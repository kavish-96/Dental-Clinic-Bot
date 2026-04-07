from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .config import settings

logger = logging.getLogger(__name__)


# ========================
# CORE RAG PARAMETERS
# ========================

# Semantic chunking defaults requested by the user.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Wider candidate pool improves recall, while post-filtering keeps context diverse.
TOP_K = 5
FETCH_K = 14
MMR_LAMBDA = 0.55
MAX_SEMANTIC_QUERIES = 4
SIMILARITY_K = 8

# Keep context clean and bounded before passing it to the answer model.
MAX_CONTEXT_CHARS_PER_CHUNK = 1400
MAX_TOTAL_CONTEXT_CHARS = 6000
MIN_CONTENT_OVERLAP_RATIO = 0.7
EXACT_PHRASE_BOOST = 0.35
TOKEN_OVERLAP_BOOST = 0.25
PDF_SOURCE_BOOST = 0.15


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en"
ALTERNATE_EMBEDDING_MODEL = "intfloat/e5-large-v2"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


SEMANTIC_QUERY_REWRITE_PROMPT = """Rewrite the user's dental-clinic search query into a concise semantic retrieval query.

Requirements:
- Expand with close synonyms, related dental or medical terms, and alternate phrasings.
- Keep the output to 1 or 2 short lines maximum.
- Preserve the user's intent exactly.
- Output only the rewritten retrieval query with no bullets, labels, or explanation.

Example:
User query: tooth pain fix
Expanded retrieval query: tooth pain treatment, dental pain relief, cavity treatment, root canal therapy
"""


SEMANTIC_ANSWER_INSTRUCTIONS = """Use the clinic knowledge below to answer the user's question accurately.
Use semantic understanding. Even if the user query uses different wording than the context, match based on meaning and provide the best possible answer.
Infer answers from related concepts when the wording differs.
Do not say the answer is not found if relevant information exists in related wording.
If the answer is only partially covered, answer with the best supported details and clearly mention the limitation."""


SEMANTIC_MULTI_QUERY_PROMPT = """Generate a few concise semantic retrieval variants for the user's dental-clinic question.

Requirements:
- Preserve the exact intent.
- Use different natural phrasings that improve semantic recall.
- Include role, specialty, treatment, and concept-level paraphrases when relevant.
- Do not add explanation, bullets, numbering, or labels.
- Return 2 to 4 short lines only, one query variant per line.

Example:
User question: who is the implantologist
implant specialist doctor
dental implant specialist
doctor for implant treatment
"""


QUERY_SYNONYM_GROUPS: dict[str, tuple[str, ...]] = {
    "price": (
        "price",
        "prices",
        "pricing",
        "cost",
        "costs",
        "costing",
        "charge",
        "charges",
        "fee",
        "fees",
        "rate",
        "rates",
        "price list",
        "price sheet",
        "quotation",
        "estimate",
    ),
    "services": (
        "service",
        "services",
        "treatment",
        "treatments",
        "procedure",
        "procedures",
        "offered",
        "available",
    ),
    "doctor": (
        "doctor",
        "doctors",
        "dentist",
        "dentists",
        "specialist",
        "specialists",
        "expert",
        "experts",
    ),
    "consultation": (
        "consultation",
        "consult",
        "visit",
        "checkup",
        "diagnosis",
        "initial diagnosis",
    ),
    "cleaning": (
        "cleaning",
        "scaling",
        "polishing",
        "oral prophylaxis",
        "dental cleaning",
    ),
    "filling": (
        "filling",
        "fillings",
        "tooth filling",
        "cavity filling",
        "restoration",
        "composite filling",
    ),
    "root_canal": (
        "root canal",
        "root canal treatment",
        "rct",
        "endodontic treatment",
        "endodontics",
        "pulp treatment",
    ),
    "extraction": (
        "extraction",
        "extractions",
        "tooth removal",
        "removal",
        "exodontia",
    ),
    "whitening": (
        "whitening",
        "bleaching",
        "teeth whitening",
        "tooth whitening",
    ),
    "implant": (
        "implant",
        "implants",
        "dental implant",
        "implantology",
        "implant treatment",
        "implantologist",
        "implant specialist",
        "implant dentist",
    ),
    "braces": (
        "braces",
        "orthodontics",
        "orthodontic treatment",
        "aligners",
        "teeth straightening",
        "orthodontist",
        "brace treatment",
    ),
}


_EMBEDDINGS_CACHE: dict[tuple[str, bool], Any] = {}
_QUERY_EXPANSION_CACHE: dict[str, str] = {}
_QUERY_VARIANTS_CACHE: dict[str, list[str]] = {}

MULTISPACE_PATTERN = re.compile(r"[ \t]+")
MULTINEWLINE_PATTERN = re.compile(r"\n{3,}")
INLINE_TABLE_SEPARATOR_PATTERN = re.compile(r"\s{2,}|\t+")
TOKEN_PATTERN = re.compile(r"\b[a-z0-9]{3,}\b")


class RagConfigurationError(RuntimeError):
    """Raised when the RAG stack is unavailable or misconfigured."""


# ========================
# PATHS
# ========================


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_backend_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (_backend_root() / path).resolve()


def get_pdf_directory() -> Path:
    return _resolve_backend_path(settings.RAG_PDF_DIR)


def get_crawl_output_path() -> Path:
    return _resolve_backend_path(settings.RAG_CRAWL_OUTPUT_PATH)


def get_faiss_index_path() -> Path:
    return _resolve_backend_path(settings.RAG_FAISS_INDEX_DIR)


# ========================
# TEXT NORMALIZATION
# ========================


def _normalize_text(text: str) -> str:
    """Normalize noisy PDF text while preserving basic table structure."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines: list[str] = []

    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            normalized_lines.append("")
            continue

        # Preserve table-ish columns from PDFs so pricing rows stay readable.
        if "\t" in line or len(INLINE_TABLE_SEPARATOR_PATTERN.findall(line)) >= 2:
            line = INLINE_TABLE_SEPARATOR_PATTERN.sub(" | ", line)

        # Join accidental single-character splits often produced by PDFs.
        line = re.sub(r"(?<=\b\w) (?=\w\b)", "", line)
        line = MULTISPACE_PATTERN.sub(" ", line).strip(" |")
        normalized_lines.append(line)

    normalized = "\n".join(normalized_lines)
    normalized = MULTINEWLINE_PATTERN.sub("\n\n", normalized)
    return normalized.strip()


def _build_pdf_semantic_variants(text: str) -> str:
    """Augment PDF table text with semantic aliases to improve retrieval."""
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    lower = normalized.lower()
    variants: list[str] = [normalized]

    if "price" in lower or "inr" in lower or "₹" in lower:
        variants.append("pricing cost charges fees rates amount rupees inr treatment price list")
    if "consultation" in lower:
        variants.append("consult visit checkup diagnosis")
    if "doctor" in lower or "dentist" in lower or "specialist" in lower:
        variants.append("doctor dentist specialist experts specialist available")
    if "scaling" in lower or "cleaning" in lower:
        variants.append("scaling cleaning polishing oral prophylaxis")
    if "filling" in lower or "cavity" in lower:
        variants.append("tooth filling cavity filling restoration composite filling")
    if "root canal" in lower:
        variants.append("root canal rct endodontic treatment pulp therapy")
    if "extraction" in lower or "removal" in lower:
        variants.append("extraction tooth removal exodontia")
    if "whitening" in lower or "bleaching" in lower:
        variants.append("teeth whitening bleaching")
    if "implant" in lower:
        variants.append("dental implant implantology implantologist implant specialist implant dentist")
    if "braces" in lower or "aligners" in lower:
        variants.append("braces orthodontics aligners teeth straightening orthodontist brace treatment")

    return "\n".join(variants)


# ========================
# QUERY EXPANSION
# ========================


def _get_query_expansion_llm() -> ChatGroq | None:
    if not settings.GROQ_API_KEY:
        return None

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )


def _heuristic_expand_query(query: str) -> str:
    normalized_query = " ".join(query.lower().split())
    expanded_terms: list[str] = [query.strip()]

    for group_terms in QUERY_SYNONYM_GROUPS.values():
        if any(term in normalized_query for term in group_terms):
            expanded_terms.extend(term for term in group_terms if term not in normalized_query)

    # Price questions often need help matching table headings and currency markers.
    if any(term in normalized_query for term in QUERY_SYNONYM_GROUPS["price"]):
        expanded_terms.extend(
            term
            for term in ("list", "details", "amount", "inr", "rupees", "tariff")
            if term not in normalized_query
        )

    # Handle common conversational or Hinglish-style dental intent phrasing.
    if any(term in normalized_query for term in ("daant", "dant", "tooth", "teeth")):
        expanded_terms.extend(
            term for term in ("dental", "tooth", "teeth") if term not in normalized_query
        )
    if "braces" in normalized_query or "aligner" in normalized_query:
        expanded_terms.extend(
            term
            for term in ("orthodontics", "orthodontist", "teeth straightening")
            if term not in normalized_query
        )
    if "implant" in normalized_query:
        expanded_terms.extend(
            term
            for term in ("implantologist", "implant specialist", "implant dentist")
            if term not in normalized_query
        )
    if any(term in normalized_query for term in ("doctor", "dentist", "specialist")):
        expanded_terms.extend(
            term
            for term in ("doctor available", "specialist available", "expert")
            if term not in normalized_query
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for term in expanded_terms:
        key = term.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(term.strip())

    return ", ".join(deduped)


def _expand_query(query: str) -> str:
    cleaned_query = query.strip()
    if not cleaned_query:
        return ""

    cached = _QUERY_EXPANSION_CACHE.get(cleaned_query)
    if cached:
        return cached

    heuristic_query = _heuristic_expand_query(cleaned_query)
    llm = _get_query_expansion_llm()
    if llm is None:
        _QUERY_EXPANSION_CACHE[cleaned_query] = heuristic_query
        return heuristic_query

    try:
        rewritten = llm.invoke(
            [
                SystemMessage(content=SEMANTIC_QUERY_REWRITE_PROMPT),
                HumanMessage(content=f"User query: {cleaned_query}"),
            ]
        )
        rewritten_text = _normalize_text(str(rewritten.content or ""))
        expanded_query = rewritten_text or heuristic_query
    except Exception as exc:
        logger.warning("Semantic query expansion failed; using heuristic expansion: %s", exc)
        expanded_query = heuristic_query

    _QUERY_EXPANSION_CACHE[cleaned_query] = expanded_query
    return expanded_query


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for query in queries:
        normalized = _normalize_text(query)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    return deduped


def _build_semantic_query_variants(query: str) -> list[str]:
    """Create multiple semantic retrieval phrasings to improve meaning-based recall."""
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    cached = _QUERY_VARIANTS_CACHE.get(cleaned_query)
    if cached:
        return cached

    expanded_query = _expand_query(cleaned_query)
    query_candidates: list[str] = [cleaned_query, expanded_query]

    llm = _get_query_expansion_llm()
    if llm is not None:
        try:
            rewritten = llm.invoke(
                [
                    SystemMessage(content=SEMANTIC_MULTI_QUERY_PROMPT),
                    HumanMessage(content=f"User question: {cleaned_query}"),
                ]
            )
            raw_variants = str(rewritten.content or "")
            query_candidates.extend(
                line.strip()
                for line in raw_variants.splitlines()
                if line.strip()
            )
        except Exception as exc:
            logger.warning("Semantic multi-query generation failed; using base variants: %s", exc)

    variants = _dedupe_queries(query_candidates)[:MAX_SEMANTIC_QUERIES]
    _QUERY_VARIANTS_CACHE[cleaned_query] = variants
    return variants


# ========================
# CONTEXT CLEANING
# ========================


def _clean_context_snippet(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""
    if len(cleaned) <= MAX_CONTEXT_CHARS_PER_CHUNK:
        return cleaned
    return cleaned[:MAX_CONTEXT_CHARS_PER_CHUNK].rsplit(" ", 1)[0].strip() + "..."


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _lexical_match_bonus(query: str, content: str) -> float:
    """General lexical boost for exact wording and close phrasing, without domain keywords."""
    normalized_query = _normalize_text(query).lower()
    normalized_content = _normalize_text(content).lower()
    if not normalized_query or not normalized_content:
        return 0.0

    bonus = 0.0

    if normalized_query in normalized_content:
        bonus += EXACT_PHRASE_BOOST

    query_tokens = _tokenize_for_overlap(normalized_query)
    content_tokens = _tokenize_for_overlap(normalized_content)
    if query_tokens and content_tokens:
        overlap_ratio = len(query_tokens & content_tokens) / len(query_tokens)
        bonus += overlap_ratio * TOKEN_OVERLAP_BOOST

    return bonus


def _content_overlap_ratio(left: str, right: str) -> float:
    left_tokens = _tokenize_for_overlap(left)
    right_tokens = _tokenize_for_overlap(right)
    if not left_tokens or not right_tokens:
        return 0.0

    shared = left_tokens & right_tokens
    return len(shared) / min(len(left_tokens), len(right_tokens))


def _select_diverse_documents(docs: list[Document]) -> list[Document]:
    """Keep highly relevant chunks while filtering near-duplicates."""
    selected: list[Document] = []
    seen_exact: set[str] = set()

    for doc in docs:
        content = _clean_context_snippet(doc.page_content)
        if not content:
            continue

        normalized = content.lower()
        if normalized in seen_exact:
            continue

        if any(
            _content_overlap_ratio(normalized, existing.page_content.lower()) >= MIN_CONTENT_OVERLAP_RATIO
            for existing in selected
        ):
            continue

        selected.append(Document(page_content=content, metadata=dict(doc.metadata)))
        seen_exact.add(normalized)

        if len(selected) >= TOP_K:
            break

    return selected


def _prefer_pdf_documents(docs: list[Document]) -> list[Document]:
    """When PDF evidence exists, prefer it over weaker website matches."""
    pdf_docs = [
        doc
        for doc in docs
        if str(doc.metadata.get("source_type", "")).strip().lower() == "pdf"
    ]
    return pdf_docs or docs


def _retrieve_semantic_documents(vector_store, query_variants: list[str]) -> list[Document]:
    """Retrieve across multiple semantic phrasings and merge results without changing architecture."""
    if not query_variants:
        return []

    aggregated: dict[tuple[str, str, Any, Any], Document] = {}

    for query_variant in query_variants:
        variant_candidates: list[tuple[Document, float]] = []

        try:
            docs = vector_store.max_marginal_relevance_search(
                query_variant,
                k=SIMILARITY_K,
                fetch_k=FETCH_K,
                lambda_mult=MMR_LAMBDA,
            )
            variant_candidates.extend((doc, 0.0) for doc in docs)
        except Exception as exc:
            logger.warning("MMR retrieval failed for variant '%s': %s", query_variant, exc)

        try:
            scored_docs = vector_store.similarity_search_with_score(
                query_variant,
                k=SIMILARITY_K,
            )
            variant_candidates.extend(scored_docs)
        except Exception as exc:
            logger.warning("Similarity retrieval failed for variant '%s': %s", query_variant, exc)

        for doc, score in variant_candidates:
            content = _clean_context_snippet(doc.page_content)
            if not content:
                continue

            signature = (
                str(doc.metadata.get("source", "")),
                content.lower(),
                doc.metadata.get("page"),
                doc.metadata.get("chunk_index"),
            )

            existing = aggregated.get(signature)
            normalized_score = float(score)
            source_type = str(doc.metadata.get("source_type", "")).strip().lower()
            lexical_bonus = _lexical_match_bonus(query_variant, content)
            source_bonus = PDF_SOURCE_BOOST if source_type == "pdf" else 0.0
            effective_score = normalized_score - lexical_bonus - source_bonus

            if existing is None:
                enriched_doc = Document(page_content=content, metadata=dict(doc.metadata))
                enriched_doc.metadata["retrieval_score"] = effective_score
                enriched_doc.metadata["query_hits"] = 1
                enriched_doc.metadata["best_query_variant"] = query_variant
                aggregated[signature] = enriched_doc
                continue

            previous_score = float(existing.metadata.get("retrieval_score", normalized_score))
            if effective_score < previous_score:
                existing.metadata["retrieval_score"] = effective_score
                existing.metadata["best_query_variant"] = query_variant
            existing.metadata["query_hits"] = int(existing.metadata.get("query_hits", 1)) + 1

    merged_docs = list(aggregated.values())
    merged_docs.sort(
        key=lambda doc: (
            -int(doc.metadata.get("query_hits", 1)),
            float(doc.metadata.get("retrieval_score", 0.0)),
        )
    )
    return _prefer_pdf_documents(merged_docs)


def _build_context_blocks(docs: list[Document]) -> list[str]:
    """Construct a clean structured context for the answer model."""
    blocks: list[str] = []
    total_chars = 0

    for index, doc in enumerate(docs, start=1):
        content = _clean_context_snippet(doc.page_content)
        if not content:
            continue

        source = str(doc.metadata.get("source", "unknown")).strip() or "unknown"
        page = doc.metadata.get("page")
        chunk_index = doc.metadata.get("chunk_index")

        location_parts: list[str] = []
        if page is not None:
            location_parts.append(f"page {page}")
        if chunk_index is not None:
            location_parts.append(f"chunk {chunk_index}")
        location = f" ({', '.join(location_parts)})" if location_parts else ""

        block = f"[Source {index}: {source}{location}]\n{content}"
        projected_total = total_chars + len(block)
        if blocks and projected_total > MAX_TOTAL_CONTEXT_CHARS:
            continue

        blocks.append(block)
        total_chars = projected_total

    return blocks


# ========================
# TEXT SPLITTING
# ========================


def _get_text_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RagConfigurationError(
            "LangChain text splitter support is unavailable. Install backend requirements to enable retrieval."
        ) from exc

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )


# ========================
# SEMANTIC EMBEDDINGS
# ========================


class SemanticEmbeddings:
    """Instruction-aware embeddings with consistent query and document encoding."""

    def __init__(self, model_name: str, *, local_files_only: bool = False) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RagConfigurationError(
                "RAG dependencies are missing. Install backend requirements to enable retrieval."
            ) from exc

        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self._encoder = SentenceTransformer(self.model_name, local_files_only=local_files_only)

    def _uses_e5_prefixes(self) -> bool:
        return "e5" in self.model_name.lower()

    def _uses_bge_instruction(self) -> bool:
        lowered = self.model_name.lower()
        return "bge-" in lowered or "/bge" in lowered

    def _prepare_document(self, text: str) -> str:
        normalized = _normalize_text(text)
        if self._uses_e5_prefixes():
            return f"passage: {normalized}"
        return normalized

    def _prepare_query(self, text: str) -> str:
        normalized = _normalize_text(text)
        if self._uses_e5_prefixes():
            return f"query: {normalized}"
        if self._uses_bge_instruction():
            return f"{BGE_QUERY_INSTRUCTION} {normalized}"
        return normalized

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._prepare_document(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode([self._prepare_query(text)])
        return vectors[0] if vectors else []

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def _embedding_model_name() -> str:
    configured = (settings.RAG_EMBEDDING_MODEL or "").strip()
    return configured or DEFAULT_EMBEDDING_MODEL


def _get_embeddings(*, local_files_only: bool = False):
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise RagConfigurationError(
            "RAG dependencies are missing. Install backend requirements to enable retrieval."
        ) from exc

    model_name = _embedding_model_name()
    cache_key = (model_name, local_files_only)
    if cache_key not in _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE[cache_key] = SemanticEmbeddings(
            model_name,
            local_files_only=local_files_only,
        )
    return _EMBEDDINGS_CACHE[cache_key]


# ========================
# PDF LOADING
# ========================


def load_pdf_documents(pdf_dir: Path | None = None) -> list[Document]:
    target_dir = pdf_dir or get_pdf_directory()
    if not target_dir.exists():
        return []

    pdf_paths = sorted(target_dir.glob("*.pdf"))
    if not pdf_paths:
        return []

    try:
        from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
    except ImportError as exc:
        raise RagConfigurationError(
            "PDF loaders are unavailable. Install backend requirements to load PDF knowledge."
        ) from exc

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        loaded: list[Document] = []

        # Unstructured often preserves layout better; PyPDFLoader is a safe fallback.
        try:
            loaded = UnstructuredFileLoader(str(pdf_path)).load()
        except Exception as exc:
            logger.warning(
                "Unstructured PDF loading failed for '%s', falling back to PyPDFLoader: %s",
                pdf_path,
                exc,
            )
            try:
                loaded = PyPDFLoader(str(pdf_path)).load()
            except Exception as fallback_exc:
                logger.warning("Failed to load PDF '%s': %s", pdf_path, fallback_exc)
                continue

        for doc in loaded:
            normalized_content = _build_pdf_semantic_variants(doc.page_content)
            if not normalized_content:
                continue

            doc.page_content = normalized_content
            doc.metadata["source"] = str(pdf_path)
            doc.metadata["source_type"] = "pdf"
            documents.append(doc)

    return documents


# ========================
# WEBSITE LOADING
# ========================


def load_crawl_documents(
    crawl_results: list[dict[str, Any]] | None = None,
    crawl_output_path: Path | None = None,
) -> list[Document]:
    if crawl_results is None:
        source_path = crawl_output_path or get_crawl_output_path()
        if not source_path.exists():
            return []
        try:
            crawl_results = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid crawl output JSON at {source_path}: {exc}") from exc

    documents: list[Document] = []
    for item in crawl_results:
        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()
        url = str(item.get("url", "")).strip()
        normalized_content = _normalize_text(content)
        if not normalized_content:
            continue

        documents.append(
            Document(
                page_content=normalized_content,
                metadata={
                    "source": url or "crawl4ai",
                    "source_type": "website",
                },
            )
        )

    return documents


def save_crawl_results(
    crawl_results: list[dict[str, str]],
    output_path: Path | None = None,
) -> Path:
    target_path = output_path or get_crawl_output_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(crawl_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return target_path


# ========================
# CRAWLING
# ========================


async def crawl_websites(urls: list[str]) -> list[dict[str, str]]:
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError as exc:
        raise RagConfigurationError(
            "crawl4ai is unavailable. Install backend requirements to enable website crawling."
        ) from exc

    crawl_results: list[dict[str, str]] = []
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            result = await crawler.arun(url=url)
            content = getattr(result, "markdown", None) or getattr(result, "cleaned_html", None) or ""
            if content:
                crawl_results.append({"url": url, "content": str(content)})

    return crawl_results


def crawl_and_save_websites(
    urls: list[str],
    output_path: Path | None = None,
) -> Path:
    crawl_results = asyncio.run(crawl_websites(urls))
    return save_crawl_results(crawl_results, output_path=output_path)


# ========================
# SPLITTING
# ========================


def split_documents(documents: list[Document]) -> list[Document]:
    if not documents:
        return []

    splitter = _get_text_splitter()
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk.page_content = _normalize_text(chunk.page_content)
        chunk.metadata["chunk_index"] = index
        chunk.metadata["page_content_length"] = len(chunk.page_content)

    return [chunk for chunk in chunks if chunk.page_content.strip()]


# ========================
# LOAD SOURCES
# ========================


def load_source_documents(
    pdf_dir: Path | None = None,
    crawl_results: list[dict[str, Any]] | None = None,
    crawl_output_path: Path | None = None,
) -> list[Document]:
    pdf_documents = load_pdf_documents(pdf_dir=pdf_dir)
    crawl_documents = load_crawl_documents(
        crawl_results=crawl_results,
        crawl_output_path=crawl_output_path,
    )
    return [*pdf_documents, *crawl_documents]


# ========================
# VECTOR STORE
# ========================


def build_vector_store(documents: list[Document]):
    if not documents:
        raise ValueError("No source documents were found for RAG ingestion.")

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise RagConfigurationError(
            "FAISS vector store support is unavailable. Install backend requirements to enable retrieval."
        ) from exc

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Source documents were loaded, but no chunks were produced.")

    try:
        embeddings = _get_embeddings(local_files_only=True)
    except Exception:
        embeddings = _get_embeddings(local_files_only=False)

    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(vector_store, index_path: Path | None = None) -> Path:
    target_path = index_path or get_faiss_index_path()
    target_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_path))
    return target_path


def load_vector_store(index_path: Path | None = None):
    target_path = index_path or get_faiss_index_path()
    index_file = target_path / "index.faiss"
    store_file = target_path / "index.pkl"
    if not index_file.exists() or not store_file.exists():
        return None

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise RagConfigurationError(
            "FAISS vector store support is unavailable. Install backend requirements to enable retrieval."
        ) from exc

    try:
        embeddings = _get_embeddings(local_files_only=True)
    except Exception:
        embeddings = _get_embeddings(local_files_only=False)

    return FAISS.load_local(
        str(target_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ========================
# INGESTION
# ========================


def ingest_knowledge_base(
    pdf_dir: Path | None = None,
    crawl_results: list[dict[str, Any]] | None = None,
    crawl_output_path: Path | None = None,
    index_path: Path | None = None,
):
    documents = load_source_documents(
        pdf_dir=pdf_dir,
        crawl_results=crawl_results,
        crawl_output_path=crawl_output_path,
    )
    vector_store = build_vector_store(documents)
    save_vector_store(vector_store, index_path=index_path)
    return vector_store


def get_or_create_vector_store():
    vector_store = load_vector_store()
    if vector_store is not None:
        return vector_store

    documents = load_source_documents()
    if not documents:
        return None

    try:
        return ingest_knowledge_base()
    except RagConfigurationError:
        raise
    except Exception as exc:
        logger.exception("Failed to build FAISS index: %s", exc)
        raise


# ========================
# QUERY
# ========================


def query_knowledge_base(query: str) -> str:
    cleaned_query = query.strip()
    if not cleaned_query:
        return "Please provide a knowledge question to search."

    query_variants = _build_semantic_query_variants(cleaned_query)
    expanded_query = query_variants[1] if len(query_variants) > 1 else query_variants[0]

    try:
        vector_store = get_or_create_vector_store()
    except RagConfigurationError as exc:
        return str(exc)
    except Exception as exc:
        return f"Unable to query the clinic knowledge base right now: {exc}"

    if vector_store is None:
        return (
            "The clinic knowledge base is empty right now. Add PDFs to "
            f"'{get_pdf_directory()}' or crawl output to '{get_crawl_output_path()}', "
            "then rebuild the FAISS index."
        )

    docs = _retrieve_semantic_documents(vector_store, query_variants)
    if not docs:
        return "No relevant clinic knowledge was found for that question."

    diverse_docs = _select_diverse_documents(_prefer_pdf_documents(docs))
    context_blocks = _build_context_blocks(diverse_docs)
    if not context_blocks:
        return "Relevant documents were retrieved, but they did not contain usable text."

    return (
        f"{SEMANTIC_ANSWER_INSTRUCTIONS}\n\n"
        f"User question: {cleaned_query}\n"
        f"Expanded retrieval query: {expanded_query}\n\n"
        + "\n\n".join(context_blocks)
    )


def rebuild_default_index() -> Path:
    vector_store = ingest_knowledge_base()
    return save_vector_store(vector_store)


# ========================
# CLI
# ========================


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the clinic FAISS knowledge index.")
    parser.add_argument(
        "--crawl-url",
        action="append",
        default=[],
        help="Website URL to crawl with crawl4ai and save into the configured crawl output JSON.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the FAISS index from the configured PDF and crawl sources.",
    )
    args = parser.parse_args()

    if args.crawl_url:
        output_path = crawl_and_save_websites(args.crawl_url)
        print(f"Crawl output saved to {output_path}")

    if not args.rebuild and not args.crawl_url:
        parser.print_help()
        return

    if args.rebuild:
        index_path = rebuild_default_index()
        print(f"FAISS index saved to {index_path}")


if __name__ == "__main__":
    main()



