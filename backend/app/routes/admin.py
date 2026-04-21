import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    AgentIntent,
    AgentPrompt,
    AgentRAGConfig,
    AgentSynonym,
    AgentToolAlias,
)
from app.rag import (
    _normalize_url,
    append_crawl_result,
    crawl_websites,
    get_knowledge_status as get_agent_knowledge_status,
    get_pdf_directory,
    load_crawl_results,
    rebuild_faiss_index,
    save_crawl_results,
)

router = APIRouter(prefix="/admin", tags=["admin"])

PROMPT_TYPES = ("system", "intent", "response", "context")
PromptType = Literal["system", "intent", "response", "context"]


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _json_aliases(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(parsed, dict):
        return {
            str(alias).strip(): str(target).strip()
            for alias, target in parsed.items()
            if str(alias).strip() and str(target).strip()
        }
    if isinstance(parsed, list):
        return {
            str(alias).strip(): str(alias).strip()
            for alias in parsed
            if str(alias).strip()
        }
    return {}


def _clean_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)
    return cleaned


class PromptItem(BaseModel):
    content: str = ""
    is_active: bool = True


class PromptsBody(BaseModel):
    prompts: dict[PromptType, PromptItem]

    @field_validator("prompts")
    @classmethod
    def require_known_prompts(cls, value: dict[PromptType, PromptItem]):
        missing = [prompt_type for prompt_type in PROMPT_TYPES if prompt_type not in value]
        if missing:
            raise ValueError(f"Missing prompt types: {', '.join(missing)}")
        return value


class RAGBody(BaseModel):
    rewrite_prompt: str = ""
    answer_instructions: str = ""
    multi_query_prompt: str = ""
    focus_keywords: list[str] = Field(default_factory=list)

    @field_validator("focus_keywords")
    @classmethod
    def clean_keywords(cls, value: list[str]):
        return _clean_unique(value)


class SynonymsBody(BaseModel):
    synonyms: dict[str, list[str]]

    @field_validator("synonyms")
    @classmethod
    def clean_synonyms(cls, value: dict[str, list[str]]):
        return {
            str(category).strip(): _clean_unique(words)
            for category, words in value.items()
            if str(category).strip()
        }


class IntentsBody(BaseModel):
    intents: list[str]

    @field_validator("intents")
    @classmethod
    def clean_intents(cls, value: list[str]):
        return _clean_unique(value)


class ToolAliasesBody(BaseModel):
    tool_aliases: dict[str, dict[str, str]]

    @field_validator("tool_aliases")
    @classmethod
    def clean_tool_aliases(cls, value: dict[str, dict[str, str]]):
        cleaned: dict[str, dict[str, str]] = {}
        for tool_name, aliases in value.items():
            clean_tool = str(tool_name).strip()
            if not clean_tool:
                continue
            cleaned_aliases = {
                str(alias).strip(): str(target).strip()
                for alias, target in aliases.items()
                if str(alias).strip() and str(target).strip()
            }
            cleaned[clean_tool] = cleaned_aliases
        return cleaned


class AddUrlBody(BaseModel):
    agent_id: str
    url: str

    @field_validator("agent_id", "url")
    @classmethod
    def require_value(cls, value: str):
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("This field is required.")
        return cleaned


def _background_rebuild(agent_id: str) -> None:
    rebuild_faiss_index(agent_id)


def _ensure_not_indexing(agent_id: str) -> None:
    if get_agent_knowledge_status(agent_id).get("indexing"):
        raise HTTPException(status_code=409, detail="Knowledge base is currently rebuilding. Please wait.")


@router.get("/prompts/{agent_id}")
def get_prompts(agent_id: str, db: Session = Depends(get_db)):
    records = db.query(AgentPrompt).filter(AgentPrompt.agent_id == agent_id).all()
    by_type = {record.prompt_type: record for record in records}
    prompts: dict[str, dict[str, Any]] = {}
    for prompt_type in PROMPT_TYPES:
        record = by_type.get(prompt_type)
        prompts[prompt_type] = {
            "content": record.content if record else "",
            "is_active": bool(record.is_active) if record else True,
            "version": record.version if record else 1,
        }
    return {"agent_id": agent_id, "prompts": prompts}


@router.put("/prompts/{agent_id}")
def update_prompts(agent_id: str, body: PromptsBody, db: Session = Depends(get_db)):
    for prompt_type, prompt in body.prompts.items():
        record = (
            db.query(AgentPrompt)
            .filter(AgentPrompt.agent_id == agent_id, AgentPrompt.prompt_type == prompt_type)
            .first()
        )
        if record:
            record.content = prompt.content
            record.is_active = prompt.is_active
            record.version = (record.version or 1) + 1
        else:
            db.add(
                AgentPrompt(
                    agent_id=agent_id,
                    prompt_type=prompt_type,
                    content=prompt.content,
                    is_active=prompt.is_active,
                )
            )
    db.commit()
    return get_prompts(agent_id, db)


@router.get("/rag/{agent_id}")
def get_rag(agent_id: str, db: Session = Depends(get_db)):
    record = (
        db.query(AgentRAGConfig)
        .filter(AgentRAGConfig.agent_id == agent_id)
        .order_by(AgentRAGConfig.created_at.desc())
        .first()
    )
    return {
        "agent_id": agent_id,
        "rewrite_prompt": record.rewrite_prompt if record else "",
        "answer_instructions": record.answer_instructions if record else "",
        "multi_query_prompt": record.multi_query_prompt if record else "",
        "focus_keywords": _json_list(record.focus_keywords if record else None),
    }


@router.put("/rag/{agent_id}")
def update_rag(agent_id: str, body: RAGBody, db: Session = Depends(get_db)):
    record = (
        db.query(AgentRAGConfig)
        .filter(AgentRAGConfig.agent_id == agent_id)
        .order_by(AgentRAGConfig.created_at.desc())
        .first()
    )
    if not record:
        record = AgentRAGConfig(agent_id=agent_id)
        db.add(record)
    record.rewrite_prompt = body.rewrite_prompt
    record.answer_instructions = body.answer_instructions
    record.multi_query_prompt = body.multi_query_prompt
    record.focus_keywords = json.dumps(body.focus_keywords)
    db.commit()
    return get_rag(agent_id, db)


@router.get("/synonyms/{agent_id}")
def get_synonyms(agent_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(AgentSynonym)
        .filter(AgentSynonym.agent_id == agent_id)
        .order_by(AgentSynonym.category)
        .all()
    )
    return {
        "agent_id": agent_id,
        "synonyms": {record.category: _json_list(record.words) for record in records},
    }


@router.put("/synonyms/{agent_id}")
def update_synonyms(agent_id: str, body: SynonymsBody, db: Session = Depends(get_db)):
    existing = {
        record.category: record
        for record in db.query(AgentSynonym).filter(AgentSynonym.agent_id == agent_id).all()
    }
    incoming_categories = set(body.synonyms.keys())
    for category, words in body.synonyms.items():
        record = existing.get(category)
        if record:
            record.words = json.dumps(words)
        else:
            db.add(AgentSynonym(agent_id=agent_id, category=category, words=json.dumps(words)))
    for category, record in existing.items():
        if category not in incoming_categories:
            db.delete(record)
    db.commit()
    return get_synonyms(agent_id, db)


@router.get("/intents/{agent_id}")
def get_intents(agent_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(AgentIntent)
        .filter(AgentIntent.agent_id == agent_id)
        .order_by(AgentIntent.label)
        .all()
    )
    return {"agent_id": agent_id, "intents": [record.label for record in records]}


@router.put("/intents/{agent_id}")
def update_intents(agent_id: str, body: IntentsBody, db: Session = Depends(get_db)):
    existing = {
        record.label: record
        for record in db.query(AgentIntent).filter(AgentIntent.agent_id == agent_id).all()
    }
    incoming = set(body.intents)
    for label in body.intents:
        if label not in existing:
            db.add(AgentIntent(agent_id=agent_id, label=label))
    for label, record in existing.items():
        if label not in incoming:
            db.delete(record)
    db.commit()
    return get_intents(agent_id, db)


@router.get("/tool-aliases/{agent_id}")
def get_tool_aliases(agent_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(AgentToolAlias)
        .filter(AgentToolAlias.agent_id == agent_id)
        .order_by(AgentToolAlias.tool_name)
        .all()
    )
    return {
        "agent_id": agent_id,
        "tool_aliases": {
            record.tool_name: _json_aliases(record.aliases)
            for record in records
        },
    }


@router.put("/tool-aliases/{agent_id}")
def update_tool_aliases(agent_id: str, body: ToolAliasesBody, db: Session = Depends(get_db)):
    existing = {
        record.tool_name: record
        for record in db.query(AgentToolAlias).filter(AgentToolAlias.agent_id == agent_id).all()
    }
    incoming_tools = set(body.tool_aliases.keys())
    for tool_name, aliases in body.tool_aliases.items():
        record = existing.get(tool_name)
        if record:
            record.aliases = json.dumps(aliases)
        else:
            db.add(AgentToolAlias(agent_id=agent_id, tool_name=tool_name, aliases=json.dumps(aliases)))
    for tool_name, record in existing.items():
        if tool_name not in incoming_tools:
            db.delete(record)
    db.commit()
    return get_tool_aliases(agent_id, db)


@router.post("/upload-pdf")
async def upload_pdf(
    agent_id: str = Form(...),
    file: UploadFile = File(...),
):
    clean_agent_id = str(agent_id).strip()
    if not clean_agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required.")
    _ensure_not_indexing(clean_agent_id)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    target_dir = get_pdf_directory({"agent_id": clean_agent_id})
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / file.filename
    contents = await file.read()
    target_path.write_bytes(contents)
    await file.close()

    return {
        "detail": "Upload successful",
        "agent_id": clean_agent_id,
        "filename": file.filename,
    }


@router.post("/add-url")
def add_url(body: AddUrlBody):
    clean_agent_id = str(body.agent_id).strip()
    _ensure_not_indexing(clean_agent_id)
    clean_url = _normalize_url(body.url)
    if not clean_url:
        raise HTTPException(status_code=400, detail="A valid URL is required.")

    existing = load_crawl_results(config={"agent_id": clean_agent_id})
    if any(_normalize_url(str(item.get("url", ""))) == clean_url for item in existing):
        return {"detail": "URL already exists", "agent_id": clean_agent_id, "url": clean_url}

    crawled_items = asyncio.run(crawl_websites([clean_url]))
    if not crawled_items:
        raise HTTPException(status_code=400, detail="No visible text could be extracted from that URL.")
    _, added = append_crawl_result(crawled_items[0], config={"agent_id": clean_agent_id})
    detail = "URL added successfully" if added else "URL already exists"
    return {"detail": detail, "agent_id": clean_agent_id, "url": clean_url}


@router.delete("/pdf/{agent_id}")
def delete_pdf(agent_id: str, filename: str):
    clean_agent_id = str(agent_id).strip()
    _ensure_not_indexing(clean_agent_id)
    clean_filename = filename.strip()
    if not clean_filename:
        raise HTTPException(status_code=400, detail="filename is required.")

    target_path = get_pdf_directory({"agent_id": clean_agent_id}) / clean_filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    target_path.unlink()
    return {"detail": "PDF removed", "agent_id": clean_agent_id, "filename": clean_filename}


@router.delete("/url/{agent_id}")
def delete_url(agent_id: str, url: str):
    clean_agent_id = str(agent_id).strip()
    _ensure_not_indexing(clean_agent_id)
    clean_url = _normalize_url(url)
    if not clean_url:
        raise HTTPException(status_code=400, detail="url is required.")

    existing = load_crawl_results(config={"agent_id": clean_agent_id})
    remaining = [
        item for item in existing
        if _normalize_url(str(item.get("url", ""))) != clean_url
    ]
    if len(remaining) == len(existing):
        raise HTTPException(status_code=404, detail="URL not found.")
    save_crawl_results(remaining, config={"agent_id": clean_agent_id})
    return {"detail": "URL removed", "agent_id": clean_agent_id, "url": clean_url}


@router.post("/rebuild-index/{agent_id}")
def rebuild_index(agent_id: str, background_tasks: BackgroundTasks):
    clean_agent_id = str(agent_id).strip()
    if not clean_agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required.")
    _ensure_not_indexing(clean_agent_id)
    background_tasks.add_task(_background_rebuild, clean_agent_id)
    return {"detail": "Index rebuild started", "agent_id": clean_agent_id}


@router.get("/knowledge-status/{agent_id}")
def knowledge_status(agent_id: str):
    clean_agent_id = str(agent_id).strip()
    if not clean_agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required.")
    return get_agent_knowledge_status(clean_agent_id)
