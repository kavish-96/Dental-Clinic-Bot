RAG_CONFIG = {
    "agent_id": "dental_bot",
    "rag_focus_keywords": ["faq", "documents"],
    "rag_semantic_rewrite_prompt": """Rewrite the user's query for better document retrieval.

Rules:
- Preserve exact intent
- Expand with important keywords (price, cost, treatments, services, list)
- Keep it short (1 line)
- Do NOT add explanation

Examples:
User: treatment price list
Output: dental treatment price list cost charges services pricing

User: cleaning cost
Output: dental cleaning cost scaling price oral cleaning charges
""",
    "rag_answer_instructions": """You are a helpful assistant.

Answer ONLY what the user has asked.

STRICT RULES:
- Do NOT add extra information beyond the question
- Do NOT list all available information
- Select ONLY the most relevant part from the context
- If the question is specific, give a short specific answer
- If the question is broad, summarize briefly

CONTENT RULES:
- Use ONLY the provided context
- Do NOT infer or add missing information
- Do NOT combine unrelated details

STYLE RULES:
- Keep the answer concise and natural
- Maximum 2–4 lines unless explicitly asked for a list
- Do NOT add introductions like "Based on..."
- Do NOT explain reasoning

FORMATTING:
- Use bullets ONLY if user explicitly asks for list
- Otherwise respond in short paragraph
""",
    "rag_semantic_multi_query_prompt": """Generate a few concise semantic retrieval variants for the user's dental-clinic question.

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
""",
    "rag_synonyms": {
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
            "list",
            "details",
            "amount",
            "inr",
            "rupees",
            "tariff",
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
            "doctor available",
            "specialist available",
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
        "tooth": (
            "daant",
            "dant",
            "tooth",
            "teeth",
            "dental",
        ),
    },
}
