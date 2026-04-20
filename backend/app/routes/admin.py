import json
from typing import Any, Literal

from fastapi import APIRouter, Depends
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
