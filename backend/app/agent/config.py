from functools import lru_cache
from sqlalchemy.orm import Session
from typing import Any
import json

from app.crud import (
    get_active_prompts,
    get_rag_config,
    get_synonyms,
    get_intents,
    get_tool_aliases,
)


class DynamicAgentConfig:
    def __init__(self, agent_id: str, db: Session):
        self.agent_id = agent_id
        
        # Load configs from DB
        self.prompts = get_active_prompts(db, agent_id)
        
        self.synonyms = get_synonyms(db, agent_id)
        self.intent_labels = get_intents(db, agent_id)
        self.tool_aliases = get_tool_aliases(db, agent_id)
        
        rag_obj = get_rag_config(db, agent_id)

        self.rag = {
            "agent_id": self.agent_id,
            "rag_synonyms": self.synonyms,
            "rag_semantic_rewrite_prompt": rag_obj.rewrite_prompt if rag_obj else "",
            "rag_answer_instructions": rag_obj.answer_instructions if rag_obj else "",
            "rag_semantic_multi_query_prompt": rag_obj.multi_query_prompt if rag_obj else "",
            "rag_focus_keywords": []
        }

        if rag_obj and rag_obj.focus_keywords:
            try:
                self.rag["rag_focus_keywords"] = json.loads(rag_obj.focus_keywords)
            except (json.JSONDecodeError, TypeError):
                self.rag["rag_focus_keywords"] = [k.strip() for k in rag_obj.focus_keywords.split(",") if k.strip()]

    @property
    @lru_cache(maxsize=None)
    def SYSTEM_PROMPT(self):
        return self.prompts.get("system", "")

    @property
    @lru_cache(maxsize=None)
    def INTENT_PROMPT(self):
        return self.prompts.get("intent", "")

    @property
    @lru_cache(maxsize=None)
    def RESPONSE_PROMPT(self):
        return self.prompts.get("response", "")

    @property
    @lru_cache(maxsize=None)
    def CONTEXT_RESOLUTION_PROMPT(self):
        return self.prompts.get("context", "")

    def get_tool_aliases(self):
        return self.tool_aliases

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        if key == "agent_id":
            return self.agent_id
        elif key == "tools":
            return [
                "book_appointment",
                "cancel_appointment",
                "update_appointment",
                "view_appointment",
                "find_next_available_slot",
                "search_clinic_knowledge"
            ]
        elif key == "intent_labels":
            return self.intent_labels
        elif key == "prompts":
            return self
        elif key == "rag":
            return self.rag
        elif key == "synonyms":
            return self.synonyms
        raise KeyError(key)

AGENT_CONFIG = {
    "agent_id": "dental_bot",
    "tools": [
        "book_appointment",
        "cancel_appointment",
        "update_appointment",
        "view_appointment",
        "find_next_available_slot",
        "search_clinic_knowledge"
    ]
}
