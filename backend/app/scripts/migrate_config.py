import sys
import os
import json

# Add backend root to Python path so we can import from app
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from app.database import SessionLocal, engine
from app.models import Base, AgentPrompt, AgentRAGConfig, AgentSynonym, AgentIntent, AgentToolAlias

from app.domain.dental.config import (
    SYSTEM_PROMPT,
    INTENT_PROMPT,
    RESPONSE_PROMPT,
    CONTEXT_RESOLUTION_PROMPT,
    INTENT_LABELS,
    TOOL_ARG_ALIASES,
)
from app.domain.dental.rag_config import RAG_CONFIG

AGENT_ID = "dental_bot"

def migrate():
    # Ensure tables exist (using create_all here is a safe bet just in case migrations aren't explicitly run)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        print(f"Starting migration for agent_id = '{AGENT_ID}'...")
        
        # 1. Migrate Prompts
        prompts = {
            "system": SYSTEM_PROMPT,
            "intent": INTENT_PROMPT,
            "response": RESPONSE_PROMPT,
            "context": CONTEXT_RESOLUTION_PROMPT
        }
        
        prompts_added = 0
        for p_type, content in prompts.items():
            existing = db.query(AgentPrompt).filter_by(agent_id=AGENT_ID, prompt_type=p_type).first()
            if not existing:
                db.add(AgentPrompt(agent_id=AGENT_ID, prompt_type=p_type, content=content, is_active=True))
                prompts_added += 1
        print(f"Added {prompts_added} new prompts.")

        # 2. Migrate RAG Config
        existing_rag = db.query(AgentRAGConfig).filter_by(agent_id=AGENT_ID).first()
        if not existing_rag:
            rag = AgentRAGConfig(
                agent_id=AGENT_ID,
                rewrite_prompt=RAG_CONFIG.get("rag_semantic_rewrite_prompt"),
                answer_instructions=RAG_CONFIG.get("rag_answer_instructions"),
                multi_query_prompt=RAG_CONFIG.get("rag_semantic_multi_query_prompt"),
                focus_keywords=json.dumps(RAG_CONFIG.get("rag_focus_keywords", [])),
            )
            db.add(rag)
            print("Added new RAG config.")
        else:
            print("RAG config already exists.")

        # 3. Migrate Synonyms
        synonyms_dict = RAG_CONFIG.get("rag_synonyms", {})
        syns_added = 0
        for category, words_tuple in synonyms_dict.items():
            existing_syn = db.query(AgentSynonym).filter_by(agent_id=AGENT_ID, category=category).first()
            if not existing_syn:
                # Save as a JSON string
                db.add(AgentSynonym(agent_id=AGENT_ID, category=category, words=json.dumps(list(words_tuple))))
                syns_added += 1
        print(f"Added {syns_added} new synonym categories.")

        # 4. Migrate Intents
        intents_added = 0
        for label in INTENT_LABELS:
            existing_intent = db.query(AgentIntent).filter_by(agent_id=AGENT_ID, label=label).first()
            if not existing_intent:
                db.add(AgentIntent(agent_id=AGENT_ID, label=label))
                intents_added += 1
        print(f"Added {intents_added} new intent labels.")

        # 5. Migrate Tool Aliases
        aliases_added = 0
        for tool_name, aliases in TOOL_ARG_ALIASES.items():
            existing_alias = db.query(AgentToolAlias).filter_by(agent_id=AGENT_ID, tool_name=tool_name).first()
            if not existing_alias:
                # `aliases` is a dict from config.py (`{"mobile": "mobile_number", ...}`)
                # Storing it directly as a JSON object, easily decoded back later
                db.add(AgentToolAlias(agent_id=AGENT_ID, tool_name=tool_name, aliases=json.dumps(aliases)))
                aliases_added += 1
        print(f"Added {aliases_added} new tool aliases.")

        # Commit all changes
        db.commit()
        print("Migration complete!")
        
    except Exception as e:
        db.rollback()
        print(f"Migration failed Exception: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    migrate()
