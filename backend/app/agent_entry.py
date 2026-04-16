import logging
from typing import List
from sqlalchemy.orm import Session
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from app.config import settings
from app.agent.core import AgentCore
from app.domain.dental.service import DentalService
from app.tools.registry import load_tools

logger = logging.getLogger(__name__)

DEFAULT_TOOLS = [
    "book_appointment",
    "cancel_appointment",
    "update_appointment",
    "view_appointment",
    "find_next_available_slot",
    "_clinic_knowledge"
]

from app.agent.config import AGENT_CONFIG

def run_agent(messages: List[BaseMessage], db: Session) -> str:
    """
    Entrypoint for the multi-agent ready architecture.
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set in environment or .env")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        # llama-3.1-8b-instant
        # llama-3.3-70b-versatile
        # openai/gpt-oss-120b
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )
    
    config = AGENT_CONFIG
    tools = load_tools(config.get("tools", DEFAULT_TOOLS), db)
    domain_service = DentalService(db)

    agent_core = AgentCore(
        llm=llm,
        tools=tools,
        domain_service=domain_service,
        config=config
    )
    
    return agent_core.run(messages)
