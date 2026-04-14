import logging
from typing import Literal, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from datetime import datetime, timezone, timedelta

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

import time
import requests
from ..database import get_db
# from ..agent import run_agent
from ..agent_entry import run_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: str


class ChatResponse(BaseModel):
    response: str


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Convert simple role/content messages into LangChain message objects.
    lc_messages: List[BaseMessage] = []
    for m in request.messages:
        content = m.content.strip()
        if not content:
            continue
        if m.role == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))

    if not lc_messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    try:
        start_time_ms = time.time() * 1000
        response = run_agent(lc_messages, db)
        end_time_ms = time.time() * 1000
        
        latency_sec = round((end_time_ms - start_time_ms) / 1000, 3)
        print(f"Latency: {latency_sec} s")
        
        question_text = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        
        def truncate_text(text: str, max_words: int = 4) -> str:
            words = text.split()
            if len(words) > max_words:
                return " ".join(words[:max_words]) + "..."
            return text
        
        try:
            ist = timezone(timedelta(hours=5, minutes=30))

            message_time = datetime.fromtimestamp(
                start_time_ms / 1000,
                tz=timezone.utc
            ).strftime("%Y-%m-%d, %H:%M:%S %Z") # isoformat()

            payload = {
                "Session_id": request.session_id,
                "Request_time": message_time,
                "Request": truncate_text(question_text),
                "AI agent": truncate_text(response),
                "Latency": latency_sec
            }
            requests.post("https://rbaskets.in/dental", json=payload, timeout=2)
        except Exception as log_err:
            logger.error(f"Failed to log to RBasket: {log_err}")

        return ChatResponse(response=response)
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Chat service not configured. Set GROQ_API_KEY.",
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error while processing chat request")
        raise HTTPException(
            status_code=500,
            detail="I'm sorry, I couldn't answer that properly just now. Please try asking again in a different way.",
        )
