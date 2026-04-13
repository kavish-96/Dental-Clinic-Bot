import logging
from typing import Literal, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

import time
from ..database import get_db
from ..models import ChatLog
from ..agent import run_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


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
        
        latency_ms_val = round(end_time_ms - start_time_ms, 3)
        print(f"Latency: {latency_ms_val} ms")
        
        question_text = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        
        def truncate_text(text: str, max_words: int = 4) -> str:
            words = text.split()
            if len(words) > max_words:
                return " ".join(words[:max_words]) + "..."
            return text
        
        chat_log = ChatLog(
            question=truncate_text(question_text),
            response=truncate_text(response),
            latency_ms=latency_ms_val
        )
        db.add(chat_log)
        db.commit()

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
