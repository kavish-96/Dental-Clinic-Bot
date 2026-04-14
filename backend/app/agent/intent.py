import logging
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.agent.prompts import get_intent_prompt

logger = logging.getLogger(__name__)

class IntentSystem:
    def __init__(self, llm: ChatGroq, config):
        self.llm = llm
        self.config = config

    def _latest_user_message(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content).strip()
        return ""

    def _recent_conversation_snippet(self, messages: List[BaseMessage], limit: int = 6) -> str:
        lines: list[str] = []
        for message in messages[-limit:]:
            if isinstance(message, HumanMessage):
                lines.append(f"user: {message.content}")
            elif hasattr(message, "content"):
                lines.append(f"assistant: {message.content}")
        return "\n".join(lines)

    def classify_intent(
        self,
        messages: List[BaseMessage],
        latest_user_message: str | None = None,
        allowed_labels: list[str] = None
    ) -> str:
        latest_user_message = latest_user_message or self._latest_user_message(messages)
        if not latest_user_message:
            return "casual"

        conversation_snippet = self._recent_conversation_snippet(messages)

        try:
            classified = self.llm.invoke(
                [
                    SystemMessage(content=get_intent_prompt(self.config.get("prompts"))),
                    HumanMessage(
                        content=(
                            f"Conversation:\n{conversation_snippet}\n\n"
                            f"Latest message: {latest_user_message}"
                        )
                    ),
                ]
            )
            label = str(classified.content or "").strip().lower()
        except Exception:
            logger.exception("Intent classification failed")
            label = ""

        valid_labels = allowed_labels if allowed_labels else self.config.get("intent_labels", [])
        return label if label in valid_labels else "casual"
