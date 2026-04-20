import json
import logging
import re
from typing import List, Any
from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from app.agent.prompts import (
    get_system_prompt,
    get_context_resolution_prompt,
)
from app.agent.intent import IntentSystem
from app.datetime_utils import current_date, current_datetime
from app.domain.dental.service import TOOL_FALLBACK_MESSAGES

logger = logging.getLogger(__name__)

TEXT_TOOL_CALL_PATTERN = re.compile(
    r"<function=(?P<name>[a-zA-Z0-9_]+)>(?P<args>\{.*?\})</function>",
    re.DOTALL,
)
TEXT_TOOL_BLOCK_PATTERN = re.compile(
    r"<function>(?P<name>[a-zA-Z0-9_]+)(?P<args>\{.*?\})</function>",
    re.DOTALL,
)
TEXT_TOOL_INLINE_PATTERN = re.compile(
    r"<function>(?P<name>[a-zA-Z0-9_]+)\((?P<args>.*?)\)</function>",
    re.DOTALL,
)

class AgentCore:
    def __init__(self, llm: ChatGroq, tools: list, domain_service: Any, config: dict):
        self.llm = llm
        self.tools = tools
        self.domain_service = domain_service
        self.config = config
        self.intent_system = IntentSystem(llm, config)

    def _latest_user_message(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content).strip()
        return ""

    def _recent_conversation_snippet(self, messages: List[BaseMessage], limit: int = 6) -> str:
        lines = []
        for message in messages[-limit:]:
            if isinstance(message, HumanMessage):
                lines.append(f"user: {message.content}")
            elif isinstance(message, AIMessage):
                lines.append(f"assistant: {message.content}")
        return "\n".join(lines)

    def _extract_text_tool_calls(self, content: str) -> list[dict]:
        tool_calls: list[dict] = []
        all_matches = list(TEXT_TOOL_CALL_PATTERN.finditer(content)) + list(
            TEXT_TOOL_BLOCK_PATTERN.finditer(content)
        )
        for index, match in enumerate(all_matches, start=1):
            tool_name = match.group("name")
            raw_args = match.group("args")
            try:
                tool_args = json.loads(raw_args)
            except json.JSONDecodeError:
                continue

            tool_calls.append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "id": f"text_tool_call_{index}",
                }
            )

        if tool_calls:
            return tool_calls

        if hasattr(self.domain_service, "parse_tool_calls"):
            domain_tools = self.domain_service.parse_tool_calls(content)
            if domain_tools:
                tool_calls.extend(domain_tools)

        return tool_calls

    def _strip_text_tool_calls(self, content: str) -> str:
        cleaned = TEXT_TOOL_CALL_PATTERN.sub("", content)
        cleaned = TEXT_TOOL_BLOCK_PATTERN.sub("", cleaned)
        cleaned = TEXT_TOOL_INLINE_PATTERN.sub("", cleaned)
        return cleaned.strip()

    def _normalize_tool_args(self, tool_name: str, tool_args: object) -> dict:
        if not isinstance(tool_args, dict):
            return {}

        aliases = self.domain_service.get_tool_aliases(
            # agent_id=self.config.get("agent_id"),
            # db=self.domain_service.db
            agent_id=self.config.get("agent_id")
        ).get(tool_name, {})
        normalized_args: dict = {}
        for key, value in tool_args.items():
            normalized_key = aliases.get(key, key)
            normalized_args[normalized_key] = value
        return normalized_args

    def _execute_tool_calls(
        self,
        history: List[BaseMessage],
        tool_calls: list[dict],
    ) -> list[tuple[str, str]]:
        history.append(AIMessage(content="", tool_calls=tool_calls))
        tool_results: list[tuple[str, str]] = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = self._normalize_tool_args(tool_name, tool_call.get("args") or {})
            tool_id = tool_call.get("id") or tool_name

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool is None:
                unavailable_message = f"The requested tool '{tool_name}' is not available."
                history.append(
                    ToolMessage(
                        content=unavailable_message,
                        name=tool_name or "unknown_tool",
                        tool_call_id=tool_id,
                    )
                )
                tool_results.append((tool_name or "unknown_tool", unavailable_message))
                continue

            try:
                # Provide an arbitrary context logic handling to tools
                ctx = {
                    "db": self.domain_service.db,
                    "config": self.config,
                    "agent_id": self.config.get("agent_id")
                }
                
                if 'context' in tool.func.__code__.co_varnames:
                    tool_args['context'] = ctx
                
                
                result = tool.invoke(tool_args)
            except Exception:
                logger.exception(
                    "Tool invocation failed for '%s' with args %s", tool_name, tool_args
                )
                result = TOOL_FALLBACK_MESSAGES.get(
                    tool_name,
                    "I couldn't complete that request because the tool input was invalid. Please try again.",
                )
            result_text = str(result)
            tool_results.append((tool_name or "unknown_tool", result_text))
            history.append(
                ToolMessage(
                    content=result_text,
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            )
        return tool_results

    def _answer_from_tool_history(self, history: List[BaseMessage]) -> str:
        final_ai: AIMessage = self.llm.invoke(history)
        content = str(final_ai.content or "").strip()
        if not content:
            return "I'm sorry, I couldn't process that. Please try again."

        if TEXT_TOOL_CALL_PATTERN.search(content) or TEXT_TOOL_BLOCK_PATTERN.search(content):
            cleaned = self._strip_text_tool_calls(content)
            if cleaned:
                return cleaned
            return "I found the relevant information, but I couldn't format the final answer clearly. Please try rephrasing your question."

        return content

    def _build_fallback_reply(self, tool_results: list[tuple[str, str]]) -> str:
        meaningful_results = [result for _, result in tool_results if result.strip()]
        if not meaningful_results:
            return "Sorry, can you say that again?"
        if len(meaningful_results) == 1:
            return meaningful_results[0]
        return "\n".join(meaningful_results)

    def run(self, messages: List[BaseMessage]) -> str:
        if hasattr(self.domain_service, "handle_small_talk"):
            small_talk_reply = self.domain_service.handle_small_talk(messages)
            if small_talk_reply:
                return small_talk_reply

        latest_user_message = self._latest_user_message(messages)
        resolved_user_message = self._resolve_user_message_with_context(messages)
        conversation_snippet = self._recent_conversation_snippet(messages)

        intent = self.intent_system.classify_intent(
            messages,
            resolved_user_message,
        )

        if hasattr(self.domain_service, "handle_intent"):
            domain_response = self.domain_service.handle_intent(
                intent=intent,
                llm=self.llm,
                messages=messages,
                resolved_user_message=resolved_user_message,
                conversation_snippet=conversation_snippet,
                tools=self.tools,
                agent_config=self.config,
                strip_func=self._strip_text_tool_calls
            )
            if domain_response:
                if isinstance(domain_response, dict) and domain_response.get("action") == "call_tool":
                    tool_call = {
                        "name": domain_response["tool_name"],
                        "args": domain_response.get("args", {}),
                        "id": "domain_triggered_tool"
                    }
                    results = self._execute_tool_calls(list(messages), [tool_call])
                    return results[0][1] if results else "Error executing forced tool."
                elif isinstance(domain_response, str) and domain_response:
                    return domain_response

        system_prompt = get_system_prompt(self.config.get("prompts"), current_date(), current_datetime())
        history = [SystemMessage(content=system_prompt), *messages]

        if resolved_user_message and resolved_user_message != latest_user_message:
            history.insert(
                1,
                SystemMessage(
                    content=f"Interpret the latest user message in context as: {resolved_user_message}"
                ),
            )

        llm_with_tools = self.llm.bind_tools(self.tools)

        try:
            ai_msg = llm_with_tools.invoke(history)
        except Exception:
            logger.exception("Initial Groq invocation failed")
            if latest_user_message:
                return "Sorry, can you say that again?"
            return "Sorry, can you say that again?"

        explicit_tool_calls = list(getattr(ai_msg, "tool_calls", None) or [])
        if not explicit_tool_calls:
            explicit_tool_calls = self._extract_text_tool_calls(str(ai_msg.content or ""))

        if not explicit_tool_calls:
            direct_reply = self._strip_text_tool_calls(str(ai_msg.content or ""))
            if direct_reply:
                return direct_reply
            return "Sorry, can you say that again?"

        tool_results = self._execute_tool_calls(history, explicit_tool_calls)

        try:
            return self._answer_from_tool_history(history)
        except Exception:
            logger.exception("Final Groq summarization failed after tool execution")
            return self._build_fallback_reply(tool_results)

    def _resolve_user_message_with_context(self, messages: List[BaseMessage]) -> str:
        latest_user_message = self._latest_user_message(messages)
        if not latest_user_message:
            return latest_user_message
        
        conversation_snippet = self._recent_conversation_snippet(messages)
        if not conversation_snippet:
            return latest_user_message

        try:
            rewritten = self.llm.invoke(
                [
                    SystemMessage(content=get_context_resolution_prompt(self.config.get("prompts"))),
                    HumanMessage(
                        content=(
                            f"Recent conversation:\n{conversation_snippet}\n\n"
                            f"Latest user message: {latest_user_message}"
                        )
                    ),
                ]
            )
            rewritten_text = self._strip_text_tool_calls(str(rewritten.content or "").strip())
            if rewritten_text:
                return rewritten_text
        except Exception:
            logger.exception("Context resolution failed")

        return latest_user_message
