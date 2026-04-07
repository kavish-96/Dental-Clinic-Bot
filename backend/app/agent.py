import json
import logging
import re
from datetime import datetime
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from sqlalchemy.orm import Session

from .config import settings
from .datetime_utils import current_date, current_datetime, parse_date_input, parse_time_input
from .tools import get_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are a friendly virtual receptionist for a dental clinic.
You help users book, cancel, update, and view appointments, and you can also share the current weather for a city.
You can also answer clinic knowledge questions by using the clinic knowledge tool when the user asks about policies, services, doctors, timings, pricing, FAQs, or other informational content from clinic documents or website pages.

Today's date is {current_date().isoformat()}.
Current local date and time is {current_datetime().strftime("%Y-%m-%d %H:%M")}.

IMPORTANT RULES:
- The user's MOBILE NUMBER is the primary identifier for all appointments.
- Ask for the user's mobile number only when they want to book, cancel, update, or view appointments
  and you do not yet know it.
- Once you know the mobile number from the conversation, remember it and REUSE it
  for all later booking, cancelling, updating and viewing actions in this chat.
- For booking or updating, use the date and time the user actually requested. Do not invent or assume a slot.
- If the user gives a day and month without a year, assume the current year.
- Never choose or suggest a past year unless the user explicitly asked for that exact year.
- If the user has not provided a date or time for booking or updating, ask for the missing detail.
- If the user asks for the next, earliest, or next possible slot, call the next-available-slot tool first instead of guessing.
- Only book or update an appointment when the requested date and time are in the future and the slot is available.
- If the user is booking or updating and some details are missing, ask only for the missing details instead of guessing.
- if the user asks about anything out of your answering capacity, back off with simple and short reply that you cant provide information on that.
- the clinic timings are 9:00 to 20:00, so if user asks for time before 9:00 or after 20:00 reply by providing the office timings
- Do not treat a request to change or update an appointment as a new booking.
- Weather questions do not need a mobile number.
- Use ONLY the provided tools to book, cancel, update, view appointments, get current weather, or retrieve clinic knowledge.
- Use the exact tool argument names:
  book_appointment(mobile_number, date, time),
  cancel_appointment(mobile_number),
  update_appointment(mobile_number, new_date, new_time),
  view_appointment(mobile_number),
  find_next_available_slot(start_date),
  get_current_weather(city),
  search_clinic_knowledge(query).
- For any question that depends on clinic documents or website content, ALWAYS call the clinic knowledge tool before answering.
- Dates may be passed as YYYY-MM-DD or natural dates like 5 April; times in HH:MM or HH:MM AM/PM.
- Keep replies short, clear, and conversational like a real clinic receptionist."""

KNOWLEDGE_KEYWORDS = {
    "pdf",
    "website",
    "document",
    "guide",
    "policy",
    "policies",
    "faq",
    "faqs",
    "service",
    "services",
    "doctor",
    "doctors",
    "timing",
    "timings",
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
    "tariff",
    "tariffs",
    "quote",
    "quotation",
    "estimate",
    "estimates",
    "price list",
    "price sheet",
    "treatment",
    "treatments",
    "clinic",
    "offer",
    "offers",
    "offered",
    "available",
    "availability",
    "procedure",
    "procedures",
    "braces",
    "aligners",
    "orthodontics",
    "orthodontist",
    "implant",
    "implants",
    "implantology",
    "implantologist",
    "root canal",
    "rct",
    "extraction",
    "whitening",
    "filling",
    "cleaning",
    "scaling",
    "cavity",
    "dentist",
    "specialist",
    "specialists",
    "doctor available",
    "teeth",
    "tooth",
    "daant",
    "dant",
    "karvane",
    "karwana",
    "karvana",
}

APPOINTMENT_KEYWORDS = {
    "appointment",
    "book",
    "booking",
    "cancel",
    "update",
    "reschedule",
    "slot",
    "mobile",
    "number",
}

WEATHER_KEYWORDS = {
    "weather",
    "temperature",
    "forecast",
    "rain",
    "wind",
}

KNOWLEDGE_INTENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bdo you have\b", re.IGNORECASE),
    re.compile(r"\bi want\b", re.IGNORECASE),
    re.compile(r"\bcan you do\b", re.IGNORECASE),
    re.compile(r"\bdo you do\b", re.IGNORECASE),
    re.compile(r"\bis there\b", re.IGNORECASE),
    re.compile(r"\bavailable\b", re.IGNORECASE),
    re.compile(r"\bbraces\b", re.IGNORECASE),
    re.compile(r"\baligners?\b", re.IGNORECASE),
    re.compile(r"\bimplantologist\b", re.IGNORECASE),
    re.compile(r"\bimplant(?:s|ology)?\b", re.IGNORECASE),
    re.compile(r"\borthodont(?:ics|ist)?\b", re.IGNORECASE),
    re.compile(r"\broot canal\b", re.IGNORECASE),
    re.compile(r"\bteeth\b", re.IGNORECASE),
    re.compile(r"\btooth\b", re.IGNORECASE),
    re.compile(r"\bdaant\b", re.IGNORECASE),
    re.compile(r"\bdant\b", re.IGNORECASE),
]

KNOWLEDGE_SUMMARY_PROMPT = """You are a dental clinic assistant.
Answer the user's question using only the clinic knowledge provided below.
Use semantic understanding, not exact keyword matching.
If the knowledge contains relevant information with different wording, use it.
If the answer is partially supported, give the supported answer clearly and briefly.
Never say you do not have access to the clinic knowledge, documents, PDF, pricing, or information when clinic knowledge is provided below.
Never ask the user to contact the clinic or visit the clinic if the answer is already present in the provided clinic knowledge.
Do not mention tools, retrieval, function calls, or internal system behavior.
Do not output XML, JSON, or function tags."""

CHAT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"^\s*(thanks|thank you|thankyou|thx|tysm)\s*[.!]*\s*$", re.IGNORECASE
        ),
        "You're welcome. If you'd like, I can also help with appointments and clinic information.",
    ),
    (
        re.compile(
            r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*[.!]*\s*$",
            re.IGNORECASE,
        ),
        "Hello. How can I help you today with your appointment or clinic questions?",
    ),
    (
        re.compile(
            r"^\s*(bye|goodbye|see you|ok bye|okay bye)\s*[.!]*\s*$", re.IGNORECASE
        ),
        "Take care. If you need anything later, I'm here to help.",
    ),
    (
        re.compile(
            r"^\s*(ok|okay|cool|alright|fine|got it|noted)\s*[.!]*\s*$", re.IGNORECASE
        ),
        "Sure. Let me know what you'd like to do next.",
    ),
    (
        re.compile(r"^\s*(who are you|what can you do)\s*[?!]*\s*$", re.IGNORECASE),
        "I'm the clinic assistant. I can help book, update, cancel, or view appointments, answer clinic questions, and share the current weather.",
    ),
]

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
RAG_SOURCE_BLOCK_PATTERN = re.compile(
    r"\[Source\s+\d+:\s*(?P<source>[^\]]+)\]\s*(?P<content>.*?)(?=\n\s*\[Source\s+\d+:|\Z)",
    re.DOTALL,
)
ACCESS_DENIAL_PATTERN = re.compile(
    r"\b(i do not have access|i don't have access|unable to provide|unable to access|contact (us|the clinic) directly|visit (us|the clinic))\b",
    re.IGNORECASE,
)
MOBILE_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{10})(?!\d)")
ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NATURAL_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)|(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)
RELATIVE_DATE_PATTERN = re.compile(
    r"\b(today|tomorrow|day after tomorrow)\b",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b([01]?\d|2[0-3]):([0-5]\d)\b",
    re.IGNORECASE,
)
NEXT_SLOT_RESULT_PATTERN = re.compile(
    r"next available slot is (?P<date>\d{4}-\d{2}-\d{2}) at (?P<time>\d{2}:\d{2})",
    re.IGNORECASE,
)

TOOL_ARG_ALIASES = {
    "book_appointment": {
        "mobile": "mobile_number",
        "mobileNumber": "mobile_number",
        "phone": "mobile_number",
        "appointment_date": "date",
        "appointment_time": "time",
    },
    "cancel_appointment": {
        "mobile": "mobile_number",
        "mobileNumber": "mobile_number",
        "phone": "mobile_number",
    },
    "update_appointment": {
        "mobile": "mobile_number",
        "mobileNumber": "mobile_number",
        "phone": "mobile_number",
        "date": "new_date",
        "time": "new_time",
        "appointment_date": "new_date",
        "appointment_time": "new_time",
    },
    "view_appointment": {
        "mobile": "mobile_number",
        "mobileNumber": "mobile_number",
        "phone": "mobile_number",
    },
    "find_next_available_slot": {
        "date": "start_date",
        "startDate": "start_date",
        "day": "start_date",
    },
    "get_current_weather": {
        "location": "city",
        "place": "city",
    },
}

TOOL_FALLBACK_MESSAGES = {
    "book_appointment": "I couldn't complete the booking request because some appointment details were missing or invalid. Please share the date, time, and mobile number again.",
    "cancel_appointment": "I couldn't cancel the appointment because I need a valid mobile number.",
    "update_appointment": "I couldn't update the appointment because some details were missing or invalid. Please share the mobile number and the new date or time again.",
    "view_appointment": "I couldn't look up the appointment because I need a valid mobile number.",
    "find_next_available_slot": "I couldn't check the next available slot just now. Please tell me the preferred date again and I'll retry.",
    "get_current_weather": "I couldn't check the weather just now. Please share the city again and I'll retry.",
    "search_clinic_knowledge": "I couldn't retrieve the clinic information just now. Please try asking the question once more.",
}


def _latest_user_message(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def _handle_small_talk(messages: List[BaseMessage]) -> str | None:
    latest_user_message = _latest_user_message(messages)
    if not latest_user_message:
        return None

    for pattern, reply in CHAT_PATTERNS:
        if pattern.match(latest_user_message):
            return reply
    return None


def _should_force_knowledge_tool(messages: List[BaseMessage]) -> bool:
    latest_user_message = _latest_user_message(messages).lower()
    if not latest_user_message:
        return False

    if any(keyword in latest_user_message for keyword in APPOINTMENT_KEYWORDS):
        return False
    if any(keyword in latest_user_message for keyword in WEATHER_KEYWORDS):
        return False

    if "?" in latest_user_message:
        return True

    if any(keyword in latest_user_message for keyword in KNOWLEDGE_KEYWORDS):
        return True

    return any(pattern.search(latest_user_message) for pattern in KNOWLEDGE_INTENT_PATTERNS)


def _extract_text_tool_calls(content: str) -> list[dict]:
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

    for index, match in enumerate(TEXT_TOOL_INLINE_PATTERN.finditer(content), start=1):
        tool_name = match.group("name")
        raw_args = match.group("args").strip()
        tool_args: dict[str, str] = {}

        if tool_name == "search_clinic_knowledge" and raw_args:
            cleaned_arg = raw_args.strip().strip("\"'")
            tool_args = {"query": cleaned_arg}
        elif tool_name == "find_next_available_slot" and raw_args:
            try:
                parsed_args = json.loads(raw_args)
                if isinstance(parsed_args, dict):
                    tool_args = {str(key): str(value) for key, value in parsed_args.items()}
            except json.JSONDecodeError:
                cleaned_arg = raw_args.strip().strip("\"'")
                if cleaned_arg:
                    tool_args = {"start_date": cleaned_arg}
        elif tool_name == "book_appointment" and raw_args:
            try:
                parsed_args = json.loads(raw_args)
                if isinstance(parsed_args, dict):
                    tool_args = {str(key): str(value) for key, value in parsed_args.items()}
            except json.JSONDecodeError:
                tool_args = {}

        if tool_args:
            tool_calls.append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "id": f"text_tool_call_inline_{index}",
                }
            )

    return tool_calls


def _strip_text_tool_calls(content: str) -> str:
    cleaned = TEXT_TOOL_CALL_PATTERN.sub("", content)
    cleaned = TEXT_TOOL_BLOCK_PATTERN.sub("", cleaned)
    cleaned = TEXT_TOOL_INLINE_PATTERN.sub("", cleaned)
    return cleaned.strip()


def _normalize_tool_args(tool_name: str, tool_args: object) -> dict:
    if not isinstance(tool_args, dict):
        return {}

    aliases = TOOL_ARG_ALIASES.get(tool_name, {})
    normalized_args: dict = {}
    for key, value in tool_args.items():
        normalized_key = aliases.get(key, key)
        normalized_args[normalized_key] = value
    return normalized_args


def _find_tool(tools: list, tool_name: str):
    return next((tool for tool in tools if tool.name == tool_name), None)


def _extract_known_mobile(messages: List[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = str(getattr(message, "content", "") or "")
        match = MOBILE_NUMBER_PATTERN.search(content)
        if match:
            return match.group(1)
    return None


def _extract_date_text(text: str) -> str | None:
    iso_match = ISO_DATE_PATTERN.search(text)
    if iso_match:
        return iso_match.group(0)

    relative_match = RELATIVE_DATE_PATTERN.search(text)
    if relative_match:
        return relative_match.group(1)

    natural_match = NATURAL_DATE_PATTERN.search(text)
    if natural_match:
        return natural_match.group(0)

    return None


def _extract_time_text(text: str) -> str | None:
    match = TIME_PATTERN.search(text)
    if not match:
        return None

    if match.group(1):
        hour = match.group(1)
        minute = match.group(2) or "00"
        meridiem = (match.group(3) or "").upper()
        return f"{hour}:{minute} {meridiem}".strip()

    if match.group(4) and match.group(5):
        return f"{match.group(4)}:{match.group(5)}"

    return None


def _extract_known_date(messages: List[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = str(getattr(message, "content", "") or "")
        date_text = _extract_date_text(content)
        if date_text:
            return date_text
    return None


def _extract_known_time(messages: List[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = str(getattr(message, "content", "") or "")
        time_text = _extract_time_text(content)
        if time_text:
            return time_text
    return None


def _parse_next_slot_result(text: str) -> tuple[str, str] | None:
    match = NEXT_SLOT_RESULT_PATTERN.search(text)
    if not match:
        return None
    return match.group("date"), match.group("time")


def _looks_like_next_slot_request(text: str) -> bool:
    lowered = text.lower()
    return (
        "next slot" in lowered
        or "next available slot" in lowered
        or "earliest slot" in lowered
        or "earliest available" in lowered
    )


def _looks_like_book_next_slot_request(text: str) -> bool:
    lowered = text.lower()
    return _looks_like_next_slot_request(lowered) and any(
        phrase in lowered
        for phrase in (
            "book the",
            "book next",
            "book me",
            "schedule the",
            "take the next",
            "reserve the next",
        )
    )


def _handle_forced_appointment_flow(messages: List[BaseMessage], tools: list) -> str | None:
    latest_user_message = _latest_user_message(messages)
    if not latest_user_message:
        return None

    if not _looks_like_next_slot_request(latest_user_message):
        return None

    next_slot_tool = _find_tool(tools, "find_next_available_slot")
    if next_slot_tool is None:
        return TOOL_FALLBACK_MESSAGES["find_next_available_slot"]

    start_date = _extract_date_text(latest_user_message)
    if start_date:
        try:
            start_date = parse_date_input(start_date).isoformat()
        except ValueError:
            start_date = current_date().isoformat()
    else:
        start_date = current_date().isoformat()

    try:
        next_slot_result = str(next_slot_tool.invoke({"start_date": start_date}))
    except Exception:
        logger.exception("Forced next-slot lookup failed")
        return TOOL_FALLBACK_MESSAGES["find_next_available_slot"]

    if not _looks_like_book_next_slot_request(latest_user_message):
        return next_slot_result

    mobile_number = _extract_known_mobile(messages)
    if not mobile_number:
        return (
            f"{next_slot_result} If you'd like me to book it, please share your mobile number."
        )

    parsed_slot = _parse_next_slot_result(next_slot_result)
    if not parsed_slot:
        return next_slot_result

    book_tool = _find_tool(tools, "book_appointment")
    if book_tool is None:
        return TOOL_FALLBACK_MESSAGES["book_appointment"]

    slot_date, slot_time = parsed_slot
    try:
        booking_result = str(
            book_tool.invoke(
                {
                    "mobile_number": mobile_number,
                    "date": slot_date,
                    "time": slot_time,
                }
            )
        )
    except Exception:
        logger.exception("Forced next-slot booking failed")
        return TOOL_FALLBACK_MESSAGES["book_appointment"]

    return booking_result


def _answer_from_tool_history(llm: ChatGroq, history: List[BaseMessage]) -> str:
    final_ai: AIMessage = llm.invoke(history)  # type: ignore[assignment]
    content = str(final_ai.content or "").strip()
    if not content:
        return "I'm sorry, I couldn't process that. Please try again."

    if TEXT_TOOL_CALL_PATTERN.search(content) or TEXT_TOOL_BLOCK_PATTERN.search(
        content
    ):
        cleaned = _strip_text_tool_calls(content)
        if cleaned:
            return cleaned
        return "I found the relevant information, but I couldn't format the final answer clearly. Please try rephrasing your question."

    return content


def _answer_from_clinic_knowledge(
    llm: ChatGroq,
    user_question: str,
    knowledge_text: str,
) -> str:
    """Use a plain answer-only prompt so tool syntax never leaks into chat."""
    final_ai: AIMessage = llm.invoke(  # type: ignore[assignment]
        [
            SystemMessage(content=KNOWLEDGE_SUMMARY_PROMPT),
            HumanMessage(
                content=(
                    f"User question: {user_question}\n\n"
                    f"Clinic knowledge:\n{knowledge_text}"
                )
            ),
        ]
    )
    content = _strip_text_tool_calls(str(final_ai.content or "").strip())
    if not content:
        return _fallback_answer_from_knowledge(user_question, knowledge_text)
    if ACCESS_DENIAL_PATTERN.search(content):
        return _fallback_answer_from_knowledge(user_question, knowledge_text)
    return content


def _extract_knowledge_source_blocks(knowledge_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for match in RAG_SOURCE_BLOCK_PATTERN.finditer(knowledge_text):
        source = match.group("source").strip()
        content = match.group("content").strip()
        if content:
            blocks.append((source, content))
    return blocks


def _clean_knowledge_content_for_user(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Use the clinic knowledge below"):
            continue
        if stripped.startswith("Use semantic understanding."):
            continue
        if stripped.startswith("Infer answers from related concepts"):
            continue
        if stripped.startswith("Do not say the answer is not found"):
            continue
        if stripped.startswith("If the answer is only partially covered"):
            continue
        if stripped.startswith("User question:"):
            continue
        if stripped.startswith("Expanded retrieval query:"):
            continue
        if stripped.startswith("[Source "):
            continue
        cleaned_lines.append(stripped)

    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\s*\|\s*", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _fallback_answer_from_knowledge(user_question: str, knowledge_text: str) -> str:
    """Never expose raw RAG prompts or chunk headers to the user."""
    blocks = _extract_knowledge_source_blocks(knowledge_text)
    if blocks:
        # Prefer clinic PDF content when it exists.
        pdf_blocks = [content for source, content in blocks if ".pdf" in source.lower()]
        candidate_text = " ".join(pdf_blocks[:2]) if pdf_blocks else " ".join(
            content for _, content in blocks[:2]
        )
    else:
        candidate_text = knowledge_text

    cleaned = _clean_knowledge_content_for_user(candidate_text)
    if not cleaned:
        return "I found relevant clinic information, but I couldn't format it clearly. Please try asking the question once more."

    sentence_split = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentence_split[:3]).strip()
    if not summary:
        summary = cleaned[:400].rsplit(" ", 1)[0].strip()

    if len(summary) > 500:
        summary = summary[:500].rsplit(" ", 1)[0].strip() + "..."

    return summary


def _execute_tool_calls(
    history: List[BaseMessage],
    tools: list,
    tool_calls: list[dict],
) -> list[tuple[str, str]]:
    history.append(AIMessage(content="", tool_calls=tool_calls))
    tool_results: list[tuple[str, str]] = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = _normalize_tool_args(tool_name, tool_call.get("args") or {})
        tool_id = tool_call.get("id") or tool_name

        tool = next((t for t in tools if t.name == tool_name), None)
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


def _build_fallback_reply(tool_results: list[tuple[str, str]]) -> str:
    meaningful_results = [result for _, result in tool_results if result.strip()]
    if not meaningful_results:
        return "I'm sorry, I couldn't find a clear answer for that just now. If you rephrase the question or ask in a bit more detail, I'll do my best to help."
    if len(tool_results) == 1 and tool_results[0][0] == "search_clinic_knowledge":
        return _fallback_answer_from_knowledge("", tool_results[0][1])
    if len(meaningful_results) == 1:
        return meaningful_results[0]
    return "\n".join(meaningful_results)


def run_agent(messages: List[BaseMessage], db: Session) -> str:
    """
    Simple tool-calling loop:
    - send chat history to the model with tools attached
    - if the model requests tools, run them
    - send tool results back to the model to get the final answer
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set in environment or .env")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )
    tools = get_tools(db)

    small_talk_reply = _handle_small_talk(messages)
    if small_talk_reply:
        return small_talk_reply

    forced_appointment_reply = _handle_forced_appointment_flow(messages, tools)
    if forced_appointment_reply:
        return forced_appointment_reply

    # First call: allow the model to decide whether to call tools.
    history: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), *messages]
    llm_with_tools = llm.bind_tools(tools)

    # Force retrieval for knowledge-style questions so the model does not skip the RAG tool.
    if _should_force_knowledge_tool(messages):
        latest_user_message = _latest_user_message(messages)
        knowledge_tool = next(
            (tool for tool in tools if tool.name == "search_clinic_knowledge"), None
        )
        if knowledge_tool is not None and latest_user_message:
            try:
                knowledge_result = str(
                    knowledge_tool.invoke({"query": latest_user_message})
                )
            except Exception:
                logger.exception("Forced clinic knowledge retrieval failed")
                return TOOL_FALLBACK_MESSAGES["search_clinic_knowledge"]

            if not knowledge_result.strip():
                return TOOL_FALLBACK_MESSAGES["search_clinic_knowledge"]

            try:
                return _answer_from_clinic_knowledge(
                    llm,
                    latest_user_message,
                    knowledge_result,
                )
            except Exception:
                logger.exception(
                    "Final Groq summarization failed after forced knowledge retrieval"
                )
                return _fallback_answer_from_knowledge(
                    latest_user_message,
                    knowledge_result,
                )

    try:
        ai_msg: AIMessage = llm_with_tools.invoke(history)  # type: ignore[assignment]
    except Exception:
        logger.exception("Initial Groq invocation failed")
        latest_user_message = _latest_user_message(messages)
        if latest_user_message:
            return "I'm sorry, I can't help you with that. I can help with appointments or clinic information, try asking again."
        return "I'm sorry, I couldn't respond properly just now. Please try again."

    explicit_tool_calls = list(getattr(ai_msg, "tool_calls", None) or [])
    if not explicit_tool_calls:
        explicit_tool_calls = _extract_text_tool_calls(str(ai_msg.content or ""))

    # If the model answered directly with no tool calls, return that reply.
    if not explicit_tool_calls:
        direct_reply = _strip_text_tool_calls(str(ai_msg.content or ""))
        if direct_reply:
            return direct_reply
        return "I'm sorry, I couldn't put that into a clear reply. Please try rephrasing it."

    # Run each requested tool once and add ToolMessage results to the history.
    tool_results = _execute_tool_calls(history, tools, explicit_tool_calls)

    # Second call: ask the model to summarise the tool results for the user.
    try:
        return _answer_from_tool_history(llm, history)
    except Exception:
        logger.exception("Final Groq summarization failed after tool execution")
        return _build_fallback_reply(tool_results)
