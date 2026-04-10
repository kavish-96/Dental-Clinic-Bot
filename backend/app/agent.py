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

from . import crud
from .config import settings
from .datetime_utils import current_date, current_datetime, parse_date_input, parse_time_input
from .tools import get_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are a friendly dental clinic receptionist.
Speak like a real human receptionist.
Keep replies short, warm, and natural.
replies should not contain information that is irrelevent to the question.
Guide the user step by step.
Ask only one thing at a time unless two missing details clearly belong together.

Today's date is {current_date().isoformat()}.
Current local date and time is {current_datetime().strftime("%Y-%m-%d %H:%M")}.

IMPORTANT RULES:
- The answer should be short and precise to question, dont overflood the information.
- Never mention knowledge base, data availability, RAG, functions, tools, internal logic, or system behavior.
- Never expose raw document text or raw tool output.
- If the user mentions whom to contact for booking, appointment, or scheduling in any form (even along with words like contact, call, or reach), prioritize helping them book the appointment directly instead of only giving contact details.
- If something is missing, ask politely for only the missing detail.
- If the user asks something unrelated, fallback by giving valid reasoning.
- If the answer is unknown, fallback by giving valid reasoning.
- The user's mobile number is the primary identifier for appointments.
- Ask for the mobile number only when needed for booking, rescheduling, cancelling, or viewing appointments.
- Reuse the mobile number naturally once the user has shared it.
- For booking or updating, use the date and time the user actually requested. Do not assume a slot unless the user asks for the next or earliest one.
- Never invent a date or time for an appointment.
- If the user gives a day and month without a year, assume the current year.
- Never choose or suggest a past year unless the user explicitly asked for that exact year.
- Only book or update an appointment when the requested date and time are in the future and the slot is available.
- If the user asks for the next, earliest, or next possible slot, call the next-available-slot tool first.
- Do not treat a request to change or update an appointment as a new booking.
- if user input is confusing or ambiguous, ask for clarification.
- Clinic timings are 9am to 8pm.
- If the user asks when they can visit or asks for clinic timings, answer directly using the clinic timings above in a short natural way.
- If the user asks for a time outside range of clinic timings, inform them about the clinic timings mentioned above.
- Use ONLY the provided tools to book, cancel, update, view appointments, or retrieve clinic information.
- Use the exact tool argument names:
  book_appointment(mobile_number, date, time),
  cancel_appointment(mobile_number),
  update_appointment(mobile_number, new_date, new_time),
  view_appointment(mobile_number),
  find_next_available_slot(start_date),
  search_clinic_knowledge(query).
- For any question that depends on clinic documents or website content, always use the clinic knowledge tool before answering.
- Dates may be passed as YYYY-MM-DD or natural dates like 5 April; times in HH:MM or HH:MM AM/PM."""

INTENT_CLASSIFIER_PROMPT = """Classify the user's latest intent for a dental clinic receptionist.

Valid labels:
- booking
- reschedule
- cancel
- clinic_info
- irrelevant
- casual

Rules:
- Use meaning, not keywords only.
- booking includes wanting to come, visit, schedule, book, or asking for the next slot.
- booking also includes asking whom to contact, call, reach, or speak to for an appointment or booking help.
- reschedule includes changing appointment date or time.
- cancel includes cancelling or removing an appointment.
- clinic_info includes services, doctors, timings, pricing, FAQs, symptoms related to dental care, or treatment information.
- casual includes greetings, thanks, who are you, or simple chit-chat.
- irrelevant includes topics unrelated to the clinic.

Reply with only one label."""

APPOINTMENT_COLLECTION_PROMPT = """You are a friendly dental clinic receptionist.
Decide whether the user's latest message means they are actively trying to manage an appointment right now, or just asking a general question.

Reply with only one label:
- collect: the user is actively trying to book, reschedule, cancel, view, or confirm an appointment now
- answer: the user is asking a general question and should be answered normally instead of collecting details

Use meaning, not keywords only.

Examples:
- "I want to book an appointment" -> collect
- "my number is 9876543210" -> collect
- "tomorrow at 5 pm" -> collect
- "whom should I contact for appointment" -> collect
- "what are your appointment timings" -> answer
- "how do I book" -> answer
"""

APPOINTMENT_FOLLOWUP_PROMPT = """You are a friendly dental clinic receptionist.
Write one short natural reply for the user.

Rules:
- Sound human and conversational.
- Ask only for the next missing details if needed.
- Do not mention tools, system behavior, or internal logic.
- Do not invent any appointment date or time.
- If the person already has an upcoming appointment and has not clearly asked to book another one, mention it briefly and ask if they want another appointment.
- Keep it concise.
"""

IRRELEVANT_REDIRECT_PROMPT = """You are a friendly dental clinic receptionist.
The user's message is unrelated to the clinic.

Rules:
- Do not answer or explain the unrelated topic.
- Do not give facts, definitions, reasoning, or educational content about it.
- Reply in short natural sentence only.
- Politely say you cant answer the unrelated topic and redirect to clinic related information or appointment booking.
- Sound human, not robotic.

Reply with only the final user-facing sentence."""

CONTEXT_RESOLUTION_PROMPT = """Rewrite the user's latest message into a standalone dental-clinic message using the recent conversation.

Rules:
- Preserve the user's latest meaning exactly.
- Resolve references like it, them, that, this, those, these, same, yes, no, okay, or follow-up wording from recent conversation.
- Keep the rewritten message concise and natural.
- If the latest message is already standalone, return it unchanged.
- Do not answer the user.
- Output only the rewritten standalone message.
"""

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

APPOINTMENT_CONTACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:who|whom)\s+(?:should\s+i\s+)?(?:contact|call|reach|message)\b.*\b(?:appointment|book|booking|schedule)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:contact|call|reach|message)\b.*\b(?:appointment|book|booking|schedule)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:appointment|book|booking|schedule)\b.*\b(?:contact|call|reach|message)\b",
        re.IGNORECASE,
    ),
]

CONTEXT_REFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(it|them|they|that|this|those|these|there|same)\b", re.IGNORECASE),
    re.compile(r"^\s*(yes|yeah|yep|yup|haan|ha|ok|okay|sure|please|no|nope|nah)\s*[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(what about|how about|and what about|and|also)\b", re.IGNORECASE),
]

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

KNOWLEDGE_SUMMARY_PROMPT = """You are a friendly dental clinic receptionist.
Answer the user's question in a short, natural, human way.
The answer should be short and precise to question, dont overflood the information.
Use only the clinic information provided below.
Use semantic understanding, not exact keyword matching.
If the answer is present with different wording, still answer naturally.
If the answer is only partly clear, share the helpful part briefly.
If the information is not clearly available, fallback by giving valid reasoning.
Never say things like:
- not explicitly mentioned
- provided clinic knowledge
- based on the information
- according to the document
- the document does not contain
- the knowledge does not contain
Never mention PDFs, tools, functions, retrieval, internal logic, or system behavior.
Do not output XML, JSON, lists of sources, or raw copied text."""

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
        "I'm the clinic assistant. I can help book, update, cancel, or view appointments and answer clinic questions.",
    ),
]

TIMINGS_INTENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bclinic timing[s]?\b", re.IGNORECASE),
    re.compile(r"\bopen(?:ing)? time[s]?\b", re.IGNORECASE),
    re.compile(r"\bwhen can i come\b", re.IGNORECASE),
    re.compile(r"\bkab aa sakta hu\b", re.IGNORECASE),
    re.compile(r"\bkab aa sakti hu\b", re.IGNORECASE),
    re.compile(r"\bkab aa sakte hain\b", re.IGNORECASE),
    re.compile(r"\bkhulne ka time\b", re.IGNORECASE),
    re.compile(r"\bclinic kab khulta hai\b", re.IGNORECASE),
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
KNOWLEDGE_BOT_PATTERN = re.compile(
    r"\b(not explicitly mentioned|provided clinic knowledge|based on the information|according to the document|the document does not contain|the knowledge does not contain|not available in the information)\b",
    re.IGNORECASE,
)
EMPTY_KNOWLEDGE_RESULT_PATTERN = re.compile(
    r"\b(no relevant clinic knowledge was found|knowledge base is empty|relevant documents were retrieved, but they did not contain usable text|unable to query the clinic knowledge base right now)\b",
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
}

TOOL_FALLBACK_MESSAGES = {
    "book_appointment": "I need the mobile number, date, and time to book that.",
    "cancel_appointment": "Please share the mobile number for that appointment.",
    "update_appointment": "Please share the mobile number and the new date or time.",
    "view_appointment": "Please share the mobile number for that appointment.",
    "find_next_available_slot": "I couldn't check that just now. Please tell me the date again.",
    "search_clinic_knowledge": "",
}


def _latest_user_message(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def _recent_conversation_snippet(messages: List[BaseMessage], limit: int = 6) -> str:
    lines: list[str] = []
    for message in messages[-limit:]:
        if isinstance(message, HumanMessage):
            lines.append(f"user: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"assistant: {message.content}")
    return "\n".join(lines)


def _needs_context_resolution(messages: List[BaseMessage]) -> bool:
    latest_user_message = _latest_user_message(messages)
    if not latest_user_message:
        return False
    if not any(isinstance(message, AIMessage) for message in messages[:-1]):
        return False
    return any(pattern.search(latest_user_message) for pattern in CONTEXT_REFERENCE_PATTERNS)


def _resolve_user_message_with_context(llm: ChatGroq, messages: List[BaseMessage]) -> str:
    latest_user_message = _latest_user_message(messages)
    if not latest_user_message or not _needs_context_resolution(messages):
        return latest_user_message

    conversation_snippet = _recent_conversation_snippet(messages)
    if not conversation_snippet:
        return latest_user_message

    try:
        rewritten = llm.invoke(
            [
                SystemMessage(content=CONTEXT_RESOLUTION_PROMPT),
                HumanMessage(
                    content=(
                        f"Recent conversation:\n{conversation_snippet}\n\n"
                        f"Latest user message: {latest_user_message}"
                    )
                ),
            ]
        )
        rewritten_text = _strip_text_tool_calls(str(rewritten.content or "").strip())
        if rewritten_text:
            return rewritten_text
    except Exception:
        logger.exception("Context resolution failed")

    return latest_user_message


def _handle_small_talk(messages: List[BaseMessage]) -> str | None:
    latest_user_message = _latest_user_message(messages)
    if not latest_user_message:
        return None
    if _needs_context_resolution(messages):
        return None

    for pattern, reply in CHAT_PATTERNS:
        if pattern.match(latest_user_message):
            return reply
    return None


def _is_clinic_timings_query(messages: List[BaseMessage]) -> bool:
    latest_user_message = _latest_user_message(messages)
    if not latest_user_message:
        return False
    return any(pattern.search(latest_user_message) for pattern in TIMINGS_INTENT_PATTERNS)


def _looks_like_appointment_contact_request(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in APPOINTMENT_CONTACT_PATTERNS)


def _classify_intent(
    llm: ChatGroq,
    messages: List[BaseMessage],
    latest_user_message: str | None = None,
) -> str:
    latest_user_message = latest_user_message or _latest_user_message(messages)
    if not latest_user_message:
        return "casual"
    if _looks_like_appointment_contact_request(latest_user_message):
        return "booking"

    conversation_snippet = _recent_conversation_snippet(messages)

    try:
        classified = llm.invoke(
            [
                SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
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

    valid_labels = {
        "booking",
        "reschedule",
        "cancel",
        "clinic_info",
        "irrelevant",
        "casual",
    }
    return label if label in valid_labels else "casual"


def _should_collect_appointment_details(
    llm: ChatGroq,
    messages: List[BaseMessage],
    latest_user_message: str | None = None,
) -> bool:
    latest_user_message = latest_user_message or _latest_user_message(messages)
    if not latest_user_message:
        return False
    if _looks_like_appointment_contact_request(latest_user_message):
        return True

    conversation_snippet = _recent_conversation_snippet(messages)

    try:
        classified = llm.invoke(
            [
                SystemMessage(content=APPOINTMENT_COLLECTION_PROMPT),
                HumanMessage(
                    content=(
                        f"Conversation:\n{conversation_snippet}\n\n"
                        f"Latest message: {latest_user_message}"
                    )
                ),
            ]
        )
        label = str(classified.content or "").strip().lower()
        return label == "collect"
    except Exception:
        logger.exception("Appointment collection classification failed")
        return False


def _should_force_knowledge_tool(
    messages: List[BaseMessage],
    latest_user_message: str | None = None,
) -> bool:
    raw_message = latest_user_message or _latest_user_message(messages)
    latest_user_message = raw_message.lower()
    if not latest_user_message:
        return False

    if _is_clinic_timings_query(messages):
        return False
    if _looks_like_appointment_contact_request(latest_user_message):
        return False

    if any(keyword in latest_user_message for keyword in APPOINTMENT_KEYWORDS):
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


def _format_existing_appointment(mobile_number: str, appointment) -> str:
    return (
        f"You already have an appointment on {appointment.date.isoformat()} at "
        f"{appointment.time.strftime('%H:%M')} for {mobile_number}. Do you want to book another one?"
    )


def _looks_like_another_booking_confirmation(text: str) -> bool:
    lowered = text.lower().strip()
    return any(
        phrase in lowered
        for phrase in (
            "another appointment",
            "book another",
            "yes another",
            "haan another",
            "yes book",
            "book one more",
        )
    )


def _handle_booking_detail_collection(
    llm: ChatGroq,
    messages: List[BaseMessage],
    db: Session,
    latest_user_message: str | None = None,
) -> str | None:
    latest_user_message = latest_user_message or _latest_user_message(messages)
    if not latest_user_message:
        return None

    if not _should_collect_appointment_details(llm, messages, latest_user_message):
        return None

    mobile_number = _extract_known_mobile(messages)
    date_text = _extract_known_date(messages)
    time_text = _extract_known_time(messages)

    existing_appointment = crud.get_upcoming_appointment_for_mobile(db, mobile_number)
    if (
        existing_appointment is not None
        and not date_text
        and not time_text
        and not _looks_like_another_booking_confirmation(latest_user_message)
    ):
        appointment_summary = _format_existing_appointment(mobile_number, existing_appointment)
    else:
        appointment_summary = ""

    if mobile_number and date_text and time_text:
        return None

    missing_details: list[str] = []
    if not mobile_number:
        missing_details.append("mobile number")
    if mobile_number and not date_text:
        missing_details.append("appointment date")
    if mobile_number and date_text and not time_text:
        missing_details.append("appointment time")

    try:
        followup_ai: AIMessage = llm.invoke(  # type: ignore[assignment]
            [
                SystemMessage(content=APPOINTMENT_FOLLOWUP_PROMPT),
                HumanMessage(
                    content=(
                        f"Latest user message: {latest_user_message}\n"
                        f"Known mobile number: {mobile_number or 'missing'}\n"
                        f"Known appointment date: {date_text or 'missing'}\n"
                        f"Known appointment time: {time_text or 'missing'}\n"
                        f"Existing upcoming appointment: {appointment_summary or 'none'}\n"
                        f"Next thing needed: {missing_details[0] if missing_details else 'nothing'}"
                    )
                ),
            ]
        )
        reply = _strip_text_tool_calls(str(followup_ai.content or "").strip())
        if reply:
            return reply
    except Exception:
        logger.exception("Appointment follow-up generation failed")

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


def _handle_forced_appointment_flow(
    messages: List[BaseMessage],
    tools: list,
    latest_user_message: str | None = None,
) -> str | None:
    latest_user_message = latest_user_message or _latest_user_message(messages)
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
    conversation_snippet: str = "",
) -> str:
    """Use a plain answer-only prompt so tool syntax never leaks into chat."""
    final_ai: AIMessage = llm.invoke(  # type: ignore[assignment]
        [
            SystemMessage(content=KNOWLEDGE_SUMMARY_PROMPT),
            HumanMessage(
                content=(
                    f"Recent conversation:\n{conversation_snippet or 'none'}\n\n"
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
    if KNOWLEDGE_BOT_PATTERN.search(content):
        return _fallback_answer_from_knowledge(user_question, knowledge_text)
    return content


def _has_meaningful_knowledge_result(knowledge_text: str) -> bool:
    cleaned = str(knowledge_text or "").strip()
    if not cleaned:
        return False
    return EMPTY_KNOWLEDGE_RESULT_PATTERN.search(cleaned) is None


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
        return ""

    sentence_split = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentence_split[:3]).strip()
    if not summary:
        summary = cleaned[:400].rsplit(" ", 1)[0].strip()

    if len(summary) > 500:
        summary = summary[:500].rsplit(" ", 1)[0].strip() + "..."

    if KNOWLEDGE_BOT_PATTERN.search(summary):
        return ""

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
        return "Sorry, can you say that again?"
    if len(tool_results) == 1 and tool_results[0][0] == "search_clinic_knowledge":
        knowledge_reply = _fallback_answer_from_knowledge("", tool_results[0][1])
        if knowledge_reply:
            return knowledge_reply
        return "Sorry, can you say that again?"
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
        # llama-3.1-8b-instant
        # llama-3.3-70b-versatile
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )
    tools = get_tools(db)

    small_talk_reply = _handle_small_talk(messages)
    if small_talk_reply:
        return small_talk_reply

    latest_user_message = _latest_user_message(messages)
    resolved_user_message = _resolve_user_message_with_context(llm, messages)
    conversation_snippet = _recent_conversation_snippet(messages)

    intent = _classify_intent(llm, messages, resolved_user_message)
    if intent == "irrelevant":
        knowledge_tool = next(
            (tool for tool in tools if tool.name == "search_clinic_knowledge"), None
        )
        if knowledge_tool is not None and resolved_user_message:
            try:
                knowledge_result = str(
                    knowledge_tool.invoke({"query": resolved_user_message})
                )
                if _has_meaningful_knowledge_result(knowledge_result):
                    return _answer_from_clinic_knowledge(
                        llm,
                        resolved_user_message,
                        knowledge_result,
                        conversation_snippet,
                    )
            except Exception:
                logger.exception("Knowledge lookup during irrelevant-intent handling failed")

        try:
            irrelevant_ai: AIMessage = llm.invoke(  # type: ignore[assignment]
                [
                    SystemMessage(content=IRRELEVANT_REDIRECT_PROMPT),
                    HumanMessage(content=resolved_user_message or latest_user_message),
                ]
            )
            irrelevant_reply = _strip_text_tool_calls(str(irrelevant_ai.content or "").strip())
            if irrelevant_reply:
                return irrelevant_reply
        except Exception:
            logger.exception("Irrelevant-intent reply generation failed")
        return "Sorry, I can help with appointments or clinic info."

    if intent == "booking":
        booking_detail_reply = _handle_booking_detail_collection(
            llm,
            messages,
            db,
            resolved_user_message,
        )
        if booking_detail_reply:
            return booking_detail_reply

    forced_appointment_reply = _handle_forced_appointment_flow(
        messages,
        tools,
        resolved_user_message,
    )
    if forced_appointment_reply:
        return forced_appointment_reply

    # First call: allow the model to decide whether to call tools.
    history: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), *messages]
    if resolved_user_message and resolved_user_message != latest_user_message:
        history.insert(
            1,
            SystemMessage(
                content=f"Interpret the latest user message in context as: {resolved_user_message}"
            ),
        )
    llm_with_tools = llm.bind_tools(tools)

    # Force retrieval for knowledge-style questions so the model does not skip the RAG tool.
    if (intent == "clinic_info" and not _is_clinic_timings_query(messages)) or _should_force_knowledge_tool(messages, resolved_user_message):
        knowledge_tool = next(
            (tool for tool in tools if tool.name == "search_clinic_knowledge"), None
        )
        if knowledge_tool is not None and resolved_user_message:
            try:
                knowledge_result = str(
                    knowledge_tool.invoke({"query": resolved_user_message})
                )
            except Exception:
                logger.exception("Forced clinic knowledge retrieval failed")
                return TOOL_FALLBACK_MESSAGES["search_clinic_knowledge"]

            if not knowledge_result.strip():
                return TOOL_FALLBACK_MESSAGES["search_clinic_knowledge"]

            try:
                return _answer_from_clinic_knowledge(
                    llm,
                    resolved_user_message,
                    knowledge_result,
                    conversation_snippet,
                )
            except Exception:
                logger.exception(
                    "Final Groq summarization failed after forced knowledge retrieval"
                )
                return _fallback_answer_from_knowledge(
                    resolved_user_message,
                    knowledge_result,
                )

    try:
        ai_msg: AIMessage = llm_with_tools.invoke(history)  # type: ignore[assignment]
    except Exception:
        logger.exception("Initial Groq invocation failed")
        latest_user_message = _latest_user_message(messages)
        if latest_user_message:
            return "Sorry, can you say that again?"
        return "Sorry, can you say that again?"

    explicit_tool_calls = list(getattr(ai_msg, "tool_calls", None) or [])
    if not explicit_tool_calls:
        explicit_tool_calls = _extract_text_tool_calls(str(ai_msg.content or ""))

    # If the model answered directly with no tool calls, return that reply.
    if not explicit_tool_calls:
        direct_reply = _strip_text_tool_calls(str(ai_msg.content or ""))
        if direct_reply:
            return direct_reply
        return "Sorry, can you say that again?"

    # Run each requested tool once and add ToolMessage results to the history.
    tool_results = _execute_tool_calls(history, tools, explicit_tool_calls)

    # Second call: ask the model to summarise the tool results for the user.
    try:
        return _answer_from_tool_history(llm, history)
    except Exception:
        logger.exception("Final Groq summarization failed after tool execution")
        return _build_fallback_reply(tool_results)
