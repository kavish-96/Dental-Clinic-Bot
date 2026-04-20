import re
import logging
import json
from typing import List, Union, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from sqlalchemy.orm import Session

from app import crud
from app.datetime_utils import current_date, parse_date_input
from app.agent.prompts import get_response_prompt
from app.domain.dental import config
logger = logging.getLogger(__name__)

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

APPOINTMENT_CONTACT_PATTERNS = [
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

APPOINTMENT_ACTION_PATTERNS = [
    re.compile(
        r"\b(?:i\s+want|want\s+to|need\s+to|would\s+like\s+to|please|help\s+me|can\s+you)\b.*\b(?:book|schedule|reserve|make|set\s+up)\b.*\b(?:appointment|booking|visit|slot)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+need|need|want|looking\s+for|looking\s+to|get|take)\b.*\b(?:appointment|booking|visit|slot)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:book|schedule|reserve|make|set\s+up)\b.*\b(?:appointment|booking|visit|slot)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:appointment|booking|visit|slot)\b.*\b(?:book|schedule|reserve|make|set\s+up)\b",
        re.IGNORECASE,
    ),
]

APPOINTMENT_GENERAL_QUESTION_PATTERNS = [
    re.compile(
        r"\b(?:how|where|what)\b.*\b(?:book|booking|appointment|schedule|visit)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:appointment|booking|clinic)\b.*\b(?:timing|timings|hours|open|close|available)\b",
        re.IGNORECASE,
    ),
]

TOOL_FALLBACK_MESSAGES = {
    "book_appointment": "I need the mobile number, date, and time to book that.",
    "cancel_appointment": "Please share the mobile number for that appointment.",
    "update_appointment": "Please share the mobile number and the new date or time.",
    "view_appointment": "Please share the mobile number for that appointment.",
    "find_next_available_slot": "I couldn't check that just now. Please tell me the date again.",
    "search_clinic_knowledge": "",
}

class DentalService:
    def __init__(self, db: Session):
        self.db = db

    def get_tool_aliases(self, agent_id: str):
        return crud.get_tool_aliases(self.db, agent_id)

    def handle_small_talk(self, messages: List[BaseMessage]) -> str | None:
        latest_user_message = self._latest_user_message(messages)
        if not latest_user_message:
            return None
        
        for pattern, reply in config.CHAT_PATTERNS:
            if pattern.match(latest_user_message):
                return reply
        return None

    def parse_tool_calls(self, content: str) -> list[dict]:
        tool_calls: list[dict] = []
        TEXT_TOOL_INLINE_PATTERN = re.compile(
            r"<function>(?P<name>[a-zA-Z0-9_]+)\((?P<args>.*?)\)</function>",
            re.DOTALL,
        )
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
                        "id": f"text_tool_call_inline_domain_{index}",
                    }
                )

        return tool_calls

    def handle_intent(
        self,
        intent: str,
        llm: ChatGroq,
        messages: List[BaseMessage],
        resolved_user_message: str | None,
        conversation_snippet: str,
        tools: list,
        agent_config: dict,
        strip_func: callable,
    ) -> Union[str, Dict, None]:
        """Central hub for all domain specific actions corresponding to an intent."""
        
        if intent == "irrelevant":
            latest_user_message = self._latest_user_message(messages)
            try:
                irrelevant_ai = llm.invoke(
                    [
                        SystemMessage(content=get_response_prompt(agent_config.get("prompts"))),
                        HumanMessage(content=resolved_user_message or latest_user_message),
                    ]
                )
                irrelevant_reply = strip_func(str(irrelevant_ai.content or "").strip())
                if irrelevant_reply:
                    return irrelevant_reply
            except Exception:
                logger.exception("Irrelevant-intent reply generation failed")
            return "Sorry, I can help with appointments or clinic info."

        if intent in ("knowledge", "clinic_info"):
            return {
                "action": "call_tool",
                "tool_name": "search_clinic_knowledge",
                "args": {"query": resolved_user_message or self._latest_user_message(messages)},
            }
            
        if intent == "booking":
            booking_detail_reply = self.handle_booking_flow(
                llm, messages, resolved_user_message, conversation_snippet, agent_config, strip_func
            )
            if booking_detail_reply:
                return booking_detail_reply

        # Handle forced slots for clinic booking intent overlap
        forced_appointment_reply = self.handle_forced_appointment(
            messages, tools, resolved_user_message
        )
        if forced_appointment_reply:
            return forced_appointment_reply

        return None

    def extract_known_mobile(self, messages: List[BaseMessage]) -> str | None:
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            content = str(getattr(message, "content", "") or "")
            match = MOBILE_NUMBER_PATTERN.search(content)
            if match:
                return match.group(1)
        return None

    def extract_date_text(self, text: str) -> str | None:
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

    def extract_time_text(self, text: str) -> str | None:
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

    def extract_known_date(self, messages: List[BaseMessage]) -> str | None:
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            content = str(getattr(message, "content", "") or "")
            date_text = self.extract_date_text(content)
            if date_text:
                return date_text
        return None

    def extract_known_time(self, messages: List[BaseMessage]) -> str | None:
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            content = str(getattr(message, "content", "") or "")
            time_text = self.extract_time_text(content)
            if time_text:
                return time_text
        return None

    def format_existing_appointment(self, mobile_number: str, appointments: list) -> str:
        if not appointments:
            return ""
        if len(appointments) == 1:
            app = appointments[0]
            return (
                f"You already have an appointment on {app.date.isoformat()} at "
                f"{app.time.strftime('%H:%M')} for {mobile_number}. Do you want to book another one?"
            )
        msg = f"You already have {len(appointments)} upcoming appointments for {mobile_number}:\n"
        for app in appointments:
            msg += f"- {app.date.isoformat()} at {app.time.strftime('%H:%M')}\n"
        msg += "Do you want to book another one?"
        return msg

    def looks_like_another_booking_confirmation(self, text: str) -> bool:
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

    def looks_like_appointment_contact_request(self, text: str) -> bool:
        if not text:
            return False
        return any(pattern.search(text) for pattern in APPOINTMENT_CONTACT_PATTERNS)

    def looks_like_appointment_general_question(self, text: str) -> bool:
        if not text:
            return False
        return any(pattern.search(text) for pattern in APPOINTMENT_GENERAL_QUESTION_PATTERNS)

    def looks_like_appointment_action(self, text: str) -> bool:
        if not text:
            return False
        if self.looks_like_next_slot_request(text):
            return True
        return any(pattern.search(text) for pattern in APPOINTMENT_ACTION_PATTERNS)

    def should_collect_appointment_details(
        self,
        llm: ChatGroq,
        messages: List[BaseMessage],
        latest_user_message: str | None = None,
        conversation_snippet: str = "",
        agent_config: dict = None
    ) -> bool:
        latest_user_message = latest_user_message or self._latest_user_message(messages)
        if not latest_user_message:
            return False
        if self.looks_like_appointment_contact_request(latest_user_message):
            return True
        if self.looks_like_appointment_general_question(latest_user_message):
            return False
        if (
            MOBILE_NUMBER_PATTERN.search(latest_user_message)
            or self.extract_date_text(latest_user_message)
            or self.extract_time_text(latest_user_message)
        ):
            return True
        if self.looks_like_another_booking_confirmation(latest_user_message):
            return True

        return self.looks_like_appointment_action(latest_user_message)

    def handle_booking_flow(
        self,
        llm: ChatGroq,
        messages: List[BaseMessage],
        latest_user_message: str | None = None,
        conversation_snippet: str = "",
        agent_config: dict = None,
        strip_func: callable = None,
    ) -> str | None:
        latest_user_message = latest_user_message or self._latest_user_message(messages)
        if not latest_user_message:
            return None

        if not self.should_collect_appointment_details(llm, messages, latest_user_message, conversation_snippet, agent_config):
            return None

        mobile_number = self.extract_known_mobile(messages)
        date_text = self.extract_known_date(messages)
        time_text = self.extract_known_time(messages)

        existing_appointments = crud.get_upcoming_appointments_for_mobile(self.db, mobile_number) if mobile_number else []
        if (
            existing_appointments
            and not date_text
            and not time_text
            and not self.looks_like_another_booking_confirmation(latest_user_message)
        ):
            appointment_summary = self.format_existing_appointment(mobile_number, existing_appointments)
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

        prompts_cfg = agent_config.get("prompts")
        try:
            followup_ai = llm.invoke(
                [
                    SystemMessage(content=get_response_prompt(prompts_cfg)),
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
            reply = strip_func(str(followup_ai.content or "").strip()) if strip_func else str(followup_ai.content or "").strip()
            if reply:
                return reply
        except Exception:
            logger.exception("Appointment follow-up generation failed")

        return None

    def parse_next_slot_result(self, text: str) -> tuple[str, str] | None:
        match = NEXT_SLOT_RESULT_PATTERN.search(text)
        if not match:
            return None
        return match.group("date"), match.group("time")

    def looks_like_next_slot_request(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "next slot" in lowered
            or "next available slot" in lowered
            or "earliest slot" in lowered
            or "earliest available" in lowered
        )

    def looks_like_book_next_slot_request(self, text: str) -> bool:
        lowered = text.lower()
        return self.looks_like_next_slot_request(lowered) and any(
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

    def handle_forced_appointment(
        self,
        messages: List[BaseMessage],
        tools: list,
        latest_user_message: str | None = None,
    ) -> Union[str, Dict, None]:
        latest_user_message = latest_user_message or self._latest_user_message(messages)
        if not latest_user_message:
            return None

        if not self.looks_like_next_slot_request(latest_user_message):
            return None

        start_date = self.extract_date_text(latest_user_message)
        if start_date:
            try:
                start_date = parse_date_input(start_date).isoformat()
            except ValueError:
                start_date = current_date().isoformat()
        else:
            start_date = current_date().isoformat()

        # Check if booking is needed
        if self.looks_like_book_next_slot_request(latest_user_message):
            mobile_number = self.extract_known_mobile(messages)
            if not mobile_number:
                return {
                    "action": "call_tool",
                    "tool_name": "find_next_available_slot",
                    "args": {"start_date": start_date}
                }
                
            # Execute slot lookup immediately to check for booking (We still have to invoke next_slot internally to get the slot first to book it, 
            # but wait, the prompt says stop executing tools. Let AgentCore handle it.
            # But we can't book unless we have the date/time of the next slot. So we just trigger a lookup tool and we can book later.
            # No, if we want to "book the next slot", the prompt implies we return action "book_appointment", but we don't know the exact slot yet unless we invoke it.
            # Let's just return the tool action for next slot lookup.)

        return {
            "action": "call_tool",
            "tool_name": "find_next_available_slot",
            "args": {"start_date": start_date}
        }

    def _latest_user_message(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content).strip()
        return ""
