import json
from datetime import date, datetime, time, timedelta

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import crud
from app.datetime_utils import current_date, current_datetime, parse_date_input, parse_time_input
from app.domain.dental.service import is_valid_date
from app.rag import get_rag_response

# --- Pydantic schemas for tool args ---

class BookAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number (primary identifier)")
    date: str = Field(description="Appointment date. Accepts YYYY-MM-DD or natural dates like 5 April.")
    time: str = Field(description="Appointment time in HH:MM or HH:MM AM/PM format")


class CancelAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number")
    date: str | None = Field(default=None, description="Date of the appointment to cancel (YYYY-MM-DD)")
    time: str | None = Field(default=None, description="Time of the appointment to cancel (HH:MM)")


class UpdateAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number")
    old_date: str | None = Field(default=None, description="Date of the existing appointment (YYYY-MM-DD)")
    old_time: str | None = Field(default=None, description="Time of the existing appointment (HH:MM)")
    new_date: str | None = Field(default=None, description="New date, e.g. YYYY-MM-DD or 5 April")
    new_time: str | None = Field(default=None, description="New time HH:MM or HH:MM AM/PM")


class ViewAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number")


class NextAvailableSlotInput(BaseModel):
    start_date: str | None = Field(
        default=None,
        description="Optional date to start searching from. Accepts YYYY-MM-DD or natural dates like 5 April.",
    )


class ClinicKnowledgeInput(BaseModel):
    query: str = Field(description="Question about clinic information from PDFs or website content")


def _resolve_rag_config(context: dict | None, db: Session) -> dict:
    agent_config = context.get("config", {}) if context else {}
    if isinstance(agent_config, dict) and "rag" in agent_config:
        return agent_config["rag"]
    if isinstance(agent_config, dict) and "rag_synonyms" in agent_config:
        return agent_config

    from app.agent.config import DynamicAgentConfig
    dyn_config = DynamicAgentConfig("dental_bot", db)
    return dyn_config.rag



def _parse_time(s: str) -> time:
    return parse_time_input(s)


def _parse_date(s: str) -> date:
    return parse_date_input(s)


def _candidate_slots_for_day(target_date: date) -> list[time]:
    return [time(hour=hour) for hour in range(9, 18)]

def book_appointment(db: Session):
    @tool(args_schema=BookAppointmentInput)
    def book_appointment_tool(
        mobile_number: str,
        date: str,
        time: str,
        context: dict = None
    ) -> str:
        """Book a new dental appointment for a mobile number."""
        if not is_valid_date(date):
            return "That date is not valid. Please provide a correct date."

        try:
            d = _parse_date(date)
            t = _parse_time(time)
        except ValueError as e:
            return str(e)
        if crud.is_slot_in_past(d, t):
            return "Appointments can only be booked for a future date and time."
        if not crud.is_slot_available(db, d, t):
            return f"The slot on {d.isoformat()} at {t.strftime('%H:%M')} is already booked. Please choose another time."

        apt = crud.create_appointment(db, mobile_number=mobile_number, date_val=d, time_val=t)
        if not apt:
            return "I couldn't book that appointment. Please try a different future time slot."
        return (
            f"Appointment booked for {mobile_number} on {apt.date.isoformat()} "
            f"at {apt.time.strftime('%H:%M')}."
        )
    book_appointment_tool.name = "book_appointment"
    return book_appointment_tool

def cancel_appointment(db: Session):
    @tool(args_schema=CancelAppointmentInput)
    def cancel_appointment_tool(mobile_number: str, date: str | None = None, time: str | None = None, context: dict = None) -> str:
        """Cancel an appointment for this mobile number."""
        upcoming = crud.get_upcoming_appointments_for_mobile(db, mobile_number)
        if not upcoming:
            return f"No upcoming appointments found for {mobile_number}."
        
        if not date or not time:
            msg = f"Upcoming appointments for {mobile_number}:\n"
            for u in upcoming:
                msg += f"- {u.date.isoformat()} at {u.time.strftime('%H:%M')}\n"
            if len(upcoming) == 1:
                msg += "\nPlease confirm that you want to cancel this appointment by specifying its date and time."
            else:
                msg += "\nPlease specify the date and time of the appointment you want to cancel."
            return msg
            
        try:
            d = _parse_date(date)
            t = _parse_time(time)
        except ValueError as e:
            return str(e)

        ok = crud.cancel_specific_appointment(db, mobile_number, d, t)
        if not ok:
            return f"No appointment found on {d.isoformat()} at {t.strftime('%H:%M')} for {mobile_number} to cancel."
        return f"Your appointment for {mobile_number} on {d.isoformat()} at {t.strftime('%H:%M')} has been cancelled."
    cancel_appointment_tool.name = "cancel_appointment"
    return cancel_appointment_tool

def update_appointment(db: Session):
    @tool(args_schema=UpdateAppointmentInput)
    def update_appointment_tool(
        mobile_number: str,
        old_date: str | None = None,
        old_time: str | None = None,
        new_date: str | None = None,
        new_time: str | None = None,
        context: dict = None
    ) -> str:
        """Update an appointment's date and/or time for this mobile number."""
        upcoming = crud.get_upcoming_appointments_for_mobile(db, mobile_number)
        if not upcoming:
            return f"No upcoming appointments found for {mobile_number}."

        if not old_date or not old_time or (not new_date and not new_time):
            msg = f"Upcoming appointments for {mobile_number}:\n"
            for u in upcoming:
                msg += f"- {u.date.isoformat()} at {u.time.strftime('%H:%M')}\n"
            
            if not old_date or not old_time:
                msg += "\nPlease specify the current date and time of the appointment you'd like to update, AND the new date and time."
            elif not new_date and not new_time:
                msg += f"\nYou selected the appointment on {old_date} at {old_time}. What is the new date and time?"
            return msg

        try:
            od = _parse_date(old_date)
            ot = _parse_time(old_time)
            nd = _parse_date(new_date) if new_date else None
            nt = _parse_time(new_time) if new_time else None
        except ValueError as e:
            return str(e)

        apt = crud.update_specific_appointment(db, mobile_number, od, ot, nd, nt)
        if not apt:
            return f"I couldn't update the appointment. Ensure you have an appointment on {od.isoformat()} at {ot.strftime('%H:%M')} and the new slot is available in the future."
            
        return (
            f"Appointment for {mobile_number} updated to "
            f"{apt.date.isoformat()} at {apt.time.strftime('%H:%M')}."
        )
    update_appointment_tool.name = "update_appointment"
    return update_appointment_tool

def view_appointment(db: Session):
    @tool(args_schema=ViewAppointmentInput)
    def view_appointment_tool(mobile_number: str, context: dict = None) -> str:
        """View the appointments for this mobile number."""
        upcoming = crud.get_upcoming_appointments_for_mobile(db, mobile_number)
        if not upcoming:
            return f"You do not have any appointments for {mobile_number}."
        
        msg = f"You have the following upcoming appointments for {mobile_number}:\n"
        for u in upcoming:
            msg += f"- {u.date.isoformat()} at {u.time.strftime('%H:%M')}\n"
        return msg.strip()
    view_appointment_tool.name = "view_appointment"
    return view_appointment_tool

def find_next_available_slot(db: Session):
    @tool(args_schema=NextAvailableSlotInput)
    def find_next_available_slot_tool(start_date: str | None = None, context: dict = None) -> str:
        """Find the next available future appointment slot. Use this before booking when the user asks for the next or earliest possible slot."""
        try:
            search_start = _parse_date(start_date) if start_date else current_date()
        except ValueError as e:
            return str(e)

        now = current_datetime().replace(tzinfo=None)
        for day_offset in range(0, 30):
            target_date = search_start + timedelta(days=day_offset)
            for candidate_time in _candidate_slots_for_day(target_date):
                if datetime.combine(target_date, candidate_time) <= now:
                    continue
                if crud.is_slot_available(db, target_date, candidate_time):
                    return (
                        f"The next available slot is {target_date.isoformat()} "
                        f"at {candidate_time.strftime('%H:%M')}."
                    )

        return "I couldn't find an available slot in the next 30 days."
    find_next_available_slot_tool.name = "find_next_available_slot"
    return find_next_available_slot_tool

def search_clinic_knowledge(db: Session):
    @tool(args_schema=ClinicKnowledgeInput)
    def search_clinic_knowledge_tool(query: str, context: dict = None) -> str:
        """Retrieve relevant clinic knowledge from indexed PDFs and website content before answering informational questions."""
        try:
            return get_rag_response(query, config=_resolve_rag_config(context, db=db))
        except Exception as exc:
            return f"Unable to retrieve clinic knowledge right now: {exc}"
    search_clinic_knowledge_tool.name = "search_clinic_knowledge"
    return search_clinic_knowledge_tool
