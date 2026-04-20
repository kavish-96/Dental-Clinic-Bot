from datetime import date, datetime, time

from sqlalchemy.orm import Session

from .models import (
    Appointment,
    AgentPrompt,
    AgentRAGConfig,
    AgentSynonym,
    AgentIntent,
    AgentToolAlias,
)
from .datetime_utils import current_datetime
import json

def _combine_slot(date_val: date, time_val: time) -> datetime:
    return datetime.combine(date_val, time_val)


def is_slot_in_past(date_val: date, time_val: time) -> bool:
    return _combine_slot(date_val, time_val) < current_datetime().replace(tzinfo=None)


def is_slot_available(
    db: Session,
    date_val: date,
    time_val: time,
    exclude_appointment_id: int | None = None,
) -> bool:
    query = db.query(Appointment).filter(
        Appointment.date == date_val,
        Appointment.time == time_val,
    )
    if exclude_appointment_id is not None:
        query = query.filter(Appointment.id != exclude_appointment_id)
    return query.first() is None


def create_appointment(
    db: Session,
    mobile_number: str,
    date_val: date,
    time_val: time,
) -> Appointment | None:
    """Create a new appointment for a mobile number if the slot is valid and free."""
    if is_slot_in_past(date_val, time_val):
        return None
    if not is_slot_available(db, date_val, time_val):
        return None

    apt = Appointment(
        mobile_number=mobile_number,
        date=date_val,
        time=time_val,
    )
    db.add(apt)
    db.commit()
    db.refresh(apt)
    return apt


def get_appointments_for_mobile(db: Session, mobile_number: str) -> list[Appointment]:
    """Return all appointments for a mobile number."""
    return (
        db.query(Appointment)
        .filter(Appointment.mobile_number == mobile_number)
        .order_by(Appointment.date, Appointment.time)
        .all()
    )


def get_upcoming_appointments_for_mobile(db: Session, mobile_number: str) -> list[Appointment]:
    """Return all upcoming appointments for a mobile number."""
    now = current_datetime().replace(tzinfo=None)
    apps = get_appointments_for_mobile(db, mobile_number)
    upcoming_apps = []
    for app in apps:
        if _combine_slot(app.date, app.time) > now:
            upcoming_apps.append(app)
    return upcoming_apps


def cancel_specific_appointment(db: Session, mobile_number: str, date_val: date, time_val: time) -> bool:
    """Cancel (delete) a specific upcoming appointment for this mobile number."""
    apt = db.query(Appointment).filter(
        Appointment.mobile_number == mobile_number,
        Appointment.date == date_val,
        Appointment.time == time_val
    ).first()
    if not apt:
        return False
    db.delete(apt)
    db.commit()
    return True


def update_specific_appointment(
    db: Session,
    mobile_number: str,
    old_date: date,
    old_time: time,
    new_date: date | None = None,
    new_time: time | None = None,
) -> Appointment | None:
    """Update date and/or time on a specific appointment for this mobile number."""
    apt = db.query(Appointment).filter(
        Appointment.mobile_number == mobile_number,
        Appointment.date == old_date,
        Appointment.time == old_time
    ).first()
    if not apt:
        return None

    target_date = new_date if new_date is not None else apt.date
    target_time = new_time if new_time is not None else apt.time

    if is_slot_in_past(target_date, target_time):
        return None
    if not is_slot_available(db, target_date, target_time, exclude_appointment_id=apt.id):
        return None

    apt.date = target_date
    apt.time = target_time
    db.commit()
    db.refresh(apt)
    return apt


def get_active_prompts(db: Session, agent_id: str) -> dict:
    """Fetch active prompts for a specific agent as a dictionary."""
    prompts = db.query(AgentPrompt).filter(
        AgentPrompt.agent_id == agent_id,
        AgentPrompt.is_active == True
    ).all()
    
    return {p.prompt_type: p.content for p in prompts}


def get_rag_config(db: Session, agent_id: str) -> AgentRAGConfig | None:
    """Fetch the latest RAG configuration for an agent."""
    return db.query(AgentRAGConfig).filter(
        AgentRAGConfig.agent_id == agent_id
    ).order_by(AgentRAGConfig.created_at.desc()).first()


def get_synonyms(db: Session, agent_id: str) -> dict:
    """Fetch synonyms grouped by category for an agent."""
    synonyms = db.query(AgentSynonym).filter(AgentSynonym.agent_id == agent_id).all()
    
    result = {}
    for syn in synonyms:
        try:
            words_list = json.loads(syn.words)
        except (json.JSONDecodeError, TypeError):
            # Fallback to comma-separated if not JSON
            words_list = [w.strip() for w in syn.words.split(",") if w.strip()]
            
        if syn.category in result:
            result[syn.category].extend(words_list)
        else:
            result[syn.category] = words_list
            
    return result


def get_intents(db: Session, agent_id: str) -> list[str]:
    """Fetch a list of supported intent labels for an agent."""
    intents = db.query(AgentIntent).filter(AgentIntent.agent_id == agent_id).all()
    return [intent.label for intent in intents]


def get_tool_aliases(db: Session, agent_id: str) -> dict:
    """Fetch tool aliases as a dictionary {tool_name: [aliases]} for an agent."""
    aliases = db.query(AgentToolAlias).filter(AgentToolAlias.agent_id == agent_id).all()
    
    result = {}
    for alias_entry in aliases:
        try:
            alias_list = json.loads(alias_entry.aliases)
        except (json.JSONDecodeError, TypeError):
            # Fallback to comma-separated if not JSON
            alias_list = [a.strip() for a in alias_entry.aliases.split(",") if a.strip()]
            
        result[alias_entry.tool_name] = alias_list
        
    return result
