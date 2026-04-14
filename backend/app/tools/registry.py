from sqlalchemy.orm import Session
from app.tool_functions import (
    book_appointment,
    cancel_appointment,
    update_appointment,
    view_appointment,
    find_next_available_slot,
    search_clinic_knowledge
)

TOOL_REGISTRY = {
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "update_appointment": update_appointment,
    "view_appointment": view_appointment,
    "find_next_available_slot": find_next_available_slot,
    "search_clinic_knowledge": search_clinic_knowledge
}

def load_tools(tool_names: list, db: Session):
    return [TOOL_REGISTRY[name](db) for name in tool_names if name in TOOL_REGISTRY]
