from app.domain.dental import config as dental_config

AGENT_CONFIG = {
    "agent_id": "dental_bot",
    "tools": [
        "book_appointment",
        "cancel_appointment",
        "update_appointment",
        "view_appointment",
        "find_next_available_slot",
        "search_clinic_knowledge"
    ],
    "intent_labels": dental_config.INTENT_LABELS,
    "prompts": dental_config
}
