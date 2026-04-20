from datetime import date, datetime

def get_system_prompt(config, current_date: date, current_datetime: datetime) -> str:
    return config.SYSTEM_PROMPT.format(
        current_date=current_date.isoformat(),
        current_datetime=current_datetime.strftime("%Y-%m-%d %H:%M")
    )

def get_intent_prompt(config) -> str:
    return config.INTENT_PROMPT

def get_response_prompt(config) -> str:
    return config.RESPONSE_PROMPT

def get_context_resolution_prompt(config) -> str:
    return config.CONTEXT_RESOLUTION_PROMPT
