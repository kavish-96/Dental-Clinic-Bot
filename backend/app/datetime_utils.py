import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .config import settings


_ORDINAL_SUFFIX_PATTERN = re.compile(r"(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)
_RELATIVE_DATE_OFFSETS = {
    "today": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
}
APP_ZONE = ZoneInfo(settings.APP_TIMEZONE)


def current_datetime() -> datetime:
    return datetime.now(APP_ZONE)


def current_date() -> date:
    return current_datetime().date()


def _normalize_date_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = _ORDINAL_SUFFIX_PATTERN.sub(r"\1", cleaned)
    cleaned = cleaned.replace(",", " ")
    return " ".join(cleaned.split())


def parse_date_input(value: str, *, default_year: int | None = None) -> date:
    normalized = _normalize_date_text(value)
    default_year = default_year or current_date().year

    relative_offset = _RELATIVE_DATE_OFFSETS.get(normalized.lower())
    if relative_offset is not None:
        return current_date() + timedelta(days=relative_offset)

    formats_with_year = (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    )
    for fmt in formats_with_year:
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue

    formats_without_year = (
        "%d %B",
        "%d %b",
        "%B %d",
        "%b %d",
    )
    for fmt in formats_without_year:
        try:
            parsed = datetime.strptime(normalized, fmt)
            return date(default_year, parsed.month, parsed.day)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format: {value}. Use YYYY-MM-DD, DD/MM/YYYY, or dates like 5 April."
    )


def parse_time_input(value: str) -> time:
    normalized = value.strip().upper()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%H:%M:%S"):
        try:
            return datetime.strptime(normalized, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Invalid time format: {value}. Use HH:MM or HH:MM AM/PM.")
