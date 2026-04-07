import json
from datetime import date, datetime, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import crud
from .config import settings
from .datetime_utils import current_date, current_datetime, parse_date_input, parse_time_input
from .rag import query_knowledge_base


# --- Pydantic schemas for tool args ---


class BookAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number (primary identifier)")
    date: str = Field(description="Appointment date. Accepts YYYY-MM-DD or natural dates like 5 April.")
    time: str = Field(description="Appointment time in HH:MM or HH:MM AM/PM format")


class CancelAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number")


class UpdateAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number")
    new_date: str | None = Field(default=None, description="New date, e.g. YYYY-MM-DD or 5 April")
    new_time: str | None = Field(default=None, description="New time HH:MM or HH:MM AM/PM")


class ViewAppointmentInput(BaseModel):
    mobile_number: str = Field(description="User's mobile number")


class NextAvailableSlotInput(BaseModel):
    start_date: str | None = Field(
        default=None,
        description="Optional date to start searching from. Accepts YYYY-MM-DD or natural dates like 5 April.",
    )


class CurrentWeatherInput(BaseModel):
    city: str = Field(description="City name to fetch current weather for")


class ClinicKnowledgeInput(BaseModel):
    query: str = Field(description="Question about clinic information from PDFs or website content")


def _parse_time(s: str) -> time:
    return parse_time_input(s)


def _parse_date(s: str) -> date:
    return parse_date_input(s)


def _candidate_slots_for_day(target_date: date) -> list[time]:
    return [time(hour=hour) for hour in range(9, 18)]


def get_tools(db: Session) -> list:
    """Build LangChain tools that operate on appointments identified by mobile number."""

    @tool(args_schema=BookAppointmentInput)
    def book_appointment(
        mobile_number: str,
        date: str,
        time: str,
    ) -> str:
        """Book a new dental appointment for a mobile number."""
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

    @tool(args_schema=CancelAppointmentInput)
    def cancel_appointment(mobile_number: str) -> str:
        """Cancel the appointment for this mobile number."""
        ok = crud.cancel_upcoming_for_mobile(db, mobile_number)
        if not ok:
            return f"No appointment found for {mobile_number}."
        return f"Your appointment for {mobile_number} has been cancelled."

    @tool(args_schema=UpdateAppointmentInput)
    def update_appointment(
        mobile_number: str,
        new_date: str | None = None,
        new_time: str | None = None,
    ) -> str:
        """Update the appointment's date and/or time for this mobile number."""
        try:
            d = _parse_date(new_date) if new_date else None
            t = _parse_time(new_time) if new_time else None
        except ValueError as e:
            return str(e)
        if d is None and t is None:
            return "Provide at least a new_date or new_time to update the appointment."
        current_appointment = crud.get_upcoming_appointment_for_mobile(db, mobile_number)
        if not current_appointment:
            return f"No appointment found for {mobile_number}."

        target_date = d if d is not None else current_appointment.date
        target_time = t if t is not None else current_appointment.time
        if crud.is_slot_in_past(target_date, target_time):
            return "Appointments can only be updated to a future date and time."
        if not crud.is_slot_available(
            db,
            target_date,
            target_time,
            exclude_appointment_id=current_appointment.id,
        ):
            return (
                f"The slot on {target_date.isoformat()} at {target_time.strftime('%H:%M')} "
                "is already booked. Please choose another time."
            )

        apt = crud.update_upcoming_for_mobile(db, mobile_number, date_val=d, time_val=t)
        if not apt:
            return "I couldn't update that appointment. Please try a different future time slot."
        return (
            f"Appointment for {mobile_number} updated to "
            f"{apt.date.isoformat()} at {apt.time.strftime('%H:%M')}."
        )

    @tool(args_schema=ViewAppointmentInput)
    def view_appointment(mobile_number: str) -> str:
        """View the appointment for this mobile number."""
        apt = crud.get_upcoming_appointment_for_mobile(db, mobile_number)
        if not apt:
            return f"You do not have any appointments for {mobile_number}."
        return (
            f"You have an appointment on {apt.date.isoformat()} at "
            f"{apt.time.strftime('%H:%M')} for mobile {mobile_number}."
        )

    @tool(args_schema=NextAvailableSlotInput)
    def find_next_available_slot(start_date: str | None = None) -> str:
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

    @tool(args_schema=CurrentWeatherInput)
    def get_current_weather(city: str) -> str:
        """Get the current weather for a city using OpenWeatherMap."""
        city_name = city.strip()
        if not city_name:
            return "Please provide a city name."

        if not settings.OPENWEATHERMAP_API_KEY:
            return "Weather service is not configured. Set OPENWEATHERMAP_API_KEY."

        encoded_city = quote_plus(city_name)
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={encoded_city}&appid={settings.OPENWEATHERMAP_API_KEY}&units=metric"
        )

        try:
            with urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                error_payload = {}
            message = error_payload.get("message")
            if exc.code == 404:
                return f"I couldn't find current weather for '{city_name}'. Please check the city name and try again."
            return f"Unable to fetch weather right now: {message or f'HTTP {exc.code}'}."
        except URLError:
            return "Unable to reach the weather service right now. Please try again in a moment."
        except Exception:
            return "Something went wrong while fetching the weather."

        weather_items = payload.get("weather") or []
        main = payload.get("main") or {}
        wind = payload.get("wind") or {}
        sys = payload.get("sys") or {}

        description = weather_items[0].get("description", "unavailable") if weather_items else "unavailable"
        temp = main.get("temp")
        feels_like = main.get("feels_like")
        humidity = main.get("humidity")
        wind_speed = wind.get("speed")
        country = sys.get("country")
        resolved_name = payload.get("name") or city_name
        location = f"{resolved_name}, {country}" if country else resolved_name

        details = [f"Current weather in {location}: {description}."]
        if temp is not None:
            details.append(f"Temperature: {round(temp)} degrees C.")
        if feels_like is not None:
            details.append(f"Feels like: {round(feels_like)} degrees C.")
        if humidity is not None:
            details.append(f"Humidity: {humidity}%.")
        if wind_speed is not None:
            details.append(f"Wind speed: {wind_speed} m/s.")
        return " ".join(details)

    @tool(args_schema=ClinicKnowledgeInput)
    def search_clinic_knowledge(query: str) -> str:
        """Retrieve relevant clinic knowledge from indexed PDFs and website content before answering informational questions."""
        try:
            return query_knowledge_base(query)
        except Exception as exc:
            return f"Unable to retrieve clinic knowledge right now: {exc}"

    return [
        book_appointment,
        cancel_appointment,
        update_appointment,
        view_appointment,
        find_next_available_slot,
        get_current_weather,
        search_clinic_knowledge,
    ]
