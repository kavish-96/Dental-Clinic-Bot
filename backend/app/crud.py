from datetime import date, datetime, time

from sqlalchemy.orm import Session

from .models import Appointment
from .datetime_utils import current_datetime


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


def get_upcoming_appointment_for_mobile(
    db: Session, mobile_number: str
) -> Appointment | None:
    """Return the appointment for a mobile number, if any.

    For simplicity we assume there is at most one active appointment per
    mobile number and just return the earliest one if multiple exist.
    """
    return (
        db.query(Appointment)
        .filter(Appointment.mobile_number == mobile_number)
        .order_by(Appointment.date, Appointment.time)
        .first()
    )


def cancel_upcoming_for_mobile(db: Session, mobile_number: str) -> bool:
    """Cancel (delete) the upcoming appointment for this mobile number."""
    apt = get_upcoming_appointment_for_mobile(db, mobile_number)
    if not apt:
        return False
    db.delete(apt)
    db.commit()
    return True


def update_upcoming_for_mobile(
    db: Session,
    mobile_number: str,
    date_val: date | None = None,
    time_val: time | None = None,
) -> Appointment | None:
    """Update date and/or time on the upcoming appointment for this mobile number."""
    apt = get_upcoming_appointment_for_mobile(db, mobile_number)
    if not apt:
        return None

    new_date = date_val if date_val is not None else apt.date
    new_time = time_val if time_val is not None else apt.time

    if is_slot_in_past(new_date, new_time):
        return None
    if not is_slot_available(db, new_date, new_time, exclude_appointment_id=apt.id):
        return None

    apt.date = new_date
    apt.time = new_time
    db.commit()
    db.refresh(apt)
    return apt
