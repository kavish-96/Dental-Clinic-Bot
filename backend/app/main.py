import logging
from contextlib import asynccontextmanager
from datetime import date, time, datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import settings
from .database import init_db, get_db
from .datetime_utils import parse_date_input, parse_time_input
from .models import Appointment
from .routes.admin import router as admin_router
from .routes.chat import router as chat_router
from . import crud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Dental Appointment Booking API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(admin_router)


@app.get("/appointments")
def list_appointments(db: Session = Depends(get_db)):
    """List all appointments (for debugging or admin)."""
    appointments = db.query(Appointment).order_by(Appointment.date, Appointment.time).all()
    return [a.to_dict() for a in appointments]


@app.get("/appointments/{mobile_number}")
def get_appointments_for_mobile(mobile_number: str, db: Session = Depends(get_db)):
    """Return all appointments for a given mobile number."""
    apps = crud.get_appointments_for_mobile(db, mobile_number)
    return [a.to_dict() for a in apps]


class UpdateAppointmentBody(BaseModel):
    date: str | None = None  # YYYY-MM-DD
    time: str | None = None  # HH:MM or HH:MM AM/PM


def _parse_date(s: str) -> date:
    try:
        return parse_date_input(s)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _parse_time(s: str) -> time:
    try:
        return parse_time_input(s)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/appointments/{mobile_number}")
def update_appointment_for_mobile(
    mobile_number: str,
    body: UpdateAppointmentBody,
    db: Session = Depends(get_db),
):
    """Update the appointment for this mobile number."""
    if not body.date and not body.time:
        raise HTTPException(status_code=400, detail="Provide at least date or time to update.")

    d = _parse_date(body.date) if body.date else None
    t = _parse_time(body.time) if body.time else None
    current_apps = crud.get_upcoming_appointments_for_mobile(db, mobile_number)
    if not current_apps:
        raise HTTPException(status_code=404, detail="No appointment found for this mobile number.")

    current = current_apps[0]
    target_date = d if d is not None else current.date
    target_time = t if t is not None else current.time
    if crud.is_slot_in_past(target_date, target_time):
        raise HTTPException(status_code=400, detail="Appointment must be scheduled for a future date and time.")
    if not crud.is_slot_available(db, target_date, target_time, exclude_appointment_id=current.id):
        raise HTTPException(status_code=409, detail="This appointment slot is already booked.")

    apt = crud.update_specific_appointment(db, mobile_number, current.date, current.time, new_date=d, new_time=t)
    if not apt:
        raise HTTPException(status_code=400, detail="Unable to update the appointment.")
    return apt.to_dict()


@app.delete("/appointments/{mobile_number}")
def cancel_appointment_for_mobile(mobile_number: str, db: Session = Depends(get_db)):
    """Cancel the appointment for this mobile number."""
    current_apps = crud.get_upcoming_appointments_for_mobile(db, mobile_number)
    if not current_apps:
        raise HTTPException(status_code=404, detail="No appointment found for this mobile number.")
    
    ok = crud.cancel_specific_appointment(db, mobile_number, current_apps[0].date, current_apps[0].time)
    if not ok:
        raise HTTPException(status_code=404, detail="No appointment found for this mobile number.")
    return {"detail": "Appointment cancelled."}


@app.get("/health")
def health():
    return {"status": "ok"}
