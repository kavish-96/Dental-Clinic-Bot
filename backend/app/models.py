from datetime import date, time, datetime
from sqlalchemy import Column, Integer, String, Date, Time, DateTime, Text, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Mobile number is the primary identifier for appointments
    mobile_number = Column(String(50), nullable=False, index=True)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mobile_number": self.mobile_number,
            "date": self.date.isoformat() if self.date else "",
            "time": self.time.strftime("%H:%M") if self.time else "",
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }
