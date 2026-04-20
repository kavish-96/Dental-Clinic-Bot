import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import Appointment, Base  # noqa: E402


def main() -> None:
    sqlite_url = os.getenv("SQLITE_DATABASE_URL", "sqlite:///./appointments.db")
    postgres_url = os.getenv(
        "POSTGRES_DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/appointments",
    )

    source_engine = create_engine(sqlite_url)
    target_engine = create_engine(postgres_url, pool_pre_ping=True)

    Base.metadata.create_all(bind=target_engine)

    with Session(source_engine) as source_session, Session(target_engine) as target_session:
        appointments = source_session.execute(select(Appointment)).scalars().all()

        imported = 0
        skipped = 0

        for appointment in appointments:
            exists = target_session.execute(
                select(Appointment).where(
                    Appointment.mobile_number == appointment.mobile_number,
                    Appointment.date == appointment.date,
                    Appointment.time == appointment.time,
                )
            ).scalar_one_or_none()

            if exists:
                skipped += 1
                continue

            target_session.add(
                Appointment(
                    mobile_number=appointment.mobile_number,
                    date=appointment.date,
                    time=appointment.time,
                    created_at=appointment.created_at,
                )
            )
            imported += 1

        target_session.commit()

    print(f"Imported {imported} appointments; skipped {skipped} duplicates.")


if __name__ == "__main__":
    main()
