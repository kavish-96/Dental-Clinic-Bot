from datetime import date, time, datetime
from sqlalchemy import Column, Integer, String, Date, Time, DateTime, Text, Float, Boolean
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

class AgentPrompt(Base):
    __tablename__ = "agent_prompts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(50), index=True, nullable=False)
    prompt_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AgentRAGConfig(Base):
    __tablename__ = "agent_rag_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(50), index=True, nullable=False)
    rewrite_prompt = Column(Text)
    answer_instructions = Column(Text)
    multi_query_prompt = Column(Text)
    focus_keywords = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class AgentSynonym(Base):
    __tablename__ = "agent_synonyms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(50), index=True, nullable=False)
    category = Column(String(100), nullable=False)
    words = Column(Text, nullable=False)

class AgentIntent(Base):
    __tablename__ = "agent_intents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(50), index=True, nullable=False)
    label = Column(String(100), nullable=False)

class AgentToolAlias(Base):
    __tablename__ = "agent_tool_aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(50), index=True, nullable=False)
    tool_name = Column(String(100), nullable=False)
    aliases = Column(Text, nullable=False)
