import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    return (value or "") if value is not None else ""


class Settings:
    # Groq API key (used by ChatGroq)
    GROQ_API_KEY: str = get_env("GROQ_API_KEY", "")
    DATABASE_URL: str = get_env(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/appointments",
    )
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    APP_TIMEZONE: str = get_env("APP_TIMEZONE", "Asia/Kolkata")
    
    RAG_PDF_DIR: str = get_env("RAG_PDF_DIR", "./data/rag/pdfs")
    RAG_CRAWL_OUTPUT_PATH: str = get_env("RAG_CRAWL_OUTPUT_PATH", "./data/rag/webcrawling")
    RAG_FAISS_INDEX_DIR: str = get_env("RAG_FAISS_INDEX_DIR", "./data/rag/faiss_index")
    RAG_EMBEDDING_MODEL: str = get_env(
        "RAG_EMBEDDING_MODEL",
        "BAAI/bge-base-en",
    )


settings = Settings()
