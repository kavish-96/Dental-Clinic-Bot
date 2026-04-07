# Dental Appointment Booking Application

Chat-based dental clinic appointment booking using LLM and LangChain. Book, cancel, update, and view appointments via natural language, answer current weather questions for any city, and retrieve clinic knowledge from PDFs and crawled website content via FAISS-backed RAG.

## Architecture

- **Backend**: FastAPI + LangChain (Groq via `langchain-groq`) with tool-calling agent; PostgreSQL for persistence; FAISS + HuggingFace embeddings for RAG.
- **Frontend**: Next.js (App Router) with a chat UI.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Groq API key (from Groq console)
- OpenWeatherMap API key (for current weather)
- PDF parsing dependencies for Unstructured

## Backend setup

1. Create a virtual environment and install dependencies:

   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set your API keys:

   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # macOS/Linux
   ```

   Edit `.env`:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
   DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/appointments
   RAG_PDF_DIR=./data/rag/pdfs
   RAG_CRAWL_OUTPUT_PATH=./data/rag/crawl_output.json
   RAG_FAISS_INDEX_DIR=./faiss_index
   RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

3. Run the API:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be at `http://localhost:8000`. Tables are created automatically on first run.

4. Optional: build the RAG index immediately:

   ```bash
   cd backend
   python -m app.rag --rebuild
   ```

   By default, PDFs are read from `backend/data/rag/pdfs`, crawl output is read from `backend/data/rag/crawl_output.json`, and the FAISS index is stored in `backend/faiss_index`.

## Frontend setup

1. Install dependencies and run the dev server:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. Optional: set `NEXT_PUBLIC_API_URL` in `.env.local` if the backend is not at `http://localhost:8000`:

   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. Open [http://localhost:3000](http://localhost:3000) and use the chat to book, cancel, update, or view appointments, ask for the current weather, or ask knowledge questions about clinic documents and website content.

## Docker setup

1. Make sure Docker Desktop is running.

2. From the project root, start PostgreSQL and the backend API:

   ```bash
   docker compose up --build
   ```

3. Open `http://localhost:8080/docs` in the browser to access the FastAPI service.

4. Stop the stack:

   ```bash
   docker compose down
   ```

## Migrating existing SQLite data to PostgreSQL

If you already have data in `backend/appointments.db`, migrate it after PostgreSQL is running:

```bash
cd backend
set POSTGRES_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/appointments
python scripts/migrate_sqlite_to_postgres.py
```

If you also want to override the SQLite file path:

```bash
set SQLITE_DATABASE_URL=sqlite:///./appointments.db
python scripts/migrate_sqlite_to_postgres.py
```

## API

- **POST /chat** — Send a message and get the assistant reply.
  - Body: `{ "message": "Book an appointment for John on 2025-03-20 at 10:00" }`
  - Response: `{ "response": "..." }`

- **GET /appointments** — List all appointments (debug/admin).

- **GET /health** — Health check.

## Usage examples

- "Book an appointment for Jane Doe on 2025-03-25 at 2:30 PM for a cleaning"
- "Show my appointments"
- "Cancel appointment ID 3"
- "Update appointment 1 to March 26 at 10am"
- "What is the current weather in Mumbai?"
- "What services does the clinic offer?"
- "What does the clinic website say about cancellation policies?"
