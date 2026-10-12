# Intelligent Legal RAG System

An AI-powered legal retrieval and guidance system for Indian law, built using retrieval-augmented generation (RAG). It expands everyday wording, searches overlapping statute chunks with FAISS and BM25, fuses results using RRF, optionally reranks them with a cross-encoder, and uses Gemini to generate cited answers.

**Live app:** [Legal Assistant on AWS EC2](http://34.238.238.94)

Email/password accounts and saved conversations use PostgreSQL. Redis caches successful answers separately for each user for 24 hours by default; errors and source-refusal answers are not cached. Sample questions remain in the frontend. See [accounts, history, caching, and deployment setup](docs/accounts-and-history.md).

**Current data coverage** (`backend/data/raw/`):
- Indian Penal Code (IPC)
- Code of Criminal Procedure, 1973
- Constitution of India
- Consumer Protection Act, 2019
- Income Tax Act, 1961
- Indian Evidence Act, 1872
- Information Technology Act, 2000
- Motor Vehicles Act, 1988

## Project structure

```
Legal-Chatbot/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app entrypoint
│   │   ├── config.py          # Settings loaded from .env
│   │   ├── api/                # routes.py, schemas.py
│   │   └── services/           # rag_engine, hybrid_search, llm_service, vector_store
│   ├── ingestion/
│   │   └── scrapers/           # indian_kanoon.py — pulls statute text from Indian Kanoon
│   ├── scripts/                # build_combined_index.py, test_rag.py, test_retrieval.py
│   └── data/
│       ├── raw/                 # source statute JSON files
│       ├── processed/           # cleaned/chunked output
│       └── faiss_index/         # built vector index (index.faiss, metadata.json)
├── frontend/
│   └── streamlit_app.py        # Streamlit UI
├── docker/                     # backend.Dockerfile, frontend.Dockerfile, docker-compose.yml
├── docs/                       # architecture.md, flow.md, poa.md
├── tests/
├── requirements.txt
└── .env.example
```

## Prerequisites

- Python 3.11+ (the Docker images use `python:3.11-slim`)
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/app/apikey)) — the default LLM provider
- Docker Desktop with Docker Compose to run PostgreSQL and Redis locally (keep Docker Desktop running even when starting Python directly)

## 1. Clone and enter the project

```bash
git clone https://github.com/Chitrangana10/Legal-Chatbot.git
cd Legal-Chatbot
```

## 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

| Shell | Command |
|---|---|
| Windows Git Bash | `source .venv/Scripts/activate` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

You should see `(.venv)` at the start of your terminal prompt once it's active. Run all following commands with this environment active.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Then edit `.env` and set your key(s):

```
GEMINI_API_KEY=your_actual_key_here
```

Other variables in `.env.example` (`APP_NAME`, `EMBEDDING_MODEL_NAME`, `DEFAULT_LLM_PROVIDER`, data directory paths, etc.) already have sensible defaults — only override them if you need to change models or paths. Confirm `.env` stays out of version control (it's already listed in `.gitignore`).

## 5. Build the vector index (Optional)

The repository already includes a pre-built FAISS index, so this step is only required if you modify the legal dataset or need to regenerate the index.

```bash
python backend/scripts/build_combined_index.py
```

Re-run this any time files in `backend/data/raw/` change.

## 6. Run the app

Open two terminals with `.venv` activated.

First start the supporting services from the project root:

```bash
docker compose up -d postgres redis
```

Direct Python connects to the same PostgreSQL database as Docker using `127.0.0.1:5433`. If you set a custom `POSTGRES_PASSWORD`, use the same password in your local `DATABASE_URL`.

**Terminal 1 — Backend**

```bash
python -m uvicorn backend.app.main:app --reload
```

**Terminal 2 — Frontend**

```bash
python -m streamlit run frontend/streamlit_app.py
```

The application will be available at:

- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

Create an account in the UI and sign in before asking questions. On `/health`, confirm `index_loaded` is `true`.

To run the entire application in Docker instead:

```bash
docker compose up --build -d
```

`docker compose down` stops it and retains PostgreSQL data. `docker compose down -v` deletes its persistent database volume.


## Quick test questions

For a comprehensive list of sample queries across all supported Acts, refer to **[`sample-ques.md`](sample-ques.md)**.

- "What is the punishment for murder?" → should cite IPC Section 302
- "What is the procedure for arrest without a warrant?" → should cite CrPC sections
- "What is the GST rate on restaurant food?" → should refuse, not answer (out of scope of the ingested statutes)

## Running tests

```bash
pytest
```

Backend account/history tests require `TEST_DATABASE_URL` pointing to a disposable PostgreSQL database with a name ending in `_test`; they recreate its tables. CI provisions that database automatically. These tests do not call Gemini. The sample retrieval check is `python backend/scripts/validate_sample_questions.py` (no Gemini calls unless `--generate` is explicitly supplied).

## Troubleshooting

- **Gemini quota exceeded:** The application falls back to returning the retrieved legal sources.
- **ModuleNotFoundError:** Ensure the virtual environment is activated before running the application.
- **Updated the legal dataset?** Rebuild the FAISS index using:

```bash
 python backend/scripts/build_combined_index.py
```

