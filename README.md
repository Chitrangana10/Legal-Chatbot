# Intelligent Legal RAG System

An AI-powered legal retrieval and guidance system for Indian law, built using retrieval-augmented generation (RAG). It combines FAISS vector search with BM25 keyword search over statute text, then uses an LLM (Gemini by default) to generate cited answers.

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
Intelligent-Legal-RAG-System/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app entrypoint
│   │   ├── config.py          # Settings loaded from .env
│   │   ├── api/                # routes.py, schemas.py
│   │   ├── core/               # logging, security
│   │   └── services/           # rag_engine, hybrid_search, llm_service, vector_store, reranker
│   ├── ingestion/              # chunker, cleaner, embedder, build_index, scrapers
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
- Optional: Docker + Docker Compose, if you'd rather run it containerized

## 1. Clone and enter the project

```bash
git clone <repo-url>
```

## 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

| Shell | Command |
|---|---|
| Windows Git Bash | `source .venv/Scripts/activate` |
| Windows CMD / PowerShell | `.venv\Scripts\activate` |
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


## Quick test questions

For a comprehensive list of sample queries across all supported Acts, refer to **[`sample-ques.md`](sample-ques.md)**.

- "What is the punishment for murder?" → should cite IPC Section 302
- "What is the procedure for arrest without a warrant?" → should cite CrPC sections
- "What is the GST rate on restaurant food?" → should refuse, not answer (out of scope of the ingested statutes)

## Running tests

```bash
pytest
```

## Troubleshooting

- **Gemini quota exceeded:** The application falls back to returning the retrieved legal sources.
- **ModuleNotFoundError:** Ensure the virtual environment is activated before running the application.
- **Updated the legal dataset?** Rebuild the FAISS index using:

```bash
 python backend/scripts/build_combined_index.py

