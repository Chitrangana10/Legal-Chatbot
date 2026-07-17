# Architecture — Intelligent Legal RAG System

## 1. One-line summary

A Streamlit chat frontend talks to a FastAPI backend over REST. The backend
holds a single `RAGEngine` in memory, built at startup from flat JSON statute
files and a pre-built FAISS index. Each question is answered by fusing FAISS
(semantic) and BM25 (keyword) retrieval, then asking Gemini to answer strictly
from the retrieved sections, with a deterministic fallback if Gemini is
unavailable.

---

## 2. Layers

### 2.1 Frontend — Streamlit (`frontend/streamlit_app.py`)

- Single Python file, no build step, no JS framework.
- Renders a chat-style UI (`st.chat_message`) with a text input form.
- On submit, POSTs `{"question": ...}` to `http://backend:8000/query`
  (service name `backend` resolves inside Docker Compose; `localhost:8000`
  when run outside Docker).
- Keeps conversation history in `st.session_state.history` — **in-browser-session
  memory only**, not persisted anywhere. Refreshing the page clears it.
- Renders the answer text plus an expandable "Sources" section listing the
  act, section number, and section title for every retrieved source.
- No authentication, no per-user accounts — it is a single shared chat
  surface today.

### 2.2 Backend API — FastAPI (`backend/app/`)

- `main.py` builds the app via `create_app()` and defines a `lifespan`
  context manager that constructs `RAGEngine()` exactly once, at process
  startup, and stores it on `app.state.rag_engine`. If construction throws
  (e.g. missing index files), `app.state.index_loaded` is set to `False`
  and the engine is `None` — the app still starts, but `/query` will
  return a 500 until fixed.
- CORS is currently locked to `http://localhost:8501` (the local Streamlit
  origin) — this will need widening before any non-local frontend is
  deployed.
- Two routes (`backend/app/api/routes.py`):
  - `GET /health` → `{status, index_loaded}` — a cheap liveness/readiness
    probe that also tells you whether the RAG engine actually loaded.
  - `POST /query` → body `{question: str}` (Pydantic-validated, min length
    1) → `{answer: str, sources: [{act, section_number, section_title}]}`.
    Any exception inside `rag_engine.answer()` is caught and turned into a
    `500 {"error": ...}` JSON response rather than an unhandled crash.
- `backend/app/core/security.py` and `backend/app/core/logging.py` exist as
  named stubs (`raise NotImplementedError`) — **there is currently no auth
  and no structured logging**; both are placeholders for later.

### 2.3 RAG engine (`backend/app/services/rag_engine.py`)

This is the orchestrator. `RAGEngine.__init__`:

1. Loads the sentence-transformer embedding model
   (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, configurable via
   `EMBEDDING_MODEL_NAME`).
2. Loads the FAISS index + metadata from `backend/data/faiss_index/`.
3. Loads every `*.json` file under `backend/data/raw/` that matches the
   expected section shape (`act`, `section_number`, `section_title`,
   `text`) via `raw_corpus.load_raw_sections` — files that don't match are
   skipped, not errored.
4. Builds a `HybridSearcher` over the combined section list.

`RAGEngine.answer(question)`:

1. `hybrid_searcher.search(question, top_k=5)` — see 2.4.
2. `build_grounded_prompt(question, retrieved)` — builds a strict,
   source-restricted prompt (see 2.5).
3. `llm_service.generate(prompt)` — calls Gemini.
4. If the response looks like an error (`is_llm_error_response` checks for
   empty text, `"gemini error:"`, `"quota"`, `"rate limit"`,
   `"retry_delay"`, `"exceeded"`), it's replaced with
   `build_fallback_answer()` — a deterministic sentence listing the
   retrieved section numbers/titles, so the user still gets something
   useful even when the LLM call fails.
5. Returns `{answer, sources}`.

### 2.4 Hybrid search (`backend/app/services/hybrid_search.py`)

- **FAISS side**: `VectorStore` (see 2.6) does a cosine-similarity search
  (vectors are L2-normalized, index is `IndexFlatIP` — inner product on
  normalized vectors *is* cosine similarity) and returns the top 10.
- **BM25 side**: `rank_bm25.BM25Okapi` over the same section texts,
  tokenized with a simple `\w+` regex, also returns the top 10.
- **Fusion**: Reciprocal Rank Fusion (RRF) with `k=60` —
  `score = Σ 1 / (60 + rank)` across whichever list(s) each section appears
  in. This is a standard, tuning-free way to combine two differently-scaled
  ranking signals (cosine similarity and BM25 scores aren't on the same
  scale, so RRF sidesteps that by using rank position instead of raw score).
- Final result: top `top_k` (5, from the engine) sections by combined RRF
  score, each carrying its FAISS rank/score and BM25 rank/score for
  debugging.
- `backend/app/services/reranker.py` exists as a stub
  (`raise NotImplementedError`) — a cross-encoder reranking step is
  planned but **not wired into the live path today**.

### 2.5 Prompt construction

`build_grounded_prompt()` concatenates the retrieved sections (act, section
number, title, full text) and wraps them in an instruction that:

- restricts the model to only the facts in the provided sections,
- explicitly allows comparing/synthesizing *across* the provided sections,
- requires a citation (Act + Section number) for every claim,
- tells the model to say "I cannot find this in the available sources."
  only when the sections genuinely don't cover the question,
- forbids inventing citations or section numbers.

This is the mechanism that keeps answers grounded and reduces hallucinated
statute citations — a hard requirement for a legal-domain assistant.

### 2.6 LLM layer (`backend/app/services/llm_service.py`)

- `GeminiProvider` wraps `google-generativeai`. Reads `GEMINI_API_KEY`
  (or falls back to `GOOGLE_API_KEY`) and `DEFAULT_LLM_MODEL` from
  settings.
- Calls `genai.GenerativeModel(model_name).generate_content(prompt,
  generation_config={"temperature": 0.1})` — a low, near-deterministic
  temperature, appropriate for a task where consistency and citation
  accuracy matter more than creative phrasing.
- Every failure mode (missing key, empty response, any exception) is
  returned as a **string** starting with `"Gemini error: ..."` rather than
  raised — this is what lets `rag_engine.answer()` detect failure and
  fall back without a stack trace reaching the user.
- `generate_answer(prompt, provider=...)` exists as a small
  multi-provider seam — passing any `provider` other than `"gemini"`
  currently returns an explicit `"LLM error: unsupported provider"`
  rather than silently ignoring it. Gemini is the only wired-up provider
  today, though `TOGETHER_API_KEY` is already present in config, hinting
  at a planned second provider.

### 2.7 Data layer — flat files, not a database

- **Raw statutes**: `backend/data/raw/*.json` — one file per Act (IPC,
  CrPC 1973, Constitution of India, Consumer Protection Act 2019, Income
  Tax Act 1961, Indian Evidence Act 1872, IT Act 2000, Motor Vehicles Act
  1988). Each file is a flat JSON array of section objects.
- **Vector index**: `backend/data/faiss_index/index.faiss` +
  `metadata.json` — a pre-built, committed-to-repo FAISS index (3,173
  sections, 384-dim vectors) plus a JSON sidecar mapping FAISS row → the
  original section metadata. `VectorStore.save/load` handles the
  serialization.
- **No SQL/NoSQL database, no user table, no session store.** Everything
  the API needs is either loaded from disk into memory at startup
  (index + statutes) or lives only in the Streamlit browser session
  (chat history). This is a meaningful and deliberate difference from a
  typical multi-user SaaS backend — see `notes.md` for the reasoning.
- Index is rebuilt offline via `backend/scripts/build_combined_index.py`
  (loads all raw sections → embeds with the same sentence-transformer
  model → builds BM25 — computed but not persisted, it's rebuilt in
  memory on every process start — → saves FAISS index + metadata to
  disk). This script, **not** the `backend/ingestion/` package, is what
  actually runs today (see 2.8).

### 2.8 Ingestion package — scaffolded, not implemented

`backend/ingestion/{cleaner,chunker,embedder,build_index}.py` are all
present as typed function signatures that `raise NotImplementedError`.
The intent is a clean pipeline: scrape/clean → chunk → embed → build
index. In the current build, `backend/scripts/build_combined_index.py`
shortcuts all of that — it treats each raw JSON section as already a
correctly-sized "chunk" (no further splitting) and embeds/indexes it
directly. `backend/ingestion/scrapers/indian_kanoon.py` (BeautifulSoup +
requests) is implemented and can pull statute text from Indian Kanoon
pages, but isn't invoked by the current build path.

### 2.9 Deployment — Docker Compose, two containers

- `docker/backend.Dockerfile`: `python:3.11-slim`, installs
  `requirements-backend.txt`, runs `uvicorn backend.app.main:app` on
  `:8000`.
- `docker/frontend.Dockerfile`: `python:3.11-slim`, installs
  `requirements-frontend.txt`, runs `streamlit run
  frontend/streamlit_app.py` on `:8501`.
- `docker-compose.yml`: two services, `backend` and `frontend` (depends on
  `backend`); the FAISS data directory is bind-mounted (`./backend/data`
  → `/app/backend/data`) so a rebuilt index doesn't require rebuilding the
  image; env vars (Gemini key, etc.) come from `.env`.
- No reverse proxy, TLS termination, or orchestration (Kubernetes, etc.)
  configured — this is a local/single-host deployment shape today.

---

## 3. Component diagram

See the architecture diagram rendered above in this conversation for the
layered view (Frontend → Backend API → RAG engine → Data layer, plus the
request-flow summary).

## 4. Key design characteristics

| Property | Current state |
|---|---|
| Multi-user | No — single shared engine instance, no auth, no per-user data |
| Statefulness | Stateless backend; state lives in the Streamlit session only |
| Persistence | Flat files (JSON + FAISS index) — no database |
| Scaling model | Vertical only (one process holds the whole index in RAM) |
| LLM provider | Gemini only, hard dependency for best-quality answers, with a non-LLM fallback |
| Retrieval | Hybrid: dense (FAISS) + sparse (BM25), fused with RRF |
| Reranking | Not implemented (stub) |
| Observability | Print statements only; no structured logging, no metrics |
| Security | No API auth, no rate limiting, no input sanitization beyond Pydantic types |