# Flow — Legal RAG System

Two flows matter here: the **offline indexing flow** (runs once, or
whenever the statute data changes) and the **online query flow** (runs on
every user question). Everything downstream of a question depends on the
indexing flow having already completed.

---

## 1. Offline flow — building the index

Trigger: `python backend/scripts/build_combined_index.py`, run manually
whenever a file under `backend/data/raw/` is added or changed. Not run
automatically — there's no file-watcher or CI hook for this yet.

```
backend/data/raw/*.json (8 Act files, section-shaped records)
        │
        ▼
load_raw_sections()            → validates each JSON file has {act,
                                  section_number, section_title, text};
                                  non-matching files are skipped, not
                                  crashed on
        │
        ▼
build_text_for_embedding()     → "<Act> Section <n>: <title>\n<text>"
        │                         (same text function used later at
        │                         query time, so index and query
        │                         embeddings are in the same space)
        ▼
SentenceTransformer.encode()   → 384-dim vector per section
        │
        ▼
VectorStore(dimension=384)     → L2-normalizes vectors, adds them to a
        │                         FAISS IndexFlatIP
        ▼
store.save(faiss_index_dir)    → writes index.faiss + metadata.json
                                  (dimension + position→section map)

(build_bm25_index() is also called in this script, but its result is
only used to confirm the corpus builds cleanly — the actual BM25 index
used at query time is rebuilt in memory every time the API process
starts, from the same raw sections.)
```

Output: `backend/data/faiss_index/index.faiss` + `metadata.json`,
committed to the repo so a fresh clone can serve queries without
re-running this step.

---

## 2. Online flow — answering a question

```
1. User types a question in Streamlit and submits the form
        │
        ▼
2. Streamlit POSTs {"question": "..."} to POST /query
        │  (requests library, 90s timeout)
        ▼
3. FastAPI validates the body (QueryRequest, min_length=1)
        │
        ▼
4. routes.query() pulls the already-loaded RAGEngine off app.state
        │  → if it's None (failed to load at startup), returns 500 immediately
        ▼
5. RAGEngine.answer(question)
        │
        ├─► 5a. HybridSearcher.search(question, top_k=5)
        │        │
        │        ├─ encode question with the same MiniLM model
        │        ├─ FAISS search → top 10 by cosine similarity
        │        ├─ BM25 search  → top 10 by keyword score
        │        └─ Reciprocal Rank Fusion (k=60) → top 5 combined
        │
        ├─► 5b. build_grounded_prompt(question, top-5 sections)
        │        → strict, citation-required, source-restricted prompt
        │
        ├─► 5c. llm_service.generate(prompt)  → Gemini API call
        │        │
        │        ├─ success → cited natural-language answer
        │        └─ failure (no key / quota / rate limit / empty
        │           response) → string starting "Gemini error: ..."
        │
        ├─► 5d. is_llm_error_response(answer)?
        │        │
        │        ├─ yes → build_fallback_answer(): a deterministic
        │        │        sentence naming the retrieved sections, so the
        │        │        user isn't left with nothing
        │        └─ no  → keep Gemini's answer as-is
        │
        └─► 5e. package {answer, sources: [{act, section_number,
                 section_title}, ...]}
        ▼
6. FastAPI serializes this as QueryResponse and returns 200 JSON
   (or a 500 {"error": ...} if any exception happened in step 5)
        │
        ▼
7. Streamlit renders the answer text, appends {question, answer, sources}
   to st.session_state.history, and shows sources in an expander
```

### Failure paths worth knowing about

- **RAG engine failed to load at startup** (e.g. missing/corrupt FAISS
  index files) → `/query` always returns 500, `/health` reports
  `index_loaded: false`. This is the first thing to check if the API is
  up but every question fails.
- **Gemini quota/rate-limit/misconfigured key** → the user still gets a
  200 response with a usable (if less fluent) answer, thanks to
  `build_fallback_answer()`. The frontend has no way to tell the user
  "this answer came from the fallback path" — it looks identical to a
  normal answer today.
- **Question with no relevant sections in any of the 8 Acts** (e.g. "What
  is the GST rate on restaurant food?") → the prompt instructs Gemini to
  say "I cannot find this in the available sources." rather than guess.
  This is a deliberate scope-refusal behavior, not a bug.
- **Backend unreachable from Streamlit** → caught as
  `requests.RequestException` and shown as a generic
  "Something went wrong — is the backend running?" error in the chat UI.

---

## 3. Request/response contract

```
POST /query
{ "question": "What is the punishment for murder?" }

200 OK
{
  "answer": "...cited answer text...",
  "sources": [
    {"act": "The Indian Penal Code, 1860", "section_number": "302", "section_title": "Punishment for murder."},
    ...
  ]
}

500 (engine not loaded, or exception during answer())
{ "error": "..." }
```