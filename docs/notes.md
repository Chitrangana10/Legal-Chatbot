# Notes — why this stack

This is a reconstruction of the reasoning implied by the actual code and
config in the repo — i.e. *why each piece plausibly makes sense here*,
not a record of decisions you documented elsewhere. Treat it as a
justification you can check against your own reasons, adjust, or push
back on.

---

## 1. Why FastAPI for the backend (not Flask/Django/Node)

- **Pydantic validation for free.** `QueryRequest`/`QueryResponse` in
  `schemas.py` get automatic request validation and response typing with
  almost no boilerplate — important for a system whose whole value
  proposition is "don't return malformed or hallucinated legal data."
- **Async-native + `lifespan`.** The `RAGEngine` (embedding model + FAISS
  index + BM25 index, all sizeable in-memory objects) needs to be built
  exactly once per process and reused across requests. FastAPI's
  `lifespan` context manager is a clean, first-class way to do that —
  Flask requires more manual wiring (app factories + globals) to get the
  same guarantee.
- **Auto-generated OpenAPI docs** (`/docs`) come for free, useful while
  the API surface is still small and evolving.
- **Why not Django**: Django's strengths (ORM, admin panel, batteries for
  a relational, multi-user app) aren't being used here — there's no
  database yet (§3). Pulling in Django's weight for two routes would be
  paying for infrastructure the project doesn't currently need.
- **Why not Node/Express**: the retrieval stack (FAISS, sentence
  transformers, BM25) is Python-native. Splitting the ML pipeline into
  Python and the API into Node would mean either a second service and a
  network hop, or fighting Python bindings from Node — Python end-to-end
  avoids that entirely.

## 2. Why Streamlit for the frontend (not React/Next.js)

- **Speed of iteration for a data/ML product.** Streamlit lets a chat UI,
  a form, and an expandable sources panel be built in ~90 lines with no
  build step, no bundler, no separate frontend deploy — appropriate when
  the interesting engineering is in retrieval and prompting, not UI.
- **It's a defensible choice for now, with a real limit.** Streamlit's
  session model (`st.session_state`) is fundamentally single-session,
  server-rendered, and doesn't persist across page reloads or devices.
  That's fine for a demo/internal tool; it's the reason `poa.md` §5
  flags frontend framework choice as something to revisit **if** the
  project moves toward persistent, multi-device chat history — Streamlit
  isn't built for that, and no amount of extra code fixes it, since it's
  a property of the framework's execution model, not a missing feature.
- **Why not React/Next.js today**: it would require a proper auth layer,
  API client, and state management to justify the extra complexity, none
  of which exist yet on the backend either (`security.py` is a stub).
  Building the React frontend before those exist would mean building UI
  for infrastructure that isn't there.

## 3. Why no database yet (flat JSON + FAISS index on disk)

This is the single biggest structural choice in the project, so it's
worth spelling out the trade-off rather than treating it as an oversight.

- **The corpus is read-mostly and small enough to fit in RAM.** 3,173
  statute sections at 384 dimensions is a few megabytes of vectors.
  Loading it once at startup (`RAGEngine.__init__`) and serving every
  query from memory is dramatically faster than round-tripping to a
  database per query, and simpler to reason about — there's no
  connection pool, no query planner, no migration system to maintain for
  data that changes only when someone re-runs the ingestion script.
- **Statutes don't change per-request.** Unlike user-generated content, a
  section of the IPC is the same for every user, every request. That's
  precisely the case where "compute once, serve from memory" beats
  "hit a database every time" — a database would be paying I/O cost for
  no benefit, since nothing about the data is per-user or write-heavy.
- **The cost of this choice**: it doesn't scale horizontally without
  extra work (each replica would reload/duplicate the same index in its
  own memory — fine up to a point, wasteful past it), and it has no
  natural home for anything that *is* inherently per-user or
  write-heavy: accounts, saved chat history, usage logs, audit trails.
  That's exactly the set of features in `poa.md` Phase 5 that are
  blocked on introducing a real database — the flat-file approach is
  correct for the read-only statute corpus and wrong for those features,
  which is why the POA treats "add a database" as a deliberate, separate
  fork rather than default infrastructure to add early.

## 4. Why FAISS specifically (not Pinecone/Weaviate/pgvector)

- **No hosted vector DB dependency, no network call for retrieval.**
  FAISS is a library, not a service — it runs in-process, so semantic
  search has zero network latency and zero external cost. For a corpus
  this size, a managed vector database would add operational overhead
  (accounts, API keys, network calls) without a corresponding benefit —
  FAISS's `IndexFlatIP` does exact (not approximate) nearest-neighbor
  search, which at ~3,000 vectors is fast enough that there's no
  accuracy/speed trade-off to make yet. Approximate indexes (IVF, HNSW)
  only start earning their complexity at a much larger scale.
- **`IndexFlatIP` + L2-normalized vectors = cosine similarity**, which is
  the standard, well-understood similarity metric for sentence-transformer
  embeddings — no custom distance function to justify or debug.

## 5. Why BM25 alongside FAISS (hybrid search, not semantic-only)

- **Legal text is citation- and term-precise.** A question like "What
  does Section 302 say?" or one naming an exact legal term needs exact
  keyword matching — pure semantic search can retrieve a *conceptually*
  similar section while missing the *literally* named one. BM25 (sparse,
  lexical) and FAISS (dense, semantic) fail on different kinds of
  queries, so combining them covers more ground than either alone.
- **Reciprocal Rank Fusion (RRF) over score-blending.** FAISS cosine
  scores and BM25 scores live on incompatible scales (roughly 0–1 vs.
  unbounded), so averaging or weighting the raw scores would require
  constant re-tuning as the corpus changes. RRF sidesteps that entirely
  by fusing on **rank position** instead of raw score — it's parameter-light
  (just `k=60`, a well-established default from the original RRF paper)
  and doesn't need retuning when sections are added or removed.

## 6. Why `sentence-transformers/all-MiniLM-L6-v2` for embeddings

- **Small (384-dim), fast, and runs on CPU** — no GPU requirement for
  either building the index or embedding a query at request time, which
  matters since the current deployment (`python:3.11-slim` in Docker) has
  no GPU. Larger embedding models would improve semantic recall somewhat
  but at real latency and infra cost for a corpus this size.
- **Well-established general-purpose model** — not fine-tuned on legal
  text specifically. This is a legitimate future upgrade path (a
  legal-domain or multilingual-legal embedding model) but MiniLM is a
  reasonable, low-risk default to start from.

## 7. Why Gemini as the LLM provider

- **`google-generativeai` is a thin, low-dependency SDK** and Gemini's
  free/low tier makes it a reasonable default for a project without
  established API spend yet — evidenced by `GEMINI_API_KEY` being the
  primary configured key and the only wired-up provider in
  `llm_service.py`, while `TOGETHER_API_KEY` is present in config as a
  placeholder for a second provider that isn't implemented yet.
- **Temperature 0.1** is a deliberate choice, not a default — a legal
  assistant should minimize creative variance in favor of consistent,
  literal grounding in the retrieved statute text.
- **Failures degrade to a fallback answer, not a crash.** Because Gemini
  quota/rate-limit errors are a real, expected operational condition
  (not an edge case), `is_llm_error_response()` treats them as a normal
  branch in the flow rather than an exception — the user still gets the
  retrieved sections even when generation fails. This is the kind of
  design decision that matters more for a legal tool than a typical
  chatbot: a hard failure with no information is worse than a rougher,
  templated answer.

## 8. Why the ingestion package is still stubbed while the app runs

This looks inconsistent at first (`ingestion/chunker.py` etc. all
`raise NotImplementedError`) but it's a reasonable **build-order**
choice, not a bug:

- The *end-to-end shape* (FastAPI ↔ RAGEngine ↔ FAISS/BM25 ↔ Gemini) was
  validated first, using a shortcut path
  (`scripts/build_combined_index.py`) that treats each raw JSON record as
  already being a well-sized chunk — true for statute sections, which are
  naturally short, numbered, and citation-ready.
- The "proper" ingestion pipeline (clean → chunk with overlap → embed →
  index) is scaffolded with real function signatures and docstrings
  ready to fill in, but deliberately deferred until it's actually needed
  — e.g. once longer documents (case law, commentary, multi-page
  provisions) are added that don't naturally fit the "one JSON record =
  one chunk" assumption the current script relies on.

## 9. Why Docker Compose (not Kubernetes / a PaaS) for deployment

- **Two services, one host, no auto-scaling requirement yet.** Compose
  gives reproducible local/single-host deployment with a fraction of the
  operational surface of Kubernetes — appropriate while the app is
  single-instance and stateless-except-for-the-index.
- **Bind-mounting `backend/data`** (rather than baking it into the image)
  means re-running the index build script updates what the running
  container serves without a rebuild — useful during active data/index
  iteration.