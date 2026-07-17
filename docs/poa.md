# Plan of Action — Intelligent Legal RAG System

Status legend: ✅ done and wired in · 🟡 exists but not wired in / stub ·
⬜ not started.

## 1. Current state audit

| Area | Status | Detail |
|---|---|---|
| FastAPI app + lifespan startup load | ✅ | `main.py` |
| `/health`, `/query` endpoints | ✅ | `api/routes.py` |
| Hybrid retrieval (FAISS + BM25 + RRF) | ✅ | `services/hybrid_search.py` |
| Grounded, citation-required prompting | ✅ | `rag_engine.build_grounded_prompt` |
| Gemini integration | ✅ | `services/llm_service.py` |
| Fallback answer when LLM fails | ✅ | `rag_engine.build_fallback_answer` |
| Streamlit chat UI | ✅ | `frontend/streamlit_app.py` |
| Docker Compose (backend + frontend) | ✅ | `docker-compose.yml` |
| Pre-built FAISS index (8 Acts, 3,173 sections) | ✅ | `backend/data/faiss_index/` |
| Offline index build script | ✅ | `backend/scripts/build_combined_index.py` |
| Indian Kanoon scraper | ✅ (implemented, not invoked by build path) | `ingestion/scrapers/indian_kanoon.py` |
| Reranker (cross-encoder) | 🟡 | `services/reranker.py` — `raise NotImplementedError` |
| Structured logging | 🟡 | `core/logging.py` — stub |
| API auth / key validation | 🟡 | `core/security.py` — stub |
| Citation formatting/validation utils | 🟡 | `utils/citations.py` — stub |
| Clean ingestion pipeline (clean → chunk → embed → index) | 🟡 | `ingestion/{cleaner,chunker,embedder,build_index}.py` — all stubs; `scripts/build_combined_index.py` bypasses this today |
| Production test suite | 🟡 | `tests/test_skeleton.py` intentionally `raise NotImplementedError`; only `test_rag_fallback.py` is a real, passing test |
| Multi-provider LLM support (Together etc.) | 🟡 | `TOGETHER_API_KEY` present in config, not wired into `llm_service.py` |
| User accounts / auth | ⬜ | Not started |
| Persistent chat history (per user) | ⬜ | Not started — history lives only in the Streamlit session |
| Rate limiting / abuse protection | ⬜ | Not started |
| CI pipeline | ⬜ | Not started |

## 2. Roadmap

### Phase 1 — Correctness & retrieval quality (near-term)
1. Implement `reranker.py` (cross-encoder, e.g.
   `cross-encoder/ms-marco-MiniLM-L-6-v2`) and wire it into
   `HybridSearcher.search()` between the RRF fusion step and the final
   top-k cut — rerank the fused top ~15-20 candidates down to the top 5
   actually sent to Gemini. This should measurably improve precision on
   ambiguous, multi-section questions.
2. Implement `utils/citations.py` so citation formatting is centralized
   (currently the frontend hand-formats `f"{act} - Section {n} - {title}"`
   inline) — one source of truth for how a citation is displayed.
3. Replace `print()` startup diagnostics with real logging via
   `core/logging.py`, at minimum: request id, question length, retrieval
   latency, LLM latency, fallback-triggered flag.
4. Add real tests: replace `tests/test_skeleton.py`'s
   `raise NotImplementedError` with actual coverage for
   `HybridSearcher.search`, `VectorStore.save/load` round-trip, and
   `build_grounded_prompt`.

### Phase 2 — Data pipeline hygiene
5. Implement `ingestion/cleaner.py` and `ingestion/chunker.py` for real,
   so long sections (e.g. lengthy Constitution articles) get split into
   overlapping, citation-preserving chunks instead of being embedded as a
   single (possibly truncated) vector each.
6. Wire `ingestion/embedder.py` and `ingestion/build_index.py` so
   `backend/scripts/build_combined_index.py` becomes a thin CLI wrapper
   around the ingestion package rather than duplicating its own
   embed/build logic.
7. Add a data-refresh path using the existing `indian_kanoon.py` scraper
   to add more Acts / stay current with amendments, with clear
   provenance metadata (source URL, date scraped) per section.

### Phase 3 — API hardening
8. Implement `core/security.py` — API key or JWT-based auth on `/query`
   before any public deployment; today anyone who can reach port 8000 can
   query it for free, unlimited, unauthenticated Gemini calls.
9. Add rate limiting (e.g. `slowapi`) — Gemini calls cost money and quota;
   right now nothing stops one user from exhausting the shared quota for
   everyone.
10. Widen/parameterize CORS (`allow_origins`) instead of the hardcoded
    `http://localhost:8501`, driven by an environment variable, before
    deploying frontend and backend on different hosts.
11. Add a `/query` response field indicating whether the answer came from
    Gemini or the fallback path, so the frontend can visually flag
    degraded answers instead of presenting them identically.

### Phase 4 — Multi-provider LLM support
12. Wire `TOGETHER_API_KEY` into `llm_service.py` as a second provider
    behind `generate_answer(provider=...)`, so a Gemini outage doesn't
    take down answer generation entirely (currently it only degrades to
    the non-LLM fallback, not to a second LLM).

### Phase 5 — Persistence & multi-user (larger scope change)
13. Decide if/when to introduce a real database. This is the biggest
    architectural fork in the project — see `notes.md` §3 for the
    trade-off. If pursued: add a `users`/`sessions` table, move chat
    history out of `st.session_state` and into that store, and add
    auth (ties into Phase 3, item 8).
14. If multi-user persistence lands, revisit whether Streamlit remains
    the right frontend (see `notes.md` §1) — Streamlit's session model
    doesn't map cleanly onto durable, cross-device chat history.

## 3. Suggested immediate next step

Of everything above, **Phase 1, item 1 (reranker)** has the best
effort-to-impact ratio: the interface (`rerank_results`) already exists,
`HybridSearcher` already returns everything the reranker needs
(query, candidates with metadata), and it directly improves the thing
users actually judge the product on — whether the cited sections are the
right ones.