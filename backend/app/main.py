"""Create and configure the FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import build_router
from backend.app.config import get_settings
from backend.app.services.rag_engine import RAGEngine
from backend.app.services.history_store import HistoryStore
from backend.app.services.auth_service import AuthService
from backend.app.services.answer_cache import AnswerCache


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the FAISS-backed RAG engine once when the API starts."""
    store = HistoryStore(get_settings().database_url)
    store.initialize()
    app.state.history_store = store
    app.state.auth_service = AuthService(store, get_settings().session_hours)
    app.state.answer_cache = AnswerCache.from_settings(get_settings())
    try:
        app.state.rag_engine = RAGEngine()
        app.state.index_loaded = True
    except Exception:
        app.state.rag_engine = None
        app.state.index_loaded = False
    try:
        yield
    finally:
        app.state.answer_cache.close()
        store.engine.dispose()


def create_app() -> FastAPI:
    """Build the FastAPI application with routers, CORS, and lifecycle hooks."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_router())
    return app


app = create_app()
