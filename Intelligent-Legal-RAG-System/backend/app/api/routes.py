"""Define HTTP routes for legal query and retrieval workflows."""

from typing import Union

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.app.api.schemas import ErrorResponse, HealthResponse, QueryRequest, QueryResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return API health and whether the FAISS-backed RAG engine loaded."""
    return HealthResponse(
        status="ok",
        index_loaded=bool(getattr(request.app.state, "index_loaded", False)),
    )


@router.post("/query", response_model=QueryResponse, responses={500: {"model": ErrorResponse}})
def query(payload: QueryRequest, request: Request) -> Union[QueryResponse, JSONResponse]:
    """Answer a legal question using the startup-loaded RAG engine."""
    rag_engine = getattr(request.app.state, "rag_engine", None)
    if rag_engine is None:
        return JSONResponse(status_code=500, content={"error": "RAG engine is not loaded."})

    try:
        result = rag_engine.answer(payload.question)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"RAG query failed: {exc}"})

    return QueryResponse(**result)


def build_router() -> APIRouter:
    """Return the API router for the legal assistant."""
    return router
