"""Define HTTP routes for legal query and retrieval workflows."""

from typing import Union

from fastapi import APIRouter, Request, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from uuid import UUID
from fastapi.responses import JSONResponse

from backend.app.api.schemas import Credentials, Registration, ErrorResponse, HealthResponse, QueryRequest, QueryResponse
from backend.app.services.answer_cache import AnswerCache


router = APIRouter()
bearer = HTTPBearer(auto_error=False)


def current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if credentials is None:
        raise HTTPException(401, "Please sign in.", headers={"WWW-Authenticate": "Bearer"})
    try:
        return request.app.state.auth_service.authenticate(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc


def auth_budget(request, email):
    # Never trust public forwarding headers. Streamlit requests share its IP,
    # so use a separate tighter budget per email/IP pair.
    identifier = request.client.host if request.client else "unknown"
    try:
        request.app.state.auth_service.throttle("ip:" + identifier, limit=100)
        request.app.state.auth_service.throttle(identifier + ":" + email)
    except PermissionError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "900"}) from exc


@router.post("/auth/register", status_code=201)
def register(payload: Registration, request: Request):
    auth_budget(request, payload.email)
    try:
        return request.app.state.auth_service.register(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/auth/login")
def login(payload: Credentials, request: Request):
    auth_budget(request, payload.email)
    try:
        return request.app.state.auth_service.login(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(401, str(exc)) from exc


@router.get("/auth/me")
def me(user=Depends(current_user)):
    return user


@router.post("/auth/logout", status_code=204)
def logout(request: Request, user=Depends(current_user), credentials=Depends(bearer)):
    request.app.state.auth_service.logout(credentials.credentials)


@router.get("/conversations")
def conversations(request: Request, limit: int = Query(50, ge=1, le=100),
                  offset: int = Query(0, ge=0), user=Depends(current_user)):
    return request.app.state.history_store.list_conversations(user["id"], limit, offset)


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: UUID, request: Request, user=Depends(current_user)):
    try:
        return request.app.state.history_store.get_conversation(user["id"], str(conversation_id))
    except LookupError as exc:
        raise HTTPException(404, "Conversation not found") from exc


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: UUID, request: Request, user=Depends(current_user)):
    try:
        request.app.state.history_store.delete_conversation(user["id"], str(conversation_id))
    except LookupError as exc:
        raise HTTPException(404, "Conversation not found") from exc


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return API health and whether the FAISS-backed RAG engine loaded."""
    return HealthResponse(
        status="ok",
        index_loaded=bool(getattr(request.app.state, "index_loaded", False)),
    )


@router.post("/query", response_model=QueryResponse, responses={500: {"model": ErrorResponse}})
def query(payload: QueryRequest, request: Request, user=Depends(current_user)) -> Union[QueryResponse, JSONResponse]:
    """Answer a legal question using the startup-loaded RAG engine."""
    conversation_id = str(payload.conversation_id) if payload.conversation_id else None
    if conversation_id:
        try:
            request.app.state.history_store.assert_owner(user["id"], conversation_id)
        except LookupError as exc:
            raise HTTPException(404, "Conversation not found") from exc
    rag_engine = getattr(request.app.state, "rag_engine", None)
    if rag_engine is None:
        return JSONResponse(status_code=500, content={"error": "RAG engine is not loaded."})

    cache = getattr(request.app.state, "answer_cache", None) or AnswerCache()
    result = cache.get(user["id"], payload.question)
    cached = result is not None
    if not cached:
        try:
            result = rag_engine.answer(payload.question)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": f"RAG query failed: {exc}"})

    # Validate generated content before committing any history.
    response = QueryResponse(**result)
    response.cached = cached
    try:
        response.conversation_id = request.app.state.history_store.save_turn(
            user["id"], conversation_id, payload.question,
            response.model_dump(exclude={"conversation_id", "cached"}),
        )
    except LookupError as exc:
        raise HTTPException(404, "Conversation not found") from exc
    if not cached:
        cache.put(user["id"], payload.question, response.model_dump())
    return response


def build_router() -> APIRouter:
    """Return the API router for the legal assistant."""
    return router
