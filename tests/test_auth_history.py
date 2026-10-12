"""Auth boundary, revocation and cross-user history checks without loading ML models."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.routes import build_router
from backend.app.services.auth_service import AuthService, token_digest, verify_password
from backend.app.services.history_store import Base, HistoryStore, LoginSession, User


@pytest.fixture
def client():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated PostgreSQL test database")
    from sqlalchemy.engine import make_url
    if not (make_url(url).database or "").endswith("_test"):
        raise ValueError("TEST_DATABASE_URL must point to a disposable database whose name ends in _test")
    store = HistoryStore(url)
    Base.metadata.drop_all(store.engine)
    store.initialize()
    app = FastAPI()
    app.state.history_store = store
    app.state.auth_service = AuthService(store)
    app.state.rag_engine = Mock()
    app.state.rag_engine.answer.return_value = {"answer": "A grounded answer", "sources": []}
    app.include_router(build_router())
    with TestClient(app) as client:
        yield client
    Base.metadata.drop_all(store.engine)
    store.engine.dispose()


def account(client, email):
    credentials = {"email": email, "password": "a-long-test-password"}
    assert client.post("/auth/register", json=credentials).status_code == 201
    response = client.post("/auth/login", json=credentials)
    assert response.status_code == 200
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def test_passwords_are_hashed_and_sessions_are_revocable(client):
    headers = account(client, "Alice@Example.com")
    assert client.get("/auth/me", headers=headers).json()["email"] == "alice@example.com"
    with Session(client.app.state.history_store.engine) as session:
        user = session.scalar(select(User))
        assert user.password_hash != "a-long-test-password"
        assert verify_password("a-long-test-password", user.password_hash)
        assert not verify_password("incorrect", user.password_hash)
        row = session.scalar(select(LoginSession))
        assert row.token_hash == token_digest(headers["Authorization"].removeprefix("Bearer "))
    assert client.post("/auth/logout", headers=headers).status_code == 204
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_queries_require_login_and_history_is_isolated(client):
    assert client.post("/query", json={"question": "Question"}).status_code == 401
    alice = account(client, "alice@example.com")
    bob = account(client, "bob@example.com")
    result = client.post("/query", headers=alice, json={"question": "Question"})
    assert result.status_code == 200
    conversation_id = result.json()["conversation_id"]
    path = "/conversations/" + conversation_id
    assert len(client.get("/conversations", headers=alice).json()) == 1
    assert client.get("/conversations", headers=bob).json() == []
    assert client.get(path, headers=bob).status_code == 404
    assert client.delete(path, headers=bob).status_code == 404
    client.app.state.rag_engine.answer.reset_mock()
    assert client.post("/query", headers=bob, json={"question": "Intrusion", "conversation_id": conversation_id}).status_code == 404
    client.app.state.rag_engine.answer.assert_not_called()
    assert client.post("/query", headers=alice, json={"question": "Second", "conversation_id": conversation_id}).status_code == 200
    assert len(client.get(path, headers=alice).json()["turns"]) == 2
    # Opening a new database connection retains saved history.
    reopened = HistoryStore(client.app.state.history_store.engine.url.render_as_string(hide_password=False))
    user_id = client.get("/auth/me", headers=alice).json()["id"]
    assert len(reopened.get_conversation(user_id, conversation_id)["turns"]) == 2
    reopened.engine.dispose()
    assert client.delete(path, headers=alice).status_code == 204
    assert client.get(path, headers=alice).status_code == 404


def test_expired_sessions_and_invalid_credentials(client):
    headers = account(client, "alice@example.com")
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": "wrong"}).status_code == 401
    assert client.post("/auth/register", json={"email": "bad", "password": "short"}).status_code == 422
    assert client.post("/query", headers=headers, json={"question": "   "}).status_code == 422
    with Session(client.app.state.history_store.engine) as session, session.begin():
        row = session.scalar(select(LoginSession))
        row.expires_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert client.get("/conversations", headers=headers).status_code == 401


def test_failed_authentication_is_rate_limited(client):
    payload = {"email": "unknown@example.com", "password": "wrong"}
    for _ in range(10):
        assert client.post("/auth/login", json=payload).status_code == 401
    assert client.post("/auth/login", json=payload).status_code == 429


def test_failed_query_does_not_create_history(client):
    headers = account(client, "alice@example.com")
    client.app.state.rag_engine.answer.side_effect = RuntimeError("LLM unavailable")
    assert client.post("/query", headers=headers, json={"question": "Question"}).status_code == 500
    assert client.get("/conversations", headers=headers).json() == []
