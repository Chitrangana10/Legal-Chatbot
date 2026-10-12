import json
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import build_router, current_user
from backend.app.services.answer_cache import AnswerCache


class MemoryRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex):
        self.values[key] = value
        self.ttls[key] = ex


def test_key_isolation_version_and_ttl():
    redis = MemoryRedis()
    cache = AnswerCache(redis, "corpus-one", 60)
    answer = {"answer": "Answer", "sources": [], "conversation_id": "private-id"}
    cache.put("alice", "Question?", answer)
    assert cache.get("alice", "Question?")["answer"] == "Answer"
    assert cache.get("bob", "Question?") is None
    assert cache.get("alice", "Different?") is None
    assert AnswerCache(redis, "corpus-two").get("alice", "Question?") is None
    assert "conversation_id" not in json.loads(redis.values[cache.key("alice", "Question?")])
    assert redis.ttls[cache.key("alice", "Question?")] == 60


def test_degraded_invalid_and_unavailable_cache():
    redis = MemoryRedis()
    cache = AnswerCache(redis)
    cache.put("alice", "Q", {"answer": "Fallback", "sources": [], "degraded": True})
    assert cache.get("alice", "Q") is None
    cache.put("alice", "Refusal", {"answer": "I cannot find this in the available sources.", "sources": []})
    assert cache.key("alice", "Refusal") not in redis.values
    redis.values[cache.key("alice", "Q")] = "invalid json"
    assert cache.get("alice", "Q") is None
    failed = Mock()
    failed.get.side_effect = ConnectionError()
    failed.set.side_effect = ConnectionError()
    cache = AnswerCache(failed)
    assert cache.get("alice", "Q") is None
    cache.put("alice", "Q", {"answer": "Answer", "sources": []})


def test_default_expiry_is_one_day():
    redis = MemoryRedis()
    cache = AnswerCache(redis)
    cache.put("alice", "Q", {"answer": "Answer", "sources": []})
    assert redis.ttls[cache.key("alice", "Q")] == 86400


def test_cache_hit_skips_generation_but_saves_each_turn():
    app = FastAPI()
    user = {"id": "alice"}
    app.dependency_overrides[current_user] = lambda: user
    app.state.answer_cache = AnswerCache(MemoryRedis())
    app.state.rag_engine = Mock()
    app.state.rag_engine.answer.return_value = {"answer": "Answer", "sources": []}
    app.state.history_store = Mock()
    app.state.history_store.save_turn.side_effect = ["chat-one", "chat-two", "chat-three"]
    app.include_router(build_router())
    with TestClient(app) as client:
        first = client.post("/query", json={"question": "Question?"}).json()
        second = client.post("/query", json={"question": "Question?"}).json()
        assert first["cached"] is False
        assert second["cached"] is True
        assert first["conversation_id"] != second["conversation_id"]
        assert app.state.rag_engine.answer.call_count == 1
        assert app.state.history_store.save_turn.call_count == 2
        user["id"] = "bob"
        assert client.post("/query", json={"question": "Question?"}).json()["cached"] is False
        assert app.state.rag_engine.answer.call_count == 2


def test_ownership_is_checked_even_for_cached_answers():
    app = FastAPI()
    app.dependency_overrides[current_user] = lambda: {"id": "alice"}
    app.state.answer_cache = AnswerCache(MemoryRedis())
    app.state.answer_cache.put("alice", "Q", {"answer": "Answer", "sources": []})
    app.state.history_store = Mock()
    app.state.history_store.assert_owner = Mock(side_effect=LookupError())
    app.state.rag_engine = Mock()
    app.include_router(build_router())
    with TestClient(app) as client:
        result = client.post("/query", json={"question": "Q", "conversation_id": "00000000-0000-0000-0000-000000000001"})
        assert result.status_code == 404
        app.state.history_store.save_turn.assert_not_called()
