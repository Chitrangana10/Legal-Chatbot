"""Optional Redis cache; outages and invalid entries fall back to normal retrieval."""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_source_refusal(response):
    return "i cannot find this in the available sources" in str(response.get("answer", "")).lower()

def cache_version(settings):
    """Invalidate on index, generation/retrieval code, or model changes."""
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    digest.update(json.dumps([settings.default_llm_model, settings.embedding_model_name,
                              settings.enable_reranker, settings.reranker_model_name,
                              settings.answer_cache_version]).encode())
    paths = [root / settings.faiss_index_dir / name for name in ("index.faiss", "metadata.json")]
    paths += [Path(__file__).parent / name for name in
              ("rag_engine.py", "llm_service.py", "query_expansion.py", "hybrid_search.py", "reranker.py")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Cannot fingerprint retrieval assets")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class AnswerCache:
    def __init__(self, client=None, version="v1", ttl=86400):
        self.client = client
        self.version = version
        self.ttl = ttl

    @classmethod
    def from_settings(cls, settings):
        if not settings.enable_answer_cache:
            return cls()
        try:
            from redis import Redis
            client = Redis.from_url(settings.redis_url, decode_responses=True,
                                    socket_connect_timeout=0.5, socket_timeout=0.5)
            return cls(client, cache_version(settings), settings.answer_cache_ttl_seconds)
        except Exception:
            logger.warning("Redis cache initialization failed; continuing without caching")
            return cls()

    def key(self, user_id, question):
        identity = hashlib.sha256(json.dumps([user_id, question.strip()], ensure_ascii=False).encode()).hexdigest()
        return f"legal:answers:{self.version}:{identity}"

    def get(self, user_id, question):
        if self.client is None:
            return None
        try:
            raw = self.client.get(self.key(user_id, question))
            if raw is None:
                return None
            from backend.app.api.schemas import QueryResponse
            payload = json.loads(raw)
            # Cache never transports another conversation's ID or failed answers.
            if (not isinstance(payload, dict) or payload.get("conversation_id") or
                    payload.get("degraded") or is_source_refusal(payload)):
                return None
            validated = QueryResponse(**payload)
            return validated.model_dump(exclude={"conversation_id", "cached"})
        except Exception:
            logger.warning("Redis read failed or invalid cache entry; using normal retrieval")
            return None

    def put(self, user_id, question, response):
        if self.client is None or response.get("degraded") or is_source_refusal(response):
            return
        payload = {key: response[key] for key in ("answer", "sources", "degraded", "notice") if key in response}
        try:
            self.client.set(self.key(user_id, question), json.dumps(payload), ex=self.ttl)
        except Exception:
            logger.warning("Redis write failed; returning the generated answer")

    def close(self):
        if self.client is not None:
            self.client.close()
