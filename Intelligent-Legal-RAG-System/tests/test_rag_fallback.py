"""Regression tests for the RAG fallback behavior when Gemini is unavailable."""

from backend.app.services.rag_engine import is_llm_error_response


def test_detects_gemini_quota_errors() -> None:
    """Quota and rate-limit responses should trigger the fallback answer path."""
    error_text = "Gemini error: 429 You exceeded your current quota"
    assert is_llm_error_response(error_text) is True

    rate_limit_text = "Gemini error: 429 rate limit exceeded"
    assert is_llm_error_response(rate_limit_text) is True
