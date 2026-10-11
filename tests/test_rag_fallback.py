"""Regression tests for the RAG fallback behavior when Gemini is unavailable.

"The law does not cover this" and "the answer service is down" need completely
different responses from the user, so a silent fallback - which looks like the
first when it is really the second - is a bug, not a graceful degrade.
"""

from backend.app.services.rag_engine import (
    build_fallback_answer,
    describe_llm_failure,
    is_llm_error_response,
)


def _retrieved(*pairs) -> list:
    return [
        {"metadata": {"act": act, "section_number": number, "section_title": title}}
        for act, number, title in pairs
    ]


def test_detects_gemini_quota_errors() -> None:
    """Quota and rate-limit responses should trigger the fallback answer path."""
    error_text = "Gemini error: 429 You exceeded your current quota"
    assert is_llm_error_response(error_text) is True

    rate_limit_text = "Gemini error: 429 rate limit exceeded"
    assert is_llm_error_response(rate_limit_text) is True


def test_a_real_answer_is_not_mistaken_for_an_error() -> None:
    assert is_llm_error_response("Whoever commits murder shall be punished with death.") is False


def test_quota_failure_is_named_explicitly() -> None:
    notice = describe_llm_failure("Gemini error: 429 You exceeded your current quota")
    assert "quota" in notice.lower()


def test_rate_limiting_is_distinguished_from_quota() -> None:
    notice = describe_llm_failure("Gemini error: rate limit, retry_delay 30s")
    assert "rate limit" in notice.lower()
    assert "quota" not in notice.lower()


def test_an_unknown_failure_still_produces_a_plain_reason() -> None:
    notice = describe_llm_failure("Gemini error: connection refused")
    assert notice
    assert "could not be reached" in notice.lower()


def test_fallback_lists_the_sections_that_were_found() -> None:
    answer = build_fallback_answer(
        "What is the punishment for murder?",
        _retrieved(
            ("The Indian Penal Code, 1860", "302", "Punishment for murder."),
            ("The Indian Penal Code, 1860", "304", "Punishment for culpable homicide."),
        ),
    )

    assert "Section 302" in answer
    assert "Section 304" in answer
    assert "not a written answer" in answer


def test_fallback_with_nothing_retrieved_does_not_claim_to_have_sources() -> None:
    answer = build_fallback_answer("something no act covers", [])

    assert "no sections" in answer.lower()
    assert "Section " not in answer
