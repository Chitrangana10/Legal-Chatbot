"""The reranker must reorder by relevance without losing or corrupting results.

Uses a stub model so the behaviour is tested without downloading or running a
real cross-encoder.
"""

from backend.app.services.reranker import CrossEncoderReranker


class _StubModel:
    """Scores a pair by how many query words appear in the section text."""

    def __init__(self) -> None:
        self.seen_pairs = []

    def predict(self, pairs):
        self.seen_pairs = list(pairs)
        scores = []
        for query, document in pairs:
            words = set(query.lower().split())
            scores.append(float(sum(word in document.lower() for word in words)))
        return scores


def _result(number: str, title: str, text: str, score: float) -> dict:
    return {
        "metadata": {
            "act": "The Motor Vehicles Act, 1988",
            "section_number": number,
            "section_title": title,
            "text": text,
        },
        "score": score,
        "combined_score": score,
    }


def _results() -> list:
    return [
        _result("39", "Necessity for registration", "No person shall drive an unregistered vehicle.", 0.9),
        _result("192", "Using vehicles without registration", "Whoever drives shall be punishable with fine.", 0.5),
        _result("40", "Registration where to be made", "The owner shall apply to the authority.", 0.4),
    ]


def test_results_are_reordered_by_the_model_not_by_retrieval() -> None:
    reranker = CrossEncoderReranker(model=_StubModel())

    reranked = reranker.rerank("punishable fine drives", _results(), top_k=3)

    assert reranked[0]["metadata"]["section_number"] == "192"
    assert [r["metadata"]["section_number"] for r in reranked] == ["192", "39", "40"]


def test_the_original_retrieval_score_is_kept_for_debugging() -> None:
    reranker = CrossEncoderReranker(model=_StubModel())

    reranked = reranker.rerank("punishable fine drives", _results(), top_k=3)

    top = reranked[0]
    assert top["retrieval_score"] == 0.5
    assert top["rerank_score"] == top["score"]
    assert top["rerank_score"] != top["retrieval_score"]


def test_top_k_is_respected_and_metadata_survives() -> None:
    reranker = CrossEncoderReranker(model=_StubModel())

    reranked = reranker.rerank("registration", _results(), top_k=2)

    assert len(reranked) == 2
    for result in reranked:
        assert set(result["metadata"]) >= {"act", "section_number", "section_title", "text"}


def test_the_model_sees_the_section_title_not_just_its_body() -> None:
    """A section is often identified by its title alone, so the pair handed to
    the model must include it."""
    stub = _StubModel()
    CrossEncoderReranker(model=stub).rerank("registration", _results(), top_k=3)

    assert all("Section" in document for _query, document in stub.seen_pairs)
    assert any("Necessity for registration" in document for _query, document in stub.seen_pairs)


def test_no_results_is_not_an_error() -> None:
    assert CrossEncoderReranker(model=_StubModel()).rerank("anything", [], top_k=5) == []
