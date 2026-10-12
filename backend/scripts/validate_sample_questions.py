"""Check UI examples against real retrieval; optionally verify Gemini answers."""

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.services.answer_cache import is_source_refusal
from backend.app.services.query_expansion import expand_query
from backend.app.services.rag_engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Also call Gemini for every example")
    args = parser.parse_args()
    tree = ast.parse((ROOT / "frontend/streamlit_app.py").read_text(encoding="utf-8"))
    questions = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "SAMPLE_QUESTIONS" for target in node.targets))
    expected = [("penal", "302"), ("penal", "379"), ("penal", "406"),
                ("criminal procedure", "154"), ("information technology", "66C"), ("motor vehicles", "181")]
    if len(questions) != len(expected):
        raise RuntimeError("Update expected sections when changing the sample list")
    questions.append("What is the penalty for driving without a valid licence?")
    expected.append(("motor vehicles", "181"))
    engine = RAGEngine()
    failures = 0
    for question, (act, section) in zip(questions, expected):
        query = expand_query(question)
        candidates = engine.hybrid_searcher.search(query, top_k=200, candidate_k=10)
        retrieved = engine.reranker.rerank(query, candidates, top_k=5) if engine.reranker else candidates[:5]
        supported = bool(retrieved) and act in retrieved[0]["metadata"]["act"].lower() and str(retrieved[0]["metadata"]["section_number"]).strip(".") == section
        generated = True
        if args.generate:
            result = engine.answer(question)
            generated = not result.get("degraded") and not is_source_refusal(result) and bool(result.get("answer"))
        passed = supported and generated
        failures += not passed
        print(f"{'PASS' if passed else 'FAIL'}: {question} (retrieval={supported}, generation={'passed' if generated else 'failed'}{'' if args.generate else ' [not requested]'})", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
