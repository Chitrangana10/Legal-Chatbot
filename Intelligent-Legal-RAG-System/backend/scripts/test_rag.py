"""Run a minimal Gemini-grounded RAG smoke test over the IPC FAISS index."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.rag_engine import RAGEngine


def main() -> None:
    """Ask a few hardcoded questions and print generated answers with sources."""
    questions = [
        "What is the punishment for murder?",
        "What acts amount to cheating under the IPC?",
        "When can several people be liable for the same criminal act?",
        "What is the punishment for theft?",
    ]

    engine = RAGEngine()
    for question in questions:
        result = engine.answer(question)
        print("=" * 80)
        print(f"Question: {question}")
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- IPC Section {source['section_number']}: {source['section_title']}")


if __name__ == "__main__":
    main()
