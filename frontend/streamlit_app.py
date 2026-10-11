"""Render a Streamlit chat interface for the legal RAG backend."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


# "backend" is the service name inside docker compose and does not resolve
# anywhere else, so running the two processes directly needs localhost.
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/") + "/query"

# Spread across the eight acts in the corpus so the first click shows range.
SAMPLE_QUESTIONS = [
    "When can the police arrest someone without a warrant?",
    "What is the punishment for criminal breach of trust?",
    "Can a WhatsApp chat be used as evidence in court?",
    "What can I do if an online seller refuses a refund?",
    "What is the penalty for driving without a valid licence?",
    "Which fundamental rights can be suspended during an Emergency?",
]

# Translucent overlays rather than fixed colours, so the accents sit on
# whatever dark surface Streamlit's theme is painting underneath.
STYLES = """
<style>
.block-container { padding-top: 3rem; }

.sq-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.55;
    margin: 0.5rem 0 0.6rem;
}

.stButton > button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    white-space: normal;
    height: auto;
    min-height: 0;
    padding: 0.6rem 0.85rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    font-weight: 400;
    font-size: 0.89rem;
    line-height: 1.4;
    transition: background 180ms ease, border-color 180ms ease, transform 180ms ease;
}
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(214, 168, 95, 0.55);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button:focus-visible { outline: 2px solid rgba(214, 168, 95, 0.8); outline-offset: 2px; }
.stButton > button p { text-align: left; }

.src-row {
    display: flex;
    gap: 0.7rem;
    align-items: baseline;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}
.src-row:last-child { border-bottom: none; }
.src-sec {
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 0.8rem;
    color: #d6a85f;
    white-space: nowrap;
}
.src-title { font-size: 0.88rem; line-height: 1.4; }
.src-act { display: block; font-size: 0.75rem; opacity: 0.55; }
</style>
"""


def submit_query(question: str) -> Dict[str, Any]:
    """Submit a legal question to the backend API and return its JSON response."""
    response = requests.post(API_URL, json={"question": question}, timeout=90)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def build_followups(sources: List[Dict[str, Any]]) -> List[str]:
    """Derive follow-up questions from the sections actually retrieved.

    Grounding the suggestions in real sources keeps them answerable - a
    generic "tell me more" would often land outside the indexed corpus.
    """
    templates = [
        "Explain Section {num} of the {act} in plain language",
        "How is Section {num} applied in practice?",
        "Which other sections deal with {title}?",
    ]
    followups: List[str] = []
    seen = set()

    for source in sources:
        num = str(source.get("section_number", "")).strip()
        act = str(source.get("act", "")).strip()
        title = str(source.get("section_title", "")).strip().lower()
        if not num or (num, act) in seen:
            continue
        seen.add((num, act))
        followups.append(
            templates[len(followups) % len(templates)].format(
                num=num, act=act or "act", title=title or "this section"
            )
        )
        if len(followups) == 3:
            break

    return followups


def ask(question: str) -> None:
    """Queue a question so every turn enters the app through one path."""
    st.session_state.pending = question.strip()


def render_question_chips(questions: List[str], key_prefix: str, columns: int = 1) -> None:
    """Render clickable question chips that queue themselves when pressed."""
    slots = st.columns(columns) if columns > 1 else [st]
    for index, question in enumerate(questions):
        slot = slots[index % len(slots)]
        slot.button(question, key=f"{key_prefix}-{index}", on_click=ask, args=(question,))


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render retrieved legal sources in an expandable section."""
    label = f"Sources ({len(sources)})" if sources else "Sources"
    with st.expander(label, expanded=False):
        if not sources:
            st.write("No sources returned.")
            return

        rows = "".join(
            "<div class='src-row'>"
            f"<span class='src-sec'>S. {source.get('section_number', '?')}</span>"
            f"<span class='src-title'>{source.get('section_title', 'Untitled section')}"
            f"<span class='src-act'>{source.get('act', 'Unknown Act')}</span>"
            "</span></div>"
            for source in sources
        )
        st.markdown(rows, unsafe_allow_html=True)


def render_notice(item: Dict[str, Any]) -> None:
    """Say when the sections below are raw retrieval, not a written answer.

    Without this a quota failure reads exactly like "the law does not cover
    your question", and those call for opposite reactions from the reader.
    """
    if item.get("degraded"):
        st.warning(item.get("notice") or "The answer service is unavailable; showing retrieved sections only.")


def render_history() -> None:
    """Render the session's previous legal Q&A pairs from state alone."""
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            render_notice(item)
            st.write(item["answer"])
            render_sources(item.get("sources", []))

    last = st.session_state.history[-1] if st.session_state.history else None
    if last and last.get("followups"):
        st.markdown("<p class='sq-label'>Follow up with</p>", unsafe_allow_html=True)
        render_question_chips(last["followups"], key_prefix=f"followup-{len(st.session_state.history)}")


def process_pending() -> None:
    """Answer the queued question, store the turn, then redraw from state."""
    question = st.session_state.pending
    st.session_state.pending = None

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the statutes..."):
            try:
                result = submit_query(question)
            except Exception:
                st.error(
                    "Couldn't reach the answer service. "
                    "Check that the backend is running, then try again."
                )
                return

    sources = result.get("sources", [])
    st.session_state.history.append(
        {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": sources,
            "degraded": result.get("degraded", False),
            "notice": result.get("notice", ""),
            "followups": [] if result.get("degraded") else build_followups(sources),
        }
    )
    # Redraw so the finished turn is painted once, from history, in its final
    # position - drawing it inline here is what made the page jump.
    st.rerun()


def render_app() -> None:
    """Render the legal assistant Streamlit application."""
    st.set_page_config(page_title="Legal Assistant", page_icon="⚖️")
    st.markdown(STYLES, unsafe_allow_html=True)

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)

    if not st.session_state.history and not st.session_state.pending:
        st.markdown("<p class='sq-label'>Start with</p>", unsafe_allow_html=True)
        render_question_chips(SAMPLE_QUESTIONS, key_prefix="sample", columns=2)

    render_history()

    typed = st.chat_input("Ask a legal question")
    if typed and typed.strip():
        ask(typed)

    if st.session_state.pending:
        process_pending()


if __name__ == "__main__":
    render_app()
