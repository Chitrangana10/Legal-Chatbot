"""Render a Streamlit chat interface for the legal RAG backend."""

from __future__ import annotations

from typing import Any, Dict, List

import requests
import streamlit as st


API_URL = "http://backend:8000/query"


def submit_query(question: str) -> Dict[str, Any]:
    """Submit a legal question to the backend API and return its JSON response."""
    response = requests.post(API_URL, json={"question": question}, timeout=90)
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render retrieved legal sources in an expandable section."""
    with st.expander("Sources", expanded=False):
        if not sources:
            st.write("No sources returned.")
            return

        for source in sources:
          st.write(f"{source.get('act', 'Unknown Act')} - Section {source['section_number']} - {source['section_title']}")

def render_history() -> None:
    """Render the session's previous legal Q&A pairs."""
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            render_sources(item.get("sources", []))


def render_app() -> None:
    """Render the legal assistant Streamlit application."""
    st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️")
    st.title("Legal RAG Assistant")

    if "history" not in st.session_state:
        st.session_state.history = []

    render_history()

    with st.form("query_form", clear_on_submit=True):
        question = st.text_input("Ask a legal question")
        submitted = st.form_submit_button("Submit")

    if not submitted:
        return

    if not question.strip():
        st.warning("Please enter a legal question.")
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal sources and generating an answer..."):
            try:
                result = submit_query(question.strip())
            except requests.RequestException:
                st.error("Something went wrong - is the backend running?")
                return
            except Exception:
                st.error("Something went wrong - is the backend running?")
                return

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        st.write(answer)
        render_sources(sources)

    st.session_state.history.append(
        {
            "question": question.strip(),
            "answer": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    render_app()
