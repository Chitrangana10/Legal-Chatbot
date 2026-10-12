"""Render a Streamlit chat interface for the legal RAG backend."""

from __future__ import annotations

import os
from html import escape
from typing import Any, Dict, List

import requests
import streamlit as st


# "backend" is the service name inside docker compose and does not resolve
# anywhere else, so running the two processes directly needs localhost.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Explicit sections keep examples anchored to the bundled statute text.
SAMPLE_QUESTIONS = [
    "What punishment does IPC Section 302 prescribe for murder?",
    "What punishment does IPC Section 379 prescribe for theft?",
    "What punishment does IPC Section 406 prescribe for criminal breach of trust?",
    "What does CrPC Section 154 say about information in cognisable cases?",
    "What punishment does IT Act Section 66C prescribe for identity theft?",
    "What penalty does Motor Vehicles Act Section 181 prescribe for driving without a valid licence?",
]

# Translucent overlays rather than fixed colours, so the accents sit on
# whatever dark surface Streamlit's theme is painting underneath.
STYLES = """
<style>
.block-container { padding-top: 2.5rem; max-width: 1000px; }
.brand-header { padding: 1rem 0 1.6rem; border-bottom: 1px solid rgba(214,168,95,.22); margin-bottom: 1.8rem; }
.brand-kicker { color: #d6a85f; font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; }
.brand-header h1 { font-size: clamp(2rem, 5vw, 3rem); letter-spacing: -.04em; margin: .35rem 0; }
.brand-header p { opacity: .72; margin: 0; font-size: 1rem; }
.welcome { padding: 1.5rem; border: 1px solid rgba(214,168,95,.2); border-radius: 18px; background: linear-gradient(120deg, rgba(214,168,95,.12), rgba(214,168,95,.02)); margin-bottom: 1.5rem; }
.welcome h2 { font-size: 1.55rem; margin: 0 0 .5rem; letter-spacing: -.025em; }
.welcome p { opacity: .72; margin: 0; line-height: 1.7; }
.coverage { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
.coverage span { font-size: .75rem; border: 1px solid rgba(214,168,95,.25); border-radius: 20px; padding: .3rem .7rem; }
[data-testid="stChatMessage"] { border: 1px solid rgba(255,255,255,.06); border-radius: 14px; }
@media (max-width: 640px) { .block-container { padding-top: 1rem; } .welcome { padding: 1rem; } }

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


def api_request(method, path, payload=None, authenticated=True):
    headers = {}
    if authenticated:
        headers["Authorization"] = "Bearer " + st.session_state.get("access_token", "")
    try:
        response = requests.request(method, BACKEND_URL + path, json=payload, headers=headers, timeout=90)
    except requests.RequestException as exc:
        raise RuntimeError("Couldn't reach the answer service. Please try again.") from exc
    if response.status_code == 401 and authenticated:
        reset_account()
        st.session_state.auth_notice = "Your session expired. Please sign in again."
        st.rerun()
    if not response.ok:
        try:
            detail = response.json().get("detail", "The request failed. Please try again.")
        except ValueError:
            detail = "The request failed. Please try again."
        if not isinstance(detail, str):
            detail = "Check your email and password. Passwords must contain 4–128 characters."
        raise RuntimeError(detail)
    return response.json() if response.content else None


def submit_query(question: str) -> Dict[str, Any]:
    """Submit a legal question to the backend API and return its JSON response."""
    return api_request("POST", "/query", {"question": question,
                                          "conversation_id": st.session_state.get("conversation_id")})


def reset_account():
    for key in ("access_token", "user", "conversation_id", "auth_notice"):
        st.session_state.pop(key, None)
    clear_chat()
    st.session_state.clear_auth_fields = True


def render_auth():
    st.markdown("### Your legal research, saved in one place")
    st.caption("Sign in to ask questions and return to your past conversations.")
    notice = st.session_state.pop("auth_notice", None)
    if notice:
        st.info(notice)
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login"):
            email = st.text_input("Email", key="login_email", max_chars=254)
            password = st.text_input("Password", type="password", key="login_password", max_chars=128)
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            try:
                result = api_request("POST", "/auth/login", {"email": email, "password": password}, False)
                clear_chat()
                st.session_state.access_token = result["access_token"]
                st.session_state.user = result["user"]
                st.session_state.clear_auth_fields = True
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
    with register_tab:
        with st.form("register"):
            email = st.text_input("Email", key="register_email", max_chars=254)
            password = st.text_input("Password", type="password", key="register_password", max_chars=128)
            confirm = st.text_input("Confirm password", type="password", key="register_confirm", max_chars=128)
            st.caption("Use at least 4 characters. Choose a password you don't use elsewhere.")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("Passwords don't match.")
            else:
                try:
                    api_request("POST", "/auth/register", {"email": email, "password": password}, False)
                    st.success("Account created. Sign in using the tab above.")
                except RuntimeError as exc:
                    st.error(str(exc))


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
            f"<span class='src-sec'>S. {escape(str(source.get('section_number', '?')))}</span>"
            f"<span class='src-title'>{escape(str(source.get('section_title', 'Untitled section')))}"
            f"<span class='src-act'>{escape(str(source.get('act', 'Unknown Act')))}</span>"
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
            except RuntimeError as exc:
                st.error(str(exc))
                return

    st.session_state.conversation_id = result["conversation_id"]
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


def clear_chat() -> None:
    """Clear only this session's conversation."""
    st.session_state.history = []
    st.session_state.pending = None
    st.session_state.conversation_id = None


def render_sidebar() -> None:
    """Navigate the authenticated user's saved conversations."""
    with st.sidebar:
        st.markdown("### ⚖️ Legal Assistant")
        st.caption("Explore Indian law, one question at a time.")
        st.button("＋ New chat", on_click=clear_chat, use_container_width=True)
        st.divider()
        st.caption(st.session_state.user["email"])
        if st.button("Sign out"):
            try:
                api_request("POST", "/auth/logout")
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                reset_account()
                st.rerun()
        st.markdown("**Saved conversations**")
        try:
            conversations = api_request("GET", "/conversations")
            if not conversations:
                st.caption("Your conversations will appear here.")
            for conversation in conversations:
                if st.button(conversation["title"], key="chat-" + conversation["id"], use_container_width=True):
                    loaded = api_request("GET", "/conversations/" + conversation["id"])
                    st.session_state.conversation_id = loaded["id"]
                    st.session_state.pending = None
                    st.session_state.history = [
                        {**turn, "followups": [] if turn.get("degraded") else build_followups(turn.get("sources", []))}
                        for turn in loaded["turns"]
                    ]
                    st.rerun()
            if st.session_state.get("conversation_id"):
                with st.expander("Delete this conversation"):
                    st.caption("This permanently removes the saved conversation.")
                    if st.button("Confirm deletion"):
                        api_request("DELETE", "/conversations/" + st.session_state.conversation_id)
                        clear_chat()
                        st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
        st.caption("Conversations are saved to your account.")
        st.divider()
        st.markdown("**Explore the collection**")
        st.caption("Constitution · Criminal law · Consumer protection · Income tax · Evidence · IT law · Motor vehicles")


def render_app() -> None:
    """Render the legal assistant Streamlit application."""
    st.set_page_config(page_title="Legal Assistant", page_icon="⚖️")
    st.markdown(STYLES, unsafe_allow_html=True)

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)
    if st.session_state.pop("clear_auth_fields", False):
        for key in ("login_email", "login_password", "register_email", "register_password", "register_confirm"):
            st.session_state.pop(key, None)

    st.markdown(
        "<header class='brand-header'><div class='brand-kicker'>Indian law · Source-backed answers</div>"
        "<h1>⚖️ Legal Assistant</h1>"
        "<p>Understand the law. Find the sections behind the answer.</p></header>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("access_token"):
        render_auth()
        return
    render_sidebar()

    if not st.session_state.history and not st.session_state.pending:
        st.markdown(
            "<section class='welcome'><h2>What would you like to understand?</h2>"
            "<p>Ask in everyday language and explore answers grounded in the available statutes, "
            "with Act and section references you can review.</p>"
            "<div class='coverage'><span>8 Acts in the collection</span><span>Act &amp; section citations</span>"
            "<span>Suggested follow-ups</span></div></section>",
            unsafe_allow_html=True,
        )
        st.markdown("<p class='sq-label'>Try a question</p>", unsafe_allow_html=True)
        render_question_chips(SAMPLE_QUESTIONS, key_prefix="sample", columns=2)

    render_history()

    typed = st.chat_input("Ask a legal question")
    if typed and typed.strip():
        ask(typed)

    if st.session_state.pending:
        process_pending()


if __name__ == "__main__":
    render_app()
