"""Exercise account screens and history navigation with a fake API."""

from unittest.mock import Mock
from pathlib import Path

import requests
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py")


def response(payload=None, status=200):
    result = Mock()
    result.status_code = status
    result.ok = status < 400
    result.content = b"json" if payload is not None else b""
    result.json.return_value = payload
    return result


def test_login_load_history_new_chat_and_logout(monkeypatch):
    calls = []

    def api(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/auth/login"):
            return response({"access_token": "test-token", "user": {"email": "alice@example.com"}})
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"
        if url.endswith("/auth/logout"):
            return response(status=204)
        if url.endswith("/conversations"):
            return response([{"id": "saved-chat", "title": "Saved question"}])
        return response({"id": "saved-chat", "turns": [
            {"question": "Saved question", "answer": "Saved answer", "sources": []}
        ]})

    monkeypatch.setattr(requests, "request", api)
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert len(app.chat_input) == 0
    app.text_input(key="login_email").set_value("alice@example.com")
    app.text_input(key="login_password").set_value("a-long-test-password")
    next(button for button in app.button if button.label == "Sign in").click().run()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert "login_password" not in app.session_state
    app.button(key="chat-saved-chat").click().run()
    assert not app.exception
    assert app.session_state.history[0]["answer"] == "Saved answer"
    next(button for button in app.button if "New chat" in button.label).click().run()
    assert app.session_state.history == []
    assert app.session_state.conversation_id is None
    next(button for button in app.button if button.label == "Sign out").click().run()
    assert not app.exception
    assert len(app.chat_input) == 0
    assert app.session_state.history == []
    assert any(url.endswith("/auth/logout") for _, url, _ in calls)


def test_expired_session_clears_private_history(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: response({"detail": "Expired"}, 401))
    app = AppTest.from_file(APP_PATH)
    app.session_state.access_token = "expired"
    app.session_state.user = {"email": "alice@example.com"}
    app.session_state.history = [{"question": "Private", "answer": "Private answer", "sources": []}]
    app.run()
    assert not app.exception
    assert app.session_state.history == []
    assert len(app.chat_input) == 0
    assert any("session expired" in notice.value for notice in app.info)


def test_registration_mismatch_does_not_call_api(monkeypatch):
    api = Mock()
    monkeypatch.setattr(requests, "request", api)
    app = AppTest.from_file(APP_PATH).run()
    app.text_input(key="register_email").set_value("alice@example.com")
    app.text_input(key="register_password").set_value("a-long-test-password")
    app.text_input(key="register_confirm").set_value("different-password")
    next(button for button in app.button if button.label == "Create account").click().run()
    assert not app.exception
    assert app.error[0].value == "Passwords don't match."
    api.assert_not_called()
