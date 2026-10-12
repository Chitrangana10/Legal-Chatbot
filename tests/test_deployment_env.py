import pytest

from backend.scripts.prepare_deployment_env import prepare


def test_initializes_secret_once_and_preserves_other_settings(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=example\nPOSTGRES_PASSWORD=\n")
    assert prepare(path)
    first = path.read_text()
    assert "GEMINI_API_KEY=example" in first
    password = first.split("POSTGRES_PASSWORD=")[1].strip()
    assert len(password) == 64
    assert not prepare(path)
    assert path.read_text() == first


def test_missing_env_or_duplicates_are_rejected(tmp_path):
    path = tmp_path / ".env"
    with pytest.raises(RuntimeError):
        prepare(path)
    path.write_text("POSTGRES_PASSWORD=one\nPOSTGRES_PASSWORD=two\n")
    with pytest.raises(RuntimeError):
        prepare(path)
