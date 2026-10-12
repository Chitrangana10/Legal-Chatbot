"""Load backend settings, API keys, and filesystem paths from environment variables."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Intelligent Legal RAG System", validation_alias="APP_NAME")
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    database_url: str = Field(default="postgresql+psycopg://legal:local-development-only@127.0.0.1:5433/legal_chatbot", validation_alias="DATABASE_URL")
    session_hours: int = Field(default=24, ge=1, le=168, validation_alias="SESSION_HOURS")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", validation_alias="REDIS_URL")
    enable_answer_cache: bool = Field(default=True, validation_alias="ENABLE_ANSWER_CACHE")
    answer_cache_ttl_seconds: int = Field(default=86400, ge=1, validation_alias="ANSWER_CACHE_TTL_SECONDS")
    answer_cache_version: str = Field(default="v1", validation_alias="ANSWER_CACHE_VERSION")

    together_api_key: Optional[str] = Field(default=None, validation_alias="TOGETHER_API_KEY")
    google_api_key: Optional[str] = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    reranker_model_name: str = Field(default="", validation_alias="RERANKER_MODEL_NAME")
    enable_reranker: bool = Field(default=True, validation_alias="ENABLE_RERANKER")
    default_llm_provider: str = Field(default="gemini", validation_alias="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field(default="gemini-3.5-flash", validation_alias="DEFAULT_LLM_MODEL")

    project_root: Path = Field(default=Path("."), validation_alias="PROJECT_ROOT")
    raw_data_dir: Path = Field(default=Path("backend/data/raw"), validation_alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(
        default=Path("backend/data/processed"),
        validation_alias="PROCESSED_DATA_DIR",
    )
    faiss_index_dir: Path = Field(default=Path("backend/data/faiss_index"), validation_alias="FAISS_INDEX_DIR")


def get_settings() -> Settings:
    """Return validated application settings."""
    return Settings()
