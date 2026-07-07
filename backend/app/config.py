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
    reranker_model_name: Optional[str] = Field(default=None, validation_alias="RERANKER_MODEL_NAME")
    default_llm_provider: str = Field(default="gemini", validation_alias="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field(default="gemini-3.5-flash", validation_alias="DEFAULT_LLM_MODEL")

    project_root: Path = Field(default=Path("."), validation_alias="PROJECT_ROOT")
    raw_data_dir: Path = Field(default=Path("backend/data/raw"), validation_alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(
        default=Path("backend/data/processed"),
        validation_alias="PROCESSED_DATA_DIR",
    )
    faiss_index_dir: Path = Field(default=Path("backend/data/faiss_index"), validation_alias="FAISS_INDEX_DIR")
    sample_ipc_path: Path = Field(default=Path("backend/data/raw/sample_ipc.json"), validation_alias="SAMPLE_IPC_PATH")
    citation_store_path: Path = Field(
        default=Path("backend/data/processed/citations.jsonl"),
        validation_alias="CITATION_STORE_PATH",
    )


def get_settings() -> Settings:
    """Return validated application settings."""
    return Settings()
