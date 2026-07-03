"""
Central configuration module for FixFirst AI.

Loads settings from environment variables (via python-dotenv) and exposes
a single `settings` object used across ingestion, labeling, training,
inference, scoring, API, and dashboard modules.

Usage:
    from fixfirst.config.settings import settings
    settings.database_url
    settings.mlflow_tracking_uri
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus
from dotenv import load_dotenv

from fixfirst.exceptions.exception import FixFirstException
# Load .env from project root regardless of caller's working directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise FixFirstException(
            f"Missing required environment variable: {key}. "
            f"Check that {ENV_PATH} exists and is populated (see .env.example).",
            sys,
        )
    return value


@dataclass(frozen=True)
class Settings:
    # --- Postgres ---
    postgres_user: str = field(default_factory=lambda: _get_env("POSTGRES_USER", "fixfirst"))
    postgres_password: str = field(default_factory=lambda: _get_env("POSTGRES_PASSWORD", "fixfirst"))
    postgres_db: str = field(default_factory=lambda: _get_env("POSTGRES_DB", "fixfirst"))
    db_host: str = field(default_factory=lambda: _get_env("DB_HOST", "localhost"))
    db_port: str = field(default_factory=lambda: _get_env("DB_PORT", "5432"))
    # --- Dashboard ---
    dashboard_api_base_url: str = field(
        default_factory=lambda: _get_env("DASHBOARD_API_BASE_URL", "http://localhost:8000")
    )
    # --- MLflow ---
    mlflow_tracking_uri: str = field(
        default_factory=lambda: _get_env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: _get_env("MLFLOW_EXPERIMENT_NAME", "fixfirst-absa")
    )

    # --- LLM fallback ---
    llm_provider: str = field(default_factory=lambda: _get_env("LLM_PROVIDER", "anthropic"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: _get_env("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: _get_env("OPENAI_API_KEY"))
    groq_api_key: Optional[str] = field(default_factory=lambda: _get_env("GROQ_API_KEY"))
    llm_model_name: str = field(default_factory=lambda: _get_env("LLM_MODEL_NAME", "claude-sonnet-4-6"))
    llm_fallback_threshold: float = field(
        default_factory=lambda: float(_get_env("LLM_FALLBACK_THRESHOLD", "0.65"))
    )

    # --- Zero-shot labeling ---
    zero_shot_model_name: str = field(
        default_factory=lambda: _get_env("ZERO_SHOT_MODEL_NAME", "facebook/bart-large-mnli")
    )
    zero_shot_category_threshold: float = field(
        default_factory=lambda: float(_get_env("ZERO_SHOT_CATEGORY_THRESHOLD", "0.30"))
    )
    zero_shot_category_fallback_threshold: float = field(
        default_factory=lambda: float(_get_env("ZERO_SHOT_CATEGORY_FALLBACK_THRESHOLD", "0.20"))
    )
    zero_shot_max_aspects_per_review: int = field(
        default_factory=lambda: int(_get_env("ZERO_SHOT_MAX_ASPECTS_PER_REVIEW", "3"))
    )
    zero_shot_batch_size: int = field(
        default_factory=lambda: int(_get_env("ZERO_SHOT_BATCH_SIZE", "8"))
    )

    # --- App / API ---
    app_env: str = field(default_factory=lambda: _get_env("APP_ENV", "development"))
    api_host: str = field(default_factory=lambda: _get_env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_get_env("API_PORT", "8000")))
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))

    # --- Paths ---
    data_raw_dir: str = field(default_factory=lambda: _get_env("DATA_RAW_DIR", "data/raw"))
    data_processed_dir: str = field(default_factory=lambda: _get_env("DATA_PROCESSED_DIR", "data/processed"))
    data_silver_labels_dir: str = field(
        default_factory=lambda: _get_env("DATA_SILVER_LABELS_DIR", "data/silver_labels")
    )
    data_gold_eval_dir: str = field(default_factory=lambda: _get_env("DATA_GOLD_EVAL_DIR", "data/gold_eval"))
    model_artifact_dir: str = field(
        default_factory=lambda: _get_env("MODEL_ARTIFACT_DIR", "artifacts/models")
    )

    @property
    def database_url(self) -> str:
        """SQLAlchemy-compatible Postgres connection string."""
        return (
                f"postgresql+psycopg2://"
                f"{quote_plus(self.postgres_user)}:"
                f"{quote_plus(self.postgres_password)}"
                f"@{self.db_host}:{self.db_port}/{self.postgres_db}"
            )

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a configured relative path against the project root."""
        return PROJECT_ROOT / relative_path

    def active_llm_api_key(self) -> str:
        """Return the API key for whichever LLM_PROVIDER is configured."""
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
        }
        key = key_map.get(self.llm_provider)
        if not key:
            raise FixFirstException(
                f"LLM_PROVIDER is set to '{self.llm_provider}' but no matching API key "
                f"was found in the environment. Set the corresponding *_API_KEY in .env.",
                sys,
            )
        return key


try:
    settings = Settings()
except FixFirstException:
    # Re-raise as-is; calling code (scripts/services) decides how to surface this.
    raise