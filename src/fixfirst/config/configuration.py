"""Central configuration for FixFirst AI.

All environment-specific values are read from the local ``.env`` file or
the process environment. Callers should obtain typed config objects from
``ConfigurationManager`` instead of reading environment variables directly.
"""

from dataclasses import dataclass, field
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv

from fixfirst.constants import ENV_FILE_PATH, PROJECT_ROOT
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

load_dotenv(dotenv_path=ENV_FILE_PATH)


def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Return an environment value or raise a typed configuration error."""
    value = os.getenv(key, default)
    if required and not value:
        raise FixFirstException(
            f"Missing required environment variable: {key}. Check {ENV_FILE_PATH}.",
            sys,
        )
    return value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str

    schema: str

    pool_size: int
    max_overflow: int
    pool_pre_ping: bool = True

    url: Optional[str] = None

    @property
    def connection_url(self) -> str:
        if self.url:
            return self.url

        return (
            f"postgresql+psycopg2://"
            f"{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class PathsConfig:
    """Project paths that may vary by environment."""

    raw_data_dir: Path
    processed_dir: Path
    extracted_labels_dir: Path
    gold_eval_dir: Path
    model_artifact_dir: Path


@dataclass(frozen=True)
class AWAREIngestionConfig:
    """Column mapping and file settings for the AWARE ingestion pipeline."""

    csv_path: Path
    domain_column: str
    app_column: str
    sentence_column: str
    rating_column: str
    category_column: str
    term_column: str
    sentiment_column: str
    from_column: Optional[str] = None
    to_column: Optional[str] = None
    batch_size: int = 500

@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    val_size: float
    random_state: int
    stratify_column: str

@dataclass(frozen=True)
class MLTrainingConfig:
    """Typed ML training settings."""
    base_model_name: str
    max_length: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    prediction_threshold: float

@dataclass(frozen=True)
class MLflowConfig:
    """Typed MLflow connection settings."""

    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class APIConfig:
    """Typed API server settings."""

    host: str
    port: int
    reload: bool


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared across scripts and services."""

    postgres_user: str = field(default_factory=lambda: _get_env("POSTGRES_USER", "fixfirst"))
    postgres_password: str = field(default_factory=lambda: _get_env("POSTGRES_PASSWORD", "fixfirst"))
    postgres_db: str = field(default_factory=lambda: _get_env("POSTGRES_DB", "fixfirst"))
    db_host: str = field(default_factory=lambda: _get_env("DB_HOST", "localhost"))
    db_port: str = field(default_factory=lambda: _get_env("DB_PORT", "5432"))
    db_pool_size: int = field(default_factory=lambda: int(_get_env("DB_POOL_SIZE", "5")))
    db_max_overflow: int = field(default_factory=lambda: int(_get_env("DB_MAX_OVERFLOW", "10")))
    dashboard_api_base_url: str = field(
        default_factory=lambda: _get_env("DASHBOARD_API_BASE_URL", "http://localhost:8000")
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: _get_env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: _get_env("MLFLOW_EXPERIMENT_NAME", "fixfirst-absa")
    )
    
    anthropic_api_key: Optional[str] = field(default_factory=lambda: _get_env("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: _get_env("OPENAI_API_KEY"))
    groq_api_key: Optional[str] = field(default_factory=lambda: _get_env("GROQ_API_KEY"))
    llm_model_name: str = field(default_factory=lambda: _get_env("LLM_MODEL_NAME", "claude-sonnet-4-6"))
    llm_fallback_threshold: float = field(default_factory=lambda: float(_get_env("LLM_FALLBACK_THRESHOLD", "0.65")))
    llm_sentiment_fallback_threshold: float = field(default_factory=lambda: float(_get_env("LLM_SENTIMENT_FALLBACK_THRESHOLD", "0.50")))
    llm_max_requests_per_minute: int = field(
        default_factory=lambda: int(_get_env("LLM_MAX_REQUESTS_PER_MINUTE", "30"))
    )
    ml_base_model_name: str = field(
        default_factory=lambda: _get_env("ML_BASE_MODEL_NAME", "microsoft/deberta-v3-base")
    )
    ml_max_length: int = field(default_factory=lambda: int(_get_env("ML_MAX_LENGTH", "128")))
    ml_batch_size: int = field(default_factory=lambda: int(_get_env("ML_BATCH_SIZE", "16")))
    ml_num_epochs: int = field(default_factory=lambda: int(_get_env("ML_NUM_EPOCHS", "4")))
    ml_learning_rate: float = field(default_factory=lambda: float(_get_env("ML_LEARNING_RATE", "2e-5")))
    ml_prediction_threshold: float = field(default_factory=lambda: float(_get_env("ML_PREDICTION_THRESHOLD", "0.5")))

    app_env: str = field(default_factory=lambda: _get_env("APP_ENV", "development"))
    api_host: str = field(default_factory=lambda: _get_env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_get_env("API_PORT", "8000")))
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))
    data_raw_dir: str = field(default_factory=lambda: _get_env("DATA_RAW_DIR", "data/raw"))
    data_processed_dir: str = field(default_factory=lambda: _get_env("DATA_PROCESSED_DIR", "data/processed"))
    data_extracted_labels_dir: str = field(
        default_factory=lambda: _get_env("DATA_EXTRACTED_LABELS_DIR", "data/extracted_labels")
    )
    data_gold_eval_dir: str = field(default_factory=lambda: _get_env("DATA_GOLD_EVAL_DIR", "data/gold_eval"))
    model_artifact_dir: str = field(default_factory=lambda: _get_env("MODEL_ARTIFACT_DIR", "artifacts/models"))

    @property
    def database_url(self) -> str:
        """Return the PostgreSQL connection string."""
        database_url = os.getenv("DB_URL")
        if database_url:
            return database_url

        return (
            f"postgresql+psycopg2://"
            f"{quote_plus(self.postgres_user)}:{quote_plus(self.postgres_password)}"
            f"@{self.db_host}:{self.db_port}/{self.postgres_db}"
        )

    def resolve_path(self, relative_path: str | Path) -> Path:
        """Resolve a path relative to the project root.

        Parameters
        ----------
        relative_path : str | Path
            Relative path to resolve against the repository root.

        Returns
        -------
        Path
            Absolute path rooted at the repository checkout.
        """
        return PROJECT_ROOT / Path(relative_path)

    def active_llm_api_key(self) -> str:
        """Return the API key matching the configured LLM provider."""
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
        }
        key = key_map.get(self.llm_provider)
        if not key:
            raise FixFirstException(
                f"LLM_PROVIDER is set to '{self.llm_provider}' but no matching API key was found.",
                sys,
            )
        return key


class ConfigurationManager:
    """Single entry point for typed runtime configuration."""

    def __init__(self) -> None:
        try:
            logging.info(f"Loading configuration from {ENV_FILE_PATH}")
            self._settings = Settings()
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def get_settings(self) -> Settings:
        """Return the shared runtime settings object."""
        try:
            return self._settings
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def get_database_config(self) -> DatabaseConfig:
        """Return the typed database configuration."""
        try:
            settings = self._settings
            return DatabaseConfig(
                host=settings.db_host,
                port=int(settings.db_port),
                name=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                schema=_get_env("DB_SCHEMA", "fixfirst"),
                pool_size=settings.db_pool_size,            # <-- Pass the pool size
                max_overflow=settings.db_max_overflow,      # <-- Pass the max overflow
                pool_pre_ping=_parse_bool(_get_env("DB_POOL_PRE_PING", "true")), # <-- Pass the ping policy
                url=os.getenv("DB_URL"),
            )
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc
    

    def get_paths_config(self) -> PathsConfig:
        """Return the typed path configuration."""
        try:
            settings = self._settings
            return PathsConfig(
                raw_data_dir=PROJECT_ROOT / settings.data_raw_dir,
                processed_dir=PROJECT_ROOT / settings.data_processed_dir,
                extracted_labels_dir=PROJECT_ROOT / settings.data_extracted_labels_dir,
                gold_eval_dir=PROJECT_ROOT / settings.data_gold_eval_dir,
                model_artifact_dir=PROJECT_ROOT / settings.model_artifact_dir,
            )
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def get_aware_ingestion_config(self) -> AWAREIngestionConfig:
        """Return the typed AWARE ingestion configuration."""
        try:
            return AWAREIngestionConfig(
                csv_path=PROJECT_ROOT / _get_env("AWARE_CSV_PATH", "data/raw/aware_reviews.csv"),
                domain_column=_get_env("AWARE_DOMAIN_COLUMN", "domain"),
                app_column=_get_env("AWARE_APP_COLUMN", "app"),
                sentence_column=_get_env("AWARE_SENTENCE_COLUMN", "sentence"),
                rating_column=_get_env("AWARE_RATING_COLUMN", "rating"),
                category_column=_get_env("AWARE_CATEGORY_COLUMN", "category"),
                term_column=_get_env("AWARE_TERM_COLUMN", "term"),
                sentiment_column=_get_env("AWARE_SENTIMENT_COLUMN", "sentiment"),
                from_column=os.getenv("AWARE_FROM_COLUMN"),
                to_column=os.getenv("AWARE_TO_COLUMN"),
                batch_size=int(_get_env("AWARE_BATCH_SIZE", "500")),
            )
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc
    
    def get_split_config(self) -> SplitConfig:
        """Return the typed dataset split configuration."""
        try:
            settings = self._settings
            return SplitConfig(
                test_size=float(_get_env("TEST_SIZE", "0.1")),
                val_size=float(_get_env("VAL_SIZE", "0.1")),
                random_state=int(_get_env("RANDOM_STATE", "42")),
                stratify_column=_get_env("STRATIFY_COLUMN", "raw_metadata"),
            )
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def get_ml_training_config(self) -> MLTrainingConfig:
        """Return the typed ML training configuration."""
        try:
            settings = self._settings
            return MLTrainingConfig(
                base_model_name=settings.ml_base_model_name,
                max_length=settings.ml_max_length,
                batch_size=settings.ml_batch_size,
                num_epochs=settings.ml_num_epochs,
                learning_rate=settings.ml_learning_rate,
                prediction_threshold=settings.ml_prediction_threshold,
            )
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def get_mlflow_config(self) -> MLflowConfig:
        """Return the typed MLflow configuration."""
        try:
            settings = self._settings
            return MLflowConfig(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow_experiment_name,
            )
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc

    def get_api_config(self) -> APIConfig:
        """Return the typed API configuration."""
        try:
            settings = self._settings
            return APIConfig(
                host=settings.api_host,
                port=settings.api_port,
                reload=_parse_bool(_get_env("API_RELOAD", "false")),
            )
        except FixFirstException:
            raise
        except Exception as exc:
            raise FixFirstException(exc, sys) from exc


settings = ConfigurationManager().get_settings()