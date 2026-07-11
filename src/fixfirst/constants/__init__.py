"""Project-wide constants for FixFirst AI.

These values are intrinsic to the application and should not vary across
deployments. Environment-specific values belong in configuration.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE_PATH = PROJECT_ROOT / ".env"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SILVER_LABELS_DIR = DATA_DIR / "silver_labels"
GOLD_EVAL_DIR = DATA_DIR / "gold_eval"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

RAW_DATA_FILE = RAW_DATA_DIR / "aware_reviews.csv"
SILVER_LABELS_FILE = SILVER_LABELS_DIR / "silver_labels.parquet"
GOLD_EVAL_FILE = GOLD_EVAL_DIR / "eval_report.json"

SCHEMA_NAME = "fixfirst"

DEFAULT_HALF_LIFE_DAYS = 90
DEFAULT_API_LIMIT = 200
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
STRATIFY_COLUMN = "category"  # column in raw_metadata to stratify by when splitting dataset


SOURCE_AWARE = "aware"
RAW_METADATA_AWARE = "aware_annotations"

AWARE_DOMAIN_COLUMN = "domain"
AWARE_APP_COLUMN = "app"
AWARE_SENTENCE_COLUMN = "sentence"
AWARE_CATEGORY_COLUMN = "category"
AWARE_TERM_COLUMN = "term"
AWARE_SENTIMENT_COLUMN = "sentiment"
AWARE_FROM_COLUMN = "from"
AWARE_TO_COLUMN = "to"

ASPECT_HYPOTHESIS_TEMPLATE = "This review discusses {}."
SENTIMENT_HYPOTHESIS_TEMPLATE = "The sentiment toward this feature is {}."
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
MAX_RETRIES = 1
DEFAULT_BATCH_SIZE = 8
RATE_LIMIT_SECONDS = 0.0
LABELS_FILENAME = "silver_labels.parquet"
FAILURES_FILENAME = "labeling_failures.parquet"
PROGRESS_FILENAME = "silver_labeling_progress.parquet"
LABEL_COLUMNS = ["review_id", "review_text", "feature_key", "sentiment"]
FAILURE_COLUMNS = ["review_id", "review_text"]
PROGRESS_COLUMNS = ["review_id", "review_text", "status"]
AWARE_SENTIMENT_NEGATIVE = "negative"
AWARE_SENTIMENT_NEUTRAL = "neutral"
AWARE_SENTIMENT_POSITIVE = "positive"

# Filenames
TRAIN_FILENAME = "train.parquet"
VAL_FILENAME = "val.parquet"
TEST_FILENAME = "test.parquet"
EXTRACTED_LABELS_FILENAME = "extracted_labels.parquet"
EXTRACTED_PROGRESS_FILENAME = "extracted_labeling_progress.parquet"

# Status & Source
STATUS_LABELED = "labeled"
STATUS_FAILED = "failed"
SOURCE_FINETUNED = "finetuned"
SOURCE_LLM_FALLBACK = "llm_fallback"

__all__ = [
    "AWARE_APP_COLUMN",
    "AWARE_CATEGORY_COLUMN",
    "AWARE_DOMAIN_COLUMN",
    "AWARE_FROM_COLUMN",
    "AWARE_SENTENCE_COLUMN",
    "AWARE_SENTIMENT_COLUMN",
    "AWARE_SENTIMENT_NEGATIVE",
    "AWARE_SENTIMENT_NEUTRAL",
    "AWARE_SENTIMENT_POSITIVE",
    "AWARE_TERM_COLUMN",
    "AWARE_TO_COLUMN",
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "DEFAULT_API_LIMIT",
    "DEFAULT_GOLD_EVAL_FILE",
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_RAW_DATA_FILE",
    "DEFAULT_SILVER_LABELS_FILE",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "ENV_FILE_PATH",
    "GOLD_EVAL_DIR",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "PROCESSED_DATA_DIR",
    "RAW_DATA_DIR",
    "RAW_DATA_FILE",
    "RAW_METADATA_AWARE",
    "SILVER_LABELS_DIR",
    "SOURCE_AWARE",
    "TRAIN_FILENAME",
    "VAL_FILENAME",
    "TEST_FILENAME",
    "EXTRACTED_LABELS_FILENAME",
    "EXTRACTED_PROGRESS_FILENAME",
    "STATUS_LABELED",
    "STATUS_FAILED",
    "SOURCE_FINETUNED",
    "SOURCE_LLM_FALLBACK",
]
