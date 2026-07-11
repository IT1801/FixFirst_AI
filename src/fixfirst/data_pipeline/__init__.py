"""Data ingestion and preprocessing."""

from fixfirst.data_pipeline.ingestion import ingest_aware_csv
from fixfirst.data_pipeline._preprocessing.pipeline import run_preprocessing_pipeline

__all__ = ["ingest_aware_csv", "run_preprocessing_pipeline"]
