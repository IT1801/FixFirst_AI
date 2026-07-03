"""
Prefect flow tying the full FixFirst AI pipeline together:
    (optional) ingest -> seed features -> preprocess -> hybrid inference -> score

Each stage wraps an already-built, independently-tested pipeline function
(preprocessing.pipeline, inference.pipeline, scoring.pipeline) — this flow
is orchestration only, no new business logic. Tasks retry transient
failures (e.g. a flaky DB connection) but do not retry on data-validation
errors (FixFirstException from a genuinely bad input), since retrying
those would just fail identically three times.

Usage:
    PYTHONPATH=src python scripts/run_pipeline_flow.py [options]

Deploying this flow on a schedule (e.g. daily) is a natural next step
once there's a live review ingestion source beyond the one-time AWARE
bootstrap — see Prefect's `serve()`/deployment docs for that step.
"""

import sys
from typing import Optional

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from fixfirst.exceptions.exception import FixFirstException

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


@task(name="ingest_aware", retries=2, retry_delay_seconds=10)
def task_ingest_aware(csv_path: str) -> int:
    from fixfirst.ingestion.aware_loader import ingest_aware_csv

    logger = get_run_logger()
    try:
        count = ingest_aware_csv(csv_path)
        logger.info(f"ingest_aware: inserted {count} rows")
        return count
    except FixFirstException:
        raise  # data/validation error — don't retry, surface immediately
    except RETRYABLE_EXCEPTIONS:
        raise  # let Prefect's retry policy handle transient infra errors


@task(name="seed_features", retries=2, retry_delay_seconds=10)
def task_seed_features() -> int:
    from scripts.seed_features import seed_features_master

    logger = get_run_logger()
    count = seed_features_master()
    logger.info(f"seed_features: {count} new features inserted")
    return count


@task(name="run_preprocessing", retries=1)
def task_run_preprocessing() -> dict:
    from fixfirst.preprocessing.pipeline import run_preprocessing_pipeline

    logger = get_run_logger()
    result = run_preprocessing_pipeline(write_output=True)
    logger.info(f"run_preprocessing: train={len(result['train'])} val={len(result['val'])} test={len(result['test'])}")
    return {k: len(v) for k, v in result.items()}


@task(name="run_hybrid_inference", retries=1)
def task_run_hybrid_inference(split: str, limit: Optional[int]) -> dict:
    import pandas as pd

    from fixfirst.config.settings import settings
    from fixfirst.inference.pipeline import run_batch_hybrid_inference

    logger = get_run_logger()
    split_path = settings.resolve_path(settings.data_processed_dir) / f"{split}.parquet"
    if not split_path.exists():
        raise FixFirstException(f"{split_path} not found — run_preprocessing must run first.", sys)

    reviews_df = pd.read_parquet(split_path)
    if limit:
        reviews_df = reviews_df.head(limit)

    result = run_batch_hybrid_inference(reviews_df, write_to_db=True)
    logger.info(
        f"run_hybrid_inference: {result['n_aspects_written']} aspects written, "
        f"llm_fallback_rate={result['fallback_stats']['llm_fallback_rate']:.1%}"
    )
    return result


@task(name="run_scoring", retries=1)
def task_run_scoring(half_life_days: int) -> int:
    from fixfirst.scoring.pipeline import run_scoring_pipeline

    logger = get_run_logger()
    n_written = run_scoring_pipeline(half_life_days=half_life_days)
    logger.info(f"run_scoring: {n_written} criticality_scores rows written")
    return n_written


@flow(name="fixfirst-ai-pipeline", task_runner=SequentialTaskRunner())
def fixfirst_pipeline_flow(
    run_ingestion: bool = False,
    aware_csv_path: Optional[str] = None,
    inference_split: str = "test",
    inference_limit: Optional[int] = None,
    half_life_days: int = 90,
) -> dict:
    """
    Full pipeline flow. run_ingestion=False by default since AWARE
    ingestion is a one-time bootstrap, not something you'd want re-running
    (and re-inserting duplicate rows) on every scheduled pipeline run.
    """
    logger = get_run_logger()
    summary = {}

    if run_ingestion:
        if not aware_csv_path:
            raise FixFirstException("run_ingestion=True requires aware_csv_path to be set.", sys)
        summary["ingested_rows"] = task_ingest_aware(aware_csv_path)

    summary["features_seeded"] = task_seed_features()
    summary["preprocessing"] = task_run_preprocessing()
    summary["inference"] = task_run_hybrid_inference(inference_split, inference_limit)
    summary["scoring_rows_written"] = task_run_scoring(half_life_days)

    logger.info(f"fixfirst_pipeline_flow: complete. Summary: {summary}")
    return summary