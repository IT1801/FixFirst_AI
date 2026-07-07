"""
Eval harness: runs the fine-tuned aspect category and aspect sentiment
models against AWARE's gold ABSA annotations (held out in test.parquet)
and reports precision/recall/F1 — the numbers that go in the README
instead of "it works."

This is deliberately separate from the training-time validation metrics
(computed on a random val split of silver/LLM labels) — this harness
evaluates against AWARE's human-annotated gold labels, which is a
meaningfully different and stronger claim than "matches what our own
LLM labeler said."

Requires trained model artifacts at:
    {model_artifact_dir}/aspect_category/final/
    {model_artifact_dir}/aspect_sentiment/final/
(produced by scripts/train_aspect_category.py and train_aspect_sentiment.py)

Usage:
    PYTHONPATH=src python scripts/run_gold_eval.py
"""

import json
import sys
from typing import Dict

import numpy as np
import pandas as pd

from fixfirst.config.configuration import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._evaluation.gold_labels import extract_gold_category_labels, extract_gold_sentiment_pairs
from fixfirst.logging.logger import logging
from fixfirst.ml._training.aspect_category.metrics import compute_metrics_from_logits
from fixfirst.ml._training.aspect_sentiment.metrics import compute_sentiment_metrics
from fixfirst.ml._training.common import build_label_index


def _load_test_df() -> pd.DataFrame:
    test_path = settings.resolve_path(settings.data_processed_dir) / "test.parquet"
    if not test_path.exists():
        raise FixFirstException(
            f"{test_path} not found — run scripts/run_preprocessing.py first.", sys
        )
    return pd.read_parquet(test_path)


def _run_category_model_inference(texts, model_dir: str, max_length: int) -> np.ndarray:
    """Runs the fine-tuned category classifier over a list of texts, batched.
    Returns raw logits, shape (n_texts, n_labels)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    all_logits = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            outputs = model(**encoded)
            all_logits.append(outputs.logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def _run_sentiment_model_inference(text_a, text_b, model_dir: str, max_length: int) -> np.ndarray:
    """Runs the fine-tuned sentiment classifier over sentence pairs, batched.
    Returns raw logits, shape (n_pairs, 3)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    all_logits = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(text_a), batch_size):
            a_batch = text_a[start : start + batch_size]
            b_batch = text_b[start : start + batch_size]
            encoded = tokenizer(
                a_batch, b_batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            )
            outputs = model(**encoded)
            all_logits.append(outputs.logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def run_gold_evaluation() -> Dict[str, Dict]:
    """
    Runs both models against AWARE gold labels. Returns a dict:
        {"category": {...metrics}, "sentiment": {...metrics}}
    and writes the same to data/gold_eval/eval_report.json.
    """
    from fixfirst.ml._labeling.taxonomy import load_active_taxonomy

    try:
        taxonomy = load_active_taxonomy()
        feature_keys = [t["feature_key"] for t in taxonomy]
        feature_display_names = {t["feature_key"]: t["display_name"] for t in taxonomy}
        sorted_feature_keys = [
            k for k, _ in sorted(build_label_index(feature_keys).items(), key=lambda kv: kv[1])
        ]

        test_df = _load_test_df()

        results: Dict[str, Dict] = {}

        # --- Category model eval ---
        category_model_dir = str(settings.resolve_path(settings.model_artifact_dir) / "aspect_category" / "final")
        with open(f"{category_model_dir}/aspect_category_meta.json") as f:
            category_meta = json.load(f)

        gold_cat_df = extract_gold_category_labels(test_df, feature_keys)
        cat_logits = _run_category_model_inference(
            gold_cat_df["review_text"].tolist(), category_model_dir, category_meta["max_length"]
        )
        gold_cat_labels = np.stack(gold_cat_df["gold_labels"].values)
        category_metrics = compute_metrics_from_logits(
            cat_logits, gold_cat_labels, sorted_feature_keys, threshold=category_meta.get("threshold", 0.5)
        )
        results["category"] = category_metrics
        logging.info(f"run_gold_evaluation: category model — f1_micro={category_metrics['f1_micro']:.3f}")

        # --- Sentiment model eval ---
        sentiment_model_dir = str(settings.resolve_path(settings.model_artifact_dir) / "aspect_sentiment" / "final")
        with open(f"{sentiment_model_dir}/aspect_sentiment_meta.json") as f:
            sentiment_meta = json.load(f)

        gold_sent_df = extract_gold_sentiment_pairs(test_df, feature_display_names)
        sentiment_label_index = {label: i for i, label in enumerate(sentiment_meta["sentiment_labels"])}
        gold_sentiment_indices = np.array(
            [sentiment_label_index[s] for s in gold_sent_df["gold_sentiment"]]
        )
        sent_logits = _run_sentiment_model_inference(
            gold_sent_df["text_a"].tolist(),
            gold_sent_df["text_b"].tolist(),
            sentiment_model_dir,
            sentiment_meta["max_length"],
        )
        sentiment_metrics = compute_sentiment_metrics(sent_logits, gold_sentiment_indices)
        results["sentiment"] = sentiment_metrics
        logging.info(f"run_gold_evaluation: sentiment model — accuracy={sentiment_metrics['accuracy']:.3f}")

        out_dir = settings.resolve_path(settings.data_gold_eval_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "eval_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"run_gold_evaluation: wrote eval report to {report_path}")

        return results
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)