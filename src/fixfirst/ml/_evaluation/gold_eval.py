"""Eval harness: runs the fine-tuned models against AWARE's gold annotations."""

import json
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from fixfirst.config.configuration import ConfigurationManager
from fixfirst.constants import TEST_FILENAME
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._evaluation.gold_labels import extract_gold_category_labels, extract_gold_sentiment_pairs
from fixfirst.logging.logger import logging
from fixfirst.ml._training.aspect_category.metrics import compute_metrics_from_logits
from fixfirst.ml._training.aspect_sentiment.metrics import compute_sentiment_metrics
from fixfirst.ml._training.common import build_label_index


class GoldEvaluator:
    """Object-oriented evaluator for the gold dataset."""

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.settings = self.config_manager.get_settings()

    def _load_test_df(self) -> pd.DataFrame:
        test_path = self.settings.resolve_path(self.settings.data_processed_dir) / TEST_FILENAME
        if not test_path.exists():
            raise FixFirstException(f"{test_path} not found — run scripts/run_preprocessing.py first.", sys)
        return pd.read_parquet(test_path)

    def _run_category_model_inference(self, texts: List[str], model_dir: str, max_length: int, num_labels: int) -> np.ndarray:
        """Runs the fine-tuned category classifier over a list of texts, batched."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
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

    def _run_sentiment_model_inference(self, text_a: List[str], text_b: List[str], model_dir: str, max_length: int, num_labels: int) -> np.ndarray:
        """Runs the fine-tuned sentiment classifier over sentence pairs, batched."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
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

    def evaluate(self) -> Dict[str, Dict]:
        """Runs both models against AWARE gold labels."""
        from fixfirst.core._db.base import get_db
        from fixfirst.core._db.models import FeatureMaster

        try:
            with get_db() as db:
                taxonomy = db.query(FeatureMaster).filter(FeatureMaster.is_active.is_(True)).all()
                feature_keys = [t.feature_key for t in taxonomy]
                feature_display_names = {t.feature_key: t.display_name for t in taxonomy}
            sorted_feature_keys = [
                k for k, _ in sorted(build_label_index(feature_keys).items(), key=lambda kv: kv[1])
            ]

            test_df = self._load_test_df()
            results: Dict[str, Dict] = {}

            # --- Category model eval ---
            category_model_dir = str(self.settings.resolve_path(self.settings.model_artifact_dir) / "aspect_category" / "final")
            with open(f"{category_model_dir}/aspect_category_meta.json") as f:
                category_meta = json.load(f)

            model_feature_keys = [k for k, _ in sorted(category_meta["label_index"].items(), key=lambda x: x[1])]

            gold_cat_df = extract_gold_category_labels(test_df, model_feature_keys)
            cat_logits = self._run_category_model_inference(
                gold_cat_df["review_text"].tolist(), category_model_dir, category_meta["max_length"], len(category_meta["label_index"])
            )
            gold_cat_labels = np.stack(gold_cat_df["gold_labels"].values)
            category_metrics = compute_metrics_from_logits(
                cat_logits, gold_cat_labels, model_feature_keys, threshold=category_meta.get("threshold", 0.5)
            )
            results["category"] = category_metrics
            logging.info(f"GoldEvaluator: category model — f1_micro={category_metrics['f1_micro']:.3f}")

            # --- Sentiment model eval ---
            sentiment_model_dir = str(self.settings.resolve_path(self.settings.model_artifact_dir) / "aspect_sentiment" / "final")
            with open(f"{sentiment_model_dir}/aspect_sentiment_meta.json") as f:
                sentiment_meta = json.load(f)

            gold_sent_df = extract_gold_sentiment_pairs(test_df, feature_display_names)
            sentiment_label_index = {label: i for i, label in enumerate(sentiment_meta["sentiment_labels"])}
            gold_sentiment_indices = np.array(
                [sentiment_label_index[s] for s in gold_sent_df["gold_sentiment"]]
            )
            sent_logits = self._run_sentiment_model_inference(
                gold_sent_df["text_a"].tolist(),
                gold_sent_df["text_b"].tolist(),
                sentiment_model_dir,
                sentiment_meta["max_length"],
                len(sentiment_meta["sentiment_labels"])
            )
            sentiment_metrics = compute_sentiment_metrics(sent_logits, gold_sentiment_indices)
            results["sentiment"] = sentiment_metrics
            logging.info(f"GoldEvaluator: sentiment model — accuracy={sentiment_metrics['accuracy']:.3f}")

            out_dir = self.settings.resolve_path(self.settings.data_gold_eval_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "eval_report.json"
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            logging.info(f"GoldEvaluator: wrote eval report to {report_path}")

            return results
        except FixFirstException:
            raise
        except Exception as e:
            raise FixFirstException(e, sys) from e


def run_gold_evaluation() -> Dict[str, Dict]:
    """Backward compatibility wrapper."""
    return GoldEvaluator().evaluate()