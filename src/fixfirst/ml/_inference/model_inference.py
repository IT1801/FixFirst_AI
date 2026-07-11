"""
Fine-tuned model inference wrappers for the hybrid ABSA pipeline.

Loads the trained category and sentiment classifiers once (models are
cached at module level to avoid reloading on every call) and exposes
simple predict functions returning probabilities, not raw logits — the
router works entirely in probability space via confidence.py.

torch/transformers imports are deferred into each function so this module
can be imported (and confidence.py tested) without those heavy
dependencies installed.
"""

import json
import sys
from typing import Dict, List, Tuple

import numpy as np

from fixfirst.config.configuration import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._inference.confidence import sigmoid, softmax
from fixfirst.logging.logger import logging

_category_model_cache = {}
_sentiment_model_cache = {}


def _load_category_model():
    if "model" in _category_model_cache:
        return _category_model_cache["model"], _category_model_cache["tokenizer"], _category_model_cache["meta"]

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    try:
        model_dir = str(settings.resolve_path(settings.model_artifact_dir) / "aspect_category" / "final")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        with open(f"{model_dir}/aspect_category_meta.json") as f:
            meta = json.load(f)

        _category_model_cache.update({"model": model, "tokenizer": tokenizer, "meta": meta})
        logging.info(f"_load_category_model: loaded model from {model_dir}")
        return model, tokenizer, meta
    except Exception as e:
        raise FixFirstException(e, sys)


def _load_sentiment_model():
    if "model" in _sentiment_model_cache:
        return _sentiment_model_cache["model"], _sentiment_model_cache["tokenizer"], _sentiment_model_cache["meta"]

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    try:
        model_dir = str(settings.resolve_path(settings.model_artifact_dir) / "aspect_sentiment" / "final")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        with open(f"{model_dir}/aspect_sentiment_meta.json") as f:
            meta = json.load(f)

        _sentiment_model_cache.update({"model": model, "tokenizer": tokenizer, "meta": meta})
        logging.info(f"_load_sentiment_model: loaded model from {model_dir}")
        return model, tokenizer, meta
    except Exception as e:
        raise FixFirstException(e, sys)


def predict_category_probs(review_text: str) -> Tuple[np.ndarray, List[str], float]:
    """
    Returns (probs, label_names, threshold): probs is shape (n_labels,) sigmoid
    outputs in [0, 1], label_names[i] corresponds to probs[i], threshold is the
    dynamically tuned optimal threshold from training (defaults to 0.5).
    """
    import torch

    try:
        model, tokenizer, meta = _load_category_model()
        label_index: Dict[str, int] = meta["label_index"]
        label_names = [k for k, _ in sorted(label_index.items(), key=lambda kv: kv[1])]
        threshold = meta.get("threshold", 0.5)

        with torch.no_grad():
            encoded = tokenizer(
                review_text, truncation=True, padding=True, max_length=meta["max_length"], return_tensors="pt"
            )
            logits = model(**encoded).logits.cpu().numpy()[0]

        probs = sigmoid(logits)
        return probs, label_names, threshold
    except Exception as e:
        raise FixFirstException(e, sys)


def predict_sentiment_probs(review_text: str, feature_display_name: str) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (probs, label_names): probs is shape (3,) softmax output,
    label_names is meta["sentiment_labels"] (e.g. ["negative","neutral","positive"]).
    """
    import torch

    try:
        model, tokenizer, meta = _load_sentiment_model()
        label_names = meta["sentiment_labels"]

        with torch.no_grad():
            encoded = tokenizer(
                review_text,
                feature_display_name,
                truncation=True,
                padding=True,
                max_length=meta["max_length"],
                return_tensors="pt",
            )
            logits = model(**encoded).logits.cpu().numpy()[0]

        probs = softmax(logits)
        return probs, label_names
    except Exception as e:
        raise FixFirstException(e, sys)