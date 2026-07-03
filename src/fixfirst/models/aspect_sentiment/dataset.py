"""
Dataset construction for the aspect SENTIMENT classifier (single-label).

Unlike the category classifier (multi-hot over the whole review), this
model is conditioned on a (review, feature) PAIR: given a review and a
specific feature that's already known to be discussed in it, predict the
sentiment expressed toward THAT feature specifically. A review mentioning
both login (negative) and UI (positive) becomes two separate training
examples, one per feature, each with its own label — this is what makes
it aspect-based rather than whole-review sentiment.

Input formatting follows the standard ABSA "sentence pair" convention:
    text_a = review_text
    text_b = feature display_name (natural language, not the raw key)
so the tokenizer's [SEP]-joined pair gives the model an explicit signal
about which aspect to focus sentiment on, rather than making it guess
from a single blended string.

Source: only silver_labels.parquet rows are used (each row IS already a
single (review, feature, sentiment) label produced by the LLM) — there is
no "zero-aspect" analog here, since a review with zero discussed features
contributes zero sentiment-classifier examples by definition.
"""

import sys
from typing import Dict, List

import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.models.common import build_label_index

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def build_sentiment_examples(
    labels_df: pd.DataFrame,
    feature_display_names: Dict[str, str],
    review_id_col: str = "review_id",
    text_col: str = "review_text",
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [review_id, feature_key, text_a, text_b,
    label], one row per (review, feature) pair in labels_df.

    text_a: the review text
    text_b: the feature's natural-language display name (e.g. "Login /
            Authentication" rather than "login_auth")
    label:  integer index into SENTIMENT_LABELS
    """
    try:
        if labels_df.empty:
            raise FixFirstException(
                "labels_df is empty — nothing to build sentiment training examples from.", sys
            )

        sentiment_index = build_label_index(SENTIMENT_LABELS)
        # SENTIMENT_LABELS is intentionally NOT alphabetical-sorted for display
        # (negative/neutral/positive reads naturally), but build_label_index
        # sorts alphabetically for determinism — reconcile explicitly here
        # rather than relying on incidental list order matching sorted order.
        sentiment_index = {label: i for i, label in enumerate(SENTIMENT_LABELS)}

        unknown_features = set(labels_df["feature_key"]) - set(feature_display_names.keys())
        if unknown_features:
            logging.warning(
                f"build_sentiment_examples: {len(unknown_features)} feature_keys in labels_df "
                f"have no display name mapping and will be dropped: {unknown_features}"
            )

        df = labels_df[labels_df["feature_key"].isin(feature_display_names.keys())].copy()

        invalid_sentiment_mask = ~df["sentiment"].isin(SENTIMENT_LABELS)
        if invalid_sentiment_mask.any():
            n_invalid = invalid_sentiment_mask.sum()
            logging.warning(f"build_sentiment_examples: dropping {n_invalid} rows with invalid sentiment values")
            df = df[~invalid_sentiment_mask]

        if df.empty:
            raise FixFirstException(
                "No valid (review, feature, sentiment) rows remain after filtering unknown "
                "features and invalid sentiment values.",
                sys,
            )

        examples_df = pd.DataFrame(
            {
                "review_id": df[review_id_col].values,
                "feature_key": df["feature_key"].values,
                "text_a": df[text_col].values,
                "text_b": [feature_display_names[fk] for fk in df["feature_key"].values],
                "label": [sentiment_index[s] for s in df["sentiment"].values],
            }
        ).reset_index(drop=True)

        logging.info(
            f"build_sentiment_examples: {len(examples_df)} (review, feature) pair examples built "
            f"from {labels_df[review_id_col].nunique()} source reviews"
        )
        return examples_df
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)


class AspectSentimentDataset:
    """
    Thin PyTorch Dataset wrapper for sentence-pair (review, feature) inputs.
    Import of torch is deferred to __init__ so build_sentiment_examples can
    be unit-tested without torch installed.
    """

    def __init__(
        self,
        text_a: List[str],
        text_b: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        import torch  # deferred import

        self._torch = torch
        self.encodings = tokenizer(
            list(text_a),
            list(text_b),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        torch = self._torch
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item