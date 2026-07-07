import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.ml._training.common import build_label_index

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def build_sentiment_examples(
    labels_df: pd.DataFrame,
    feature_display_names: Dict[str, str],
    review_id_col: str = "review_id",
    text_col: str = "review_text",
) -> Tuple[pd.DataFrame, List[float]]:
    try:
        if labels_df.empty:
            raise FixFirstException(
                "labels_df is empty — nothing to build sentiment training examples from.", sys
            )

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
                "No valid (review, feature, sentiment) rows remain after filtering.",
                sys,
            )
            
        labels_list = [sentiment_index[s] for s in df["sentiment"].values]

        examples_df = pd.DataFrame(
            {
                "review_id": df[review_id_col].values,
                "feature_key": df["feature_key"].values,
                "text_a": df[text_col].values,
                "text_b": [feature_display_names[fk] for fk in df["feature_key"].values],
                "label": labels_list,
            }
        ).reset_index(drop=True)
        
        total_samples = len(labels_list)
        num_classes = len(SENTIMENT_LABELS)
        
        class_counts = np.bincount(labels_list, minlength=num_classes)
        
        class_weights = []
        for count in class_counts:
            weight = total_samples / (num_classes * (count + 1e-5))
            class_weights.append(min(float(weight), 10.0))

        logging.info(
            f"build_sentiment_examples: {len(examples_df)} pair examples built. "
            f"Sentiment distribution: Negative={class_counts[0]}, Neutral={class_counts[1]}, Positive={class_counts[2]}. "
            f"Computed weights: {class_weights}"
        )
        
        return examples_df, class_weights
        
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)


class AspectSentimentDataset:
    """Thin PyTorch Dataset wrapper for sentence-pair (review, feature) inputs."""

    def __init__(
        self,
        text_a: List[str],
        text_b: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        import torch

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
