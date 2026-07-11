"""Dataset construction for the multi-label aspect category classifier."""

import sys
from typing import List, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.ml._training.common import build_label_index


def build_category_examples(
    labels_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    feature_keys: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Build one review-level row with a multi-hot target per feature.
    Also computes dynamic positive class weights to counter severe class imbalance.
    """
    try:
        if progress_df.empty:
            raise FixFirstException(
                "Extracted-label progress is empty — run `make label` before training.", sys
            )

        label_index = build_label_index(feature_keys)
        labeled_progress = progress_df[progress_df["status"] == "labeled"][
            ["review_id", "review_text"]
        ].drop_duplicates("review_id")
        if labeled_progress.empty:
            raise FixFirstException("No successfully labeled reviews are available for training.", sys)

        known_labels = labels_df[labels_df["feature_key"].isin(label_index)].copy()
        features_by_review = known_labels.groupby("review_id")["feature_key"].agg(set).to_dict()

        rows = []
        # Tracker for counting positive occurrences of each feature key
        positive_counts = np.zeros(len(label_index), dtype=np.float32)

        for review in labeled_progress.itertuples(index=False):
            target = np.zeros(len(label_index), dtype=np.float32)
            active_features = features_by_review.get(review.review_id, set())
            
            for feature_key in active_features:
                idx = label_index[feature_key]
                target[idx] = 1.0
                positive_counts[idx] += 1.0
                
            rows.append(
                {
                    "review_id": review.review_id,
                    "review_text": review.review_text,
                    "labels": target,
                }
            )
            
        dataset_df = pd.DataFrame(rows)
        total_samples = len(dataset_df)

        # Compute pos_weight for BCEWithLogitsLoss with a protective cap
        MAX_WEIGHT_CAP = 15.0 
        pos_weights = {}
        
        for feature_key, idx in label_index.items():
            pos_count = positive_counts[idx]
            neg_count = total_samples - pos_count
            
            # 1. Calculate the raw ratio
            raw_weight = float(neg_count / (pos_count + 1e-5))
            
            # 2. Apply the mathematical ceiling to prevent overfitting
            capped_weight = min(raw_weight, MAX_WEIGHT_CAP)
            
            # 3. Store the capped weight
            pos_weights[feature_key] = capped_weight

        return dataset_df, pos_weights
        
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


class AspectCategoryDataset:
    """Tokenized review texts with float multi-hot labels."""

    def __init__(
        self,
        texts: List[str],
        labels: Sequence[np.ndarray],
        tokenizer,
        max_length: int = 128,
    ):
        import torch

        self._torch = torch
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: self._torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = self._torch.tensor(self.labels[index], dtype=self._torch.float32)
        return item