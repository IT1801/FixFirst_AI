"""
Train/val/test split for FixFirst AI.

Splits at the REVIEW level to prevent data leakage: all sentences that belong
to the same source review (identified by ``aware_review_id`` stored in
raw_metadata by the ingestion layer) are always placed in the same split.

Stratifies by domain when available (AWARE rows carry domain in raw_metadata)
so productivity/social/games are represented proportionally in every split;
falls back to a plain random split when no stratification column is present or
a stratum is too small.
"""

import sys
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.config.configuration import ConfigurationManager, SplitConfig


def _extract_domain(raw_metadata) -> Optional[str]:
    if isinstance(raw_metadata, dict):
        return raw_metadata.get("domain")
    return None


def split_dataset(
    df: pd.DataFrame,
    config: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df).

    The split is performed at the *review* level using ``aware_review_id``
    stored in each row's ``raw_metadata``.  This guarantees that every
    sentence from the same source review lands in the same split, eliminating
    cross-split data leakage.

    ``test_size`` and ``val_size`` are fractions of the full dataset
    (e.g. 0.1 + 0.1 → 80 / 10 / 10 split).
    """
    try:
        if config is None:
            config = ConfigurationManager().get_split_config()

        test_size  = config.test_size
        val_size   = config.val_size
        stratify_col = config.stratify_column
        random_state = config.random_state

        if test_size + val_size >= 1.0:
            raise ValueError(
                f"test_size + val_size must be < 1.0, got {test_size + val_size}"
            )

        out = df.copy()

        # ------------------------------------------------------------------
        # Step 1 — assign a group key per SOURCE REVIEW (not per sentence).
        # The ingestion layer stores the original AWARE review_id in
        # raw_metadata["aware_review_id"].  Rows without one (other sources)
        # get a synthetic per-row key so they are still included but treated
        # as independent groups (no leakage risk there).
        # ------------------------------------------------------------------
        out["_group_key"] = out["raw_metadata"].apply(
            lambda m: (
                m.get("aware_review_id")
                if isinstance(m, dict) and m.get("aware_review_id")
                else None
            )
        )
        missing_mask = out["_group_key"].isna()
        out.loc[missing_mask, "_group_key"] = out.loc[missing_mask, "id"].astype(str)

        # ------------------------------------------------------------------
        # Step 2 — build a deduplicated group-level series for splitting.
        # ------------------------------------------------------------------
        domains_by_group = (
            out.groupby("_group_key")["raw_metadata"]
            .first()
            .apply(_extract_domain)
        )
        group_index = domains_by_group.index  # one entry per unique review

        # ------------------------------------------------------------------
        # Step 3 — determine stratification strata (domain) at group level.
        # ------------------------------------------------------------------
        strata = None
        if stratify_col in out.columns:
            if domains_by_group.notna().all() and domains_by_group.nunique() > 1:
                min_class_count = domains_by_group.value_counts().min()
                if min_class_count >= 2:
                    strata = domains_by_group
        if strata is None:
            logging.info(
                "split_dataset: no usable stratification column found, "
                "using plain random split"
            )

        # ------------------------------------------------------------------
        # Step 4 — split GROUP IDs (not rows) so no review straddles splits.
        # ------------------------------------------------------------------
        group_train_val, group_test = train_test_split(
            group_index,
            test_size=test_size,
            random_state=random_state,
            stratify=strata,
        )

        strata_2 = strata.loc[group_train_val] if strata is not None else None
        relative_val_size = val_size / (1.0 - test_size)

        group_train, group_val = train_test_split(
            group_train_val,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=strata_2,
        )

        # ------------------------------------------------------------------
        # Step 5 — map every sentence row to its split via the group key.
        # ------------------------------------------------------------------
        split_map = (
            {g: "train" for g in group_train}
            | {g: "val"   for g in group_val}
            | {g: "test"  for g in group_test}
        )
        out["_split"] = out["_group_key"].map(split_map)

        train_df = (
            out[out["_split"] == "train"]
            .drop(columns=["_group_key", "_split"])
            .reset_index(drop=True)
        )
        val_df = (
            out[out["_split"] == "val"]
            .drop(columns=["_group_key", "_split"])
            .reset_index(drop=True)
        )
        test_df = (
            out[out["_split"] == "test"]
            .drop(columns=["_group_key", "_split"])
            .reset_index(drop=True)
        )

        logging.info(
            f"split_dataset (review-level, leak-free): total={len(out)} -> "
            f"train={len(train_df)} ({len(train_df)/len(out):.1%}), "
            f"val={len(val_df)} ({len(val_df)/len(out):.1%}), "
            f"test={len(test_df)} ({len(test_df)/len(out):.1%})"
        )
        return train_df, val_df, test_df
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)