"""
Train/val/test split for FixFirst AI.

Splits at the review level (each row is already one unique review/sentence
post-dedup, so there's no group-leakage risk like there would be with
multiple rows per review). Stratifies by domain when available (AWARE rows
carry domain in raw_metadata) so productivity/social/games are represented
proportionally in every split; falls back to a plain random split when no
stratification column is present or a stratum is too small.
"""

import sys
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging
from fixfirst.config.configuration import ConfigurationManager,SplitConfig

def _extract_domain(raw_metadata) -> Optional[str]:
    if isinstance(raw_metadata, dict):
        return raw_metadata.get("domain")
    return None


def split_dataset(
    df: pd.DataFrame,
    config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df). test_size and val_size are each
    a fraction of the FULL dataset (e.g. 0.1 + 0.1 -> 80/10/10 split).
    """
    try:
        if config is None:
            config = ConfigurationManager().get_split_config()

        test_size = config.test_size
        val_size = config.val_size
        stratify_col = config.stratify_column
        random_state = config.random_state

        if test_size + val_size >= 1.0:
            raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")

        out = df.copy()

        strata = None
        if stratify_col in out.columns:
            domains = out[stratify_col].apply(_extract_domain)
            if domains.notna().all() and domains.nunique() > 1:
                # Guard against strata too small for sklearn's stratify requirement
                min_class_count = domains.value_counts().min()
                if min_class_count >= 2:
                    strata = domains

        if strata is None:
            logging.info("split_dataset: no usable stratification column found, using plain random split")

        train_val_df, test_df = train_test_split(
            out,
            test_size=test_size,
            random_state=random_state,
            stratify=strata,
        )

        # Recompute strata for the train/val split on the remaining subset
        strata_2 = strata.loc[train_val_df.index] if strata is not None else None
        relative_val_size = val_size / (1.0 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=strata_2,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logging.info(
            f"split_dataset: total={len(out)} -> "
            f"train={len(train_df)} ({len(train_df)/len(out):.1%}), "
            f"val={len(val_df)} ({len(val_df)/len(out):.1%}), "
            f"test={len(test_df)} ({len(test_df)/len(out):.1%})"
        )
        return train_df, val_df, test_df
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)