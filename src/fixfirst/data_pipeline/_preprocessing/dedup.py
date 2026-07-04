"""
Deduplication for FixFirst AI.

Two passes:
  1. Exact duplicates — same (app_id, cleaned review_text). Common with
     scraped sources that paginate inconsistently.
  2. Near-duplicates — same normalized text (lowercased, punctuation
     stripped) within the same app. Catches "Great app!" vs "great app"
     vs "Great app!!" without being aggressive enough to merge genuinely
     different short reviews.
"""

import re
import sys

import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

_NORMALIZE_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_for_near_dup(text: str) -> str:
    text = text.lower()
    text = _NORMALIZE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate_reviews(
    df: pd.DataFrame,
    text_col: str = "review_text",
    app_col: str = "app_id",
    near_duplicate: bool = True,
) -> pd.DataFrame:
    """
    Returns a deduplicated copy of df. Assumes text_col has already been
    cleaned (see text_cleaning.clean_dataframe).
    """
    try:
        out = df.copy()
        before = len(out)

        out = out.drop_duplicates(subset=[app_col, text_col], keep="first").reset_index(drop=True)
        exact_dropped = before - len(out)

        near_dropped = 0
        if near_duplicate:
            before_near = len(out)
            out["_norm_key"] = out[text_col].apply(_normalize_for_near_dup)
            out = out.drop_duplicates(subset=[app_col, "_norm_key"], keep="first").reset_index(drop=True)
            out = out.drop(columns=["_norm_key"])
            near_dropped = before_near - len(out)

        logging.info(
            f"deduplicate_reviews: {before} -> {len(out)} rows "
            f"(exact_dupes_dropped={exact_dropped}, near_dupes_dropped={near_dropped})"
        )
        return out
    except Exception as e:
        raise FixFirstException(e, sys)