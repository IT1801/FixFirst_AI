"""
Text cleaning for FixFirst AI.
"""

import re
import sys
import unicodedata

import pandas as pd

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.])\1{2,}")  # "!!!!" -> "!!", "......" -> ".."
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Cleans a single review string. Returns '' for null/empty input."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1", text)
    text = _MULTI_WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Returns a copy of df with `text_col` cleaned in place and empty-after-
    cleaning rows dropped (e.g. reviews that were only a URL or emoji spam).
    """
    try:
        out = df.copy()
        out[text_col] = out[text_col].apply(clean_text)

        before = len(out)
        out = out[out[text_col].str.len() > 0].reset_index(drop=True)
        dropped = before - len(out)

        if dropped:
            logging.info(f"clean_dataframe: dropped {dropped} rows that were empty after cleaning")

        return out
    except Exception as e:
        raise FixFirstException(e, sys)