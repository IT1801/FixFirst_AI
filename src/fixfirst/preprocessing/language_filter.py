"""
Language filtering for FixFirst AI.

AWARE is English-only, but future sources (Google Play scrapes, App Store)
are multilingual. The ABSA models (fine-tuned + LLM fallback) target
English, so non-English reviews are filtered here rather than silently
mispredicted downstream.

Uses langdetect (pure Python, no model download required). It is not
perfectly accurate on very short strings (<5 words) — those are kept by
default rather than dropped, since a 3-word review like "Great app!" is
unambiguous to a human but easy for langdetect to misclassify.
"""

import sys

import pandas as pd
from langdetect import detect, LangDetectException, DetectorFactory

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

# Pin the seed so detection is deterministic across runs.
DetectorFactory.seed = 42

MIN_WORDS_FOR_DETECTION = 5


def _safe_detect(text: str) -> str:
    """Returns ISO 639-1 language code, or 'en' as a permissive default
    for very short text where detection is unreliable, or 'unknown' on
    detector failure."""
    if len(text.split()) < MIN_WORDS_FOR_DETECTION:
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def filter_english(
    df: pd.DataFrame,
    text_col: str = "review_text",
    keep_langs: tuple = ("en",),
) -> pd.DataFrame:
    """Returns a copy of df containing only rows detected as English
    (or too short to reliably detect, which default to 'en')."""
    try:
        out = df.copy()
        out["_detected_lang"] = out[text_col].apply(_safe_detect)

        before = len(out)
        out = out[out["_detected_lang"].isin(keep_langs)].drop(columns=["_detected_lang"]).reset_index(drop=True)
        dropped = before - len(out)

        logging.info(f"filter_english: {before} -> {len(out)} rows (dropped {dropped} non-English)")
        return out
    except Exception as e:
        raise FixFirstException(e, sys)