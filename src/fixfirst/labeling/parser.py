"""
Parser/validator for LLM ABSA responses.

Handles the realistic failure modes of LLM JSON output:
  - Wrapped in markdown code fences despite instructions not to.
  - Trailing prose before/after the JSON object.
  - Invented feature_key values not in the taxonomy (dropped, logged).
  - Invalid sentiment values (dropped, logged).
  - Duplicate (feature_key, sentiment) pairs for the same review (deduped).

This module has NO network dependency and is fully unit-testable.
"""

import json
import re
import sys
from typing import List, Dict, Set, Optional

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_str(raw_text: str) -> str:
    """Strips markdown fences and isolates the outermost JSON object."""
    fenced = _CODE_FENCE_RE.search(raw_text)
    candidate = fenced.group(1) if fenced else raw_text

    obj_match = _JSON_OBJECT_RE.search(candidate)
    if obj_match:
        return obj_match.group(0)
    return candidate


def _validate_aspects(aspects, valid_feature_keys: Set[str]) -> List[Dict[str, str]]:
    validated: List[Dict[str, str]] = []
    seen: Set[tuple] = set()

    if not isinstance(aspects, list):
        raise FixFirstException(f"'aspects' must be a list, got: {type(aspects)}", sys)

    for item in aspects:
        if not isinstance(item, dict):
            logging.warning(f"parse_llm_response: skipping non-dict aspect item: {item!r}")
            continue

        feature_key = item.get("feature_key")
        sentiment = item.get("sentiment")

        if feature_key not in valid_feature_keys:
            logging.warning(f"parse_llm_response: dropping unknown feature_key {feature_key!r}")
            continue

        if sentiment not in VALID_SENTIMENTS:
            logging.warning(
                f"parse_llm_response: dropping invalid sentiment {sentiment!r} for feature {feature_key!r}"
            )
            continue

        key = (feature_key, sentiment)
        if key in seen:
            continue
        seen.add(key)

        validated.append({"feature_key": feature_key, "sentiment": sentiment})

    return validated


def parse_llm_response(raw_text: str, valid_feature_keys: Set[str]) -> List[Dict[str, str]]:
    """
    Parses and validates an LLM ABSA response.

    Returns a list of {feature_key, sentiment} dicts, silently dropping
    (and logging) any entries that fail validation rather than raising —
    a single malformed aspect in an otherwise-valid response shouldn't
    discard the whole review's labels.
    """
    try:
        json_str = _extract_json_str(raw_text)
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        raise FixFirstException(f"Failed to parse LLM response as JSON: {e}. Raw text: {raw_text[:200]!r}", sys)

    if not isinstance(parsed, dict) or "aspects" not in parsed:
        raise FixFirstException(
            f"LLM response JSON missing expected 'aspects' key. Got: {parsed!r}", sys
        )

    aspects = parsed["aspects"]
    if not isinstance(aspects, list):
        raise FixFirstException(f"'aspects' must be a list, got: {type(aspects)}", sys)

    return _validate_aspects(aspects, valid_feature_keys)


def parse_llm_batch_response(
    raw_text: str,
    valid_feature_keys: Set[str],
    expected_review_ids: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    """
    Parses a batched LLM response shaped like {"reviews": [{"review_id": ..., "aspects": [...]}]}.

    Returns a list of {review_id, aspects} dicts. Unknown review_ids are
    dropped with a warning; missing review_ids are handled by the caller.
    """
    try:
        json_str = _extract_json_str(raw_text)
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        raise FixFirstException(f"Failed to parse LLM batch response as JSON: {e}. Raw text: {raw_text[:200]!r}", sys)

    if not isinstance(parsed, dict) or "reviews" not in parsed:
        raise FixFirstException(
            f"LLM batch response JSON missing expected 'reviews' key. Got: {parsed!r}", sys
        )

    reviews = parsed["reviews"]
    if not isinstance(reviews, list):
        raise FixFirstException(f"'reviews' must be a list, got: {type(reviews)}", sys)

    validated: List[Dict[str, object]] = []
    seen_review_ids: Set[str] = set()

    for item in reviews:
        if not isinstance(item, dict):
            logging.warning(f"parse_llm_batch_response: skipping non-dict review item: {item!r}")
            continue

        review_id = item.get("review_id")
        if review_id is None:
            logging.warning(f"parse_llm_batch_response: skipping review item without review_id: {item!r}")
            continue

        review_id_str = str(review_id)
        if expected_review_ids is not None and review_id_str not in expected_review_ids:
            logging.warning(f"parse_llm_batch_response: dropping unexpected review_id {review_id_str!r}")
            continue

        if review_id_str in seen_review_ids:
            continue
        seen_review_ids.add(review_id_str)

        aspects = item.get("aspects", [])
        validated.append(
            {
                "review_id": review_id_str,
                "aspects": _validate_aspects(aspects, valid_feature_keys),
            }
        )

    return validated