"""
Prompt construction for LLM-based silver labeling.

Design choices:
  - The feature taxonomy is embedded as a closed set with key + description,
    so the LLM classifies into OUR categories rather than inventing its own.
  - The LLM is asked to return ONLY features explicitly discussed in the
    review, not to force-fit every review into a category (most reviews
    only touch 1-2 features).
  - Output is strict JSON with no prose, parsed downstream by parser.py.
"""

import json
from typing import List, Dict

SYSTEM_PROMPT = """You are an expert annotator for Aspect-Based Sentiment Analysis (ABSA) on \
software/app reviews. Given a review and a closed set of feature categories, identify which \
categories are explicitly discussed in the review and the sentiment expressed toward each one.

Rules:
- Only include a feature category if the review explicitly discusses it (do not guess or infer \
categories that aren't actually mentioned).
- A review may mention zero, one, or multiple feature categories.
- sentiment must be exactly one of: "positive", "negative", "neutral".
- feature_key must be exactly one of the provided category keys — never invent a new one.
- Respond with ONLY a JSON object, no other text, no markdown code fences.
"""

RESPONSE_SCHEMA_HINT = '{"aspects": [{"feature_key": "<key>", "sentiment": "<positive|negative|neutral>"}]}'
BATCH_RESPONSE_SCHEMA_HINT = (
  '{"reviews": [{"review_id": "<id>", "aspects": '
  '[{"feature_key": "<key>", "sentiment": "<positive|negative|neutral>"}]}]}'
)


def _format_taxonomy(taxonomy: List[Dict[str, str]]) -> str:
    lines = []
    for feature in taxonomy:
        desc = f" — {feature['description']}" if feature.get("description") else ""
        lines.append(f"- {feature['feature_key']}: {feature['display_name']}{desc}")
    return "\n".join(lines)


def build_absa_prompt(review_text: str, taxonomy: List[Dict[str, str]]) -> str:
    """Builds the user-turn prompt for a single review."""
    taxonomy_block = _format_taxonomy(taxonomy)
    return (
        f"Feature categories:\n{taxonomy_block}\n\n"
        f'Review: "{review_text}"\n\n'
        f"Respond with JSON matching exactly this shape:\n{RESPONSE_SCHEMA_HINT}\n"
        f'If no categories are discussed, respond with {{"aspects": []}}.'
    )


def build_absa_batch_prompt(reviews: List[Dict[str, str]], taxonomy: List[Dict[str, str]]) -> str:
  """Builds the user-turn prompt for a batch of reviews."""
  taxonomy_block = _format_taxonomy(taxonomy)
  review_lines = []
  for review in reviews:
    review_id = json.dumps(str(review["review_id"]))
    review_text = json.dumps(review["review_text"])
    review_lines.append(f"- review_id: {review_id}\n  review_text: {review_text}")

  review_block = "\n".join(review_lines)
  return (
    f"Feature categories:\n{taxonomy_block}\n\n"
    f"Reviews to label:\n{review_block}\n\n"
    f"Respond with JSON matching exactly this shape:\n{BATCH_RESPONSE_SCHEMA_HINT}\n"
    f"Include exactly one entry per input review_id, in the same order. "
    f'If a review has no categories discussed, use {{"review_id": "...", "aspects": []}} for that review.'
  )