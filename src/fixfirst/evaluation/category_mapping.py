"""
Maps AWARE's native aspect_category vocabulary onto FixFirst AI's
features_master taxonomy, so AWARE's gold annotations (preserved in
raw_metadata.aware_annotations at ingestion time) can be used as a
held-out evaluation set for the fine-tuned models.

WHY THIS EXISTS: AWARE's aspect categories are dataset-specific and were
defined independently of features_master — they do not match 1:1. Some
AWARE categories map cleanly onto one of our feature_keys; others don't
correspond to anything in our taxonomy and are intentionally left
unmapped (None), which means gold-label rows using them are excluded
from evaluation rather than silently mismatched.

ACTION REQUIRED: this mapping is a starting point based on category names
commonly used in ABSA app-review literature. Before running the eval
harness, inspect the actual distinct values in your downloaded AWARE
file (`df['aspect_category'].unique()`) and correct AWARE_CATEGORY_MAP
below — mismatched or missing entries will silently shrink your gold
evaluation set rather than crash, so check the coverage log line the
harness prints on each run.
"""

import sys
from typing import Dict, Optional, Set

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

# --- EDIT THIS after inspecting the real AWARE aspect_category values -----
# Left side: AWARE's raw category string. Right side: our feature_key, or
# None to explicitly exclude that AWARE category from gold-label mapping.
AWARE_CATEGORY_MAP: Dict[str, Optional[str]] = {
    "login": "login_auth",
    "authentication": "login_auth",
    "sync": "sync_speed",
    "synchronization": "sync_speed",
    "crash": "crashes_stability",
    "stability": "crashes_stability",
    "bug": "crashes_stability",
    "performance": "performance_speed",
    "speed": "performance_speed",
    "responsiveness": "ui_responsiveness",
    "ui": "ui_ux_design",
    "ux": "ui_ux_design",
    "design": "ui_ux_design",
    "layout": "ui_ux_design",
    "offline": "offline_mode",
    "notification": "notifications",
    "billing": "billing_subscription",
    "subscription": "billing_subscription",
    "price": "billing_subscription",
    "search": "search",
    "support": "customer_support",
    "help": "customer_support",
    "onboarding": "onboarding",
    "tutorial": "onboarding",
    "privacy": "data_privacy",
    "permissions": "data_privacy",
    "battery": "battery_usage",
    "compatibility": "compatibility",
    "device": "compatibility",
    "update": "updates",
    "ads": "ads_monetization",
    "advertisement": "ads_monetization",
    "monetization": "ads_monetization",
    "gameplay": "gameplay_mechanics",
    "mechanics": "gameplay_mechanics",
    "social": "social_features",
    "sharing": "social_features",
    # AWARE categories with no meaningful counterpart in our taxonomy are
    # intentionally mapped to None rather than omitted, so it's explicit
    # that they were considered and deliberately excluded:
    "general": None,
    "other": None,
}
# ---------------------------------------------------------------------------


def map_aware_category(raw_category: str) -> Optional[str]:
    """
    Case-insensitive, whitespace-normalized lookup. Returns None (excluded
    from gold eval) for anything not present in AWARE_CATEGORY_MAP —
    unrecognized values are logged, not silently dropped, so mapping gaps
    are visible.
    """
    if not raw_category:
        return None

    normalized = str(raw_category).strip().lower()
    if normalized not in AWARE_CATEGORY_MAP:
        logging.warning(f"map_aware_category: unmapped AWARE category {raw_category!r} — add it to AWARE_CATEGORY_MAP")
        return None

    return AWARE_CATEGORY_MAP[normalized]


def report_mapping_coverage(raw_categories: Set[str]) -> None:
    """Logs how many distinct AWARE categories in the dataset are mapped vs unmapped."""
    try:
        mapped = {c for c in raw_categories if map_aware_category(c) is not None}
        unmapped = raw_categories - mapped
        total = len(raw_categories)

        logging.info(
            f"report_mapping_coverage: {len(mapped)}/{total} distinct AWARE categories mapped "
            f"to features_master. Unmapped: {sorted(unmapped) if unmapped else 'none'}"
        )
    except Exception as e:
        raise FixFirstException(e, sys)