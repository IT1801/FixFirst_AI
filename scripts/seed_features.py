"""
Seed data and seeding script for features_master.

This taxonomy is a starting point, curated to cover common software/app
complaint categories seen across productivity, social, and games apps
(the same three AWARE domains). It is intentionally generic so it transfers
across app types; refine feature_key entries as real review data surfaces
categories this list misses.

Usage:
    PYTHONPATH=src python scripts/seed_features.py
"""

import sys
from typing import List, Dict

FEATURES_SEED: List[Dict[str, str]] = [
    {
        "feature_key": "login_auth",
        "display_name": "Login / Authentication",
        "description": "Sign-in, sign-up, password reset, OAuth, session expiry issues.",
    },
    {
        "feature_key": "crashes_stability",
        "display_name": "Crashes / Stability",
        "description": "App crashes, freezes, force-closes, unresponsive UI.",
    },
    {
        "feature_key": "sync_speed",
        "display_name": "Sync Speed",
        "description": "Data sync delays, sync failures, cross-device consistency.",
    },
    {
        "feature_key": "performance_speed",
        "display_name": "Performance / Loading Speed",
        "description": "General app speed, lag, slow load times unrelated to sync.",
    },
    {
        "feature_key": "ui_responsiveness",
        "display_name": "UI Responsiveness",
        "description": "UI lag, jank, dropped touch input, animation stutter.",
    },
    {
        "feature_key": "ui_ux_design",
        "display_name": "UI/UX Design",
        "description": "Layout, navigation clarity, visual design complaints/praise.",
    },
    {
        "feature_key": "offline_mode",
        "display_name": "Offline Mode",
        "description": "Functionality (or lack thereof) without an internet connection.",
    },
    {
        "feature_key": "notifications",
        "display_name": "Notifications",
        "description": "Push notification reliability, frequency, relevance.",
    },
    {
        "feature_key": "billing_subscription",
        "display_name": "Billing / Subscription",
        "description": "Payment failures, subscription management, pricing complaints.",
    },
    {
        "feature_key": "search",
        "display_name": "Search",
        "description": "In-app search accuracy, speed, filtering.",
    },
    {
        "feature_key": "customer_support",
        "display_name": "Customer Support",
        "description": "Responsiveness and quality of support/help channels.",
    },
    {
        "feature_key": "onboarding",
        "display_name": "Onboarding",
        "description": "First-run experience, tutorials, setup friction.",
    },
    {
        "feature_key": "data_privacy",
        "display_name": "Data & Privacy",
        "description": "Data handling, permissions, privacy policy concerns.",
    },
    {
        "feature_key": "battery_usage",
        "display_name": "Battery Usage",
        "description": "Excessive battery drain attributed to the app.",
    },
    {
        "feature_key": "compatibility",
        "display_name": "Device / OS Compatibility",
        "description": "Issues specific to OS versions, device models, screen sizes.",
    },
    {
        "feature_key": "updates",
        "display_name": "App Updates",
        "description": "Update-induced regressions, forced updates, changelog complaints.",
    },
    {
        "feature_key": "ads_monetization",
        "display_name": "Ads / Monetization",
        "description": "Ad frequency, intrusiveness, in-app purchase friction.",
    },
    {
        "feature_key": "gameplay_mechanics",
        "display_name": "Gameplay / Core Mechanics",
        "description": "Games domain: balance, difficulty, controls, core loop feedback.",
    },
    {
        "feature_key": "social_features",
        "display_name": "Social / Sharing Features",
        "description": "Friend lists, sharing, messaging, social-networking-specific features.",
    },
]


def seed_features_master() -> int:
    """
    Inserts FEATURES_SEED into features_master, skipping rows whose
    feature_key already exists (idempotent — safe to re-run).
    Returns the number of newly inserted rows.
    """
    from fixfirst.db.base import get_db
    from fixfirst.db.models import FeatureMaster
    from fixfirst.exceptions.exception import FixFirstException
    from fixfirst.logging.logger import logging

    try:
        inserted = 0
        with get_db() as db:
            existing_keys = {row[0] for row in db.query(FeatureMaster.feature_key).all()}
            for feature in FEATURES_SEED:
                if feature["feature_key"] in existing_keys:
                    continue
                db.add(FeatureMaster(**feature))
                inserted += 1

        logging.info(
            f"Seeded {inserted} new features_master rows "
            f"({len(FEATURES_SEED) - inserted} already existed, skipped)."
        )
        return inserted
    except Exception as e:
        raise FixFirstException(e, sys)


if __name__ == "__main__":
    from fixfirst.exceptions.exception import FixFirstException
    from fixfirst.logging.logger import logging

    try:
        count = seed_features_master()
        logging.info(f"Done. {count} features inserted.")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)