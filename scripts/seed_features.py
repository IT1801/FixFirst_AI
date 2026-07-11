"""Seed data and seeding script for features_master."""

import sys
from typing import Dict, List

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

FEATURES_SEED: List[Dict[str, str]] = [
    {"feature_key": "usability", "display_name": "Usability", "description": "Ease of use and user interface."},
    {"feature_key": "effectiveness", "display_name": "Effectiveness", "description": "How well the application achieves its goals."},
    {"feature_key": "general", "display_name": "General", "description": "General feedback and comments."},
    {"feature_key": "safety", "display_name": "Safety", "description": "Physical safety or mental well-being related."},
    {"feature_key": "learnability", "display_name": "Learnability", "description": "Ease of learning the application."},
    {"feature_key": "cost", "display_name": "Cost", "description": "Pricing, subscriptions, and value for money."},
    {"feature_key": "efficiency", "display_name": "Efficiency", "description": "Performance, speed, and resource usage."},
    {"feature_key": "compatibility", "display_name": "Compatibility", "description": "Device and OS compatibility."},
    {"feature_key": "security", "display_name": "Security", "description": "Data security and privacy concerns."},
    {"feature_key": "aesthetics", "display_name": "Aesthetics", "description": "Visual design and look-and-feel."},
    {"feature_key": "reliability", "display_name": "Reliability", "description": "Bugs, crashes, and stability issues."},
    {"feature_key": "enjoyability", "display_name": "Enjoyability", "description": "Fun, engaging, and pleasant to use."},
]


def seed_features_master() -> int:
    """
    Inserts FEATURES_SEED into features_master, skipping rows whose
    feature_key already exists (idempotent — safe to re-run).
    Returns the number of newly inserted rows.
    """
    try:
        from fixfirst.core.db import FeatureMaster, get_db

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
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def main() -> int:
    """Run the features seeding CLI."""
    try:
        count = seed_features_master()
        logging.info(f"Done. {count} features inserted.")
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())