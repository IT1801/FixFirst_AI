"""
Feature taxonomy loader for FixFirst AI.

The LLM silver-labeler must be constrained to the SAME closed set of
feature categories used everywhere else in the system (features_master).
Without this constraint, the LLM would invent free-text aspect categories
that never match anything in review_aspects/criticality_scores, silently
breaking the aggregation layer.
"""

import sys
from typing import List, Dict

from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def load_active_taxonomy() -> List[Dict[str, str]]:
    """
    Returns active features_master rows as a list of
    {feature_key, display_name, description} dicts, used to build the
    closed label set embedded in LLM prompts.
    """
    from fixfirst.db.base import get_db
    from fixfirst.db.models import FeatureMaster

    try:
        with get_db() as db:
            rows = (
                db.query(FeatureMaster)
                .filter(FeatureMaster.is_active.is_(True))
                .order_by(FeatureMaster.feature_key)
                .all()
            )
            taxonomy = [
                {
                    "feature_key": r.feature_key,
                    "display_name": r.display_name,
                    "description": r.description or "",
                }
                for r in rows
            ]

        if not taxonomy:
            raise FixFirstException(
                "features_master has no active rows — run scripts/seed_features.py first.", sys
            )

        logging.info(f"load_active_taxonomy: loaded {len(taxonomy)} active feature categories")
        return taxonomy
    except FixFirstException:
        raise
    except Exception as e:
        raise FixFirstException(e, sys)