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

# PERFECTLY ENGINEERED HYPOTHESES FOR DEBERTA-V3 NLI
# Creates clear mathematical separation between overlapping categories
PERFECT_HYPOTHESES: Dict[str, str] = {
    "customer_support": "This text discusses customer support, contacting help desks, support tickets, agent responses, or reporting issues to developers.",
    "social_features": "This text mentions social features, group chats, messaging friends, server communities, voice calls, or sharing content with other people.",
    "gameplay_mechanics": "This text talks about the core gameplay mechanics, game rules, crafting recipes, levels, matches, or specific in-game interactive elements.",
    "crashes_stability": "This text explicitly mentions the application crashing, freezing, force-closing, breaking, or completely stopping working.",
    "data_privacy": "This text discusses data privacy, account security, data breaches, information tracking, privacy policies, or personal data tracking safety.",
    "ui_ux_design": "This text describes the user interface design, visual layout, color palette, font style, themes, look and feel, or cosmetic aesthetics.",
    "ui_responsiveness": "This text complains about slow UI responsiveness, delayed button taps, input lag, screen freezing while typing, or sluggish transitions.",
    "compatibility": "This text mentions hardware or OS compatibility, running on specific device models like iPhone or iPad, trackpad support, or operating system versions.",
    "ads_monetization": "This text references advertisements, pop-up ads, video ads, monetization models, or being forced to watch ads to earn currency.",
    "offline_mode": "This text mentions using the application without internet, offline mode, local data storage, or lack of network connectivity features.",
    "search": "This text focuses specifically on finding items via a search bar, applying search filters, sorting query results, or tagging search items.",
    "battery_usage": "This text complains about heavy battery usage, battery drain, or the device overheating and getting extremely hot while using the app.",
    "login_auth": "This text is about authentication, logging in, signing up for an account, password errors, or being locked out of an account.",
    "notifications": "This text discusses system notifications, push notification alerts, unread badging icons, loud notification sounds, or missed reminders.",
    "performance_speed": "This text mentions overall performance speed, server lag, loading times, long waiting screens, or general network latency.",
    "updates": "This text talks about a software update, a new version release, patch notes, or changes introduced in the latest version.",
    "billing_subscription": "This text references billing, subscription prices, premium memberships, purchasing plans, bank accounts, transactions, or refund requests.",
    "onboarding": "This text describes the initial onboarding experience, welcome screens, introductory tutorials, or the setup flow for new users.",
    "sync_speed": "This text explicitly monitors data sync speed, cross-device syncing delays, cloud saving lag, or the time it takes to backup data."
}


def load_active_taxonomy() -> List[Dict[str, str]]:
    """
    Returns active features_master rows as a list of
    {feature_key, display_name, description, hypothesis} dicts, used to build the
    closed label set embedded in LLM prompts and zero-shot models.
    """
    from fixfirst.core._db.base import get_db
    from fixfirst.core._db.models import FeatureMaster

    try:
        with get_db() as db:
            rows = (
                db.query(FeatureMaster)
                .filter(FeatureMaster.is_active.is_(True))
                .order_by(FeatureMaster.feature_key)
                .all()
            )
            
            taxonomy = []
            for r in rows:
                desc = r.description.strip() if r.description else ""
                
                # Check if we have a hand-engineered perfect hypothesis ready
                if r.feature_key in PERFECT_HYPOTHESES:
                    hypothesis = PERFECT_HYPOTHESES[r.feature_key]
                # Fallback to dynamic creation if a new row is added to the DB later
                elif desc:
                    hypothesis = f"This review discusses {r.display_name}, specifically: {desc}."
                else:
                    hypothesis = f"This review discusses {r.display_name}."

                taxonomy.append({
                    "feature_key": r.feature_key,
                    "display_name": r.display_name,
                    "description": desc,
                    "hypothesis": hypothesis,
                })

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