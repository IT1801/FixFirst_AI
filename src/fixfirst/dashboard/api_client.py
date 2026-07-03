"""
Thin API client for the FixFirst AI dashboard.

Kept separate from app.py (which contains only Streamlit rendering calls)
so these HTTP + data-shaping functions are unit-testable without a
Streamlit runtime — app.py should never construct a request or parse a
response body itself, only call functions from this module and render
the result.
"""

import sys
from typing import Dict, List, Optional

import requests

from fixfirst.config.settings import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

DEFAULT_TIMEOUT_SECONDS = 10


class ApiUnavailableError(FixFirstException):
    """Raised when the FastAPI backend can't be reached at all (connection
    refused, timeout) — distinct from a normal 4xx/5xx HTTP error, so the
    dashboard can show a specific "backend is down" message rather than a
    generic error."""


def _get(path: str, params: Optional[Dict] = None) -> Dict:
    url = f"{settings.dashboard_api_base_url}{path}"
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        raise ApiUnavailableError(
            f"Could not reach the FixFirst API at {settings.dashboard_api_base_url}. "
            f"Is the API service running? ({e})",
            sys,
        )
    except requests.exceptions.Timeout as e:
        raise ApiUnavailableError(f"Request to {url} timed out after {DEFAULT_TIMEOUT_SECONDS}s.", sys)
    except requests.exceptions.HTTPError as e:
        raise FixFirstException(f"API returned an error for {url}: {e}", sys)
    except Exception as e:
        raise FixFirstException(e, sys)


def get_features(active_only: bool = True) -> List[Dict]:
    return _get("/features", params={"active_only": active_only})


def get_criticality_scores(priority: Optional[str] = None, limit: int = 50) -> List[Dict]:
    params = {"limit": limit}
    if priority:
        params["priority"] = priority
    return _get("/criticality-scores", params=params)


def get_feature_trend(feature_key: str) -> Optional[Dict]:
    try:
        return _get(f"/trends/{feature_key}")
    except FixFirstException as e:
        if "404" in str(e):
            return None
        raise


def get_reviews(
    feature_key: Optional[str] = None,
    sentiment: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict:
    params = {"limit": limit, "offset": offset}
    if feature_key:
        params["feature_key"] = feature_key
    if sentiment:
        params["sentiment"] = sentiment
    if source:
        params["source"] = source
    return _get("/reviews", params=params)


def check_api_health() -> bool:
    try:
        result = _get("/health")
        return result.get("status") == "ok"
    except FixFirstException:
        return False