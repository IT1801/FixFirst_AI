"""Thin API client for the FixFirst AI dashboard."""

import sys
from typing import Dict, List, Optional

import requests

from fixfirst.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS
from fixfirst.config.configuration import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


class ApiUnavailableError(FixFirstException):
    """Raised when the FastAPI backend can't be reached at all (connection
    refused, timeout) — distinct from a normal 4xx/5xx HTTP error, so the
    dashboard can show a specific "backend is down" message rather than a
    generic error."""


def _get(path: str, params: Optional[Dict] = None) -> Dict:
    try:
        url = f"{settings.dashboard_api_base_url}{path}"
        logging.info(f"dashboard.api_client: GET {url}")
        response = requests.get(url, params=params, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise ApiUnavailableError(
            f"Could not reach the FixFirst API at {settings.dashboard_api_base_url}. Is the API service running? ({exc})",
            sys,
        )
    except requests.exceptions.Timeout as exc:
        raise ApiUnavailableError(
            f"Request to {settings.dashboard_api_base_url}{path} timed out after {DEFAULT_REQUEST_TIMEOUT_SECONDS}s.",
            sys,
        )
    except requests.exceptions.HTTPError as exc:
        raise FixFirstException(f"API returned an error for {settings.dashboard_api_base_url}{path}: {exc}", sys)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def get_features(active_only: bool = True) -> List[Dict]:
    """Fetch feature metadata for the dashboard."""
    try:
        return _get("/features", params={"active_only": active_only})
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def get_criticality_scores(priority: Optional[str] = None, limit: int = 50) -> List[Dict]:
    """Fetch the latest criticality scores for the dashboard."""
    try:
        params = {"limit": limit}
        if priority:
            params["priority"] = priority
        return _get("/criticality-scores", params=params)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def get_feature_trend(feature_key: str) -> Optional[Dict]:
    """Fetch a feature trend or return None for missing features."""
    try:
        return _get(f"/trends/{feature_key}")
    except FixFirstException as exc:
        if "404" in str(exc):
            return None
        raise


def get_reviews(
    feature_key: Optional[str] = None,
    sentiment: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict:
    """Fetch paginated reviews for the dashboard browser."""
    try:
        params = {"limit": limit, "offset": offset}
        if feature_key:
            params["feature_key"] = feature_key
        if sentiment:
            params["sentiment"] = sentiment
        if source:
            params["source"] = source
        return _get("/reviews", params=params)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def check_api_health() -> bool:
    """Check whether the FastAPI backend is reachable."""
    try:
        result = _get("/health")
        return result.get("status") == "ok"
    except FixFirstException:
        return False