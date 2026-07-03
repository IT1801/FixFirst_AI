"""
Pydantic response schemas for the FixFirst AI API.

Separated from queries.py/routes so response shape is defined once and
reused across all route handlers — no endpoint hand-builds its own dict.
"""

from datetime import date, datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class FeatureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    feature_key: str
    display_name: str
    description: Optional[str] = None
    is_active: bool


class ReviewAspectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    feature_key: str
    sentiment: str
    confidence: float
    source: str


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source: str
    app_id: str
    review_text: str
    rating: Optional[int] = None
    review_date: Optional[date] = None
    aspects: List[ReviewAspectOut] = []


class PaginatedReviews(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[ReviewOut]


class CriticalityScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    feature_key: str
    display_name: str
    window_start: date
    window_end: date
    score: float
    mention_count: int
    negative_ratio: float
    trend_delta: Optional[float] = None


class TrendPointOut(BaseModel):
    window_start: date
    window_end: date
    score: float
    mention_count: int
    negative_ratio: float


class FeatureTrendOut(BaseModel):
    feature_key: str
    display_name: str
    points: List[TrendPointOut]


class HealthOut(BaseModel):
    status: str
    version: str