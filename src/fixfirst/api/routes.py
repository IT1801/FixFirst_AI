"""HTTP route handlers for the FixFirst API."""

import sys
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from fixfirst.api import queries
from fixfirst.api.deps import get_db_session
from fixfirst.api.schemas import (
    CriticalityScoreOut,
    FeatureOut,
    FeatureTrendOut,
    PaginatedReviews,
    ReviewAspectOut,
    ReviewOut,
    TrendPointOut,
)
from fixfirst.constants import DEFAULT_API_LIMIT
from fixfirst.core.db import RawReview
from fixfirst.exceptions.exception import FixFirstException

router = APIRouter()


@router.get("/features", response_model=List[FeatureOut], tags=["features"])
def get_features(
    active_only: bool = Query(True),
    db: Session = Depends(get_db_session),
) -> List[FeatureOut]:
    try:
        return queries.list_features(db, active_only=active_only)
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def _serialize_review(review: RawReview) -> ReviewOut:
    aspects = [
        ReviewAspectOut(
            feature_key=aspect.feature.feature_key,
            sentiment=aspect.sentiment.value,
            confidence=aspect.confidence,
            source=aspect.source.value,
        )
        for aspect in review.aspects
    ]
    return ReviewOut(
        id=review.id,
        source=review.source,
        app_id=review.app_id,
        review_text=review.review_text,
        rating=review.rating,
        review_date=review.review_date,
        aspects=aspects,
    )


@router.get("/reviews", response_model=PaginatedReviews, tags=["reviews"])
def get_reviews(
    feature_key: Optional[str] = Query(None, description="Filter by feature"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    source: Optional[str] = Query(None, description="Filter by ingestion source"),
    limit: int = Query(50, ge=1, le=DEFAULT_API_LIMIT),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
) -> PaginatedReviews:
    try:
        rows, total = queries.list_reviews(
            db,
            feature_key=feature_key,
            sentiment=sentiment,
            source=source,
            limit=limit,
            offset=offset,
        )
        return PaginatedReviews(
            total=total,
            limit=limit,
            offset=offset,
            items=[_serialize_review(review) for review in rows],
        )
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


@router.get(
    "/criticality-scores",
    response_model=List[CriticalityScoreOut],
    tags=["criticality"],
)
def get_criticality_scores(
    priority: Optional[str] = Query(None, pattern="^(high|low)$"),
    limit: int = Query(50, ge=1, le=DEFAULT_API_LIMIT),
    db: Session = Depends(get_db_session),
) -> List[CriticalityScoreOut]:
    try:
        rows = queries.list_criticality_scores(db, priority=priority, limit=limit)
        return [
            CriticalityScoreOut(
                feature_key=feature.feature_key,
                display_name=feature.display_name,
                window_start=score.window_start,
                window_end=score.window_end,
                score=score.score,
                mention_count=score.mention_count,
                negative_ratio=score.negative_ratio,
                trend_delta=score.trend_delta,
            )
            for score, feature in rows
        ]
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


@router.get(
    "/trends/{feature_key}",
    response_model=FeatureTrendOut,
    tags=["criticality"],
)
def get_feature_trend(
    feature_key: str,
    db: Session = Depends(get_db_session),
) -> FeatureTrendOut:
    try:
        result = queries.get_feature_trend(db, feature_key)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown feature_key: {feature_key!r}")

        feature, scores = result
        return FeatureTrendOut(
            feature_key=feature.feature_key,
            display_name=feature.display_name,
            points=[
                TrendPointOut(
                    window_start=score.window_start,
                    window_end=score.window_end,
                    score=score.score,
                    mention_count=score.mention_count,
                    negative_ratio=score.negative_ratio,
                )
                for score in scores
            ],
        )
    except FixFirstException:
        raise
    except HTTPException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc
