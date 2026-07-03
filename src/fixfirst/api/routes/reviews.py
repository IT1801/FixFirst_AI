from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from fixfirst.api.deps import get_db_session
from fixfirst.api.schemas import PaginatedReviews, ReviewAspectOut, ReviewOut
from fixfirst.api import queries
from fixfirst.db.models import RawReview

router = APIRouter(prefix="/reviews", tags=["reviews"])

MAX_LIMIT = 200


def _serialize_review(review: RawReview) -> ReviewOut:
    """
    Manual serialization rather than pure from_attributes: ReviewAspect's
    ORM relationship exposes `.feature` (a FeatureMaster object), not a
    flat `feature_key` string — the API schema flattens that for a
    simpler client-facing shape, so this mapping step is necessary.
    """
    aspects = [
        ReviewAspectOut(
            feature_key=a.feature.feature_key,
            sentiment=a.sentiment.value,
            confidence=a.confidence,
            source=a.source.value,
        )
        for a in review.aspects
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


@router.get("", response_model=PaginatedReviews)
def get_reviews(
    feature_key: Optional[str] = Query(None, description="Filter to reviews discussing this feature"),
    sentiment: Optional[str] = Query(None, description="Filter to reviews with this sentiment for the given feature"),
    source: Optional[str] = Query(None, description="Filter by ingestion source, e.g. 'aware'"),
    limit: int = Query(50, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db_session),
):
    rows, total = queries.list_reviews(
        db, feature_key=feature_key, sentiment=sentiment, source=source, limit=limit, offset=offset
    )
    return PaginatedReviews(
        total=total,
        limit=limit,
        offset=offset,
        items=[_serialize_review(r) for r in rows],
    )