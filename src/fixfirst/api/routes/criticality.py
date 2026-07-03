from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from fixfirst.api.deps import get_db_session
from fixfirst.api.schemas import CriticalityScoreOut, FeatureTrendOut, TrendPointOut
from fixfirst.api import queries

router = APIRouter(tags=["criticality"])

MAX_LIMIT = 200


@router.get("/criticality-scores", response_model=List[CriticalityScoreOut])
def get_criticality_scores(
    priority: Optional[str] = Query(
        None, pattern="^(high|low)$", description="'high' = Needs Work (sorted desc), 'low' = Backlog/Stable (sorted asc)"
    ),
    limit: int = Query(50, ge=1, le=MAX_LIMIT),
    db: Session = Depends(get_db_session),
):
    """
    Returns each feature's MOST RECENT window score — this is the
    dashboard's headline view ("High Priority: Needs Work" /
    "Low Priority: Backlog / Stable"). For historical trend data on a
    single feature, use /trends/{feature_key}.
    """
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


@router.get("/trends/{feature_key}", response_model=FeatureTrendOut)
def get_feature_trend(feature_key: str, db: Session = Depends(get_db_session)):
    """
    Full historical window-by-window score series for one feature — powers
    the trend chart ("did the last deployment improve or worsen sentiment
    for this feature").
    """
    result = queries.get_feature_trend(db, feature_key)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown feature_key: {feature_key!r}")

    feature, scores = result
    points = [
        TrendPointOut(
            window_start=s.window_start,
            window_end=s.window_end,
            score=s.score,
            mention_count=s.mention_count,
            negative_ratio=s.negative_ratio,
        )
        for s in scores
    ]
    return FeatureTrendOut(feature_key=feature.feature_key, display_name=feature.display_name, points=points)