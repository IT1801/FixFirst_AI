"""
ORM models for FixFirst AI — six tables under the `fixfirst` Postgres schema.

    raw_reviews         Ingested reviews, as-is, from any source.
    features_master     Curated feature/aspect taxonomy (login, sync, etc).
    review_aspects      Per-review, per-feature sentiment (the ABSA output).
    criticality_scores  Aggregated per-feature score over a time window.
    deployments         Release log, used to correlate trend shifts.
    model_runs          MLflow run metadata mirrored into Postgres.
"""

import enum
import uuid

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    Date,
    ForeignKey,
    JSON,
    Enum,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from fixfirst.db.base import Base, SCHEMA_NAME


class SentimentLabel(str, enum.Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class AspectSource(str, enum.Enum):
    finetuned = "finetuned"
    llm_fallback = "llm_fallback"


class ModelTask(str, enum.Enum):
    aspect_category = "aspect_category"
    aspect_sentiment = "aspect_sentiment"


class RawReview(Base):
    __tablename__ = "raw_reviews"
    __table_args__ = (
        Index("ix_raw_reviews_app_id", "app_id"),
        Index("ix_raw_reviews_review_date", "review_date"),
        {"schema": SCHEMA_NAME},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(50), nullable=False)  # e.g. 'aware', 'google_play', 'app_store', 'github_issues'
    app_id = Column(String(255), nullable=False)
    review_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=True)  # 1-5 stars, nullable since not all sources have it
    review_date = Column(Date, nullable=True)
    raw_metadata = Column(JSON, nullable=True)  # original payload for traceability
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())

    aspects = relationship("ReviewAspect", back_populates="review", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<RawReview id={self.id} source={self.source} app_id={self.app_id}>"


class FeatureMaster(Base):
    __tablename__ = "features_master"
    __table_args__ = (
        UniqueConstraint("feature_key", name="uq_features_master_feature_key"),
        {"schema": SCHEMA_NAME},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feature_key = Column(String(100), nullable=False)  # e.g. 'login_auth', 'sync_speed'
    display_name = Column(String(150), nullable=False)  # e.g. 'Login / Authentication'
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    aspects = relationship("ReviewAspect", back_populates="feature")
    criticality_scores = relationship("CriticalityScore", back_populates="feature")

    def __repr__(self) -> str:
        return f"<FeatureMaster key={self.feature_key} name={self.display_name}>"


class ReviewAspect(Base):
    __tablename__ = "review_aspects"
    __table_args__ = (
        Index("ix_review_aspects_review_id", "review_id"),
        Index("ix_review_aspects_feature_id", "feature_id"),
        {"schema": SCHEMA_NAME},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(UUID(as_uuid=True), ForeignKey(f"{SCHEMA_NAME}.raw_reviews.id"), nullable=False)
    feature_id = Column(UUID(as_uuid=True), ForeignKey(f"{SCHEMA_NAME}.features_master.id"), nullable=False)
    sentiment = Column(Enum(SentimentLabel, name="sentiment_label"), nullable=False)
    confidence = Column(Float, nullable=False)  # model confidence for this prediction, 0-1
    source = Column(Enum(AspectSource, name="aspect_source"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    review = relationship("RawReview", back_populates="aspects")
    feature = relationship("FeatureMaster", back_populates="aspects")

    def __repr__(self) -> str:
        return f"<ReviewAspect review={self.review_id} feature={self.feature_id} sentiment={self.sentiment}>"


class CriticalityScore(Base):
    __tablename__ = "criticality_scores"
    __table_args__ = (
        Index("ix_criticality_scores_feature_window", "feature_id", "window_start", "window_end"),
        {"schema": SCHEMA_NAME},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feature_id = Column(UUID(as_uuid=True), ForeignKey(f"{SCHEMA_NAME}.features_master.id"), nullable=False)
    window_start = Column(Date, nullable=False)
    window_end = Column(Date, nullable=False)
    score = Column(Float, nullable=False)
    mention_count = Column(Integer, nullable=False)
    negative_ratio = Column(Float, nullable=False)
    trend_delta = Column(Float, nullable=True)  # change vs. previous window, null for first window
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    feature = relationship("FeatureMaster", back_populates="criticality_scores")

    def __repr__(self) -> str:
        return f"<CriticalityScore feature={self.feature_id} score={self.score} window=({self.window_start}..{self.window_end})>"


class Deployment(Base):
    __tablename__ = "deployments"
    __table_args__ = (
        Index("ix_deployments_deploy_date", "deploy_date"),
        {"schema": SCHEMA_NAME},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(50), nullable=False)
    deploy_date = Column(Date, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<Deployment version={self.version} date={self.deploy_date}>"


class ModelRun(Base):
    __tablename__ = "model_runs"
    __table_args__ = (
        Index("ix_model_runs_mlflow_run_id", "mlflow_run_id"),
        {"schema": SCHEMA_NAME},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mlflow_run_id = Column(String(64), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    task = Column(Enum(ModelTask, name="model_task"), nullable=False)
    metrics = Column(JSON, nullable=True)  # precision/recall/F1 etc, mirrored from MLflow
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<ModelRun mlflow_run_id={self.mlflow_run_id} task={self.task}>"