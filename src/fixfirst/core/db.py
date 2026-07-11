"""Database session, schema, and ORM models.

This module is the canonical database entry point. Implementation details
live in ``core._db`` to keep this public module focused.
"""


from fixfirst.constants import SCHEMA_NAME
from fixfirst.core._db.base import Base, SessionLocal, engine, get_db, init_schema_and_tables
from fixfirst.core._db.models import (
    AspectSource,
    CriticalityScore,
    Deployment,
    FeatureMaster,
    ModelRun,
    ModelTask,
    RawReview,
    ReviewAspect,
    SentimentLabel,
)

__all__ = [
    "AspectSource",
    "Base",
    "CriticalityScore",
    "Deployment",
    "FeatureMaster",
    "ModelRun",
    "ModelTask",
    "RawReview",
    "ReviewAspect",
    "SCHEMA_NAME",
    "SessionLocal",
    "SentimentLabel",
    "engine",
    "get_db",
    "init_schema_and_tables",
]
