"""
FastAPI dependency for DB session injection.

Separate from db.base.get_db (a context manager used by scripts/pipelines
that auto-commits on success) because API read endpoints don't need or
want a commit on every GET request — this is a plain generator-style
dependency, the standard FastAPI pattern.
"""

from typing import Generator

from sqlalchemy.orm import Session

from fixfirst.db.base import SessionLocal


def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()