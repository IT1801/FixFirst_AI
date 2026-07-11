"""FastAPI dependency for DB session injection."""

from typing import Generator

from sqlalchemy.orm import Session

from fixfirst.core.db import SessionLocal


def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session for request-scoped API work."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()