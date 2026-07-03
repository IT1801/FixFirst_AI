"""
Database engine and session management for FixFirst AI.

Convention:
    from fixfirst.db.base import Base, engine, SessionLocal, get_db

All ORM models live under the `fixfirst` Postgres schema (see models.py),
mirroring the customer-segmentation-retention project's use of a dedicated
`retail` schema rather than the public schema.
"""

import sys
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from fixfirst.config.settings import settings
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

SCHEMA_NAME = "fixfirst"

Base = declarative_base()

try:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )
except Exception as e:
    raise FixFirstException(e, sys)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def get_db():
    """
    Context-managed DB session.

    Usage:
        with get_db() as db:
            db.query(...)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"DB session rolled back due to error: {e}")
        raise FixFirstException(e, sys)
    finally:
        db.close()


def init_schema_and_tables() -> None:
    """
    Creates the `fixfirst` schema (if missing) and all tables registered
    on Base.metadata. Safe to call repeatedly — idempotent.
    """
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
            conn.commit()

        # Import models so they register on Base.metadata before create_all
        from fixfirst.db import models  # noqa: F401

        Base.metadata.create_all(bind=engine)
        logging.info(f"Schema '{SCHEMA_NAME}' and all tables created/verified successfully.")
    except Exception as e:
        raise FixFirstException(e, sys)