"""Database engine and session management for FixFirst AI."""

import sys
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from fixfirst.config.configuration import (
    ConfigurationManager,
    DatabaseConfig,
)
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

Base = declarative_base()


def create_db_engine(config: DatabaseConfig) -> Engine:
    """
    Create and configure the SQLAlchemy database engine.
    """
    try:
        logging.info("Creating PostgreSQL database engine...")

        engine = create_engine(
            config.connection_url,
            pool_pre_ping=config.pool_pre_ping,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            future=True,
        )

        logging.info("Database engine created successfully.")
        return engine

    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


def create_session_factory(engine: Engine) -> sessionmaker:
    """
    Create a SQLAlchemy session factory.
    """
    try:
        return sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )
    except FixFirstException:
        raise
    except Exception as exc:
        raise FixFirstException(exc, sys) from exc


# ---------------------------------------------------------------------
# Initialize shared database objects
# ---------------------------------------------------------------------

database_config = ConfigurationManager().get_database_config()

engine = create_db_engine(database_config)

SessionLocal = create_session_factory(engine)


@contextmanager
def get_db():
    """
    Yield a transactional SQLAlchemy session.

    Automatically commits on success,
    rolls back on failure,
    and always closes the session.
    """
    db = SessionLocal()

    try:
        yield db
        db.commit()

    except FixFirstException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()
        logging.error(f"Database transaction rolled back: {exc}")
        raise FixFirstException(exc, sys) from exc

    finally:
        db.close()


def init_schema_and_tables() -> None:
    """
    Create the configured schema and all ORM tables
    if they do not already exist.
    """
    try:
        logging.info(
            f"Initializing database schema '{database_config.schema}'..."
        )

        with engine.connect() as connection:
            connection.execute(
                text(
                    f"CREATE SCHEMA IF NOT EXISTS {database_config.schema}"
                )
            )
            connection.commit()
        from fixfirst.core._db import models
        
        Base.metadata.create_all(bind=engine)
        
        logging.info("Database schema and tables initialized successfully.")

    except FixFirstException:
        raise

    except Exception as exc:
        raise FixFirstException(exc, sys) from exc