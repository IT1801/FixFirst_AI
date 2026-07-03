"""
One-off script to create the `fixfirst` Postgres schema and all tables.

Usage:
    PYTHONPATH=src python scripts/init_db.py

Requires the `db` service (Postgres) to be running — e.g. via:
    docker compose up -d db
"""

import sys

from fixfirst.db.base import init_schema_and_tables
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Initializing fixfirst schema and tables...")
        init_schema_and_tables()
        logging.info("Done.")
    except FixFirstException as e:
        logging.error(str(e))
        sys.exit(1)