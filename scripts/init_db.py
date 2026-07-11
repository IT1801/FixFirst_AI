"""One-off script to create the FixFirst schema and all tables."""

import sys

from fixfirst.core.db import init_schema_and_tables
from fixfirst.exceptions.exception import FixFirstException
from fixfirst.logging.logger import logging


def main() -> int:
    """Create the database schema and tables."""
    try:
        logging.info("Initializing fixfirst schema and tables...")
        init_schema_and_tables()
        logging.info("Done.")
        return 0
    except FixFirstException as exc:
        logging.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
