"""
Logging configuration for FixFirst AI.

Convention (consistent across the codebase):
    from fixfirst.logging.logger import logging
    logging.info("message")

Logs to both console and a timestamped file under logs/, formatted with
module, line number, and level for traceability across the pipeline.
"""

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(lineno)d - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(),
    ],
)