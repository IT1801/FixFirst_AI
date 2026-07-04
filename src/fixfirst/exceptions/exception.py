"""Custom exception class for FixFirst AI."""

import sys
from typing import Any


def _error_message_detail(error: Any, error_detail: Any) -> str:
    """Build a traceable error message from the active exception context."""
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script: [{file_name}] at line number [{line_number}] error message: [{error}]"


class FixFirstException(Exception):
    """Application exception that preserves file and line context."""

    def __init__(self, error_message: Any, error_detail: Any = sys):
        super().__init__(str(error_message))
        self.error_message = _error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message