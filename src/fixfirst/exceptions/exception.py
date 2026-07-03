"""
Custom exception class for FixFirst AI.

Convention (consistent across the codebase):
    from fixfirst.exceptions.exception import FixFirstException
    try:
        ...
    except Exception as e:
        raise FixFirstException(e, sys)

Captures the originating file name and line number so errors are traceable
across ingestion, labeling, training, inference, scoring, and API layers.
"""

import sys


def _error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error occurred in script: [{file_name}] "
        f"at line number [{line_number}] "
        f"error message: [{error}]"
    )


class FixFirstException(Exception):
    def __init__(self, error_message, error_detail: sys = sys):
        super().__init__(str(error_message))
        self.error_message = _error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message