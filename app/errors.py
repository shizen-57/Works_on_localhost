"""Sanitized, controlled error responses (Problem Statement Sec. 06.1).

Every error path returns app.schemas.ErrorResponse JSON -- never a raw
stack trace, exception repr, or provider/internal detail. Callers that
need the real reason for logs use `log_and_error`, which logs server-side
at ERROR and returns a generic client-safe message.
"""

from __future__ import annotations

import logging

from fastapi.responses import JSONResponse

logger = logging.getLogger("gridwise")


def json_error(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


def log_and_error(status_code: int, error: str, public_detail: str, *, exc: Exception | None = None) -> JSONResponse:
    """Log the real exception server-side, return only `public_detail` to the caller."""
    if exc is not None:
        logger.error("%s: %s", error, exc, exc_info=exc)
    else:
        logger.error("%s: %s", error, public_detail)
    return json_error(status_code, error, public_detail)
