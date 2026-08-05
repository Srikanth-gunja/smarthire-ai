"""Small, dependency-free execution logging for SmartHire.

Logs are intentionally concise and avoid raw resume/JD/chat content.  They
are emitted to the terminal that runs ``streamlit run app.py``.
"""

from __future__ import annotations

import functools
import logging
from time import perf_counter
from typing import Any, Callable, TypeVar


logger = logging.getLogger("smarthire.execution")
F = TypeVar("F", bound=Callable[..., Any])


def timed_operation(kind: str, name: str | None = None) -> Callable[[F], F]:
    """Log start, finish, failure, and elapsed time for an operation."""

    def decorate(func: F) -> F:
        operation = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            logger.info("event=start kind=%s operation=%s", kind, operation)
            try:
                result = func(*args, **kwargs)
            except Exception:
                elapsed_ms = (perf_counter() - started) * 1000
                logger.exception(
                    "event=failed kind=%s operation=%s duration_ms=%.1f",
                    kind,
                    operation,
                    elapsed_ms,
                )
                raise
            elapsed_ms = (perf_counter() - started) * 1000
            logger.info(
                "event=complete kind=%s operation=%s duration_ms=%.1f",
                kind,
                operation,
                elapsed_ms,
            )
            return result

        return wrapped  # type: ignore[return-value]

    return decorate


def instrument_tool_methods(cls: type) -> type:
    """Instrument every public instance method on a tool class."""
    for method_name, method in vars(cls).items():
        if method_name.startswith("_") or method_name == "__init__" or not callable(method):
            continue
        setattr(cls, method_name, timed_operation("tool", f"{cls.__name__}.{method_name}")(method))
    return cls
