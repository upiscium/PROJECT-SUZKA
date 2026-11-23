"""Read-only polling of live stores (D1 sec. 4-3).

SubjectHistory and the target use ``journal_mode = DELETE``: a reader holds a
read lock that can block the server's writer.  Every poll is therefore ONE
short read transaction, opened and closed per call (``mode=ro``,
``query_only``, a small busy timeout, no held cursor).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import sqlite3
import time
from typing import Any

BUSY_TIMEOUT_S = 0.25


def short_read(
    path: Path, sql: str, params: Sequence[Any] = (), *, busy_timeout_s: float = BUSY_TIMEOUT_S
) -> list[tuple[Any, ...]]:
    """One SELECT in one read transaction on a ``mode=ro`` connection."""

    if not sql.lstrip().upper().startswith("SELECT"):
        raise ValueError("short_read runs one SELECT only")
    connection = sqlite3.connect(
        f"{Path(path).resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=busy_timeout_s,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        try:
            return connection.execute(sql, tuple(params)).fetchall()
        finally:
            connection.execute("COMMIT")
    finally:
        connection.close()


def poll_until(
    condition: Callable[[], Any],
    *,
    timeout_s: float,
    interval_s: float = 0.05,
) -> Any:
    """Call ``condition`` until it returns something truthy; a locked store
    (the server's writer) is a retry, never a failure.  Returns the value, or
    ``None`` on timeout."""

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            value = condition()
        except sqlite3.OperationalError:
            value = None
        if value:
            return value
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval_s)


__all__ = ["BUSY_TIMEOUT_S", "poll_until", "short_read"]
