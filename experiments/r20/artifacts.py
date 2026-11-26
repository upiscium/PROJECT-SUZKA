"""Raw-artifact handling of a real run.

Raw transcripts (the R15 protocol capture holds real model text) live in
``<base>/<run_id>/private/`` (0700 directory, 0600 files), never in the closed-key
evidence directory, never in the repository, never uploaded.  A retention note says:
kept 14 days for audit, then removed by the Human.  The closed-key manifest carries no
raw text and may be copied.
"""

from __future__ import annotations

import os
from pathlib import Path

PRIVATE_NAME = "private"
RETENTION_DAYS = 14
RETENTION_NOTE = (
    "RAW REAL-MODEL OUTPUT. Kept at most 14 days for audit, then removed by the Human.\n"
    "Never copy this directory into the repository or an issue; the closed-key manifest (no raw text)\n"
    "is what may leave this host.\n"
)


def private_dir(run_dir: Path) -> Path:
    directory = Path(run_dir) / PRIVATE_NAME
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    note = directory / "RETENTION.txt"
    if not note.exists():
        write_private(note, RETENTION_NOTE.encode("utf-8"))
    return directory


def write_private(path: Path, data: bytes) -> Path:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    return Path(path)


__all__ = ["PRIVATE_NAME", "RETENTION_DAYS", "private_dir", "write_private"]
