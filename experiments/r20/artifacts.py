"""Raw-artifact handling of a real run (IP-23a; HD-23-10 (a)).

Raw transcripts (the R15 protocol capture holds real model text) live in
``<base>/<run_id>/private/`` (0700 directory, 0600 files), never in the closed-key
evidence directory, never in the repository, never uploaded.  A retention note
says: kept 14 days for audit, then erased by the Human with ``erase``.  The
manifest, the pre-registration and the report carry no raw text and may be copied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import time

from experiments.r20.guards import MARKER_NAME, GuardRefusal, check_not_production

PRIVATE_NAME = "private"
RETENTION_DAYS = 14
BASE_MARKER = ".r20_real_base"
RETENTION_NOTE = (
    "RAW REAL-MODEL OUTPUT. Kept at most 14 days for audit, then erased by the Human:\n"
    "  python -m experiments.r20.real_run erase --base <base> --older-than-days 14 --confirm-erase\n"
    "Never copy this directory into the repository or an issue; the manifest, the pre-registration and the\n"
    "report (no raw text) are what may leave this host.\n"
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


def mark_base(base: Path, prereg_sha256: str) -> None:
    marker = Path(base) / BASE_MARKER
    if not marker.exists():
        write_private(marker, json.dumps({"kind": "r20 real-run base", "prereg": prereg_sha256}).encode("utf-8"))
    os.chmod(base, 0o700)


def private_problems(base: Path) -> list[str]:
    """Every ``private`` directory under ``base`` must be 0700 with 0600 files."""

    problems: list[str] = []
    for private in Path(base).glob(f"*/{PRIVATE_NAME}"):
        if private.is_symlink() or (private.stat().st_mode & 0o777) != 0o700:
            problems.append(str(private))
        for item in private.rglob("*"):
            mode = item.lstat().st_mode & 0o777
            if item.is_symlink() or (item.is_dir() and mode != 0o700) or (item.is_file() and mode != 0o600):
                problems.append(str(item))
    return problems


@dataclass(frozen=True, slots=True)
class Erased:
    run_id: str
    removed: tuple[str, ...]


def erase(
    base: Path,
    *,
    run_ids: list[str] | None = None,
    older_than_days: float | None = None,
    confirmed: bool,
    now: float | None = None,
) -> list[Erased]:
    """Remove the RAW parts (subject root, private transcripts, serve log, home)
    of runs under a real-run base; the closed-key ``evidence/`` stays.  Needs the
    explicit confirmation; only a base carrying the real-run marker; never follows a link."""

    if not confirmed:
        raise GuardRefusal("ERASE_NOT_CONFIRMED", "--confirm-erase is required")
    base = Path(base)
    check_not_production(base)
    if not (base / BASE_MARKER).is_file() or base.is_symlink():
        raise GuardRefusal("ERASE_NOT_A_REAL_RUN_BASE", str(base))
    moment = time.time() if now is None else now
    done: list[Erased] = []
    for run_dir in sorted(path for path in base.iterdir() if path.is_dir() and not path.is_symlink()):
        if not (run_dir / "subject" / MARKER_NAME).is_file():
            continue
        if run_ids is not None and run_dir.name not in run_ids:
            continue
        if older_than_days is not None:
            reference = (run_dir / "evidence").stat().st_mtime if (run_dir / "evidence").exists() else run_dir.stat().st_mtime
            if moment - reference < older_than_days * 86400:
                continue
        removed: list[str] = []
        for name in ("subject", PRIVATE_NAME, "home"):
            target = run_dir / name
            if target.is_symlink():
                raise GuardRefusal("ERASE_SYMLINK", str(target))
            if target.is_dir():
                shutil.rmtree(target)
                removed.append(name)
        log = run_dir / "serve.log"
        if log.is_file():
            log.unlink()
            removed.append("serve.log")
        done.append(Erased(run_dir.name, tuple(removed)))
    return done


__all__ = [
    "BASE_MARKER", "PRIVATE_NAME", "RETENTION_DAYS", "Erased", "erase", "mark_base", "private_dir",
    "private_problems", "write_private",
]
