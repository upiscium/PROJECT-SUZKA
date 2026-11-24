"""The Human's caps and the run ledger (IP-23a; HD-23-2 (a), the 2026-11-24 decision).

Caps: at most 42 subjects, 10 h of real-run wall clock in total, 2.5 h per
session (one launch), 120 s per turn, >= 14 GiB free GPU memory at launch.
They are CONSTANTS here; a command-line value can only LOWER a cap (a larger
one is a refusal), so no launch flag raises a cap.  Exceeding a cap stops the
run and asks the Human: nothing is extended silently, and a stopped attempt
still counts (no silent restart).

The ledger (``<base>/ledger.jsonl``, 0600, append-only lines) is what makes the
subject and hour totals hold ACROSS launches; the pre-registration and the
manifests copy the numbers, the ledger is the running count.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

from experiments.r20.gpu import MIN_FREE_BYTES
from experiments.r20.guards import GuardRefusal

HOUR = 3600.0
HARD_MAX_SUBJECTS = 42
HARD_TOTAL_SECONDS = 10 * HOUR
HARD_SESSION_SECONDS = 2.5 * HOUR
HARD_TURN_SECONDS = 120.0
# Subjects per stage of the plan sec. 4: G1 1; S0 <= 3 + 1 negative variant; G2 1; S1 <= 2; G3 <= 4;
# F1 24 + retries <= 6.  The sum (1 + 4 + 1 + 2 + 4 + 30 = 42) is the total cap.
STAGE_LIMITS = {"G1": 1, "S0": 4, "G2": 1, "S1": 2, "G3": 4, "F1": 30}
LEDGER_NAME = "ledger.jsonl"


@dataclass(frozen=True, slots=True)
class Caps:
    max_subjects: int = HARD_MAX_SUBJECTS
    total_seconds: float = HARD_TOTAL_SECONDS
    session_seconds: float = HARD_SESSION_SECONDS
    turn_seconds: float = HARD_TURN_SECONDS
    min_free_bytes: int = MIN_FREE_BYTES

    def __post_init__(self) -> None:
        if self.max_subjects < 1 or self.max_subjects > HARD_MAX_SUBJECTS:
            raise GuardRefusal("CAP_ABOVE_LIMIT", f"max_subjects {self.max_subjects} > {HARD_MAX_SUBJECTS}")
        if not 0 < self.total_seconds <= HARD_TOTAL_SECONDS:
            raise GuardRefusal("CAP_ABOVE_LIMIT", f"total hours {self.total_seconds / HOUR} > {HARD_TOTAL_SECONDS / HOUR}")
        if not 0 < self.session_seconds <= HARD_SESSION_SECONDS:
            raise GuardRefusal("CAP_ABOVE_LIMIT", f"session hours {self.session_seconds / HOUR} > {HARD_SESSION_SECONDS / HOUR}")
        if not 0 < self.turn_seconds <= HARD_TURN_SECONDS:
            raise GuardRefusal("CAP_ABOVE_LIMIT", f"turn seconds {self.turn_seconds} > {HARD_TURN_SECONDS}")
        if self.min_free_bytes < MIN_FREE_BYTES:
            raise GuardRefusal("CAP_ABOVE_LIMIT", "the free-GPU minimum cannot be lowered below 14 GiB")

    def record(self) -> dict[str, object]:
        return {**asdict(self), "stage_limits": dict(STAGE_LIMITS)}


class Ledger:
    """Append-only counts of subjects started and wall-clock seconds used."""

    def __init__(self, base: Path) -> None:
        self.path = Path(base) / LEDGER_NAME

    def _append(self, row: dict[str, object]) -> None:
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def rows(self) -> list[dict[str, object]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    def subjects_used(self, stage: str | None = None) -> int:
        return sum(
            1 for row in self.rows() if row.get("kind") == "subject" and (stage is None or row.get("stage") == stage)
        )

    def seconds_used(self) -> float:
        """Per launch the LARGEST elapsed time recorded (a killed launcher
        still counts what it had recorded)."""

        best: dict[str, float] = {}
        for row in self.rows():
            if row.get("kind") in {"tick", "subject", "end"}:
                launch = str(row.get("launch"))
                best[launch] = max(best.get(launch, 0.0), float(str(row.get("elapsed_s", 0.0))))
        return sum(best.values())

    def append(self, **row: object) -> None:
        self._append(row)


class Session:
    """One launch's budget.  Every subject start goes through ``begin_subject``;
    every long wait calls ``check``."""

    def __init__(
        self, ledger: Ledger, caps: Caps, launch_id: str, stage: str, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if stage not in STAGE_LIMITS:
            raise GuardRefusal("STAGE_UNKNOWN", stage)
        self.ledger = ledger
        self.caps = caps
        self.launch_id = launch_id
        self.stage = stage
        self._clock = clock
        self._started = clock()
        self._prior = ledger.seconds_used()
        ledger.append(kind="start", launch=launch_id, stage=stage, elapsed_s=0.0)

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    def check(self) -> None:
        if self.elapsed >= self.caps.session_seconds:
            raise GuardRefusal("CAP_SESSION_HOURS", f"{self.elapsed / HOUR:.2f} h")
        if self._prior + self.elapsed >= self.caps.total_seconds:
            raise GuardRefusal("CAP_TOTAL_HOURS", f"{(self._prior + self.elapsed) / HOUR:.2f} h")
        self.ledger.append(kind="tick", launch=self.launch_id, elapsed_s=round(self.elapsed, 1))

    def over(self) -> str | None:
        """The cap code this session has reached, or ``None`` (no ledger write)."""

        if self.elapsed >= self.caps.session_seconds:
            return "CAP_SESSION_HOURS"
        if self._prior + self.elapsed >= self.caps.total_seconds:
            return "CAP_TOTAL_HOURS"
        return None

    def begin_subject(self, run_id: str, *, stage: str | None = None) -> None:
        """Count one subject BEFORE it is created (a stopped attempt counts)."""

        named = stage or self.stage
        self.check()
        if self.ledger.subjects_used() >= self.caps.max_subjects:
            raise GuardRefusal("CAP_MAX_SUBJECTS", str(self.caps.max_subjects))
        if named not in STAGE_LIMITS or self.ledger.subjects_used(named) >= STAGE_LIMITS[named]:
            raise GuardRefusal("CAP_STAGE_SUBJECTS", f"{named} <= {STAGE_LIMITS.get(named)}")
        self.ledger.append(kind="subject", launch=self.launch_id, stage=named, run_id=run_id,
                           elapsed_s=round(self.elapsed, 1))

    def end(self, reason: str) -> None:
        self.ledger.append(kind="end", launch=self.launch_id, reason=reason, elapsed_s=round(self.elapsed, 1))


__all__ = [
    "HARD_MAX_SUBJECTS", "HARD_SESSION_SECONDS", "HARD_TOTAL_SECONDS", "HARD_TURN_SECONDS", "STAGE_LIMITS",
    "Caps", "Ledger", "Session",
]
