"""F15 degraded-continuation primitives (#411, F15 slice D1 r3, DC-1).

Everything here is INERT until a later unit calls it: nothing in the server
start path, the Runtime or the participants reads these primitives yet (the
supervisor is DC-5, the participant quarantine is DC-4, the HARD ladder is
DC-6).  The module offers:

* ``IntegrityVerdict`` and the classifiers, a pure function of the exception
  CLASS name (and, for the memory domains, the participant id).  Unknown is
  ``PROTECTED_LOST``: never quarantinable.
* ``QuarantineWriter``: a damaged artifact is moved by ``os.rename`` to a
  sibling ``<name>.quarantine-<epoch_id>`` in the same parent.  It is never
  repaired, rewritten, copied, truncated or deleted; there is no copy or
  delete fallback because there is nothing to fall back to (same filesystem by
  construction).
* the four-state manifest PENDING -> MOVED -> JOURNALED -> COMMITTED and its
  crash-resume planner.  The manifest is a sidecar, never an authority.
* ``Deadline``, the probe / back-off schedules and the status record behind
  the ``dc_*`` health fields.

Known limits (stated, not hidden):

* ``T_hash`` bounds hashing between syscalls.  One ``open`` / ``read`` /
  ``scandir`` / ``rename`` / ``fsync`` on a hung filesystem is not
  interruptible, so the bound is not a hard wall-clock guarantee there.
* ``os.rename`` of a directory replaces an EMPTY directory that appears at the
  target between the existence check and the rename (no ``RENAME_NOREPLACE``
  in the standard library).  No artifact data can be lost that way.
* A busy lease and any other ``OSError`` of ``EventJournalLease`` share one
  exception class and message, so both are labelled ``event_journal_lease_busy``
  (TRANSIENT); only the label is affected, nothing is quarantined on it.
* A post-move check after the hash budget ran out compares paths, kinds and
  sizes only; the manifest then records ``verified: null``, not ``true``.
* DC-4 must create the empty root together with its marker atomically (temp
  directory, then rename): a root without the marker is ``BLOCKED`` here.

Prohibitions kept here (D1 sec. 10): no ``unlink`` / ``rmtree`` / truncate /
copy anywhere in this module, writes only create NEW files (``O_EXCL``), no
path under a ``*.quarantine-*`` component is ever written, and no quarantined
tree is ever opened for writing.  ``tests/test_dc_architecture.py`` checks
this by AST.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from suzka.runtime.event_journal import EventJournalAdmissionBand
from suzka.runtime.thread_liveness import ThreadLedger, ThreadsAlive

__all__ = [
    "BackoffSchedule",
    "CycleFailure",
    "Deadline",
    "DegradedContinuationStatus",
    "DcState",
    "EPOCH_MARKER_NAME",
    "IntegrityVerdict",
    "MANIFEST_SCHEMA",
    "ManifestError",
    "ManifestState",
    "QUARANTINE_NAME_INFIX",
    "QuarantineArtifact",
    "QuarantineError",
    "QuarantineManifest",
    "QuarantineRefused",
    "QuarantineWriter",
    "ResumeAction",
    "ResumeFacts",
    "ResumePlan",
    "ThreadLedger",
    "ThreadsAlive",
    "classify_cycle_exception",
    "classify_participant_error",
    "is_quarantinable",
    "must_propagate",
    "plan_resume",
    "probe_schedule",
    "reeval_delay",
    "restart_delay",
    "verdict_for_band",
]

# --------------------------------------------------------------------------
# Verdicts (D1 sec. 3)
# --------------------------------------------------------------------------


class IntegrityVerdict(str, Enum):
    OK = "OK"
    TRANSIENT = "TRANSIENT"
    DIVERGED = "DIVERGED"
    AMBIGUOUS = "AMBIGUOUS"
    PROTECTED_LOST = "PROTECTED_LOST"
    BAND_HARD = "BAND_HARD"


MEMORY_EXPERIENCE_PARTICIPANT = "memory.experience"
MEMORY_SEMANTIC_PARTICIPANT = "memory.semantic"
# Only these two participants may ever be quarantined (HD-F15S-1, D1 sec. 4.2).
# ``subject.history``, ``memory.episodic`` and ``session.turn`` never are.
QUARANTINABLE_PARTICIPANTS: frozenset[str] = frozenset(
    {MEMORY_EXPERIENCE_PARTICIPANT, MEMORY_SEMANTIC_PARTICIPANT}
)

# Qualified ``module.Class`` names; an EXACT match on the raised class, never a
# subclass match: a subclass added later is unknown, hence ``PROTECTED_LOST``.
_PARTICIPANT_UNAVAILABLE = (
    "suzka.runtime.transaction_coordinator.ParticipantUnavailableError"
)
_PARTICIPANT_DIVERGED = "suzka.runtime.transaction_coordinator.ParticipantDivergedError"
# New in DC-4 (a subclass of ``UnsupportedParticipantReconciliationError``); the
# classifier matches it by name already so DC-4 needs no change here.
_PARTICIPANT_EVIDENCE_ABSENT = (
    "suzka.runtime.transaction_coordinator.ParticipantEvidenceAbsentError"
)
_SEMANTIC_STORE_UNAVAILABLE = "suzka.memory.semantic_store.SemanticStoreUnavailable"
_EXPERIENCE_STORE_UNAVAILABLE = (
    "suzka.memory.experience_store.ExperienceStoreUnavailable"
)
_STARTUP_RECONCILIATION_ERROR = (
    "suzka.runtime.startup_reconciliation.StartupReconciliationError"
)
_EVENT_JOURNAL_LOAD_ERROR = "suzka.runtime.event_journal.EventJournalLoadError"

# ``EventJournalLease`` reports a busy lease as a plain ``EventJournalLoadError``
# with this message (``event_journal.py``, ``EventJournalLease.__init__``); the
# class alone cannot tell it from a corrupt Journal.  Pinned by a test that
# provokes the real raise.
LEASE_BUSY_MESSAGE = "EventJournal exclusive authority is unavailable"
LEASE_BUSY_LABEL = "event_journal_lease_busy"

#: Names the table refers to that may not exist yet (added by a later unit).
PENDING_CLASS_NAMES: frozenset[str] = frozenset({_PARTICIPANT_EVIDENCE_ABSENT})

_TRANSIENT_CLASS_NAMES: frozenset[str] = frozenset(
    {
        _PARTICIPANT_UNAVAILABLE,
        _SEMANTIC_STORE_UNAVAILABLE,
        _EXPERIENCE_STORE_UNAVAILABLE,
    }
)
_DIVERGED_CLASS_NAMES: frozenset[str] = frozenset(
    {_PARTICIPANT_DIVERGED, _PARTICIPANT_EVIDENCE_ABSENT}
)
# Commit evidence that cannot be told apart is raised as this class
# (``reconcile_open_transactions``); it is also raised for other reconciliation
# refusals.  Both are non-quarantinable and re-evaluated, so the coarse name is
# the safe reading.
_AMBIGUOUS_CLASS_NAMES: frozenset[str] = frozenset({_STARTUP_RECONCILIATION_ERROR})

#: ``classify_*`` table keys, for the table-resolves test and DC-4/5.
CLASSIFIED_CLASS_NAMES: frozenset[str] = (
    _TRANSIENT_CLASS_NAMES
    | _DIVERGED_CLASS_NAMES
    | _AMBIGUOUS_CLASS_NAMES
    | frozenset({_EVENT_JOURNAL_LOAD_ERROR})
)


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_builtin_os_error(error: BaseException) -> bool:
    """The ``OSError`` family (``FileNotFoundError``, ``PermissionError``,
    ``TimeoutError``...) raised as a BUILTIN class.  A suzka subclass of
    ``OSError`` is unknown, so it is not matched here."""

    return type(error).__module__ == "builtins" and isinstance(error, OSError)


def _is_lease_busy(error: BaseException) -> bool:
    return (
        _qualified_name(type(error)) == _EVENT_JOURNAL_LOAD_ERROR
        and str(error) == LEASE_BUSY_MESSAGE
    )


@dataclass(frozen=True)
class CycleFailure:
    """``label`` is what ``dc_reason`` shows: the qualified class name, or a
    short reason for a case the class name cannot name (lease busy)."""

    label: str
    verdict: IntegrityVerdict


def must_propagate(error: BaseException) -> bool:
    """Only these are not a failure of the cycle: shutdown, exit and
    cancellation (``asyncio.CancelledError`` is a ``BaseException``).  Every
    other ``Exception`` is classified and ends in a re-evaluated state."""

    return not isinstance(error, Exception)


def classify_cycle_exception(error: BaseException) -> CycleFailure:
    """Classify an exception that ended a cycle build, at the first start or at
    a restart.  Total: every exception class gets a verdict and a label, none
    is left undefined, none raises.  Unknown is ``PROTECTED_LOST``.

    Without a participant id nothing is ``DIVERGED``: a divergence verdict is
    for one participant (``classify_participant_error``)."""

    name = _qualified_name(type(error))
    if _is_lease_busy(error):
        return CycleFailure(LEASE_BUSY_LABEL, IntegrityVerdict.TRANSIENT)
    if name in _AMBIGUOUS_CLASS_NAMES:
        return CycleFailure(name, IntegrityVerdict.AMBIGUOUS)
    if name in _TRANSIENT_CLASS_NAMES or _is_builtin_os_error(error):
        return CycleFailure(name, IntegrityVerdict.TRANSIENT)
    return CycleFailure(name, IntegrityVerdict.PROTECTED_LOST)


def classify_participant_error(
    participant_id: str, error: BaseException
) -> IntegrityVerdict:
    """Verdict of an error one participant raised during reconciliation.

    ``DIVERGED`` (quarantinable) only for ``memory.experience`` /
    ``memory.semantic`` and only for ``ParticipantDivergedError`` or
    ``ParticipantEvidenceAbsentError``.  ``TRANSIENT`` only for the same two
    participants.  Everything else, including every error from
    ``memory.episodic`` / ``subject.history`` / ``session.turn`` and every
    unknown class, is ``PROTECTED_LOST``."""

    if participant_id not in QUARANTINABLE_PARTICIPANTS:
        return IntegrityVerdict.PROTECTED_LOST
    name = _qualified_name(type(error))
    if name in _DIVERGED_CLASS_NAMES:
        return IntegrityVerdict.DIVERGED
    if name in _TRANSIENT_CLASS_NAMES or _is_builtin_os_error(error):
        return IntegrityVerdict.TRANSIENT
    return IntegrityVerdict.PROTECTED_LOST


def verdict_for_band(band: object) -> IntegrityVerdict:
    """``BAND_HARD`` when the Journal admission band is HARD, else ``OK``.  An
    unknown value is ``PROTECTED_LOST``, never ``OK``."""

    if band is EventJournalAdmissionBand.HARD:
        return IntegrityVerdict.BAND_HARD
    if isinstance(band, EventJournalAdmissionBand):
        return IntegrityVerdict.OK
    return IntegrityVerdict.PROTECTED_LOST


def is_quarantinable(
    participant_id: str,
    verdict: IntegrityVerdict,
    *,
    quarantine_unavailable: bool = False,
) -> bool:
    """``DIVERGED`` of a memory-domain participant only.  ``TRANSIENT`` is
    quarantinable only when the Human chose HD-new-2 (b)/(c)
    (``quarantine_unavailable``), which the caller derives from settings and
    elapsed time; the default is never."""

    if participant_id not in QUARANTINABLE_PARTICIPANTS:
        return False
    if verdict is IntegrityVerdict.DIVERGED:
        return True
    return quarantine_unavailable and verdict is IntegrityVerdict.TRANSIENT


# --------------------------------------------------------------------------
# Bounded time (D1 sec. 3)
# --------------------------------------------------------------------------

#: Default schedules (seconds).  Not measured (D1 sec. 14).
RESTART_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 4.0, 16.0, 60.0)
REEVAL_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 4.0, 16.0, 60.0, 300.0)
T_TERM_GRACE_SECONDS = 30.0
T_LADDER_RUNG_SECONDS = 60.0

BackoffSchedule = tuple[float, ...]


class Deadline:
    """A shared monotonic budget (``T_retry``, ``T_hash``): every user draws on
    the same instant, so the total stays bounded however many users there are."""

    def __init__(
        self, budget_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not budget_s >= 0:
            raise ValueError("budget_s must be zero or more")
        self._clock = clock
        self._end = clock() + budget_s

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def expired(self) -> bool:
        return self._clock() >= self._end


def probe_schedule(total_s: float) -> tuple[float, ...]:
    """Doubling probe delays 1, 2, 4... whose SUM is exactly ``total_s``
    (60 s -> 1, 2, 4, 8, 16, 29).  The SLEEPS add up to ``total_s``; the time
    an attempt itself takes is not in it, so a caller that wants a hard end
    must also check a ``Deadline`` after each attempt."""

    if not total_s > 0:
        return ()
    delays: list[float] = []
    remaining = float(total_s)
    step = 1.0
    while step < remaining:
        delays.append(step)
        remaining -= step
        step *= 2
    delays.append(remaining)
    return tuple(delays)


def _schedule_delay(
    schedule: BackoffSchedule, attempt: int, cap_s: float | None
) -> float:
    if attempt < 0:
        raise ValueError("attempt must be zero or more")
    delay = schedule[min(attempt, len(schedule) - 1)]
    return delay if cap_s is None else min(delay, cap_s)


def restart_delay(attempt: int) -> float:
    """Back-off before restart ``attempt`` (0-based): 1, 4, 16, 60, 60..."""

    return _schedule_delay(RESTART_BACKOFF_SECONDS, attempt, None)


def reeval_delay(attempt: int, max_backoff_s: float = 300.0) -> float:
    """Re-evaluation delay (0-based): 1, 4, 16, 60, 300, then every 300 s,
    never above ``max_backoff_s``."""

    return _schedule_delay(REEVAL_BACKOFF_SECONDS, attempt, max_backoff_s)


# --------------------------------------------------------------------------
# Health fields (D1 sec. 5.1)
# --------------------------------------------------------------------------


class DcState(str, Enum):
    INACTIVE = "INACTIVE"  # nothing wired in: the legacy behaviour
    STARTING = "STARTING"
    SERVING = "SERVING"
    RESTARTING = "RESTARTING"
    RECONCILING = "RECONCILING"
    REFUSED_RUNNING = "REFUSED_RUNNING"
    RESTART_LIMIT = "RESTART_LIMIT"
    HARD_WAIT = "HARD_WAIT"
    HARD_LADDER = "HARD_LADDER"
    HARD_STUCK = "HARD_STUCK"


_DC_REASON_MAX = 200


def _bounded(value: str) -> str:
    return value if len(value) <= _DC_REASON_MAX else value[:_DC_REASON_MAX]


class DegradedContinuationStatus:
    """Thread-safe record behind ``dc_state`` / ``dc_reason`` / ``dc_since`` /
    ``dc_participants`` / ``dc_epoch_id``.  Read-only for ``/health`` and
    ``/health/dc``: it holds no reference to ``app.state`` or to any store, so
    reading it works in every phase.  Values are short strings (``/health`` is
    ``dict[str, str]``) and never carry a path, a payload or an exception
    message."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._state = DcState.INACTIVE
        self._reason: str | None = None
        self._since: datetime | None = None
        self._participants: tuple[str, ...] = ()
        self._epoch_id: str | None = None

    def publish(
        self,
        state: DcState,
        *,
        reason: str | None = None,
        participants: Iterable[str] = (),
        epoch_id: str | None = None,
    ) -> None:
        participants = tuple(sorted(set(participants)))
        with self._lock:
            changed = (state, reason) != (self._state, self._reason)
            self._state = state
            self._reason = None if reason is None else _bounded(reason)
            self._participants = participants
            self._epoch_id = epoch_id
            if changed or self._since is None:
                self._since = self._clock()

    def snapshot(self) -> dict[str, str]:
        """Every field, for ``GET /health/dc``.  ``dc_state`` is always there."""

        with self._lock:
            fields = {"dc_state": self._state.value}
            if self._reason:
                fields["dc_reason"] = self._reason
            if self._since is not None:
                fields["dc_since"] = self._since.isoformat()
            if self._participants:
                fields["dc_participants"] = ",".join(self._participants)
            if self._epoch_id:
                fields["dc_epoch_id"] = self._epoch_id
            return fields

    def health_fields(self) -> dict[str, str]:
        """What ``/health`` adds.  Nothing while inactive, so a deployment
        that wires nothing in answers exactly as before; nothing for a plain
        ``SERVING`` either.  Present for every other state and for a
        quarantined domain."""

        with self._lock:
            quiet = self._state in (DcState.INACTIVE, DcState.SERVING) and not (
                self._reason or self._participants
            )
        return {} if quiet else self.snapshot()


# --------------------------------------------------------------------------
# Quarantine (D1 sec. 4.2 steps 1-2, 5.1)
# --------------------------------------------------------------------------

MANIFEST_SCHEMA = "suzka.dc.manifest.v1"
MANIFEST_NAME = "manifest.json"
#: Marker DC-4 puts in the empty root it creates after a quarantine.  DC-1 only
#: reads it, to plan a resume.
EPOCH_MARKER_NAME = ".dc_epoch.json"
QUARANTINE_NAME_INFIX = ".quarantine-"
_EPOCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CHUNK = 1 << 20


class QuarantineError(Exception):
    """A quarantine step failed.  Whatever was moved stays where it is and
    nothing was deleted; the caller ends in a re-evaluated state."""


class QuarantineRefused(QuarantineError):
    """The action was refused BEFORE it changed anything on disk."""


class ManifestError(QuarantineError):
    """A manifest is missing, malformed or asked to move against its order."""


class ManifestState(str, Enum):
    PENDING = "PENDING"
    MOVED = "MOVED"
    JOURNALED = "JOURNALED"
    COMMITTED = "COMMITTED"


_STATE_ORDER = (
    ManifestState.PENDING,
    ManifestState.MOVED,
    ManifestState.JOURNALED,
    ManifestState.COMMITTED,
)


@dataclass(frozen=True)
class QuarantineArtifact:
    """One quarantined tree.  ``entries`` are ``(relpath, kind, size, sha256)``
    sorted by relpath; ``sha256`` is None for a directory, a symlink (recorded,
    never followed), an unreadable file and a file the hash budget did not
    reach (``unhashed``).  Hashes and sizes only, never payload bytes."""

    origin_path: str
    quarantine_path: str
    tree_sha256: str
    entries: tuple[tuple[str, str, int, str | None], ...]
    read_errors: tuple[tuple[str, str], ...] = ()
    unhashed: tuple[str, ...] = ()
    # None until the moved tree was re-hashed and compared.
    verified: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "origin_path": self.origin_path,
            "quarantine_path": self.quarantine_path,
            "tree_sha256": self.tree_sha256,
            "entries": [list(entry) for entry in self.entries],
            "read_errors": [list(item) for item in self.read_errors],
            "unhashed": list(self.unhashed),
            "verified": self.verified,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "QuarantineArtifact":
        try:
            return cls(
                origin_path=_str(raw["origin_path"]),
                quarantine_path=_str(raw["quarantine_path"]),
                tree_sha256=_str(raw["tree_sha256"]),
                entries=tuple(
                    (
                        _str(e[0]),
                        _str(e[1]),
                        _int(e[2]),
                        None if e[3] is None else _str(e[3]),
                    )
                    for e in raw["entries"]
                ),
                read_errors=tuple((_str(e[0]), _str(e[1])) for e in raw["read_errors"]),
                unhashed=tuple(_str(e) for e in raw["unhashed"]),
                verified=_opt_bool(raw["verified"]),
            )
        except (KeyError, IndexError, TypeError) as error:
            raise ManifestError("manifest artifact is malformed") from error


def _str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected an integer")
    return value


def _opt_bool(value: Any) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError("expected a boolean or null")
    return value


@dataclass(frozen=True)
class QuarantineManifest:
    """Sidecar record of one epoch.  Never an authority: it never narrows what
    is verified (the participant floor comes from Journal records only)."""

    epoch_id: str
    state: ManifestState
    created_at: str
    participant_id: str
    verdict: str
    reason: str
    journal_lineage_id: str | None
    journal_last_record_hash: str | None
    runtime_instance_id: str | None
    artifacts: tuple[QuarantineArtifact, ...]
    transaction_ids: tuple[str, ...] = ()
    schema: str = MANIFEST_SCHEMA

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state": self.state.value,
            "epoch_id": self.epoch_id,
            "created_at": self.created_at,
            "participant_id": self.participant_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "journal_lineage_id": self.journal_lineage_id,
            "journal_last_record_hash": self.journal_last_record_hash,
            "runtime_instance_id": self.runtime_instance_id,
            "artifacts": [artifact.to_json() for artifact in self.artifacts],
            "transaction_ids": list(self.transaction_ids),
        }

    @classmethod
    def from_json(cls, raw: object) -> "QuarantineManifest":
        if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
            raise ManifestError("manifest schema is not suzka.dc.manifest.v1")
        try:
            epoch_id = _str(raw["epoch_id"])
            if _EPOCH_ID.match(epoch_id) is None:
                raise ManifestError("manifest epoch_id is invalid")
            return cls(
                epoch_id=epoch_id,
                state=ManifestState(_str(raw["state"])),
                created_at=_str(raw["created_at"]),
                participant_id=_str(raw["participant_id"]),
                verdict=_str(raw["verdict"]),
                reason=_str(raw["reason"]),
                journal_lineage_id=_opt_str(raw["journal_lineage_id"]),
                journal_last_record_hash=_opt_str(raw["journal_last_record_hash"]),
                runtime_instance_id=_opt_str(raw["runtime_instance_id"]),
                artifacts=tuple(
                    QuarantineArtifact.from_json(a) for a in raw["artifacts"]
                ),
                transaction_ids=tuple(_str(t) for t in raw["transaction_ids"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestError("manifest is malformed") from error


def _opt_str(value: Any) -> str | None:
    return None if value is None else _str(value)


# -- file system primitives.  These three are the ONLY places this module
# -- touches the disk destructively or creatively; tests patch them to inject
# -- a crash, and the AST test pins their number.


def _assert_writable_path(path: Path) -> None:
    """Refuse to write under any ``*.quarantine-*`` component: a quarantined
    tree is never opened for writing."""

    for part in path.parts:
        if QUARANTINE_NAME_INFIX in part:
            raise QuarantineRefused("refusing to write under a quarantined path")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory(source: Path, target: Path) -> None:
    """The one and only move of a damaged artifact."""

    os.rename(source, target)


def _write_new_file(path: Path, data: bytes) -> None:
    """Create ``path`` (it must not exist: ``O_EXCL``), write, fsync."""

    _assert_writable_path(path)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_file(temporary: Path, final: Path) -> None:
    _assert_writable_path(final)
    os.replace(temporary, final)


# -- tree hashing -----------------------------------------------------------


@dataclass
class _TreeScan:
    entries: list[tuple[str, str, int, str | None]] = field(default_factory=list)
    read_errors: list[tuple[str, str]] = field(default_factory=list)
    unhashed: list[str] = field(default_factory=list)


def _hash_file(path: Path, deadline: Deadline) -> tuple[str | None, str | None]:
    """sha256 of a regular file, or ``(None, reason)``.  Opens read-only,
    without following a symlink, without blocking on a special file.  The budget is checked BEFORE the file
    is opened and between chunks; a single ``open`` / ``read`` on a hung
    filesystem is not interruptible and is not bounded by it."""

    if deadline.expired():
        return None, "unhashed"
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0),
        )
    except OSError as error:
        return None, f"open:{error.errno}"
    digest = hashlib.sha256()
    try:
        while True:
            if deadline.expired():
                return None, "unhashed"
            chunk = os.read(descriptor, _CHUNK)
            if not chunk:
                return digest.hexdigest(), None
            digest.update(chunk)
    except OSError as error:
        return None, f"read:{error.errno}"
    finally:
        os.close(descriptor)


def _scan_tree(
    root: Path,
    deadline: Deadline,
    *,
    only_hash: frozenset[str] | None = None,
) -> _TreeScan:
    """Read-only walk, sorted, symlinks recorded and never followed, never
    raising: an unreadable entry is a ``read_error``, a file the budget did not
    reach is ``unhashed``.  ``only_hash`` limits hashing to those relpaths (the
    re-hash after a move compares what the first pass reached)."""

    scan = _TreeScan()

    def walk(directory: Path, prefix: str) -> None:
        try:
            with os.scandir(directory) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError as error:
            scan.read_errors.append((prefix or ".", f"scandir:{error.errno}"))
            return
        for name in names:
            relpath = f"{prefix}{name}"
            path = directory / name
            try:
                status = os.lstat(path)
            except OSError as error:
                scan.read_errors.append((relpath, f"lstat:{error.errno}"))
                continue
            if stat.S_ISLNK(status.st_mode):
                scan.entries.append((relpath, "symlink", status.st_size, None))
            elif stat.S_ISDIR(status.st_mode):
                scan.entries.append((relpath, "dir", 0, None))
                walk(path, f"{relpath}/")
            elif stat.S_ISREG(status.st_mode):
                if only_hash is not None and relpath not in only_hash:
                    scan.entries.append((relpath, "file", status.st_size, None))
                    continue
                sha, problem = _hash_file(path, deadline)
                if sha is None:
                    if problem == "unhashed":
                        scan.unhashed.append(relpath)
                    else:
                        scan.read_errors.append((relpath, problem or "unknown"))
                scan.entries.append((relpath, "file", status.st_size, sha))
            else:
                scan.entries.append((relpath, "other", status.st_size, None))

    walk(root, "")
    return scan


def _fully_compared(
    before: Iterable[tuple[str, str, int, str | None]],
    after: Iterable[tuple[str, str, int, str | None]],
) -> bool:
    """Every file that had a sha256 before the move has one after it, so the
    comparison covered all hashed content (``verified=True`` needs this)."""

    new = {entry[0]: entry for entry in after}
    return all(
        new[entry[0]][3] is not None
        for entry in before
        if entry[1] == "file" and entry[3] is not None
    )


def _tree_digest(entries: Iterable[tuple[str, str, int, str | None]]) -> str:
    payload = json.dumps(
        sorted(list(entry) for entry in entries), separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entries_match(
    before: Iterable[tuple[str, str, int, str | None]],
    after: Iterable[tuple[str, str, int, str | None]],
) -> bool:
    """The moved tree equals the recorded one: same relpaths, kinds and sizes,
    and equal sha256 wherever BOTH passes computed one."""

    old = {entry[0]: entry for entry in before}
    new = {entry[0]: entry for entry in after}
    if old.keys() != new.keys():
        return False
    for relpath, (_, kind, size, sha) in old.items():
        _, new_kind, new_size, new_sha = new[relpath]
        if (kind, size) != (new_kind, new_size):
            return False
        if sha is not None and new_sha is not None and sha != new_sha:
            return False
    return True


# -- manifest store ---------------------------------------------------------


class QuarantineWriter:
    """Byte-preserving quarantine of one directory by rename.

    ``quarantine_directory`` does steps 1-2 of D1 sec. 4.2 (PENDING, then MOVED).
    The Journal append (JOURNALED) and the empty-root creation (COMMITTED) are
    the caller's (DC-4); this class only records their order
    (``mark_journaled`` / ``mark_committed``) and plans a resume
    (``plan_resume`` / ``resume``).
    """

    def __init__(
        self,
        quarantine_dir: Path,
        *,
        has_exclusive_authority: Callable[[], bool],
        t_hash_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        quarantinable_verdicts: frozenset[IntegrityVerdict] = frozenset(
            {IntegrityVerdict.DIVERGED}
        ),
    ) -> None:
        self._dir = Path(quarantine_dir)
        # Prohibition 8: no DC action without the Journal lease.  Asked before
        # every move (the caller passes e.g. ``journal.has_exclusive_authority``).
        self._authority = has_exclusive_authority
        self._t_hash_s = t_hash_s
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._verdicts = quarantinable_verdicts

    # -- paths

    def manifest_path(self, epoch_id: str) -> Path:
        self._check_epoch_id(epoch_id)
        return self._dir / epoch_id / MANIFEST_NAME

    @staticmethod
    def quarantine_path_for(root: Path, epoch_id: str) -> Path:
        return root.parent / f"{root.name}{QUARANTINE_NAME_INFIX}{epoch_id}"

    @staticmethod
    def _check_epoch_id(epoch_id: str) -> None:
        if _EPOCH_ID.match(epoch_id) is None:
            raise QuarantineRefused("epoch_id must be 1-64 of [A-Za-z0-9_-]")

    # -- the action

    def quarantine_directory(
        self,
        root: Path,
        *,
        reason: str,
        participant_id: str,
        epoch_id: str,
        verdict: IntegrityVerdict = IntegrityVerdict.DIVERGED,
        journal_lineage_id: str | None = None,
        journal_last_record_hash: str | None = None,
        runtime_instance_id: str | None = None,
        transaction_ids: Iterable[str] = (),
    ) -> QuarantineManifest:
        """PENDING -> rename -> MOVED.  Returns the MOVED manifest.

        Refuses (and changes nothing) when the participant or verdict is not
        quarantinable, when the target name exists, when ``root`` is not a
        real directory or is a symlink, when the parent is not owned by this
        process's uid, or when this epoch already has a manifest.  A failure
        after the rename leaves the tree where it is and raises; it is never
        moved back, copied or deleted."""

        self._check_epoch_id(epoch_id)
        self._require_authority()
        if participant_id not in QUARANTINABLE_PARTICIPANTS:
            raise QuarantineRefused("participant is not quarantinable")
        if verdict not in self._verdicts:
            raise QuarantineRefused("verdict is not quarantinable")
        root = Path(root)
        target = self.quarantine_path_for(root, epoch_id)
        manifest_dir = self._dir / epoch_id
        self._refuse_unsafe(root, target, manifest_dir)

        # Step 1, PENDING: hash the live tree (read-only, shared budget).
        deadline = Deadline(self._t_hash_s, self._clock)
        scan = _scan_tree(root, deadline)
        artifact = QuarantineArtifact(
            origin_path=str(root),
            quarantine_path=str(target),
            tree_sha256=_tree_digest(scan.entries),
            entries=tuple(scan.entries),
            read_errors=tuple(scan.read_errors),
            unhashed=tuple(scan.unhashed),
        )
        manifest = QuarantineManifest(
            epoch_id=epoch_id,
            state=ManifestState.PENDING,
            created_at=self._wall_clock().isoformat(),
            participant_id=participant_id,
            verdict=verdict.value,
            reason=_bounded(reason),
            journal_lineage_id=journal_lineage_id,
            journal_last_record_hash=journal_last_record_hash,
            runtime_instance_id=runtime_instance_id,
            artifacts=(artifact,),
            transaction_ids=tuple(transaction_ids),
        )
        self._create_manifest_dir(manifest_dir)
        self._write_manifest(manifest)

        # Step 2, MOVED.
        return self._move_and_verify(manifest, root, deadline)

    def _require_authority(self) -> None:
        if not self._authority():
            raise QuarantineRefused("no exclusive Journal authority; nothing was done")

    def _refuse_unsafe(self, root: Path, target: Path, manifest_dir: Path) -> None:
        try:
            status = os.lstat(root)
        except OSError as error:
            raise QuarantineRefused("root cannot be inspected") from error
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise QuarantineRefused("root is not a real directory")
        try:
            parent_status = os.stat(root.parent)
        except OSError as error:
            raise QuarantineRefused("parent cannot be inspected") from error
        if parent_status.st_uid != os.geteuid():
            raise QuarantineRefused("parent is not owned by this process")
        if os.path.lexists(target):
            raise QuarantineRefused("quarantine target already exists")
        if os.path.lexists(manifest_dir):
            raise QuarantineRefused("epoch already has a manifest directory")
        if QUARANTINE_NAME_INFIX in root.name:
            raise QuarantineRefused("root is itself a quarantined name")

    def _move_and_verify(
        self, manifest: QuarantineManifest, root: Path, deadline: Deadline
    ) -> QuarantineManifest:
        (artifact,) = manifest.artifacts
        target = Path(artifact.quarantine_path)
        try:
            _rename_directory(root, target)
            _fsync_directory(root.parent)
        except OSError as error:
            raise QuarantineError(
                "quarantine rename or parent fsync failed (the tree may or may "
                "not have moved; nothing was deleted)"
            ) from error
        hashed = frozenset(
            entry[0]
            for entry in artifact.entries
            if entry[1] == "file" and entry[3] is not None
        )
        after = _scan_tree(target, deadline, only_hash=hashed)
        unreadable = {item[0] for item in after.read_errors} & hashed
        if unreadable or not _entries_match(artifact.entries, after.entries):
            raise QuarantineError(
                "moved tree differs from the recorded one; manifest stays PENDING"
            )
        # True only when every previously hashed file was hashed again and
        # equal; when the budget ran out the check was partial (paths, kinds,
        # sizes) and the manifest says so with None.
        complete = (
            not artifact.unhashed
            and not artifact.read_errors
            and _fully_compared(artifact.entries, after.entries)
        )
        moved = replace(
            manifest,
            state=ManifestState.MOVED,
            artifacts=(replace(artifact, verified=True if complete else None),),
        )
        self._write_manifest(moved)
        return moved

    # -- manifest persistence

    def _create_manifest_dir(self, manifest_dir: Path) -> None:
        _assert_writable_path(manifest_dir)
        manifest_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.mkdir(manifest_dir, 0o700)
        _fsync_directory(manifest_dir.parent)

    def _write_manifest(self, manifest: QuarantineManifest) -> None:
        """Write-tmp, fsync, rename, fsync-dir.  The temporary name is unique
        per write (``O_EXCL``); a leftover from a crash is harmless and is
        never deleted."""

        final = self.manifest_path(manifest.epoch_id)
        data = json.dumps(manifest.to_json(), sort_keys=True, indent=1).encode("utf-8")
        temporary = final.with_name(
            f"{MANIFEST_NAME}.tmp-{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}"
        )
        _write_new_file(temporary, data)
        _publish_file(temporary, final)
        _fsync_directory(final.parent)

    def read_manifest(self, epoch_id: str) -> QuarantineManifest:
        path = self.manifest_path(epoch_id)
        try:
            raw = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise ManifestError("manifest cannot be read") from error
        manifest = QuarantineManifest.from_json(raw)
        if manifest.epoch_id != epoch_id:
            raise ManifestError("manifest epoch_id does not match its directory")
        return manifest

    def advance(
        self,
        epoch_id: str,
        to: ManifestState,
        *,
        transaction_ids: Iterable[str] | None = None,
    ) -> QuarantineManifest:
        """Move the manifest ONE state forward (idempotent for the current
        state).  Skipping or going back raises ``ManifestError``."""

        manifest = self.read_manifest(epoch_id)
        if to is manifest.state:
            return manifest
        if _STATE_ORDER.index(to) != _STATE_ORDER.index(manifest.state) + 1:
            raise ManifestError(
                f"manifest cannot go from {manifest.state.value} to {to.value}"
            )
        ids = (
            manifest.transaction_ids
            if transaction_ids is None
            else tuple(transaction_ids)
        )
        advanced = replace(manifest, state=to, transaction_ids=ids)
        self._write_manifest(advanced)
        return advanced

    def mark_journaled(
        self, epoch_id: str, *, transaction_ids: Iterable[str] | None = None
    ) -> QuarantineManifest:
        """Step 3 done: the outcomes are in the Journal."""

        return self.advance(
            epoch_id, ManifestState.JOURNALED, transaction_ids=transaction_ids
        )

    def mark_committed(self, epoch_id: str) -> QuarantineManifest:
        """Step 4 done: the empty root with its marker exists."""

        return self.advance(epoch_id, ManifestState.COMMITTED)

    def list_epochs(self) -> tuple[str, ...]:
        """Epoch ids that have a manifest directory, sorted.  Read-only."""

        try:
            with os.scandir(self._dir) as iterator:
                names = sorted(
                    e.name for e in iterator if e.is_dir(follow_symlinks=False)
                )
        except OSError:
            return ()
        return tuple(name for name in names if _EPOCH_ID.match(name))

    # -- resume

    def observe(self, manifest: QuarantineManifest, root: Path) -> "ResumeFacts":
        """Read-only facts the resume planner needs."""

        (artifact,) = manifest.artifacts
        target = Path(artifact.quarantine_path)
        marker_epoch: str | None = None
        try:
            marker = json.loads((Path(root) / EPOCH_MARKER_NAME).read_bytes())
            if isinstance(marker, dict) and isinstance(marker.get("epoch_id"), str):
                marker_epoch = marker["epoch_id"]
        except (OSError, ValueError):
            marker_epoch = None
        hashes_equal = False
        if os.path.lexists(target) and not os.path.islink(target):
            hashed = frozenset(
                e[0] for e in artifact.entries if e[1] == "file" and e[3] is not None
            )
            after = _scan_tree(
                target, Deadline(self._t_hash_s, self._clock), only_hash=hashed
            )
            hashes_equal = _entries_match(artifact.entries, after.entries)
        return ResumeFacts(
            original_present=os.path.lexists(root),
            quarantined_present=os.path.lexists(target),
            quarantined_hashes_equal=hashes_equal,
            root_marker_epoch_id=marker_epoch,
        )

    def resume(self, epoch_id: str, root: Path) -> "ResumePlan":
        """Plan from the manifest and the disk, and carry out the steps this
        class owns (redo the rename, record MOVED).  Returns the plan; what
        remains (Journal outcomes, the empty root) is the caller's."""

        manifest = self.read_manifest(epoch_id)
        plan = plan_resume(manifest, self.observe(manifest, Path(root)))
        if plan.action is ResumeAction.REDO_MOVE:
            self._require_authority()
            root = Path(root)
            (artifact,) = manifest.artifacts
            target = self.quarantine_path_for(root, epoch_id)
            # The manifest never steers the rename: the destination is
            # recomputed and must equal what it recorded.
            if artifact.origin_path != str(root) or artifact.quarantine_path != str(
                target
            ):
                raise QuarantineRefused("manifest does not match this root and epoch")
            status = os.lstat(root)
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise QuarantineRefused("root is not a real directory")
            if os.stat(root.parent).st_uid != os.geteuid():
                raise QuarantineRefused("parent is not owned by this process")
            if QUARANTINE_NAME_INFIX in root.name:
                raise QuarantineRefused("root is itself a quarantined name")
            # The original may have changed since the PENDING hashes were
            # taken (a live writer, the crash window): hash it again and
            # record that before moving.
            deadline = Deadline(self._t_hash_s, self._clock)
            scan = _scan_tree(root, deadline)
            manifest = replace(
                manifest,
                artifacts=(
                    replace(
                        artifact,
                        tree_sha256=_tree_digest(scan.entries),
                        entries=tuple(scan.entries),
                        read_errors=tuple(scan.read_errors),
                        unhashed=tuple(scan.unhashed),
                    ),
                ),
            )
            self._write_manifest(manifest)
            moved = self._move_and_verify(manifest, root, deadline)
            return ResumePlan(ResumeAction.CONTINUE_JOURNAL, "moved again", moved.state)
        if plan.action is ResumeAction.MARK_MOVED:
            moved = self.advance(epoch_id, ManifestState.MOVED)
            return ResumePlan(ResumeAction.CONTINUE_JOURNAL, plan.reason, moved.state)
        if plan.action is ResumeAction.MARK_COMMITTED:
            done = self.advance(epoch_id, ManifestState.COMMITTED)
            return ResumePlan(ResumeAction.DONE, plan.reason, done.state)
        return plan


class ResumeAction(str, Enum):
    REDO_MOVE = "REDO_MOVE"  # PENDING, original present: redo the rename (step 2)
    MARK_MOVED = "MARK_MOVED"  # PENDING, rename already happened: record it
    CONTINUE_JOURNAL = "CONTINUE_JOURNAL"  # MOVED: continue at step 3
    CREATE_EMPTY_ROOT = "CREATE_EMPTY_ROOT"  # JOURNALED, no live root: step 4
    MARK_COMMITTED = "MARK_COMMITTED"  # JOURNALED, the epoch's empty root exists
    DONE = "DONE"  # COMMITTED
    BLOCKED = "BLOCKED"  # a state that no rule covers: no automatic action


@dataclass(frozen=True)
class ResumeFacts:
    original_present: bool
    quarantined_present: bool
    quarantined_hashes_equal: bool
    root_marker_epoch_id: str | None


@dataclass(frozen=True)
class ResumePlan:
    action: ResumeAction
    reason: str
    state: ManifestState | None = None


def plan_resume(manifest: QuarantineManifest, facts: ResumeFacts) -> ResumePlan:
    """The crash matrix, with no cross-state guessing (D1 sec. 4.2).

    * PENDING, original present, target absent -> redo from step 2.
    * PENDING, original absent, target present, hashes equal -> treat as MOVED.
    * MOVED -> continue at step 3 (the Journal's existing "outcome already
      known" checks give idempotence, so a crash between the last append and
      the JOURNALED write is safe to repeat).
    * JOURNALED -> step 4: create the empty root, or, when the live root
      already carries this epoch's marker, record COMMITTED.
    * COMMITTED -> nothing to do.
    * everything else (both present, both gone, a mismatch, a live root that is
      not this epoch's) -> BLOCKED: no automatic action, nothing deleted.
    """

    state = manifest.state
    if state is ManifestState.PENDING:
        if facts.original_present and not facts.quarantined_present:
            return ResumePlan(ResumeAction.REDO_MOVE, "original still in place", state)
        if (
            not facts.original_present
            and facts.quarantined_present
            and facts.quarantined_hashes_equal
        ):
            return ResumePlan(ResumeAction.MARK_MOVED, "rename completed", state)
        return ResumePlan(
            ResumeAction.BLOCKED, "pending quarantine is not in a known shape", state
        )
    # The moved tree must still be where the manifest says for every later state.
    if not facts.quarantined_present:
        return ResumePlan(ResumeAction.BLOCKED, "quarantined tree is missing", state)
    if state is ManifestState.MOVED:
        if facts.original_present and facts.root_marker_epoch_id != manifest.epoch_id:
            return ResumePlan(
                ResumeAction.BLOCKED,
                "a live root exists that is not this epoch's",
                state,
            )
        return ResumePlan(ResumeAction.CONTINUE_JOURNAL, "outcomes not recorded", state)
    if state is ManifestState.JOURNALED:
        if not facts.original_present:
            return ResumePlan(
                ResumeAction.CREATE_EMPTY_ROOT, "outcomes recorded", state
            )
        if facts.root_marker_epoch_id == manifest.epoch_id:
            return ResumePlan(ResumeAction.MARK_COMMITTED, "empty root exists", state)
        return ResumePlan(
            ResumeAction.BLOCKED, "a live root exists that is not this epoch's", state
        )
    return ResumePlan(ResumeAction.DONE, "committed", state)
