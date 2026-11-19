"""R18 IP-15: the fixed target and its operation ledger (#368 D1, #380 D1/D2).

The target is its own SQLite file (HP-2: ``/var/lib/suzka/target/``, never
the SubjectHistory file). The only effect it can hold is one ReviewArtifact
per obligation group, and every effect or fence is a row of the append-only
operation ledger written in the same target transaction:

- ``FORWARD_WRITE_REVIEW_ARTIFACT_V1``: ABSENT → ACTIVE under CAS.
- ``INVERSE_WRITE_REVIEW_ARTIFACT_V1``: ACTIVE → REVERTED under CAS.
- ``fence``: the startup sweep's FENCED row (#380 D1 P1-1). Because of
  ``UNIQUE(operation_key)`` and ``UNIQUE(obligation_group_key, direction)``
  a late CAS for a fenced key always fails, and a FENCED key is never
  reissued (#380 D2 P1-A).
- ``READ_REVIEW_ARTIFACT_STATUS_V1``: ``TargetReader``, a read-only
  connection used only by the verifier and the sweep (``suzka.r18.verifier``).

A write is accepted only from the exact challenge and claim (#380 §2-3 / D2
addendum 4): the bytes written must hash to the approved payload digest, and
the claim's ``runtime_instance_id`` must equal the one row the sweep rotates
(#380 D2 P2-1 / addendum 1). A mismatch is a definite CAS_REJECTED, not
IN_DOUBT. The executor's receipt is only its own report.

Capacity (#378 V ``target_ledger_reserve``): every applied forward effect
without an inverse row keeps room for its own inverse (``inverse_room``).
Before COMMIT a forward must leave ``reserve_pages`` plus the inverse room of
all such effects (itself included) free; a fence may use the reserve but not
the inverse room; an inverse may use everything left. So every effect can
always be undone, and fences can never take that room (the HP-8 rule: only
capacity-increasing forward work is refused). A fence past the inverse room
fails as TargetCapacityError, which leaves the sweep BLOCKED; IP-16 / IP-18
keep the number of unapplied intents bounded.

A reverted effect keeps its bytes (only the state and revision change), so
the verifier can recompute their digest for the inverse too.

Nothing here is wired to R16, the runtime or any setting (IP-18 does that).
A lost target is never replaced by a new empty one: ``open`` never creates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
from threading import Lock
from typing import Final
from uuid import uuid4

from suzka.r16.contracts import digest, identifier
from suzka.r16.history_codec import canonical_uuid
from suzka.r18.contracts import (
    MAX_ARTIFACT_BYTES,
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    ExecutionReceiptV1,
    ExecutorResult,
    InverseIntentV1,
    operation_key,
    payload_digest,
)

TARGET_SCHEMA_VERSION: Final = 1
# Proposed bounds (IP-15; HP-2 asks the PR to state them). Measured: one
# forward effect adds about 0.64 pages (a ~1.3 KiB artifact and its ledger
# row) plus its inverse room; the defaults hold 4264 forward effects.
DEFAULT_MAX_PAGE_COUNT: Final = 4096  # 16 MiB with 4 KiB pages
DEFAULT_RESERVE_PAGES: Final = 256  # 1 MiB kept for fences
# Measured (IP-15): one inverse adds about 0.13 pages on average and at most
# 3 in a single B-tree split. The room keeps twice the average plus slack.
INVERSE_ROWS_PER_PAGE: Final = 4
INVERSE_SLACK_PAGES: Final = 8
_SQLITE_BUSY_TIMEOUT_SECONDS: Final = 5.0
_HEX = "length({0}) = 64 AND {0} NOT GLOB '*[^0-9a-f]*'"

SCHEMA_STATEMENTS: Final = (
    "CREATE TABLE target_meta ("
    " id INTEGER PRIMARY KEY CHECK (id = 1),"
    " schema_version INTEGER NOT NULL CHECK (schema_version = 1),"
    " target_instance_id TEXT NOT NULL CHECK (length(target_instance_id) = 36),"
    " target_namespace TEXT NOT NULL CHECK (length(target_namespace) BETWEEN 1 AND 128),"
    " runtime_instance_id TEXT NOT NULL CHECK (length(runtime_instance_id) = 36)"
    ") STRICT",
    "CREATE TABLE target_operations ("
    " operation_key TEXT PRIMARY KEY CHECK (" + _HEX.format("operation_key") + "),"
    " obligation_group_key TEXT NOT NULL CHECK (" + _HEX.format("obligation_group_key") + "),"
    " direction TEXT NOT NULL CHECK (direction IN ('FORWARD', 'INVERSE')),"
    " kind TEXT NOT NULL CHECK (kind IN ('EFFECT', 'FENCED')),"
    " payload_digest TEXT CHECK (payload_digest IS NULL OR (" + _HEX.format("payload_digest") + ")),"
    " approval_ref TEXT CHECK (approval_ref IS NULL OR (" + _HEX.format("approval_ref") + ")),"
    " runtime_instance_id TEXT CHECK (runtime_instance_id IS NULL OR length(runtime_instance_id) = 36),"
    " revision_after INTEGER CHECK (revision_after IS NULL OR revision_after > 0),"
    " CHECK ((kind = 'EFFECT') = (payload_digest IS NOT NULL AND approval_ref IS NOT NULL"
    " AND runtime_instance_id IS NOT NULL AND revision_after IS NOT NULL)),"
    " CHECK (kind = 'EFFECT' OR (payload_digest IS NULL AND approval_ref IS NULL"
    " AND runtime_instance_id IS NULL AND revision_after IS NULL)),"
    " UNIQUE (obligation_group_key, direction)"
    ") STRICT",
    "CREATE TABLE target_artifacts ("
    " obligation_group_key TEXT PRIMARY KEY CHECK (" + _HEX.format("obligation_group_key") + "),"
    " state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'REVERTED')),"
    " revision INTEGER NOT NULL CHECK (revision > 0),"
    f" payload BLOB NOT NULL CHECK (length(payload) <= {MAX_ARTIFACT_BYTES})"
    ") STRICT",
    # Append-only ledger; the effect row moves only ACTIVE -> REVERTED once.
    "CREATE TRIGGER target_operations_no_update BEFORE UPDATE ON target_operations"
    " BEGIN SELECT RAISE(ABORT, 'target ledger is append-only'); END",
    "CREATE TRIGGER target_operations_no_delete BEFORE DELETE ON target_operations"
    " BEGIN SELECT RAISE(ABORT, 'target ledger is append-only'); END",
    "CREATE TRIGGER target_artifacts_no_delete BEFORE DELETE ON target_artifacts"
    " BEGIN SELECT RAISE(ABORT, 'target effects are not deleted'); END",
    "CREATE TRIGGER target_artifacts_revert_only BEFORE UPDATE ON target_artifacts"
    " WHEN NOT (OLD.state = 'ACTIVE' AND NEW.state = 'REVERTED'"
    " AND NEW.revision = OLD.revision + 1 AND NEW.payload = OLD.payload"
    " AND NEW.obligation_group_key = OLD.obligation_group_key)"
    " BEGIN SELECT RAISE(ABORT, 'a target effect only reverts once'); END",
    "CREATE TRIGGER target_meta_no_delete BEFORE DELETE ON target_meta"
    " BEGIN SELECT RAISE(ABORT, 'target meta is permanent'); END",
    "CREATE TRIGGER target_meta_instance_only BEFORE UPDATE ON target_meta"
    " WHEN NOT (NEW.id = OLD.id AND NEW.schema_version = OLD.schema_version"
    " AND NEW.target_instance_id = OLD.target_instance_id"
    " AND NEW.target_namespace = OLD.target_namespace)"
    " BEGIN SELECT RAISE(ABORT, 'only runtime_instance_id rotates'); END",
)


class TargetError(Exception):
    """Base of every target failure."""


class TargetUnavailable(TargetError):
    """The target cannot be reached now (absent, locked, I/O). Not ABSENT."""


class TargetCorrupt(TargetError):
    """Custody, schema or integrity is wrong: BLOCKED, never recreated."""


class TargetNotStarted(TargetUnavailable):
    """``BEGIN IMMEDIATE`` failed: provably nothing ran (IP-18a, F07 refined).

    Raised only from the ``BEGIN`` branch of a write, so a caller may retry
    the same write; every other failure is an unknown result."""


class TargetCapacityError(TargetUnavailable):
    """No room for a capacity-increasing forward effect."""


class TargetBindingError(TargetError):
    """The intent, challenge and claim do not describe one approved write."""


class LedgerKind(str, Enum):
    EFFECT = "EFFECT"
    FENCED = "FENCED"


class ArtifactState(str, Enum):
    ACTIVE = "ACTIVE"
    REVERTED = "REVERTED"


@dataclass(frozen=True, slots=True)
class LedgerRow:
    operation_key: str
    obligation_group_key: str
    direction: Direction
    kind: LedgerKind
    payload_digest: str | None
    approval_ref: str | None
    runtime_instance_id: str | None
    revision_after: int | None


@dataclass(frozen=True, slots=True)
class ArtifactRow:
    obligation_group_key: str
    state: ArtifactState
    revision: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class TargetMeta:
    target_instance_id: str
    target_namespace: str
    runtime_instance_id: str


@dataclass(frozen=True, slots=True)
class GroupStatus:
    """``READ_REVIEW_ARTIFACT_STATUS_V1``: one group's effect and ledger rows."""

    meta: TargetMeta
    artifact: ArtifactRow | None
    forward: LedgerRow | None
    inverse: LedgerRow | None


# --------------------------------------------------------------- custody


def _trusted_path(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise TargetCorrupt("target path must be absolute")
    if os.path.realpath(path.parent) != str(path.parent) or path.name in {"", ".", ".."}:
        raise TargetCorrupt("target path must be a normal path without symlinks")
    return path


def _check_private_directory(directory: Path) -> None:
    try:
        info = os.lstat(directory)
    except OSError as error:
        raise TargetUnavailable("target directory is unavailable") from error
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise TargetCorrupt("target directory custody is invalid")


def _check_private_file(path: Path, *, required: bool) -> os.stat_result | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        if required:
            raise TargetUnavailable("target file is absent") from None
        return None
    except OSError as error:
        raise TargetUnavailable("target file is unavailable") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
    ):
        raise TargetCorrupt("target file custody is invalid")
    return info


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + "-journal")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finish_interrupted_create(path: Path) -> None:
    """Remove the one staging link a crash between link and unlink left."""
    try:
        info = os.lstat(path)
    except OSError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 2:
        return
    pattern = re.compile(rf"\.{re.escape(path.name)}\.[0-9a-f]{{32}}\.creating")
    try:
        names = [name for name in os.listdir(path.parent) if pattern.fullmatch(name)]
    except OSError:
        return
    matches = []
    for name in names:
        try:
            other = os.lstat(path.parent / name)
        except OSError:
            continue
        if (other.st_dev, other.st_ino) == (info.st_dev, info.st_ino):
            matches.append(name)
    if len(matches) == 1:
        try:
            os.unlink(path.parent / matches[0])
            _fsync_directory(path.parent)
        except OSError:
            return


def _error(error: sqlite3.Error, action: str) -> TargetError:
    primary = getattr(error, "sqlite_errorcode", 0) & 0xFF
    if primary == sqlite3.SQLITE_FULL:
        return TargetCapacityError(f"target {action} has no capacity")
    if primary in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}:
        return TargetCorrupt(f"target {action} found a corrupt file")
    if isinstance(error, sqlite3.IntegrityError):
        return TargetCorrupt(f"target {action} broke a ledger constraint")
    return TargetUnavailable(f"target {action} is unavailable")


def _blob(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TargetCorrupt("target artifact payload is not a blob")
    return value


def _uuid() -> str:
    return str(uuid4())


def _schema_rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    return connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()


def _expected_schema() -> list[tuple[object, ...]]:
    reference = sqlite3.connect(":memory:")
    try:
        for statement in SCHEMA_STATEMENTS:
            reference.execute(statement)
        return _schema_rows(reference)
    finally:
        reference.close()


def _verify(connection: sqlite3.Connection) -> None:
    if _schema_rows(connection) != _expected_schema():
        raise TargetCorrupt("target schema is not schema v1")
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise TargetCorrupt("target integrity check failed")


def _connect(path: Path, mode: str) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode={mode}",
        uri=True,
        timeout=_SQLITE_BUSY_TIMEOUT_SECONDS,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        if sys.version_info >= (3, 12):
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
        assignments: tuple[tuple[str, str, object], ...] = (
            ("trusted_schema = OFF", "trusted_schema", 0),
            ("cell_size_check = ON", "cell_size_check", 1),
        )
        if mode == "rw":
            assignments = (
                ("journal_mode = DELETE", "journal_mode", "delete"),
                ("synchronous = EXTRA", "synchronous", 3),
                *assignments,
            )
        for assignment, name, value in assignments:
            connection.execute(f"PRAGMA {assignment}").fetchall()
            actual = connection.execute(f"PRAGMA {name}").fetchone()
            if actual is None or actual[0] != value:
                raise TargetCorrupt(f"target PRAGMA {name} could not be enforced")
        return connection
    except BaseException:
        connection.close()
        raise


def _open_checked(path: Path, mode: str) -> sqlite3.Connection:
    path = _trusted_path(path)
    _check_private_directory(path.parent)
    if mode == "rw":
        _finish_interrupted_create(path)
    before = _check_private_file(path, required=True)
    _check_private_file(_sidecar(path), required=False)
    for suffix in ("-wal", "-shm"):
        if os.path.lexists(path.with_name(path.name + suffix)):
            raise TargetCorrupt("target has an unexpected WAL sidecar")
    try:
        connection = _connect(path, mode)
    except sqlite3.Error as error:
        raise _error(error, "open") from error
    try:
        after = _check_private_file(path, required=True)
        assert before is not None and after is not None
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise TargetCorrupt("target file was replaced")
        _verify(connection)
        return connection
    except sqlite3.Error as error:
        connection.close()
        raise _error(error, "open") from error
    except BaseException:
        connection.close()
        raise


# ------------------------------------------------------------------ rows


def _ledger(row: tuple[object, ...]) -> LedgerRow:
    try:
        return LedgerRow(
            operation_key=str(row[0]),
            obligation_group_key=str(row[1]),
            direction=Direction(row[2]),
            kind=LedgerKind(row[3]),
            payload_digest=None if row[4] is None else str(row[4]),
            approval_ref=None if row[5] is None else str(row[5]),
            runtime_instance_id=None if row[6] is None else str(row[6]),
            revision_after=None if row[7] is None else int(str(row[7])),
        )
    except (TypeError, ValueError) as error:
        raise TargetCorrupt("target ledger row is invalid") from error


_LEDGER_COLUMNS: Final = (
    "operation_key, obligation_group_key, direction, kind, payload_digest, "
    "approval_ref, runtime_instance_id, revision_after"
)


_META_SQL: Final = "SELECT target_instance_id, target_namespace, runtime_instance_id FROM target_meta"


def _meta(rows: list[tuple[object, ...]]) -> TargetMeta:
    if len(rows) != 1:
        raise TargetCorrupt("target meta is missing")
    try:
        return TargetMeta(
            canonical_uuid(rows[0][0], "target_instance_id"),
            identifier(rows[0][1], "target_namespace"),
            canonical_uuid(rows[0][2], "runtime_instance_id"),
        )
    except (TypeError, ValueError) as error:
        raise TargetCorrupt("target meta is invalid") from error


class _Connection:
    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection
        self._lock = Lock()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _read(self, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
        with self._lock:
            try:
                return self._connection.execute(sql, parameters).fetchall()
            except sqlite3.Error as error:
                raise _error(error, "read") from error

    def meta(self) -> TargetMeta:
        return _meta(self._read(_META_SQL))

    def ledger(self) -> tuple[LedgerRow, ...]:
        """Every ledger row, for the sweep's target → source direction."""
        return tuple(
            _ledger(row)
            for row in self._read(f"SELECT {_LEDGER_COLUMNS} FROM target_operations ORDER BY operation_key")
        )

    def forward_snapshot(self) -> tuple[tuple[LedgerRow, ArtifactRow | None], ...]:
        """Every forward ledger row with its group's artifact, in one read
        transaction (the sweep's target -> source direction, IP-18b)."""

        with self._lock:
            connection = self._connection
            try:
                connection.execute("BEGIN")
                try:
                    rows = connection.execute(
                        f"SELECT {_LEDGER_COLUMNS} FROM target_operations "
                        "WHERE direction = 'FORWARD' ORDER BY operation_key"
                    ).fetchall()
                    artifact_rows = connection.execute(
                        "SELECT obligation_group_key, state, revision, payload FROM target_artifacts"
                    ).fetchall()
                finally:
                    connection.execute("COMMIT")
            except sqlite3.Error as error:
                raise _error(error, "read") from error
        artifacts: dict[str, ArtifactRow] = {}
        for group, state, revision, payload in artifact_rows:
            try:
                artifacts[str(group)] = ArtifactRow(
                    str(group), ArtifactState(state), int(str(revision)), _blob(payload)
                )
            except (TypeError, ValueError) as error:
                raise TargetCorrupt("target artifact row is invalid") from error
        result = []
        for raw in rows:
            row = _ledger(raw)
            result.append((row, artifacts.get(row.obligation_group_key)))
        return tuple(result)

    def status(self, group_key: str) -> GroupStatus:
        """``READ_REVIEW_ARTIFACT_STATUS_V1`` for one obligation group, read
        in one snapshot (one read transaction)."""
        digest(group_key, "obligation_group_key")
        with self._lock:
            connection = self._connection
            try:
                connection.execute("BEGIN")
                try:
                    meta_rows = connection.execute(_META_SQL).fetchall()
                    rows = connection.execute(
                        f"SELECT {_LEDGER_COLUMNS} FROM target_operations WHERE obligation_group_key = ?",
                        (group_key,),
                    ).fetchall()
                    artifact_rows = connection.execute(
                        "SELECT state, revision, payload FROM target_artifacts WHERE obligation_group_key = ?",
                        (group_key,),
                    ).fetchall()
                finally:
                    connection.execute("COMMIT")
            except sqlite3.Error as error:
                raise _error(error, "read") from error
        by_direction = {row.direction: row for row in (_ledger(row) for row in rows)}
        artifact = None
        if artifact_rows:
            state, revision, payload = artifact_rows[0]
            try:
                artifact = ArtifactRow(group_key, ArtifactState(state), int(str(revision)), _blob(payload))
            except (TypeError, ValueError) as error:
                raise TargetCorrupt("target artifact row is invalid") from error
        return GroupStatus(
            _meta(meta_rows), artifact, by_direction.get(Direction.FORWARD), by_direction.get(Direction.INVERSE)
        )


class TargetReader(_Connection):
    """A read-only connection (``mode=ro``): the verifier's only view."""

    @classmethod
    def open(cls, path: Path) -> TargetReader:
        return cls(_trusted_path(path), _open_checked(path, "ro"))


class TargetStore(_Connection):
    """The one writable connection: executor writes and sweep fences."""

    def __init__(
        self,
        path: Path,
        connection: sqlite3.Connection,
        *,
        max_page_count: int,
        reserve_pages: int,
    ) -> None:
        super().__init__(path, connection)
        self._max_page_count = max_page_count
        self._reserve_pages = reserve_pages

    @staticmethod
    def _bounds(max_page_count: int, reserve_pages: int) -> None:
        if type(max_page_count) is not int or not 64 <= max_page_count <= 2**31 - 2:
            raise TargetCorrupt("target max_page_count is invalid")
        if type(reserve_pages) is not int or not 8 <= reserve_pages < max_page_count // 2:
            raise TargetCorrupt("target reserve_pages is invalid")

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        target_namespace: str,
        max_page_count: int = DEFAULT_MAX_PAGE_COUNT,
        reserve_pages: int = DEFAULT_RESERVE_PAGES,
    ) -> TargetStore:
        """Create an empty target; refuses an existing path. Explicit only."""

        path = _trusted_path(path)
        identifier(target_namespace, "target_namespace")
        cls._bounds(max_page_count, reserve_pages)
        _check_private_directory(path.parent)
        if os.path.lexists(path) or os.path.lexists(_sidecar(path)):
            raise TargetCorrupt("target file already exists")
        staging = path.with_name(f".{path.name}.{uuid4().hex}.creating")
        try:
            descriptor = os.open(staging, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            os.close(descriptor)
            connection = _connect(staging, "rw")
            try:
                connection.execute("BEGIN IMMEDIATE")
                for statement in SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO target_meta (id, schema_version, target_instance_id, "
                    "target_namespace, runtime_instance_id) VALUES (1, ?, ?, ?, ?)",
                    (TARGET_SCHEMA_VERSION, _uuid(), target_namespace, _uuid()),
                )
                connection.execute("COMMIT")
            finally:
                connection.close()
            os.link(staging, path)
        except sqlite3.Error as error:
            raise _error(error, "creation") from error
        except FileExistsError as error:
            raise TargetCorrupt("target file already exists") from error
        except OSError as error:
            raise TargetUnavailable("target creation failed") from error
        finally:
            try:
                os.unlink(staging)
            except FileNotFoundError:
                pass
        _fsync_directory(path.parent)
        return cls.open(path, max_page_count=max_page_count, reserve_pages=reserve_pages)

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        max_page_count: int = DEFAULT_MAX_PAGE_COUNT,
        reserve_pages: int = DEFAULT_RESERVE_PAGES,
    ) -> TargetStore:
        """Open an existing target read-write; never creates or migrates."""

        cls._bounds(max_page_count, reserve_pages)
        connection = _open_checked(path, "rw")
        try:
            connection.execute(f"PRAGMA max_page_count = {max_page_count}").fetchall()
            actual = connection.execute("PRAGMA max_page_count").fetchone()
            if actual is None or actual[0] != max_page_count:
                raise TargetCapacityError("target is larger than its configured max_page_count")
        except sqlite3.Error as error:
            connection.close()
            raise _error(error, "open") from error
        except BaseException:
            connection.close()
            raise
        store = cls(
            _trusted_path(path), connection, max_page_count=max_page_count, reserve_pages=reserve_pages
        )
        store.meta()
        return store

    @contextmanager
    def _write(self, action: str) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connection
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as error:
                failure = _error(error, action)
                if type(failure) is TargetUnavailable:
                    raise TargetNotStarted(str(failure)) from error
                raise failure from error
            try:
                yield connection
                connection.execute("COMMIT")
            except sqlite3.Error as error:
                self._rollback()
                raise _error(error, action) from error
            except BaseException:
                self._rollback()
                raise

    def _rollback(self) -> None:
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    @staticmethod
    def _inverse_room(connection: sqlite3.Connection) -> int:
        """Pages kept so that every applied effect without an inverse row can
        still be undone."""
        row = connection.execute(
            "SELECT count(*) FROM target_operations AS f WHERE f.direction = 'FORWARD' "
            "AND f.kind = 'EFFECT' AND NOT EXISTS (SELECT 1 FROM target_operations AS i "
            "WHERE i.obligation_group_key = f.obligation_group_key AND i.direction = 'INVERSE')"
        ).fetchone()
        return int(row[0]) // INVERSE_ROWS_PER_PAGE + INVERSE_SLACK_PAGES

    def _free_pages(self, connection: sqlite3.Connection) -> int:
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        freelist = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        return self._max_page_count - page_count + freelist

    # ------------------------------------------------------------ sweep

    def rotate_runtime_instance(self, runtime_instance_id: str) -> None:
        """#380 D2 P2-1: the first step of the startup sweep.

        The caller must hold the R07 exclusive lease first and must commit
        this before scanning (#380 D2 addendum 2). Every CAS from an older
        process then fails as CAS_REJECTED.
        """
        canonical_uuid(runtime_instance_id, "runtime_instance_id")
        with self._write("instance rotation") as connection:
            current = connection.execute("SELECT runtime_instance_id FROM target_meta").fetchone()
            if current is not None and current[0] == runtime_instance_id:
                raise TargetBindingError("a rotation needs a new runtime_instance_id")
            connection.execute(
                "UPDATE target_meta SET runtime_instance_id = ? WHERE id = 1", (runtime_instance_id,)
            )

    def fence(self, group_key: str, direction: Direction) -> LedgerRow:
        """Insert the FENCED row for ``(group, direction)``, or return the row
        that is already there (an EFFECT means: reconcile as an effect).

        A successful fence is not yet VERIFIED_ABSENT: the verifier decides
        that from what it reads (#380 D2 P2-2). Fences may use the reserve,
        but never the room kept for inverses.
        """
        digest(group_key, "obligation_group_key")
        if type(direction) is not Direction:
            raise TypeError("direction must be a Direction")
        key = operation_key(group_key, direction)
        with self._write("fence") as connection:
            rows = connection.execute(
                f"SELECT {_LEDGER_COLUMNS} FROM target_operations "
                "WHERE obligation_group_key = ? AND direction = ?",
                (group_key, direction.value),
            ).fetchall()
            if rows:
                return _ledger(rows[0])
            connection.execute(
                "INSERT INTO target_operations (operation_key, obligation_group_key, direction, kind) "
                "VALUES (?, ?, ?, 'FENCED')",
                (key, group_key, direction.value),
            )
            if self._free_pages(connection) < self._inverse_room(connection):
                raise TargetCapacityError("a fence may not use the room kept for inverses")
        return LedgerRow(key, group_key, direction, LedgerKind.FENCED, None, None, None, None)

    # ----------------------------------------------------------- effects

    def forward_write(
        self,
        intent: ActionIntentV1,
        challenge: ApprovalChallengeV1,
        claim: ExecutionClaimV1,
        *,
        now: Callable[[], datetime],
    ) -> ExecutionReceiptV1:
        """``FORWARD_WRITE_REVIEW_ARTIFACT_V1`` (ABSENT → ACTIVE).

        HP-7 (a): inside the ``BEGIN IMMEDIATE`` transaction, ``now()`` must be
        before the claim's Commitment deadline (none: no check), else the
        receipt is ``DEADLINE_ELAPSED`` and nothing is written."""

        if type(intent) is not ActionIntentV1:
            raise TypeError("intent must be an ActionIntentV1")
        payload = intent.artifact.payload_bytes()
        _bind(challenge, claim, Direction.FORWARD, intent.intent_id, intent.intent_digest)
        if (
            challenge.operation_key != intent.forward_key
            or challenge.target_instance_id != intent.target_instance_id
            or challenge.target_expected_revision != intent.target_expected_revision
            or challenge.payload_digest != payload_digest(payload)
        ):
            raise TargetBindingError("the challenge does not approve these bytes")
        return self._apply(
            claim,
            direction=Direction.FORWARD,
            group_key=intent.obligation_group_key,
            namespace=intent.target_namespace,
            target_instance_id=intent.target_instance_id,
            expected_revision=intent.target_expected_revision,
            payload=payload,
            approved_digest=challenge.payload_digest,
            now=now,
        )

    def inverse_write(
        self,
        intent: InverseIntentV1,
        challenge: ApprovalChallengeV1,
        claim: ExecutionClaimV1,
    ) -> ExecutionReceiptV1:
        """``INVERSE_WRITE_REVIEW_ARTIFACT_V1`` (ACTIVE → REVERTED)."""

        if type(intent) is not InverseIntentV1:
            raise TypeError("intent must be an InverseIntentV1")
        _bind(challenge, claim, Direction.INVERSE, intent.intent_id, intent.intent_digest)
        if (
            challenge.operation_key != intent.inverse_key
            or challenge.target_instance_id != intent.target_instance_id
            or challenge.target_expected_revision != intent.target_expected_revision
            or challenge.payload_digest != intent.forward_payload_digest
        ):
            raise TargetBindingError("the challenge does not approve this inverse")
        return self._apply(
            claim,
            direction=Direction.INVERSE,
            group_key=intent.obligation_group_key,
            namespace=intent.target_namespace,
            target_instance_id=intent.target_instance_id,
            expected_revision=intent.target_expected_revision,
            payload=None,
            approved_digest=intent.forward_payload_digest,
        )

    def _apply(
        self,
        claim: ExecutionClaimV1,
        *,
        direction: Direction,
        group_key: str,
        namespace: str,
        target_instance_id: str,
        expected_revision: int,
        payload: bytes | None,
        approved_digest: str,
        now: Callable[[], datetime] | None = None,
    ) -> ExecutionReceiptV1:
        def receipt(result: ExecutorResult, revision: int | None = None) -> ExecutionReceiptV1:
            return ExecutionReceiptV1.for_claim(
                claim,
                result,
                reported_revision=revision,
                reported_payload_digest=approved_digest
                if revision is not None and direction is Direction.FORWARD
                else None,
            )

        key = operation_key(group_key, direction)
        with self._write(f"{direction.value.lower()} write") as connection:
            meta = connection.execute(
                "SELECT target_instance_id, target_namespace, runtime_instance_id FROM target_meta"
            ).fetchone()
            if meta is None:
                raise TargetCorrupt("target meta is missing")
            if (meta[0], meta[1]) != (target_instance_id, namespace):
                raise TargetBindingError("the intent names another target")
            if meta[2] != claim.runtime_instance_id:
                # An older process after the sweep rotated the instance.
                return receipt(ExecutorResult.CAS_REJECTED)
            existing = connection.execute(
                "SELECT kind FROM target_operations WHERE operation_key = ? "
                "OR (obligation_group_key = ? AND direction = ?)",
                (key, group_key, direction.value),
            ).fetchall()
            if any(row[0] == LedgerKind.FENCED.value for row in existing):
                return receipt(ExecutorResult.FENCED)
            artifact = connection.execute(
                "SELECT state, revision, payload FROM target_artifacts WHERE obligation_group_key = ?",
                (group_key,),
            ).fetchone()
            if existing:
                return receipt(ExecutorResult.CAS_REJECTED)  # already applied: never twice
            if direction is Direction.FORWARD and claim.commitment_deadline_utc is not None:
                assert now is not None
                # Read after BEGIN IMMEDIATE returned: check and commit adjoin.
                moment = now()
                if moment.tzinfo is None:
                    raise TargetBindingError("the target clock must be aware")
                deadline = datetime.strptime(
                    claim.commitment_deadline_utc, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)
                if moment >= deadline:
                    return receipt(ExecutorResult.DEADLINE_ELAPSED)
            if direction is Direction.FORWARD:
                if artifact is not None or expected_revision != 0:
                    return receipt(ExecutorResult.CAS_REJECTED)
                assert payload is not None
                revision = 1
                connection.execute(
                    "INSERT INTO target_artifacts (obligation_group_key, state, revision, payload) "
                    "VALUES (?, 'ACTIVE', ?, ?)",
                    (group_key, revision, payload),
                )
            else:
                forward_effect = connection.execute(
                    "SELECT 1 FROM target_operations WHERE obligation_group_key = ? "
                    "AND direction = 'FORWARD' AND kind = 'EFFECT'",
                    (group_key,),
                ).fetchone()
                if (
                    forward_effect is None
                    or artifact is None
                    or artifact[0] != ArtifactState.ACTIVE.value
                    or artifact[1] != expected_revision
                    or payload_digest(_blob(artifact[2])) != approved_digest
                ):
                    return receipt(ExecutorResult.CAS_REJECTED)
                revision = expected_revision + 1
                connection.execute(
                    "UPDATE target_artifacts SET state = 'REVERTED', revision = ? "
                    "WHERE obligation_group_key = ?",
                    (revision, group_key),
                )
            connection.execute(
                "INSERT INTO target_operations (operation_key, obligation_group_key, direction, kind, "
                "payload_digest, approval_ref, runtime_instance_id, revision_after) "
                "VALUES (?, ?, ?, 'EFFECT', ?, ?, ?, ?)",
                (key, group_key, direction.value, approved_digest, claim.nonce, claim.runtime_instance_id, revision),
            )
            if direction is Direction.FORWARD and self._free_pages(connection) < (
                self._reserve_pages + self._inverse_room(connection)
            ):
                # Checked after the writes: the whole transaction rolls back.
                raise TargetCapacityError("no room for a forward effect above the reserve")
        return receipt(ExecutorResult.APPLIED, revision)


def _bind(
    challenge: ApprovalChallengeV1,
    claim: ExecutionClaimV1,
    direction: Direction,
    intent_id: str,
    intent_digest: str,
) -> None:
    if type(challenge) is not ApprovalChallengeV1 or type(claim) is not ExecutionClaimV1:
        raise TypeError("expected an ApprovalChallengeV1 and an ExecutionClaimV1")
    if (
        challenge.direction is not direction
        or claim.direction is not direction
        or (challenge.intent_id, challenge.intent_digest) != (intent_id, intent_digest)
        or (claim.intent_id, claim.intent_digest) != (intent_id, intent_digest)
        or claim.challenge_digest != challenge.challenge_digest
        or claim.nonce != challenge.nonce
        or claim.operation_key != challenge.operation_key
        or claim.boot_id != challenge.boot_id
        or claim.commitment_deadline_utc != challenge.commitment_deadline_utc
    ):
        raise TargetBindingError("the claim is not this challenge's claim for this intent")


def new_runtime_instance_id() -> str:
    """A fresh immutable instance id for one Runtime process (#380 D2 addendum 1)."""
    return str(uuid4())


__all__ = [
    "DEFAULT_MAX_PAGE_COUNT",
    "DEFAULT_RESERVE_PAGES",
    "INVERSE_ROWS_PER_PAGE",
    "INVERSE_SLACK_PAGES",
    "SCHEMA_STATEMENTS",
    "TARGET_SCHEMA_VERSION",
    "ArtifactRow",
    "ArtifactState",
    "GroupStatus",
    "LedgerKind",
    "LedgerRow",
    "TargetBindingError",
    "TargetCapacityError",
    "TargetCorrupt",
    "TargetError",
    "TargetNotStarted",
    "TargetMeta",
    "TargetReader",
    "TargetStore",
    "TargetUnavailable",
    "new_runtime_instance_id",
]
