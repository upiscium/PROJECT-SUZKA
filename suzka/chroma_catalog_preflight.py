"""Disconnected Chroma 1.5.9 catalog observation; never startup authority.

Assertions are supplied by a caller, not an authenticated registration. SQLite
opens by pathname: the descriptor checks are NOT an atomic pathname binding.
"""

from __future__ import annotations

import math
import os
import sqlite3
import stat
import time
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import UUID

_MAX_CATALOG_TEXT_BYTES = 1024


class Observation(StrEnum):
    VERIFIED_UUIDS = "verified_uuids"
    REFUSED = "refused"


class Refusal(StrEnum):
    INVALID_POLICY = "invalid_policy"
    PATH_UNSAFE = "path_unsafe"
    ABSENT = "absent"
    IDENTITY_MISMATCH = "identity_mismatch"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    CATALOG_MISMATCH = "catalog_mismatch"
    RESOURCE_EXCEEDED = "resource_exceeded"
    UNSTABLE = "unstable"


@dataclass(frozen=True, slots=True)
class Identity:
    path: str
    device: int
    inode: int
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True, slots=True)
class Collection:
    name: str
    uuid: str


@dataclass(frozen=True, slots=True)
class Limits:
    max_file_bytes: int
    max_pages: int
    max_path_bytes: int
    max_components: int
    max_entries: int
    max_vm_steps: int
    deadline_seconds: float
    busy_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class Policy:
    source: Identity
    chroma_root: Identity
    catalog: Identity
    schema_tag: str
    tenant_id: str
    database_name: str
    database_uuid: str
    db1: Collection
    db2: Collection
    limits: Limits


@dataclass(frozen=True, slots=True)
class Result:
    observation: Observation
    startup: str = "DISABLED_PENDING_I2"
    refusal: Refusal | None = None


class _Denied(Exception):
    def __init__(self, reason: Refusal):
        self.reason = reason


def _require(condition: bool, reason: Refusal) -> None:
    if not condition:
        raise _Denied(reason)


def _uuid(value: object) -> bool:
    try:
        return type(value) is str and str(UUID(value)) == value
    except (TypeError, ValueError, AttributeError):
        return False


def _catalog_text(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        return 0 < len(value.encode("utf-8")) <= _MAX_CATALOG_TEXT_BYTES
    except UnicodeError:
        return False


def _parts(path: object, limits: Limits) -> tuple[str, ...]:
    if type(path) is not str:
        raise _Denied(Refusal.INVALID_POLICY)
    _require(
        path.startswith("/")
        and not path.startswith("//")
        and 0 < len(os.fsencode(path)) <= limits.max_path_bytes,
        Refusal.PATH_UNSAFE,
    )
    parts = tuple(path.split("/"))[1:]
    _require(
        0 < len(parts) <= limits.max_components
        and all(part not in ("", ".", "..") and "\x00" not in part for part in parts),
        Refusal.PATH_UNSAFE,
    )
    return parts


def _validate(policy: Policy) -> None:
    _require(
        isinstance(policy, Policy) and isinstance(policy.limits, Limits),
        Refusal.INVALID_POLICY,
    )
    limits = policy.limits
    _require(
        all(
            type(getattr(limits, key)) is int and getattr(limits, key) > 0
            for key in (
                "max_file_bytes",
                "max_pages",
                "max_path_bytes",
                "max_components",
                "max_entries",
                "max_vm_steps",
            )
        )
        and all(
            type(getattr(limits, key)) in (int, float)
            and math.isfinite(getattr(limits, key))
            and getattr(limits, key) > 0
            for key in ("deadline_seconds", "busy_timeout_seconds")
        ),
        Refusal.INVALID_POLICY,
    )
    _require(
        isinstance(policy.source, Identity)
        and isinstance(policy.chroma_root, Identity)
        and isinstance(policy.catalog, Identity)
        and isinstance(policy.db1, Collection)
        and isinstance(policy.db2, Collection),
        Refusal.INVALID_POLICY,
    )
    roots = [
        _parts(item.path, limits)
        for item in (policy.source, policy.chroma_root, policy.catalog)
    ]
    _require(
        len(roots[1]) > len(roots[0])
        and roots[1][: len(roots[0])] == roots[0]
        and roots[2] == (*roots[1], "chroma.sqlite3"),
        Refusal.PATH_UNSAFE,
    )
    for item in (policy.source, policy.chroma_root, policy.catalog):
        _require(
            all(
                type(getattr(item, key)) is int
                and getattr(item, key) >= (1 if key in ("device", "inode") else 0)
                for key in ("device", "inode", "uid", "gid")
            )
            and type(item.mode) is int
            and 0 <= item.mode <= 0o7777,
            Refusal.INVALID_POLICY,
        )
    _require(
        policy.schema_tag == "1.5.9"
        and type(policy.schema_tag) is str
        and _catalog_text(policy.tenant_id)
        and _catalog_text(policy.database_name)
        and _uuid(policy.database_uuid)
        and all(
            _catalog_text(c.name) and _uuid(c.uuid) for c in (policy.db1, policy.db2)
        )
        and policy.db1.name != policy.db2.name
        and policy.db1.uuid != policy.db2.uuid,
        Refusal.INVALID_POLICY,
    )


def _matches(fd: int, expected: Identity, directory: bool) -> None:
    s = os.fstat(fd)
    _require(
        stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode),
        Refusal.IDENTITY_MISMATCH,
    )
    _require(
        (s.st_dev, s.st_ino, s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode))
        == (expected.device, expected.inode, expected.uid, expected.gid, expected.mode),
        Refusal.IDENTITY_MISMATCH,
    )
    # SQLite read-only mode may touch WAL/SHM sidecars by pathname.  An
    # observed Source, Chroma directory, or main catalog writable by other
    # credentials cannot safely support the future cooperative P2 fence.
    _require(stat.S_IMODE(s.st_mode) & 0o022 == 0, Refusal.PATH_UNSAFE)
    if not directory:
        _require(s.st_nlink == 1, Refusal.IDENTITY_MISMATCH)


def _open_path(path: str, limits: Limits, *, directory: bool) -> int:
    parts = _parts(path, limits)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            if index < len(parts) - 1 or directory:
                flags |= os.O_DIRECTORY
            else:
                flags |= os.O_NONBLOCK
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _uri(path: str) -> str:
    uri = Path(path).as_uri() + "?mode=ro"
    parsed = urlsplit(uri)
    _require(
        parsed.scheme == "file"
        and not parsed.netloc
        and not parsed.fragment
        and parsed.query == "mode=ro"
        and unquote(parsed.path) == path,
        Refusal.PATH_UNSAFE,
    )
    return uri


def _check_schema(connection: sqlite3.Connection) -> None:
    expected = {
        "tenants": {"id"},
        "databases": {"id", "name", "tenant_id"},
        "collections": {"id", "name", "database_id"},
    }
    for table, columns in expected.items():
        rows = connection.execute(
            "SELECT type FROM sqlite_master WHERE name = ?", (table,)
        ).fetchmany(2)
        _require(rows == [("table",)], Refusal.UNSUPPORTED_SCHEMA)
        # Fixed identifiers, never interpolated caller input.
        fields = connection.execute(f"PRAGMA table_info({table})").fetchall()
        _require(columns <= {row[1] for row in fields}, Refusal.UNSUPPORTED_SCHEMA)


def _sidecars(root_fd: int, limits: Limits, main_size: int) -> None:
    total = main_size
    for suffix in ("-wal", "-shm"):
        try:
            s = os.stat(
                "chroma.sqlite3" + suffix, dir_fd=root_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            continue
        _require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, Refusal.PATH_UNSAFE)
        total += s.st_size
        _require(total <= limits.max_file_bytes, Refusal.RESOURCE_EXCEEDED)


def _observe(policy: Policy) -> None:
    _validate(policy)
    limits = policy.limits
    deadline = time.monotonic() + limits.deadline_seconds
    fds: list[int] = []
    try:
        for item, directory in (
            (policy.source, True),
            (policy.chroma_root, True),
            (policy.catalog, False),
        ):
            fd = _open_path(item.path, limits, directory=directory)
            fds.append(fd)
            _matches(fd, item, directory)
        _require(
            os.fstat(fds[-1]).st_size <= limits.max_file_bytes,
            Refusal.RESOURCE_EXCEEDED,
        )
        _sidecars(fds[1], limits, os.fstat(fds[-1]).st_size)
        _require(time.monotonic() < deadline, Refusal.RESOURCE_EXCEEDED)
        uri = _uri(policy.catalog.path)
        steps = 0

        def progress() -> int:
            nonlocal steps
            steps += 1
            return int(steps > limits.max_vm_steps or time.monotonic() >= deadline)

        _require(time.monotonic() < deadline, Refusal.RESOURCE_EXCEEDED)
        try:
            with closing(
                sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=min(
                        limits.busy_timeout_seconds,
                        max(0.001, deadline - time.monotonic()),
                    ),
                )
            ) as connection:
                connection.set_progress_handler(progress, 1)
                connection.execute("PRAGMA query_only=ON")
                connection.execute("BEGIN")
                dbs = connection.execute("PRAGMA database_list").fetchmany(3)
                _require(
                    len(dbs) == 1 and dbs[0][1:] == ("main", policy.catalog.path),
                    Refusal.PATH_UNSAFE,
                )
                pages = connection.execute("PRAGMA page_count").fetchone()
                _require(
                    pages is not None and 0 < pages[0] <= limits.max_pages,
                    Refusal.RESOURCE_EXCEEDED,
                )
                _check_schema(connection)
                tenants = connection.execute(
                    "SELECT id FROM tenants WHERE id = ?", (policy.tenant_id,)
                ).fetchmany(limits.max_entries + 1)
                _require(len(tenants) <= limits.max_entries, Refusal.RESOURCE_EXCEEDED)
                _require(
                    len(tenants) == 1 and tenants[0] == (policy.tenant_id,),
                    Refusal.CATALOG_MISMATCH,
                )
                databases = connection.execute(
                    "SELECT id, name, tenant_id FROM databases "
                    "WHERE name = ? AND tenant_id = ?",
                    (policy.database_name, policy.tenant_id),
                ).fetchmany(limits.max_entries + 1)
                _require(
                    len(databases) <= limits.max_entries, Refusal.RESOURCE_EXCEEDED
                )
                _require(
                    len(databases) == 1
                    and databases[0]
                    == (policy.database_uuid, policy.database_name, policy.tenant_id),
                    Refusal.CATALOG_MISMATCH,
                )
                _require(
                    all(_catalog_text(value) for value in databases[0]),
                    Refusal.UNSUPPORTED_SCHEMA,
                )
                for collection in (policy.db1, policy.db2):
                    rows = connection.execute(
                        "SELECT id, name FROM collections "
                        "WHERE database_id = ? AND name = ?",
                        (policy.database_uuid, collection.name),
                    ).fetchmany(limits.max_entries + 1)
                    _require(len(rows) <= limits.max_entries, Refusal.RESOURCE_EXCEEDED)
                    _require(
                        len(rows) == 1
                        and rows[0] == (collection.uuid, collection.name),
                        Refusal.CATALOG_MISMATCH,
                    )
                    _require(_uuid(rows[0][0]), Refusal.UNSUPPORTED_SCHEMA)
                connection.execute("ROLLBACK")
        except sqlite3.Error:
            if steps > limits.max_vm_steps or time.monotonic() >= deadline:
                raise _Denied(Refusal.RESOURCE_EXCEEDED) from None
            raise
        _require(
            time.monotonic() < deadline and steps <= limits.max_vm_steps,
            Refusal.RESOURCE_EXCEEDED,
        )
        for fd, item, directory in zip(
            fds,
            (policy.source, policy.chroma_root, policy.catalog),
            (True, True, False),
        ):
            _matches(fd, item, directory)
            named = _open_path(item.path, limits, directory=directory)
            try:
                _matches(named, item, directory)
            finally:
                os.close(named)
        _require(
            os.fstat(fds[-1]).st_size <= limits.max_file_bytes,
            Refusal.RESOURCE_EXCEEDED,
        )
        _sidecars(fds[1], limits, os.fstat(fds[-1]).st_size)
        _require(time.monotonic() < deadline, Refusal.RESOURCE_EXCEEDED)
    finally:
        for fd in fds:
            os.close(fd)


def observe(policy: Policy) -> Result:
    """Observe two pinned UUIDs in one catalog snapshot; startup stays disabled."""
    try:
        _observe(policy)
        return Result(Observation.VERIFIED_UUIDS)
    except _Denied as error:
        return Result(Observation.REFUSED, refusal=error.reason)
    except sqlite3.Error:
        return Result(Observation.REFUSED, refusal=Refusal.UNSTABLE)
    except (OSError, ValueError, TypeError, OverflowError):
        return Result(Observation.REFUSED, refusal=Refusal.ABSENT)
