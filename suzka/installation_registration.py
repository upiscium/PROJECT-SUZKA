"""Nonauthorizing, caller-custodied local registration record mechanics.

W0-I2-P0 NEVER authenticates an installation, issues a boot permit, imports an
application/source owner, or starts a process. COMMITTED_RECORD is ONLY a
local persistence outcome under *unverified caller-provided* custody facts.
The future protected operator/supervisor is a separate Human-gated project.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import Enum
import fcntl
import json
import os
import re
import secrets
import stat
from typing import Iterator, NoReturn
from uuid import UUID


class RegistrationReason(str, Enum):
    INVALID_POLICY = "INVALID_POLICY"
    INVALID_RECORD = "INVALID_RECORD"
    RECORD_MISSING = "RECORD_MISSING"
    RECORD_EXISTS = "RECORD_EXISTS"
    RECORD_MISMATCH = "RECORD_MISMATCH"
    STALE_REVISION = "STALE_REVISION"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    RESOURCE_EXCEEDED = "RESOURCE_EXCEEDED"
    CUSTODY_MISMATCH = "CUSTODY_MISMATCH"
    CUSTODY_UNAVAILABLE = "CUSTODY_UNAVAILABLE"
    LOCK_UNAVAILABLE = "LOCK_UNAVAILABLE"
    UNCERTAIN_COMMIT = "UNCERTAIN_COMMIT"
    ABANDONED_STAGE = "ABANDONED_STAGE"


class RegistrationDenied(RuntimeError):
    """Fail closed without including private paths, record bytes or data."""

    def __init__(self, reason: RegistrationReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _deny(reason: RegistrationReason) -> NoReturn:
    raise RegistrationDenied(reason)


class RegistrationState(str, Enum):
    FIRST_INSTALL_PREPARED = "FIRST_INSTALL_PREPARED"
    GENESIS_IN_PROGRESS = "GENESIS_IN_PROGRESS"
    CONTINUATION_READY = "CONTINUATION_READY"
    START_CLAIMED = "START_CLAIMED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class RegistrationOutcome(str, Enum):
    """A write acknowledgement, NEVER a credential or startup permit."""

    COMMITTED_RECORD = "COMMITTED_RECORD"


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_ALLOWED = {
    RegistrationState.FIRST_INSTALL_PREPARED: frozenset(
        {RegistrationState.GENESIS_IN_PROGRESS, RegistrationState.RECOVERY_REQUIRED}
    ),
    RegistrationState.GENESIS_IN_PROGRESS: frozenset(
        {RegistrationState.CONTINUATION_READY, RegistrationState.RECOVERY_REQUIRED}
    ),
    RegistrationState.CONTINUATION_READY: frozenset(
        {RegistrationState.START_CLAIMED, RegistrationState.RECOVERY_REQUIRED}
    ),
    RegistrationState.START_CLAIMED: frozenset(
        {RegistrationState.CONTINUATION_READY, RegistrationState.RECOVERY_REQUIRED}
    ),
    RegistrationState.RECOVERY_REQUIRED: frozenset(),
}
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _canonical_uuid(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        return str(UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def _hex_digest(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _utf8_length(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeError:
        _deny(RegistrationReason.INVALID_RECORD)


def _absolute_components(
    raw: object, *, max_bytes: int, max_count: int
) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw.startswith("/") or "\x00" in raw:
        _deny(RegistrationReason.INVALID_POLICY)
    try:
        path_bytes = os.fsencode(raw)
    except UnicodeError:
        _deny(RegistrationReason.INVALID_POLICY)
    if len(path_bytes) > max_bytes or raw.endswith("/") or "//" in raw:
        _deny(RegistrationReason.INVALID_POLICY)
    components = tuple(raw.split("/")[1:])
    if (
        not components
        or len(components) > max_count
        or any(not p or p in {".", ".."} for p in components)
    ):
        _deny(RegistrationReason.INVALID_POLICY)
    return components


@dataclass(frozen=True, slots=True)
class RegistryCustody:
    """Explicit, UNAUTHENTICATED custody expectations; no policy defaults."""

    directory: str
    directory_dev: int
    directory_ino: int
    directory_uid: int
    directory_gid: int
    directory_mode: int
    lock_name: str
    lock_dev: int
    lock_ino: int
    lock_uid: int
    lock_gid: int
    lock_mode: int
    record_name: str
    record_uid: int
    record_gid: int
    record_mode: int
    max_record_bytes: int
    max_path_bytes: int
    max_path_components: int
    max_directory_entries: int

    def __post_init__(self) -> None:
        for name in (
            "directory_dev",
            "directory_ino",
            "directory_uid",
            "directory_gid",
            "directory_mode",
            "lock_dev",
            "lock_ino",
            "lock_uid",
            "lock_gid",
            "lock_mode",
            "record_uid",
            "record_gid",
            "record_mode",
            "max_record_bytes",
            "max_path_bytes",
            "max_path_components",
            "max_directory_entries",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _deny(RegistrationReason.INVALID_POLICY)
        if (
            self.directory_ino == 0
            or self.lock_ino == 0
            or not 1 <= self.max_record_bytes <= 1024 * 1024
            or not 1 <= self.max_path_bytes <= 4096
            or not 1 <= self.max_path_components <= 256
            or not 1 <= self.max_directory_entries <= 4096
        ):
            _deny(RegistrationReason.INVALID_POLICY)
        _absolute_components(
            self.directory,
            max_bytes=self.max_path_bytes,
            max_count=self.max_path_components,
        )
        if (
            type(self.lock_name) is not str
            or type(self.record_name) is not str
            or not _FILENAME.fullmatch(self.lock_name)
            or not _FILENAME.fullmatch(self.record_name)
            or self.lock_name == self.record_name
        ):
            _deny(RegistrationReason.INVALID_POLICY)
        # Neither records nor locks may be broadly accessible at the
        # final owner-controlled directory level. This does NOT validate
        # independent operator authority for the directory or its ancestors.
        for mode in (self.directory_mode, self.lock_mode, self.record_mode):
            if mode > 0o7777 or (mode & 0o077):
                _deny(RegistrationReason.INVALID_POLICY)
        if self.directory_mode & 0o700 != 0o700:
            _deny(RegistrationReason.INVALID_POLICY)
        if self.lock_mode & 0o600 != 0o600:
            _deny(RegistrationReason.INVALID_POLICY)
        if self.record_mode & 0o600 != 0o600:
            _deny(RegistrationReason.INVALID_POLICY)


@dataclass(frozen=True, slots=True)
class RegistrationRecord:
    """No state, digest or caller token here is independently authenticated."""

    schema_version: int
    state: RegistrationState
    revision: int
    installation_id: str
    first_install_attempt: str
    config_sha256: str
    installed_code_sha256: str
    source_root: str
    source_root_dev: int
    source_root_ino: int
    asgi_target: str
    owner_baseline_sha256: str | None
    active_claim_id: str | None
    last_claim_id: str | None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.state) is not RegistrationState
            or type(self.revision) is not int
            or not 1 <= self.revision < 2**63
            or not _canonical_uuid(self.installation_id)
            or not _canonical_uuid(self.first_install_attempt)
            or not _hex_digest(self.config_sha256)
            or not _hex_digest(self.installed_code_sha256)
            or type(self.source_root_dev) is not int
            or type(self.source_root_ino) is not int
            or self.source_root_dev < 0
            or self.source_root_ino <= 0
            or type(self.source_root) is not str
            or _utf8_length(self.source_root) > 4096
            or type(self.asgi_target) is not str
            or len(self.asgi_target) > 200
            or not re.fullmatch(
                r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*:[A-Za-z_][A-Za-z_0-9]*",
                self.asgi_target,
            )
        ):
            _deny(RegistrationReason.INVALID_RECORD)
        # This check is only lexical identity, NOT path custody/attestation.
        try:
            _absolute_components(self.source_root, max_bytes=4096, max_count=256)
        except RegistrationDenied:
            _deny(RegistrationReason.INVALID_RECORD)
        if any(
            value is not None and not _hex_digest(value)
            for value in (self.owner_baseline_sha256,)
        ) or any(
            value is not None and not _canonical_uuid(value)
            for value in (self.active_claim_id, self.last_claim_id)
        ):
            _deny(RegistrationReason.INVALID_RECORD)
        if self.state in {
            RegistrationState.FIRST_INSTALL_PREPARED,
            RegistrationState.GENESIS_IN_PROGRESS,
        } and (
            self.owner_baseline_sha256 is not None
            or self.active_claim_id is not None
            or self.last_claim_id is not None
        ):
            _deny(RegistrationReason.INVALID_RECORD)
        if (
            self.state
            in {
                RegistrationState.CONTINUATION_READY,
                RegistrationState.START_CLAIMED,
            }
            and self.owner_baseline_sha256 is None
        ):
            _deny(RegistrationReason.INVALID_RECORD)
        if (self.state is RegistrationState.START_CLAIMED) != (
            self.active_claim_id is not None
        ):
            _deny(RegistrationReason.INVALID_RECORD)
        if self.active_claim_id == self.first_install_attempt:
            _deny(RegistrationReason.INVALID_RECORD)

    def to_json_bytes(self, *, max_bytes: int) -> bytes:
        obj = {item.name: getattr(self, item.name) for item in fields(self)}
        obj["state"] = self.state.value
        raw = json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        if len(raw) > max_bytes:
            _deny(RegistrationReason.RESOURCE_EXCEEDED)
        return raw

    @classmethod
    def from_json_bytes(cls, data: bytes, *, max_bytes: int) -> RegistrationRecord:
        if type(max_bytes) is not int or not 1 <= max_bytes <= 1024 * 1024:
            _deny(RegistrationReason.INVALID_POLICY)
        if type(data) is not bytes:
            _deny(RegistrationReason.INVALID_RECORD)
        if len(data) > max_bytes:
            _deny(RegistrationReason.RESOURCE_EXCEEDED)

        def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            out: dict[str, object] = {}
            for key, value in pairs:
                if key in out:
                    _deny(RegistrationReason.INVALID_RECORD)
                out[key] = value
            return out

        def no_constant(_: str) -> None:
            _deny(RegistrationReason.INVALID_RECORD)

        try:
            raw = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=no_duplicates,
                parse_constant=no_constant,
            )
            if type(raw) is not dict or set(raw) != {f.name for f in fields(cls)}:
                _deny(RegistrationReason.INVALID_RECORD)
            state = raw["state"]
            if type(state) is not str:
                _deny(RegistrationReason.INVALID_RECORD)
            raw["state"] = RegistrationState(state)
            return cls(**raw)
        except RegistrationDenied:
            raise
        except (UnicodeError, ValueError, TypeError, KeyError, RecursionError):
            _deny(RegistrationReason.INVALID_RECORD)


def _identity_fields(record: RegistrationRecord) -> tuple[object, ...]:
    return (
        record.schema_version,
        record.installation_id,
        record.first_install_attempt,
        record.config_sha256,
        record.installed_code_sha256,
        record.source_root,
        record.source_root_dev,
        record.source_root_ino,
        record.asgi_target,
    )


def _validate_transition(old: RegistrationRecord, new: RegistrationRecord) -> None:
    if new.revision != old.revision + 1 or new.revision >= 2**63:
        _deny(RegistrationReason.STALE_REVISION)
    if (
        _identity_fields(old) != _identity_fields(new)
        or new.state not in _ALLOWED[old.state]
    ):
        _deny(RegistrationReason.INVALID_TRANSITION)
    if new.state is RegistrationState.RECOVERY_REQUIRED:
        if (
            new.owner_baseline_sha256 != old.owner_baseline_sha256
            or new.active_claim_id is not None
            or new.last_claim_id
            != (old.active_claim_id if old.active_claim_id else old.last_claim_id)
        ):
            _deny(RegistrationReason.INVALID_TRANSITION)
    elif old.state is RegistrationState.FIRST_INSTALL_PREPARED:
        if new.owner_baseline_sha256 is not None:
            _deny(RegistrationReason.INVALID_TRANSITION)
    elif old.state is RegistrationState.GENESIS_IN_PROGRESS:
        if new.owner_baseline_sha256 is None or new.last_claim_id is not None:
            _deny(RegistrationReason.INVALID_TRANSITION)
    elif old.state is RegistrationState.CONTINUATION_READY:
        if (
            new.owner_baseline_sha256 != old.owner_baseline_sha256
            or new.last_claim_id != old.last_claim_id
            or new.active_claim_id
            in {old.first_install_attempt, old.last_claim_id, None}
        ):
            _deny(RegistrationReason.INVALID_TRANSITION)
    elif old.state is RegistrationState.START_CLAIMED and (
        new.owner_baseline_sha256 != old.owner_baseline_sha256
        or new.active_claim_id is not None
        or new.last_claim_id != old.active_claim_id
    ):
        _deny(RegistrationReason.INVALID_TRANSITION)


def _stat_tuple(s: os.stat_result) -> tuple[int, int, int, int, int]:
    return (s.st_dev, s.st_ino, s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode))


def _open_protected_directory(c: RegistryCustody) -> int:
    parts = _absolute_components(
        c.directory, max_bytes=c.max_path_bytes, max_count=c.max_path_components
    )
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in parts:
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        s = os.fstat(fd)
        if not stat.S_ISDIR(s.st_mode) or _stat_tuple(s) != (
            c.directory_dev,
            c.directory_ino,
            c.directory_uid,
            c.directory_gid,
            c.directory_mode,
        ):
            _deny(RegistrationReason.CUSTODY_MISMATCH)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _assert_directory(c: RegistryCustody, dfd: int) -> None:
    now = os.fstat(dfd)
    if _stat_tuple(now) != (
        c.directory_dev,
        c.directory_ino,
        c.directory_uid,
        c.directory_gid,
        c.directory_mode,
    ) or not stat.S_ISDIR(now.st_mode):
        _deny(RegistrationReason.CUSTODY_MISMATCH)
    other = _open_protected_directory(c)
    try:
        if (os.fstat(other).st_dev, os.fstat(other).st_ino) != (now.st_dev, now.st_ino):
            _deny(RegistrationReason.CUSTODY_MISMATCH)
    finally:
        os.close(other)


def _assert_lock(c: RegistryCustody, dfd: int, lockfd: int) -> None:
    s = os.fstat(lockfd)
    name = os.stat(c.lock_name, dir_fd=dfd, follow_symlinks=False)
    expected = (c.lock_dev, c.lock_ino, c.lock_uid, c.lock_gid, c.lock_mode)
    if (
        not stat.S_ISREG(s.st_mode)
        or s.st_nlink != 1
        or s.st_size != 0
        or _stat_tuple(s) != expected
        or _stat_tuple(name) != expected
        or not stat.S_ISREG(name.st_mode)
    ):
        _deny(RegistrationReason.CUSTODY_MISMATCH)


def _assert_no_abandoned_stage(c: RegistryCustody, dfd: int) -> int:
    """Fail closed after an interrupted stage; never reap unknown files.

    A separate operator recovery workflow must classify and clean orphan
    temp artifacts. Prevent repeated failed starts from creating unbounded
    further temp files while an old stage remains.
    """
    prefix = "." + c.record_name + "."
    # On supported CPython, os.scandir(fd) does NOT consume its caller's
    # directory descriptor; the context closes only its own iterator.
    # The caller retains and closes dfd. Characterized by the adjacent
    # explicit fd-survival regression test.
    number = 0
    with os.scandir(dfd) as entries:
        for number, entry in enumerate(entries, 1):
            if number > c.max_directory_entries:
                _deny(RegistrationReason.RESOURCE_EXCEEDED)
            if entry.name.startswith(prefix) and entry.name.endswith(".tmp"):
                _deny(RegistrationReason.ABANDONED_STAGE)
        return number


@contextmanager
def _locked(c: RegistryCustody, *, mutation: bool = False) -> Iterator[tuple[int, int]]:
    if type(c) is not RegistryCustody:
        _deny(RegistrationReason.INVALID_POLICY)
    dfd: int | None = None
    lockfd: int | None = None
    try:
        dfd = _open_protected_directory(c)
        lockfd = os.open(
            c.lock_name,
            os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=dfd,
        )
        _assert_lock(c, dfd, lockfd)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _deny(RegistrationReason.LOCK_UNAVAILABLE)
        _assert_lock(c, dfd, lockfd)
        _assert_directory(c, dfd)
        _assert_no_abandoned_stage(c, dfd)
        yield dfd, lockfd
        # The mutation may have been published by the yielded caller.
        # Failure of an AFTER-yield check must not mislabel its
        # potentially durable effect as an ordinary custody refusal.
        try:
            _assert_lock(c, dfd, lockfd)
            _assert_directory(c, dfd)
        except (RegistrationDenied, OSError, ValueError, TypeError):
            if mutation:
                _deny(RegistrationReason.UNCERTAIN_COMMIT)
            raise
    except RegistrationDenied:
        raise
    except (OSError, ValueError, TypeError):
        _deny(RegistrationReason.CUSTODY_UNAVAILABLE)
    finally:
        if lockfd is not None:
            os.close(lockfd)
        if dfd is not None:
            os.close(dfd)


def _read_record(
    c: RegistryCustody, dfd: int, *, missing_ok: bool
) -> tuple[RegistrationRecord | None, os.stat_result | None]:
    try:
        fd = os.open(c.record_name, _FILE_READ_FLAGS, dir_fd=dfd)
    except FileNotFoundError:
        if missing_ok:
            return None, None
        _deny(RegistrationReason.RECORD_MISSING)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_dev != c.directory_dev
            or before.st_uid != c.record_uid
            or before.st_gid != c.record_gid
            or stat.S_IMODE(before.st_mode) != c.record_mode
        ):
            _deny(RegistrationReason.CUSTODY_MISMATCH)
        if before.st_size > c.max_record_bytes:
            _deny(RegistrationReason.RESOURCE_EXCEEDED)
        if before.st_size <= 0:
            _deny(RegistrationReason.INVALID_RECORD)
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            part = os.read(fd, min(remaining, 64 * 1024))
            if not part:
                _deny(RegistrationReason.RECORD_MISMATCH)
            chunks.append(part)
            remaining -= len(part)
        data = b"".join(chunks)
        if len(data) != before.st_size:
            _deny(RegistrationReason.RECORD_MISMATCH)
        after = os.fstat(fd)
        if len(data) > c.max_record_bytes:
            _deny(RegistrationReason.RESOURCE_EXCEEDED)
        name = os.stat(c.record_name, dir_fd=dfd, follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (
            name.st_dev,
            name.st_ino,
            name.st_size,
            name.st_mtime_ns,
            name.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _deny(RegistrationReason.RECORD_MISMATCH)
        return RegistrationRecord.from_json_bytes(
            data, max_bytes=c.max_record_bytes
        ), after
    finally:
        os.close(fd)


def _assert_old_record(
    c: RegistryCustody, dfd: int, old: os.stat_result | None
) -> None:
    try:
        now = os.stat(c.record_name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        if old is None:
            return
        _deny(RegistrationReason.RECORD_MISMATCH)
    if old is None or (
        now.st_dev,
        now.st_ino,
        now.st_mtime_ns,
        now.st_ctime_ns,
        now.st_size,
    ) != (old.st_dev, old.st_ino, old.st_mtime_ns, old.st_ctime_ns, old.st_size):
        _deny(RegistrationReason.RECORD_MISMATCH)


def _publish(
    c: RegistryCustody,
    dfd: int,
    lockfd: int,
    record: RegistrationRecord,
    old_stat: os.stat_result | None,
) -> RegistrationOutcome:
    payload = record.to_json_bytes(max_bytes=c.max_record_bytes)
    # A new record must remain readable under this SAME explicit policy.
    # Never acknowledge initialization if adding the record exceeds the
    # next operation's bounded directory scan.
    if (
        old_stat is None
        and _assert_no_abandoned_stage(c, dfd) >= c.max_directory_entries
    ):
        _deny(RegistrationReason.RESOURCE_EXCEEDED)
    temp_name = "." + c.record_name + "." + secrets.token_hex(12) + ".tmp"
    tempfd: int | None = None
    created_identity: tuple[int, int] | None = None
    renamed = False
    try:
        tempfd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            c.record_mode,
            dir_fd=dfd,
        )
        initial_tempstat = os.fstat(tempfd)
        created_identity = (initial_tempstat.st_dev, initial_tempstat.st_ino)
        os.fchmod(tempfd, c.record_mode)
        tempstat = os.fstat(tempfd)
        if (
            not stat.S_ISREG(tempstat.st_mode)
            or tempstat.st_dev != c.directory_dev
            or tempstat.st_uid != c.record_uid
            or tempstat.st_gid != c.record_gid
            or stat.S_IMODE(tempstat.st_mode) != c.record_mode
            or tempstat.st_nlink != 1
        ):
            _deny(RegistrationReason.CUSTODY_MISMATCH)
        written = 0
        while written < len(payload):
            n = os.write(tempfd, payload[written:])
            if n <= 0:
                _deny(RegistrationReason.CUSTODY_UNAVAILABLE)
            written += n
        os.fsync(tempfd)
        os.close(tempfd)
        tempfd = None
        _assert_lock(c, dfd, lockfd)
        _assert_directory(c, dfd)
        _assert_old_record(c, dfd, old_stat)
        os.replace(temp_name, c.record_name, src_dir_fd=dfd, dst_dir_fd=dfd)
        renamed = True
        os.fsync(dfd)
        _assert_lock(c, dfd, lockfd)
        _assert_directory(c, dfd)
        published, _ = _read_record(c, dfd, missing_ok=False)
        if published != record:
            _deny(RegistrationReason.UNCERTAIN_COMMIT)
        return RegistrationOutcome.COMMITTED_RECORD
    except RegistrationDenied:
        if renamed:
            _deny(RegistrationReason.UNCERTAIN_COMMIT)
        raise
    except (OSError, ValueError, TypeError):
        _deny(
            RegistrationReason.UNCERTAIN_COMMIT
            if renamed
            else RegistrationReason.CUSTODY_UNAVAILABLE
        )
    finally:
        if tempfd is not None:
            os.close(tempfd)
        if not renamed and created_identity is not None:
            # O_EXCL failure must NEVER remove a pre-existing name. Verify
            # identity of our own staged inode before cleaning it up. A
            # concurrent attacker of the asserted directory custody is
            # outside P0 and remains a reason to deny, never to authorize.
            try:
                now = os.stat(temp_name, dir_fd=dfd, follow_symlinks=False)
                if (now.st_dev, now.st_ino) == created_identity:
                    os.unlink(temp_name, dir_fd=dfd)
            except (FileNotFoundError, OSError):
                pass  # Retained stage forces a future ABANDONED_STAGE denial.


class InstallationRegistrationStore:
    """Caller-asserted record mechanics only; never an I2 authority."""

    def __init__(self, custody: RegistryCustody) -> None:
        if type(custody) is not RegistryCustody:
            _deny(RegistrationReason.INVALID_POLICY)
        self._custody = custody

    def read(self) -> RegistrationRecord:
        """A data snapshot, not an authorization decision."""
        with _locked(self._custody) as (dfd, _lockfd):
            record, _ = _read_record(self._custody, dfd, missing_ok=False)
            assert record is not None
            return record

    def initialize(self, prepared: RegistrationRecord) -> RegistrationOutcome:
        """Explicit first record. Mere absent record is NEVER an install permit."""
        if type(prepared) is not RegistrationRecord:
            _deny(RegistrationReason.INVALID_RECORD)
        if (
            prepared.state is not RegistrationState.FIRST_INSTALL_PREPARED
            or prepared.revision != 1
        ):
            _deny(RegistrationReason.INVALID_TRANSITION)
        with _locked(self._custody, mutation=True) as (dfd, lockfd):
            previous, _ = _read_record(self._custody, dfd, missing_ok=True)
            if previous is not None:
                _deny(RegistrationReason.RECORD_EXISTS)
            return _publish(self._custody, dfd, lockfd, prepared, None)

    def transition(
        self, expected: RegistrationRecord, target: RegistrationRecord
    ) -> RegistrationOutcome:
        """CAS only. The record is not a trusted root/controller boot claim."""
        if (
            type(expected) is not RegistrationRecord
            or type(target) is not RegistrationRecord
        ):
            _deny(RegistrationReason.INVALID_RECORD)
        _validate_transition(expected, target)
        with _locked(self._custody, mutation=True) as (dfd, lockfd):
            current, old_stat = _read_record(self._custody, dfd, missing_ok=False)
            assert current is not None
            if current.revision != expected.revision:
                _deny(RegistrationReason.STALE_REVISION)
            if current != expected:
                _deny(RegistrationReason.RECORD_MISMATCH)
            return _publish(self._custody, dfd, lockfd, target, old_stat)
