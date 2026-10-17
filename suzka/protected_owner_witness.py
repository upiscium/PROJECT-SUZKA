"""Nonauthorizing, operator-custodied, one-shot OwnerWitness storage mechanics.

This helper only operates on explicitly selected PRE-EXISTING operator custody,
and makes no attempt to authenticate the caller or to authorize genesis.
Its publisher is a no-replace durable file operation, not a P0 state change,
worker admission, protected service credential, or historical integrity proof.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn
from uuid import UUID

from suzka.owner_witness_codec import (
    MAX_WITNESS_BYTES,
    OwnerWitness,
    WitnessDenied,
    canonicalize_witness_bytes,
    parse_witness_exact_bytes,
)


class StoreReason(StrEnum):
    INVALID_POLICY = "invalid_policy"
    INVALID_WITNESS = "invalid_witness"
    CUSTODY_CHANGED = "custody_changed"
    LOCK_MISSING = "lock_missing"
    LOCK_BUSY = "lock_busy"
    LOCK_CHANGED = "lock_changed"
    ALREADY_PUBLISHED = "already_published"
    WITNESS_MISSING = "witness_missing"
    WITNESS_MISMATCH = "witness_mismatch"
    ABANDONED_STAGE = "abandoned_stage"
    RESOURCE_EXCEEDED = "resource_exceeded"
    UNCERTAIN_COMMIT = "uncertain_commit"
    UNAVAILABLE = "unavailable"


class StoreDenied(RuntimeError):
    """Coarse refusal without private filenames or file contents."""

    def __init__(self, reason: StoreReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _deny(reason: StoreReason) -> NoReturn:
    raise StoreDenied(reason)


@dataclass(frozen=True, slots=True)
class DirectoryCustody:
    path: str
    device: int
    inode: int
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True, slots=True)
class LockFile:
    name: str
    device: int
    inode: int
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True, slots=True)
class Limits:
    max_path_bytes: int
    max_path_components: int
    max_directory_entries: int
    max_witness_bytes: int


@dataclass(frozen=True, slots=True)
class Policy:
    """Caller assertions: NOT independent protected installation authority."""

    custody: DirectoryCustody
    lock: LockFile
    installation_id: str
    record_uid: int
    record_gid: int
    record_mode: int
    limits: Limits


@dataclass(frozen=True, slots=True)
class Published:
    """Durable byte publication acknowledgment, never a startup token."""

    digest_sha256: str
    outcome: str = "COMMITTED_WITNESS"


@dataclass(frozen=True, slots=True)
class StoredWitness:
    witness: OwnerWitness
    digest_sha256: str
    status: str = "OBSERVED_WITNESS_NOT_AUTHORITY"


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_CREATE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _canonical_uuid(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        return str(UUID(value)) == value
    except (ValueError, AttributeError, TypeError):
        return False


def record_name(installation_id: str) -> str:
    if not _canonical_uuid(installation_id):
        _deny(StoreReason.INVALID_POLICY)
    return f"witness-{installation_id}.json"


def _absolute_parts(path: object, limits: Limits) -> tuple[str, ...]:
    if type(path) is not str:
        _deny(StoreReason.INVALID_POLICY)
    try:
        raw = os.fsencode(path)
    except UnicodeError:
        _deny(StoreReason.INVALID_POLICY)
    if not path.startswith("/") or path.startswith("//") or b"\x00" in raw:
        _deny(StoreReason.INVALID_POLICY)
    if not 0 < len(raw) <= limits.max_path_bytes:
        _deny(StoreReason.INVALID_POLICY)
    parts = tuple(path.split("/")[1:])
    if not 0 < len(parts) <= limits.max_path_components or any(
        not item or item in {".", ".."} for item in parts
    ):
        _deny(StoreReason.INVALID_POLICY)
    return parts


def _metadata(
    value: object, fields: tuple[str, ...], *, allow_zero: bool = False
) -> bool:
    return all(
        type(getattr(value, name, None)) is int
        and (0 if allow_zero or name in ("uid", "gid") else 1)
        <= getattr(value, name)
        < 2**63
        for name in fields
    )


def _validate(policy: Policy) -> str:
    if type(policy) is not Policy or type(policy.limits) is not Limits:
        _deny(StoreReason.INVALID_POLICY)
    lim = policy.limits
    if (
        not _metadata(
            lim,
            (
                "max_path_bytes",
                "max_path_components",
                "max_directory_entries",
                "max_witness_bytes",
            ),
        )
        or not 1 <= lim.max_path_bytes <= 4096
        or not 1 <= lim.max_path_components <= 256
        or not 1 <= lim.max_directory_entries <= 4096
        or not 1 <= lim.max_witness_bytes <= MAX_WITNESS_BYTES
    ):
        _deny(StoreReason.INVALID_POLICY)
    custody = policy.custody
    lock = policy.lock
    if type(custody) is not DirectoryCustody or type(lock) is not LockFile:
        _deny(StoreReason.INVALID_POLICY)
    _absolute_parts(custody.path, lim)
    if (
        not _metadata(custody, ("device", "inode", "uid", "gid"))
        or not _metadata(lock, ("device", "inode", "uid", "gid"))
        or lock.device != custody.device
        or type(custody.mode) is not int
        or custody.mode != 0o700
        or type(lock.mode) is not int
        or lock.mode != 0o600
        or type(lock.name) is not str
        or _NAME.fullmatch(lock.name) is None
        or type(policy.record_uid) is not int
        or not 0 <= policy.record_uid < 2**63
        or type(policy.record_gid) is not int
        or not 0 <= policy.record_gid < 2**63
        or policy.record_uid != custody.uid
        or policy.record_gid != custody.gid
        or type(policy.record_mode) is not int
        or policy.record_mode != 0o600
    ):
        _deny(StoreReason.INVALID_POLICY)
    name = record_name(policy.installation_id)
    if name == lock.name:
        _deny(StoreReason.INVALID_POLICY)
    return name


def _stat_tuple(s: os.stat_result) -> tuple[int, int, int, int, int]:
    return s.st_dev, s.st_ino, s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)


def _directory_tuple(value: DirectoryCustody) -> tuple[int, int, int, int, int]:
    return value.device, value.inode, value.uid, value.gid, value.mode


def _lock_tuple(value: LockFile) -> tuple[int, int, int, int, int]:
    return value.device, value.inode, value.uid, value.gid, value.mode


def _open_custody(policy: Policy) -> int:
    parts = _absolute_parts(policy.custody.path, policy.limits)
    fd = os.open("/", _DIR)
    try:
        for component in parts:
            next_fd = os.open(component, _DIR, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        observed = os.fstat(fd)
        if not stat.S_ISDIR(observed.st_mode) or _stat_tuple(
            observed
        ) != _directory_tuple(policy.custody):
            _deny(StoreReason.CUSTODY_CHANGED)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_custody(policy: Policy, dfd: int) -> None:
    current = os.fstat(dfd)
    if not stat.S_ISDIR(current.st_mode) or _stat_tuple(current) != _directory_tuple(
        policy.custody
    ):
        _deny(StoreReason.CUSTODY_CHANGED)
    other = _open_custody(policy)
    try:
        if _stat_tuple(os.fstat(other)) != _stat_tuple(current):
            _deny(StoreReason.CUSTODY_CHANGED)
    finally:
        os.close(other)


def _check_lock(policy: Policy, dfd: int, lockfd: int) -> None:
    expected = _lock_tuple(policy.lock)
    saved = os.fstat(lockfd)
    seen = os.stat(policy.lock.name, dir_fd=dfd, follow_symlinks=False)
    if (
        not stat.S_ISREG(saved.st_mode)
        or not stat.S_ISREG(seen.st_mode)
        or saved.st_nlink != 1
        or seen.st_nlink != 1
        or _stat_tuple(saved) != expected
        or _stat_tuple(seen) != expected
    ):
        _deny(StoreReason.LOCK_CHANGED)


def _check_abandoned(policy: Policy, dfd: int, name: str) -> int:
    prefix = f".{name}.stage-"
    entries = 0
    with os.scandir(dfd) as it:
        for item in it:
            entries += 1
            if entries > policy.limits.max_directory_entries:
                _deny(StoreReason.RESOURCE_EXCEEDED)
            if item.name.startswith(prefix):
                _deny(StoreReason.ABANDONED_STAGE)
    return entries


def _read_exact(
    policy: Policy, dfd: int, name: str, expected: str | None
) -> StoredWitness:
    try:
        fd = os.open(name, _READ, dir_fd=dfd)
    except FileNotFoundError:
        _deny(StoreReason.WITNESS_MISSING)
    try:
        first = os.fstat(fd)
        if (
            not stat.S_ISREG(first.st_mode)
            or first.st_nlink != 1
            or first.st_dev != policy.custody.device
            or first.st_uid != policy.record_uid
            or first.st_gid != policy.record_gid
            or stat.S_IMODE(first.st_mode) != policy.record_mode
        ):
            _deny(StoreReason.WITNESS_MISMATCH)
        if not 0 < first.st_size <= policy.limits.max_witness_bytes:
            _deny(StoreReason.RESOURCE_EXCEEDED)
        data = bytearray()
        while len(data) < first.st_size:
            piece = os.read(fd, min(first.st_size - len(data), 8192))
            if not piece:
                _deny(StoreReason.WITNESS_MISMATCH)
            data.extend(piece)
        end = os.fstat(fd)
        named = os.stat(name, dir_fd=dfd, follow_symlinks=False)
        if _stat_tuple(first) != _stat_tuple(end) or _stat_tuple(first) != _stat_tuple(
            named
        ):
            _deny(StoreReason.WITNESS_MISMATCH)
        if end.st_size != first.st_size or end.st_mtime_ns != first.st_mtime_ns:
            _deny(StoreReason.WITNESS_MISMATCH)
        try:
            obj = parse_witness_exact_bytes(bytes(data))
        except WitnessDenied:
            _deny(StoreReason.INVALID_WITNESS)
        digest = hashlib.sha256(data).hexdigest()
        if obj.installation_id != policy.installation_id:
            _deny(StoreReason.WITNESS_MISMATCH)
        if expected is not None and digest != expected:
            _deny(StoreReason.WITNESS_MISMATCH)
        return StoredWitness(obj, digest)
    finally:
        os.close(fd)


@contextmanager
def _locked(policy: Policy) -> Iterator[tuple[int, int, str, int]]:
    name = _validate(policy)
    dfd = -1
    lockfd = -1
    try:
        dfd = _open_custody(policy)
        try:
            lockfd = os.open(policy.lock.name, _READ, dir_fd=dfd)
        except FileNotFoundError:
            _deny(StoreReason.LOCK_MISSING)
        _check_lock(policy, dfd, lockfd)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _deny(StoreReason.LOCK_BUSY)
        _check_lock(policy, dfd, lockfd)
        _check_custody(policy, dfd)
        entry_count = _check_abandoned(policy, dfd, name)
        yield dfd, lockfd, name, entry_count
    except StoreDenied:
        raise
    except (OSError, UnicodeError, ValueError, OverflowError):
        _deny(StoreReason.UNAVAILABLE)
    finally:
        if lockfd >= 0:
            os.close(lockfd)
        if dfd >= 0:
            os.close(dfd)


def observe(policy: Policy, *, expected_sha256: str) -> StoredWitness:
    """Verify exact canonical byte/digest and protected file metadata, not authority."""
    if (
        type(expected_sha256) is not str
        or re.fullmatch(r"[a-f0-9]{64}\Z", expected_sha256) is None
    ):
        _deny(StoreReason.INVALID_POLICY)
    with _locked(policy) as (dfd, _lockfd, name, _entry_count):
        result = _read_exact(policy, dfd, name, expected_sha256)
        _check_custody(policy, dfd)
        return result


def publish_once(policy: Policy, witness: OwnerWitness) -> Published:
    """Publish one exact witness with no replacement; cannot authorize genesis.

    First-genesis operator must independently prove no old P0, no old witness,
    and no prior owner state, then durably consume P0 GENESIS_IN_PROGRESS
    BEFORE any owner effect. This function does NONE of those checks.
    """
    _validate(policy)
    if type(witness) is not OwnerWitness:
        _deny(StoreReason.INVALID_WITNESS)
    try:
        payload = canonicalize_witness_bytes(witness)
    except WitnessDenied:
        _deny(StoreReason.INVALID_WITNESS)
    if witness.installation_id != policy.installation_id:
        _deny(StoreReason.INVALID_WITNESS)
    with _locked(policy) as (dfd, lockfd, name, entry_count):
        if len(payload) > policy.limits.max_witness_bytes:
            _deny(StoreReason.RESOURCE_EXCEEDED)
        try:
            os.stat(name, dir_fd=dfd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _deny(StoreReason.ALREADY_PUBLISHED)
        # Reserve two slots: transient stage AND final hard-link. Otherwise
        # a successful COMMITTED_WITNESS may exceed the accepted directory
        # budget and become unreadable by the next observe() call.
        if entry_count > policy.limits.max_directory_entries - 2:
            _deny(StoreReason.RESOURCE_EXCEEDED)
        # A new private stage is only created INSIDE explicitly preprovisioned
        # operator custody, never inside the service-owned Source.
        stage = f".{name}.stage-{secrets.token_hex(12)}"
        stagefd = -1
        staged = False
        try:
            stagefd = os.open(stage, _CREATE, 0o600, dir_fd=dfd)
            staged = True
            # os.open creation mode is filtered by the process umask.
            os.fchmod(stagefd, policy.record_mode)
            view = memoryview(payload)
            while view:
                written = os.write(stagefd, view)
                if written <= 0:
                    raise OSError("short stage write")
                view = view[written:]
            meta = os.fstat(stagefd)
            if (
                not stat.S_ISREG(meta.st_mode)
                or meta.st_nlink != 1
                or meta.st_dev != policy.custody.device
                or meta.st_uid != policy.record_uid
                or meta.st_gid != policy.record_gid
                or stat.S_IMODE(meta.st_mode) != policy.record_mode
            ):
                _deny(StoreReason.CUSTODY_CHANGED)
            os.fsync(stagefd)
            os.close(stagefd)
            stagefd = -1
            _check_lock(policy, dfd, lockfd)
            _check_custody(policy, dfd)
            # POSIX hard-link publishing fails if the target exists, even
            # with two contenders or a crash. Never replace the existing witness.
            os.link(stage, name, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False)
            os.fsync(dfd)
            os.unlink(stage, dir_fd=dfd)
            os.fsync(dfd)
            result = _read_exact(policy, dfd, name, hashlib.sha256(payload).hexdigest())
            _check_custody(policy, dfd)
            return Published(result.digest_sha256)
        except StoreDenied:
            if staged:
                # Once a stage exists, its partial effects cannot be presumed
                # durably absent or safe to retry. Leave it for operator review.
                _deny(StoreReason.UNCERTAIN_COMMIT)
            raise
        except OSError:
            if staged:
                _deny(StoreReason.UNCERTAIN_COMMIT)
            _deny(StoreReason.UNAVAILABLE)
        finally:
            if stagefd >= 0:
                os.close(stagefd)
