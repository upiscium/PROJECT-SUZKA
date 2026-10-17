"""Operator-supplied, Source-physical-identity keyed lifetime flock mechanics.

This module does NOT authenticate operator custody, establish worker death, or
issue boot permission. A later independently authorized supervisor must hold a
LiveClaim through positive process/cgroup drain and trusted P0 transitions.
No lock, Source, directory or registration object is created by acquire().
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn


class ClaimReason(StrEnum):
    INVALID_POLICY = "invalid_policy"
    PATH_UNSAFE = "path_unsafe"
    SOURCE_CHANGED = "source_changed"
    CUSTODY_CHANGED = "custody_changed"
    LOCK_MISSING = "lock_missing"
    LOCK_BUSY = "lock_busy"
    LOCK_CHANGED = "lock_changed"
    UNSTABLE = "unstable"
    CLOSED = "closed"


class ClaimDenied(RuntimeError):
    """Path-free denial: no file bytes or caller-selected paths in errors."""

    def __init__(self, reason: ClaimReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _deny(reason: ClaimReason) -> NoReturn:
    raise ClaimDenied(reason)


@dataclass(frozen=True, slots=True)
class Directory:
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


@dataclass(frozen=True, slots=True)
class Policy:
    """Only asserted metadata, never independently authenticated trust roots."""

    source: Directory
    custody: Directory
    lock: LockFile
    limits: Limits


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_LOCK_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def lock_name(device: int, inode: int) -> str:
    """Physical dev/inode-based global key shared by distinct installations.

    The later operator must pre-provision exactly this lock name in a SINGLE
    shared protected custody directory for all managed installations.
    """
    if (
        type(device) is not int
        or type(inode) is not int
        or not 0 <= device < 2**63
        or not 0 < inode < 2**63
    ):
        _deny(ClaimReason.INVALID_POLICY)
    digest = hashlib.sha256(f"source-physical-v1:{device}:{inode}".encode()).hexdigest()
    return f"source-v1-{digest}.lock"


def _parts(path: object, limits: Limits) -> tuple[str, ...]:
    if type(path) is not str:
        _deny(ClaimReason.INVALID_POLICY)
    try:
        encoded = os.fsencode(path)
    except UnicodeError:
        _deny(ClaimReason.PATH_UNSAFE)
    if not path.startswith("/") or path.startswith("//") or b"\x00" in encoded:
        _deny(ClaimReason.PATH_UNSAFE)
    if not 1 <= len(encoded) <= limits.max_path_bytes:
        _deny(ClaimReason.PATH_UNSAFE)
    parts = tuple(path.split("/")[1:])
    if not 1 <= len(parts) <= limits.max_path_components or any(
        part in ("", ".", "..") for part in parts
    ):
        _deny(ClaimReason.PATH_UNSAFE)
    return parts


def _validate_directory(
    value: Directory, limits: Limits, *, custody: bool
) -> tuple[str, ...]:
    if type(value) is not Directory:
        _deny(ClaimReason.INVALID_POLICY)
    parts = _parts(value.path, limits)
    if not all(
        type(getattr(value, item)) is int and 0 <= getattr(value, item) < 2**63
        for item in ("device", "inode", "uid", "gid")
    ):
        _deny(ClaimReason.INVALID_POLICY)
    if (custody and not value.device) or not value.inode:
        _deny(ClaimReason.INVALID_POLICY)
    if type(value.mode) is not int or not 0 <= value.mode <= 0o7777:
        _deny(ClaimReason.INVALID_POLICY)
    if custody and (value.mode & 0o077 or value.mode & 0o700 != 0o700):
        _deny(ClaimReason.INVALID_POLICY)
    return parts


def _validate(policy: Policy) -> None:
    if type(policy) is not Policy or type(policy.limits) is not Limits:
        _deny(ClaimReason.INVALID_POLICY)
    limits = policy.limits
    if (
        type(limits.max_path_bytes) is not int
        or type(limits.max_path_components) is not int
        or not 1 <= limits.max_path_bytes <= 4096
        or not 1 <= limits.max_path_components <= 256
    ):
        _deny(ClaimReason.INVALID_POLICY)
    src = _validate_directory(policy.source, limits, custody=False)
    operator = _validate_directory(policy.custody, limits, custody=True)
    # An operator custody root must never be inside worker-writable Source.
    if operator[: len(src)] == src or src == operator:
        _deny(ClaimReason.INVALID_POLICY)
    if type(policy.lock) is not LockFile:
        _deny(ClaimReason.INVALID_POLICY)
    lock = policy.lock
    if lock.name != lock_name(policy.source.device, policy.source.inode):
        _deny(ClaimReason.INVALID_POLICY)
    if (
        not all(
            type(getattr(lock, field)) is int and 0 <= getattr(lock, field) < 2**63
            for field in ("device", "inode", "uid", "gid")
        )
        or lock.device == 0
        or lock.inode == 0
        or type(lock.mode) is not int
        or lock.mode != 0o600
    ):
        _deny(ClaimReason.INVALID_POLICY)
    if lock.device != policy.custody.device:
        _deny(ClaimReason.INVALID_POLICY)


def _identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_uid,
        stat_result.st_gid,
        stat.S_IMODE(stat_result.st_mode),
    )


def _dir_identity(value: Directory) -> tuple[int, int, int, int, int]:
    return (value.device, value.inode, value.uid, value.gid, value.mode)


def _lock_identity(value: LockFile) -> tuple[int, int, int, int, int]:
    return (value.device, value.inode, value.uid, value.gid, value.mode)


def _open_directory(value: Directory, limits: Limits, reason: ClaimReason) -> int:
    parts = _parts(value.path, limits)
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in parts:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        observed = os.fstat(fd)
        if not stat.S_ISDIR(observed.st_mode) or _identity(observed) != _dir_identity(
            value
        ):
            _deny(reason)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_file(lockfd: int, dfd: int, expected: LockFile) -> None:
    actual = os.fstat(lockfd)
    if (
        not stat.S_ISREG(actual.st_mode)
        or actual.st_nlink != 1
        or _identity(actual) != _lock_identity(expected)
    ):
        _deny(ClaimReason.LOCK_CHANGED)
    entry = os.stat(expected.name, dir_fd=dfd, follow_symlinks=False)
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or _identity(entry) != _lock_identity(expected)
    ):
        _deny(ClaimReason.LOCK_CHANGED)


class LiveClaim:
    """Held local flock and descriptor observations, NOT process-lifetime proof."""

    __slots__ = (
        "_closed",
        "_custody_fd",
        "_lock_fd",
        "_owner_pid",
        "_policy",
        "_source_fd",
    )

    def __init__(
        self, policy: Policy, custody_fd: int, lock_fd: int, source_fd: int
    ) -> None:
        self._policy = policy
        self._custody_fd = custody_fd
        self._lock_fd = lock_fd
        self._source_fd = source_fd
        self._owner_pid = os.getpid()
        self._closed = False

    @property
    def held(self) -> bool:
        return not self._closed and self._owner_pid == os.getpid()

    def recheck(self) -> None:
        """Coarse path-free refusal if Source, custody, or lock has moved.

        The mechanical claim does NOT establish old writer/cgroup drainage;
        a future supervisor must independently prove that before release.
        """
        if not self.held:
            _deny(ClaimReason.CLOSED)
        p = self._policy
        custody_fd = -1
        source_fd = -1
        try:
            try:
                custody_fd = _open_directory(
                    p.custody, p.limits, ClaimReason.CUSTODY_CHANGED
                )
                if _identity(os.fstat(self._custody_fd)) != _dir_identity(p.custody):
                    _deny(ClaimReason.CUSTODY_CHANGED)
            except (OSError, UnicodeError, ValueError, OverflowError):
                _deny(ClaimReason.CUSTODY_CHANGED)
            try:
                source_fd = _open_directory(
                    p.source, p.limits, ClaimReason.SOURCE_CHANGED
                )
                if _identity(os.fstat(self._source_fd)) != _dir_identity(p.source):
                    _deny(ClaimReason.SOURCE_CHANGED)
            except (OSError, UnicodeError, ValueError, OverflowError):
                _deny(ClaimReason.SOURCE_CHANGED)
            try:
                _check_file(self._lock_fd, custody_fd, p.lock)
            except (OSError, UnicodeError, ValueError, OverflowError):
                _deny(ClaimReason.LOCK_CHANGED)
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            if custody_fd >= 0:
                os.close(custody_fd)

    def close(self) -> None:
        """Explicit release only after a future caller proves writer drain.

        Do not issue LOCK_UN: inherited copies of the open-file description
        must keep the flock until the last descriptor is closed.
        """
        if self._closed:
            return
        if self._owner_pid != os.getpid():
            _deny(ClaimReason.CLOSED)
        self._closed = True
        try:
            os.close(self._lock_fd)
        finally:
            try:
                os.close(self._custody_fd)
            finally:
                os.close(self._source_fd)

    # Deliberately no context manager: implicit release on cancellation
    # would unlock a still-live managed writer before cgroup drain proof.


def acquire(policy: Policy) -> LiveClaim:
    """Acquire existing global physical Source lock, refusing all absent state.

    The P2 supervisor may use this held mechanical claim, but it MUST also
    prove generation-specific old worker/cgroup drain, protected Source and
    owner witness continuity. No caller can bootstrap from lock absence.
    """
    _validate(policy)
    sourcefd = -1
    custodyfd = -1
    lockfd = -1
    handed_off = False
    try:
        sourcefd = _open_directory(
            policy.source, policy.limits, ClaimReason.SOURCE_CHANGED
        )
        custodyfd = _open_directory(
            policy.custody, policy.limits, ClaimReason.CUSTODY_CHANGED
        )
        try:
            lockfd = os.open(policy.lock.name, _LOCK_FLAGS, dir_fd=custodyfd)
        except FileNotFoundError:
            _deny(ClaimReason.LOCK_MISSING)
        _check_file(lockfd, custodyfd, policy.lock)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _deny(ClaimReason.LOCK_BUSY)
        # Read-back under the acquired lifetime lock.
        _check_file(lockfd, custodyfd, policy.lock)
        source_verify = _open_directory(
            policy.source, policy.limits, ClaimReason.SOURCE_CHANGED
        )
        try:
            if _identity(os.fstat(sourcefd)) != _identity(os.fstat(source_verify)):
                _deny(ClaimReason.SOURCE_CHANGED)
        finally:
            os.close(source_verify)
        claim = LiveClaim(policy, custodyfd, lockfd, sourcefd)
        # Ownership transfers only after the final recheck succeeds.
        # A failed recheck takes the outer finally path and closes all FDs.
        claim.recheck()
        handed_off = True
        return claim
    except ClaimDenied:
        raise
    except (OSError, OverflowError, UnicodeError, ValueError):
        _deny(ClaimReason.UNSTABLE)
    finally:
        # Never rely on locals() or a deleted variable to decide FD ownership.
        if not handed_off:
            if lockfd >= 0:
                os.close(lockfd)
            if custodyfd >= 0:
                os.close(custodyfd)
            if sourcefd >= 0:
                os.close(sourcefd)
