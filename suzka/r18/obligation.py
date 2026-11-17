"""IP-16a: the shared obligation identity (#411 comment 6916372789 §3a, §8-1).

Human decision §8-1 (a), 2026-11-16: one obligation is
``HMAC(secret, context_id, normalized request text) + scope`` — independent of
events and incarnations (#373 A1-5), so the same request in the same Context
is the same obligation, and a true rollback cannot mint a second effect for
it. The text appears only inside the HMAC; the secret is 32 random bytes the
operator's ``target_init`` writes next to the target, outside the history
store's rollback domain (§3a). A lost or replaced secret is BLOCKED, never
recreated (it would silently change every group key).

The Intent issuer (IP-16a) and the Decision issuer's ledger reader (IP-16b)
compute the key only through ``identity_for_source``, which reads the secret,
the ``user_input`` of X's committed episode and then calls
``obligation_identity``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
import stat
from typing import Final
from typing import TYPE_CHECKING, Any
import unicodedata

from suzka.experience import ExperienceLifecycle, experience_record_digest
from suzka.memory.dual_memory_system import (
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
)
from suzka.memory.experience_store import ExperienceStoreError
from suzka.memory.memory_schema import MemoryRecordType
from suzka.r16.contracts import canonical_json, identifier
from suzka.r18.contracts import obligation_group_key

if TYPE_CHECKING:
    from suzka.r16.plan_preview import _QualifiedPreview

TARGET_NAMESPACE: Final = "review-artifact"
SECRET_FILE: Final = "obligation.key"
SECRET_BYTES: Final = 32
MAX_REQUEST_TEXT: Final = 16_384
_TEXT_DOMAIN: Final = b"PROJECT-SUZKA:R18:OBLIGATION-TEXT:V1\0"
_SCOPE_DOMAIN: Final = b"PROJECT-SUZKA:R18:OBLIGATION-SCOPE:V1\0"


class ObligationSecretError(Exception):
    """The obligation secret is missing or not trusted: BLOCKED (§3a)."""


class ObligationSourceUnavailable(Exception):
    """X's committed episode (the request text) cannot be read: HOLD (§3a)."""


@dataclass(frozen=True, slots=True)
class ObligationIdentity:
    experience_content_digest: str
    obligation_scope: str
    group_key: str


def normalize_request_text(text: str) -> str:
    """V1 normalization: Unicode NFC, whitespace runs to one space, stripped."""
    if type(text) is not str:
        raise TypeError("request text must be a string")
    if len(text) > 4 * MAX_REQUEST_TEXT:
        raise ValueError("request text exceeds its bound")
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalized:
        raise ValueError("request text is empty")
    if len(normalized) > MAX_REQUEST_TEXT:
        raise ValueError("request text exceeds its bound")
    return normalized


def obligation_identity(
    secret: bytes,
    *,
    subject_id: str,
    context_id: str,
    request_text: str,
    scope: tuple[str, ...],
) -> ObligationIdentity:
    """The incarnation-independent obligation group of one request."""

    if type(secret) is not bytes or len(secret) != SECRET_BYTES:
        raise ObligationSecretError("the obligation secret is not 32 bytes")
    identifier(context_id, "context_id")
    if type(scope) is not tuple or not scope or any(type(item) is not str or not item for item in scope):
        raise ValueError("scope must be a non-empty tuple of strings")
    content = hmac.new(
        secret,
        _TEXT_DOMAIN + canonical_json({"context_id": context_id, "text": normalize_request_text(request_text)}),
        hashlib.sha256,
    ).hexdigest()
    scope_id = "scope-" + hashlib.sha256(_SCOPE_DOMAIN + canonical_json(list(scope))).hexdigest()
    return ObligationIdentity(
        content,
        scope_id,
        obligation_group_key(subject_id, content, scope_id, TARGET_NAMESPACE),
    )


def request_text(source: Any, q: _QualifiedPreview) -> str:
    """§3a: the ``user_input`` of X's committed episode (never logged).

    ``source`` is the preview owner's source owner (its committed Experience
    and episodic memory reads). Every mismatch with the qualified X, an
    archived or missing episode, or a read failure is
    ``ObligationSourceUnavailable``.
    """

    try:
        record = source._experience.load_committed_current(q.experience_id).record
        if (
            record.experience_id != q.experience_id
            or record.revision != q.experience_revision
            or record.lifecycle is not ExperienceLifecycle.ACTIVE
            or experience_record_digest(record) != q.experience_digest
        ):
            raise ObligationSourceUnavailable("X is not the qualified Experience")
        committed = source._memory.get_committed_episodic(record.source_episode_id)
        if committed is None:
            raise ObligationSourceUnavailable("X's episode is missing")
        episode = committed.record
        if (
            episode.id != record.source_episode_id
            or episode.record_type is not MemoryRecordType.EPISODIC_LOG
            or episode.archived
            or episode.context_id != record.context_id
            or record.context_id != q.context_id
            or type(episode.user_input) is not str
        ):
            raise ObligationSourceUnavailable("X's episode is not current")
        return episode.user_input
    except (
        ExperienceStoreError,
        EpisodicMemoryReadError,
        EpisodicMemoryFormatError,
        ValueError,
        OSError,
    ):
        raise ObligationSourceUnavailable("X's episode is unavailable") from None


def identity_for_source(
    secret_directory: Path,
    *,
    source: Any,
    subject_id: str,
    q: _QualifiedPreview,
    scope: tuple[str, ...],
) -> ObligationIdentity:
    """The shared §3a computation: secret, then request text, then the key.

    Raises ``ObligationSecretError`` (no / untrusted / wrong-size secret) or
    ``ObligationSourceUnavailable`` (episode unavailable, or an input the key
    refuses).
    """

    secret = load_obligation_secret(secret_directory)
    text = request_text(source, q)
    try:
        return obligation_identity(
            secret,
            subject_id=subject_id,
            context_id=q.context_id,
            request_text=text,
            scope=scope,
        )
    except (TypeError, ValueError):
        raise ObligationSourceUnavailable("the obligation inputs are invalid") from None


def check_secret_directory(directory: Path) -> None:
    """The same rule as the target's directory: owned, not group/other writable.

    The secret file itself is 0600; the directory rule keeps anyone else from
    replacing it.
    """
    if not isinstance(directory, Path) or not directory.is_absolute():
        raise ObligationSecretError("the target directory must be absolute")
    if os.path.realpath(directory) != str(directory):
        raise ObligationSecretError("the target directory must be a normal path without symlinks")
    try:
        info = os.lstat(directory)
    except OSError as error:
        raise ObligationSecretError("the target directory is unavailable") from error
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise ObligationSecretError("the target directory custody is invalid")


def _check_secret_file(info: os.stat_result) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        or info.st_size != SECRET_BYTES
    ):
        raise ObligationSecretError("the obligation secret custody is invalid")


def load_obligation_secret(directory: Path) -> bytes:
    """Read the 32-byte secret; refuse anything but a private regular file."""

    check_secret_directory(directory)
    path = directory / SECRET_FILE
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise ObligationSecretError("the obligation secret is missing (run target_init)") from None
    except OSError as error:
        raise ObligationSecretError("the obligation secret is unavailable") from error
    _check_secret_file(info)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        raise ObligationSecretError("the obligation secret is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ObligationSecretError("the obligation secret changed while it was opened")
        _check_secret_file(opened)
        data = os.read(descriptor, SECRET_BYTES + 1)
    except OSError as error:
        raise ObligationSecretError("the obligation secret is unavailable") from error
    finally:
        os.close(descriptor)
    if len(data) != SECRET_BYTES:
        raise ObligationSecretError("the obligation secret is not 32 bytes")
    return data


def create_obligation_secret(directory: Path) -> Path:
    """Write a fresh secret once (O_EXCL, 0600, fsync); refuse to overwrite."""

    check_secret_directory(directory)
    path = directory / SECRET_FILE
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    except FileExistsError as error:
        raise ObligationSecretError("the obligation secret already exists; it is never replaced") from error
    try:
        if os.write(descriptor, os.urandom(SECRET_BYTES)) != SECRET_BYTES:
            raise ObligationSecretError("the obligation secret was not fully written")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


__all__ = [
    "MAX_REQUEST_TEXT",
    "SECRET_BYTES",
    "SECRET_FILE",
    "TARGET_NAMESPACE",
    "ObligationIdentity",
    "ObligationSecretError",
    "ObligationSourceUnavailable",
    "check_secret_directory",
    "create_obligation_secret",
    "load_obligation_secret",
    "identity_for_source",
    "normalize_request_text",
    "obligation_identity",
    "request_text",
]
