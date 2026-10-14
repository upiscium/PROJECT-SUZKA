"""W0-I1a read-only admission of source locations and config/evaluation bytes.

This module deliberately does not start a service, create any source owner,
authorize bootstrap, or implement runtime revalidation (W0-I1b/I2).
All custody expectations and resource limits must come from a trusted caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, NoReturn

import yaml  # type: ignore[import-untyped]

from suzka.config.schema import Settings


class AdmissionReason(StrEnum):
    POLICY_INVALID = "policy_invalid"
    CUSTODY_UNAVAILABLE = "custody_unavailable"
    IDENTITY_MISMATCH = "identity_mismatch"
    PATH_UNSUPPORTED = "path_unsupported"
    CONFIG_INVALID = "config_invalid"
    RESOURCE_EXCEEDED = "resource_exceeded"
    SOURCE_MISMATCH = "source_mismatch"
    EVAL_UNAVAILABLE = "eval_unavailable"


class SourceAdmissionDenied(ValueError):
    """Typed refusal without leaking private config content or file paths."""

    def __init__(self, reason: AdmissionReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class TrustedDirectory:
    """Caller-expected identity of an already-provisioned private root."""

    path: Path
    device: int
    inode: int
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True, slots=True)
class ExpectedInput:
    """Caller-expected original evaluation source, not future run authority."""

    path: Path
    device: int
    inode: int
    digest: str


@dataclass(frozen=True, slots=True)
class TrustedAdmissionPolicy:
    """Mandatory and immutable *data*, not a transferable trust capability.

    The caller must separately authenticate the policy's origin. Numbers,
    ownership, location, symlink/hardlink and mount rules have NO defaults.
    """

    config_root: TrustedDirectory
    state_root: TrustedDirectory
    eval_root: TrustedDirectory
    config_path: Path
    config_device: int
    config_inode: int
    config_digest: str
    config_uid: int
    config_gid: int
    config_mode: int
    eval_uid: int
    eval_gid: int
    eval_mode: int
    expected_source_paths: tuple[tuple[str, Path], ...]
    expected_eval_inputs: tuple[ExpectedInput, ...]
    trusted_ancestor_uids: tuple[int, ...]
    allowed_sticky_ancestors: tuple[Path, ...]
    require_single_link: bool
    forbid_mount_crossing: bool
    max_config_bytes: int
    max_yaml_nodes: int
    max_yaml_depth: int
    max_path_components: int
    max_eval_input_bytes: int
    max_eval_cases: int
    max_metadata_visits: int


@dataclass(frozen=True, slots=True)
class FileProvenance:
    path: str
    device: int
    inode: int
    digest: str


@dataclass(frozen=True, slots=True)
class AdmittedEvaluation:
    """Private case contents: repr=False is NOT redaction for serializers."""

    provenance: FileProvenance
    cases: tuple[tuple[str, str], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class AdmittedSourceSelection:
    """Immutable observation only: NEVER a permit to start source owners.

    The private config bytes are exactly the previously FD-verified and
    parsed input, not a second pathname read. They may contain private data:
    repr=False is NOT redaction by generic serializers. Never log/publish.
    """

    config: FileProvenance
    _validated_config_bytes: bytes = field(repr=False)
    source_paths: tuple[tuple[str, str], ...]
    derived_paths: tuple[tuple[str, str], ...]
    evaluation_inputs: tuple[AdmittedEvaluation, ...] = field(repr=False)
    roots: tuple[tuple[str, int, int], ...]
    source_start_status: str = "SOURCE_START_BLOCKED_PENDING_I2"


_SOURCE_KEYS = (
    "memory.persist_directory",
    "sleep.dream_dataset_path",
    "qlora.output_dir",
    "adapter_registry.path",
    "adapter_registry.eval_result_dir",
    "agent_state.path",
    "event_journal.path",
    "state_wal.directory",
)
_DIR_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
# Nonblocking is REQUIRED: opening a FIFO must not hang before fstat.
_FILE_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _deny(reason: AdmissionReason) -> NoReturn:
    raise SourceAdmissionDenied(reason)


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def _identity(value: object) -> bool:
    return type(value) is int and value >= 0


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _parts(value: Path, limit: int) -> tuple[str, ...]:
    if not isinstance(value, Path):
        _deny(AdmissionReason.POLICY_INVALID)
    raw = str(value)
    if not raw.startswith("/") or raw.startswith("//"):
        _deny(AdmissionReason.PATH_UNSUPPORTED)
    parts = tuple(raw.split("/")[1:])
    if any(part in ("", ".", "..") or "\x00" in part for part in parts):
        _deny(AdmissionReason.PATH_UNSUPPORTED)
    if len(parts) > limit:
        _deny(AdmissionReason.RESOURCE_EXCEEDED)
    return parts


def _validate_policy(policy: TrustedAdmissionPolicy | None) -> None:
    if type(policy) is not TrustedAdmissionPolicy:
        _deny(AdmissionReason.POLICY_INVALID)
    assert isinstance(policy, TrustedAdmissionPolicy)
    for name in (
        "max_config_bytes",
        "max_yaml_nodes",
        "max_yaml_depth",
        "max_path_components",
        "max_eval_input_bytes",
        "max_eval_cases",
        "max_metadata_visits",
    ):
        if not _positive(getattr(policy, name)):
            _deny(AdmissionReason.POLICY_INVALID)
    if (
        type(policy.require_single_link) is not bool
        or type(policy.forbid_mount_crossing) is not bool
        or not isinstance(policy.expected_source_paths, tuple)
        or not isinstance(policy.expected_eval_inputs, tuple)
        or not policy.expected_eval_inputs
        or len(policy.expected_eval_inputs) > policy.max_eval_cases
    ):
        _deny(AdmissionReason.POLICY_INVALID)
    for root in (policy.config_root, policy.state_root, policy.eval_root):
        if type(root) is not TrustedDirectory:
            _deny(AdmissionReason.POLICY_INVALID)
        _parts(root.path, policy.max_path_components)
        if (
            not all(
                _identity(getattr(root, x)) for x in ("device", "inode", "uid", "gid")
            )
            or type(root.mode) is not int
            or not 0 <= root.mode <= 0o777
            or root.mode & 0o022
        ):
            _deny(AdmissionReason.POLICY_INVALID)
    for key in (
        "config_uid",
        "config_gid",
        "eval_uid",
        "eval_gid",
        "config_device",
        "config_inode",
    ):
        if not _identity(getattr(policy, key)):
            _deny(AdmissionReason.POLICY_INVALID)
    for key in ("config_mode", "eval_mode"):
        mode = getattr(policy, key)
        if type(mode) is not int or not 0 <= mode <= 0o777 or mode & 0o022:
            _deny(AdmissionReason.POLICY_INVALID)
    if not _digest(policy.config_digest):
        _deny(AdmissionReason.POLICY_INVALID)
    _parts(policy.config_path, policy.max_path_components)
    if policy.config_path == policy.config_root.path:
        _deny(AdmissionReason.POLICY_INVALID)
    if not policy.config_path.is_relative_to(policy.config_root.path):
        _deny(AdmissionReason.POLICY_INVALID)
    root_paths = (
        policy.config_root.path,
        policy.state_root.path,
        policy.eval_root.path,
    )
    if len(set(root_paths)) != 3 or any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(root_paths)
        for b in root_paths[i + 1 :]
    ):
        _deny(AdmissionReason.POLICY_INVALID)
    if (
        not isinstance(policy.trusted_ancestor_uids, tuple)
        or not policy.trusted_ancestor_uids
        or not all(_identity(uid) for uid in policy.trusted_ancestor_uids)
        or not isinstance(policy.allowed_sticky_ancestors, tuple)
    ):
        _deny(AdmissionReason.POLICY_INVALID)
    for parent in policy.allowed_sticky_ancestors:
        _parts(parent, policy.max_path_components)
    if len(set(policy.allowed_sticky_ancestors)) != len(
        policy.allowed_sticky_ancestors
    ):
        _deny(AdmissionReason.POLICY_INVALID)
    if len(policy.expected_source_paths) != len(_SOURCE_KEYS):
        _deny(AdmissionReason.POLICY_INVALID)
    names: set[str] = set()
    paths: list[Path] = []
    for item in policy.expected_source_paths:
        if not isinstance(item, tuple) or len(item) != 2:
            _deny(AdmissionReason.POLICY_INVALID)
        name, path = item
        if not isinstance(name, str) or name in names or name not in _SOURCE_KEYS:
            _deny(AdmissionReason.POLICY_INVALID)
        names.add(name)
        _parts(path, policy.max_path_components)
        if path == policy.state_root.path or not path.is_relative_to(
            policy.state_root.path
        ):
            _deny(AdmissionReason.POLICY_INVALID)
        paths.append(path)
    if names != set(_SOURCE_KEYS):
        _deny(AdmissionReason.POLICY_INVALID)
    for i, left in enumerate(paths):
        if any(
            left == right or left in right.parents or right in left.parents
            for right in paths[i + 1 :]
        ):
            _deny(AdmissionReason.POLICY_INVALID)
    for eval_item in policy.expected_eval_inputs:
        if type(eval_item) is not ExpectedInput:
            _deny(AdmissionReason.POLICY_INVALID)
        _parts(eval_item.path, policy.max_path_components)
        if (
            eval_item.path == policy.eval_root.path
            or not eval_item.path.is_relative_to(policy.eval_root.path)
            or not _identity(eval_item.device)
            or not _identity(eval_item.inode)
            or not _digest(eval_item.digest)
        ):
            _deny(AdmissionReason.POLICY_INVALID)
    if len({item.path for item in policy.expected_eval_inputs}) != len(
        policy.expected_eval_inputs
    ):
        _deny(AdmissionReason.POLICY_INVALID)


class _Budget:
    def __init__(self, limit: int) -> None:
        self.remaining = limit

    def visit(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            _deny(AdmissionReason.RESOURCE_EXCEEDED)


def _open_root(
    root: TrustedDirectory, policy: TrustedAdmissionPolicy, budget: _Budget
) -> int:
    """Follow no symlinks from /, checking the caller-pinned root identity."""
    flags = _DIR_FLAGS
    budget.visit()
    fd = os.open("/", flags)
    try:
        current = Path("/")
        for part in _parts(root.path, policy.max_path_components):
            budget.visit()
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
            current = current / part
            budget.visit()
            ancestor = os.fstat(fd)
            sticky_exception = (
                current in policy.allowed_sticky_ancestors
                and ancestor.st_uid == 0
                and bool(ancestor.st_mode & stat.S_ISVTX)
            )
            if (
                not stat.S_ISDIR(ancestor.st_mode)
                or ancestor.st_uid not in policy.trusted_ancestor_uids
                or (ancestor.st_mode & 0o022 and not sticky_exception)
            ):
                _deny(AdmissionReason.IDENTITY_MISMATCH)
        budget.visit()
        s = os.fstat(fd)
        if (
            not stat.S_ISDIR(s.st_mode)
            or (s.st_dev, s.st_ino) != (root.device, root.inode)
            or (s.st_uid, s.st_gid) != (root.uid, root.gid)
            or stat.S_IMODE(s.st_mode) != root.mode
        ):
            _deny(AdmissionReason.IDENTITY_MISMATCH)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _walk_leaf(
    root_fd: int,
    root: TrustedDirectory,
    relative: tuple[str, ...],
    policy: TrustedAdmissionPolicy,
    budget: _Budget,
) -> int:
    """Walk only inside a previously verified root, without file creation."""
    fd = os.dup(root_fd)
    try:
        for part in relative[:-1]:
            budget.visit()
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
            budget.visit()
            s = os.fstat(fd)
            if (
                not stat.S_ISDIR(s.st_mode)
                or s.st_uid != root.uid
                or s.st_gid != root.gid
                or s.st_mode & 0o022
                or (policy.forbid_mount_crossing and s.st_dev != root.device)
            ):
                _deny(AdmissionReason.IDENTITY_MISMATCH)
        budget.visit()
        result = os.open(relative[-1], _FILE_FLAGS, dir_fd=fd)
        return result
    finally:
        os.close(fd)


def _read_bounded(fd: int, limit: int) -> bytes:
    """Bound allocations per read even if the *caller* supplied a huge limit."""
    pieces: list[bytes] = []
    used = 0
    while used <= limit:
        piece = os.read(fd, min(16384, limit + 1 - used))
        if not piece:
            break
        used += len(piece)
        if used > limit:
            _deny(AdmissionReason.RESOURCE_EXCEEDED)
        pieces.append(piece)
    return b"".join(pieces)


def _stat_signature(s: os.stat_result) -> tuple[int, ...]:
    return (
        s.st_dev,
        s.st_ino,
        s.st_uid,
        s.st_gid,
        stat.S_IMODE(s.st_mode),
        s.st_size,
        s.st_mtime_ns,
        s.st_ctime_ns,
    )


def _read_verified_file(
    path: Path,
    root: TrustedDirectory,
    policy: TrustedAdmissionPolicy,
    budget: _Budget,
    *,
    expected_dev: int,
    expected_ino: int,
    expected_digest: str,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
    limit: int,
) -> tuple[bytes, FileProvenance]:
    _parts(path, policy.max_path_components)
    if path == root.path or not path.is_relative_to(root.path):
        _deny(AdmissionReason.PATH_UNSUPPORTED)
    relative = path.relative_to(root.path).parts
    if not relative or len(relative) > policy.max_path_components:
        _deny(AdmissionReason.PATH_UNSUPPORTED)
    anchor = _open_root(root, policy, budget)
    try:
        fd = _walk_leaf(anchor, root, relative, policy, budget)
        try:
            budget.visit()
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != (expected_dev, expected_ino)
                or (before.st_uid, before.st_gid) != (expected_uid, expected_gid)
                or stat.S_IMODE(before.st_mode) != expected_mode
                or (policy.require_single_link and before.st_nlink != 1)
                or (policy.forbid_mount_crossing and before.st_dev != root.device)
            ):
                _deny(AdmissionReason.IDENTITY_MISMATCH)
            contents = _read_bounded(fd, limit)
            budget.visit()
            after = os.fstat(fd)
            if _stat_signature(before) != _stat_signature(after):
                _deny(AdmissionReason.IDENTITY_MISMATCH)
            digest = hashlib.sha256(contents).hexdigest()
            if digest != expected_digest:
                _deny(AdmissionReason.IDENTITY_MISMATCH)
            # Check both the original root and the current name. The FD is
            # still the source of the bytes: there is NO pathname re-read.
            second_anchor = _open_root(root, policy, budget)
            try:
                second_fd = _walk_leaf(second_anchor, root, relative, policy, budget)
                try:
                    budget.visit()
                    named = os.fstat(second_fd)
                    if _stat_signature(named) != _stat_signature(after):
                        _deny(AdmissionReason.IDENTITY_MISMATCH)
                finally:
                    os.close(second_fd)
            finally:
                os.close(second_anchor)
            return contents, FileProvenance(
                str(path), before.st_dev, before.st_ino, digest
            )
        finally:
            os.close(fd)
    finally:
        os.close(anchor)


class _NoDuplicateKeysLoader(yaml.SafeLoader):
    """Reject ambiguous mapping overwrites, after the event bound is proved."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        if not isinstance(node, yaml.MappingNode):
            _deny(AdmissionReason.CONFIG_INVALID)
        seen: set[object] = set()
        for key_node, _value in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                if key in seen:
                    _deny(AdmissionReason.CONFIG_INVALID)
                seen.add(key)
            except TypeError:
                _deny(AdmissionReason.CONFIG_INVALID)
        return super().construct_mapping(node, deep=deep)


def _parse_config(
    content: bytes, policy: TrustedAdmissionPolicy
) -> tuple[dict, Settings]:
    try:
        text = content.decode("utf-8")
        count = depth = 0
        for event in yaml.parse(text):
            count += 1
            if count > policy.max_yaml_nodes:
                _deny(AdmissionReason.RESOURCE_EXCEEDED)
            if isinstance(event, yaml.events.AliasEvent):
                _deny(AdmissionReason.CONFIG_INVALID)
            if isinstance(
                event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)
            ):
                depth += 1
                if depth > policy.max_yaml_depth:
                    _deny(AdmissionReason.RESOURCE_EXCEEDED)
            if isinstance(
                event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)
            ):
                depth -= 1
        if depth != 0:
            _deny(AdmissionReason.CONFIG_INVALID)
        raw = yaml.load(text, Loader=_NoDuplicateKeysLoader)
        if not isinstance(raw, dict):
            _deny(AdmissionReason.CONFIG_INVALID)
        settings = Settings.model_validate(raw)
    except SourceAdmissionDenied:
        raise
    except (
        ValueError,
        TypeError,
        UnicodeError,
        yaml.YAMLError,
        RecursionError,
        OverflowError,
    ):
        _deny(AdmissionReason.CONFIG_INVALID)
    return raw, settings


def _raw_source_paths(
    raw: dict, policy: TrustedAdmissionPolicy
) -> tuple[tuple[str, str], ...]:
    # Read original YAML strings; pathlib silently normalizes /./ before Pydantic.
    fields: list[tuple[str, str]] = []
    expected = dict(policy.expected_source_paths)
    for name in _SOURCE_KEYS:
        section, key = name.split(".")
        try:
            value = raw[section][key]
        except (KeyError, TypeError):
            _deny(AdmissionReason.CONFIG_INVALID)
        if not isinstance(value, str):
            _deny(AdmissionReason.CONFIG_INVALID)
        segments = _parts(Path(value), policy.max_path_components)
        if value != "/" + "/".join(segments):
            _deny(AdmissionReason.PATH_UNSUPPORTED)
        actual = Path(value)
        if actual != expected[name]:
            _deny(AdmissionReason.SOURCE_MISMATCH)
        fields.append((name, value))

    # The effective paths must not reuse any foreign root, and cannot overlap.
    source_paths = [Path(v) for _n, v in fields]
    state = policy.state_root.path
    if any(path == state or not path.is_relative_to(state) for path in source_paths):
        _deny(AdmissionReason.SOURCE_MISMATCH)
    for i, path in enumerate(source_paths):
        if any(
            path == other or path in other.parents or other in path.parents
            for other in source_paths[i + 1 :]
        ):
            _deny(AdmissionReason.SOURCE_MISMATCH)
    return tuple(fields)


def _raw_eval_paths(raw: dict, policy: TrustedAdmissionPolicy) -> None:
    try:
        values = raw["adapter_registry"]["eval_sets"]
    except (KeyError, TypeError):
        _deny(AdmissionReason.CONFIG_INVALID)
    if not isinstance(values, list) or len(values) != len(policy.expected_eval_inputs):
        _deny(AdmissionReason.EVAL_UNAVAILABLE)
    for value, expected in zip(values, policy.expected_eval_inputs, strict=True):
        if not isinstance(value, str):
            _deny(AdmissionReason.EVAL_UNAVAILABLE)
        segments = _parts(Path(value), policy.max_path_components)
        if value != "/" + "/".join(segments) or Path(value) != expected.path:
            _deny(AdmissionReason.EVAL_UNAVAILABLE)


def _eval_cases(content: bytes, max_cases: int) -> tuple[tuple[str, str], ...]:
    try:

        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result

        parsed = json.loads(content, object_pairs_hook=no_duplicates)
    except (ValueError, UnicodeError, RecursionError, TypeError):
        _deny(AdmissionReason.EVAL_UNAVAILABLE)
    cases = parsed.get("cases") if isinstance(parsed, dict) else None
    if not isinstance(cases, list) or not 0 < len(cases) <= max_cases:
        _deny(AdmissionReason.EVAL_UNAVAILABLE)
    accepted: list[tuple[str, str]] = []
    for row in cases:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("prompt"), str)
            or not row["prompt"]
            or not isinstance(row.get("expected"), str)
        ):
            _deny(AdmissionReason.EVAL_UNAVAILABLE)
        accepted.append((row["prompt"], row["expected"]))
    return tuple(accepted)


def admit_source_configuration(
    policy: TrustedAdmissionPolicy | None,
) -> AdmittedSourceSelection:
    """Observe config/eval provenance, returning NEVER a service startup permit.

    Caller controls policy origin; I1a does not authenticate an installation
    or allow I2 first-install/continuation. All file opens are read-only.
    """
    _validate_policy(policy)
    assert policy is not None  # narrowing only; invalid None denied above
    budget = _Budget(policy.max_metadata_visits)
    try:
        # Even an EMPTY pre-existing state root is not a first-install permit.
        fd = _open_root(policy.state_root, policy, budget)
        os.close(fd)
        config_bytes, config_proof = _read_verified_file(
            policy.config_path,
            policy.config_root,
            policy,
            budget,
            expected_dev=policy.config_device,
            expected_ino=policy.config_inode,
            expected_digest=policy.config_digest,
            expected_uid=policy.config_uid,
            expected_gid=policy.config_gid,
            expected_mode=policy.config_mode,
            limit=policy.max_config_bytes,
        )
        raw, _validated = _parse_config(config_bytes, policy)
        paths = _raw_source_paths(raw, policy)
        _raw_eval_paths(raw, policy)
        eval_items: list[AdmittedEvaluation] = []
        total_eval_cases = 0
        for expected in policy.expected_eval_inputs:
            content, provenance = _read_verified_file(
                expected.path,
                policy.eval_root,
                policy,
                budget,
                expected_dev=expected.device,
                expected_ino=expected.inode,
                expected_digest=expected.digest,
                expected_uid=policy.eval_uid,
                expected_gid=policy.eval_gid,
                expected_mode=policy.eval_mode,
                limit=policy.max_eval_input_bytes,
            )
            cases = _eval_cases(content, policy.max_eval_cases)
            total_eval_cases += len(cases)
            if total_eval_cases > policy.max_eval_cases:
                _deny(AdmissionReason.RESOURCE_EXCEEDED)
            eval_items.append(AdmittedEvaluation(provenance, cases))
        memory_root = dict(paths)["memory.persist_directory"]
        memory = Path(memory_root)
        derived = (
            ("experience", str(memory.parent / "experience")),
            ("semantic", str(memory.parent / "semantic")),
            ("episodic_pending", str(memory / ".r07-episodic-pending")),
        )
        all_other = [Path(value) for _key, value in paths]
        for key, name in derived:
            path = Path(name)
            if any(
                path == other
                or (
                    key != "episodic_pending"
                    and (path in other.parents or other in path.parents)
                )
                for other in all_other
            ):
                _deny(AdmissionReason.SOURCE_MISMATCH)
        return AdmittedSourceSelection(
            config=config_proof,
            _validated_config_bytes=config_bytes,
            source_paths=paths,
            derived_paths=derived,
            evaluation_inputs=tuple(eval_items),
            roots=tuple(
                (str(root.path), root.device, root.inode)
                for root in (policy.config_root, policy.state_root, policy.eval_root)
            ),
        )
    except SourceAdmissionDenied:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        _deny(AdmissionReason.CUSTODY_UNAVAILABLE)
