"""Disconnected, read-only, partial Source OWNER filesystem observation.

All inputs are unauthenticated caller assertions. No result authorizes startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import os
import re
import stat
from uuid import UUID


class Refusal(StrEnum):
    INVALID_POLICY = "invalid_policy"
    PATH_UNSAFE = "path_unsafe"
    ABSENT = "absent"
    IDENTITY_MISMATCH = "identity_mismatch"
    INVALID_ARTIFACT = "invalid_artifact"
    RESOURCE_EXCEEDED = "resource_exceeded"
    UNSTABLE = "unstable"


class Observation(StrEnum):
    P1A_PARTIAL_UNVERIFIED_CHROMA = "p1a_partial_unverified_chroma"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class Directory:
    path: str
    device: int
    inode: int
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    device: int
    uid: int
    gid: int
    mode: int
    # Only supply when a protected witness guarantees a stable file incarnation.
    stable_inode: int | None = None


@dataclass(frozen=True, slots=True)
class Limits:
    max_file_bytes: int
    max_metadata_visits: int
    max_path_bytes: int
    max_path_components: int
    max_journal_entries: int
    max_rotated_segments: int


@dataclass(frozen=True, slots=True)
class Policy:
    source: Directory
    journal_dir: Directory
    wal_dir: Directory
    generations_dir: Directory
    experience_dir: Directory
    semantic_dir: Directory
    snapshot: Artifact
    journal_active: Artifact
    wal_manifest: Artifact
    boot_anchor: Artifact
    adapter_registry: Artifact
    generation: Artifact
    rotated: tuple[Artifact, ...]
    limits: Limits


@dataclass(frozen=True, slots=True)
class Result:
    observation: Observation
    startup: str = "DISABLED_IN_P1A"
    refusal: Refusal | None = None


class _Denied(Exception):
    def __init__(self, reason: Refusal):
        self.reason = reason


def _check(condition: bool, reason: Refusal) -> None:
    if not condition:
        raise _Denied(reason)


def _number(value: object, *, positive: bool = False) -> bool:
    return type(value) is int and (value > 0 if positive else value >= 0)


def _parts(path: object, limits: Limits) -> tuple[str, ...]:
    if not isinstance(path, str):
        raise _Denied(Refusal.INVALID_POLICY)
    _check(0 < len(os.fsencode(path)) <= limits.max_path_bytes, Refusal.PATH_UNSAFE)
    _check(path.startswith("/") and not path.startswith("//"), Refusal.PATH_UNSAFE)
    parts = tuple(path.split("/"))[1:]
    _check(
        0 < len(parts) <= limits.max_path_components
        and all(p not in ("", ".", "..") and "\x00" not in p for p in parts),
        Refusal.PATH_UNSAFE,
    )
    return parts


def _validate(policy: Policy) -> None:
    _check(
        isinstance(policy, Policy) and isinstance(policy.limits, Limits),
        Refusal.INVALID_POLICY,
    )
    limits = policy.limits
    _check(
        all(
            _number(getattr(limits, name), positive=True)
            for name in Limits.__dataclass_fields__
        ),
        Refusal.INVALID_POLICY,
    )
    dirs = (
        policy.source,
        policy.journal_dir,
        policy.wal_dir,
        policy.generations_dir,
        policy.experience_dir,
        policy.semantic_dir,
    )
    # Validate tuple shape and bound before expanding untrusted caller input.
    _check(type(policy.rotated) is tuple, Refusal.INVALID_POLICY)
    _check(
        len(policy.rotated) <= limits.max_rotated_segments, Refusal.RESOURCE_EXCEEDED
    )
    files = (
        policy.snapshot,
        policy.journal_active,
        policy.wal_manifest,
        policy.boot_anchor,
        policy.adapter_registry,
        policy.generation,
        *policy.rotated,
    )
    _check(
        all(isinstance(d, Directory) for d in dirs)
        and all(isinstance(f, Artifact) for f in files)
        and isinstance(policy.rotated, tuple),
        Refusal.INVALID_POLICY,
    )
    source = _parts(policy.source.path, limits)
    for directory in dirs:
        parts = _parts(directory.path, limits)
        _check(parts[: len(source)] == source, Refusal.PATH_UNSAFE)
        _check(
            all(
                _number(getattr(directory, n), positive=n in ("device", "inode"))
                for n in ("device", "inode", "uid", "gid")
            )
            and type(directory.mode) is int
            and 0 <= directory.mode <= 0o7777,
            Refusal.INVALID_POLICY,
        )
    _check(len({d.path for d in dirs}) == len(dirs), Refusal.INVALID_POLICY)
    for file in files:
        parts = _parts(file.path, limits)
        _check(
            parts[: len(source)] == source and len(parts) > len(source),
            Refusal.PATH_UNSAFE,
        )
        _check(
            all(
                _number(getattr(file, n), positive=n == "device")
                for n in ("device", "uid", "gid")
            )
            and type(file.mode) is int
            and 0 <= file.mode <= 0o7777
            and (
                file.stable_inode is None or _number(file.stable_inode, positive=True)
            ),
            Refusal.INVALID_POLICY,
        )
    _check(len({f.path for f in files}) == len(files), Refusal.INVALID_POLICY)
    _check(
        all(f.path.rpartition("/")[0] in {d.path for d in dirs} for f in files),
        Refusal.INVALID_POLICY,
    )

    def parent(file: Artifact, directory: Directory) -> bool:
        return file.path.rpartition("/")[0] == directory.path

    _check(
        parent(policy.journal_active, policy.journal_dir)
        and parent(policy.wal_manifest, policy.wal_dir)
        and parent(policy.boot_anchor, policy.wal_dir)
        and parent(policy.generation, policy.generations_dir)
        and policy.generations_dir.path.rpartition("/")[0] == policy.wal_dir.path
        and policy.wal_manifest.path.endswith("/manifest.json")
        and policy.boot_anchor.path.endswith("/boot_anchor.json"),
        Refusal.INVALID_POLICY,
    )
    _check(
        all(parent(f, policy.journal_dir) for f in policy.rotated)
        and len(policy.rotated) <= limits.max_rotated_segments,
        Refusal.INVALID_POLICY,
    )
    base = policy.journal_active.path.rpartition("/")[2]
    _check(
        all(
            re.fullmatch(re.escape(base) + r"\.\d{8}", f.path.rpartition("/")[2])
            for f in policy.rotated
        ),
        Refusal.INVALID_POLICY,
    )
    _check(
        all(
            d.path == policy.source.path
            or d.path.rpartition("/")[0] in {x.path for x in dirs}
            for d in dirs
        ),
        Refusal.INVALID_POLICY,
    )


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return (
        a.st_dev,
        a.st_ino,
        a.st_mode,
        a.st_uid,
        a.st_gid,
        a.st_size,
        a.st_mtime_ns,
        a.st_ctime_ns,
    ) == (
        b.st_dev,
        b.st_ino,
        b.st_mode,
        b.st_uid,
        b.st_gid,
        b.st_size,
        b.st_mtime_ns,
        b.st_ctime_ns,
    )


class _Reader:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.visits = 0
        self.directories = {
            d.path: d
            for d in (
                policy.source,
                policy.journal_dir,
                policy.wal_dir,
                policy.generations_dir,
                policy.experience_dir,
                policy.semantic_dir,
            )
        }

    def visit(self) -> None:
        self.visits += 1
        _check(
            self.visits <= self.policy.limits.max_metadata_visits,
            Refusal.RESOURCE_EXCEEDED,
        )

    def directory(self, path: str) -> int:
        parts = _parts(path, self.policy.limits)
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            prefix = ""
            for part in parts:
                self.visit()
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
                os.close(fd)
                fd = child
                prefix += "/" + part
                actual = os.fstat(fd)
                expected = self.directories.get(prefix)
                _check(stat.S_ISDIR(actual.st_mode), Refusal.IDENTITY_MISMATCH)
                if expected is not None:
                    _check(
                        (
                            actual.st_dev,
                            actual.st_ino,
                            actual.st_uid,
                            actual.st_gid,
                            stat.S_IMODE(actual.st_mode),
                        )
                        == (
                            expected.device,
                            expected.inode,
                            expected.uid,
                            expected.gid,
                            expected.mode,
                        ),
                        Refusal.IDENTITY_MISMATCH,
                    )
            return fd
        except BaseException:
            os.close(fd)
            raise

    def file(self, expected: Artifact) -> bytes:
        parent, _, name = expected.path.rpartition("/")
        fd = self.directory(parent)
        try:
            before_parent = os.fstat(fd)
            self.visit()
            leaf = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=fd,
            )
            try:
                before = os.fstat(leaf)
                _check(
                    stat.S_ISREG(before.st_mode)
                    and before.st_nlink == 1
                    and before.st_size > 0,
                    Refusal.INVALID_ARTIFACT,
                )
                _check(
                    (
                        before.st_dev,
                        before.st_uid,
                        before.st_gid,
                        stat.S_IMODE(before.st_mode),
                    )
                    == (expected.device, expected.uid, expected.gid, expected.mode)
                    and (
                        expected.stable_inode is None
                        or before.st_ino == expected.stable_inode
                    ),
                    Refusal.IDENTITY_MISMATCH,
                )
                _check(
                    before.st_size <= self.policy.limits.max_file_bytes,
                    Refusal.RESOURCE_EXCEEDED,
                )
                remaining = before.st_size
                chunks: list[bytes] = []
                while remaining > 0:
                    chunk = os.read(leaf, min(remaining, 65536))
                    _check(bool(chunk), Refusal.UNSTABLE)
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                _check(_same(before, os.fstat(leaf)), Refusal.UNSTABLE)
                self.visit()
                named = os.stat(name, dir_fd=fd, follow_symlinks=False)
                _check(
                    _same(before, named) and _same(before_parent, os.fstat(fd)),
                    Refusal.UNSTABLE,
                )
                return data
            finally:
                os.close(leaf)
        finally:
            os.close(fd)

    def journal(self) -> None:
        policy = self.policy
        fd = self.directory(policy.journal_dir.path)
        try:
            before = os.fstat(fd)
            names: list[str] = []
            # Avoid allocating the whole directory before enforcing its limit.
            with os.scandir(fd) as entries:
                for entry in entries:
                    self.visit()
                    _check(
                        len(names) < policy.limits.max_journal_entries,
                        Refusal.RESOURCE_EXCEEDED,
                    )
                    names.append(entry.name)
            base = policy.journal_active.path.rpartition("/")[2]
            suffixes = sorted(name for name in names if name.startswith(base + "."))
            _check(
                not any(name.startswith("." + base + ".rotation") for name in names),
                Refusal.INVALID_ARTIFACT,
            )
            _check(
                all(re.fullmatch(re.escape(base) + r"\.\d{8}", n) for n in suffixes),
                Refusal.INVALID_ARTIFACT,
            )
            _check(
                len(suffixes) <= policy.limits.max_rotated_segments,
                Refusal.RESOURCE_EXCEEDED,
            )
            numbers = [int(name[-8:]) for name in suffixes]
            _check(
                all(right == left + 1 for left, right in zip(numbers, numbers[1:])),
                Refusal.INVALID_ARTIFACT,
            )
            _check(
                set(suffixes) == {f.path.rpartition("/")[2] for f in policy.rotated},
                Refusal.ABSENT,
            )
            _check(_same(before, os.fstat(fd)), Refusal.UNSTABLE)
        finally:
            os.close(fd)
        for segment in policy.rotated:
            _json_lines(self.file(segment), "journal")
        _json_lines(self.file(policy.journal_active), "journal")


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate")
        result[key] = value
    return result


def _json(raw: bytes) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, RecursionError):
        raise _Denied(Refusal.INVALID_ARTIFACT) from None


def _uuid(value: object) -> bool:
    try:
        return type(value) is str and str(UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def _json_lines(raw: bytes, kind: str) -> None:
    _check(raw.endswith(b"\n"), Refusal.INVALID_ARTIFACT)
    for line in raw.split(b"\n")[:-1]:
        _check(bool(line), Refusal.INVALID_ARTIFACT)
        value = _json(line)
        if not isinstance(value, dict):
            raise _Denied(Refusal.INVALID_ARTIFACT)
        _check(
            type(value) is dict
            and type(value.get("schema_version")) is int
            and value["schema_version"] in ({1, 2, 3} if kind == "journal" else {1}),
            Refusal.INVALID_ARTIFACT,
        )


def _wal(manifest: object, anchor: object, policy: Policy) -> None:
    if not isinstance(manifest, dict) or not isinstance(anchor, dict):
        raise _Denied(Refusal.INVALID_ARTIFACT)
    _check(
        set(manifest)
        == {
            "schema_version",
            "command_version",
            "active_generation_id",
            "active_baseline_record_id",
            "active_baseline_record_hash",
            "predecessor_generation_id",
            "predecessor_generation_hash",
            "external_reconciliation_required",
            "manifest_hash",
        }
        and set(anchor)
        == {
            "schema_version",
            "command_version",
            "snapshot_sequence",
            "snapshot_hash",
            "generation_id",
            "anchored_record_id",
            "anchored_record_hash",
            "journal_processing_high_water",
            "journal_tail_record_id",
            "journal_tail_record_hash",
            "journal_lineage_id",
            "anchor_hash",
        },
        Refusal.INVALID_ARTIFACT,
    )
    _check(
        type(manifest["schema_version"]) is int
        and manifest["schema_version"] == 1
        and type(manifest["command_version"]) is int
        and manifest["command_version"] == 1
        and type(anchor["schema_version"]) is int
        and anchor["schema_version"] == 1
        and type(anchor["command_version"]) is int
        and anchor["command_version"] == 1
        and type(manifest["external_reconciliation_required"]) is bool,
        Refusal.INVALID_ARTIFACT,
    )
    for key in ("active_generation_id", "active_baseline_record_id"):
        _check(_uuid(manifest[key]), Refusal.INVALID_ARTIFACT)
    for key in ("generation_id", "anchored_record_id", "journal_lineage_id"):
        _check(_uuid(anchor[key]), Refusal.INVALID_ARTIFACT)
    for obj, keys in (
        (manifest, ("active_baseline_record_hash", "manifest_hash")),
        (anchor, ("snapshot_hash", "anchored_record_hash", "anchor_hash")),
    ):
        _check(
            all(
                type(obj[k]) is str and re.fullmatch(r"[0-9a-f]{64}", obj[k])
                for k in keys
            ),
            Refusal.INVALID_ARTIFACT,
        )
    for id_key, hash_key in (
        ("predecessor_generation_id", "predecessor_generation_hash"),
    ):
        _check(
            (manifest[id_key] is None and manifest[hash_key] is None)
            or (
                _uuid(manifest[id_key])
                and type(manifest[hash_key]) is str
                and re.fullmatch(r"[0-9a-f]{64}", manifest[hash_key]) is not None
            ),
            Refusal.INVALID_ARTIFACT,
        )
    _check(
        (
            anchor["journal_tail_record_id"] is None
            and anchor["journal_tail_record_hash"] is None
        )
        or (
            _uuid(anchor["journal_tail_record_id"])
            and type(anchor["journal_tail_record_hash"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", anchor["journal_tail_record_hash"])
            is not None
        ),
        Refusal.INVALID_ARTIFACT,
    )
    _check(
        all(
            _number(anchor[k])
            for k in ("snapshot_sequence", "journal_processing_high_water")
        )
        # BootAnchor may remain on an older, separately valid generation
        # during recovery/rebaseline; only the manifest pins the active file.
        and policy.generation.path
        == (
            policy.generations_dir.path
            + "/"
            + manifest["active_generation_id"]
            + ".jsonl"
        ),
        Refusal.INVALID_ARTIFACT,
    )


def _registry(raw: bytes) -> None:
    data = _json(raw)
    if not isinstance(data, dict):
        raise _Denied(Refusal.INVALID_ARTIFACT)
    _check(
        type(data) is dict
        and set(data) == {"adapters"}
        and type(data["adapters"]) is list,
        Refusal.INVALID_ARTIFACT,
    )
    keys = {
        "adapter_id",
        "base_model",
        "path",
        "status",
        "state",
        "dataset_path",
        "dataset_hash",
        "eval_score",
        "eval_result_path",
        "created_at",
        "updated_at",
        "notes",
    }
    statuses = {
        "candidate",
        "trial_active",
        "approved",
        "active",
        "rejected",
        "archived",
    }
    for entry in data["adapters"]:
        _check(type(entry) is dict and set(entry) == keys, Refusal.INVALID_ARTIFACT)
        _check(
            all(
                type(entry[k]) is str for k in keys - {"eval_score", "eval_result_path"}
            )
            and entry["status"] in statuses
            and entry["state"] == entry["status"]
            and (
                entry["eval_score"] is None
                or (
                    type(entry["eval_score"]) in (int, float)
                    and -1e308 < entry["eval_score"] < 1e308
                )
            )
            and (
                entry["eval_result_path"] is None
                or type(entry["eval_result_path"]) is str
            ),
            Refusal.INVALID_ARTIFACT,
        )


def observe(policy: Policy) -> Result:
    """Observe asserted required artifacts; even success is NOT startup authority."""
    try:
        _validate(policy)
        reader = _Reader(policy)
        for directory in (
            policy.source,
            policy.journal_dir,
            policy.wal_dir,
            policy.generations_dir,
            policy.experience_dir,
            policy.semantic_dir,
        ):
            fd = reader.directory(directory.path)
            os.close(fd)
        _check(
            type(_json(reader.file(policy.snapshot))) is dict, Refusal.INVALID_ARTIFACT
        )
        reader.journal()
        manifest = _json(reader.file(policy.wal_manifest))
        anchor = _json(reader.file(policy.boot_anchor))
        _wal(manifest, anchor, policy)
        _json_lines(reader.file(policy.generation), "generation")
        _registry(reader.file(policy.adapter_registry))
        # Detect changed registered root incarnations after the observation.
        for directory in (
            policy.source,
            policy.journal_dir,
            policy.wal_dir,
            policy.generations_dir,
            policy.experience_dir,
            policy.semantic_dir,
        ):
            fd = reader.directory(directory.path)
            os.close(fd)
        return Result(Observation.P1A_PARTIAL_UNVERIFIED_CHROMA)
    except _Denied as error:
        return Result(Observation.REFUSED, refusal=error.reason)
    except (OSError, OverflowError, TypeError, ValueError):
        return Result(Observation.REFUSED, refusal=Refusal.ABSENT)
