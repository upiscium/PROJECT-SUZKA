"""Private rootless tests for the nonauthorizing partial observation."""

from dataclasses import replace
import json
import os
from pathlib import Path
import stat
from uuid import uuid4

import pytest

from suzka import source_owner_preflight as p


def _directory(path: Path) -> p.Directory:
    s = path.stat()
    return p.Directory(
        str(path), s.st_dev, s.st_ino, s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)
    )


def _artifact(path: Path) -> p.Artifact:
    s = path.stat()
    return p.Artifact(str(path), s.st_dev, s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode))


@pytest.fixture
def installed(tmp_path: Path) -> p.Policy:
    source = tmp_path / "source"
    source.mkdir()
    journal = source / "journal"
    wal = source / "wal"
    generations = wal / "generations"
    experience = source / "experience"
    semantic = source / "semantic"
    for path in (journal, wal, generations, experience, semantic):
        path.mkdir()
    generation_id = str(uuid4())
    zero = "0" * 64
    manifest = dict(
        schema_version=1,
        command_version=1,
        active_generation_id=generation_id,
        active_baseline_record_id=str(uuid4()),
        active_baseline_record_hash=zero,
        predecessor_generation_id=None,
        predecessor_generation_hash=None,
        external_reconciliation_required=False,
        manifest_hash=zero,
    )
    anchor = dict(
        schema_version=1,
        command_version=1,
        snapshot_sequence=0,
        snapshot_hash=zero,
        generation_id=generation_id,
        anchored_record_id=str(uuid4()),
        anchored_record_hash=zero,
        journal_processing_high_water=0,
        journal_tail_record_id=None,
        journal_tail_record_hash=None,
        journal_lineage_id=str(uuid4()),
        anchor_hash=zero,
    )
    files = {
        "snapshot": source / "state.json",
        "journal_active": journal / "events.jsonl",
        "wal_manifest": wal / "manifest.json",
        "boot_anchor": wal / "boot_anchor.json",
        "generation": generations / f"{generation_id}.jsonl",
        "adapter_registry": source / "adapters.json",
    }
    for name, path in files.items():
        content = {
            "wal_manifest": json.dumps(manifest),
            "boot_anchor": json.dumps(anchor),
            "adapter_registry": '{"adapters":[]}',
            "snapshot": "{}",
        }.get(name, '{"schema_version":1}\n')
        path.write_text(content)
        path.chmod(0o600)
    return p.Policy(
        _directory(source),
        _directory(journal),
        _directory(wal),
        _directory(generations),
        _directory(experience),
        _directory(semantic),
        *(
            _artifact(files[name])
            for name in (
                "snapshot",
                "journal_active",
                "wal_manifest",
                "boot_anchor",
                "adapter_registry",
                "generation",
            )
        ),
        (),
        p.Limits(4096, 400, 512, 40, 30, 10),
    )


def _refused(policy: p.Policy) -> None:
    result = p.observe(policy)
    assert result.observation is p.Observation.REFUSED
    assert result.refusal is not None
    assert result.startup == "DISABLED_IN_P1A"


def test_initialized_empty_is_only_partial(installed: p.Policy) -> None:
    result = p.observe(installed)
    assert result == p.Result(p.Observation.P1A_PARTIAL_UNVERIFIED_CHROMA)
    assert not any(
        "allow" in name or "ready" in name
        for name in vars(p.Result).get("__dataclass_fields__", {})
    )
    assert "chroma" not in result.startup.lower()


@pytest.mark.parametrize(
    "field",
    [
        "snapshot",
        "journal_active",
        "wal_manifest",
        "boot_anchor",
        "adapter_registry",
        "generation",
        "experience_dir",
        "semantic_dir",
    ],
)
def test_missing_required(installed: p.Policy, field: str) -> None:
    target = Path(getattr(installed, field).path)
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()
    _refused(installed)


def test_root_replaced(installed: p.Policy) -> None:
    root = Path(installed.source.path)
    moved = root.with_name("former")
    root.rename(moved)
    root.mkdir()
    _refused(installed)


def test_directory_custody_mismatch(installed: p.Policy) -> None:
    _refused(
        replace(
            installed, source=replace(installed.source, uid=installed.source.uid + 1)
        )
    )
    root = Path(installed.experience_dir.path)
    root.chmod(0o700 if installed.experience_dir.mode != 0o700 else 0o711)
    _refused(installed)


@pytest.mark.parametrize(
    "kind",
    ["symlink", "fifo", "directory", "mode", "owner", "hardlink", "empty", "oversize"],
)
def test_unsafe_leaf(installed: p.Policy, kind: str) -> None:
    path = Path(installed.snapshot.path)
    if kind in {"symlink", "fifo", "directory", "hardlink"}:
        path.unlink()
        if kind == "symlink":
            path.symlink_to(installed.adapter_registry.path)
        elif kind == "fifo":
            os.mkfifo(path)
        elif kind == "directory":
            path.mkdir()
        else:
            os.link(installed.adapter_registry.path, path)
    elif kind == "mode":
        path.chmod(0o644)
    elif kind == "owner":
        installed = replace(
            installed,
            snapshot=replace(installed.snapshot, uid=installed.snapshot.uid + 1),
        )
    elif kind == "empty":
        path.write_bytes(b"")
    else:
        path.write_bytes(b"x" * 4097)
    _refused(installed)


def test_symlink_intermediate(installed: p.Policy) -> None:
    path = Path(installed.wal_dir.path)
    path.rename(path.with_name("wal_old"))
    path.symlink_to(path.with_name("wal_old"), target_is_directory=True)
    _refused(installed)


@pytest.mark.parametrize(
    "content",
    [
        '{"adapters":',
        '{"adapters":[],"adapters":[]}',
        '{"adapters":[],"extra":1}',
        '{"adapters":null}',
        '{"adapters":[{}]}',
    ],
)
def test_bad_registry(installed: p.Policy, content: str) -> None:
    Path(installed.adapter_registry.path).write_text(content)
    _refused(installed)


@pytest.mark.parametrize("field", ["wal_manifest", "boot_anchor"])
def test_bad_wal_json(installed: p.Policy, field: str) -> None:
    Path(getattr(installed, field).path).write_text('{"schema_version":')
    _refused(installed)


def test_truncated_records_and_snapshot(installed: p.Policy) -> None:
    for field in ("snapshot", "journal_active", "generation"):
        original = Path(getattr(installed, field).path).read_bytes()
        Path(getattr(installed, field).path).write_bytes(b'{"schema_version":')
        _refused(installed)
        Path(getattr(installed, field).path).write_bytes(original)


@pytest.mark.parametrize(
    "name", [".events.jsonl.rotation.tmp", "events.jsonl.bad", "events.jsonl.00000002"]
)
def test_journal_interrupted_unchanged(installed: p.Policy, name: str) -> None:
    artifact = Path(installed.journal_dir.path) / name
    artifact.write_bytes(b"private evidence")
    before = (artifact.read_bytes(), artifact.stat())
    _refused(installed)
    after = artifact.stat()
    assert artifact.read_bytes() == before[0]
    assert (after.st_ino, after.st_mode, after.st_mtime_ns) == (
        before[1].st_ino,
        before[1].st_mode,
        before[1].st_mtime_ns,
    )


def test_rotated_sequence(installed: p.Policy) -> None:
    journal = Path(installed.journal_dir.path)
    first = journal / "events.jsonl.00000001"
    second = journal / "events.jsonl.00000003"
    first.write_text('{"schema_version":1}\n')
    second.write_text('{"schema_version":1}\n')
    policy = replace(installed, rotated=(_artifact(first), _artifact(second)))
    _refused(policy)


def test_policy_invalid_and_traversal(installed: p.Policy) -> None:
    for budget in (None, False, 0, -1, "12"):
        _refused(
            replace(installed, limits=replace(installed.limits, max_file_bytes=budget))
        )  # type: ignore[arg-type]
    _refused(replace(installed, source=replace(installed.source, inode=None)))  # type: ignore[arg-type]
    _refused(replace(installed, snapshot=replace(installed.snapshot, path="/outside")))
    _refused(
        replace(
            installed,
            snapshot=replace(
                installed.snapshot, path=installed.source.path + "/../state"
            ),
        )
    )
    _refused(replace(installed, snapshot=replace(installed.snapshot, path="/" * 520)))
    _refused(
        replace(installed, limits=replace(installed.limits, max_metadata_visits=1))
    )


def test_atomic_mutable_replacement(installed: p.Policy) -> None:
    old = Path(installed.adapter_registry.path)
    replacement = old.with_name("replacement")
    replacement.write_text('{"adapters": []}')
    replacement.chmod(0o600)
    os.replace(replacement, old)
    assert (
        p.observe(installed).observation is p.Observation.P1A_PARTIAL_UNVERIFIED_CHROMA
    )


def test_observation_never_mutates(
    installed: p.Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [
        Path(getattr(installed, field).path)
        for field in (
            "snapshot",
            "journal_active",
            "wal_manifest",
            "boot_anchor",
            "adapter_registry",
            "generation",
        )
    ]
    before = [(x.read_bytes(), x.stat()) for x in paths]

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("preflight attempted mutation")

    for name in ("mkdir", "chmod", "fchmod", "replace", "rename", "unlink"):
        monkeypatch.setattr(p.os, name, forbidden)
    assert (
        p.observe(installed).observation is p.Observation.P1A_PARTIAL_UNVERIFIED_CHROMA
    )
    _refused(replace(installed, snapshot=replace(installed.snapshot, uid=999999)))
    for path, (bytes_before, stat_before) in zip(paths, before, strict=True):
        now = path.stat()
        assert path.read_bytes() == bytes_before
        assert (now.st_dev, now.st_ino, now.st_mode, now.st_mtime_ns) == (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_mode,
            stat_before.st_mtime_ns,
        )


def test_journal_over_limit_is_streamed(
    installed: p.Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = Path(installed.journal_dir.path)
    for n in range(12):
        (journal / f"unrelated-{n}").write_text("x")
    limit = 3
    policy = replace(
        installed, limits=replace(installed.limits, max_journal_entries=limit)
    )
    real_scandir = os.scandir
    consumed = [0]

    def bounded_scandir(fd: int) -> object:
        underlying = real_scandir(fd)

        class Scan:
            def __enter__(self) -> "Scan":
                underlying.__enter__()
                return self

            def __exit__(self, *exc: object) -> object:
                return underlying.__exit__(*exc)

            def __iter__(self) -> "Scan":
                return self

            def __next__(self) -> os.DirEntry[str]:
                consumed[0] += 1
                assert consumed[0] <= limit + 1, "Unbounded directory enumeration"
                return next(underlying)

        return Scan()

    def no_listdir(*_args: object, **_kwargs: object) -> None:
        pytest.fail("os.listdir materializes an unbounded directory")

    with monkeypatch.context() as mp:
        mp.setattr(p.os, "scandir", bounded_scandir)
        mp.setattr(p.os, "listdir", no_listdir)
        result = p.observe(policy)
    assert result.refusal is p.Refusal.RESOURCE_EXCEEDED
    assert consumed[0] == limit + 1


def test_huge_rotated_suffix_gap_is_bounded(installed: p.Policy) -> None:
    journal = Path(installed.journal_dir.path)
    a = journal / "events.jsonl.00000000"
    b = journal / "events.jsonl.99999999"
    for artifact in (a, b):
        artifact.write_text('{"schema_version":1}\n')
        artifact.chmod(0o600)
    policy = replace(installed, rotated=(_artifact(a), _artifact(b)))
    result = p.observe(policy)
    assert result.refusal is p.Refusal.INVALID_ARTIFACT


def test_rotated_tuple_validated_before_expansion(installed: p.Policy) -> None:
    _refused(replace(installed, rotated=[]))  # type: ignore[arg-type]
    too_many = replace(
        installed,
        rotated=(installed.journal_active,)
        * (installed.limits.max_rotated_segments + 1),
    )
    assert p.observe(too_many).refusal is p.Refusal.RESOURCE_EXCEEDED


def test_older_boot_anchor_generation_is_partial(installed: p.Policy) -> None:
    path = Path(installed.boot_anchor.path)
    anchor = json.loads(path.read_text())
    anchor["generation_id"] = str(uuid4())
    path.write_text(json.dumps(anchor))
    assert (
        p.observe(installed).observation is p.Observation.P1A_PARTIAL_UNVERIFIED_CHROMA
    )


def test_stable_short_reads_are_accepted(
    installed: p.Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = os.read
    reads = [0]

    def short_read(fd: int, count: int) -> bytes:
        reads[0] += 1
        return original_read(fd, min(count, 3))

    with monkeypatch.context() as mp:
        mp.setattr(p.os, "read", short_read)
        result = p.observe(installed)
    assert reads[0] > 10
    assert result.observation is p.Observation.P1A_PARTIAL_UNVERIFIED_CHROMA


def test_premature_eof_is_denied(
    installed: p.Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as mp:
        mp.setattr(p.os, "read", lambda _fd, _count: b"")
        result = p.observe(installed)
    assert result.refusal is p.Refusal.UNSTABLE


@pytest.mark.parametrize("kind", ["file", "journal"])
def test_initial_parent_fstat_failure_closes_fd(
    installed: p.Policy, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    target = (
        str(Path(installed.snapshot.path).parent)
        if kind == "file"
        else installed.journal_dir.path
    )
    original_dir = p._Reader.directory
    original_fstat = os.fstat
    armed: set[int] = set()
    recorded: list[int] = []

    def tracked_directory(reader: p._Reader, path: str) -> int:
        fd = original_dir(reader, path)
        if path == target:
            recorded.append(fd)
            armed.add(fd)
        return fd

    def faulty_fstat(fd: int) -> os.stat_result:
        if fd in armed:
            armed.remove(fd)
            raise OSError("injected parent fstat failure")
        return original_fstat(fd)

    with monkeypatch.context() as mp:
        mp.setattr(p._Reader, "directory", tracked_directory)
        mp.setattr(p.os, "fstat", faulty_fstat)
        reader = p._Reader(installed)
        with pytest.raises(OSError, match="injected parent fstat"):
            if kind == "file":
                reader.file(installed.snapshot)
            else:
                reader.journal()

    assert len(recorded) == 1
    with pytest.raises(OSError):
        original_fstat(recorded[0])
