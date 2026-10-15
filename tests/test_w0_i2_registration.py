"""Nonauthorizing W0-I2-P0 record mechanics on disposable private roots.

These tests intentionally never import an ASGI app, construct a Source Owner,
access installed systemd units, or create operator-owned registration records.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from uuid import uuid4

import pytest

from suzka.installation_registration import (
    InstallationRegistrationStore,
    RegistrationDenied,
    RegistrationOutcome,
    RegistrationReason,
    RegistrationRecord,
    RegistrationState,
    RegistryCustody,
)

ROOT = Path(__file__).resolve().parents[1]


def _custody(tmp_path: Path, *, max_record_bytes: int = 4096) -> RegistryCustody:
    directory = tmp_path / "operator-like-private-fixture"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    lock = directory / "registration.lock"
    lock.touch(mode=0o600)
    lock.chmod(0o600)
    ds, ls = directory.stat(), lock.stat()
    return RegistryCustody(
        directory=str(directory),
        directory_dev=ds.st_dev,
        directory_ino=ds.st_ino,
        directory_uid=ds.st_uid,
        directory_gid=ds.st_gid,
        directory_mode=0o700,
        lock_name=lock.name,
        lock_dev=ls.st_dev,
        lock_ino=ls.st_ino,
        lock_uid=ls.st_uid,
        lock_gid=ls.st_gid,
        lock_mode=0o600,
        record_name="registration.json",
        record_uid=ds.st_uid,
        record_gid=ds.st_gid,
        record_mode=0o600,
        max_record_bytes=max_record_bytes,
        max_path_bytes=4096,
        max_path_components=256,
        max_directory_entries=128,
    )


def _prepared(tmp_path: Path) -> RegistrationRecord:
    ds = tmp_path.stat()
    return RegistrationRecord(
        schema_version=1,
        state=RegistrationState.FIRST_INSTALL_PREPARED,
        revision=1,
        installation_id=str(uuid4()),
        first_install_attempt=str(uuid4()),
        config_sha256="a" * 64,
        installed_code_sha256="b" * 64,
        source_root=str(tmp_path / "future-state-not-created"),
        source_root_dev=ds.st_dev,
        source_root_ino=ds.st_ino,
        asgi_target="suzka.api.verified_service:create_served_app",
        owner_baseline_sha256=None,
        active_claim_id=None,
        last_claim_id=None,
    )


def _assert_denied(reason: RegistrationReason, action: object) -> None:
    assert callable(action)
    with pytest.raises(RegistrationDenied) as error:
        action()
    assert error.value.reason is reason
    assert str(error.value) == reason.value


def _write_untrusted_raw(policy: RegistryCustody, data: bytes) -> None:
    path = Path(policy.directory) / policy.record_name
    path.write_bytes(data)
    path.chmod(policy.record_mode)


def _in_progress(prepared: RegistrationRecord) -> RegistrationRecord:
    return replace(prepared, state=RegistrationState.GENESIS_IN_PROGRESS, revision=2)


def _ready(in_progress: RegistrationRecord) -> RegistrationRecord:
    return replace(
        in_progress,
        state=RegistrationState.CONTINUATION_READY,
        revision=3,
        owner_baseline_sha256="c" * 64,
    )


def test_missing_registration_is_not_first_install_permission(tmp_path: Path) -> None:
    custody = _custody(tmp_path)
    store = InstallationRegistrationStore(custody)
    _assert_denied(RegistrationReason.RECORD_MISSING, store.read)
    assert not (Path(custody.directory) / custody.record_name).exists()


def test_explicit_provision_and_genesis_transition_are_data_only(
    tmp_path: Path,
) -> None:
    custody = _custody(tmp_path)
    store = InstallationRegistrationStore(custody)
    p = _prepared(tmp_path)
    assert store.initialize(p) is RegistrationOutcome.COMMITTED_RECORD
    assert store.read() == p
    assert store.transition(p, _in_progress(p)) is RegistrationOutcome.COMMITTED_RECORD
    assert store.read() == _in_progress(p)
    assert not any(
        hasattr(store, name)
        for name in ("allow_start", "issue_boot_permit", "authorize", "start_service")
    )
    assert set(Path(custody.directory).iterdir()) == {
        Path(custody.directory) / custody.lock_name,
        Path(custody.directory) / custody.record_name,
    }


def test_full_lifecycle_records_are_not_a_startup_permit(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    p = _prepared(tmp_path)
    store.initialize(p)
    g, ready = _in_progress(p), _ready(_in_progress(p))
    assert store.transition(p, g) is RegistrationOutcome.COMMITTED_RECORD
    assert store.transition(g, ready) is RegistrationOutcome.COMMITTED_RECORD
    claimed = replace(
        ready,
        state=RegistrationState.START_CLAIMED,
        revision=4,
        active_claim_id=str(uuid4()),
    )
    assert store.transition(ready, claimed) is RegistrationOutcome.COMMITTED_RECORD
    released = replace(
        claimed,
        state=RegistrationState.CONTINUATION_READY,
        revision=5,
        active_claim_id=None,
        last_claim_id=claimed.active_claim_id,
    )
    assert store.transition(claimed, released) is RegistrationOutcome.COMMITTED_RECORD
    assert store.read() == released
    # The immediately previous claim cannot be issued again, even after a
    # mechanically recorded clean stop; no process identity is authenticated.
    reused = replace(
        released,
        state=RegistrationState.START_CLAIMED,
        revision=6,
        active_claim_id=claimed.active_claim_id,
    )
    _assert_denied(
        RegistrationReason.INVALID_TRANSITION,
        lambda: store.transition(released, reused),
    )
    assert store.read() == released


def test_genesis_requires_explicit_external_transition(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    p = _prepared(tmp_path)
    store.initialize(p)
    g = _in_progress(p)
    store.transition(p, g)
    _assert_denied(
        RegistrationReason.INVALID_TRANSITION,
        lambda: store.transition(g, replace(g, revision=3)),
    )
    _assert_denied(
        RegistrationReason.INVALID_TRANSITION,
        lambda: store.transition(
            g, replace(g, state=RegistrationState.FIRST_INSTALL_PREPARED, revision=3)
        ),
    )
    assert store.read().state is RegistrationState.GENESIS_IN_PROGRESS


def test_terminal_recovery_state_cannot_resume_without_new_human_gate(
    tmp_path: Path,
) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    p = _prepared(tmp_path)
    store.initialize(p)
    g = _in_progress(p)
    store.transition(p, g)
    recovery = replace(g, state=RegistrationState.RECOVERY_REQUIRED, revision=3)
    assert store.transition(g, recovery) is RegistrationOutcome.COMMITTED_RECORD
    _assert_denied(
        RegistrationReason.INVALID_TRANSITION,
        lambda: store.transition(
            recovery,
            replace(
                recovery,
                revision=4,
                state=RegistrationState.CONTINUATION_READY,
                owner_baseline_sha256="c" * 64,
            ),
        ),
    )


def test_duplicate_init_stale_revision_and_identity_changes_refused(
    tmp_path: Path,
) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    p = _prepared(tmp_path)
    store.initialize(p)
    _assert_denied(RegistrationReason.RECORD_EXISTS, lambda: store.initialize(p))
    g = _in_progress(p)
    store.transition(p, g)
    _assert_denied(RegistrationReason.STALE_REVISION, lambda: store.transition(p, g))
    _assert_denied(
        RegistrationReason.INVALID_TRANSITION,
        lambda: store.transition(g, replace(_ready(g), config_sha256="e" * 64)),
    )
    assert store.read() == g


@pytest.mark.parametrize(
    "mutator",
    [
        lambda obj: {**obj, "unexpected": "x"},
        lambda obj: {k: v for k, v in obj.items() if k != "asgi_target"},
        lambda obj: {**obj, "schema_version": True},
        lambda obj: {**obj, "schema_version": 2},
        lambda obj: {**obj, "revision": 1.0},
        lambda obj: {**obj, "revision": False},
        lambda obj: {**obj, "state": "AUTOMATIC_BOOT"},
        lambda obj: {**obj, "config_sha256": "not-a-sha256"},
        lambda obj: {**obj, "source_root": "/bad/../relative"},
        lambda obj: {**obj, "asgi_target": "suzka.api.server:app; rm -rf /"},
    ],
)
def test_malformed_or_unknown_record_denied(tmp_path: Path, mutator: object) -> None:
    assert callable(mutator)
    policy = _custody(tmp_path)
    prepared = _prepared(tmp_path)
    raw = json.loads(prepared.to_json_bytes(max_bytes=policy.max_record_bytes))
    _write_untrusted_raw(policy, json.dumps(mutator(raw)).encode())
    _assert_denied(
        RegistrationReason.INVALID_RECORD,
        InstallationRegistrationStore(policy).read,
    )


@pytest.mark.parametrize(
    "broken",
    [
        b"not-json",
        b"\xff\xfe",
        b'{"revision":1,"revision":2}',
        b'{"x":NaN}',
        b"",
    ],
)
def test_corrupt_invalid_utf8_duplicate_json_and_constant_refused(
    tmp_path: Path, broken: bytes
) -> None:
    policy = _custody(tmp_path)
    _write_untrusted_raw(policy, broken)
    _assert_denied(
        RegistrationReason.INVALID_RECORD,
        InstallationRegistrationStore(policy).read,
    )


def test_oversized_record_is_rejected_before_json_parsing(tmp_path: Path) -> None:
    policy = _custody(tmp_path, max_record_bytes=300)
    _write_untrusted_raw(policy, b"?" * 301)
    _assert_denied(
        RegistrationReason.RESOURCE_EXCEEDED,
        InstallationRegistrationStore(policy).read,
    )


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_false_explicit_custody_expectations_are_rejected(
    tmp_path: Path, field: str
) -> None:
    policy = _custody(tmp_path)
    InstallationRegistrationStore(policy).initialize(_prepared(tmp_path))
    changed = {field: getattr(policy, field) + 1}
    if field.endswith("_mode"):
        _assert_denied(
            RegistrationReason.INVALID_POLICY,
            lambda: replace(policy, **changed),
        )
        return
    other = replace(policy, **changed)
    _assert_denied(
        RegistrationReason.CUSTODY_MISMATCH,
        InstallationRegistrationStore(other).read,
    )


def test_lock_is_preprovisioned_not_implicitly_created(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    (Path(policy.directory) / policy.lock_name).unlink()
    _assert_denied(
        RegistrationReason.CUSTODY_UNAVAILABLE,
        lambda: InstallationRegistrationStore(policy).initialize(_prepared(tmp_path)),
    )
    assert list(Path(policy.directory).iterdir()) == []


def test_lock_inode_replacement_even_same_name_refused(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    lock = Path(policy.directory) / policy.lock_name
    lock.rename(lock.with_name("old-lock-keeps-original-inode"))
    lock.write_bytes(b"")
    lock.chmod(policy.lock_mode)
    _assert_denied(
        RegistrationReason.CUSTODY_MISMATCH,
        lambda: InstallationRegistrationStore(policy).initialize(_prepared(tmp_path)),
    )


def test_symlink_lock_or_directory_component_denied(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    lock = Path(policy.directory) / policy.lock_name
    lock.unlink()
    lock.symlink_to(tmp_path / "missing-target")
    _assert_denied(
        RegistrationReason.CUSTODY_UNAVAILABLE,
        lambda: InstallationRegistrationStore(policy).initialize(_prepared(tmp_path)),
    )
    direct = tmp_path / "symlink"
    direct.symlink_to(Path(policy.directory), target_is_directory=True)
    wrapped = replace(policy, directory=str(direct))
    _assert_denied(
        RegistrationReason.CUSTODY_UNAVAILABLE,
        lambda: InstallationRegistrationStore(wrapped).initialize(_prepared(tmp_path)),
    )


def test_hardlinked_lock_and_registry_are_rejected(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    lock = Path(policy.directory) / policy.lock_name
    os.link(lock, Path(policy.directory) / "lock-extra")
    _assert_denied(
        RegistrationReason.CUSTODY_MISMATCH,
        InstallationRegistrationStore(policy).read,
    )


def test_record_symlink_and_fifo_are_refused_without_blocking(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    path = Path(policy.directory) / policy.record_name
    path.symlink_to(tmp_path / "absent")
    _assert_denied(
        RegistrationReason.CUSTODY_UNAVAILABLE,
        InstallationRegistrationStore(policy).read,
    )
    path.unlink()
    if hasattr(os, "mkfifo"):
        os.mkfifo(path, 0o600)
        _assert_denied(
            RegistrationReason.CUSTODY_MISMATCH,
            InstallationRegistrationStore(policy).read,
        )


def test_policy_must_be_explicit_absolute_private_and_bounded(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    for bad in (
        {"directory": "relative"},
        {"directory": "/path/../not-canonical"},
        {"directory": "/path//double"},
        {"directory_mode": 0o777},
        {"max_record_bytes": 0},
        {"max_record_bytes": 1024 * 1024 + 1},
        {"lock_name": "../elsewhere"},
        {"record_name": policy.lock_name},
    ):
        _assert_denied(
            RegistrationReason.INVALID_POLICY,
            lambda bad=bad: replace(policy, **bad),
        )


def test_file_fsync_failure_keeps_record_absent_and_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)
    real_fsync = os.fsync

    def broken_fsync(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("injected before publication")
        real_fsync(fd)

    monkeypatch.setattr(reg.os, "fsync", broken_fsync)
    _assert_denied(
        RegistrationReason.CUSTODY_UNAVAILABLE,
        lambda: InstallationRegistrationStore(policy).initialize(_prepared(tmp_path)),
    )
    assert not (Path(policy.directory) / policy.record_name).exists()
    assert not list(Path(policy.directory).glob("*.tmp"))


def test_atomic_replace_failure_denies_with_no_partial_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)

    def broken_replace(*args: object, **kwargs: object) -> None:
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(reg.os, "replace", broken_replace)
    _assert_denied(
        RegistrationReason.CUSTODY_UNAVAILABLE,
        lambda: InstallationRegistrationStore(policy).initialize(_prepared(tmp_path)),
    )
    assert not (Path(policy.directory) / policy.record_name).exists()


def test_directory_fsync_failure_returns_uncertain_never_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)
    real_fsync = os.fsync

    def broken_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected post-replace directory fsync")
        real_fsync(fd)

    record = _prepared(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(reg.os, "fsync", broken_fsync)
        _assert_denied(
            RegistrationReason.UNCERTAIN_COMMIT,
            lambda: InstallationRegistrationStore(policy).initialize(record),
        )
    # A process crash rather than power loss may leave the new bytes visible.
    # This is not a success result and cannot be reinitialized silently.
    assert InstallationRegistrationStore(policy).read() == record
    _assert_denied(
        RegistrationReason.RECORD_EXISTS,
        lambda: InstallationRegistrationStore(policy).initialize(record),
    )


def test_no_empty_policy_or_fake_record_accepted(tmp_path: Path) -> None:
    _assert_denied(
        RegistrationReason.INVALID_POLICY,
        lambda: InstallationRegistrationStore(None),  # type: ignore[arg-type]
    )
    p = _custody(tmp_path)
    _assert_denied(
        RegistrationReason.INVALID_RECORD,
        lambda: InstallationRegistrationStore(p).initialize(None),  # type: ignore[arg-type]
    )


_CLAIMANT = r"""
import json,os,sys,time
from pathlib import Path
from dataclasses import replace
from uuid import uuid4
from suzka.installation_registration import (
 InstallationRegistrationStore,RegistryCustody,RegistrationState,
 RegistrationOutcome,RegistrationDenied,RegistrationReason
)
p=RegistryCustody(**json.loads(sys.argv[1]))
store=InstallationRegistrationStore(p)
kind=sys.argv[2]
barrier=Path(sys.argv[3])
expected_rev=int(sys.argv[4])

# This is a test-only retry of lock BUSY while reading, not a product-side
# authorization/replay retry after an ambiguous durable commit.
for attempt in range(250):
 try:old=store.read();break
 except RegistrationDenied as err:
  if err.reason is not RegistrationReason.LOCK_UNAVAILABLE:raise
  time.sleep(.01)
else:raise AssertionError("lock remained busy")
assert old.revision==expected_rev,old
if kind=="genesis":
 target=replace(old,state=RegistrationState.GENESIS_IN_PROGRESS,revision=old.revision+1)
else:
 target=replace(old,state=RegistrationState.START_CLAIMED,revision=old.revision+1,
                active_claim_id=str(uuid4()))
(barrier/("ready-"+str(os.getpid()))).write_text(str(old.revision))
for attempt in range(600):
 if (barrier/"release").is_file():break
 time.sleep(.01)
else:raise AssertionError("no barrier release")
for attempt in range(250):
 try:
  result=store.transition(old,target)
  assert result is RegistrationOutcome.COMMITTED_RECORD
  print("ONE_COMMITTED")
  break
 except RegistrationDenied as err:
  if err.reason is RegistrationReason.LOCK_UNAVAILABLE:
   time.sleep(.01)
   continue
  print("REFUSED:"+err.reason.value)
  break
else:raise AssertionError("retry budget exceeded")
"""


@pytest.mark.parametrize("kind", ["genesis", "continuation"])
def test_eight_independent_processes_get_only_one_cas_commit(
    tmp_path: Path, kind: str
) -> None:
    import time

    policy = _custody(tmp_path)
    prepared = _prepared(tmp_path)
    store = InstallationRegistrationStore(policy)
    store.initialize(prepared)
    expected_revision = 1
    if kind == "continuation":
        genesis = _in_progress(prepared)
        ready = _ready(genesis)
        store.transition(prepared, genesis)
        store.transition(genesis, ready)
        expected_revision = 3
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    barrier = tmp_path / "claimant-barrier"
    barrier.mkdir(mode=0o700)
    command = [
        sys.executable,
        "-c",
        _CLAIMANT,
        json.dumps(asdict(policy)),
        kind,
        str(barrier),
        str(expected_revision),
    ]
    children = [
        subprocess.Popen(
            command,
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(8)
    ]
    results: list[str] = []
    try:
        for _ in range(250):
            ready_files = tuple(barrier.glob("ready-*"))
            if len(ready_files) == 8:
                break
            assert not any(child.poll() is not None for child in children)
            time.sleep(0.04)
        else:
            raise AssertionError("all eight claimants must observe same revision")
        assert {path.read_text() for path in ready_files} == {str(expected_revision)}
        (barrier / "release").write_text("go")
        for child in children:
            output, errors = child.communicate(timeout=25)
            assert child.returncode == 0, errors
            results.append(output.strip())
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
    assert results.count("ONE_COMMITTED") == 1, results
    assert results.count("REFUSED:STALE_REVISION") == 7, results
    expected_state = (
        RegistrationState.GENESIS_IN_PROGRESS
        if kind == "genesis"
        else RegistrationState.START_CLAIMED
    )
    assert store.read().state is expected_state


def test_public_module_has_no_service_or_owner_imports(tmp_path: Path) -> None:
    import ast

    module = ROOT / "suzka" / "installation_registration.py"
    raw = module.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    assert all(
        not item.startswith(("suzka", "fastapi", "chromadb")) for item in imports
    )
    assert "ALLOW_START" not in raw
    assert "BootPermit" not in raw
    assert not (tmp_path / "service-state").exists()


def test_public_parser_enforces_mandatory_resource_limit(tmp_path: Path) -> None:
    record = _prepared(tmp_path)
    payload = record.to_json_bytes(max_bytes=4096)
    _assert_denied(
        RegistrationReason.RESOURCE_EXCEEDED,
        lambda: RegistrationRecord.from_json_bytes(payload, max_bytes=10),
    )
    _assert_denied(
        RegistrationReason.INVALID_POLICY,
        lambda: RegistrationRecord.from_json_bytes(payload, max_bytes=0),
    )
    _assert_denied(
        RegistrationReason.INVALID_RECORD,
        lambda: RegistrationRecord.from_json_bytes("not bytes", max_bytes=4096),  # type: ignore[arg-type]
    )


def test_deeply_nested_bounded_json_gets_typed_denial(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    # Well below the record-size bound yet above Python JSON nesting limits.
    nested = b"[" * 1300 + b"0" + b"]" * 1300
    assert len(nested) < policy.max_record_bytes
    _write_untrusted_raw(policy, nested)
    _assert_denied(
        RegistrationReason.INVALID_RECORD,
        InstallationRegistrationStore(policy).read,
    )


def test_escaped_surrogate_path_is_invalid_record_not_unicode_exception(
    tmp_path: Path,
) -> None:
    policy = _custody(tmp_path)
    record = _prepared(tmp_path)
    raw = json.loads(record.to_json_bytes(max_bytes=4096))
    raw["source_root"] = "/\ud800"
    _write_untrusted_raw(policy, json.dumps(raw).encode())
    _assert_denied(
        RegistrationReason.INVALID_RECORD,
        InstallationRegistrationStore(policy).read,
    )
    _assert_denied(
        RegistrationReason.INVALID_POLICY,
        lambda: replace(policy, directory="/\ud800"),
    )


def test_genesis_cannot_invent_a_previous_continuation_claim(
    tmp_path: Path,
) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    prepared = _prepared(tmp_path)
    store.initialize(prepared)
    inprogress = _in_progress(prepared)
    store.transition(prepared, inprogress)
    invalid_ready = replace(_ready(inprogress), last_claim_id=str(uuid4()))
    _assert_denied(
        RegistrationReason.INVALID_TRANSITION,
        lambda: store.transition(inprogress, invalid_ready),
    )
    assert store.read() == inprogress


def test_failure_after_postyield_identity_check_is_uncertain_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)
    original = reg._assert_directory
    check_count = 0

    def after_yield_failure(c: RegistryCustody, fd: int) -> None:
        nonlocal check_count
        check_count += 1
        if check_count == 4:
            raise RegistrationDenied(RegistrationReason.CUSTODY_MISMATCH)
        original(c, fd)

    record = _prepared(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(reg, "_assert_directory", after_yield_failure)
        _assert_denied(
            RegistrationReason.UNCERTAIN_COMMIT,
            lambda: InstallationRegistrationStore(policy).initialize(record),
        )
    assert check_count == 4
    # The published record may now exist, but the caller was NOT told it
    # could start anything; a repeat initializer refuses an existing record.
    assert InstallationRegistrationStore(policy).read() == record


def test_record_mode_changed_on_disk_and_hardlinked_record_refused(
    tmp_path: Path,
) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    store.initialize(_prepared(tmp_path))
    path = Path(policy.directory) / policy.record_name
    path.chmod(0o644)
    _assert_denied(RegistrationReason.CUSTODY_MISMATCH, store.read)
    path.chmod(0o600)
    extra = Path(policy.directory) / "registry-hardlink"
    os.link(path, extra)
    _assert_denied(RegistrationReason.CUSTODY_MISMATCH, store.read)


def test_nonblocking_lock_rejects_other_process_holder(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    store.initialize(_prepared(tmp_path))
    lock = str(Path(policy.directory) / policy.lock_name)
    cmd = (
        "import fcntl,os,sys;"
        "fd=os.open(sys.argv[1],os.O_RDWR);"
        "fcntl.flock(fd,fcntl.LOCK_EX);"
        "print('LOCKED',flush=True);"
        "sys.stdin.read()"
    )
    holder = subprocess.Popen(
        [sys.executable, "-u", "-c", cmd, lock],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "LOCKED"
        _assert_denied(RegistrationReason.LOCK_UNAVAILABLE, store.read)
    finally:
        if holder.stdin is not None:
            holder.stdin.close()
        holder.wait(timeout=8)


_CRASH_CHILD = r"""
import json,os,stat,sys
from dataclasses import replace
import suzka.installation_registration as reg

store=reg.InstallationRegistrationStore(
    reg.RegistryCustody(**json.loads(sys.argv[1]))
)
prior=store.read()
next_record=replace(
    prior,state=reg.RegistrationState.GENESIS_IN_PROGRESS,
    revision=prior.revision+1
)
stage=sys.argv[2]
original_write=reg.os.write
original_fsync=reg.os.fsync
original_replace=reg.os.replace

def fault_write(fd,data):
    if stage=="before_write":os._exit(77)
    count=original_write(fd,data)
    if stage=="after_write":os._exit(77)
    return count
def fault_fsync(fd):
    original_fsync(fd)
    s=os.fstat(fd)
    if stage=="after_file_fsync" and stat.S_ISREG(s.st_mode):os._exit(77)
    if stage=="after_directory_fsync" and stat.S_ISDIR(s.st_mode):os._exit(77)
def fault_replace(*args,**kwargs):
    result=original_replace(*args,**kwargs)
    if stage=="after_replace":os._exit(77)
    return result

reg.os.write=fault_write
reg.os.fsync=fault_fsync
reg.os.replace=fault_replace
store.transition(prior,next_record)
raise AssertionError("expected subprocess to crash before success")
"""


@pytest.mark.parametrize(
    ("stage", "expected_state"),
    [
        ("before_write", RegistrationState.FIRST_INSTALL_PREPARED),
        ("after_write", RegistrationState.FIRST_INSTALL_PREPARED),
        ("after_file_fsync", RegistrationState.FIRST_INSTALL_PREPARED),
        ("after_replace", RegistrationState.GENESIS_IN_PROGRESS),
        ("after_directory_fsync", RegistrationState.GENESIS_IN_PROGRESS),
    ],
)
def test_subprocess_crash_at_publication_boundaries_never_allows_replay(
    tmp_path: Path,
    stage: str,
    expected_state: RegistrationState,
) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    prepared = _prepared(tmp_path)
    store.initialize(prepared)
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH_CHILD,
            json.dumps(asdict(policy)),
            stage,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 77, (stage, child.stderr)
    # Inspect disk data in this isolated fault test; the product helper
    # separately FAILS CLOSED on an abandoned stage after a pre-replace crash.
    disk = (Path(policy.directory) / policy.record_name).read_bytes()
    actual = RegistrationRecord.from_json_bytes(disk, max_bytes=policy.max_record_bytes)
    assert actual.state is expected_state
    assert actual.revision == (1 if expected_state is prepared.state else 2)
    abandoned = tuple(Path(policy.directory).glob("." + policy.record_name + ".*.tmp"))
    if expected_state is RegistrationState.FIRST_INSTALL_PREPARED:
        assert abandoned
        _assert_denied(RegistrationReason.ABANDONED_STAGE, store.read)
        # Simulates an explicit EXTERNAL fixture-only operator decision;
        # the library NEVER cleans up stale stages itself.
        for stage_path in abandoned:
            stage_path.unlink()
        assert store.read() == prepared
    else:
        assert not abandoned
        # A failed process is not authorized to re-run its consumed intent.
        _assert_denied(
            RegistrationReason.STALE_REVISION,
            lambda: store.transition(prepared, _in_progress(prepared)),
        )


def test_short_read_valid_json_prefix_cannot_authorize_malformed_full_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)
    valid = _prepared(tmp_path).to_json_bytes(max_bytes=policy.max_record_bytes)
    _write_untrusted_raw(policy, valid + b"UNEXPECTED_TRAILING_BYTES")
    original_read = os.read
    calls = 0

    def short_read(fd: int, count: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_read(fd, len(valid))
        return b""  # Simulates a premature, zero-byte read before st_size.

    with monkeypatch.context() as m:
        m.setattr(reg.os, "read", short_read)
        _assert_denied(
            RegistrationReason.RECORD_MISMATCH,
            InstallationRegistrationStore(policy).read,
        )
    assert calls == 2


def test_abandoned_stage_requires_explicit_external_recovery(
    tmp_path: Path,
) -> None:
    policy = _custody(tmp_path)
    store = InstallationRegistrationStore(policy)
    prepared = _prepared(tmp_path)
    store.initialize(prepared)
    stage = Path(policy.directory) / (
        "." + policy.record_name + "." + "a" * 24 + ".tmp"
    )
    stage.write_text("fixture from interrupted write")
    stage.chmod(0o600)
    _assert_denied(RegistrationReason.ABANDONED_STAGE, store.read)
    assert stage.exists()  # The library must not erase prior owner evidence.
    _assert_denied(
        RegistrationReason.ABANDONED_STAGE,
        lambda: store.transition(prepared, _in_progress(prepared)),
    )
    stage.unlink()  # Explicit test-only simulated external classification.
    assert store.read() == prepared


def test_bounded_directory_scan_rejects_excess_entries(tmp_path: Path) -> None:
    policy = _custody(tmp_path)
    InstallationRegistrationStore(policy).initialize(_prepared(tmp_path))
    too_small = replace(policy, max_directory_entries=1)
    _assert_denied(
        RegistrationReason.RESOURCE_EXCEEDED,
        InstallationRegistrationStore(too_small).read,
    )


def test_failed_exclusive_temp_create_cannot_delete_another_existing_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)
    stage_name = "." + policy.record_name + "." + "a" * 24 + ".tmp"
    stage = Path(policy.directory) / stage_name
    original_open = os.open

    def create_collision_during_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == stage_name and flags & os.O_EXCL:
            stage.write_text("PREEXISTING_NOT_CREATED_BY_HELPER")
            stage.chmod(0o600)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as m:
        m.setattr(reg.secrets, "token_hex", lambda length: "a" * (2 * length))
        m.setattr(reg.os, "open", create_collision_during_open)
        _assert_denied(
            RegistrationReason.CUSTODY_UNAVAILABLE,
            lambda: InstallationRegistrationStore(policy).initialize(
                _prepared(tmp_path)
            ),
        )
    assert stage.read_text() == "PREEXISTING_NOT_CREATED_BY_HELPER"
    assert not (Path(policy.directory) / policy.record_name).exists()
    # Subsequent attempts cannot blindly overwrite or clean that artifact.
    _assert_denied(
        RegistrationReason.ABANDONED_STAGE,
        lambda: InstallationRegistrationStore(policy).initialize(_prepared(tmp_path)),
    )


def test_directory_scandir_does_not_consume_callers_fd(tmp_path: Path) -> None:
    """The registry owns the parent FD until the outer lock scope exits."""
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)
    fd = os.open(policy.directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        before = os.fstat(fd)
        reg._assert_no_abandoned_stage(policy, fd)
        after = os.fstat(fd)
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    finally:
        os.close(fd)


def test_initial_publish_cannot_strand_record_beyond_directory_budget(
    tmp_path: Path,
) -> None:
    policy = _custody(tmp_path)
    tiny = replace(policy, max_directory_entries=1)
    _assert_denied(
        RegistrationReason.RESOURCE_EXCEEDED,
        lambda: InstallationRegistrationStore(tiny).initialize(_prepared(tmp_path)),
    )
    assert not (Path(policy.directory) / policy.record_name).exists()
    enough = replace(policy, max_directory_entries=2)
    prepared = _prepared(tmp_path)
    assert (
        InstallationRegistrationStore(enough).initialize(prepared)
        is RegistrationOutcome.COMMITTED_RECORD
    )
    assert InstallationRegistrationStore(enough).read() == prepared


def test_fchmod_failure_cleans_own_created_temp_without_stranding_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suzka.installation_registration as reg

    policy = _custody(tmp_path)

    def fail_fchmod(fd: int, mode: int) -> None:
        raise OSError("injected chmod failure after temp creation")

    with monkeypatch.context() as m:
        m.setattr(reg.os, "fchmod", fail_fchmod)
        _assert_denied(
            RegistrationReason.CUSTODY_UNAVAILABLE,
            lambda: InstallationRegistrationStore(policy).initialize(
                _prepared(tmp_path)
            ),
        )
    assert not (Path(policy.directory) / policy.record_name).exists()
    assert not tuple(Path(policy.directory).glob("." + policy.record_name + ".*.tmp"))
    assert tuple(Path(policy.directory).iterdir()) == (
        Path(policy.directory) / policy.lock_name,
    )
