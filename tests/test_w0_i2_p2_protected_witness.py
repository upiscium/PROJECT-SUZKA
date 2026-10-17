"""Rootless P2 protected-witness mechanics, never a live Source or P0 record."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import stat
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from suzka import protected_owner_witness as store
from suzka.owner_witness_codec import (
    CatalogIdentity,
    CollectionIdentity,
    DirectoryIdentity,
    OwnerPresence,
    OwnerWitness,
    canonicalize_witness_bytes,
    digest_witness_bytes,
)


def witness(installation: int = 1) -> OwnerWitness:
    d = DirectoryIdentity
    return OwnerWitness(
        schema_version=1,
        installation_id=str(UUID(int=installation)),
        first_install_attempt=str(UUID(int=2)),
        source_root=d("/example/source", 1, 11, 1000, 1000, 0o700),
        source_ancestors=(
            d("/", 1, 9, 0, 0, 0o755),
            d("/example", 1, 10, 0, 0, 0o755),
        ),
        source_mount=d("/example", 1, 10, 0, 0, 0o755),
        chroma_root=d("/example/source/chroma", 1, 12, 1000, 1000, 0o700),
        chroma_catalog=CatalogIdentity(
            "/example/source/chroma/chroma.sqlite3", 1, 13, 1000, 1000, 0o600
        ),
        tenant_name="default_tenant",
        database_name="default_database",
        database_id=str(UUID(int=3)),
        db1=CollectionIdentity("db1", str(UUID(int=4))),
        db2=CollectionIdentity("db2", str(UUID(int=5))),
        journal_lineage_id=str(UUID(int=6)),
        experience_root=d("/example/source/experience", 1, 14, 1000, 1000, 0o700),
        semantic_root=d("/example/source/semantic", 1, 15, 1000, 1000, 0o700),
        adapter_registry_empty_at_genesis=True,
        required_owners=tuple(
            OwnerPresence(owner_class, owner_class.lower())
            for owner_class in (
                "AgentState",
                "Journal",
                "WAL",
                "AdapterRegistry",
                "Experience",
                "Semantic",
            )
        ),
    )


@pytest.fixture
def policy(tmp_path: Path) -> store.Policy:
    operator = tmp_path / "operator-records"
    operator.mkdir(mode=0o700)
    operator.chmod(0o700)
    lock = operator / "custody.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    dstat = operator.stat()
    lstat = lock.stat()
    return store.Policy(
        custody=store.DirectoryCustody(
            str(operator),
            dstat.st_dev,
            dstat.st_ino,
            dstat.st_uid,
            dstat.st_gid,
            stat.S_IMODE(dstat.st_mode),
        ),
        lock=store.LockFile(
            "custody.lock",
            lstat.st_dev,
            lstat.st_ino,
            lstat.st_uid,
            lstat.st_gid,
            stat.S_IMODE(lstat.st_mode),
        ),
        installation_id=witness().installation_id,
        record_uid=dstat.st_uid,
        record_gid=dstat.st_gid,
        record_mode=0o600,
        limits=store.Limits(4096, 256, 40, 32768),
    )


def denied(call, reason: store.StoreReason) -> None:
    with pytest.raises(store.StoreDenied) as error:
        call()
    assert error.value.reason is reason
    assert "/" not in str(error.value)


def owner_file(p: store.Policy) -> Path:
    return Path(p.custody.path) / store.record_name(p.installation_id)


def test_publish_exact_canonical_once_and_observe(policy: store.Policy) -> None:
    expected = witness()
    payload = canonicalize_witness_bytes(expected)
    before = set(Path(policy.custody.path).iterdir())
    assert len(before) == 1
    committed = store.publish_once(policy, expected)
    assert committed.outcome == "COMMITTED_WITNESS"
    assert committed.digest_sha256 == hashlib.sha256(payload).hexdigest()
    assert committed.digest_sha256 == digest_witness_bytes(payload)
    path = owner_file(policy)
    assert path.read_bytes() == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_nlink == 1
    proof = store.observe(policy, expected_sha256=committed.digest_sha256)
    assert (
        proof.witness == expected
        or proof.witness == store.parse_witness_exact_bytes(payload)
    )
    assert proof.status == "OBSERVED_WITNESS_NOT_AUTHORITY"
    assert proof.digest_sha256 == committed.digest_sha256
    denied(
        lambda: store.publish_once(policy, expected),
        store.StoreReason.ALREADY_PUBLISHED,
    )
    assert path.read_bytes() == payload
    assert set(Path(policy.custody.path).iterdir()) == before | {path}


def test_missing_witness_never_bootstraps(policy: store.Policy) -> None:
    snapshot = list(Path(policy.custody.path).iterdir())
    denied(
        lambda: store.observe(policy, expected_sha256="0" * 64),
        store.StoreReason.WITNESS_MISSING,
    )
    assert list(Path(policy.custody.path).iterdir()) == snapshot


def test_digest_mismatch_and_wrong_identity(policy: store.Policy) -> None:
    witness_data = witness()
    result = store.publish_once(policy, witness_data)
    denied(
        lambda: store.observe(policy, expected_sha256="0" * 64),
        store.StoreReason.WITNESS_MISMATCH,
    )
    denied(
        lambda: store.publish_once(policy, witness(installation=9)),
        store.StoreReason.INVALID_WITNESS,
    )
    assert (
        store.observe(
            policy, expected_sha256=result.digest_sha256
        ).witness.installation_id
        == policy.installation_id
    )


def test_invalid_json_or_size_never_adopted(policy: store.Policy) -> None:
    f = owner_file(policy)
    f.write_text('{"installation_id":"forged"}')
    f.chmod(0o600)
    denied(
        lambda: store.observe(policy, expected_sha256="0" * 64),
        store.StoreReason.INVALID_WITNESS,
    )
    f.write_bytes(b"x" * (policy.limits.max_witness_bytes + 1))
    denied(
        lambda: store.observe(policy, expected_sha256="0" * 64),
        store.StoreReason.RESOURCE_EXCEEDED,
    )


def test_symlinked_record_never_followed(policy: store.Policy) -> None:
    f = owner_file(policy)
    f.symlink_to(Path(policy.custody.path) / "missing-external")
    denied(
        lambda: store.observe(policy, expected_sha256="0" * 64),
        store.StoreReason.UNAVAILABLE,
    )
    denied(
        lambda: store.publish_once(policy, witness()),
        store.StoreReason.ALREADY_PUBLISHED,
    )
    assert f.is_symlink()


def test_preexisting_stage_blocks_read_and_publish(policy: store.Policy) -> None:
    stray = (
        Path(policy.custody.path)
        / f".{store.record_name(policy.installation_id)}.stage-orphan"
    )
    stray.write_bytes(b"unfinished")
    snapshot = list(Path(policy.custody.path).iterdir())
    denied(
        lambda: store.publish_once(policy, witness()), store.StoreReason.ABANDONED_STAGE
    )
    denied(
        lambda: store.observe(policy, expected_sha256="0" * 64),
        store.StoreReason.ABANDONED_STAGE,
    )
    assert list(Path(policy.custody.path).iterdir()) == snapshot


def test_error_before_link_retains_stage_and_refuses_retry(
    policy: store.Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_link(*_args, **_kwargs):
        raise OSError("synthetic interrupted publication before link")

    with monkeypatch.context() as patcher:
        patcher.setattr(store.os, "link", fail_link)
        denied(
            lambda: store.publish_once(policy, witness()),
            store.StoreReason.UNCERTAIN_COMMIT,
        )
    assert not owner_file(policy).exists()
    stage = list(Path(policy.custody.path).glob(".*.stage-*"))
    assert len(stage) == 1
    denied(
        lambda: store.publish_once(policy, witness()), store.StoreReason.ABANDONED_STAGE
    )


def test_error_after_link_never_replaces_or_repairs(
    policy: store.Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    orig = store.os.fsync
    calls = 0

    def fail_dir_sync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic failure after no-replace link")
        orig(fd)

    with monkeypatch.context() as patcher:
        patcher.setattr(store.os, "fsync", fail_dir_sync)
        denied(
            lambda: store.publish_once(policy, witness()),
            store.StoreReason.UNCERTAIN_COMMIT,
        )
    assert owner_file(policy).exists()
    assert owner_file(policy).stat().st_nlink == 2
    assert list(Path(policy.custody.path).glob(".*.stage-*"))
    denied(
        lambda: store.publish_once(policy, witness()), store.StoreReason.ABANDONED_STAGE
    )
    denied(
        lambda: store.observe(policy, expected_sha256="0" * 64),
        store.StoreReason.ABANDONED_STAGE,
    )


def test_lock_busy_independent_process(policy: store.Policy) -> None:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()

    def holder(conn, p):
        import fcntl

        descriptor = os.open(
            p.lock.name,
            os.O_RDONLY,
            dir_fd=os.open(p.custody.path, os.O_RDONLY | os.O_DIRECTORY),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            conn.send(True)
            conn.recv()
        finally:
            os.close(descriptor)
            conn.close()

    proc = context.Process(target=holder, args=(child, policy))
    proc.start()
    child.close()
    try:
        assert parent.poll(8) and parent.recv()
        denied(
            lambda: store.publish_once(policy, witness()), store.StoreReason.LOCK_BUSY
        )
        parent.send("done")
        proc.join(8)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()
        parent.close()


def test_lock_and_directory_replacement_refuse(policy: store.Policy) -> None:
    lock_path = Path(policy.custody.path) / policy.lock.name
    retired = lock_path.with_name("retired-lock")
    lock_path.rename(retired)
    lock_path.write_bytes(b"")
    lock_path.chmod(0o600)
    denied(
        lambda: store.publish_once(policy, witness()), store.StoreReason.LOCK_CHANGED
    )
    d = Path(policy.custody.path)
    previous = d.with_name("old-custody")
    d.rename(previous)
    d.mkdir(mode=0o700)
    d.chmod(0o700)
    denied(
        lambda: store.publish_once(policy, witness()), store.StoreReason.CUSTODY_CHANGED
    )


def test_invalid_policy_limits_and_wrong_lock_name(policy: store.Policy) -> None:
    bad_policies = (
        replace(policy, record_mode=0o666),
        replace(policy, record_uid=True),
        replace(policy, installation_id="bad!"),
        replace(policy, limits=replace(policy.limits, max_witness_bytes=999999)),
        replace(
            policy,
            lock=replace(
                policy.lock, name="witness-" + policy.installation_id + ".json"
            ),
        ),
        replace(policy, custody=replace(policy.custody, path="/tmp/../unsafe")),
    )
    for bad in bad_policies:
        denied(
            lambda bad=bad: store.publish_once(bad, witness()),
            store.StoreReason.INVALID_POLICY,
        )


def test_no_owner_or_p0_side_effects_in_module() -> None:
    from pathlib import Path

    module_source = Path(store.__file__).read_text()
    # Static guard only: the module must not import runtime owner constructors.
    assert "from suzka.api" not in module_source
    assert "from suzka.installation_registration" not in module_source
    assert "os.replace" not in module_source
    assert "get_or_create_collection" not in module_source


def test_eight_independent_publishers_never_replace(policy: store.Policy) -> None:
    """One durable witness, or a refusal; never eight overwrites."""
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(8)
    results = ctx.Queue()

    def publish_contender(i, p, sync, queue):
        try:
            sync.wait(timeout=8)
            acknowledged = store.publish_once(p, witness())
            queue.put((i, acknowledged.outcome))
        except store.StoreDenied as exc:
            queue.put((i, exc.reason.value))

    workers = [
        ctx.Process(target=publish_contender, args=(i, policy, barrier, results))
        for i in range(8)
    ]
    for worker in workers:
        worker.start()
    try:
        outcomes = [results.get(timeout=9)[1] for _ in workers]
        assert outcomes.count("COMMITTED_WITNESS") == 1, outcomes
        assert all(
            result in {"COMMITTED_WITNESS", "lock_busy", "already_published"}
            for result in outcomes
        ), outcomes
    finally:
        for worker in workers:
            worker.join(3)
            if worker.is_alive():
                worker.kill()
                worker.join()
        results.close()
    expected = digest_witness_bytes(canonicalize_witness_bytes(witness()))
    assert store.observe(policy, expected_sha256=expected).digest_sha256 == expected


def test_entry_budget_reserves_stage_and_permanent_record(policy: store.Policy) -> None:
    baseline = {p.name for p in Path(policy.custody.path).iterdir()}
    assert baseline == {policy.lock.name}
    for limit in (1, 2):
        too_small = replace(
            policy, limits=replace(policy.limits, max_directory_entries=limit)
        )
        denied(
            lambda selected=too_small: store.publish_once(selected, witness()),
            store.StoreReason.RESOURCE_EXCEEDED,
        )
        assert {p.name for p in Path(policy.custody.path).iterdir()} == baseline
    # At most lock + stage + final name at the publication peak.
    exact = replace(policy, limits=replace(policy.limits, max_directory_entries=3))
    committed = store.publish_once(exact, witness())
    assert committed.outcome == "COMMITTED_WITNESS"
    assert (
        store.observe(exact, expected_sha256=committed.digest_sha256).digest_sha256
        == committed.digest_sha256
    )
    assert not list(Path(policy.custody.path).glob(".*.stage-*"))


def test_stage_mode_is_independent_of_restrictive_umask(policy: store.Policy) -> None:
    saved_umask = os.umask(0o777)
    try:
        committed = store.publish_once(policy, witness())
    finally:
        os.umask(saved_umask)
    assert committed.outcome == "COMMITTED_WITNESS"
    assert stat.S_IMODE(owner_file(policy).stat().st_mode) == 0o600
    assert (
        store.observe(policy, expected_sha256=committed.digest_sha256).digest_sha256
        == committed.digest_sha256
    )


def test_second_parent_fsync_failure_after_stage_unlink(
    policy: store.Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_fsync = store.os.fsync
    dir_calls = 0

    def inject_second_dir_sync(fd: int) -> None:
        nonlocal dir_calls
        observation = os.fstat(fd)
        if stat.S_ISDIR(observation.st_mode):
            dir_calls += 1
            if dir_calls == 2:
                raise OSError("synthetic second parent directory fsync failure")
        actual_fsync(fd)

    with monkeypatch.context() as patcher:
        patcher.setattr(store.os, "fsync", inject_second_dir_sync)
        denied(
            lambda: store.publish_once(policy, witness()),
            store.StoreReason.UNCERTAIN_COMMIT,
        )
    assert dir_calls == 2
    target = owner_file(policy)
    assert target.exists() and target.stat().st_nlink == 1
    assert not list(Path(policy.custody.path).glob(".*.stage-*"))
    # Visible canonical bytes are DATA for future manual operator
    # classification, not a new genesis or startup authorization.
    exact_digest = digest_witness_bytes(canonicalize_witness_bytes(witness()))
    assert (
        store.observe(policy, expected_sha256=exact_digest).digest_sha256
        == exact_digest
    )
    denied(
        lambda: store.publish_once(policy, witness()),
        store.StoreReason.ALREADY_PUBLISHED,
    )
