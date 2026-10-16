"""Rootless lifetime lock mechanics: only disposable operator/Source fixtures."""

from __future__ import annotations

import multiprocessing
import os
import stat
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from suzka import source_lifetime_lock as s


def directory(path: Path) -> s.Directory:
    st = path.stat()
    return s.Directory(
        str(path), st.st_dev, st.st_ino, st.st_uid, st.st_gid, stat.S_IMODE(st.st_mode)
    )


@pytest.fixture
def policy(tmp_path: Path) -> Generator[s.Policy, None, None]:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    source.chmod(0o700)
    operator = tmp_path / "operator"
    operator.mkdir(mode=0o700)
    operator.chmod(0o700)
    identity = directory(source)
    name = s.lock_name(identity.device, identity.inode)
    lock_path = operator / name
    lock_path.write_bytes(b"")
    lock_path.chmod(0o600)
    ls = lock_path.stat()
    p = s.Policy(
        source=identity,
        custody=directory(operator),
        lock=s.LockFile(
            name, ls.st_dev, ls.st_ino, ls.st_uid, ls.st_gid, stat.S_IMODE(ls.st_mode)
        ),
        limits=s.Limits(max_path_bytes=4096, max_path_components=256),
    )
    yield p


def denied(call, reason: s.ClaimReason) -> None:
    with pytest.raises(s.ClaimDenied) as exc:
        call()
    assert exc.value.reason == reason
    assert "/" not in str(exc.value)


@contextmanager
def disposable_claim(policy: s.Policy) -> Generator[s.LiveClaim, None, None]:
    """TEST-ONLY cleanup: no managed worker is running in these fixtures."""
    claim = s.acquire(policy)
    try:
        yield claim
    finally:
        claim.close()


def test_one_source_one_holder_two_installations(policy: s.Policy) -> None:
    # Policy has deliberately no installation id: distinct installation
    # registries referencing one physical dev+inode must contend on one lock.
    assert policy.lock.name == s.lock_name(policy.source.device, policy.source.inode)
    before = set(Path(policy.custody.path).iterdir())
    first = s.acquire(policy)
    try:
        assert first.held
        first.recheck()
        denied(lambda: s.acquire(replace(policy)), s.ClaimReason.LOCK_BUSY)
        assert set(Path(policy.custody.path).iterdir()) == before
    finally:
        first.close()
    assert not first.held
    denied(first.recheck, s.ClaimReason.CLOSED)
    with disposable_claim(policy) as next_claim:
        assert next_claim.held
        next_claim.recheck()


def test_separate_process_owner_blocks_other_installation(policy: s.Policy) -> None:
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe()

    def worker(pipe, source_policy):
        try:
            with disposable_claim(source_policy) as claim:
                pipe.send(("held", claim.held))
                pipe.recv()
                claim.recheck()
            pipe.send(("released", True))
        except (s.ClaimDenied, OSError, ValueError) as exc:
            pipe.send(("error", type(exc).__name__))
        finally:
            pipe.close()

    proc = ctx.Process(target=worker, args=(child, policy))
    proc.start()
    child.close()
    try:
        assert parent.poll(6)
        assert parent.recv() == ("held", True)
        denied(lambda: s.acquire(policy), s.ClaimReason.LOCK_BUSY)
        parent.send("release")
        assert parent.poll(6)
        assert parent.recv() == ("released", True)
        proc.join(6)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()
        parent.close()
    with disposable_claim(policy):
        pass


def test_source_inode_replacement_denied_without_lock_creation(
    policy: s.Policy,
) -> None:
    path = Path(policy.source.path)
    old = path.with_name("retired-root")
    path.rename(old)
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    snapshot = list(Path(policy.custody.path).iterdir())
    denied(lambda: s.acquire(policy), s.ClaimReason.SOURCE_CHANGED)
    assert list(Path(policy.custody.path).iterdir()) == snapshot


def test_source_replaced_during_lifetime_detected(policy: s.Policy) -> None:
    with disposable_claim(policy) as claim:
        original = Path(policy.source.path)
        original.rename(original.with_name("old-source"))
        original.mkdir(mode=0o700)
        original.chmod(0o700)
        denied(claim.recheck, s.ClaimReason.SOURCE_CHANGED)


def test_missing_lock_never_bootstraps(policy: s.Policy) -> None:
    lock = Path(policy.custody.path) / policy.lock.name
    lock.unlink()
    before = list(Path(policy.custody.path).iterdir())
    denied(lambda: s.acquire(policy), s.ClaimReason.LOCK_MISSING)
    assert list(Path(policy.custody.path).iterdir()) == before


def test_lock_swap_detected_even_while_old_fd_held(policy: s.Policy) -> None:
    with disposable_claim(policy) as held:
        old = Path(policy.custody.path) / policy.lock.name
        old.rename(old.with_name("operator-old-claim"))
        replacement = Path(policy.custody.path) / policy.lock.name
        replacement.touch(mode=0o600)
        replacement.chmod(0o600)
        denied(held.recheck, s.ClaimReason.LOCK_CHANGED)
        denied(lambda: s.acquire(policy), s.ClaimReason.LOCK_CHANGED)


def test_lock_symlink_denied(policy: s.Policy) -> None:
    lock = Path(policy.custody.path) / policy.lock.name
    original = lock.with_name("original-lock")
    lock.rename(original)
    lock.symlink_to(original.name)
    denied(lambda: s.acquire(policy), s.ClaimReason.UNSTABLE)
    # Although future operator recovery is required, the normal caller never
    # removes this symlink or reconstructs a new lock.
    assert lock.is_symlink()


def test_source_symlink_denied(policy: s.Policy) -> None:
    original = Path(policy.source.path)
    old = original.with_name("old-source")
    original.rename(old)
    original.symlink_to(old, target_is_directory=True)
    denied(lambda: s.acquire(policy), s.ClaimReason.UNSTABLE)


def test_operator_custody_replacement_detected(policy: s.Policy) -> None:
    with disposable_claim(policy) as lease:
        root = Path(policy.custody.path)
        root.rename(root.with_name("retired-operator"))
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        denied(lease.recheck, s.ClaimReason.CUSTODY_CHANGED)


def test_fail_closed_invalid_keys_paths_modes(policy: s.Policy) -> None:
    cases = (
        replace(policy, lock=replace(policy.lock, name="per-install-id.lock")),
        replace(policy, source=replace(policy.source, device=policy.source.device + 1)),
        replace(policy, custody=replace(policy.custody, mode=0o777)),
        replace(policy, custody=replace(policy.custody, path=policy.source.path)),
        replace(policy, source=replace(policy.source, path="/tmp/../source")),
        replace(policy, limits=replace(policy.limits, max_path_bytes=10000)),
        replace(policy, lock=replace(policy.lock, mode=0o666)),
    )
    for wrong in cases:
        with pytest.raises(s.ClaimDenied):
            s.acquire(wrong)


def test_old_lock_fd_survives_short_registration_window(policy: s.Policy) -> None:
    # A caller may operate a separate short-lived P0 lock while this lock
    # stays held; here a second preprovisioned file stands in for P0 mechanics.
    registration_lock = Path(policy.custody.path) / "registration.lock"
    registration_lock.touch(mode=0o600)
    with disposable_claim(policy) as lease:
        for _ in range(4):
            with registration_lock.open("rb") as handle:
                os.fstat(handle.fileno())
            lease.recheck()
            denied(lambda: s.acquire(policy), s.ClaimReason.LOCK_BUSY)


def test_lock_key_rejects_boolean_empty_or_negative_values() -> None:
    for dev, inode in ((True, 1), (1, 0), (-1, 2), (3, False), (2**63, 2)):
        denied(
            lambda dev=dev, inode=inode: s.lock_name(dev, inode),
            s.ClaimReason.INVALID_POLICY,
        )


def test_eight_installation_contenders_only_one_holder(policy: s.Policy) -> None:
    """Distinct managed instances sharing physical Source race for one lock."""
    import time

    ctx = multiprocessing.get_context("fork")
    start = ctx.Barrier(8)
    results = ctx.Queue()

    def contender(i, bound_policy, barrier, output):
        try:
            barrier.wait(timeout=6)
            with disposable_claim(bound_policy):
                output.put((i, "held"))
                time.sleep(0.45)
        except s.ClaimDenied as exc:
            output.put((i, exc.reason.value))

    processes = [
        ctx.Process(target=contender, args=(i, policy, start, results))
        for i in range(8)
    ]
    for proc in processes:
        proc.start()
    try:
        actual = [results.get(timeout=7) for _ in processes]
        assert sum(result == "held" for _, result in actual) == 1, actual
        assert sum(result == "lock_busy" for _, result in actual) == 7, actual
    finally:
        for proc in processes:
            proc.join(3)
            if proc.is_alive():
                proc.kill()
                proc.join()
        results.close()
    with disposable_claim(policy):
        pass


def test_source_zero_device_static_compatibility(policy: s.Policy) -> None:
    assert s.lock_name(0, 1) != s.lock_name(1, 1)
    p = replace(
        policy,
        source=replace(policy.source, device=0),
        lock=replace(policy.lock, name=s.lock_name(0, policy.source.inode)),
    )
    # No fake real Source: only the static P0/P1b-compatible policy shape.
    s._validate(p)
    denied(lambda: s.acquire(p), s.ClaimReason.SOURCE_CHANGED)


def test_no_automatic_release_after_cancelled_coordinator(policy: s.Policy) -> None:
    assert not hasattr(s.LiveClaim, "__enter__")
    assert not hasattr(s.LiveClaim, "__exit__")
    claim = s.acquire(policy)
    try:
        with pytest.raises(RuntimeError, match="synthetic cancellation"):
            raise RuntimeError("synthetic cancellation")
        assert claim.held
        denied(lambda: s.acquire(policy), s.ClaimReason.LOCK_BUSY)
    finally:
        # In this isolated fixture no managed child exists.
        claim.close()
    with disposable_claim(policy) as fresh:
        fresh.recheck()


def test_recheck_missing_paths_are_typed_and_path_free(policy: s.Policy) -> None:
    claim = s.acquire(policy)
    source = Path(policy.source.path)
    retired_source = source.with_name("retired-source")
    custody = Path(policy.custody.path)
    retired_custody = custody.with_name("retired-custody")
    lock = custody / policy.lock.name
    retired_lock = custody / "retired-lock"
    try:
        source.rename(retired_source)
        denied(claim.recheck, s.ClaimReason.SOURCE_CHANGED)
        retired_source.rename(source)
        claim.recheck()
        lock.rename(retired_lock)
        denied(claim.recheck, s.ClaimReason.LOCK_CHANGED)
        retired_lock.rename(lock)
        claim.recheck()
        custody.rename(retired_custody)
        denied(claim.recheck, s.ClaimReason.CUSTODY_CHANGED)
        retired_custody.rename(custody)
        claim.recheck()
    finally:
        if retired_source.exists() and not source.exists():
            retired_source.rename(source)
        if retired_custody.exists() and not custody.exists():
            retired_custody.rename(custody)
        if retired_lock.exists() and not lock.exists():
            retired_lock.rename(lock)
        claim.close()


def test_fork_inherited_lock_survives_parent_close(policy: s.Policy) -> None:
    """Avoid explicit LOCK_UN releasing an inherited file-description lock."""
    context = multiprocessing.get_context("fork")
    claim = s.acquire(policy)
    parent, child = context.Pipe()

    def inherited_holder(pipe):
        try:
            pipe.send(("inherited", claim.held))
            pipe.recv()
        finally:
            pipe.close()

    process = context.Process(target=inherited_holder, args=(child,))
    process.start()
    child.close()
    try:
        assert parent.poll(6)
        assert parent.recv() == ("inherited", False)
        claim.close()  # Child still has the same flock file description.
        denied(lambda: s.acquire(policy), s.ClaimReason.LOCK_BUSY)
        parent.send("exit")
        process.join(6)
        assert process.exitcode == 0
    finally:
        if claim.held:
            claim.close()
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
    with disposable_claim(policy) as reacquired:
        reacquired.recheck()
