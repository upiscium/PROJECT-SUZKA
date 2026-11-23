"""DC-1 (F15 slice D1 r3 sec. 5.1 / 10): classifier, quarantine writer,
four-state manifest and its crash matrix, schedules, health record."""

from __future__ import annotations

from datetime import datetime, timezone
import errno
import importlib
import json
import os
from pathlib import Path
import pkgutil
import stat
from typing import Any

import pytest

import suzka
from suzka.runtime import degraded_continuation as dc
from suzka.runtime.degraded_continuation import (
    DcState,
    DegradedContinuationStatus,
    Deadline,
    IntegrityVerdict,
    ManifestError,
    ManifestState,
    QuarantineError,
    QuarantineRefused,
    QuarantineWriter,
    ResumeAction,
    ResumeFacts,
    classify_cycle_exception,
    classify_participant_error,
    is_quarantinable,
    must_propagate,
    plan_resume,
    probe_schedule,
    reeval_delay,
    restart_delay,
    verdict_for_band,
)
from suzka.runtime.event_journal import (
    EventJournalAdmissionBand,
    EventJournalIntegrityError,
    EventJournalLease,
    EventJournalLoadError,
)
from suzka.runtime.startup_reconciliation import StartupReconciliationError
from suzka.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    ParticipantUnavailableError,
    UnsupportedParticipantReconciliationError,
)
from tests.dc_kit import (
    QuarantineAccessSpy,
    QuarantineAccessViolation,
    assert_trees_equal,
    tree_snapshot,
)

EXPERIENCE = "memory.experience"
SEMANTIC = "memory.semantic"
V = IntegrityVerdict


def _named(name: str, module: str, base: type[BaseException] = Exception) -> type:
    return type(name, (base,), {"__module__": module})


EvidenceAbsent = _named(
    "ParticipantEvidenceAbsentError",
    "suzka.runtime.transaction_coordinator",
    UnsupportedParticipantReconciliationError,
)


# ------------------------------------------------------------ classifier


@pytest.mark.parametrize("participant", [EXPERIENCE, SEMANTIC])
def test_memory_domain_participant_verdicts_by_class_name(participant: str) -> None:
    assert (
        classify_participant_error(participant, ParticipantDivergedError())
        is V.DIVERGED
    )
    assert classify_participant_error(participant, EvidenceAbsent()) is V.DIVERGED
    assert (
        classify_participant_error(participant, ParticipantUnavailableError())
        is V.TRANSIENT
    )
    assert classify_participant_error(participant, PermissionError()) is V.TRANSIENT
    assert classify_participant_error(participant, FileNotFoundError()) is V.TRANSIENT
    # A plain "resolver not registered" is a code defect: never quarantinable.
    assert (
        classify_participant_error(
            participant, UnsupportedParticipantReconciliationError()
        )
        is V.PROTECTED_LOST
    )
    assert classify_participant_error(participant, RuntimeError()) is V.PROTECTED_LOST


@pytest.mark.parametrize(
    "participant", ["subject.history", "memory.episodic", "session.turn", "x.unknown"]
)
def test_other_participants_are_never_diverged_or_transient(participant: str) -> None:
    for error in (
        ParticipantDivergedError(),
        EvidenceAbsent(),
        ParticipantUnavailableError(),
        PermissionError(),
        UnsupportedParticipantReconciliationError(),
    ):
        assert classify_participant_error(participant, error) is V.PROTECTED_LOST
        assert not is_quarantinable(participant, V.DIVERGED)


def test_unknown_class_and_subclass_of_a_known_class_are_protected_lost() -> None:
    Unknown = _named("Whatever", "somewhere.else")
    Subclass = type("Sub", (ParticipantDivergedError,), {})
    assert classify_participant_error(EXPERIENCE, Unknown()) is V.PROTECTED_LOST
    # exact class names only: a subclass added later is not quarantinable
    assert classify_participant_error(EXPERIENCE, Subclass()) is V.PROTECTED_LOST
    assert classify_cycle_exception(Unknown()).verdict is V.PROTECTED_LOST
    # a suzka subclass of OSError is not "the OSError family"
    SuzkaOSError = _named("StoreOSError", "suzka.memory.x", OSError)
    assert classify_cycle_exception(SuzkaOSError()).verdict is V.PROTECTED_LOST
    assert classify_participant_error(EXPERIENCE, SuzkaOSError()) is V.PROTECTED_LOST


def test_cycle_exception_classes_at_start_and_restart() -> None:
    assert (
        classify_cycle_exception(StartupReconciliationError("x")).verdict is V.AMBIGUOUS
    )
    assert (
        classify_cycle_exception(PermissionError(errno.EACCES, "x")).verdict
        is V.TRANSIENT
    )
    assert classify_cycle_exception(OSError()).verdict is V.TRANSIENT
    assert classify_cycle_exception(TimeoutError()).verdict is V.TRANSIENT
    for error in (
        RuntimeError("x"),
        ValueError("x"),
        AssertionError("x"),
        MemoryError(),
        KeyError("x"),
        EventJournalIntegrityError("torn tail"),
        EventJournalLoadError("corrupt"),
        ParticipantDivergedError(),
    ):
        failure = classify_cycle_exception(error)
        assert failure.verdict is V.PROTECTED_LOST, error
        assert failure.label == f"{type(error).__module__}.{type(error).__qualname__}"


def test_a_busy_lease_is_transient_and_a_corrupt_journal_is_not(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    journal_path = tmp_path / "journal.jsonl"
    first = EventJournalLease(journal_path)
    try:
        # The REAL raise site: pins the message the classifier keys on.
        with pytest.raises(EventJournalLoadError) as busy:
            EventJournalLease(journal_path)
        failure = classify_cycle_exception(busy.value)
        assert failure.verdict is V.TRANSIENT
        assert failure.label == dc.LEASE_BUSY_LABEL
    finally:
        first.close()
    corrupt = classify_cycle_exception(
        EventJournalLoadError("Journal record is corrupt")
    )
    assert corrupt.verdict is V.PROTECTED_LOST


def _all_suzka_exception_classes() -> set[type[BaseException]]:
    for module in pkgutil.walk_packages(suzka.__path__, "suzka."):
        importlib.import_module(module.name)
    found: set[type[BaseException]] = set()
    pending = [BaseException]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            if sub not in found:
                found.add(sub)
                pending.append(sub)
    return found


def test_every_exception_class_has_a_verdict_none_is_undefined() -> None:
    classes = _all_suzka_exception_classes()
    assert len(classes) > 50  # builtins + the suzka hierarchy were really walked
    assert any(c.__module__.startswith("suzka.") for c in classes)
    for cls in classes:
        try:
            error = cls.__new__(cls)  # no __init__: some classes need arguments
        except TypeError:
            continue
        failure = classify_cycle_exception(error)
        assert isinstance(failure.verdict, IntegrityVerdict), cls
        assert failure.label, cls
        assert classify_participant_error(EXPERIENCE, error) in (
            V.DIVERGED,
            V.TRANSIENT,
            V.PROTECTED_LOST,
        ), cls
        assert classify_participant_error("subject.history", error) is V.PROTECTED_LOST
        # nothing but the three exits and cancellation propagates
        assert must_propagate(error) == (not isinstance(error, Exception)), cls


def test_only_exit_and_cancellation_propagate() -> None:
    import asyncio

    for exit_like in (
        KeyboardInterrupt(),
        SystemExit(),
        GeneratorExit(),
        asyncio.CancelledError(),
    ):
        assert must_propagate(exit_like)
    for failure in (RuntimeError(), OSError(), MemoryError(), AssertionError()):
        assert not must_propagate(failure)


def test_table_names_resolve_to_real_classes_except_the_pending_ones() -> None:
    unresolved = set()
    for name in dc.CLASSIFIED_CLASS_NAMES:
        module_name, _, class_name = name.rpartition(".")
        module = importlib.import_module(module_name)
        if not hasattr(module, class_name):
            unresolved.add(name)
    # Only names a later unit introduces (DC-4's ParticipantEvidenceAbsentError)
    # may be unresolved; a renamed existing class must fail this test.
    assert unresolved <= dc.PENDING_CLASS_NAMES


def test_band_and_quarantinable_verdicts() -> None:
    assert verdict_for_band(EventJournalAdmissionBand.HARD) is V.BAND_HARD
    assert verdict_for_band(EventJournalAdmissionBand.NORMAL) is V.OK
    assert verdict_for_band(EventJournalAdmissionBand.SOFT) is V.OK
    assert verdict_for_band("hard") is V.PROTECTED_LOST
    assert verdict_for_band(None) is V.PROTECTED_LOST
    assert is_quarantinable(EXPERIENCE, V.DIVERGED)
    assert is_quarantinable(SEMANTIC, V.DIVERGED)
    for verdict in (V.TRANSIENT, V.AMBIGUOUS, V.PROTECTED_LOST, V.BAND_HARD, V.OK):
        assert not is_quarantinable(EXPERIENCE, verdict)
    # HD-new-2 (b)/(c) only widens it to TRANSIENT, and only when told to
    assert is_quarantinable(EXPERIENCE, V.TRANSIENT, quarantine_unavailable=True)
    assert not is_quarantinable(
        "memory.episodic", V.DIVERGED, quarantine_unavailable=True
    )


# --------------------------------------------------------------- schedules


def test_probe_schedule_sums_to_the_shared_deadline() -> None:
    assert probe_schedule(60) == (1, 2, 4, 8, 16, 29)
    for total in (1, 2, 3, 10, 60, 61, 300.5):
        assert sum(probe_schedule(total)) == pytest.approx(total)
    assert probe_schedule(0) == ()


def test_backoff_schedules_are_bounded() -> None:
    assert [restart_delay(i) for i in range(6)] == [1, 4, 16, 60, 60, 60]
    assert [reeval_delay(i) for i in range(8)] == [1, 4, 16, 60, 300, 300, 300, 300]
    assert reeval_delay(7, max_backoff_s=120) == 120
    with pytest.raises(ValueError):
        restart_delay(-1)


def test_deadline_is_shared_and_monotonic() -> None:
    now = [100.0]
    deadline = Deadline(30, lambda: now[0])
    assert not deadline.expired() and deadline.remaining() == 30
    now[0] = 129.5
    assert not deadline.expired()
    now[0] = 130.0
    assert deadline.expired() and deadline.remaining() == 0


# -------------------------------------------------------------- the writer


def _make_store(parent: Path, name: str = "experience") -> Path:
    root = parent / name
    (root / "records").mkdir(parents=True)
    (root / "pending").mkdir()
    (root / "receipts").mkdir()
    (root / "records" / "a.json").write_bytes(b'{"id": "a"}\n')
    (root / "records" / "b.json").write_bytes(os.urandom(4096))
    (root / "pending" / "p.json").write_bytes(b"torn{")
    os.symlink("records/a.json", root / "link")
    return root


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, Path, QuarantineWriter]:
    tmp_path.chmod(0o700)
    memory = tmp_path / "memory"
    memory.mkdir()
    qdir = tmp_path / "quarantine"
    return (
        memory,
        qdir,
        QuarantineWriter(
            qdir,
            has_exclusive_authority=lambda: True,
            wall_clock=lambda: datetime(2026, 11, 24, tzinfo=timezone.utc),
        ),
    )


def _quarantine(writer: QuarantineWriter, root: Path, epoch: str = "e1", **kw: Any):
    return writer.quarantine_directory(
        root,
        reason="participant_diverged",
        participant_id=EXPERIENCE,
        epoch_id=epoch,
        journal_lineage_id="lineage",
        journal_last_record_hash="h" * 64,
        runtime_instance_id="run-1",
        transaction_ids=("t1",),
        **kw,
    )


def test_quarantine_moves_by_rename_byte_for_byte(world) -> None:  # type: ignore[no-untyped-def]
    memory, qdir, writer = world
    root = _make_store(memory)
    before = tree_snapshot(root)
    inode = root.stat().st_ino

    manifest = _quarantine(writer, root)

    target = memory / "experience.quarantine-e1"
    assert not root.exists() and target.is_dir()
    assert target.stat().st_ino == inode  # a rename, not a copy
    assert_trees_equal(before, tree_snapshot(target))
    assert manifest.state is ManifestState.MOVED
    (artifact,) = manifest.artifacts
    assert artifact.verified is True
    assert artifact.quarantine_path == str(target)
    paths = {entry[0]: entry for entry in artifact.entries}
    assert paths["link"][1] == "symlink" and paths["link"][3] is None
    assert paths["records"][1] == "dir"
    assert paths["records/a.json"][3] is not None
    assert os.readlink(target / "link") == "records/a.json"  # recorded, not followed
    # the sidecar holds hashes, sizes and paths only, never payload bytes
    text = (qdir / "e1" / "manifest.json").read_text()
    assert "torn{" not in text and '"id": "a"' not in text
    assert json.loads(text)["schema"] == "suzka.dc.manifest.v1"
    assert writer.read_manifest("e1") == manifest
    assert writer.list_epochs() == ("e1",)


def test_manifest_walks_four_states_in_order_and_never_back(world) -> None:  # type: ignore[no-untyped-def]
    memory, _, writer = world
    _quarantine(writer, _make_store(memory))
    with pytest.raises(ManifestError):
        writer.mark_committed("e1")  # skipping JOURNALED
    journaled = writer.mark_journaled("e1", transaction_ids=("t1", "t2"))
    assert journaled.state is ManifestState.JOURNALED
    assert journaled.transaction_ids == ("t1", "t2")
    assert writer.mark_journaled("e1").state is ManifestState.JOURNALED  # idempotent
    with pytest.raises(ManifestError):
        writer.advance("e1", ManifestState.MOVED)  # backwards
    assert writer.mark_committed("e1").state is ManifestState.COMMITTED
    with pytest.raises(ManifestError):
        writer.advance("e1", ManifestState.PENDING)


@pytest.mark.parametrize(
    "participant", ["subject.history", "memory.episodic", "session.turn", "other"]
)
def test_refuses_a_participant_that_is_never_quarantined(
    world, participant: str
) -> None:  # type: ignore[no-untyped-def]
    memory, qdir, writer = world
    root = _make_store(memory)
    before = tree_snapshot(root)
    with pytest.raises(QuarantineRefused):
        writer.quarantine_directory(
            root, reason="r", participant_id=participant, epoch_id="e1"
        )
    assert_trees_equal(before, tree_snapshot(root))
    assert not qdir.exists()


@pytest.mark.parametrize(
    "verdict", [V.TRANSIENT, V.AMBIGUOUS, V.PROTECTED_LOST, V.BAND_HARD, V.OK]
)
def test_refuses_a_verdict_that_is_not_diverged_by_default(world, verdict) -> None:  # type: ignore[no-untyped-def]
    memory, qdir, writer = world
    root = _make_store(memory)
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, root, verdict=verdict)
    assert root.is_dir() and not qdir.exists()


def test_refusals_change_nothing(world, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    memory, qdir, writer = world
    root = _make_store(memory)
    before = tree_snapshot(root)

    # the target name already exists
    taken = memory / "experience.quarantine-e1"
    taken.mkdir()
    (taken / "keep").write_bytes(b"keep")
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, root)
    assert (taken / "keep").read_bytes() == b"keep"
    assert_trees_equal(before, tree_snapshot(root))
    assert not qdir.exists()

    # bad epoch ids (path traversal included)
    for bad in ("../x", "a/b", "", "x" * 65, ".hidden", "a b"):
        with pytest.raises(QuarantineRefused):
            _quarantine(writer, root, epoch=bad)

    # root is a symlink to a directory
    link = memory / "linked"
    link.symlink_to(root)
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, link)
    assert link.is_symlink()
    # root is a file / missing
    plain = memory / "plain"
    plain.write_bytes(b"x")
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, plain)
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, memory / "absent")
    # root is itself a quarantined name
    already = memory / "x.quarantine-old"
    already.mkdir()
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, already, epoch="e2")
    # the parent is not owned by this process's uid
    real = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real + 1)
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, root, epoch="e3")
    monkeypatch.undo()
    assert_trees_equal(before, tree_snapshot(root))
    assert not qdir.exists()

    # one epoch is used once
    _quarantine(writer, root, epoch="e4")
    other = _make_store(memory, "semantic")
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, other, epoch="e4")
    assert other.is_dir()


def test_unreadable_file_is_recorded_and_the_move_still_happens(world) -> None:  # type: ignore[no-untyped-def]
    memory, _, writer = world
    root = _make_store(memory)
    secret = root / "records" / "b.json"
    secret.chmod(0)
    try:
        if os.access(secret, os.R_OK):
            pytest.skip("permission bits are not enforced for this user")
        manifest = _quarantine(writer, root)
    finally:
        (memory / "experience.quarantine-e1" / "records" / "b.json").chmod(0o600)
    (artifact,) = manifest.artifacts
    assert [item[0] for item in artifact.read_errors] == ["records/b.json"]
    assert not root.exists()
    assert manifest.state is ManifestState.MOVED


def test_hash_budget_exhaustion_records_unhashed_and_still_renames(world) -> None:  # type: ignore[no-untyped-def]
    memory, qdir, _ = world
    root = _make_store(memory)
    before = tree_snapshot(root)
    ticks = iter(range(10_000))
    # every clock read advances 10 s: the 30 s budget ends inside the first pass
    writer = QuarantineWriter(
        qdir,
        has_exclusive_authority=lambda: True,
        t_hash_s=30,
        clock=lambda: next(ticks) * 10.0,
    )
    manifest = _quarantine(writer, root)
    (artifact,) = manifest.artifacts
    assert artifact.unhashed  # recorded, not silently dropped
    # the post-move check was partial, and the manifest must not claim more
    assert artifact.verified is None
    assert all(
        entry[3] is None for entry in artifact.entries if entry[0] in artifact.unhashed
    )
    assert not root.exists()
    assert_trees_equal(before, tree_snapshot(memory / "experience.quarantine-e1"))
    assert manifest.state is ManifestState.MOVED


def test_a_fifo_in_the_tree_cannot_block_the_hash(world) -> None:  # type: ignore[no-untyped-def]
    memory, _, writer = world
    root = _make_store(memory)
    os.mkfifo(root / "fifo")
    manifest = _quarantine(writer, root)
    kinds = {e[0]: e[1] for e in manifest.artifacts[0].entries}
    assert kinds["fifo"] == "other"
    assert stat.S_ISFIFO(os.lstat(memory / "experience.quarantine-e1" / "fifo").st_mode)


def test_rename_failure_moves_nothing_deletes_nothing_and_stays_pending(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, qdir, writer = world
    root = _make_store(memory)
    before = tree_snapshot(root)

    def denied(source: Path, target: Path) -> None:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(dc, "_rename_directory", denied)
    with pytest.raises(QuarantineError):
        _quarantine(writer, root)
    assert_trees_equal(before, tree_snapshot(root))
    assert not (memory / "experience.quarantine-e1").exists()
    assert writer.read_manifest("e1").state is ManifestState.PENDING


def test_a_changed_tree_after_the_move_is_not_marked_moved(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _, writer = world
    root = _make_store(memory)
    real = dc._rename_directory

    def rename_then_damage(source: Path, target: Path) -> None:
        real(source, target)
        # same size, different bytes: only the sha256 compare can see it
        (target / "records" / "a.json").write_bytes(b'{"id": "Z"}\n')

    monkeypatch.setattr(dc, "_rename_directory", rename_then_damage)
    with pytest.raises(QuarantineError):
        _quarantine(writer, root)
    assert writer.read_manifest("e1").state is ManifestState.PENDING
    monkeypatch.undo()
    plan = writer.resume("e1", root)
    assert plan.action is ResumeAction.BLOCKED  # hashes differ: no automatic action


def test_the_writer_never_writes_under_a_quarantined_path(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    with pytest.raises(QuarantineRefused):
        dc._write_new_file(tmp_path / "x.quarantine-e" / "f", b"x")
    with pytest.raises(QuarantineRefused):
        dc._publish_file(tmp_path / "a", tmp_path / "x.quarantine-e" / "f")
    # a quarantine_dir under such a name is refused too (nothing is created)
    memory = tmp_path / "memory"
    memory.mkdir()
    root = _make_store(memory)
    writer = QuarantineWriter(
        tmp_path / "q.quarantine-z", has_exclusive_authority=lambda: True
    )
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, root)
    assert root.is_dir() and not (tmp_path / "q.quarantine-z").exists()


# ---------------------------------------------------------- crash matrix


class _Crash(BaseException):
    """The process dies here (BaseException: no ``except Exception`` runs)."""


def _crash_before_manifest_write(monkeypatch: pytest.MonkeyPatch, number: int) -> None:
    """Die just before the ``number``-th manifest write (1 PENDING, 2 MOVED,
    3 JOURNALED, 4 COMMITTED)."""

    real = QuarantineWriter._write_manifest
    count = [0]

    def write(self: QuarantineWriter, manifest: Any) -> None:
        count[0] += 1
        if count[0] == number:
            raise _Crash
        real(self, manifest)

    monkeypatch.setattr(QuarantineWriter, "_write_manifest", write)


def _reborn(writer: QuarantineWriter) -> QuarantineWriter:
    """A new process: a new writer over the same directory."""

    return QuarantineWriter(writer._dir, has_exclusive_authority=lambda: True)


def test_crash_before_the_pending_manifest_moves_nothing(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _, writer = world
    root = _make_store(memory)
    before = tree_snapshot(root)
    _crash_before_manifest_write(monkeypatch, 1)
    with pytest.raises(_Crash):
        _quarantine(writer, root)
    monkeypatch.undo()
    assert_trees_equal(before, tree_snapshot(root))
    assert not (memory / "experience.quarantine-e1").exists()
    with pytest.raises(ManifestError):  # no manifest: nothing to resume
        _reborn(writer).read_manifest("e1")


def test_crash_after_pending_before_the_rename_redoes_the_move(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _, writer = world
    root = _make_store(memory)
    before = tree_snapshot(root)

    def die(source: Path, target: Path) -> None:
        raise _Crash

    monkeypatch.setattr(dc, "_rename_directory", die)
    with pytest.raises(_Crash):
        _quarantine(writer, root)
    monkeypatch.undo()
    assert writer.read_manifest("e1").state is ManifestState.PENDING
    plan = _reborn(writer).resume("e1", root)
    assert plan.action is ResumeAction.CONTINUE_JOURNAL
    assert _reborn(writer).read_manifest("e1").state is ManifestState.MOVED
    assert_trees_equal(before, tree_snapshot(memory / "experience.quarantine-e1"))


def test_crash_after_the_rename_before_moved_is_recorded_as_moved(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _, writer = world
    root = _make_store(memory)
    before = tree_snapshot(root)
    _crash_before_manifest_write(monkeypatch, 2)
    with pytest.raises(_Crash):
        _quarantine(writer, root)
    monkeypatch.undo()
    assert not root.exists()
    again = _reborn(writer)
    assert again.read_manifest("e1").state is ManifestState.PENDING
    facts = again.observe(again.read_manifest("e1"), root)
    assert (facts.original_present, facts.quarantined_present) == (False, True)
    assert facts.quarantined_hashes_equal
    plan = again.resume("e1", root)
    assert plan.action is ResumeAction.CONTINUE_JOURNAL
    assert again.read_manifest("e1").state is ManifestState.MOVED
    assert_trees_equal(before, tree_snapshot(memory / "experience.quarantine-e1"))


def test_crash_after_moved_continues_at_the_journal_step(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _, writer = world
    _quarantine(writer, _make_store(memory))
    _crash_before_manifest_write(monkeypatch, 1)  # the JOURNALED write dies
    with pytest.raises(_Crash):
        writer.mark_journaled("e1")
    monkeypatch.undo()
    again = _reborn(writer)
    assert again.read_manifest("e1").state is ManifestState.MOVED
    assert (
        again.resume("e1", memory / "experience").action
        is ResumeAction.CONTINUE_JOURNAL
    )
    # idempotent: resuming does not change the state
    assert again.read_manifest("e1").state is ManifestState.MOVED


def test_crash_after_journaled_creates_the_empty_root_or_commits(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _, writer = world
    root = _make_store(memory)
    _quarantine(writer, root)
    writer.mark_journaled("e1")
    again = _reborn(writer)
    # no live root yet: step 4 (the caller creates the empty root + marker)
    assert again.resume("e1", root).action is ResumeAction.CREATE_EMPTY_ROOT
    # the empty root with THIS epoch's marker exists, the COMMITTED write died
    root.mkdir()
    (root / dc.EPOCH_MARKER_NAME).write_text(json.dumps({"epoch_id": "e1"}))
    _crash_before_manifest_write(monkeypatch, 1)
    with pytest.raises(_Crash):
        _reborn(writer).resume("e1", root)
    monkeypatch.undo()
    assert _reborn(writer).read_manifest("e1").state is ManifestState.JOURNALED
    plan = _reborn(writer).resume("e1", root)
    assert plan.action is ResumeAction.DONE
    assert _reborn(writer).read_manifest("e1").state is ManifestState.COMMITTED
    assert _reborn(writer).resume("e1", root).action is ResumeAction.DONE


def test_a_live_root_that_is_not_this_epochs_blocks_the_resume(world) -> None:  # type: ignore[no-untyped-def]
    memory, _, writer = world
    root = _make_store(memory)
    _quarantine(writer, root)
    writer.mark_journaled("e1")
    root.mkdir()  # a restored / foreign directory, no marker
    plan = writer.resume("e1", root)
    assert plan.action is ResumeAction.BLOCKED
    assert writer.read_manifest("e1").state is ManifestState.JOURNALED
    assert root.is_dir() and not any(root.iterdir())  # nothing was deleted or written


def test_a_torn_manifest_write_leaves_the_old_manifest_and_deletes_nothing(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, qdir, writer = world
    _quarantine(writer, _make_store(memory))

    def die(temporary: Path, final: Path) -> None:
        raise _Crash

    monkeypatch.setattr(dc, "_publish_file", die)
    with pytest.raises(_Crash):
        writer.mark_journaled("e1")
    monkeypatch.undo()
    assert _reborn(writer).read_manifest("e1").state is ManifestState.MOVED
    leftovers = [p.name for p in (qdir / "e1").iterdir() if ".tmp-" in p.name]
    assert len(leftovers) == 1  # the stray temporary stays: nothing is deleted
    # the next write works and does not collide with the leftover
    assert writer.mark_journaled("e1").state is ManifestState.JOURNALED


def test_no_exclusive_authority_no_quarantine(world) -> None:  # type: ignore[no-untyped-def]
    memory, qdir, _ = world
    root = _make_store(memory)
    before = tree_snapshot(root)
    held = [False]
    writer = QuarantineWriter(qdir, has_exclusive_authority=lambda: held[0])
    with pytest.raises(QuarantineRefused):
        _quarantine(writer, root)
    assert_trees_equal(before, tree_snapshot(root))
    assert not qdir.exists()
    held[0] = True
    assert _quarantine(writer, root).state is ManifestState.MOVED


def test_redo_rehashes_a_root_that_changed_after_the_pending_write(  # type: ignore[no-untyped-def]
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, _, writer = world
    root = _make_store(memory)

    def die(source: Path, target: Path) -> None:
        raise _Crash

    monkeypatch.setattr(dc, "_rename_directory", die)
    with pytest.raises(_Crash):
        _quarantine(writer, root)
    monkeypatch.undo()
    (root / "records" / "new.json").write_bytes(b"written after the crash")
    changed = tree_snapshot(root)
    plan = _reborn(writer).resume("e1", root)
    assert plan.action is ResumeAction.CONTINUE_JOURNAL
    moved = _reborn(writer).read_manifest("e1")
    assert moved.state is ManifestState.MOVED
    assert any(e[0] == "records/new.json" for e in moved.artifacts[0].entries)
    assert_trees_equal(changed, tree_snapshot(memory / "experience.quarantine-e1"))


def test_redo_refuses_a_manifest_that_does_not_match_the_root(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    memory, _, writer = world
    root = _make_store(memory)

    def die(source: Path, target: Path) -> None:
        raise _Crash

    monkeypatch.setattr(dc, "_rename_directory", die)
    with pytest.raises(_Crash):
        _quarantine(writer, root)
    monkeypatch.undo()
    other = _make_store(memory, "semantic")
    before = tree_snapshot(other)
    with pytest.raises(QuarantineRefused):  # the manifest names "experience"
        _reborn(writer).resume("e1", other)
    assert_trees_equal(before, tree_snapshot(other))
    assert root.is_dir()
    link = memory / "experience-link"
    link.symlink_to(root)
    with pytest.raises(QuarantineRefused):
        _reborn(writer).resume("e1", link)


@pytest.mark.parametrize(
    "facts, expected",
    [
        # PENDING
        (ResumeFacts(True, False, False, None), ResumeAction.REDO_MOVE),
        (ResumeFacts(False, True, True, None), ResumeAction.MARK_MOVED),
        (ResumeFacts(False, True, False, None), ResumeAction.BLOCKED),  # hashes differ
        (ResumeFacts(True, True, True, None), ResumeAction.BLOCKED),  # both present
        (ResumeFacts(False, False, False, None), ResumeAction.BLOCKED),  # both gone
    ],
)
def test_plan_resume_pending_rows(facts: ResumeFacts, expected: ResumeAction) -> None:
    manifest = _manifest(ManifestState.PENDING)
    assert plan_resume(manifest, facts).action is expected


def _manifest(state: ManifestState) -> Any:
    return dc.QuarantineManifest(
        epoch_id="e1",
        state=state,
        created_at="t",
        participant_id=EXPERIENCE,
        verdict="DIVERGED",
        reason="r",
        journal_lineage_id=None,
        journal_last_record_hash=None,
        runtime_instance_id=None,
        artifacts=(),
    )


@pytest.mark.parametrize(
    "state", [ManifestState.MOVED, ManifestState.JOURNALED, ManifestState.COMMITTED]
)
def test_plan_resume_later_states_need_the_quarantined_tree(
    state: ManifestState,
) -> None:
    gone = ResumeFacts(False, False, False, None)
    # a vanished quarantined tree is never "done" (nothing is guessed)
    assert plan_resume(_manifest(state), gone).action is ResumeAction.BLOCKED


def test_manifest_rejects_a_malformed_or_foreign_document(world) -> None:  # type: ignore[no-untyped-def]
    memory, qdir, writer = world
    _quarantine(writer, _make_store(memory))
    path = qdir / "e1" / "manifest.json"
    good = json.loads(path.read_text())
    for broken in (
        {**good, "schema": "other"},
        {**good, "state": "DONE"},
        {**good, "epoch_id": "../x"},
        {k: v for k, v in good.items() if k != "artifacts"},
        [],
    ):
        path.write_text(json.dumps(broken))
        with pytest.raises(ManifestError):
            writer.read_manifest("e1")
    path.write_text("{not json")
    with pytest.raises(ManifestError):
        writer.read_manifest("e1")
    path.write_text(json.dumps({**good, "epoch_id": "other"}))
    with pytest.raises(ManifestError):  # a manifest in the wrong directory
        writer.read_manifest("e1")


# ------------------------------------------------------- the kit itself


def test_access_spy_refuses_a_quarantined_path_and_passes_others(
    tmp_path: Path,
) -> None:
    quarantined = tmp_path / "x.quarantine-e1"
    quarantined.mkdir()
    (quarantined / "f").write_bytes(b"1")
    live = tmp_path / "live.json"
    live.write_bytes(b"2")
    with QuarantineAccessSpy() as refused:
        assert live.read_bytes() == b"2"
        with pytest.raises(QuarantineAccessViolation):
            open(quarantined / "f", "rb")
        with pytest.raises(QuarantineAccessViolation):
            os.open(quarantined / "f", os.O_RDONLY)
        with pytest.raises(QuarantineAccessViolation):
            os.listdir(quarantined)
        with pytest.raises(QuarantineAccessViolation):
            os.scandir(quarantined)
    assert len(refused) == 4
    assert (quarantined / "f").read_bytes() == b"1"  # patches are removed again


# ------------------------------------------------------------- health record


def test_status_is_quiet_while_inactive_or_plainly_serving() -> None:
    status = DegradedContinuationStatus()
    assert status.health_fields() == {}
    assert status.snapshot() == {"dc_state": "INACTIVE"}
    status.publish(DcState.SERVING)
    assert status.health_fields() == {}
    assert status.snapshot()["dc_state"] == "SERVING"


def test_status_reports_every_other_state_and_a_quarantined_domain() -> None:
    now = [datetime(2026, 11, 24, 1, tzinfo=timezone.utc)]
    status = DegradedContinuationStatus(clock=lambda: now[0])
    status.publish(DcState.REFUSED_RUNNING, reason="suzka.x.SomeError")
    first = status.health_fields()
    assert first == {
        "dc_state": "REFUSED_RUNNING",
        "dc_reason": "suzka.x.SomeError",
        "dc_since": "2026-11-24T01:00:00+00:00",
    }
    now[0] = datetime(2026, 11, 24, 2, tzinfo=timezone.utc)
    status.publish(DcState.REFUSED_RUNNING, reason="suzka.x.SomeError")
    assert status.health_fields()["dc_since"] == first["dc_since"]  # same state: kept
    status.publish(
        DcState.SERVING,
        reason="domain_quarantined",
        participants=[SEMANTIC, EXPERIENCE],
        epoch_id="e1",
    )
    fields = status.health_fields()
    assert fields["dc_state"] == "SERVING"
    assert fields["dc_participants"] == f"{EXPERIENCE},{SEMANTIC}"
    assert fields["dc_epoch_id"] == "e1"
    assert fields["dc_since"] == "2026-11-24T02:00:00+00:00"
    status.publish(DcState.RECONCILING, reason="x" * 1000)
    assert len(status.snapshot()["dc_reason"]) <= 200
    assert all(isinstance(v, str) for v in status.snapshot().values())
