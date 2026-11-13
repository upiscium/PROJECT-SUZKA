"""IP-9 U4-5: SubjectHistory crash matrix (#376 §C, D1-1, D1-4, D2-5).

Each case stops the ordinary R07 -> SQLite -> R06 -> SQLite FINALIZE -> R07
sequence at one boundary, restarts the startup reconciliation, and checks
both the R07 outcome and what the head reader adopts.  No model calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.history_participant import SubjectHistoryParticipant  # noqa: E402
from suzka.runtime.transaction_coordinator import ParticipantDivergedError  # noqa: E402
from suzka.r16.history_reader import HeadBlockReason, HeadStatus  # noqa: E402
from suzka.r16.history_store import (  # noqa: E402
    HistoryPhase,
    HistoryStoreConflict,
    SubjectHistoryStore,
)
from suzka.runtime.event_journal import (  # noqa: E402
    AbortOutcome,
    ParticipantOutcome,
)
from suzka.runtime.startup_reconciliation import (  # noqa: E402
    StartupReconciliationCoordinator,
)
from suzka.runtime.state_recovery import StateRecoveryCoordinator  # noqa: E402
from test_r16_ip9_history_reader import SCOPE, _World  # noqa: E402
from test_r16_ip9_subject_history import (  # noqa: E402
    _binding,
    _prepared,
    _snapshot,
)


def _startup(world: _World) -> tuple[bool, str | None]:
    return StartupReconciliationCoordinator(
        world.journal, world.recovery, world.memory, history_store=world.store
    ).reconcile_open_transactions()


def _row(world: _World, transaction_id: str):
    return world.store.operation(world.incarnation, transaction_id)


def test_crash_after_r07_prepared_before_sqlite_prepare(tmp_path: Path) -> None:
    """No SQLite row: startup proves PRE_INTERNAL and writes a tombstone."""

    world = _World(tmp_path)
    event, decision = world.decision("no-sqlite-row", intent="a" * 64)

    class Crashing(SubjectHistoryParticipant):
        def prepare(self, binding) -> None:
            raise KeyboardInterrupt

    participant = Crashing(
        world.store, world.incarnation, decision, world.participant(decision).proof_source
    )
    with pytest.raises(KeyboardInterrupt):
        _prepared(world.journal, event, participant)
    assert world.reader().head(SCOPE).reason is HeadBlockReason.UNRESOLVED_OPERATION

    assert _startup(world) == (True, None)
    aborted = world.journal.inspect().aborted_transactions[0]
    assert aborted.abort_outcomes == (("subject.history", AbortOutcome.ALREADY_ABSENT),)
    row = _row(world, aborted.transaction_id)
    assert row is not None and row.phase is HistoryPhase.ABORTED
    assert row.pending_bytes is None  # binding-only tombstone
    # The tombstone is proven by the R07 abort and never blocks the scope.
    world.journal.append_failed(
        event,
        world.journal.inspect().snapshot_sequence,
        world.journal.inspect().snapshot_hash or "",
    )
    assert world.reader().head(SCOPE).status is HeadStatus.EMPTY
    # The same transaction can never be prepared later.
    with pytest.raises(ParticipantDivergedError):
        world.participant(decision).prepare(_binding(event, decision))


def test_crash_inside_the_sqlite_finalize_rolls_back_then_rolls_forward(
    tmp_path: Path,
) -> None:
    """A FINALIZE interrupted mid-transaction leaves only PREPARED."""

    world = _World(tmp_path)
    event, decision = world.decision("mid-finalize", intent="a" * 64)
    participant = world.participant(decision)
    _prepared(world.journal, event, participant)
    world.recovery.commit_internal_candidate(
        event, world.state.load(), _snapshot(world.sequence)
    )
    binding = _binding(event, decision)
    original = world.store._connection
    raw = sqlite3.connect(world.store.path)
    raw.execute(
        "CREATE TEMP TRIGGER crash BEFORE INSERT ON history_postcommit_receipts "
        "BEGIN SELECT RAISE(ABORT, 'simulated crash'); END"
    )
    world.store._connection = raw
    try:
        with pytest.raises(ParticipantDivergedError) as raised:
            participant.finalize(binding)
        cause = raised.value.__cause__
        assert isinstance(cause, HistoryStoreConflict) or "refused" in str(raised.value)
    finally:
        world.store._connection = original
        raw.close()
    row = _row(world, binding.transaction_id)
    assert row is not None and row.phase is HistoryPhase.PREPARED
    assert world.store.finalized_decision(world.incarnation, binding.transaction_id) is None

    assert _startup(world) == (True, None)
    completed = world.journal.inspect().completed_transactions[0]
    assert completed.participant_outcomes == (
        ("subject.history", ParticipantOutcome.FINALIZED),
    )


def test_crash_after_participant_finalized_before_transaction_completed(
    tmp_path: Path,
) -> None:
    world = _World(tmp_path)
    event, decision = world.decision("finalized-recorded", intent="a" * 64)
    _prepared(world.journal, event, world.participant(decision))
    evidence = world.recovery.commit_internal_candidate(
        event, world.state.load(), _snapshot(world.sequence)
    )
    binding = _binding(event, decision)
    assert world.participant(decision).finalize(binding) is ParticipantOutcome.FINALIZED
    world.journal.append_participant_finalized(
        event,
        binding.transaction_id,
        "subject.history",
        binding.operation_digest,
        ParticipantOutcome.FINALIZED,
    )
    # Not adopted yet: the transaction and event are still open.
    assert world.reader().head(SCOPE).reason is HeadBlockReason.UNRESOLVED_OPERATION

    assert _startup(world) == (True, None)
    world.recovery.complete_committed_event(event, evidence)
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision) == (HeadStatus.ADOPTED, decision)


def test_crash_after_ambiguous_state_stays_blocked(tmp_path: Path) -> None:
    """Ambiguous R06 evidence never aborts or finalizes the operation."""

    from suzka.runtime.startup_reconciliation import StartupReconciliationError

    world = _World(tmp_path)
    event, decision = world.decision("ambiguous", intent="a" * 64)
    _prepared(world.journal, event, world.participant(decision))
    world.recovery.commit_internal_candidate(
        event, world.state.load(), _snapshot(world.sequence)
    )
    world.state.save(_snapshot(world.sequence, 0.9))
    with pytest.raises(StartupReconciliationError, match="ambiguous"):
        _startup(world)
    row = _row(world, world.journal.inspect().open_transactions[0].transaction_id)
    assert row is not None and row.phase is HistoryPhase.PREPARED
    assert world.reader().head(SCOPE).reason is HeadBlockReason.UNRESOLVED_OPERATION


def test_tombstone_and_prepare_race_has_one_winner(tmp_path: Path) -> None:
    """Review P2-7: both read "no row", then race; exactly one write wins."""

    import suzka.r16.history_participant as participant_module

    world = _World(tmp_path)
    event, decision = world.decision("race", intent="a" * 64)
    binding = _binding(event, decision)
    second = SubjectHistoryStore.open(
        world.store.path, max_page_count=world.store._max_page_count
    )
    barrier = threading.Barrier(2)
    real_load = participant_module._load

    def load_then_wait(store, incarnation, transaction_id):
        row = real_load(store, incarnation, transaction_id)
        barrier.wait(timeout=10)  # both threads have seen "no row"
        return row

    outcomes: list[object] = []

    def prepare() -> None:
        try:
            world.participant(decision).prepare(binding)
            outcomes.append("prepared")
        except ParticipantDivergedError as error:
            outcomes.append(error)

    def tombstone() -> None:
        try:
            outcomes.append(
                SubjectHistoryParticipant.abort_pending(second, world.incarnation, binding)
            )
        except ParticipantDivergedError as error:
            outcomes.append(error)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(participant_module, "_load", load_then_wait)
        threads = [threading.Thread(target=prepare), threading.Thread(target=tombstone)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    second.close()
    winners = [item for item in outcomes if not isinstance(item, Exception)]
    losers = [item for item in outcomes if isinstance(item, ParticipantDivergedError)]
    assert len(winners) == 1 and len(losers) == 1
    row = _row(world, binding.transaction_id)
    assert row is not None and len(world.store.operations(world.incarnation)) == 1
    if winners == ["prepared"]:
        assert row.phase is HistoryPhase.PREPARED
    else:
        assert winners == [AbortOutcome.ALREADY_ABSENT]
        assert row.phase is HistoryPhase.ABORTED and row.pending_bytes is None


def test_interrupted_create_is_completed_on_open(tmp_path: Path) -> None:
    """Review P2-4: crash between link() and unlink(staging) during create."""

    directory = tmp_path / "history"
    directory.mkdir(mode=0o700)
    path = directory / "subject_history.sqlite3"
    store = SubjectHistoryStore.create(
        path, subject_id="suzka", journal_lineage_id="lineage-1", max_page_count=4096
    )
    store.close()
    staging = directory / f".{path.name}.{'0' * 32}.creating"
    os.link(path, staging)
    assert path.stat().st_nlink == 2
    reopened = SubjectHistoryStore.open(path, max_page_count=4096)
    assert path.stat().st_nlink == 1 and not staging.exists()
    reopened.close()

    # Any other second link is refused, not "repaired".
    other = directory / "unrelated-link"
    os.link(path, other)
    from suzka.r16.history_store import HistoryStoreCorrupt

    with pytest.raises(HistoryStoreCorrupt):
        SubjectHistoryStore.open(path, max_page_count=4096)


def test_true_rollback_starts_a_new_incarnation(tmp_path: Path) -> None:
    """D1-4: after a proven TRUE_ROLLBACK the old chain is never a current source."""

    world = _World(tmp_path)
    world.recovery.publish_boot_anchor(world.recovery.prepare_startup())
    _event1, old_root = world.commit("before-rollback", intent="a" * 64)
    assert world.reader().head(SCOPE).decision == old_root
    old_incarnation = world.incarnation

    # Corrupt the committed WAL transition and drop the published state, as
    # the existing true-rollback startup tests do.
    wal = world.recovery.wal
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    world.state.path.unlink()
    rolled_back = StateRecoveryCoordinator(
        world.state, world.journal, wal
    ).prepare_startup()
    assert rolled_back.true_rollback_performed
    result = StartupReconciliationCoordinator(
        world.journal, world.recovery, world.memory, history_store=world.store
    ).reconcile_recovery_gate(rolled_back)
    assert result.participants_consistent
    assert world.journal.inspect().terminal_gate_clear is not None

    from suzka.r16.history_adoption import resolve_incarnation

    new_incarnation, horizon = resolve_incarnation(world.journal.inspect(), world.store)
    assert new_incarnation != old_incarnation and horizon >= world.sequence
    # The old incarnation's rows stay as immutable history only.
    assert len(world.store.operations(old_incarnation)) == 1
    assert world.reader().head(SCOPE).status is HeadStatus.EMPTY

    # A new GENESIS Decision in the new incarnation is adopted normally.
    world.incarnation = new_incarnation
    world.sequence = world.journal.inspect().processing_high_water
    _event2, new_root = world.commit("after-rollback", intent="a" * 64)
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, new_root, 1)
    assert new_root.decision_id != old_root.decision_id
