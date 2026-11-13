"""IP-9 U4-4: SubjectHistory lineage head reader and projection (#376 §C, D2-5)."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.history_codec import GENESIS  # noqa: E402
from suzka.r16.history_participant import (  # noqa: E402
    JournalCommitProofSource,
    SubjectHistoryParticipant,
)
from suzka.r16.history_reader import (  # noqa: E402
    HeadBlockReason,
    HeadStatus,
    SubjectHistoryReader,
    project_head,
)
from suzka.r16.history_store import SubjectHistoryStore  # noqa: E402
from test_r16_ip9_history_adoption import PAGES, _adopt  # noqa: E402
from test_r16_ip9_subject_history import (  # noqa: E402
    OTHER_INCARNATION,
    _committed,
    _event,
    _prepared,
    _snapshot,
)


SCOPE = "goal-lineage-1"


class _World:
    def __init__(self, tmp_path: Path) -> None:
        (
            self.memory,
            self.state,
            self.journal,
            self.recovery,
            self.store,
            self.coordinator,
            _baseline,
        ) = _adopt(tmp_path)
        assert self.coordinator.history_incarnation is not None
        self.incarnation = self.coordinator.history_incarnation()
        self.sequence = 0

    def reader(self, incarnation: str | None = None) -> SubjectHistoryReader:
        if incarnation is None:
            return SubjectHistoryReader(self.journal, self.store)
        return SubjectHistoryReader(
            self.journal, self.store, lambda _inspection: (incarnation, 0)
        )

    def decision(self, name: str, *, intent: str, parent=None):
        self.sequence += 1
        event = _event(name, self.sequence)
        predecessor = (GENESIS, None) if parent is None else (
            parent.decision_id,
            parent.content_digest,
        )
        decision = _committed(
            event, incarnation=self.incarnation, intent=intent, predecessor=predecessor
        )
        return event, decision

    def participant(self, decision) -> SubjectHistoryParticipant:
        return SubjectHistoryParticipant(
            self.store,
            self.incarnation,
            decision,
            JournalCommitProofSource(self.journal, self.recovery),
        )

    def commit(self, name: str, *, intent: str, parent=None, complete: bool = True):
        event, decision = self.decision(name, intent=intent, parent=parent)
        coordinator = _prepared(self.journal, event, self.participant(decision))
        evidence = self.recovery.commit_internal_candidate(
            event, self.state.load(), _snapshot(self.sequence)
        )
        coordinator.finalize_event(event, evidence)
        if complete:
            self.recovery.complete_committed_event(event, evidence)
        return event, decision


def test_empty_then_single_then_chained_head(tmp_path: Path) -> None:
    world = _World(tmp_path)
    assert world.reader().head(SCOPE).status is HeadStatus.EMPTY

    _event1, root = world.commit("root", intent="a" * 64)
    head = world.reader().head(SCOPE)
    assert (head.status, head.depth, head.decision) == (HeadStatus.ADOPTED, 1, root)

    _event2, child = world.commit("child", intent="b" * 64, parent=root)
    head = world.reader().head(SCOPE)
    assert (head.status, head.depth, head.decision) == (HeadStatus.ADOPTED, 2, child)
    assert world.reader().head("another-lineage").status is HeadStatus.EMPTY


def test_finalized_row_without_event_terminal_is_not_adopted(tmp_path: Path) -> None:
    world = _World(tmp_path)
    world.commit("uncompleted", intent="a" * 64, complete=False)
    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (
        HeadStatus.BLOCKED,
        HeadBlockReason.MISSING_JOURNAL_PROOF,
    )
    assert head.decision is None


def test_unresolved_operations_block_the_whole_scope(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _event1, root = world.commit("root", intent="a" * 64)
    # A newer PREPARED child with an open R07 transaction.
    event, child = world.decision("pending-child", intent="b" * 64, parent=root)
    _prepared(world.journal, event, world.participant(child))
    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (
        HeadStatus.BLOCKED,
        HeadBlockReason.UNRESOLVED_OPERATION,
    )

    # SQLite FINALIZE before the R07 terminal is still unresolved.
    evidence = world.recovery.commit_internal_candidate(
        event, world.state.load(), _snapshot(world.sequence)
    )
    binding_participant = world.participant(child)
    from test_r16_ip9_subject_history import _binding

    binding_participant.finalize(_binding(event, child))
    assert world.reader().head(SCOPE).reason is HeadBlockReason.UNRESOLVED_OPERATION
    del evidence


def test_store_rollback_under_a_terminal_transaction_diverges(tmp_path: Path) -> None:
    world = _World(tmp_path)
    event, decision = world.decision("rolled-back-store", intent="a" * 64)
    coordinator = _prepared(world.journal, event, world.participant(decision))
    backup = tmp_path / "prepared-copy.sqlite3"
    shutil.copyfile(world.store.path, backup)
    evidence = world.recovery.commit_internal_candidate(
        event, world.state.load(), _snapshot(world.sequence)
    )
    coordinator.finalize_event(event, evidence)
    world.recovery.complete_committed_event(event, evidence)
    path = world.store.path
    world.store.close()
    shutil.copyfile(backup, path)
    world.store = SubjectHistoryStore.open(path, max_page_count=PAGES)

    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (HeadStatus.BLOCKED, HeadBlockReason.DIVERGED)


def test_lineage_gap_and_other_incarnations_are_not_sources(tmp_path: Path) -> None:
    world = _World(tmp_path)
    # A parent that was never issued (not even accepted by the Journal).
    orphan_parent = _committed(
        _event("never-committed", 999), incarnation=world.incarnation, intent="a" * 64
    )
    world.commit("orphan", intent="b" * 64, parent=orphan_parent)
    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (HeadStatus.BLOCKED, HeadBlockReason.LINEAGE_GAP)

    # Rows of the current incarnation are never read as another incarnation's.
    assert world.reader(OTHER_INCARNATION).head(SCOPE).reason is HeadBlockReason.DIVERGED

    def unavailable(_inspection) -> tuple[str, int]:
        raise RuntimeError("gate clear open")

    blocked = SubjectHistoryReader(world.journal, world.store, unavailable).head(SCOPE)
    assert blocked.reason is HeadBlockReason.INCARNATION_UNAVAILABLE


def test_aborted_operations_are_ignored(tmp_path: Path) -> None:
    world = _World(tmp_path)
    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator

    event, decision = world.decision("aborted", intent="a" * 64)
    _prepared(world.journal, event, world.participant(decision))
    StartupReconciliationCoordinator(
        world.journal, world.recovery, world.memory, history_store=world.store
    ).reconcile_open_transactions()
    assert world.journal.inspect().aborted_transactions
    assert world.reader().head(SCOPE).status is HeadStatus.EMPTY
    inspection = world.journal.inspect()
    world.journal.append_failed(
        event, inspection.snapshot_sequence, inspection.snapshot_hash or ""
    )

    # The aborted intent slot is free again for a real Decision.
    _event2, real = world.commit("after-abort", intent="a" * 64)
    assert world.reader().head(SCOPE).decision == real


def test_projection_is_public_safe(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _event1, root = world.commit("projected", intent="a" * 64)
    projection = project_head(world.reader().head(SCOPE))
    value = projection.canonical_value()
    assert value["status"] == "ADOPTED"
    assert value["decision_id"] == root.decision_id
    assert value["policy_row"] == "4" and value["selected_kind"] == "plan"
    assert value["source_refs"] == [
        [item.role.value, item.kind, item.ref_id, item.revision, item.digest]
        for item in root.source_refs
    ]
    rendered = repr(value)
    assert "policy_input" not in rendered and "'row'" not in rendered
    assert "assess" not in rendered  # plan steps are never projected

    blocked = project_head(
        SubjectHistoryReader(
            world.journal, world.store, lambda _inspection: (OTHER_INCARNATION, 0)
        ).head("x")
    )
    assert blocked.decision_id is None and blocked.source_refs == ()


def test_startup_roll_forward_terminal_is_adopted(tmp_path: Path) -> None:
    """Review P1-1: a crash before the event COMPLETED record still adopts."""

    world = _World(tmp_path)
    event, decision = world.decision("rolled-forward", intent="a" * 64)
    coordinator = _prepared(world.journal, event, world.participant(decision))
    evidence = world.recovery.commit_internal_candidate(
        event, world.state.load(), _snapshot(world.sequence)
    )
    coordinator.finalize_event(event, evidence)
    # Crash before complete_committed_event; startup classifies the event.
    world.recovery.prepare_startup()
    terminals = [
        record.lifecycle.value
        for record in world.journal.inspect().records
        if record.event_id == event.event_id
    ]
    assert "recovery_classified" in terminals and "completed" not in terminals
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision) == (HeadStatus.ADOPTED, decision)


def test_store_restored_to_an_older_copy_blocks_every_scope(tmp_path: Path) -> None:
    """Review P1-2(a): missing rows are found from the Journal side."""

    world = _World(tmp_path)
    _event1, root = world.commit("root", intent="a" * 64)
    backup = tmp_path / "older.sqlite3"
    shutil.copyfile(world.store.path, backup)
    world.commit("child", intent="b" * 64, parent=root)
    path = world.store.path
    world.store.close()
    shutil.copyfile(backup, path)
    world.store = SubjectHistoryStore.open(path, max_page_count=PAGES)

    for scope in (SCOPE, "another-lineage"):
        head = world.reader().head(scope)
        assert (head.status, head.reason) == (HeadStatus.BLOCKED, HeadBlockReason.DIVERGED)


def test_open_transaction_without_a_row_blocks(tmp_path: Path) -> None:
    """Review P1-2(b): TRANSACTION_PREPARED before the SQLite PREPARE."""

    world = _World(tmp_path)
    world.commit("root", intent="a" * 64)
    event, decision = world.decision("no-row", intent="b" * 64)

    class Refusing(SubjectHistoryParticipant):
        def prepare(self, binding) -> None:  # crash before the SQLite row
            raise KeyboardInterrupt

    participant = Refusing(
        world.store,
        world.incarnation,
        decision,
        JournalCommitProofSource(world.journal, world.recovery),
    )
    with pytest.raises(KeyboardInterrupt):
        _prepared(world.journal, event, participant)
    assert world.journal.inspect().open_transactions
    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (
        HeadStatus.BLOCKED,
        HeadBlockReason.UNRESOLVED_OPERATION,
    )
