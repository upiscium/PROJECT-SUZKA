"""IP-14b-2: participant / reader / C10 / slot generalization for schema v4.

Design: IP-14b D1 r2 FINAL §4-3 / §6-1 / §7 (IP-14b-2 rows). There is still no
writer (IP-14b-3, IP-20 / IP-21): operations are built from the real R16 /
R19 values (the IP-14b-1 fixtures) and driven through the R07 participant,
the R07 world and startup reconciliation directly. Every value here is a
FIXTURE ONLY.
"""

from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.history_codec import GENESIS, PolicyRow  # noqa: E402
from suzka.r16.history_participant import (  # noqa: E402
    SubjectHistoryParticipant,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import (  # noqa: E402
    HeadBlockReason,
    HeadStatus,
    SubjectHistoryReader,
)
from suzka.r16.history_records import (  # noqa: E402
    DECISION_OPERATION_KINDS,
    PLAN_HOLD_GATED_KINDS,
    R19_GATED_KINDS,
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_store import (  # noqa: E402
    HistoryPhase,
    HistoryStoreRecordsDisabled,
    SubjectHistoryStore,
    decision_slot_occupied,
)
from suzka.r19.contracts import AttemptOutcome, TerminalReason  # noqa: E402
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType  # noqa: E402
from suzka.runtime.event_journal import (  # noqa: E402
    AbortOutcome,
    ParticipantOutcome,
    StartupParticipantOutcome,
)
from suzka.runtime.transaction_coordinator import (  # noqa: E402
    ParticipantDivergedError,
    ParticipantUnavailableError,
)
from test_r16_ip14a2_participant_reader import NOW, _binding, _Proof, _RecordWorld  # noqa: E402
from test_r16_ip14a_history_v2 import _all_rows, _apply, _prepare  # noqa: E402
from test_r16_ip14b1_history_v4 import (  # noqa: E402
    INC,
    Cycle,
    _one_of_each,
    _plan_hold,
    _plan_hold_op,
    _store,
)
from test_r16_ip9_history_reader import SCOPE  # noqa: E402
from test_r16_ip9_subject_history import _committed, _event, _prepared, _snapshot  # noqa: E402

K = HistoryOperationKind
R = RecordKind
NEW_KINDS = sorted(R19_GATED_KINDS | PLAN_HOLD_GATED_KINDS, key=lambda kind: kind.value)


def _mutation(operation: HistoryRecordOperationV1) -> HistoryRecordOperationV1:
    """The R07 transaction kind the participant binding carries."""
    return replace(operation, transaction_kind="event_mutation")


class _WorldCycle(Cycle):
    """A Cycle of the R07 world's incarnation, bound as event mutations, whose
    events continue the world's contiguous processing sequence."""

    def __init__(self, world: _RecordWorld, name: str = "goal-1") -> None:
        super().__init__(name, incarnation=world.incarnation)
        self.seq = world.sequence
        self.wake = replace(self.wake, horizon_sequence=max(world.sequence, 1), cause_sequence=max(world.sequence, 1))

    def op(self, kind: K, record_kind: R, record: object, event: tuple[str, int] | None = None) -> HistoryRecordOperationV1:
        return _mutation(super().op(kind, record_kind, record, event))


_NEEDS_CLAIM_2 = frozenset({K.R16_CYCLE_DECISION, K.R16_CYCLE_RECEIPT, K.R19_WAKE_ATTEMPT_OUTCOME})


def _ready(store: SubjectHistoryStore, kind: K) -> HistoryRecordOperationV1:
    """IP-14b-1's PREPARE-able operation of ``kind``, made FINALIZE-able too:
    the closures and the outcome name the second claim, which is applied."""

    operations = _one_of_each(store)
    if kind in _NEEDS_CLAIM_2:
        _apply(store, operations[K.R19_WAKE_CLAIM])
    return _mutation(operations[kind])


def _world(tmp_path: Path) -> _RecordWorld:
    world = _RecordWorld(tmp_path)
    world.store._r19_enabled = True
    world.store._plan_hold_receipts = True
    return world


def _plan_hold_at(incarnation: str, sequence: int) -> Any:
    """``_plan_hold`` of IP-14b-1, at another event sequence."""

    from suzka.r16.contracts import ReasonCode
    from suzka.r16.cycle_contracts import PlanHoldReceiptV1
    from suzka.r16.history_codec import PolicyKind
    from suzka.r16.successor_policy import Interpretation, InterpretationStage, SuccessorRow
    from test_r16_ip14b1_history_v4 import _h, _uuid
    from test_r16_ip8b_successor_policy import _active

    event_id = _uuid("plan-at")
    sealed = replace(
        _active(), event_id=event_id, event_sequence=sequence, now_utc="2026-11-16T03:00:00Z",
        interpretation=Interpretation.CONTRADICTORY, interpretation_stage=InterpretationStage.S2,
    )
    return PlanHoldReceiptV1(
        incarnation_id=incarnation, lineage_scope=SCOPE, review_intent_key=_h("plan-rik-at"),
        predecessor_key=GENESIS, predecessor_digest=None, event_id=event_id, event_sequence=sequence,
        policy_kind=PolicyKind.SUCCESSOR_POLICY_V2, policy_row=SuccessorRow.ROW_2B,
        hold_reason=ReasonCode.CONFLICTING_EVIDENCE, policy_input_bytes=sealed.canonical_bytes(),
        prepare_checked_at_utc="2026-11-16T03:00:00Z",
    )


# ------------------------------------------------------------ participant


@pytest.mark.parametrize("kind", NEW_KINDS, ids=lambda kind: kind.value)
def test_the_record_participant_covers_every_new_kind(tmp_path: Path, kind: K) -> None:
    """PREPARE (idempotent), FINALIZE, ALREADY_CONSISTENT, rebuild from the
    pending bytes, and startup inspection, for each v4 kind (D1 r2 §4-3)."""

    store = _store(tmp_path)
    operation = _ready(store, kind)
    participant = SubjectHistoryRecordParticipant(store, INC, operation, _Proof())
    binding = _binding(operation)
    participant.prepare(binding)
    participant.prepare(binding)
    assert store.live_keys(operation.reserved_keys) == frozenset(operation.reserved_keys)
    rebuilt = SubjectHistoryParticipant.from_pending(store, INC, _Proof(), binding)
    assert type(rebuilt) is SubjectHistoryRecordParticipant and rebuilt.operation == operation
    with pytest.raises(ParticipantUnavailableError, match="roll-forward"):
        participant.inspect_reconciliation(binding)
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    assert participant.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert participant.inspect_reconciliation(binding) is StartupParticipantOutcome.VERIFIED_CONSISTENT
    assert participant.reconcile(binding) is StartupParticipantOutcome.VERIFIED_CONSISTENT
    assert store.record_rows_match(INC, binding.transaction_id, operation)
    with pytest.raises(ParticipantDivergedError):
        participant.abort(binding)  # a FINALIZED operation never aborts
    with pytest.raises(ParticipantDivergedError):
        participant.prepare(binding)  # the transaction was used


@pytest.mark.parametrize("kind", NEW_KINDS, ids=lambda kind: kind.value)
def test_abort_tombstone_and_roll_forward_for_every_new_kind(tmp_path: Path, kind: K) -> None:
    def ready(name: str) -> tuple[SubjectHistoryStore, HistoryRecordOperationV1, SubjectHistoryRecordParticipant]:
        store = _store(tmp_path / name)
        operation = _ready(store, kind)
        return store, operation, SubjectHistoryRecordParticipant(store, INC, operation, _Proof())

    # Nothing reached SQLite before the abort: a binding-only tombstone, and
    # the transaction can never be prepared later.
    store, operation, participant = ready("tombstone")
    binding = _binding(operation)
    assert participant.abort(binding) is AbortOutcome.ALREADY_ABSENT
    row = store.operation(INC, binding.transaction_id)
    assert row is not None and (row.operation_kind, row.phase) == (K.R07_BINDING_ONLY_ABORT, HistoryPhase.ABORTED)
    with pytest.raises(ParticipantDivergedError):
        participant.prepare(binding)
    assert store.live_keys(operation.reserved_keys) == frozenset()
    # PREPARED, then aborted before the internal commit: the keys are released.
    store, operation, participant = ready("abort")
    participant.prepare(binding)
    assert participant.abort(binding) is AbortOutcome.ABORTED
    assert participant.abort(binding) is AbortOutcome.ALREADY_ABSENT
    assert store.live_keys(operation.reserved_keys) == frozenset()
    assert store.prepare_collisions(operation).clear
    with pytest.raises(ParticipantDivergedError):
        participant.finalize(binding)
    # PREPARED, then a crash past the internal commit: rolled forward.
    store, operation, participant = ready("forward")
    participant.prepare(binding)
    assert participant.reconcile(binding) is StartupParticipantOutcome.ROLLED_FORWARD
    assert store.record_rows_match(INC, binding.transaction_id, operation)


def test_the_participant_refuses_a_mismatched_binding_or_envelope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    operations = _one_of_each(store)
    _apply(store, operations[K.R19_WAKE_CLAIM])
    operation = _mutation(operations[K.R16_CYCLE_DECISION])
    participant = SubjectHistoryRecordParticipant(store, INC, operation, _Proof())
    binding = _binding(operation)
    for bad in (
        replace(binding, processing_sequence=binding.processing_sequence + 1),
        replace(binding, event_id="00000000-0000-4000-8000-000000000001"),
        replace(binding, operation_digest="0" * 64),
    ):
        with pytest.raises(ParticipantDivergedError):
            participant.prepare(bad)
    with pytest.raises(ValueError, match="incarnation"):
        SubjectHistoryRecordParticipant(store, "9" * 64, operation, _Proof())
    # A stored row whose kind does not match its envelope is not rebuilt.
    participant.prepare(binding)
    connection = sqlite3.connect(store.path, isolation_level=None)
    connection.execute("DROP TRIGGER history_operations_transition")
    connection.execute(
        "UPDATE history_operations SET operation_kind = 'R16_DECISION_ISSUE' WHERE transaction_id = ?",
        (binding.transaction_id,),
    )
    connection.close()
    with pytest.raises(ParticipantDivergedError):
        SubjectHistoryParticipant.from_pending(store, INC, _Proof(), binding)


def test_record_rows_match_covers_the_cycle_decision_rows(tmp_path: Path) -> None:
    """D1 r2 §4-3: Decision + sources + resolution for R16_CYCLE_DECISION."""

    for table, column in (
        (None, None),
        ("history_source_refs", "digest"),
        ("history_decisions", "policy_input_digest"),
        ("logical_cycle_resolutions", "record_digest"),
    ):
        store = _store(tmp_path / str(table))
        operation = _ready(store, K.R16_CYCLE_DECISION)
        transaction_id = _apply(store, operation)
        if table is None:
            assert store.record_rows_match(INC, transaction_id, operation)
            # The same rows checked against another record's envelope.
            assert not store.record_rows_match(INC, transaction_id, Cycle("goal-other").materialize())
            continue
        connection = sqlite3.connect(store.path, isolation_level=None)
        for (trigger,) in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'trigger' AND tbl_name = ? AND name LIKE '%update%'", (table,)).fetchall():
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(f"UPDATE {table} SET {column} = ?", ("0" * 64,))
        connection.close()
        assert not store.record_rows_match(INC, transaction_id, operation), table


# --------------------------------------------------------- raw pre-checks


def test_prepare_collisions_read_raw_prepared_and_finalized_rows(tmp_path: Path) -> None:
    """D1 r2 §2-6: keys and Decision slots are read from raw store state, so a
    PREPARED or FINALIZED-not-terminal holder is seen before PREPARE."""

    from suzka.r16.history_codec import history_operation_digest

    store = _store(tmp_path)
    operations = _one_of_each(store)
    for operation in operations.values():
        assert store.prepare_collisions(operation).clear, operation.operation_kind
    # Keys: a PREPARED holder of the cycle key collides for every closure.
    hold = operations[K.R16_CYCLE_RECEIPT]
    transaction_id = _prepare(store, hold)
    for kind in (K.R16_CYCLE_DECISION, K.R19_WAKE_TERMINAL, K.R19_WAKE_ATTEMPT_OUTCOME):
        collisions = store.prepare_collisions(operations[kind])
        assert not collisions.clear and collisions.live_keys and not collisions.decision_slot_occupied, kind
    store.mark_aborted(INC, transaction_id)
    assert store.prepare_collisions(operations[K.R16_CYCLE_DECISION]).clear
    # Slots: a PREPARED PLAN_PREVIEW Decision on the same predecessor slot.
    decision = operations[K.R16_CYCLE_DECISION].cycle_decision
    assert decision is not None
    event = _event("plan-preview", 90)
    preview = _committed(event, incarnation=INC, intent="7" * 64)
    preview_tx = "00000000-0000-4000-8000-0000000000aa"
    store.insert_prepared(transaction_id=preview_tx, event_id=event.event_id, processing_sequence=90, transaction_kind="event_mutation", operation_digest=history_operation_digest(preview), decision=preview)
    collisions = store.prepare_collisions(operations[K.R16_CYCLE_DECISION])
    assert collisions.decision_slot_occupied and not collisions.live_keys
    store.mark_aborted(INC, preview_tx)
    # ... and a FINALIZED one on the same review-intent slot (not yet terminal
    # in R07 is invisible to the adopted path, but not to this check).
    same_intent = _committed(_event("plan-preview-2", 91), incarnation=INC, intent=decision.review_intent_key, predecessor=("decision-" + "f" * 64, "f" * 64))
    finalized_tx = "00000000-0000-4000-8000-0000000000ab"
    store.insert_prepared(transaction_id=finalized_tx, event_id=same_intent.issuer_event_id, processing_sequence=91, transaction_kind="event_mutation", operation_digest=history_operation_digest(same_intent), decision=same_intent)
    from test_r16_ip14a_history_v2 import _receipt

    store.finalize(transaction_id=finalized_tx, decision=same_intent, receipt=_receipt(finalized_tx, same_intent.issuer_event_id, 91))
    assert store.prepare_collisions(operations[K.R16_CYCLE_DECISION]).decision_slot_occupied


def test_decision_slot_occupied_sees_both_decision_kinds_and_skips_aborted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    operation = _one_of_each(store)[K.R16_CYCLE_DECISION]
    decision = operation.cycle_decision
    assert decision is not None and DECISION_OPERATION_KINDS == {K.R16_DECISION_ISSUE, K.R16_CYCLE_DECISION}
    slots = {"lineage_scope": decision.lineage_scope, "predecessor_key": decision.predecessor_key}
    transaction_id = _prepare(store, operation)
    rows = store.operations(INC)
    assert decision_slot_occupied(rows, review_intent_key="1" * 64, **slots)  # predecessor slot
    assert decision_slot_occupied(rows, review_intent_key=decision.review_intent_key, lineage_scope="x", predecessor_key=GENESIS)
    assert not decision_slot_occupied(rows, review_intent_key="1" * 64, lineage_scope="other-scope", predecessor_key=GENESIS)
    store.mark_aborted(INC, transaction_id)
    assert not decision_slot_occupied(store.operations(INC), review_intent_key=decision.review_intent_key, **slots)


def test_the_issuer_slot_check_sees_a_prepared_cycle_decision(tmp_path: Path) -> None:
    """D1 r2 §4-3: otherwise a PLAN_PREVIEW would PREPARE into a held slot,
    and the conflict would fail-stop the runtime (§2-6)."""

    from suzka.r16.decision_issuer import _DecisionIssuer, _IssuerHold
    from suzka.r16.outcomes import R16OwnerReason

    store = _store(tmp_path)
    operation = _one_of_each(store)[K.R16_CYCLE_DECISION]
    decision = operation.cycle_decision
    assert decision is not None
    preview = _committed(_event("preview", 95), incarnation=INC, intent="8" * 64, predecessor=(decision.predecessor_key, decision.predecessor_digest))

    class _Issuer:
        _store = store

    _DecisionIssuer._check_slots(_Issuer(), preview, INC)  # type: ignore[arg-type]  # free
    transaction_id = _prepare(store, operation)
    with pytest.raises(_IssuerHold) as held:
        _DecisionIssuer._check_slots(_Issuer(), preview, INC)  # type: ignore[arg-type]
    assert held.value.reason is R16OwnerReason.DECISION_SLOT_OCCUPIED
    store.mark_aborted(INC, transaction_id)
    _DecisionIssuer._check_slots(_Issuer(), preview, INC)  # type: ignore[arg-type]


# ------------------------------------------------------------- the reader


def _cycle_to_decision(world: _RecordWorld, root: Any, *, name: str = "goal-1") -> tuple[_WorldCycle, list[HistoryRecordOperationV1]]:
    cycle = _WorldCycle(world, name)
    operations = [cycle.materialize(), cycle.claim(1)]
    for operation in operations:
        world.commit_record(operation)
    closure = cycle.decision(predecessor=(root.decision_id, root.content_digest), row=PolicyRow.ROW_5)
    operations.append(closure)
    return cycle, operations


def test_a_cycle_decision_moves_the_head_and_its_resolution_is_adopted(tmp_path: Path) -> None:
    """S0 step 8: the Wake-origin successor (row 5) is a chain node, and a
    later PLAN_PREVIEW Decision chains on it."""

    world = _world(tmp_path)
    _event_, root = world.commit("root", intent="a" * 64)
    _cycle, operations = _cycle_to_decision(world, root)
    world.commit_record(operations[-1])
    decision = operations[-1].cycle_decision
    reader = SubjectHistoryReader(world.journal, world.store)
    head = reader.head(SCOPE)
    assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, decision, 2)
    adopted = reader.adopted_records()
    assert adopted.blocked is None
    assert [(item.operation_kind, item.record_kind) for item in adopted.records] == [
        (K.R19_WAKE_MATERIALIZE, R.WAKE), (K.R19_WAKE_CLAIM, R.WAKE_CLAIM), (K.R16_CYCLE_DECISION, R.CYCLE_CLOSURE),
    ]
    assert adopted.records[-1].record == operations[-1].record
    world.sequence = operations[-1].processing_sequence
    _event_, child = world.commit("child", intent="b" * 64, parent=decision)
    head = SubjectHistoryReader(world.journal, world.store).head(SCOPE)
    assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, child, 3)
    chain = SubjectHistoryReader(world.journal, world.store).classify((SCOPE,)).chains[SCOPE]
    assert [node.decision for node in chain] == [root, decision, child]


def test_every_r19_and_plan_hold_record_is_read_only_when_adopted(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _event_, root = world.commit("root", intent="a" * 64)
    committed: list[HistoryRecordOperationV1] = []

    def commit(operation: HistoryRecordOperationV1) -> None:
        world.commit_record(operation)
        committed.append(operation)

    cycle = _WorldCycle(world, "goal-r")
    commit(cycle.materialize())
    for attempt, outcome in ((1, AttemptOutcome.REFUSED), (2, AttemptOutcome.DEFERRED_BUDGET)):
        commit(cycle.claim(attempt))
        commit(cycle.outcome(outcome))
    commit(cycle.claim(3))
    hold = cycle.hold()
    commit(hold)
    message = cycle.message(hold)
    commit(message)
    commit(cycle.ack(message.record))  # type: ignore[arg-type]
    failed = _WorldCycle(world, "goal-f")
    commit(failed.materialize())
    commit(failed.terminal(TerminalReason.EXPIRED_WAKE))
    terminal = committed[-1]
    commit(failed.message(terminal))
    commit(_mutation(_plan_hold_op(_plan_hold_at(world.incarnation, world.sequence + 1))))
    reader = SubjectHistoryReader(world.journal, world.store)
    adopted = reader.adopted_records()
    assert adopted.blocked is None
    assert [item.record for item in adopted.records] == [operation.record for operation in committed]
    assert {item.record_kind for item in adopted.records} == {
        R.WAKE, R.WAKE_CLAIM, R.WAKE_ATTEMPT_OUTCOME, R.CYCLE_CLOSURE, R.OUTBOX_MESSAGE, R.INBOX_ACK, R.WAKE_TERMINAL, R.PLAN_HOLD,
    }
    head = reader.head(SCOPE)  # no record moves the head
    assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, root, 1)
    # FINALIZED but not yet terminal: invisible, nothing else blocked.
    late = _WorldCycle(world, "goal-late")
    world.commit_record(late.materialize(), complete=False)
    pending = SubjectHistoryReader(world.journal, world.store).adopted_records()
    assert pending.blocked is None and [item.record for item in pending.records] == [item.record for item in adopted.records]


def test_a_cycle_decision_is_not_a_head_until_terminal(tmp_path: Path) -> None:
    world = _world(tmp_path)
    _event_, root = world.commit("root", intent="a" * 64)
    _cycle, operations = _cycle_to_decision(world, root)
    closure = operations[-1]
    # PREPARED with an open R07 transaction: the scope is unresolved.
    event = AgentEvent(closure.event_id, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, closure.processing_sequence)
    coordinator = _prepared(world.journal, event, world.record_participant(closure))
    assert world.reader().head(SCOPE).reason is HeadBlockReason.UNRESOLVED_OPERATION
    # FINALIZED, R07 committed, but the event has no terminal yet: neither a
    # head nor a record (the same rule as a PLAN_PREVIEW Decision).
    evidence = world.recovery.commit_internal_candidate(event, world.state.load(), _snapshot(closure.processing_sequence))
    coordinator.finalize_event(event, evidence)
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision) == (HeadStatus.BLOCKED, None)
    adopted = world.reader().adopted_records()
    assert adopted.blocked is None and K.R16_CYCLE_DECISION not in {item.operation_kind for item in adopted.records}
    world.recovery.complete_committed_event(event, evidence)
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision) == (HeadStatus.ADOPTED, closure.cycle_decision)


def test_a_cycle_decision_or_resolution_mismatch_diverges(tmp_path: Path) -> None:
    def tampered(name: str, table: str, column: str, value: object) -> HeadBlockReason | None:
        world = _world(tmp_path / name)
        _event_, root = world.commit("root", intent="a" * 64)
        _cycle, operations = _cycle_to_decision(world, root)
        world.commit_record(operations[-1])
        assert world.reader().head(SCOPE).status is HeadStatus.ADOPTED
        connection = sqlite3.connect(world.store.path, isolation_level=None)
        for (trigger,) in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'trigger' AND tbl_name = ? AND name LIKE '%update%'", (table,)).fetchall():
            connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(f"UPDATE {table} SET {column} = ?", (value,))
        connection.close()
        reader = world.reader()
        assert reader.adopted_records().blocked is HeadBlockReason.DIVERGED
        return reader.head(SCOPE).reason

    assert tampered("resolution", "logical_cycle_resolutions", "record_digest", "0" * 64) is HeadBlockReason.DIVERGED
    assert tampered("source", "history_source_refs", "digest", "0" * 64) is HeadBlockReason.DIVERGED
    assert tampered("decision", "history_decisions", "policy_input_digest", "0" * 64) is HeadBlockReason.DIVERGED


def test_a_cycle_decision_envelope_mismatch_blocks_its_scope(tmp_path: Path) -> None:
    """The per-kind operation-digest check: the envelope must embed exactly
    the stored Decision and hash to the row's operation digest."""

    world = _world(tmp_path)
    _event_, root = world.commit("root", intent="a" * 64)
    _cycle, operations = _cycle_to_decision(world, root)
    world.commit_record(operations[-1])
    connection = sqlite3.connect(world.store.path, isolation_level=None)
    connection.execute("DROP TRIGGER history_operations_transition")
    connection.execute(
        "UPDATE history_operations SET operation_digest = ? WHERE operation_kind = 'R16_CYCLE_DECISION'",
        ("0" * 64,),
    )
    connection.close()
    # The R07 binding no longer matches the row: the whole incarnation diverges.
    assert world.reader().head(SCOPE).reason is HeadBlockReason.DIVERGED


def test_startup_rolls_a_committed_cycle_decision_forward(tmp_path: Path) -> None:
    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator

    world = _world(tmp_path)
    _event_, root = world.commit("root", intent="a" * 64)
    _cycle, operations = _cycle_to_decision(world, root)
    closure = operations[-1]
    event = AgentEvent(closure.event_id, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, closure.processing_sequence)
    _prepared(world.journal, event, world.record_participant(closure))
    world.recovery.commit_internal_candidate(event, world.state.load(), _snapshot(closure.processing_sequence))
    world.store._r19_enabled = False  # restarted switched off: FINALIZE is not gated
    coordinator = StartupReconciliationCoordinator(
        world.journal, world.recovery, world.memory, history_store=world.store, history_incarnation=lambda: world.incarnation,
    )
    assert coordinator.reconcile_open_transactions()[0] is True
    world.recovery.prepare_startup()
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, closure.cycle_decision, 2)
    assert world.reader().adopted_records().records[-1].operation_kind is K.R16_CYCLE_DECISION


def test_startup_aborts_a_cycle_decision_prepared_before_the_internal_commit(tmp_path: Path) -> None:
    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator

    world = _world(tmp_path)
    _event_, root = world.commit("root", intent="a" * 64)
    _cycle, operations = _cycle_to_decision(world, root)
    closure = operations[-1]
    event = AgentEvent(closure.event_id, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, closure.processing_sequence)
    _prepared(world.journal, event, world.record_participant(closure))
    # Crash before R06: no internal commit.
    coordinator = StartupReconciliationCoordinator(
        world.journal, world.recovery, world.memory, history_store=world.store, history_incarnation=lambda: world.incarnation,
    )
    coordinator.reconcile_open_transactions()
    world.recovery.prepare_startup()
    rows = [row for row in world.store.operations(world.incarnation) if row.operation_kind is K.R16_CYCLE_DECISION]
    assert [row.phase for row in rows] == [HistoryPhase.ABORTED]
    assert world.store.live_keys(closure.reserved_keys) == frozenset()
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, root, 1)
    assert world.store.prepare_collisions(closure).clear


# ------------------------------------------------------------- C10 attested


def test_attested_cycle_records_and_decision_survive_compaction(tmp_path: Path) -> None:
    from test_r16_ip9_c10_history_attestation import _chain, _compact, _provider, _small

    world = _world(tmp_path)
    _small(world)
    _event_, root = world.commit("root", intent="a" * 64)
    _cycle, operations = _cycle_to_decision(world, root)
    world.commit_record(operations[-1])
    decision = operations[-1].cycle_decision
    fillers = _chain(world, 4, prefix="filler", parent=decision)
    outcome = _compact(world, _provider(world))
    assert outcome.performed, outcome.reason
    reader = SubjectHistoryReader(world.journal, world.store)
    classified = reader.classify((SCOPE,))
    assert classified.global_block is None
    high_water = classified.attested_high_water
    assert operations[-1].processing_sequence <= high_water  # the attested path
    assert [node.decision for node in classified.chains[SCOPE]] == [root, decision, *fillers]
    adopted = reader.adopted_records()
    assert adopted.blocked is None
    assert [item.operation_kind for item in adopted.records] == [K.R19_WAKE_MATERIALIZE, K.R19_WAKE_CLAIM, K.R16_CYCLE_DECISION]
    connection = sqlite3.connect(world.store.path, isolation_level=None)
    connection.execute("DROP TRIGGER logical_cycle_resolutions_no_update")
    connection.execute("UPDATE logical_cycle_resolutions SET record_digest = ?", ("1" * 64,))
    connection.close()
    assert SubjectHistoryReader(world.journal, world.store).adopted_records().blocked is HeadBlockReason.DIVERGED


# ---------------------------------------------------------------- startup


class _Journal:
    def inspect(self):  # noqa: ANN202
        from types import SimpleNamespace

        return SimpleNamespace(journal_lineage_id="lineage-1", latest_compaction=None, baselines=(), active_baseline=None)


def _open(path: Path, caplog: pytest.LogCaptureFixture, **switches: bool) -> tuple[SubjectHistoryStore, list[str]]:
    from suzka.r16 import history_adoption
    from suzka.r16.history_adoption import open_or_create_store
    from test_r16_ip9_subject_history import PAGES

    caplog.clear()
    original = history_adoption.adopted_history_baseline
    history_adoption.adopted_history_baseline = lambda inspection: object()  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.WARNING, logger="suzka.r16.history_adoption"):
            store = open_or_create_store(_Journal(), path, subject_id="suzka", max_page_count=PAGES, **switches)  # type: ignore[arg-type]
    finally:
        history_adoption.adopted_history_baseline = original  # type: ignore[assignment]
    return store, [r.getMessage() for r in caplog.records if r.name == "suzka.r16.history_adoption"]


def test_r19_rows_with_r19_off_start_switched_off(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """D1 r2 §6-1 (mirrors IP-16b §6): start, log once, refuse new, settle
    pending, readers keep running; never a refusal to start."""

    store = _store(tmp_path)
    cycle = Cycle()
    _apply(store, cycle.materialize())
    claim = cycle.claim(1)
    pending = _prepare(store, claim)
    path = store.path
    store.close()
    off, codes = _open(path, caplog)
    assert codes == ["R19_RECORDS_PRESENT_BUT_DISABLED"]
    with pytest.raises(HistoryStoreRecordsDisabled, match="r19"):
        _prepare(off, Cycle("goal-x").materialize())
    from test_r16_ip14a_history_v2 import _receipt

    off.finalize_record(transaction_id=pending, operation=claim, receipt=_receipt(pending, claim.event_id, claim.processing_sequence))
    assert off.record_rows_match(INC, pending, claim)
    off.close()
    on, codes = _open(path, caplog, r19_enabled=True)
    assert codes == []
    on.close()
    # r18 / plan-hold switches do not silence it, and do not add their codes.
    other, codes = _open(path, caplog, r18_enabled=True, plan_hold_receipts=True)
    assert codes == ["R19_RECORDS_PRESENT_BUT_DISABLED"]
    other.close()


def test_plan_hold_rows_with_the_switch_off_start_switched_off(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """D1 r2 §6-1 (r2 final review P3-2) and §4-3 (P3-1): plan-hold rows have
    their own code and never raise the R18 or R19 one."""

    store = _store(tmp_path)
    _apply(store, _plan_hold_op(_plan_hold()))
    path = store.path
    store.close()
    off, codes = _open(path, caplog)
    assert codes == ["PLAN_HOLD_RECORDS_PRESENT_BUT_DISABLED"]
    with pytest.raises(HistoryStoreRecordsDisabled, match="plan_hold"):
        _prepare(off, _plan_hold_op(_plan_hold("second")))
    off.close()
    on, codes = _open(path, caplog, plan_hold_receipts=True)
    assert codes == []
    on.close()


def test_an_r16_only_store_logs_nothing_new(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from test_r16_ip14a_history_v2 import _decision

    store = _store(tmp_path)
    _decision(store)
    path = store.path
    store.close()
    off, codes = _open(path, caplog)
    assert codes == []
    off.close()


def test_the_r07_world_starts_switched_off_and_still_reads(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The real Journal: a store with an adopted cycle Decision and a pending
    claim opens with every switch off, logs once, settles the claim at
    startup reconcile, and the head reader still returns the cycle Decision."""

    from suzka.r16.history_adoption import open_or_create_store
    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator
    from test_r16_ip9_subject_history import PAGES

    world = _world(tmp_path)
    _event_, root = world.commit("root", intent="a" * 64)
    cycle, operations = _cycle_to_decision(world, root)
    world.commit_record(operations[-1])
    other = _WorldCycle(world, "goal-2")
    world.commit_record(other.materialize())
    claim = other.claim(1)
    event = AgentEvent(claim.event_id, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, claim.processing_sequence)
    _prepared(world.journal, event, world.record_participant(claim))
    world.recovery.commit_internal_candidate(event, world.state.load(), _snapshot(claim.processing_sequence))
    path = world.store.path
    world.store.close()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="suzka.r16.history_adoption"):
        store = open_or_create_store(world.journal, path, subject_id="suzka", max_page_count=PAGES)
    assert [r.getMessage() for r in caplog.records if r.name == "suzka.r16.history_adoption"] == ["R19_RECORDS_PRESENT_BUT_DISABLED"]
    world.store = store
    coordinator = StartupReconciliationCoordinator(
        world.journal, world.recovery, world.memory, history_store=store, history_incarnation=lambda: world.incarnation,
    )
    assert coordinator.reconcile_open_transactions()[0] is True
    world.recovery.prepare_startup()
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision) == (HeadStatus.ADOPTED, operations[-1].cycle_decision)
    assert world.reader().adopted_records().records[-1].record == claim.record
    second = other.claim(2)
    with pytest.raises(ParticipantUnavailableError):  # a new R19 PREPARE is refused
        world.record_participant(second).prepare(_binding(second))


# ------------------------------------- R16 equivalence with 64e5bbe (all off)

BASE = "64e5bbe"
_SCENARIO = """
import json, logging, sys
sys.path[:0] = [{code!r}, {tests!r}]
from pathlib import Path
from fastapi.testclient import TestClient
from suzka.r16.history_reader import SubjectHistoryReader
from test_r16_ip10_decision_issuer import _app, _issue, _settings
from test_r16_u3a_plan_preview import _qualified_goal

logs = []
class _Capture(logging.Handler):
    def emit(self, record):
        if record.name.startswith("suzka.r16") and record.levelno >= logging.WARNING:
            logs.append([record.name, record.levelname, record.getMessage()])
logging.getLogger().addHandler(_Capture())

def result(value):
    return [value.status.value, None if value.row is None else value.row.value, None if value.owner_reason is None else value.owner_reason.value, value.decision_id]

root = Path({root!r})
if {gid!r} is None:
    settings = _settings(root)
else:
    # Reopen an existing root: settings_in makes a fresh root, so build them on
    # a probe root and point every path at the existing one (IP-14b-1's way).
    probe = str(root) + "-probe-" + Path(sys.executable).name
    def move(value):
        if isinstance(value, Path) and str(value).startswith(probe):
            return Path(str(value).replace(probe, str(root), 1))
        if hasattr(value, "model_copy"):
            return value.model_copy(update={{name: move(getattr(value, name)) for name in type(value).model_fields}})
        if isinstance(value, list):
            return [move(item) for item in value]
        return value
    settings = move(_settings(Path(probe)))
with TestClient(_app(settings)) as client:
    state = client.app.state
    gid = {gid!r}
    out = {{}}
    if gid is None:
        _x, gid = _qualified_goal(state)
        out["first"] = result(_issue(state, gid))
    out["again"] = result(_issue(state, gid))
    store = state.subject_history_store
    reader = SubjectHistoryReader(state.event_journal, store)
    classified = reader.classify()
    heads = {{scope: [head.status.value, None if head.reason is None else head.reason.value, None if head.decision is None else head.decision.decision_id, head.depth] for scope, head in classified.heads.items()}}
    out.update(gid=gid, heads=heads, block=classified.global_block,
               rows=[[row.operation_kind.value, row.phase.value, row.transaction_kind] for row in classified.rows],
               records=len(classified.records), adopted=len(reader.adopted_records().records), logs=logs)
print(json.dumps(out))
"""


def _code(tmp_path: Path) -> Path:
    root = Path(__file__).parents[1]
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{BASE}^{{commit}}"], check=False, capture_output=True).returncode != 0:
        pytest.skip(f"{BASE} is not in this clone")
    target = tmp_path / "base"
    target.mkdir()
    archive = subprocess.run(["git", "-C", str(root), "archive", BASE, "suzka", "tests", "config.yaml"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive, check=True)
    return target


def _run(code: Path, root: Path, gid: str | None = None) -> dict[str, Any]:
    script = _SCENARIO.format(code=str(code), tests=str(code / "tests"), root=str(root), gid=gid)
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=code, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _without_ids(run: dict[str, Any]) -> dict[str, Any]:
    def strip(value: list[Any] | None) -> list[Any] | None:
        return None if value is None else value[:-1]

    heads = sorted([*head[:2], head[3]] for head in run["heads"].values())
    return {
        "first": strip(run.get("first")), "again": strip(run["again"]), "heads": heads, "block": run["block"],
        "rows": run["rows"], "records": run["records"], "adopted": run["adopted"], "logs": run["logs"],
    }


def test_r16_with_every_switch_off_is_identical_to_the_base(tmp_path: Path) -> None:
    """The R16 path with R18 / R19 / plan-hold off behaves exactly as at
    64e5bbe (IP-14b-1): the same results, heads, rows and r16 warning logs
    for the same scenario, and each code reads the other's store to the same
    head with the re-issue EXISTING and no SubjectHistory row added."""

    base = _code(tmp_path)
    here = Path(__file__).parents[1]
    old_root, new_root = tmp_path / "made-by-base", tmp_path / "made-by-this"
    old = _run(base, old_root)
    new = _run(here, new_root)
    assert old["first"][0] == "decision_issued" and old["again"][0] == "decision_existing"
    assert _without_ids(new) == _without_ids(old)
    assert old["logs"] == [] and old["block"] is None
    for code, root, made in ((here, old_root, old), (base, new_root, new)):
        path = root / "history" / "subject_history.sqlite3"
        before = _all_rows(path)
        read = _run(code, root, made["gid"])
        assert read["heads"] == made["heads"]
        assert read["again"] == made["again"] and read["again"][-1] == made["first"][-1]
        assert read["rows"] == made["rows"] and read["logs"] == []
        assert _all_rows(path) == before
