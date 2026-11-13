"""IP-9 U4-1/U4-2: SubjectHistory codec, trusted store and R07 participant (#376)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
from uuid import NAMESPACE_URL, uuid5

import pytest

from suzka.r16.contracts import (
    CandidateKind,
    CandidateSelection,
    DecisionRecord,
    GoalEvidenceRef,
    PlanCandidate,
    PlanDAG,
    PlanStep,
    ReasonCode,
    StepKind,
    ValueEvidenceRef,
    canonical_json,
)
from suzka.r16.history_codec import (
    GENESIS,
    MAX_POLICY_INPUT_BYTES,
    SUBJECT_HISTORY_PARTICIPANT_ID,
    CommittedDecisionV1,
    HistorySourceRef,
    PolicyKind,
    PolicyRow,
    PostcommitReceiptV1,
    SourceRole,
    decision_id_for,
    history_operation_digest,
    policy_input_digest,
)
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryParticipant,
)
from suzka.r16.history_store import (
    HistoryMetaPhase,
    HistoryOperationKind,
    HistoryPhase,
    HistoryStoreCapacityError,
    HistoryStoreConflict,
    HistoryStoreCorrupt,
    HistoryStoreUnavailable,
    SubjectHistoryStore,
)
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from suzka.runtime.event_journal import (
    AbortOutcome,
    EventJournal,
    ParticipantOutcome,
    StartupParticipantOutcome,
    TransactionKind,
)
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    ParticipantDivergedError,
    ParticipantUnavailableError,
    TransactionBinding,
    TransactionCoordinator,
    TransactionCoordinator as _Coordinator,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)
INCARNATION = "1" * 64
OTHER_INCARNATION = "2" * 64
PAGES = 4096


# ------------------------------------------------------------------ builders


def _plan(index: int) -> PlanDAG:
    return PlanDAG(
        1,
        f"plan-{index}",
        (
            PlanStep(1, "assess", StepKind.ASSESS, ()),
            PlanStep(
                1, "review", StepKind.CHECK if index == 0 else StepKind.COMPARE, ("assess",)
            ),
        ),
    )


def _u1(decision_id: str, *, no_op: bool = False) -> DecisionRecord:
    goal = GoalEvidenceRef(1, "a" * 64, 2, "b" * 64)
    value = ValueEvidenceRef(1, "value-1", 3, "c" * 64)
    options = (
        PlanCandidate(1, "candidate-a", CandidateKind.PLAN, goal, value, _plan(0)),
        PlanCandidate(1, "candidate-b", CandidateKind.PLAN, goal, value, _plan(1)),
        PlanCandidate(1, "candidate-noop", CandidateKind.NO_OP, goal, value, None),
    )
    chosen = options[2] if no_op else options[0]
    return DecisionRecord(
        1,
        decision_id,
        options,
        CandidateSelection(1, chosen.candidate_id, chosen.content_digest),
        ReasonCode.NO_OP_PREFERRED if no_op else ReasonCode.PLAN_PREFERRED,
        250_000,
    )


def _event(name: str, sequence: int = 1) -> AgentEvent:
    return AgentEvent(
        str(uuid5(NAMESPACE_URL, name)),
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        NOW,
        sequence,
    )


def _committed(
    event: AgentEvent,
    *,
    incarnation: str = INCARNATION,
    intent: str = "e" * 64,
    predecessor: tuple[str, str | None] = (GENESIS, None),
    row: PolicyRow = PolicyRow.ROW_4,
    kind: PolicyKind = PolicyKind.SUCCESSOR_POLICY_V2,
    policy_input: bytes = b'{"row":"4"}',
    sources: tuple[HistorySourceRef, ...] | None = None,
) -> CommittedDecisionV1:
    assert event.processing_sequence is not None
    identity = decision_id_for(
        incarnation,
        event.event_id,
        event.processing_sequence,
        intent,
        policy_input_digest(policy_input),
    )
    return CommittedDecisionV1(
        incarnation_id=incarnation,
        decision_id=identity,
        issuer_event_id=event.event_id,
        issuer_sequence=event.processing_sequence,
        lineage_scope="goal-lineage-1",
        review_intent_key=intent,
        predecessor_key=predecessor[0],
        predecessor_digest=predecessor[1],
        decision=_u1(identity, no_op=row is PolicyRow.ROW_5),
        policy_kind=kind,
        policy_row=row,
        policy_input_bytes=policy_input,
        source_refs=sources
        if sources is not None
        else (
            HistorySourceRef(SourceRole.EXPERIENCE, "experience", "experience-1", 0, "d" * 64),
            HistorySourceRef(SourceRole.GOAL, "goal", "goal-1", 2, "b" * 64),
        ),
    )


def _binding(event: AgentEvent, decision: CommittedDecisionV1) -> TransactionBinding:
    return TransactionBinding(
        transaction_id=_Coordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        ),
        event_id=event.event_id,
        processing_sequence=event.processing_sequence or 0,
        participant_id=SUBJECT_HISTORY_PARTICIPANT_ID,
        operation_digest=history_operation_digest(decision),
        transaction_kind=TransactionKind.EVENT_MUTATION,
    )


def _receipt(binding: TransactionBinding) -> PostcommitReceiptV1:
    return PostcommitReceiptV1(
        transaction_id=binding.transaction_id,
        event_id=binding.event_id,
        processing_sequence=binding.processing_sequence,
        prepared_state_hash_before="3" * 64,
        prepared_state_hash_after="4" * 64,
        snapshot_sequence=binding.processing_sequence,
        snapshot_hash="4" * 64,
        wal_generation_id=str(uuid5(NAMESPACE_URL, "generation")),
        wal_record_id=str(uuid5(NAMESPACE_URL, f"record-{binding.transaction_id}")),
        wal_record_hash="5" * 64,
    )


class FakeProof:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.calls = 0

    def postcommit_receipt(self, binding: TransactionBinding) -> PostcommitReceiptV1:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _receipt(binding)


def _private_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "history"
    directory.mkdir(mode=0o700, parents=True)
    directory.chmod(0o700)
    return directory


def _store(tmp_path: Path, *, active: bool = True, pages: int = PAGES) -> SubjectHistoryStore:
    store = SubjectHistoryStore.create(
        _private_dir(tmp_path) / "subject_history.sqlite3",
        subject_id="suzka",
        journal_lineage_id="lineage-1",
        max_page_count=pages,
    )
    if active:
        store.activate(
            adopted_baseline_id="baseline-1",
            adopted_baseline_hash="9" * 64,
            genesis_incarnation_id=INCARNATION,
        )
    return store


# --------------------------------------------------------------------- codec


def test_committed_decision_roundtrips_exactly_and_binds_its_identity() -> None:
    event = _event("codec")
    value = _committed(event)
    payload = value.canonical_bytes()

    assert CommittedDecisionV1.from_json(payload) == value
    assert value.decision.decision_id == value.decision_id
    assert history_operation_digest(value) == history_operation_digest(
        CommittedDecisionV1.from_json(payload)
    )
    # Any byte change, including a noncanonical respelling, is refused.
    with pytest.raises(ValueError):
        CommittedDecisionV1.from_json(payload.replace(b'"row":"4"', b'"row":"5"'))
    with pytest.raises(ValueError):
        CommittedDecisionV1.from_json(payload + b" ")


def test_decision_identity_is_bound_to_incarnation_event_and_policy_input() -> None:
    event = _event("identity")
    value = _committed(event)
    other = decision_id_for(
        OTHER_INCARNATION, event.event_id, 1, value.review_intent_key,
        value.policy_input_digest,
    )
    assert other != value.decision_id
    with pytest.raises(ValueError, match="decision_id"):
        replace(value, policy_input_bytes=b'{"row":"other"}')
    with pytest.raises(ValueError, match="incarnation|decision_id"):
        replace(value, incarnation_id=OTHER_INCARNATION)


@pytest.mark.parametrize(
    ("row", "kind", "no_op_mismatch"),
    [
        (PolicyRow.ROW_4B, PolicyKind.SUCCESSOR_POLICY_V1, False),
        (PolicyRow.ROW_5, PolicyKind.SUCCESSOR_POLICY_V2, True),
    ],
)
def test_policy_row_must_match_its_policy_version_and_u1_reason(
    row: PolicyRow, kind: PolicyKind, no_op_mismatch: bool
) -> None:
    event = _event(f"row-{row.value}-{kind.value}")
    if no_op_mismatch:
        valid = _committed(event, row=row, kind=kind)
        assert valid.decision.reason is ReasonCode.NO_OP_PREFERRED
        with pytest.raises(ValueError, match="reason"):
            replace(valid, decision=_u1(valid.decision_id))
    else:
        with pytest.raises(ValueError, match="b rows"):
            _committed(event, row=row, kind=kind)


def test_predecessor_sources_and_policy_input_bounds_fail_closed() -> None:
    event = _event("bounds")
    value = _committed(event)
    with pytest.raises(ValueError, match="GENESIS"):
        replace(value, predecessor_digest="7" * 64)
    with pytest.raises(ValueError):
        replace(value, predecessor_key="decision-" + "7" * 64, predecessor_digest=None)
    with pytest.raises(ValueError, match="sorted"):
        replace(value, source_refs=tuple(reversed(value.source_refs)))
    with pytest.raises(ValueError, match="at least one"):
        replace(value, source_refs=())
    oversized = canonical_json({"x": "y" * MAX_POLICY_INPUT_BYTES})
    with pytest.raises(ValueError, match="byte bound"):
        _committed(event, policy_input=oversized)
    with pytest.raises(ValueError, match="object"):
        _committed(event, policy_input=b"[1]")
    chained = _committed(
        event, intent="f" * 64, predecessor=(value.decision_id, value.content_digest)
    )
    assert CommittedDecisionV1.from_json(chained.canonical_bytes()) == chained


# --------------------------------------------------------------------- store


def test_store_creation_is_explicit_and_open_never_creates(tmp_path: Path) -> None:
    store = _store(tmp_path, active=False)
    path = store.path
    meta = store.meta()
    assert meta.phase is HistoryMetaPhase.PROVISIONAL
    assert (path.stat().st_mode & 0o777) == 0o600
    store.close()

    with pytest.raises(HistoryStoreConflict):
        SubjectHistoryStore.create(
            path, subject_id="suzka", journal_lineage_id="lineage-1", max_page_count=PAGES
        )
    with pytest.raises(HistoryStoreUnavailable):
        SubjectHistoryStore.open(path.with_name("absent.sqlite3"), max_page_count=PAGES)
    assert not path.with_name("absent.sqlite3").exists()
    reopened = SubjectHistoryStore.open(path, max_page_count=PAGES)
    assert reopened.meta() == meta
    assert [name for name in os.listdir(path.parent)] == [path.name]


def test_activation_is_one_way_and_identity_is_immutable(tmp_path: Path) -> None:
    store = _store(tmp_path, active=False)
    event = _event("inactive")
    decision = _committed(event)
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof())
    with pytest.raises(ParticipantUnavailableError):
        participant.prepare(_binding(event, decision))

    active = store.activate(
        adopted_baseline_id="baseline-1",
        adopted_baseline_hash="9" * 64,
        genesis_incarnation_id=INCARNATION,
    )
    assert active.phase is HistoryMetaPhase.ACTIVE
    assert store.activate(
        adopted_baseline_id="baseline-1",
        adopted_baseline_hash="9" * 64,
        genesis_incarnation_id=INCARNATION,
    ) == active
    with pytest.raises(HistoryStoreConflict):
        store.activate(
            adopted_baseline_id="baseline-2",
            adopted_baseline_hash="9" * 64,
            genesis_incarnation_id=INCARNATION,
        )
    raw = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("UPDATE history_meta SET subject_id = 'other'")
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("DELETE FROM history_meta")
    raw.close()


@pytest.mark.parametrize("case", ["symlink", "mode", "directory_mode", "relative"])
def test_store_custody_refuses_untrusted_files(tmp_path: Path, case: str) -> None:
    store = _store(tmp_path)
    path = store.path
    store.close()
    target = path
    if case == "symlink":
        target = path.with_name("link.sqlite3")
        target.symlink_to(path)
    elif case == "mode":
        path.chmod(0o644)
    elif case == "directory_mode":
        path.parent.chmod(0o777)
    elif case == "relative":
        target = Path("relative.sqlite3")
    with pytest.raises(HistoryStoreCorrupt):
        SubjectHistoryStore.open(target, max_page_count=PAGES)


def test_tampered_schema_or_pragmas_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.path
    store.close()
    raw = sqlite3.connect(path)
    raw.execute("DROP TRIGGER history_decisions_no_update")
    raw.commit()
    raw.close()
    with pytest.raises(HistoryStoreCorrupt, match="schema"):
        SubjectHistoryStore.open(path, max_page_count=PAGES)

    fresh = _store(tmp_path / "second")
    fresh_path = fresh.path
    fresh.close()
    # A page cap below the current size is a capacity condition (#376 D1-9).
    with pytest.raises(HistoryStoreCapacityError, match="max_page_count"):
        SubjectHistoryStore.open(fresh_path, max_page_count=16)


def test_sql_triggers_keep_operations_append_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = _event("triggers")
    decision = _committed(event)
    binding = _binding(event, decision)
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof())
    participant.prepare(binding)
    raw = sqlite3.connect(store.path)
    raw.execute("PRAGMA foreign_keys = ON")
    # FINALIZE without its receipt and Decision is refused.
    with pytest.raises(sqlite3.IntegrityError, match="incomplete"):
        raw.execute("UPDATE history_operations SET phase = 'FINALIZED'")
    # Bound columns are immutable.
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("UPDATE history_operations SET operation_digest = ?", ("0" * 64,))
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("DELETE FROM history_operations")
    raw.close()

    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    raw = sqlite3.connect(store.path)
    for statement in (
        "UPDATE history_operations SET phase = 'ABORTED'",
        "UPDATE history_decisions SET policy_row = '6'",
        "DELETE FROM history_decisions",
        "UPDATE history_source_refs SET revision = 9",
        "DELETE FROM history_postcommit_receipts",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(statement)
    raw.close()


# --------------------------------------------------------------- participant


def test_prepare_finalize_is_exact_and_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    proof = FakeProof()
    event = _event("finalize")
    decision = _committed(event)
    binding = _binding(event, decision)
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, proof)

    participant.prepare(binding)
    participant.prepare(binding)
    row = store.operation(INCARNATION, binding.transaction_id)
    assert row is not None and row.phase is HistoryPhase.PREPARED
    assert row.pending_bytes == decision.canonical_bytes()

    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    assert participant.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    assert proof.calls == 1
    finalized = store.finalized_decision(INCARNATION, binding.transaction_id)
    assert finalized is not None
    assert finalized.decision_id == decision.decision_id
    assert finalized.receipt == _receipt(binding)
    assert (
        participant.inspect_reconciliation(binding)
        is StartupParticipantOutcome.VERIFIED_CONSISTENT
    )
    rebuilt = SubjectHistoryParticipant.from_pending(store, INCARNATION, proof, binding)
    assert rebuilt.decision == decision


def test_finalize_needs_proof_and_never_aborts_after_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    proof = FakeProof()
    event = _event("unproven")
    decision = _committed(event)
    binding = _binding(event, decision)
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, proof)
    participant.prepare(binding)

    proof.error = ParticipantUnavailableError("ambiguous")
    with pytest.raises(ParticipantUnavailableError):
        participant.finalize(binding)
    row = store.operation(INCARNATION, binding.transaction_id)
    assert row is not None and row.phase is HistoryPhase.PREPARED
    with pytest.raises(ParticipantUnavailableError, match="roll-forward"):
        participant.inspect_reconciliation(binding)

    proof.error = None
    assert participant.reconcile(binding) is StartupParticipantOutcome.ROLLED_FORWARD
    with pytest.raises(ParticipantDivergedError, match="finalized"):
        participant.abort(binding)


def test_abort_tombstone_makes_a_transaction_unreusable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = _event("tombstone")
    decision = _committed(event)
    binding = _binding(event, decision)

    assert (
        SubjectHistoryParticipant.abort_pending(store, INCARNATION, binding)
        is AbortOutcome.ALREADY_ABSENT
    )
    row = store.operation(INCARNATION, binding.transaction_id)
    assert row is not None
    assert row.operation_kind is HistoryOperationKind.R07_BINDING_ONLY_ABORT
    assert row.phase is HistoryPhase.ABORTED and row.pending_bytes is None
    assert (
        SubjectHistoryParticipant.abort_pending(store, INCARNATION, binding)
        is AbortOutcome.ALREADY_ABSENT
    )
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof())
    with pytest.raises(ParticipantDivergedError, match="already used"):
        participant.prepare(binding)
    with pytest.raises(ParticipantDivergedError):
        participant.finalize(binding)
    with pytest.raises(ParticipantDivergedError):
        SubjectHistoryParticipant.from_pending(store, INCARNATION, FakeProof(), binding)


def test_prepared_abort_releases_the_lineage_slot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_event = _event("abort-first")
    first = _committed(first_event)
    first_binding = _binding(first_event, first)
    SubjectHistoryParticipant(store, INCARNATION, first, FakeProof()).prepare(first_binding)

    second_event = _event("abort-second", 2)
    second = _committed(second_event)
    second_participant = SubjectHistoryParticipant(store, INCARNATION, second, FakeProof())
    second_binding = _binding(second_event, second)
    # Same review intent and GENESIS predecessor: a second PREPARED child is refused.
    with pytest.raises(ParticipantDivergedError):
        second_participant.prepare(second_binding)

    assert (
        SubjectHistoryParticipant.abort_pending(store, INCARNATION, first_binding)
        is AbortOutcome.ABORTED
    )
    second_participant.prepare(second_binding)
    assert second_participant.finalize(second_binding) is ParticipantOutcome.FINALIZED


def test_single_child_per_predecessor_and_incarnation_scope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root_event = _event("lineage-root")
    root = _committed(root_event)
    root_binding = _binding(root_event, root)
    root_participant = SubjectHistoryParticipant(store, INCARNATION, root, FakeProof())
    root_participant.prepare(root_binding)
    root_participant.finalize(root_binding)

    child_event = _event("lineage-child", 2)
    child = _committed(
        child_event, intent="f" * 64, predecessor=(root.decision_id, root.content_digest)
    )
    child_participant = SubjectHistoryParticipant(store, INCARNATION, child, FakeProof())
    child_binding = _binding(child_event, child)
    child_participant.prepare(child_binding)
    child_participant.finalize(child_binding)

    sibling_event = _event("lineage-sibling", 3)
    sibling = _committed(
        sibling_event, intent="a" * 64, predecessor=(root.decision_id, root.content_digest)
    )
    with pytest.raises(ParticipantDivergedError):
        SubjectHistoryParticipant(store, INCARNATION, sibling, FakeProof()).prepare(
            _binding(sibling_event, sibling)
        )

    # A new incarnation starts at its own GENESIS without colliding (#376 D1-4).
    store_meta_event = _event("lineage-new-incarnation", 4)
    reborn = _committed(store_meta_event, incarnation=OTHER_INCARNATION)
    reborn_participant = SubjectHistoryParticipant(
        store, OTHER_INCARNATION, reborn, FakeProof()
    )
    reborn_binding = _binding(store_meta_event, reborn)
    reborn_participant.prepare(reborn_binding)
    assert reborn_participant.finalize(reborn_binding) is ParticipantOutcome.FINALIZED
    assert len(store.operations(INCARNATION)) == 2
    assert len(store.operations(OTHER_INCARNATION)) == 1


def test_binding_mismatches_and_missing_pending_diverge(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = _event("mismatch")
    decision = _committed(event)
    binding = _binding(event, decision)
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof())

    with pytest.raises(ParticipantDivergedError):
        participant.prepare(replace(binding, operation_digest="0" * 64))
    with pytest.raises(ParticipantDivergedError):
        participant.prepare(replace(binding, participant_id="memory.experience"))
    with pytest.raises(ParticipantDivergedError):
        participant.prepare(replace(binding, transaction_id=str(uuid5(NAMESPACE_URL, "x"))))
    # Committed but never prepared: never fabricated, never aborted.
    with pytest.raises(ParticipantDivergedError, match="absent"):
        participant.finalize(binding)
    with pytest.raises(ParticipantDivergedError, match="absent"):
        SubjectHistoryParticipant.from_pending(store, INCARNATION, FakeProof(), binding)
    with pytest.raises(ValueError, match="incarnation"):
        SubjectHistoryParticipant(store, OTHER_INCARNATION, decision, FakeProof())


def test_capacity_is_refused_before_prepare(tmp_path: Path) -> None:
    store = _store(tmp_path, pages=32)
    event = _event("capacity")
    decision = _committed(event)
    assert store.free_bytes() < store.required_headroom(len(decision.canonical_bytes()))
    with pytest.raises(HistoryStoreCapacityError):
        store.check_headroom(len(decision.canonical_bytes()))
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof())
    with pytest.raises(ParticipantUnavailableError):
        participant.prepare(_binding(event, decision))
    assert store.operations(INCARNATION) == ()


# -------------------------------------------- R07 / R06 integration (no model)


def _graph(tmp_path: Path):
    from tests.test_startup_reconciliation import _graph as graph

    return graph(tmp_path)


def _snapshot(sequence: int, valence: float = 0.4):
    from tests.test_startup_reconciliation import _snapshot as snapshot

    return snapshot(sequence, valence)


def _prepared(journal: EventJournal, event: AgentEvent, participant) -> TransactionCoordinator:
    journal.append_accepted(event)
    journal.append_started(event)
    coordinator = TransactionCoordinator(journal, lambda _event, _evidence: None)
    coordinator.prepare_result(event, CoordinatedResult("public", (participant,)))
    return coordinator


def _startup(journal, recovery, memory, store):
    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator

    return StartupReconciliationCoordinator(
        journal,
        recovery,
        memory,
        history_store=store,
        history_incarnation=lambda: INCARNATION,
    )


def test_live_finalize_records_the_exact_r06_receipt(tmp_path: Path) -> None:
    memory, state, journal, wal, recovery = _graph(tmp_path)
    store = _store(tmp_path)
    event = _event("live")
    decision = _committed(event)
    participant = SubjectHistoryParticipant(
        store, INCARNATION, decision, JournalCommitProofSource(journal, recovery)
    )
    coordinator = _prepared(journal, event, participant)
    evidence = recovery.commit_internal_candidate(event, state.load(), _snapshot(1))
    coordinator.finalize_event(event, evidence)

    completed = journal.inspect().completed_transactions
    assert completed[0].participant_outcomes == (
        (SUBJECT_HISTORY_PARTICIPANT_ID, ParticipantOutcome.FINALIZED),
    )
    finalized = store.finalized_decision(INCARNATION, completed[0].transaction_id)
    assert finalized is not None
    receipt = finalized.receipt
    assert (receipt.snapshot_sequence, receipt.snapshot_hash) == (
        evidence.snapshot_sequence,
        evidence.snapshot_hash,
    )
    assert (receipt.wal_generation_id, receipt.wal_record_id, receipt.wal_record_hash) == (
        evidence.wal_generation_id,
        evidence.wal_record_id,
        evidence.wal_record_hash,
    )


def test_startup_aborts_a_pre_internal_history_operation(tmp_path: Path) -> None:
    memory, _state, journal, _wal, recovery = _graph(tmp_path)
    store = _store(tmp_path)
    event = _event("startup-pre-internal")
    decision = _committed(event)
    participant = SubjectHistoryParticipant(
        store, INCARNATION, decision, JournalCommitProofSource(journal, recovery)
    )
    _prepared(journal, event, participant)

    assert _startup(journal, recovery, memory, store).reconcile_open_transactions() == (
        True,
        None,
    )
    aborted = journal.inspect().aborted_transactions
    assert aborted[0].abort_outcomes == (
        (SUBJECT_HISTORY_PARTICIPANT_ID, AbortOutcome.ABORTED),
    )
    row = store.operation(INCARNATION, aborted[0].transaction_id)
    assert row is not None and row.phase is HistoryPhase.ABORTED


def test_startup_rolls_a_committed_operation_forward(tmp_path: Path) -> None:
    memory, state, journal, _wal, recovery = _graph(tmp_path)
    store = _store(tmp_path)
    event = _event("startup-committed")
    decision = _committed(event)
    participant = SubjectHistoryParticipant(
        store, INCARNATION, decision, JournalCommitProofSource(journal, recovery)
    )
    _prepared(journal, event, participant)
    recovery.commit_internal_candidate(event, state.load(), _snapshot(1))

    assert _startup(journal, recovery, memory, store).reconcile_open_transactions() == (
        True,
        None,
    )
    completed = journal.inspect().completed_transactions[0]
    assert completed.participant_outcomes == (
        (SUBJECT_HISTORY_PARTICIPANT_ID, ParticipantOutcome.FINALIZED),
    )
    assert store.finalized_decision(INCARNATION, completed.transaction_id) is not None


def test_startup_after_sqlite_finalize_before_journal_is_consistent(tmp_path: Path) -> None:
    memory, state, journal, _wal, recovery = _graph(tmp_path)
    store = _store(tmp_path)
    event = _event("startup-finalized-first")
    decision = _committed(event)
    participant = SubjectHistoryParticipant(
        store, INCARNATION, decision, JournalCommitProofSource(journal, recovery)
    )
    _prepared(journal, event, participant)
    recovery.commit_internal_candidate(event, state.load(), _snapshot(1))
    transaction = journal.inspect().open_transactions[0]
    # Crash after the SQLite FINALIZE but before PARTICIPANT_FINALIZED.
    binding = _binding(event, decision)
    assert binding.transaction_id == transaction.transaction_id
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED

    assert _startup(journal, recovery, memory, store).reconcile_open_transactions() == (
        True,
        None,
    )
    assert journal.inspect().completed_transactions[0].participant_outcomes == (
        (SUBJECT_HISTORY_PARTICIPANT_ID, ParticipantOutcome.ALREADY_CONSISTENT),
    )


def test_startup_fails_closed_on_lost_pending_or_ambiguous_proof(tmp_path: Path) -> None:
    from suzka.runtime.startup_reconciliation import StartupReconciliationError

    memory, state, journal, _wal, recovery = _graph(tmp_path)
    store = _store(tmp_path)
    event = _event("startup-lost")
    decision = _committed(event)
    participant = SubjectHistoryParticipant(
        store, INCARNATION, decision, JournalCommitProofSource(journal, recovery)
    )
    _prepared(journal, event, participant)
    recovery.commit_internal_candidate(event, state.load(), _snapshot(1))

    # The pending row belongs to a different incarnation: nothing to roll forward.
    lost = _startup(journal, recovery, memory, store)
    lost.history_incarnation = lambda: OTHER_INCARNATION
    assert lost.reconcile_open_transactions() == (
        False,
        "external_participant_reconciliation_required",
    )
    assert journal.inspect().open_transactions

    # No resolver at all (pre-U4-3 wiring): an unregistered participant.
    unregistered = _startup(journal, recovery, memory, None)
    assert unregistered.reconcile_open_transactions() == (
        False,
        "external_participant_reconciliation_required",
    )

    state.save(_snapshot(1, 0.9))
    with pytest.raises(StartupReconciliationError, match="ambiguous"):
        _startup(journal, recovery, memory, store).reconcile_open_transactions()
    row = store.operation(INCARNATION, journal.inspect().open_transactions[0].transaction_id)
    assert row is not None and row.phase is HistoryPhase.PREPARED


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_foreign_wal_sidecars_are_refused(tmp_path: Path, suffix: str) -> None:
    store = _store(tmp_path)
    path = store.path
    store.close()
    sidecar = path.with_name(path.name + suffix)
    sidecar.write_bytes(b"")
    sidecar.chmod(0o600)
    with pytest.raises(HistoryStoreCorrupt, match="WAL"):
        SubjectHistoryStore.open(path, max_page_count=PAGES)


def test_operation_with_finalize_rows_cannot_be_aborted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = _event("abort-with-rows")
    decision = _committed(event)
    binding = _binding(event, decision)
    SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof()).prepare(binding)
    raw = sqlite3.connect(store.path)
    raw.execute("PRAGMA foreign_keys = ON")
    # A direct writer plants a decision row outside any FINALIZE (review P2-1).
    raw.execute(
        "INSERT INTO history_decisions (decision_id, incarnation_id, transaction_id, "
        "lineage_scope, review_intent_key, predecessor_key, predecessor_digest, "
        "issuer_event_id, issuer_sequence, decision_bytes, decision_digest, "
        "committed_bytes, committed_digest, policy_kind, policy_row, "
        "policy_input_bytes, policy_input_digest) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            decision.decision_id, INCARNATION, binding.transaction_id,
            decision.lineage_scope, decision.review_intent_key, GENESIS,
            event.event_id, 1, decision.decision_bytes, "0" * 64,
            decision.canonical_bytes(), "0" * 64, decision.policy_kind.value,
            decision.policy_row.value, decision.policy_input_bytes, "0" * 64,
        ),
    )
    raw.commit()
    raw.close()
    with pytest.raises(ParticipantDivergedError):
        SubjectHistoryParticipant.abort_pending(store, INCARNATION, binding)
    row = store.operation(INCARNATION, binding.transaction_id)
    assert row is not None and row.phase is HistoryPhase.PREPARED
