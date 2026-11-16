"""IP-14a-2: record participant, reader record checks, adopted-only API,
and the two-way V2 / V3 head comparison (#411 comment 6910958369 §6 / §8).

Every R17 / R18 value is a FIXTURE ONLY.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.history_codec import PolicyKind  # noqa: E402
from suzka.r16.history_participant import (  # noqa: E402
    JournalCommitProofSource,
    SubjectHistoryParticipant,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import HeadBlockReason, SubjectHistoryReader  # noqa: E402
from suzka.r16.history_records import HistoryOperationKind, RecordKind  # noqa: E402
from suzka.r16.successor_policy import LedgerState, as_version  # noqa: E402
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType  # noqa: E402
from suzka.runtime.event_journal import ParticipantOutcome  # noqa: E402
from suzka.runtime.transaction_coordinator import (  # noqa: E402
    ParticipantDivergedError,
    ParticipantUnavailableError,
    TransactionBinding,
    TransactionCoordinator,
)
from test_r16_ip14a_history_v2 import Chain, _steps  # noqa: E402
from test_r16_ip9_history_reader import _World  # noqa: E402
from test_r16_ip9_subject_history import _prepared, _snapshot  # noqa: E402

K = HistoryOperationKind
R = RecordKind
NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _RecordWorld(_World):
    """IP-9's adopted R07 world, with R18 records enabled."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.store._r18_enabled = True

    def chain(self) -> Chain:
        _event, decision = self.commit("decision", intent="a" * 64)
        return Chain(
            decision,
            incarnation=self.incarnation,
            base=self.sequence + 1,
            transaction_kind="event_mutation",
        )

    def record_participant(self, operation) -> SubjectHistoryRecordParticipant:  # noqa: ANN001
        return SubjectHistoryRecordParticipant(
            self.store, self.incarnation, operation, JournalCommitProofSource(self.journal, self.recovery)
        )

    def commit_record(self, operation, *, complete: bool = True) -> AgentEvent:  # noqa: ANN001
        event = AgentEvent(
            operation.event_id, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, operation.processing_sequence
        )
        coordinator = _prepared(self.journal, event, self.record_participant(operation))
        evidence = self.recovery.commit_internal_candidate(
            event, self.state.load(), _snapshot(operation.processing_sequence)
        )
        coordinator.finalize_event(event, evidence)
        if complete:
            self.recovery.complete_committed_event(event, evidence)
        self.sequence = operation.processing_sequence
        return event


def _binding(operation) -> TransactionBinding:  # noqa: ANN001
    from suzka.r16.history_codec import SUBJECT_HISTORY_PARTICIPANT_ID
    from suzka.runtime.event_journal import TransactionKind

    event = AgentEvent(operation.event_id, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, operation.processing_sequence)
    return TransactionBinding(
        transaction_id=TransactionCoordinator.derive_transaction_id(event, TransactionKind.EVENT_MUTATION),
        event_id=operation.event_id,
        processing_sequence=operation.processing_sequence,
        participant_id=SUBJECT_HISTORY_PARTICIPANT_ID,
        operation_digest=operation.operation_digest,
        transaction_kind=TransactionKind.EVENT_MUTATION,
    )


class _Proof:
    def postcommit_receipt(self, binding: TransactionBinding):  # noqa: ANN201
        from test_r16_ip9_subject_history import _receipt

        return _receipt(binding)


# ------------------------------------------------------- record participant


def test_record_participant_prepares_finalizes_and_is_idempotent(tmp_path: Path) -> None:
    world = _RecordWorld(tmp_path)
    chain = world.chain()
    operation = _steps(chain)[0]
    participant = SubjectHistoryRecordParticipant(world.store, world.incarnation, operation, _Proof())
    binding = _binding(operation)
    participant.prepare(binding)
    participant.prepare(binding)  # idempotent PREPARE
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED
    assert participant.finalize(binding) is ParticipantOutcome.ALREADY_CONSISTENT
    rebuilt = SubjectHistoryParticipant.from_pending(world.store, world.incarnation, _Proof(), binding)
    assert type(rebuilt) is SubjectHistoryRecordParticipant and rebuilt.operation == operation


def test_record_participant_refuses_a_foreign_binding(tmp_path: Path) -> None:
    world = _RecordWorld(tmp_path)
    operation = _steps(world.chain())[0]
    participant = SubjectHistoryRecordParticipant(world.store, world.incarnation, operation, _Proof())
    binding = _binding(operation)
    for bad in (
        replace(binding, processing_sequence=binding.processing_sequence + 1),
        replace(binding, operation_digest="0" * 64),
    ):
        with pytest.raises(ParticipantDivergedError):
            participant.prepare(bad)
    with pytest.raises(ValueError):
        SubjectHistoryRecordParticipant(world.store, "9" * 64, operation, _Proof())


def test_disabled_r18_refuses_prepare_but_still_settles_a_pending_record(tmp_path: Path) -> None:
    world = _RecordWorld(tmp_path)
    chain = world.chain()
    operation = _steps(chain)[0]
    binding = _binding(operation)
    participant = SubjectHistoryRecordParticipant(world.store, world.incarnation, operation, _Proof())
    participant.prepare(binding)
    world.store._r18_enabled = False
    other = _steps(Chain(chain.decision, group="b" * 64, incarnation=world.incarnation, base=500, transaction_kind="event_mutation"))[0]
    with pytest.raises(ParticipantUnavailableError):
        SubjectHistoryRecordParticipant(world.store, world.incarnation, other, _Proof()).prepare(_binding(other))
    # FINALIZE / reconcile of the pending one still works (the sweep must run).
    assert participant.finalize(binding) is ParticipantOutcome.FINALIZED


def test_abort_releases_and_tombstones_like_a_decision(tmp_path: Path) -> None:
    world = _RecordWorld(tmp_path)
    operation = _steps(world.chain())[0]
    binding = _binding(operation)
    participant = SubjectHistoryRecordParticipant(world.store, world.incarnation, operation, _Proof())
    participant.prepare(binding)
    participant.abort(binding)
    assert world.store.live_keys(operation.reserved_keys) == frozenset()
    with pytest.raises(ParticipantDivergedError):
        participant.finalize(binding)


# --------------------------------------------------------------- reader


def test_adopted_records_only_after_the_event_terminal(tmp_path: Path) -> None:
    world = _RecordWorld(tmp_path)
    chain = world.chain()
    steps = _steps(chain)
    world.commit_record(steps[0])
    reader = SubjectHistoryReader(world.journal, world.store)
    adopted = reader.adopted_records()
    assert adopted.blocked is None
    assert [(item.operation_kind, item.record) for item in adopted.records] == [(K.R18_INTENT_ISSUE, chain.intent)]
    # FINALIZED but the event is not terminal yet: simply not adopted, and
    # neither the other records nor any R16 head are blocked (review P2).
    head = reader.head("goal-lineage-1")
    world.commit_record(steps[1], complete=False)
    pending = reader.adopted_records()
    assert pending.blocked is None
    assert [item.record for item in pending.records] == [chain.intent]
    later = reader.head("goal-lineage-1")
    assert (later.status, later.decision) == (head.status, head.decision)


def test_record_rows_that_differ_from_their_envelope_block_the_reader(tmp_path: Path) -> None:
    world = _RecordWorld(tmp_path)
    chain = world.chain()
    world.commit_record(_steps(chain)[0])
    connection = sqlite3.connect(world.store.path, isolation_level=None)
    connection.execute("DROP TRIGGER action_intents_no_update")
    connection.execute("UPDATE action_intents SET record_digest = ?", ("0" * 64,))
    connection.close()
    reader = SubjectHistoryReader(world.journal, world.store)
    result = reader.classify(("goal-lineage-1",))
    assert result.global_block is HeadBlockReason.DIVERGED
    assert reader.adopted_records().blocked is HeadBlockReason.DIVERGED


def test_record_rows_do_not_change_r16_heads(tmp_path: Path) -> None:
    world = _RecordWorld(tmp_path)
    chain = world.chain()
    head_before = SubjectHistoryReader(world.journal, world.store).head("goal-lineage-1")
    for operation in _steps(chain)[:3]:
        world.commit_record(operation)
    head_after = SubjectHistoryReader(world.journal, world.store).head("goal-lineage-1")
    assert (head_after.status, head_after.decision, head_after.depth) == (
        head_before.status, head_before.decision, head_before.depth,
    )


def test_startup_rolls_a_committed_record_forward(tmp_path: Path) -> None:
    """A committed record whose FINALIZE did not run is rebuilt from its
    envelope and rolled forward at startup (same path as a Decision)."""

    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator

    world = _RecordWorld(tmp_path)
    operation = _steps(world.chain())[0]
    event = AgentEvent(operation.event_id, AgentEventType.CHAT, AgentEventSource.API_CHAT, NOW, operation.processing_sequence)
    _prepared(world.journal, event, world.record_participant(operation))
    world.recovery.commit_internal_candidate(event, world.state.load(), _snapshot(operation.processing_sequence))
    # Crash: no FINALIZE, no terminal.
    coordinator = StartupReconciliationCoordinator(
        world.journal,
        world.recovery,
        world.memory,
        history_store=world.store,
        history_incarnation=lambda: world.incarnation,
    )
    assert coordinator.reconcile_open_transactions()[0] is True
    world.recovery.prepare_startup()  # classifies the event terminal, as at startup
    adopted = SubjectHistoryReader(world.journal, world.store).adopted_records()
    assert adopted.blocked is None and [item.record for item in adopted.records][-1] == operation.record


# --------------------------------------------------- V2 / V3 comparison


def test_as_version_converts_both_ways_when_representable() -> None:
    from test_r16_ip8b_successor_policy import LEDGER, _verified

    v2 = _verified()
    v3 = as_version(v2, PolicyKind.SUCCESSOR_POLICY_V3)
    assert v3 is not None and v3.version is PolicyKind.SUCCESSOR_POLICY_V3
    assert as_version(v3, PolicyKind.SUCCESSOR_POLICY_V2) == v2
    assert as_version(v2, PolicyKind.SUCCESSOR_POLICY_V2) is v2
    prior = replace(
        v3, ledger=LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT, ledger_refs=LEDGER,
        verified_outcome_ref=None, feedback=v2.feedback.__class__.ABSENT, feedback_ref=None,
    )
    assert as_version(prior, PolicyKind.SUCCESSOR_POLICY_V2) is None
    with pytest.raises(ValueError):
        as_version(v2, PolicyKind.SUCCESSOR_POLICY_V1)


def test_a_v2_head_stays_existing_when_the_issuer_seals_v3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D1 r3 §8 (a): the switch in IP-14a-3 keeps unchanged lineages EXISTING.
    The reverse leg (a V3 head under a rolled-back V2 issuer) is IP-14a-3's
    acceptance (D1 §10); ``as_version`` covers its conversion here."""

    from fastapi.testclient import TestClient

    from suzka.r16 import decision_issuer as issuer_module
    from suzka.r16.decision_issuer import DecisionIssueStatus

    from test_r16_ip10_decision_issuer import _app, _issue, _settings
    from test_r16_u3a_plan_preview import _qualified_goal

    settings = _settings(tmp_path / "v3")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        first = _issue(state, gid)
        assert first.status is DecisionIssueStatus.ISSUED
        assert first.decision.policy_kind is PolicyKind.SUCCESSOR_POLICY_V2
        monkeypatch.setattr(issuer_module, "POLICY_KIND", PolicyKind.SUCCESSOR_POLICY_V3)
        again = _issue(state, gid)
        assert again.status is DecisionIssueStatus.EXISTING
        assert again.decision.decision_id == first.decision.decision_id


# ------------------------------------------------------------- C10 attested


def test_attested_records_stay_adopted_and_tampering_diverges(tmp_path: Path) -> None:
    """D1 r3 §6: after a C10 COMPACTION archives their R07 facts, records at or
    below the high-water are checked internally (bytes against the envelope)."""

    from test_r16_ip9_c10_history_attestation import _chain, _compact, _provider, _small

    world = _RecordWorld(tmp_path)
    _small(world)
    chain = world.chain()
    for operation in _steps(chain)[:3]:
        world.commit_record(operation)
    _chain(world, 4, prefix="filler", parent=chain.decision)
    outcome = _compact(world, _provider(world))
    assert outcome.performed, outcome.reason
    reader = SubjectHistoryReader(world.journal, world.store)
    adopted = reader.adopted_records()
    assert adopted.blocked is None
    assert [item.operation_kind for item in adopted.records] == [
        K.R18_INTENT_ISSUE, K.R17_EXPECTATION_COMMIT, K.R18_POLICY_COMMIT,
    ]
    high_water = reader.classify().attested_high_water
    assert all(item.processing_sequence <= high_water for item in adopted.records)  # attested path
    connection = sqlite3.connect(world.store.path, isolation_level=None)
    connection.execute("DROP TRIGGER action_forecasts_no_update")
    connection.execute("UPDATE action_forecasts SET record_digest = ?", ("1" * 64,))
    connection.close()
    assert SubjectHistoryReader(world.journal, world.store).adopted_records().blocked is HeadBlockReason.DIVERGED
