"""IP-20c test harness: a real app with R19 on, a hand-materialized Wake, and a
settable clock (IP-20 D1 r3 §9-2).

There is no materializer in IP-20c (IP-20b), so ``materialize`` plays its role
with the real issuer: it evaluates the Goal at the event's own instant and
sequence (``_evaluate``), freezes the fingerprint and the predecessor, and writes
the Wake with the real R07 coordinator and the subject.history record
participant.  Every value is a FIXTURE; no production target is touched.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from suzka.r16.cycle_contracts import FrozenPredecessor
from suzka.r16.history_participant import JournalCommitProofSource, SubjectHistoryRecordParticipant
from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import HistoryOperationKind, HistoryRecordOperationV1, RecordKind
from suzka.r16.history_codec import PolicyKind
from suzka.r16.successor_policy import PolicyRef, as_version
from suzka.r18.contracts import GoalOrigin
from suzka.r19.contracts import WakeCause, WakeKind, WakeV1
from suzka.r19.ledger import WakeEntry, derive_wake_view
from suzka.runtime import agent_runtime
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import TransactionKind
from suzka.runtime.transaction_coordinator import CoordinatedResult

import test_r16_ip10_decision_issuer as ip10

UTC = timezone.utc
STAMP = "%Y-%m-%dT%H:%M:%SZ"
_REAL_DATETIME = datetime


def settings(root: Any, *, r19: bool = True) -> Any:
    base = ip10._settings(root)
    return base.model_copy(update={"r19": base.r19.model_copy(update={"enabled": r19})})


class Time:
    """A forward-only clock for the runtime (``requested_at``, T) and the Wake
    owners' live clock (L).  ``skew`` makes L run ahead of T: the queue wait."""

    def __init__(self, state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.offset = timedelta(0)
        self.skew = timedelta(0)
        outer = self

        class _Datetime(_REAL_DATETIME):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
                return _REAL_DATETIME.now(tz) + outer.offset

        monkeypatch.setattr(agent_runtime, "datetime", _Datetime)
        for owner in (state.wake_claim_owner, state.wake_review_owner):
            owner._clock = self.live
        state.decision_issuer._clock = self.live

    def now(self) -> datetime:
        return _REAL_DATETIME.now(UTC) + self.offset

    def live(self) -> datetime:
        return self.now() + self.skew

    def advance(self, seconds: float) -> None:
        self.offset += timedelta(seconds=seconds)

    def stamp(self, seconds: float = 0) -> str:
        return (self.now() + timedelta(seconds=seconds)).strftime(STAMP)


def incarnation(state: Any) -> str:
    return ip10._incarnation(state)


def operations(state: Any, kind: HistoryOperationKind | None = None) -> list[Any]:
    rows = state.subject_history_store.operations(incarnation(state))
    return [row for row in rows if kind is None or row.operation_kind is kind]


def adopted(state: Any, record_type: type) -> list[Any]:
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return [item.record for item in found.records if type(item.record) is record_type]


def view(state: Any, at: datetime | None = None) -> Any:
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    moment = at or _REAL_DATETIME.now(UTC)
    return derive_wake_view(found.records, moment.strftime(STAMP), clock_floor=None)


def entry(state: Any, wake: WakeV1, at: datetime | None = None) -> WakeEntry:
    found = view(state, at).get(wake.wakeup_id)
    assert found is not None
    return found


def write_record(
    state: Any,
    kind: HistoryOperationKind,
    record_kind: RecordKind,
    build: Any,
    event_type: AgentEventType = AgentEventType.R19_WAKE_MATERIALIZE,
    source: AgentEventSource = AgentEventSource.RUNTIME_R19_WAKE_MATERIALIZE,
) -> Any:
    """Commit one FIXTURE record through the real runtime and participant."""

    runtime = state.agent_runtime
    built: list[Any] = []

    def handle() -> object:
        event = runtime.current_event()
        record = build(event)
        operation = HistoryRecordOperationV1(
            kind,
            incarnation(state),
            event.event_id,
            event.processing_sequence,
            TransactionKind.EVENT_MUTATION.value,
            record_kind,
            record,
        )
        built.append(record)
        participant = SubjectHistoryRecordParticipant(
            state.subject_history_store,
            operation.incarnation_id,
            operation,
            JournalCommitProofSource(state.event_journal, state.state_recovery),
        )
        return CoordinatedResult("fixture_record", (participant,))

    assert runtime.submit(event_type, source, handle).result(timeout=30).value == "fixture_record"
    return built[0]


def materialize(
    state: Any,
    goal_id: str,
    time: Time | None = None,
    *,
    not_before: float = -600,
    expires: float = 7 * 86_400 - 60,
    lineage: str | None = None,
    policy_kind: PolicyKind | None = None,
    **changes: Any,
) -> WakeV1:
    """The materializer's role: evaluate ``goal_id`` at the event's own instant
    and sequence, freeze its fingerprint and head, write one GOAL_ADMITTED Wake.
    ``not_before`` / ``expires`` are seconds from the (fake) now."""

    clock = time.now() if time is not None else _REAL_DATETIME.now(UTC)
    issuer = state.decision_issuer

    def build(event: Any) -> WakeV1:
        evaluated = issuer._evaluate(
            event, goal_id, now=event.requested_at, before_sequence=event.processing_sequence
        )
        value = evaluated.value
        head = evaluated.head
        if head is not None and head.decision is not None:
            frozen = FrozenPredecessor.decision(
                head.decision.decision_id, head.decision.content_digest
            )
        else:
            frozen = FrozenPredecessor.absence(evaluated.as_of)
        goal = state.main_loop._goal_system.get(goal_id)
        admission = goal.subject_admission
        q = evaluated.q
        frozen_value = value if policy_kind is None else as_version(value, policy_kind)
        assert frozen_value is not None
        wake = WakeV1(
            kind=WakeKind.GOAL_REEVALUATION,
            cause=WakeCause.GOAL_ADMITTED,
            cause_event_id=admission.event_id,
            cause_sequence=admission.event_sequence,
            cause_ref=PolicyRef("goal-admission", goal_id, q.goal_revision, admission.admission_digest),
            goal_id=goal_id,
            goal_revision=q.goal_revision,
            goal_digest=q.goal_digest,
            experience_id=q.experience_id,
            goal_origin=GoalOrigin[q.goal_origin.name],
            origin_refs=(),
            evidence_fingerprint=frozen_value.evidence_fingerprint(),
            policy_kind=frozen_value.version,
            frozen_predecessor=frozen,
            horizon_sequence=evaluated.as_of,
            incarnation_id=evaluated.incarnation,
            journal_lineage_id=lineage or value.lineage_ref.journal_lineage_id,
            not_before_utc=(clock + timedelta(seconds=not_before)).strftime(STAMP),
            expires_at_utc=(clock + timedelta(seconds=expires)).strftime(STAMP),
            commitment_deadline_utc=None,
            context_id=q.context_id,
        )
        return replace(wake, **changes)

    return write_record(
        state, HistoryOperationKind.R19_WAKE_MATERIALIZE, RecordKind.WAKE, build
    )


def claim(state: Any, wake: WakeV1) -> Any:
    return state.wake_claim_owner._submit_claim(wake.wakeup_id).result(timeout=30).value


def review(state: Any, claim_id: str) -> Any:
    return state.wake_review_owner._submit_review(claim_id).result(timeout=30).value


def terminal(state: Any, wake: WakeV1, reason: Any, superseding: str | None = None) -> Any:
    return (
        state.wake_claim_owner._submit_terminal(wake.wakeup_id, reason, superseding)
        .result(timeout=30)
        .value
    )
