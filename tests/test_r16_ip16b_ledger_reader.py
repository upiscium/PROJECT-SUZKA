"""IP-16b: the Decision issuer's ledger reader and the switched-off startup
(#411 D1 r3 §5, §6, §7 "Ledger" / "Switching off", §10).

The IP-17+ records (verdicts, challenges, expiries, Claims, Outcomes,
inverse records) cannot be produced by a production writer yet: here they
are FIXTURES committed through the real R07 runtime and the subject.history
record participant, so the reader adopts them exactly as it will adopt the
real ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
import sqlite3
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16 import decision_issuer as decision_module
from suzka.r16.decision_issuer import DecisionIssueStatus, _DecisionIssueResult
from suzka.r16.history_participant import (
    JournalCommitProofSource,
    SubjectHistoryRecordParticipant,
)
from suzka.r16.history_reader import HeadStatus, SubjectHistoryReader
from suzka.r16.history_records import (
    MAX_CHALLENGES_PER_INTENT,
    ApprovalExpirationV1,
    ExpirationReason,
    HistoryOperationKind as K,
    HistoryRecordOperationV1,
    RecordKind as R,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreRecordsDisabled
from suzka.r16.successor_policy import (
    CLOSED_UNCLAIMED_LEDGER,
    MAX_LEDGER_REFS,
    CommitmentState,
    FeedbackState,
    Interpretation,
    InterpretationStage,
    InverseAuthorization,
    LedgerState,
    LineageRef,
    LineageState,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorRow,
    evaluate,
)
from suzka.r17.contracts import ExpectedObservationV1, ObservedState
from suzka.r17.expectation_issuer import ExpectationCommitStatus, R17ExpectationReason
from suzka.r18 import ledger as ledger_module
from suzka.r18 import obligation as obligation_module
from suzka.r18.contracts import (
    OUTCOME_REF_KIND,
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    InverseIntentV1,
    PolicyDenyReason,
    PolicyVerdict,
    PolicyVerdictV1,
    VerifiedOutcomeV1,
    VerifiedState,
    operation_key,
)
from suzka.r18.intent_issuer import IntentIssueStatus, R18IntentReason
from suzka.r18.ledger import (
    CHALLENGE_REF_KIND,
    CLAIM_REF_KIND,
    EXPIRY_REF_KIND,
    INTENT_REF_KIND,
    NO_LEDGER,
    PRIOR_CLAIM_REF_KIND,
    VERDICT_REF_KIND,
    LedgerRead,
    RecordAt,
    derive_ledger,
)
from suzka.r18.obligation import SECRET_FILE, ObligationSourceUnavailable
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeDurabilityError,
)
from suzka.runtime.event_journal import TransactionKind
from suzka.runtime.transaction_coordinator import CoordinatedResult

import test_r16_ip10_decision_issuer as ip10
import test_r17_ip16a3_expectation_issuer as ip16a3
import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_ip14a_history_v2 import Chain, _apply, _prepare, _steps
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_commitment, _qualified_goal

_settings = ip16a2._settings
_row4 = ip16a2._row4
_intent = ip16a2._intent
_issue = ip10._issue
_no_failed_events = ip16a2._no_failed_events
OTHER = "2" * 64
BOOT = "00000000-0000-4000-8000-0000000000bb"
RUNTIME = "00000000-0000-4000-8000-0000000000cc"
H = "a" * 64
# IP-17a: the pure derivation's clock, before the fixture window end.
NOW = datetime(2026, 11, 16, 0, 30, tzinfo=timezone.utc)


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip16b"))


# ------------------------------------------------------------------ helpers


class _Seen:
    """Every SuccessorPolicy input the issuer evaluates (the sealed witness)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.inputs: list[SuccessorPolicyInputV2] = []
        real = decision_module.evaluate

        def spy(value: SuccessorPolicyInputV2):  # type: ignore[no-untyped-def]
            self.inputs.append(value)
            return real(value)

        monkeypatch.setattr(decision_module, "evaluate", spy)

    @property
    def last(self) -> SuccessorPolicyInputV2:
        return self.inputs[-1]


def _preview(state, gid, seen: _Seen | None = None):  # type: ignore[no-untyped-def]
    if seen is not None:
        seen.inputs.clear()
    return _issue(state, gid)


def _incarnation(state) -> str:  # type: ignore[no-untyped-def]
    return ip10._incarnation(state)


def _commit_record(state, build: Callable[[Any], HistoryRecordOperationV1]) -> HistoryRecordOperationV1:  # type: ignore[no-untyped-def]
    """Commit one FIXTURE record through the real R07 runtime and the
    subject.history record participant (IP-17+ writers do not exist yet)."""

    runtime = state.agent_runtime
    built: list[HistoryRecordOperationV1] = []

    def handle() -> object:
        event = runtime.current_event()
        operation = build(event)
        built.append(operation)
        participant = SubjectHistoryRecordParticipant(
            state.subject_history_store,
            operation.incarnation_id,
            operation,
            JournalCommitProofSource(state.event_journal, state.state_recovery),
        )
        return CoordinatedResult("fixture_record", (participant,))

    outcome = runtime.submit(
        AgentEventType.R18_ACTION_INTENT_ISSUE,
        AgentEventSource.RUNTIME_R18_ACTION_INTENT_ISSUE,
        handle,
    ).result(timeout=30)
    assert outcome.value == "fixture_record"
    return built[0]


def _operation(state, event, kind, record_kind, record, attempt=None):  # type: ignore[no-untyped-def]
    return HistoryRecordOperationV1(
        kind,
        _incarnation(state),
        event.event_id,
        event.processing_sequence,
        TransactionKind.EVENT_MUTATION.value,
        record_kind,
        record,
        attempt,
    )


def _verdict(state, intent, verdict=PolicyVerdict.ALLOW, *, direction=Direction.FORWARD, key=None) -> PolicyVerdictV1:  # type: ignore[no-untyped-def]
    reason = None if verdict is PolicyVerdict.ALLOW else PolicyDenyReason.SOURCE_NOT_CURRENT
    key = intent.forward_key if key is None else key

    def build(event):  # type: ignore[no-untyped-def]
        record = PolicyVerdictV1(
            direction, intent.intent_id, intent.intent_digest, key, verdict, reason,
            event.event_id, event.processing_sequence,
        )
        return _operation(state, event, K.R18_POLICY_COMMIT, R.POLICY_VERDICT, record)

    return _commit_record(state, build).record  # type: ignore[return-value]


def _challenge(state, intent, verdict, attempt, forecast=None) -> ApprovalChallengeV1:  # type: ignore[no-untyped-def]
    nonce = f"{attempt:x}".rjust(64, "d" if forecast is not None else "7")
    if forecast is not None:
        record = ApprovalChallengeV1.for_forward(
            intent, forecast.forecast_digest, verdict, nonce=nonce,
            issued_utc="2026-11-16T00:00:00Z", expires_utc="2026-11-16T00:10:00Z", boot_id=BOOT,
        )
    else:
        record = ApprovalChallengeV1.for_inverse(
            intent, verdict, nonce=nonce,
            issued_utc="2026-11-16T01:00:00Z", expires_utc="2026-11-16T01:10:00Z", boot_id=BOOT,
        )
    _commit_record(
        state,
        lambda event: _operation(state, event, K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, record, attempt),
    )
    return record


def _expire(state, challenge) -> ApprovalExpirationV1:  # type: ignore[no-untyped-def]
    record = ApprovalExpirationV1.for_challenge(challenge, ExpirationReason.EXPIRED)
    _commit_record(
        state,
        lambda event: _operation(state, event, K.R18_APPROVAL_EXPIRE, R.APPROVAL_EXPIRATION, record),
    )
    return record


def _claim(state, challenge, claimed_utc="2026-11-16T00:05:00Z") -> ExecutionClaimV1:  # type: ignore[no-untyped-def]
    def build(event):  # type: ignore[no-untyped-def]
        record = ExecutionClaimV1.claim(
            challenge, runtime_instance_id=RUNTIME, boot_id=BOOT,
            claim_event_id=event.event_id, claim_sequence=event.processing_sequence,
            claimed_utc=claimed_utc,
        )
        return _operation(state, event, K.R18_EXECUTION_CLAIM, R.EXECUTION_CLAIM, record)

    return _commit_record(state, build).record  # type: ignore[return-value]


def _outcome_record(state, key, outcome_state, *, payload, effect_key):  # type: ignore[no-untyped-def]
    authorization = {
        VerifiedState.ACTIVE_VERIFIED_AUTHORIZED: InverseAuthorization.AUTHORIZED,
        VerifiedState.UNAUTHORIZED_EFFECT: InverseAuthorization.UNPROVEN,
        VerifiedState.REVERTED: InverseAuthorization.AUTHORIZED,
    }.get(outcome_state)
    active = outcome_state in (
        VerifiedState.ACTIVE_VERIFIED_AUTHORIZED,
        VerifiedState.UNAUTHORIZED_EFFECT,
        VerifiedState.INVERSE_VERIFIED_ABSENT,
    )

    def build(event):  # type: ignore[no-untyped-def]
        record = VerifiedOutcomeV1(
            operation_key=key,
            direction=Direction.INVERSE
            if outcome_state in (VerifiedState.REVERTED, VerifiedState.INVERSE_VERIFIED_ABSENT)
            else Direction.FORWARD,
            outcome_revision=1,
            outcome_state=outcome_state,
            authorization=authorization,
            target_instance_id=state._ip16b_target,
            observed_state=ObservedState.ACTIVE if active else ObservedState.ABSENT,
            observed_payload_digest=payload if active else None,
            observed_revision=1,
            effect_operation_key=effect_key if active else None,
            foreign_change=False,
            verifier_event_id=event.event_id,
            verifier_sequence=event.processing_sequence,
        )
        return _operation(state, event, K.R18_OUTCOME_COMMIT, R.VERIFIED_OUTCOME, record)

    return _commit_record(state, build).record


def _forward(state):  # type: ignore[no-untyped-def]
    """A real row-4 Decision, its real Intent and real forecast."""

    _x, gid, decision = _row4(state)
    issued = _intent(state, decision.lineage_scope)
    assert issued.status is IntentIssueStatus.ISSUED
    forecast = ip16a3._forecast(state, issued.intent_id)
    assert forecast.status is ExpectationCommitStatus.COMMITTED
    state._ip16b_target = issued.intent.target_instance_id
    return gid, decision, issued.intent, forecast.forecast


def _refs(value: SuccessorPolicyInputV2) -> list[tuple[str, str]]:
    return [(ref.kind, ref.ref_id) for ref in value.ledger_refs]


def _expected_group(state) -> str:  # type: ignore[no-untyped-def]
    return ip16a2._expected_group(state)


# ----------------------------------------------- no R18 data: production path


@pytest.mark.parametrize("r18", (False, True))
def test_no_r18_data_is_byte_identical_and_computes_no_key(tmp_path, monkeypatch, r18) -> None:  # type: ignore[no-untyped-def]
    """§5 / §10-1 (r3 P2-2): with no R17 / R18 record and no live claim key,
    the sealed policy input, the result, the rows and the logs are exactly
    those the pre-IP-16b issuer produced (ledger NONE, no refs, Feedback
    ABSENT, no Outcome) and no obligation secret or episode text is read.

    Proof by twin evaluation inside the real handler: every qualified issue
    is run once with the pre-IP-16b witness spelled literally and once for
    real, and both must agree byte for byte (Decision bytes, sealed input,
    row, status, log lines)."""

    reads: list[str] = []

    def forbidden(name: str) -> Callable[..., Any]:
        def read(*args: Any, **kwargs: Any) -> Any:
            reads.append(name)
            raise AssertionError(f"{name} read without R18 data")

        return read

    monkeypatch.setattr(obligation_module, "load_obligation_secret", forbidden("secret"))
    monkeypatch.setattr(obligation_module, "request_text", forbidden("episode"))
    monkeypatch.setattr(decision_module, "identity_for_source", forbidden("identity"))
    pre_ip16b = LedgerRead(LedgerState.NONE, (), None, None)
    lines: list[str] = []

    class _Lines(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    handler = _Lines()
    decision_module.LOGGER.addHandler(handler)
    decision_module.LOGGER.setLevel(logging.INFO)
    twins: list[tuple[object, object]] = []

    def summary(result: object) -> tuple[object, ...]:
        value = result.value if type(result) is CoordinatedResult else result
        assert type(value) is _DecisionIssueResult
        decision = value.decision
        return (
            value.status,
            value.row,
            value.hold,
            value.owner_reason,
            value.decision_id,
            None if decision is None else decision.canonical_bytes(),
            None if decision is None else decision.policy_input_bytes,
        )

    try:
        with TestClient(_app(_settings(tmp_path / "plain", r18=r18))) as client:
            state = client.app.state
            issuer = state.decision_issuer
            real = issuer._issue_qualified

            def twin(event, q, initial):  # type: ignore[no-untyped-def]
                lines.clear()
                with monkeypatch.context() as patch:
                    patch.setattr(issuer, "_ledger", lambda *args: pre_ip16b)
                    before = real(event, q, initial)
                before_lines = list(lines)
                lines.clear()
                after = real(event, q, initial)
                assert summary(after) == summary(before)
                assert lines == before_lines
                twins.append((summary(before), summary(after)))
                return after

            monkeypatch.setattr(issuer, "_issue_qualified", twin)
            x, gid = _qualified_goal(state)
            row3 = _issue(state, gid)
            assert row3.row is SuccessorRow.ROW_3
            _qualified_commitment(state, x)
            row4 = _issue(state, gid)
            assert (row4.status, row4.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_4)
            again = _issue(state, gid)
            assert again.status is DecisionIssueStatus.EXISTING
            with monkeypatch.context() as patch:
                patch.setattr(
                    issuer,
                    "_interpretation",
                    lambda *args: SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTRADICTORY"),
                )
                held = _issue(state, gid)
            assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)
            assert len(twins) == 4
            for result in (row3, row4):
                sealed = json.loads(result.decision.policy_input_bytes)
                assert sealed["ledger"] == {"inverse_authorization": None, "refs": [], "state": "NONE"}
                assert sealed["feedback"] == {"ref": None, "state": "ABSENT"}
                assert sealed["verified_outcome"] is None
            assert state.subject_history_store.has_record_operations() is False
            assert reads == []
            _no_failed_events(state)
    finally:
        decision_module.LOGGER.removeHandler(handler)


def test_no_key_without_data_even_with_no_secret_or_target(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Production has no target and no secret: R16 previews are unchanged."""

    with TestClient(_app(_settings(tmp_path / "bare", r18=False, init=False))) as client:
        state = client.app.state
        assert not (state.settings.r18.target_path.parent / SECRET_FILE).exists()
        x, gid = _qualified_goal(state)
        assert _issue(state, gid).row is SuccessorRow.ROW_3
        _qualified_commitment(state, x)
        assert _issue(state, gid).row is SuccessorRow.ROW_4


# ----------------------------------------------------------------- the ledger


def test_reissue_after_an_intent_is_row_2_not_source_drift(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A head sealed with ledger NONE vs INTENT_ISSUED now: a changed witness,
    so the successor path, which the policy holds at row 2 (§5)."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        _x, gid, decision = _row4(state)
        intent = _intent(state, decision.lineage_scope).intent
        before = len(ip10._history_transactions(state))
        held = _preview(state, gid, seen)
        assert (held.status, held.row, held.owner_reason) == (
            DecisionIssueStatus.HOLD,
            SuccessorRow.ROW_2,
            None,
        )
        assert seen.last.ledger is LedgerState.INTENT_ISSUED
        assert seen.last.predecessor is PredecessorState.QUALIFIED
        assert seen.last.ledger_refs == (
            PolicyRef(INTENT_REF_KIND, intent.intent_id, 1, intent.intent_digest),
        )
        assert len(ip10._history_transactions(state)) == before
        _no_failed_events(state)


def test_each_forward_and_inverse_state_maps_to_its_row(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§5 table / §7: every stage of one obligation, from real Intent and
    forecast through FIXTURE verdict, challenges, expiries, Claim, Outcome
    and the inverse chain, read by the real issuer."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        gid, _decision, intent, forecast = _forward(state)

        def expect(ledger: LedgerState, row: SuccessorRow, refs: list[tuple[str, str]]) -> _DecisionIssueResult:
            result = _preview(state, gid, seen)
            assert seen.last.ledger is ledger
            assert _refs(seen.last) == sorted(refs)
            assert result.row is row
            return result

        intent_ref = (INTENT_REF_KIND, intent.intent_id)
        expect(LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2, [intent_ref])
        verdict = _verdict(state, intent)
        expect(LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2, [intent_ref, (VERDICT_REF_KIND, verdict.verdict_digest)])
        challenges = []
        for attempt in range(1, MAX_CHALLENGES_PER_INTENT + 1):
            challenge = _challenge(state, intent, verdict, attempt, forecast)
            challenges.append(challenge)
            expect(LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2, [intent_ref, (CHALLENGE_REF_KIND, challenge.challenge_digest)])
            if attempt < MAX_CHALLENGES_PER_INTENT:
                expiry = _expire(state, challenge)
                # An expiry while attempts remain is still INTENT_ISSUED (§5 amendment).
                expect(LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2, [intent_ref, (EXPIRY_REF_KIND, expiry.challenge_digest)])
        claim = _claim(state, challenges[-1])
        claimed = [intent_ref, (CHALLENGE_REF_KIND, challenges[-1].challenge_digest), (CLAIM_REF_KIND, claim.claim_id)]
        expect(LedgerState.CLAIMED, SuccessorRow.ROW_2, claimed)
        outcome = _outcome_record(
            state, intent.forward_key, VerifiedState.ACTIVE_VERIFIED_AUTHORIZED,
            payload=intent.payload_digest, effect_key=intent.forward_key,
        )
        # ACTIVE_VERIFIED_AUTHORIZED is not row 0: the Outcome is only the
        # verified_outcome_ref, never a ledger ref (r2 P1-2) -> row 6 Plan A.
        verified = expect(LedgerState.ACTIVE_VERIFIED_AUTHORIZED, SuccessorRow.ROW_6, claimed)
        assert verified.status is DecisionIssueStatus.ISSUED
        assert seen.last.verified_outcome_ref == PolicyRef(
            OUTCOME_REF_KIND, outcome.verified_outcome_id, 1, outcome.outcome_digest
        )
        assert verified.decision.decision.selected.candidate_id == "candidate-a"
        assert _preview(state, gid, seen).status is DecisionIssueStatus.EXISTING
        # The inverse chain (operator-initiated; a separate authority).
        inverse = InverseIntentV1(
            incarnation_id=_incarnation(state),
            admitting_event_id="00000000-0000-4000-8000-000000000000",
            admitting_sequence=1,
            obligation_group_key=intent.obligation_group_key,
            target_instance_id=intent.target_instance_id,
            target_namespace=intent.target_namespace,
            target_expected_revision=1,
            forward_outcome_id=outcome.verified_outcome_id,
            forward_outcome_revision=1,
            forward_outcome_digest=outcome.outcome_digest,
            forward_outcome_state=VerifiedState.ACTIVE_VERIFIED_AUTHORIZED,
            forward_payload_digest=intent.payload_digest,
        )

        def build_inverse(event):  # type: ignore[no-untyped-def]
            record = replace(inverse, admitting_event_id=event.event_id, admitting_sequence=event.processing_sequence)
            return _operation(state, event, K.R18_INTENT_ISSUE, R.INVERSE_INTENT, record)

        inverse = _commit_record(state, build_inverse).record
        claim_ref = (CLAIM_REF_KIND, claim.claim_id)
        inverse_ref = (INTENT_REF_KIND, inverse.intent_id)
        expect(LedgerState.INVERSE_INTENT_ISSUED, SuccessorRow.ROW_2, [claim_ref, inverse_ref])
        inverse_verdict = _verdict(state, inverse, direction=Direction.INVERSE, key=inverse.inverse_key)
        inverse_challenge = _challenge(state, inverse, inverse_verdict, 1)
        expect(
            LedgerState.INVERSE_INTENT_ISSUED,
            SuccessorRow.ROW_2,
            [claim_ref, inverse_ref, (CHALLENGE_REF_KIND, inverse_challenge.challenge_digest)],
        )
        inverse_claim = _claim(state, inverse_challenge, "2026-11-16T01:05:00Z")
        full = [
            claim_ref,
            inverse_ref,
            (CHALLENGE_REF_KIND, inverse_challenge.challenge_digest),
            (CLAIM_REF_KIND, inverse_claim.claim_id),
        ]
        expect(LedgerState.INVERSE_CLAIMED, SuccessorRow.ROW_2, full)
        assert len(seen.last.ledger_refs) == MAX_LEDGER_REFS
        _outcome_record(
            state, inverse.inverse_key, VerifiedState.REVERTED, payload=None, effect_key=None
        )
        expect(LedgerState.REVERTED, SuccessorRow.ROW_2, full)
        assert seen.last.inverse_authorization is InverseAuthorization.AUTHORIZED
        assert seen.last.verified_outcome_ref is not None
        assert seen.last.verified_outcome_ref.kind == OUTCOME_REF_KIND
        _no_failed_events(state)


def test_a_denied_intent_is_policy_denied_row_2c(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """IP-17a (§12-2): POLICY_DENIED closed the obligation; V3 row 2c."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        gid, _decision, intent, _forecast = _forward(state)
        verdict = _verdict(state, intent, PolicyVerdict.DENY)
        issued = _preview(state, gid, seen)
        assert (issued.status, issued.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_2C)
        assert seen.last.ledger is LedgerState.POLICY_DENIED
        assert _refs(seen.last) == sorted(
            [(INTENT_REF_KIND, intent.intent_id), (VERDICT_REF_KIND, verdict.verdict_digest)]
        )


def test_a_second_identical_request_is_row_0_and_issues_nothing(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§5 / §7 (r3 P1): a new X with the same text is a new lineage with no
    head; with a non-NONE ledger the predecessor contradicts it -> row 0
    CONFLICTING_EVIDENCE, before and after its Commitment."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        _x, _gid, decision = _row4(state)
        assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.ISSUED
        before = len(ip10._history_transactions(state))
        gid = ip16a2._second_goal(state)
        held = _preview(state, gid, seen)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_0)
        assert held.hold.reason.value == "conflicting_evidence"
        assert seen.last.predecessor is PredecessorState.ABSENT_INITIAL
        assert seen.last.ledger is LedgerState.INTENT_ISSUED
        _qualified_commitment(state, state._ip16_second_x)
        assert _preview(state, gid, seen).row is SuccessorRow.ROW_0
        assert len(ip10._history_transactions(state)) == before
        _no_failed_events(state)


def test_a_different_request_is_another_obligation(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§8-1 (a): the ledger is the group's; other text is another group."""

    from suzka.r16.goal_owner import _GoalSourceSelection
    from test_r16_u2b_goal_owner import chat, x_id

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        _x, _gid, decision = _row4(state)
        assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.ISSUED
        x = chat(state, "a different source experience")
        assert state._motivation_source_owner._submit(x).result(timeout=30).value == "interest_applied"
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        owner = state._goal_source_owner
        proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(timeout=30).value
        owner._submit_review(proposed.subject_id).result(timeout=30)
        result = _preview(state, proposed.subject_id, seen)
        assert result.row is SuccessorRow.ROW_3
        assert seen.last.ledger is LedgerState.NONE


def test_an_unavailable_key_is_row_1(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§5: with R18 data, a key that cannot be computed (no secret, an
    unavailable episode, no target configured) is the row-1 HOLD."""

    with TestClient(app) as client:
        state = client.app.state
        _x, gid, decision = _row4(state)
        assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.ISSUED
        secret = state.settings.r18.target_path.parent / SECRET_FILE
        data = secret.read_bytes()
        secret.unlink()
        held = _issue(state, gid)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_1)
        fd = os.open(secret, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, data)
        os.close(fd)
        assert _issue(state, gid).row is SuccessorRow.ROW_2

        def unavailable(*args: Any, **kwargs: Any) -> Any:
            raise ObligationSourceUnavailable("archived")

        with monkeypatch.context() as patch:
            patch.setattr(decision_module, "identity_for_source", unavailable)
            assert _issue(state, gid).row is SuccessorRow.ROW_1
        with monkeypatch.context() as patch:
            patch.setattr(state.decision_issuer, "_target_path", None)
            assert _issue(state, gid).row is SuccessorRow.ROW_1
        _no_failed_events(state)


def test_an_archived_episode_is_row_1(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _x, gid, decision = _row4(state)
        assert _intent(state, decision.lineage_scope).status is IntentIssueStatus.ISSUED
        source = state._plan_preview_owner._source
        record = source._experience.load_committed_current(
            [ref for ref in decision.decision.source_refs if ref.kind == "experience"][0].ref_id
        ).record
        real = source._memory.get_committed_episodic

        def archived(episode_id):  # type: ignore[no-untyped-def]
            committed = real(episode_id)
            if committed is None or episode_id != record.source_episode_id:
                return committed
            return replace(committed, record=replace(committed.record, archived=True))

        source._memory.get_committed_episodic = archived
        try:
            assert _issue(state, gid).row is SuccessorRow.ROW_1
        finally:
            del source._memory.get_committed_episodic
        assert _issue(state, gid).row is SuccessorRow.ROW_2


# ------------------------------------------------------ another incarnation


def _other_chain(state, decision, *, upto: int, base: int = 9000) -> Chain:  # type: ignore[no-untyped-def]
    """Another incarnation's records of this obligation, written straight to
    the store (the reader never reads another incarnation's rows)."""

    chain = Chain(decision, _expected_group(state), incarnation=OTHER, base=base)
    for operation in _steps(chain)[:upto]:
        _apply(state.subject_history_store, operation)
    return chain


def test_another_incarnations_claim_is_claimed_row_2_with_a_head(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        _x, gid, decision = _row4(state)
        chain = _other_chain(state, decision.decision, upto=5)
        held = _preview(state, gid, seen)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2)
        assert seen.last.ledger is LedgerState.CLAIMED
        assert seen.last.ledger_refs == (
            PolicyRef(PRIOR_CLAIM_REF_KIND, chain.claim.claim_id, 1, chain.claim.claim_digest),
        )
        assert seen.last.verified_outcome_ref is None
        # The Intent issuer refuses the obligation too (its claim key is live).
        assert _intent(state, decision.lineage_scope).reason is R18IntentReason.PRIOR_OBLIGATION
        _no_failed_events(state)


def test_another_incarnations_claim_is_row_0_for_a_new_lineage(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        x, gid = _qualified_goal(state)
        assert _issue(state, gid).row is SuccessorRow.ROW_3
        _qualified_commitment(state, x)
        decision = _issue(state, gid)
        assert decision.row is SuccessorRow.ROW_4
        _other_chain(state, decision.decision, upto=5)
        gid2 = ip16a2._second_goal(state)
        held = _preview(state, gid2, seen)
        assert held.row is SuccessorRow.ROW_0
        assert seen.last.ledger is LedgerState.CLAIMED
        assert [ref.kind for ref in seen.last.ledger_refs] == [PRIOR_CLAIM_REF_KIND]


def test_another_incarnations_unclaimed_intent_does_not_block(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A1-5 / §9: an unclaimed abandoned Intent is not a blanket denial."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        _x, gid, row4 = _row4(state)
        _other_chain(state, row4.decision, upto=3)  # Intent, forecast, ALLOW
        assert state.subject_history_store.has_record_operations()
        again = _preview(state, gid, seen)
        assert (again.status, again.decision_id) == (DecisionIssueStatus.EXISTING, row4.decision_id)
        assert seen.last.ledger is LedgerState.NONE
        assert _intent(state, row4.lineage_scope).status is IntentIssueStatus.ISSUED


def test_a_missing_or_mismatched_prior_claim_row_is_corrupt(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§10-2: the live claim key's raw row is missing (PREPARED only) or its
    digest does not match -> CORRUPT (row 0), not CLAIMED."""

    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        _x, gid, decision = _row4(state)
        chain = Chain(decision.decision, _expected_group(state), incarnation=OTHER, base=9000)
        steps = _steps(chain)
        store = state.subject_history_store
        for operation in steps[:4]:
            _apply(store, operation)
        _prepare(store, steps[4])  # the Claim key is live, the row is not there
        held = _preview(state, gid, seen)
        assert held.row is SuccessorRow.ROW_0
        assert seen.last.ledger is LedgerState.CORRUPT
        assert seen.last.ledger_refs == ()


def test_a_tampered_prior_claim_digest_is_corrupt(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        seen = _Seen(monkeypatch)
        _x, gid, decision = _row4(state)
        _other_chain(state, decision.decision, upto=5)
        connection = sqlite3.connect(state.subject_history_store.path, isolation_level=None)
        try:
            connection.execute("DROP TRIGGER execution_claims_no_update")
            connection.execute("UPDATE execution_claims SET record_digest = ?", ("1" * 64,))
        finally:
            connection.close()
        assert _preview(state, gid, seen).row is SuccessorRow.ROW_0
        assert seen.last.ledger is LedgerState.CORRUPT


# --------------------------------------------------------- end to end (lab)


def test_end_to_end_chat_to_intent_and_forecast_then_row_2(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§7 end to end (HP-6, lab only): CHAT -> ... -> Decision row 4 ->
    Intent -> forecast (-> IP-17a ALLOW) through the real driver; PLAN_PREVIEW afterwards is
    row 2 and issues nothing."""

    import time

    from test_r16_ip6b_driver import _interest, _settled

    os.environ["SUZKA_P1B_TEST_TOKEN"] = ip16a2.TOKEN
    try:
        with TestClient(_app(_settings(tmp_path / "e2e"))) as client:
            state = client.app.state
            assert client.post("/api/r16/driver/activate", headers=ip16a2.ADMIN).status_code == 200
            x = ip16a2._chat(client, "something wonderful happened today")
            _interest(client, x["experience_id"])
            review = ip16a2._chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}")
            assert _settled(client, review["command_ticket"]["ticket_id"])["status"] == "adopted"
            commit = ip16a2._chat(client, f"COMMIT_REVIEW_EXPERIENCE:1:{x['experience_id']}")
            assert _settled(client, commit["command_ticket"]["ticket_id"])["status"] == "accepted"
            reader = SubjectHistoryReader(state.event_journal, state.subject_history_store)
            # IP-17a: the driver's Policy stage follows the forecast (ALLOW).
            ip16a2._until(lambda: len(reader.adopted_records().records) >= 3, timeout=30)
            time.sleep(0.5)
            records = reader.adopted_records()
            kinds = [item.record_kind for item in records.records]
            assert kinds == [R.ACTION_INTENT, R.ACTION_FORECAST, R.POLICY_VERDICT]
            intent = records.records[0].record
            assert type(intent) is ActionIntentV1
            assert type(records.records[1].record) is ExpectedObservationV1
            head = SubjectHistoryReader(state.event_journal, state.subject_history_store).head(intent.lineage_scope)
            assert head.status is HeadStatus.ADOPTED and head.decision.decision_id == intent.decision_id
            goal = [ref for ref in head.decision.source_refs if ref.kind == "goal"][0].ref_id
            before = len(ip10._history_transactions(state))
            after = _issue(state, goal)
            assert (after.status, after.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2)
            assert len(ip10._history_transactions(state)) == before
            _no_failed_events(state)
    finally:
        del os.environ["SUZKA_P1B_TEST_TOKEN"]


# -------------------------------------------------------------- switching off


def _switched_off(settings):  # type: ignore[no-untyped-def]
    return settings.model_copy(update={"r18": settings.r18.model_copy(update={"enabled": False})})


def test_startup_with_records_and_r18_disabled_starts_switched_off(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    """§6: starts, logs R18_RECORDS_PRESENT_BUT_DISABLED once, makes nothing
    new (issuers HOLD, the store refuses PREPARE), and the ledger reader keeps
    row 4 from being issued again."""

    settings = _settings(tmp_path / "off")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, gid, decision = _row4(state)
        issued = _intent(state, decision.lineage_scope)
        assert issued.status is IntentIssueStatus.ISSUED
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="suzka.r16.history_adoption"):
        with TestClient(_app(_switched_off(settings))) as client:
            state = client.app.state
            codes = [r.getMessage() for r in caplog.records if r.name == "suzka.r16.history_adoption"]
            assert codes == ["R18_RECORDS_PRESENT_BUT_DISABLED"]
            assert not state.intent_issuer.enabled and state.r16_driver._intent is None
            held = _intent(state, decision.lineage_scope)
            assert (held.status, held.reason) == (IntentIssueStatus.HOLD, R18IntentReason.R18_DISABLED)
            forecast = ip16a3._scan(state)
            assert (forecast.status, forecast.reason) == (
                ExpectationCommitStatus.HOLD,
                R17ExpectationReason.R18_DISABLED,
            )
            with pytest.raises(HistoryStoreRecordsDisabled):
                state.subject_history_store.insert_prepared_record(
                    transaction_id="00000000-0000-4000-8000-0000000000ff",
                    operation_digest="0" * 64,
                    operation=_steps(Chain(decision.decision, incarnation=_incarnation(state), base=9000))[0],
                )
            preview = _issue(state, gid)
            assert (preview.status, preview.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2)
            rows = state.subject_history_store.operations(_incarnation(state))
            assert [row.operation_kind for row in rows].count(K.R18_INTENT_ISSUE) == 1
            _no_failed_events(state)
            codes = [r.getMessage() for r in caplog.records if r.name == "suzka.r16.history_adoption"]
            assert codes == ["R18_RECORDS_PRESENT_BUT_DISABLED"]


@pytest.mark.parametrize("crash", ("after_prepare", "after_internal_commit"))
def test_a_pending_record_settles_while_switched_off(tmp_path, crash) -> None:  # type: ignore[no-untyped-def]
    """§6 (r2 P3-13): a pending Intent left by a crash settles through R07's
    startup reconcile with r18.enabled false (FINALIZE is not gated)."""

    settings = _settings(tmp_path / crash)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, gid, decision = _row4(state)
        if crash == "after_prepare":
            def fail(event):  # type: ignore[no-untyped-def]
                if event.event_type is AgentEventType.R18_ACTION_INTENT_ISSUE:
                    raise RuntimeError("simulated pre-WAL failure")

            state.intent_issuer._validate_commit = fail  # type: ignore[method-assign]
            with pytest.raises(AgentRuntimeDurabilityError):
                _intent(state, decision.lineage_scope)
        else:
            real = SubjectHistoryRecordParticipant.finalize

            def crash_finalize(self, binding):  # type: ignore[no-untyped-def]
                raise RuntimeError("simulated post-R06 failure")

            SubjectHistoryRecordParticipant.finalize = crash_finalize  # type: ignore[method-assign]
            try:
                with pytest.raises(AgentRuntimeDurabilityError):
                    _intent(state, decision.lineage_scope)
            finally:
                SubjectHistoryRecordParticipant.finalize = real  # type: ignore[method-assign]
        rows = ip16a2._intent_rows(state)
        assert [row.phase for row in rows] == [HistoryPhase.PREPARED]
    with TestClient(_app(_switched_off(settings))) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        phases = [row.phase for row in ip16a2._intent_rows(state)]
        expected = HistoryPhase.ABORTED if crash == "after_prepare" else HistoryPhase.FINALIZED
        assert phases == [expected]
        result = _issue(state, gid)
        if crash == "after_prepare":
            # The tombstoned Intent never existed: the head stays EXISTING.
            assert result.status is DecisionIssueStatus.EXISTING
        else:
            # The settled Intent is adopted: the ledger reader holds row 2.
            assert (result.status, result.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2)
        assert not state.subject_history_store.live_keys(
            (f"claim:{operation_key(_expected_group(state), Direction.FORWARD)}",)
        )


# ------------------------------------------------------------ pure derivation


def _fixture_chain() -> Chain:
    from test_r16_ip9_subject_history import _committed, _event

    from suzka.r16.history_codec import PolicyRow

    return Chain(_committed(_event("ledger", 3), row=PolicyRow.ROW_4))


def _at(*records: object) -> list[RecordAt]:
    return [RecordAt(record, index + 1) for index, record in enumerate(records)]


def _inverse_for(chain: Chain, **changes: Any) -> InverseIntentV1:
    values: dict[str, Any] = {
        "incarnation_id": chain.incarnation,
        "admitting_event_id": chain.event("inverse"),
        "admitting_sequence": chain.seq(9),
        "obligation_group_key": chain.intent.obligation_group_key,
        "target_instance_id": chain.intent.target_instance_id,
        "target_namespace": "review-artifact",
        "target_expected_revision": 1,
        "forward_outcome_id": chain.outcome.verified_outcome_id,
        "forward_outcome_revision": 1,
        "forward_outcome_digest": chain.outcome.outcome_digest,
        "forward_outcome_state": chain.outcome.outcome_state,
        "forward_payload_digest": chain.intent.payload_digest,
    }
    values.update(changes)
    return InverseIntentV1(**values)


def _policy(read: LedgerRead, predecessor: PredecessorState = PredecessorState.QUALIFIED) -> SuccessorRow:
    """The policy row of one ledger read, everything else a qualified head."""

    ref = (
        PolicyRef("decision", "d-1", 2, H)
        if predecessor is PredecessorState.QUALIFIED
        else PolicyRef("history-horizon", "x-1", 0, H)
    )
    value = SuccessorPolicyInputV2(
        required_source=RequiredSourceState.QUALIFIED,
        source_refs=tuple(
            PolicyRef(kind, f"{kind}-1", 1, H) for kind in ("context", "experience", "goal", "value")
        ),
        predecessor=predecessor,
        predecessor_ref=ref,
        lineage=LineageState.CONSISTENT,
        lineage_ref=LineageRef("lineage-1", "baseline-1", H, None, H),
        commitment=CommitmentState.ACTIVE,
        commitment_ref=PolicyRef("commitment", "c-1", 1, H),
        commitment_deadline_utc="9999-12-31T23:59:59Z",
        ledger=read.state,
        ledger_refs=read.refs,
        inverse_authorization=read.inverse_authorization,
        feedback=FeedbackState.ABSENT,
        feedback_ref=None,
        verified_outcome_ref=read.verified_outcome_ref,
        interpretation=Interpretation.SETTLED,
        interpretation_stage=InterpretationStage.S1,
        as_of_r07_sequence=1,
        as_of_snapshot_hash=H,
        event_id="event-1",
        event_sequence=2,
        now_utc="2026-11-17T00:00:00Z",
        version=decision_module.POLICY_KIND,
    )
    return evaluate(value).row


def _outcome_of(chain: Chain, state: VerifiedState) -> VerifiedOutcomeV1:
    from test_r18_ip13_contracts import _outcome

    inverse = state in (VerifiedState.REVERTED, VerifiedState.INVERSE_VERIFIED_ABSENT)
    key = operation_key(chain.intent.obligation_group_key, Direction.INVERSE if inverse else Direction.FORWARD)
    changes: dict[str, Any] = {"operation_key": key, "target_instance_id": chain.intent.target_instance_id}
    if state in (VerifiedState.ACTIVE_VERIFIED_AUTHORIZED, VerifiedState.UNAUTHORIZED_EFFECT):
        changes["effect_operation_key"] = key
    if state is VerifiedState.INVERSE_VERIFIED_ABSENT:
        changes["effect_operation_key"] = chain.intent.forward_key
    return _outcome(state, **changes)


def test_derived_forward_states_and_their_policy_rows() -> None:
    chain = _fixture_chain()
    group = chain.intent.obligation_group_key
    c = chain.challenges
    base = [chain.intent, chain.forecast, chain.verdict]
    expiries = [ApprovalExpirationV1.for_challenge(item, ExpirationReason.EXPIRED) for item in c[:3]]
    cases: list[tuple[list[object], LedgerState, SuccessorRow]] = [
        ([], LedgerState.NONE, SuccessorRow.ROW_4),
        ([chain.intent], LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2),
        (base, LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2),
        ([*base, c[0]], LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2),
        ([*base, c[0], expiries[0]], LedgerState.INTENT_ISSUED, SuccessorRow.ROW_2),
        # IP-17a (§12-2): a closed unclaimed obligation is V3 row 2c.
        (
            [*base, c[0], expiries[0], c[1], expiries[1], c[2], expiries[2]],
            LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
            SuccessorRow.ROW_2C,
        ),
        (
            [chain.intent, chain.forecast, replace(chain.verdict, verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.SOURCE_NOT_CURRENT)],
            LedgerState.POLICY_DENIED,
            SuccessorRow.ROW_2C,
        ),
        ([*base, c[0], chain.claim], LedgerState.CLAIMED, SuccessorRow.ROW_2),
    ]
    for state in (VerifiedState.ACTIVE_VERIFIED_AUTHORIZED, VerifiedState.VERIFIED_ABSENT, VerifiedState.UNAUTHORIZED_EFFECT):
        row = SuccessorRow.ROW_6 if state is VerifiedState.ACTIVE_VERIFIED_AUTHORIZED else SuccessorRow.ROW_2
        cases.append(([*base, c[0], chain.claim, _outcome_of(chain, state)], LedgerState(state.value), row))
    for records, ledger, row in cases:
        read = derive_ledger(_at(*records), group, NOW)
        assert read.state is ledger, (ledger, read)
        assert _policy(read) is row, ledger
        assert list(read.refs) == sorted(read.refs, key=lambda ref: ref.sort_key)
        assert len(read.refs) <= MAX_LEDGER_REFS
        assert all(ref.kind != OUTCOME_REF_KIND for ref in read.refs)
        if ledger in CLOSED_UNCLAIMED_LEDGER:
            # IP-17a (§12-2): a new lineage on a closed group re-decides.
            assert _policy(read, PredecessorState.ABSENT_INITIAL) is SuccessorRow.ROW_2C
        elif ledger is not LedgerState.NONE:
            # A new lineage (no head) contradicts any other non-NONE ledger.
            assert _policy(read, PredecessorState.ABSENT_INITIAL) is SuccessorRow.ROW_0
    # Records of another group are not this ledger.
    assert derive_ledger(_at(chain.intent, chain.verdict), "f" * 64, NOW) == NO_LEDGER


def test_derived_inverse_states_and_their_policy_rows() -> None:
    chain = _fixture_chain()
    group = chain.intent.obligation_group_key
    forward = [chain.intent, chain.forecast, chain.verdict, chain.challenges[0], chain.claim, chain.outcome]
    inverse = _inverse_for(chain)
    verdict = PolicyVerdictV1(
        Direction.INVERSE, inverse.intent_id, inverse.intent_digest, inverse.inverse_key,
        PolicyVerdict.ALLOW, None, chain.event("iv"), chain.seq(11),
    )
    challenge = ApprovalChallengeV1.for_inverse(
        inverse, verdict, nonce="7" * 64, issued_utc="2026-11-16T01:00:00Z",
        expires_utc="2026-11-16T01:10:00Z", boot_id=BOOT,
    )
    claim = ExecutionClaimV1.claim(
        challenge, runtime_instance_id=RUNTIME, boot_id=BOOT, claim_event_id=chain.event("ic"),
        claim_sequence=chain.seq(13), claimed_utc="2026-11-16T01:05:00Z",
    )
    denied = replace(verdict, verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.SOURCE_NOT_CURRENT)
    challenges = [
        replace(challenge, nonce=f"{n:x}".rjust(64, "7")) for n in range(1, MAX_CHALLENGES_PER_INTENT + 1)
    ]
    expired = [
        item
        for ch in challenges
        for item in (ch, ApprovalExpirationV1.for_challenge(ch, ExpirationReason.BOOT_CHANGED))
    ]
    cases = [
        ([inverse], LedgerState.INVERSE_INTENT_ISSUED),
        ([inverse, denied], LedgerState.INVERSE_POLICY_DENIED),
        ([inverse, verdict, *expired], LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED),
        ([inverse, verdict, challenge, claim], LedgerState.INVERSE_CLAIMED),
        ([inverse, verdict, challenge, claim, _outcome_of(chain, VerifiedState.INVERSE_VERIFIED_ABSENT)], LedgerState.INVERSE_VERIFIED_ABSENT),
        ([inverse, verdict, challenge, claim, _outcome_of(chain, VerifiedState.REVERTED)], LedgerState.REVERTED),
    ]
    for extra, ledger in cases:
        read = derive_ledger(_at(*forward, *extra), group, NOW)
        assert read.state is ledger, (ledger, read)
        assert _policy(read) is SuccessorRow.ROW_2
        assert read.verified_outcome_ref is not None
        assert read.verified_outcome_ref.kind == OUTCOME_REF_KIND
        assert len(read.refs) <= MAX_LEDGER_REFS
        assert (CLAIM_REF_KIND, chain.claim.claim_id) in [ref.sort_key for ref in read.refs]
        assert (INTENT_REF_KIND, chain.intent.intent_id) not in [ref.sort_key for ref in read.refs]
    reverted = derive_ledger(_at(*forward, *cases[-1][0]), group, NOW)
    assert reverted.inverse_authorization is InverseAuthorization.AUTHORIZED
    full = derive_ledger(_at(*forward, inverse, verdict, challenge, claim), group, NOW)
    assert len(full.refs) == MAX_LEDGER_REFS
    assert [ref.kind for ref in full.refs] == sorted(ref.kind for ref in full.refs)
    # An inverse after UNAUTHORIZED_EFFECT is the same table.
    unauthorized = _outcome_of(chain, VerifiedState.UNAUTHORIZED_EFFECT)
    after = derive_ledger(
        _at(
            *forward[:-1],
            unauthorized,
            _inverse_for(
                chain,
                forward_outcome_id=unauthorized.verified_outcome_id,
                forward_outcome_digest=unauthorized.outcome_digest,
                forward_outcome_state=VerifiedState.UNAUTHORIZED_EFFECT,
            ),
        ),
        group,
        NOW,
    )
    assert after.state is LedgerState.INVERSE_INTENT_ISSUED


def test_inconsistent_record_sets_are_corrupt() -> None:
    chain = _fixture_chain()
    group = chain.intent.obligation_group_key
    second = replace(chain.intent, admitting_event_id=chain.event("again"), admitting_sequence=chain.seq(50))
    denied = replace(chain.verdict, verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.SOURCE_NOT_CURRENT)
    inverse = _inverse_for(chain)
    corrupt = [
        [chain.intent, second],  # two forward Intents
        [chain.intent, chain.verdict, chain.challenges[0], chain.claim, replace(chain.verdict, policy_sequence=999)],
        [chain.intent, denied, chain.challenges[0]],  # a challenge without ALLOW
        [chain.intent, chain.claim],  # a Claim without ALLOW or its challenge
        [chain.intent, denied, chain.claim],  # a Claim after DENY
        [chain.intent, chain.verdict, chain.challenges[0], chain.outcome],  # Outcome without Claim
        [chain.verdict],  # a verdict with no Intent
        [chain.outcome],  # an Outcome with no Intent
        [chain.intent, ApprovalExpirationV1.for_challenge(chain.challenges[0], ExpirationReason.EXPIRED)],
        [
            chain.intent, chain.verdict, chain.challenges[0],
            ApprovalExpirationV1.for_challenge(chain.challenges[0], ExpirationReason.EXPIRED), chain.claim,
        ],  # a Claim of an expired challenge
        [inverse],  # an inverse with no forward effect
        [chain.intent, chain.verdict, chain.challenges[0], chain.claim, inverse],  # before any Outcome
        [
            chain.intent, chain.verdict, chain.challenges[0], chain.claim,
            _outcome_of(chain, VerifiedState.VERIFIED_ABSENT), inverse,
        ],  # an inverse after VERIFIED_ABSENT
        [
            chain.intent, chain.verdict, chain.challenges[0], chain.claim, chain.outcome,
            _inverse_for(chain, forward_outcome_digest="9" * 64),
        ],  # an inverse of another Outcome
    ]
    for records in corrupt:
        read = derive_ledger(_at(*records), group, NOW)
        assert read == LedgerRead(LedgerState.CORRUPT, (), None, None), records
        assert _policy(read) is SuccessorRow.ROW_0
    # A prior-incarnation Claim beside a forward record of this one.
    prior = PolicyRef(PRIOR_CLAIM_REF_KIND, chain.claim.claim_id, 1, chain.claim.claim_digest)
    assert derive_ledger(_at(chain.intent), group, NOW, prior_claim=prior).state is LedgerState.CORRUPT
    assert derive_ledger((), group, NOW, prior_claim=prior) == LedgerRead(LedgerState.CLAIMED, (prior,), None, None)


def test_prior_claim_ref_checks_the_raw_row() -> None:
    chain = _fixture_chain()
    claim = chain.claim
    row = {
        "record_id": claim.claim_id,
        "record_bytes": claim.canonical_bytes(),
        "record_digest": claim.claim_digest,
        "operation_key": claim.operation_key,
    }
    ref = ledger_module.prior_claim_ref(row, claim.operation_key)
    assert ref == PolicyRef(PRIOR_CLAIM_REF_KIND, claim.claim_id, 1, claim.claim_digest)
    assert ledger_module.prior_claim_ref(None, claim.operation_key) is None
    for broken in (
        {**row, "record_digest": "1" * 64},
        {**row, "record_id": "claim-" + "1" * 64},
        {**row, "operation_key": "1" * 64},
        {**row, "record_bytes": b"{}"},
    ):
        assert ledger_module.prior_claim_ref(broken, claim.operation_key) is None
    assert ledger_module.prior_claim_ref(row, "1" * 64) is None


def test_latest_stage_ref_tie_break_is_expiry_then_challenge_then_verdict() -> None:
    """Review P3-4: on an equal processing sequence the later stage wins."""

    chain = _fixture_chain()
    group = chain.intent.obligation_group_key
    expiry = ApprovalExpirationV1.for_challenge(chain.challenges[0], ExpirationReason.EXPIRED)

    def stage_kinds(*records: object) -> list[str]:
        at = [RecordAt(chain.intent, 1), *(RecordAt(record, 5) for record in records)]
        read = derive_ledger(at, group, NOW)
        return [ref.kind for ref in read.refs if ref.kind != INTENT_REF_KIND]

    for order in ((chain.verdict, chain.challenges[0], expiry), (expiry, chain.challenges[0], chain.verdict)):
        assert stage_kinds(*order) == [EXPIRY_REF_KIND]
    assert stage_kinds(chain.challenges[0], chain.verdict) == [CHALLENGE_REF_KIND]
    assert stage_kinds(chain.verdict) == [VERDICT_REF_KIND]
    # A higher sequence still wins over the stage order.
    later = derive_ledger(
        [RecordAt(chain.intent, 1), RecordAt(chain.challenges[0], 4), RecordAt(chain.verdict, 6)], group, NOW
    )
    assert [ref.kind for ref in later.refs if ref.kind != INTENT_REF_KIND] == [VERDICT_REF_KIND]


def test_has_live_claim_keys_matches_only_live_claim_keys(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Review P3-5: the range predicate keeps the ``claim:`` prefix semantics."""

    from test_r16_ip14a_history_v2 import _decision, _v2_store

    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    steps = _steps(chain)
    for operation in steps[:4]:
        _apply(store, operation)
    assert store.live_keys((f"challenge:{chain.intent.intent_id}:1",))  # other live keys
    assert store.has_live_claim_keys() is False
    transaction_id = _prepare(store, steps[4])
    assert store.has_live_claim_keys() is True
    store.mark_aborted(chain.incarnation, transaction_id)
    assert store.has_live_claim_keys() is False
