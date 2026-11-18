"""IP-13: R18 pure governed-Action values and lifecycles (#368, #380 under H3).

Every value built here is a FIXTURE ONLY: nothing is issued, approved,
executed or verified, and no target is touched.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from typing import Any

import pytest

import suzka.r18 as r18
from suzka.r16.history_codec import PolicyKind, PolicyRow
from suzka.r16.successor_policy import InverseAuthorization, LedgerState
from suzka.r17.contracts import ObservedState, OutcomeState
from suzka.r18 import contracts
from suzka.r18.contracts import (
    CONFIRMATION_CODE_LENGTH,
    MAX_APPROVAL_TTL,
    MAX_ARTIFACT_BYTES,
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    ExecutionReceiptV1,
    ExecutorResult,
    GateResult,
    GateStatus,
    GoalOrigin,
    InverseIntentV1,
    PolicyDenyReason,
    PolicyVerdict,
    PolicyVerdictV1,
    ReviewArtifactV1,
    ReviewGate,
    ReviewSourceRef,
    SourceRole,
    VerifiedOutcomeV1,
    VerifiedState,
    allowed_transitions,
    ledger_state_of,
    obligation_group_key,
    operation_key,
    payload_digest,
    transition,
)

H = "a" * 64
EXPECTATION = "e" * 64
TARGET = "00000000-0000-4000-8000-0000000000aa"
BOOT = "00000000-0000-4000-8000-0000000000bb"
RUNTIME = "00000000-0000-4000-8000-0000000000cc"
EVENT_I = "00000000-0000-4000-8000-000000000001"
EVENT_P = "00000000-0000-4000-8000-000000000002"
EVENT_C = "00000000-0000-4000-8000-000000000003"
EVENT_V = "00000000-0000-4000-8000-000000000004"
NONCE = "d" * 64
ISSUED = "2026-11-16T00:00:00Z"
EXPIRES = "2026-11-16T00:10:00Z"
# IP-17a: the forward Intent's approval window end.
WINDOW_END = "2026-11-16T01:00:00Z"
DECISION = "decision-" + "f" * 64
GROUP = obligation_group_key("subject-1", "1" * 64, "review", "review-artifact")


def _sources(**digests: str) -> tuple[ReviewSourceRef, ...]:
    refs = []
    for role in sorted(SourceRole):
        if role is SourceRole.DECISION:
            refs.append(ReviewSourceRef(role, DECISION, 1, H))
        else:
            refs.append(ReviewSourceRef(role, f"{role.value.lower()}-1", 1, digests.get(role.value, H)))
    return tuple(refs)


def _gates(**statuses: GateStatus) -> tuple[GateResult, ...]:
    return tuple(GateResult(gate, statuses.get(gate.value, GateStatus.PASS)) for gate in sorted(ReviewGate))


def _artifact(**statuses: GateStatus) -> ReviewArtifactV1:
    return ReviewArtifactV1("review-artifact", GROUP, _sources(), _gates(**statuses))


def _intent(**changes: Any) -> ActionIntentV1:
    base = ActionIntentV1(
        lineage_scope="x-1",
        incarnation_id=H,
        admitting_event_id=EVENT_I,
        admitting_sequence=20,
        decision_id=DECISION,
        decision_committed_digest=H,
        policy_kind=PolicyKind.SUCCESSOR_POLICY_V2,
        policy_row=PolicyRow.ROW_4,
        selected_candidate_id="candidate-b",
        selected_candidate_digest=H,
        goal_origin=GoalOrigin.USER_REQUEST,
        target_instance_id=TARGET,
        target_expected_revision=0,
        artifact=_artifact(),
        approval_window_end_utc=WINDOW_END,
    )
    return replace(base, **changes)


def _allow(intent: ActionIntentV1 | InverseIntentV1, **changes: Any) -> PolicyVerdictV1:
    forward = type(intent) is ActionIntentV1
    key = intent.forward_key if isinstance(intent, ActionIntentV1) else intent.inverse_key
    base = PolicyVerdictV1(
        Direction.FORWARD if forward else Direction.INVERSE,
        intent.intent_id,
        intent.intent_digest,
        key,
        PolicyVerdict.ALLOW,
        None,
        EVENT_P,
        21,
    )
    return replace(base, **changes)


def _challenge(intent: ActionIntentV1 | None = None, **changes: Any) -> ApprovalChallengeV1:
    intent = intent or _intent()
    base = ApprovalChallengeV1.for_forward(
        intent, EXPECTATION, _allow(intent), nonce=NONCE, issued_utc=ISSUED, expires_utc=EXPIRES, boot_id=BOOT
    )
    return replace(base, **changes)


def _claim(challenge: ApprovalChallengeV1 | None = None) -> ExecutionClaimV1:
    return ExecutionClaimV1.claim(
        challenge or _challenge(),
        runtime_instance_id=RUNTIME,
        boot_id=BOOT,
        claim_event_id=EVENT_C,
        claim_sequence=22,
        claimed_utc="2026-11-16T00:05:00Z",
    )


def _outcome(state: VerifiedState = VerifiedState.ACTIVE_VERIFIED_AUTHORIZED, **changes: Any) -> VerifiedOutcomeV1:
    forward = _intent().forward_key
    by_state: dict[VerifiedState, dict[str, Any]] = {
        VerifiedState.ACTIVE_VERIFIED_AUTHORIZED: {
            "authorization": InverseAuthorization.AUTHORIZED,
            "observed_state": ObservedState.ACTIVE,
            "observed_payload_digest": _intent().payload_digest,
            "effect_operation_key": forward,
        },
        VerifiedState.UNAUTHORIZED_EFFECT: {
            "authorization": InverseAuthorization.UNPROVEN,
            "observed_state": ObservedState.ACTIVE,
            "observed_payload_digest": _intent().payload_digest,
            "effect_operation_key": forward,
        },
        VerifiedState.VERIFIED_ABSENT: {
            "authorization": None,
            "observed_state": ObservedState.ABSENT,
            "observed_payload_digest": None,
            "effect_operation_key": None,
        },
        VerifiedState.REVERTED: {
            "authorization": InverseAuthorization.AUTHORIZED,
            "observed_state": ObservedState.ABSENT,
            "observed_payload_digest": None,
            "effect_operation_key": None,
        },
        VerifiedState.INVERSE_VERIFIED_ABSENT: {
            "authorization": None,
            "observed_state": ObservedState.ACTIVE,
            "observed_payload_digest": _intent().payload_digest,
            "effect_operation_key": forward,
        },
    }
    inverse = state in (VerifiedState.REVERTED, VerifiedState.INVERSE_VERIFIED_ABSENT)
    values: dict[str, Any] = {
        "operation_key": operation_key(GROUP, Direction.INVERSE) if inverse else forward,
        "direction": Direction.INVERSE if inverse else Direction.FORWARD,
        "outcome_revision": 1,
        "outcome_state": state,
        "target_instance_id": TARGET,
        "observed_revision": 1,
        "foreign_change": False,
        "verifier_event_id": EVENT_V,
        "verifier_sequence": 23,
        **by_state[state],
    }
    values.update(changes)
    return VerifiedOutcomeV1(**values)


def _inverse(**changes: Any) -> InverseIntentV1:
    forward = _outcome()
    base = InverseIntentV1(
        incarnation_id=H,
        admitting_event_id=EVENT_I,
        admitting_sequence=30,
        obligation_group_key=GROUP,
        target_instance_id=TARGET,
        target_namespace="review-artifact",
        target_expected_revision=1,
        forward_outcome_id=forward.verified_outcome_id,
        forward_outcome_revision=forward.outcome_revision,
        forward_outcome_digest=forward.outcome_digest,
        forward_outcome_state=forward.outcome_state,
        forward_payload_digest=_intent().payload_digest,
    )
    return replace(base, **changes)


def _tamper(payload: bytes, **changes: object) -> bytes:
    row = json.loads(payload)
    row.update(changes)
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


# --------------------------------------------------------------------- keys


def test_obligation_group_key_uses_only_incarnation_independent_content() -> None:
    """#373 A1-5 / A2-4: no incarnation, receipt, revision or event input."""

    params = inspect.signature(obligation_group_key).parameters
    assert tuple(params) == (
        "subject_id",
        "experience_content_digest",
        "obligation_scope",
        "target_namespace",
    )
    base = ("subject-1", "1" * 64, "review", "review-artifact")
    assert obligation_group_key(*base) == GROUP
    for index, other in enumerate(("subject-2", "2" * 64, "other", "other-ns")):
        changed = list(base)
        changed[index] = other
        assert obligation_group_key(*changed) != GROUP
    with pytest.raises(ValueError):
        obligation_group_key("subject-1", "not-a-digest", "review", "review-artifact")


def test_operation_keys_are_stable_and_direction_bound() -> None:
    forward = operation_key(GROUP, Direction.FORWARD)
    assert forward == operation_key(GROUP, Direction.FORWARD)
    assert forward != operation_key(GROUP, Direction.INVERSE)
    assert forward != GROUP
    with pytest.raises(TypeError):
        operation_key(GROUP, "FORWARD")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        operation_key("x", Direction.FORWARD)


# ---------------------------------------------------------- review artifact


def test_artifact_round_trips_and_binds_its_payload_digest() -> None:
    artifact = _artifact()
    payload = artifact.payload_bytes()
    assert ReviewArtifactV1.from_payload(payload) == artifact
    assert artifact.payload_digest == payload_digest(payload)
    assert artifact.payload_digest == sha256(b"PROJECT-SUZKA:R18:REVIEW-ARTIFACT:V1\0" + payload).hexdigest()
    assert json.loads(payload)["summary"] == "PASS"
    assert len(payload) <= MAX_ARTIFACT_BYTES
    # No raw subject ID or text goes in: only the group key and digests.
    assert b"subject-1" not in payload


def test_artifact_keeps_unknown_and_fail_and_is_not_actionable() -> None:
    """#371 D6-A9-R1: any required gate UNKNOWN or FAIL means no Action."""

    for status in (GateStatus.UNKNOWN, GateStatus.FAIL):
        for gate in ReviewGate:
            artifact = _artifact(**{gate.value: status})
            assert not artifact.actionable
            row = json.loads(artifact.payload_bytes())
            assert row["summary"] == "BLOCKED"
            assert {"gate": gate.value, "status": status.value} in row["gates"]
            with pytest.raises(ValueError, match="not PASS"):
                _intent(artifact=artifact)


def test_artifact_needs_every_role_and_gate_once_in_order() -> None:
    sources, gates = _sources(), _gates()
    for bad_sources in (sources[:-1], sources[::-1], (*sources[:-1], sources[0])):
        with pytest.raises(ValueError):
            ReviewArtifactV1("review-artifact", GROUP, bad_sources, gates)
    for bad_gates in (gates[:-1], gates[::-1], (*gates[:-1], gates[0])):
        with pytest.raises(ValueError):
            ReviewArtifactV1("review-artifact", GROUP, sources, bad_gates)
    with pytest.raises(TypeError):
        ReviewArtifactV1("review-artifact", GROUP, sources, ("PASS",) * 6)  # type: ignore[arg-type]
    # The gate roster maps one-to-one onto the six source roles.
    assert len(ReviewGate) == len(SourceRole) == 6


def test_artifact_payload_is_strict() -> None:
    payload = _artifact().payload_bytes()
    with pytest.raises(ValueError, match="summary"):
        ReviewArtifactV1.from_payload(_tamper(_artifact(CONTEXT_CURRENT=GateStatus.FAIL).payload_bytes(), summary="PASS"))
    with pytest.raises(ValueError):
        ReviewArtifactV1.from_payload(b" " + payload)
    with pytest.raises(ValueError):
        ReviewArtifactV1.from_payload(_tamper(payload, schema_version=2))
    with pytest.raises(ValueError):
        ReviewArtifactV1.from_payload(_tamper(payload, extra=1))
    with pytest.raises(ValueError):
        ReviewArtifactV1.from_payload(b"{" + b" " * MAX_ARTIFACT_BYTES + b"}")


# ------------------------------------------------------------ action intent


def test_action_intent_round_trips_with_stable_identity() -> None:
    intent = _intent()
    payload = intent.canonical_bytes()
    assert ActionIntentV1.from_json(payload) == intent
    assert intent.intent_id.startswith("intent-")
    assert intent.forward_key == operation_key(GROUP, Direction.FORWARD)
    assert intent.payload_digest == intent.artifact.payload_digest
    assert _intent().intent_id == intent.intent_id
    assert _intent(admitting_sequence=21).intent_id != intent.intent_id
    # A new Decision of the same obligation keeps the forward key (#368).
    assert _intent(selected_candidate_digest="9" * 64).forward_key == intent.forward_key


def test_action_intent_refuses_self_origin_and_non_row_4() -> None:
    with pytest.raises(ValueError, match="self-origin"):
        _intent(goal_origin=GoalOrigin.SUBJECT_MOTIVATION)
    for row in PolicyRow:
        if row is not PolicyRow.ROW_4:
            with pytest.raises(ValueError, match="row 4"):
                _intent(policy_row=row)
    with pytest.raises(TypeError):
        _intent(goal_origin="USER_REQUEST")
    with pytest.raises(ValueError, match="V2 or V3"):
        _intent(policy_kind=PolicyKind.SUCCESSOR_POLICY_V1)
    _intent(policy_kind=PolicyKind.SUCCESSOR_POLICY_V3)


def test_action_intent_artifact_must_name_this_decision() -> None:
    with pytest.raises(ValueError, match="DECISION source"):
        _intent(decision_committed_digest="9" * 64)
    with pytest.raises(ValueError, match="DECISION source"):
        _intent(decision_id="decision-" + "9" * 64)


def test_action_intent_bytes_are_strict() -> None:
    payload = _intent().canonical_bytes()
    for changes in (
        {"direction": "INVERSE"},
        {"prestate": "ACTIVE"},
        {"intent_id": "intent-" + "0" * 64},
        {"forward_key": "0" * 64},
        {"goal_origin": "SUBJECT_MOTIVATION"},
        {"policy_row": "ROW_3"},
        {"schema_version": 2},
    ):
        with pytest.raises((ValueError, TypeError)):
            ActionIntentV1.from_json(_tamper(payload, **changes))
    with pytest.raises(ValueError):
        ActionIntentV1.from_json(b" " + payload)


# ----------------------------------------------------------- inverse intent


def test_inverse_intent_only_after_a_verified_forward_effect() -> None:
    for state in (VerifiedState.ACTIVE_VERIFIED_AUTHORIZED, VerifiedState.UNAUTHORIZED_EFFECT):
        intent = _inverse(forward_outcome_state=state)
        assert InverseIntentV1.from_json(intent.canonical_bytes()) == intent
        assert intent.inverse_key == operation_key(GROUP, Direction.INVERSE)
        assert intent.forward_key == _intent().forward_key
        assert intent.intent_id.startswith("inverse-intent-")
    for state in (VerifiedState.VERIFIED_ABSENT, VerifiedState.REVERTED, VerifiedState.INVERSE_VERIFIED_ABSENT):
        with pytest.raises(ValueError, match="verified forward"):
            _inverse(forward_outcome_state=state)
    with pytest.raises(ValueError):
        InverseIntentV1.from_json(_tamper(_inverse().canonical_bytes(), direction="FORWARD"))


# ------------------------------------------------------------------- policy


def test_policy_verdict_deny_reason_and_terminal_states() -> None:
    intent = _intent()
    allow = _allow(intent)
    assert allow.denied_state is None
    assert PolicyVerdictV1.from_json(allow.canonical_bytes()) == allow
    deny = _allow(intent, verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.ADMISSION_BLOCKED)
    assert deny.denied_state is LedgerState.POLICY_DENIED
    assert PolicyVerdictV1.from_json(deny.canonical_bytes()) == deny
    inverse_deny = _allow(_inverse(), verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.TARGET_MISMATCH)
    assert inverse_deny.denied_state is LedgerState.INVERSE_POLICY_DENIED
    with pytest.raises(ValueError, match="deny reason"):
        _allow(intent, verdict=PolicyVerdict.DENY)
    with pytest.raises(ValueError, match="deny reason"):
        _allow(intent, deny_reason=PolicyDenyReason.SOURCE_NOT_CURRENT)
    with pytest.raises(ValueError):
        _allow(intent, intent_id="inverse-intent-" + "0" * 64)


# ---------------------------------------------------------------- challenge


def test_challenge_binds_every_required_value() -> None:
    """#380 §2-3 + D2 addendum 4: intent, expectation, revision, key,
    payload, expiry and boot."""

    intent = _intent()
    challenge = _challenge(intent)
    row = json.loads(challenge.canonical_bytes())
    assert (
        row["intent_digest"],
        row["expectation_digest"],
        row["target_expected_revision"],
        row["operation_key"],
        row["payload_digest"],
        row["expires_utc"],
        row["boot_id"],
    ) == (
        intent.intent_digest,
        EXPECTATION,
        intent.target_expected_revision,
        intent.forward_key,
        intent.payload_digest,
        EXPIRES,
        BOOT,
    )
    assert ApprovalChallengeV1.from_json(challenge.canonical_bytes()) == challenge


def test_challenge_cannot_be_built_with_a_missing_binding() -> None:
    intent = _intent()
    with pytest.raises(ValueError):
        ApprovalChallengeV1.for_forward(
            intent, None, _allow(intent), nonce=NONCE, issued_utc=ISSUED, expires_utc=EXPIRES, boot_id=BOOT  # type: ignore[arg-type]
        )
    for field, value in (
        ("intent_digest", ""),
        ("operation_key", None),
        ("payload_digest", None),
        ("expires_utc", None),
        ("boot_id", ""),
        ("nonce", "short"),
        ("target_expected_revision", None),
    ):
        with pytest.raises((ValueError, TypeError)):
            _challenge(intent, **{field: value})
    with pytest.raises(ValueError, match="R17 expectation"):
        ApprovalChallengeV1(
            NONCE, Direction.INVERSE, _inverse().intent_id, H, EXPECTATION, TARGET, 1, H, H, ISSUED, EXPIRES, BOOT
        )
    with pytest.raises(ValueError):
        _challenge(intent, intent_id=_inverse().intent_id)


def test_challenge_needs_this_intents_policy_allow() -> None:
    intent = _intent()
    deny = _allow(intent, verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.SOURCE_NOT_CURRENT)
    other = _allow(_intent(admitting_sequence=21))

    def build(policy: Any) -> ApprovalChallengeV1:
        return ApprovalChallengeV1.for_forward(
            intent, EXPECTATION, policy, nonce=NONCE, issued_utc=ISSUED, expires_utc=EXPIRES, boot_id=BOOT
        )

    with pytest.raises(ValueError, match="denied"):
        build(deny)
    with pytest.raises(ValueError, match="another intent"):
        build(other)
    with pytest.raises(TypeError):
        build(None)
    inverse = _inverse()
    assert (
        ApprovalChallengeV1.for_inverse(
            inverse, _allow(inverse), nonce=NONCE, issued_utc=ISSUED, expires_utc=EXPIRES, boot_id=BOOT
        ).operation_key
        == inverse.inverse_key
    )
    with pytest.raises(ValueError, match="another intent"):
        ApprovalChallengeV1.for_inverse(
            inverse, _allow(intent), nonce=NONCE, issued_utc=ISSUED, expires_utc=EXPIRES, boot_id=BOOT
        )


def test_challenge_ttl_is_bounded() -> None:
    for expires in (ISSUED, "2026-11-15T23:59:59Z", "2026-11-16T00:15:01Z", "2026-11-16 00:10:00"):
        with pytest.raises(ValueError):
            _challenge(expires_utc=expires)
    assert MAX_APPROVAL_TTL.total_seconds() == 900
    _challenge(expires_utc="2026-11-16T00:15:00Z")


def test_confirmation_code_follows_every_bound_value() -> None:
    challenge = _challenge()
    code = challenge.confirmation_code
    assert len(code) == CONFIRMATION_CODE_LENGTH and set(code) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    for field, value in (
        ("nonce", "0" * 64),
        ("expectation_digest", "0" * 64),
        ("target_expected_revision", 1),
        ("payload_digest", "0" * 64),
        ("expires_utc", "2026-11-16T00:09:00Z"),
        ("boot_id", RUNTIME),
    ):
        assert _challenge(**{field: value}).confirmation_code != code, field


def test_challenge_liveness_is_bounded_by_issue_expiry_and_boot() -> None:
    challenge = _challenge()
    assert challenge.is_live(ISSUED, BOOT)
    assert challenge.is_live("2026-11-16T00:09:59Z", BOOT)
    assert not challenge.is_live(EXPIRES, BOOT)
    assert not challenge.is_live("2026-11-15T23:59:59Z", BOOT)
    assert not challenge.is_live(ISSUED, RUNTIME)  # a restart voids it


# -------------------------------------------------------------------- claim


def test_claim_copies_the_challenge_and_refuses_stale_ones() -> None:
    challenge = _challenge()
    claim = _claim(challenge)
    assert (claim.operation_key, claim.nonce, claim.challenge_digest) == (
        challenge.operation_key,
        challenge.nonce,
        challenge.challenge_digest,
    )
    assert claim.claim_id.startswith("claim-")
    assert ExecutionClaimV1.from_json(claim.canonical_bytes()) == claim
    for now, boot in ((EXPIRES, BOOT), ("2026-11-16T00:05:00Z", RUNTIME)):
        with pytest.raises(ValueError, match="expired"):
            ExecutionClaimV1.claim(
                challenge,
                runtime_instance_id=RUNTIME,
                boot_id=boot,
                claim_event_id=EVENT_C,
                claim_sequence=22,
                claimed_utc=now,
            )
    with pytest.raises(ValueError):
        ExecutionClaimV1.from_json(_tamper(claim.canonical_bytes(), claim_id="claim-" + "0" * 64))


# ------------------------------------------------------------------ receipt


def test_receipt_is_bound_to_its_claim_and_carries_no_proof() -> None:
    claim = _claim()
    applied = ExecutionReceiptV1.for_claim(
        claim, ExecutorResult.APPLIED, reported_revision=1, reported_payload_digest=_intent().payload_digest
    )
    assert (applied.claim_id, applied.operation_key, applied.runtime_instance_id) == (
        claim.claim_id,
        claim.operation_key,
        RUNTIME,
    )
    assert ExecutionReceiptV1.from_json(applied.canonical_bytes()) == applied
    for result in (ExecutorResult.FENCED, ExecutorResult.CAS_REJECTED):
        receipt = ExecutionReceiptV1.for_claim(claim, result, reported_revision=None, reported_payload_digest=None)
        assert ExecutionReceiptV1.from_json(receipt.canonical_bytes()) == receipt
        with pytest.raises(ValueError):
            ExecutionReceiptV1.for_claim(claim, result, reported_revision=1, reported_payload_digest=None)
    with pytest.raises(ValueError):
        ExecutionReceiptV1.for_claim(claim, ExecutorResult.APPLIED, reported_revision=None, reported_payload_digest=H)
    with pytest.raises(ValueError):
        ExecutionReceiptV1.for_claim(claim, ExecutorResult.APPLIED, reported_revision=1, reported_payload_digest=None)
    inverse_claim = replace(claim, direction=Direction.INVERSE, intent_id=_inverse().intent_id)
    ExecutionReceiptV1.for_claim(
        inverse_claim, ExecutorResult.APPLIED, reported_revision=2, reported_payload_digest=None
    )
    # The verifier never reads a receipt: VerifiedOutcomeV1 has no receipt field.
    assert not any("receipt" in name for name in VerifiedOutcomeV1.__dataclass_fields__)


# ------------------------------------------------------------------ outcome


@pytest.mark.parametrize("state", list(VerifiedState))
def test_each_verified_state_round_trips_and_maps_to_the_ledger(state: VerifiedState) -> None:
    outcome = _outcome(state)
    assert VerifiedOutcomeV1.from_json(outcome.canonical_bytes()) == outcome
    assert ledger_state_of(outcome) is LedgerState(state.value)
    assert outcome.verified_outcome_id == _outcome(state, outcome_revision=2).verified_outcome_id
    assert outcome.outcome_digest != _outcome(state, outcome_revision=2).outcome_digest


def test_outcome_states_follow_direction_and_authorization() -> None:
    with pytest.raises(ValueError, match="direction"):
        _outcome(VerifiedState.REVERTED, direction=Direction.FORWARD)
    with pytest.raises(ValueError, match="direction"):
        _outcome(VerifiedState.VERIFIED_ABSENT, direction=Direction.INVERSE)
    with pytest.raises(ValueError, match="authorization"):
        _outcome(authorization=InverseAuthorization.UNPROVEN)
    with pytest.raises(ValueError, match="authorization"):
        _outcome(VerifiedState.UNAUTHORIZED_EFFECT, authorization=None)
    with pytest.raises(ValueError, match="authorization"):
        _outcome(VerifiedState.REVERTED, authorization=None)
    _outcome(VerifiedState.REVERTED, authorization=InverseAuthorization.UNPROVEN)


def test_outcome_observation_must_fit_the_state() -> None:
    with pytest.raises(ValueError):
        _outcome(effect_operation_key=operation_key(GROUP, Direction.INVERSE))
    with pytest.raises(ValueError):
        _outcome(observed_state=ObservedState.ABSENT, observed_payload_digest=None)
    with pytest.raises(ValueError):
        _outcome(
            VerifiedState.VERIFIED_ABSENT,
            observed_state=ObservedState.ACTIVE,
            observed_payload_digest=H,
            effect_operation_key=_intent().forward_key,
        )
    with pytest.raises(ValueError):
        _outcome(
            VerifiedState.REVERTED,
            observed_state=ObservedState.ACTIVE,
            observed_payload_digest=H,
        )
    for state in VerifiedState:
        with pytest.raises(ValueError):
            _outcome(state, observed_state=ObservedState.UNKNOWN, observed_payload_digest=None)
    # An unapplied inverse leaves the forward row: never ABSENT (that is
    # REVERTED) and never the inverse's own key.
    with pytest.raises(ValueError, match="forward effect ACTIVE"):
        _outcome(
            VerifiedState.INVERSE_VERIFIED_ABSENT,
            observed_state=ObservedState.ABSENT,
            observed_payload_digest=None,
            effect_operation_key=None,
        )
    with pytest.raises(ValueError, match="forward effect ACTIVE"):
        _outcome(VerifiedState.INVERSE_VERIFIED_ABSENT, effect_operation_key=operation_key(GROUP, Direction.INVERSE))
    with pytest.raises(ValueError, match="forward effect ACTIVE"):
        _outcome(VerifiedState.INVERSE_VERIFIED_ABSENT, effect_operation_key=None)


def test_forward_outcome_becomes_the_r17_view() -> None:
    outcome = _outcome()
    view = outcome.to_r17_view()
    assert (view.verified_outcome_id, view.outcome_digest, view.outcome_state) == (
        outcome.verified_outcome_id,
        outcome.outcome_digest,
        OutcomeState.ACTIVE_VERIFIED_AUTHORIZED,
    )
    assert _outcome(VerifiedState.VERIFIED_ABSENT).to_r17_view().observed_state is ObservedState.ABSENT
    with pytest.raises(ValueError, match="forward"):
        _outcome(VerifiedState.REVERTED).to_r17_view()


def test_outcome_json_is_strict() -> None:
    payload = _outcome().canonical_bytes()
    with pytest.raises(TypeError):
        VerifiedOutcomeV1.from_json(_tamper(payload, foreign_change=0))
    with pytest.raises(ValueError):
        VerifiedOutcomeV1.from_json(_tamper(payload, outcome_digest="0" * 64))
    with pytest.raises(ValueError):
        VerifiedOutcomeV1.from_json(b" " + payload)


# ---------------------------------------------------------------- lifecycle


def test_transition_table_is_total_and_corrupt_is_always_reachable() -> None:
    for state in LedgerState:
        allowed = allowed_transitions(state)
        assert state not in allowed or state is LedgerState.CORRUPT
        if state is LedgerState.CORRUPT:
            assert allowed == frozenset()
        else:
            assert LedgerState.CORRUPT in allowed
    with pytest.raises(TypeError):
        allowed_transitions("NONE")  # type: ignore[arg-type]


def test_forward_and_inverse_paths_and_terminals() -> None:
    path = (
        LedgerState.NONE,
        LedgerState.INTENT_ISSUED,
        LedgerState.CLAIMED,
        LedgerState.IN_DOUBT,
        LedgerState.ACTIVE_VERIFIED_AUTHORIZED,
        LedgerState.INVERSE_INTENT_ISSUED,
        LedgerState.INVERSE_CLAIMED,
        LedgerState.INVERSE_IN_DOUBT,
        LedgerState.REVERTED,
    )
    for current, new in zip(path, path[1:], strict=False):
        assert transition(current, new) is new
    terminals = {
        LedgerState.VERIFIED_ABSENT,
        LedgerState.INVERSE_POLICY_DENIED,
        LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
        LedgerState.INVERSE_VERIFIED_ABSENT,
        LedgerState.REVERTED,
        LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT,
    }
    for state in terminals:
        assert allowed_transitions(state) == {LedgerState.CORRUPT}, state
    # Unclaimed terminals are never reissued, but the sweep may still find
    # a prior-incarnation effect of the same group (#380 D1 P1-2).
    for state in (LedgerState.POLICY_DENIED, LedgerState.APPROVAL_EXPIRED_UNCLAIMED):
        assert allowed_transitions(state) == {
            LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT,
            LedgerState.CORRUPT,
        }
    # The inverse has its own closed set of states (#380 D1 P1-5).
    assert allowed_transitions(LedgerState.INVERSE_INTENT_ISSUED) == {
        LedgerState.INVERSE_POLICY_DENIED,
        LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
        LedgerState.INVERSE_CLAIMED,
        LedgerState.CORRUPT,
    }


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (LedgerState.NONE, LedgerState.CLAIMED),
        (LedgerState.INTENT_ISSUED, LedgerState.ACTIVE_VERIFIED_AUTHORIZED),
        (LedgerState.CLAIMED, LedgerState.INTENT_ISSUED),
        (LedgerState.CLAIMED, LedgerState.APPROVAL_EXPIRED_UNCLAIMED),  # no CANCELLED after a claim
        (LedgerState.IN_DOUBT, LedgerState.CLAIMED),  # no blind retry
        (LedgerState.VERIFIED_ABSENT, LedgerState.INTENT_ISSUED),  # a FENCED key is never reissued
        (LedgerState.VERIFIED_ABSENT, LedgerState.INVERSE_INTENT_ISSUED),
        (LedgerState.CLAIMED, LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT),  # a Claim exists
        (LedgerState.IN_DOUBT, LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT),
        (LedgerState.VERIFIED_ABSENT, LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT),
        (LedgerState.INVERSE_INTENT_ISSUED, LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT),
        (LedgerState.POLICY_DENIED, LedgerState.INTENT_ISSUED),
        (LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT, LedgerState.INVERSE_INTENT_ISSUED),
        (LedgerState.REVERTED, LedgerState.INVERSE_INTENT_ISSUED),
        (LedgerState.CORRUPT, LedgerState.NONE),
    ],
)
def test_forbidden_transitions_are_refused(current: LedgerState, new: LedgerState) -> None:
    with pytest.raises(ValueError, match="cannot move"):
        transition(current, new)


def test_prior_incarnation_effect_is_reachable_from_every_unclaimed_state() -> None:
    """A restored source with no Claim: NONE, an issued intent, or one denied
    or expired before any Claim."""

    for state in (
        LedgerState.NONE,
        LedgerState.INTENT_ISSUED,
        LedgerState.POLICY_DENIED,
        LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
    ):
        assert transition(state, LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT) is (
            LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT
        )


def test_verified_outcomes_land_only_on_reachable_states() -> None:
    for state in VerifiedState:
        landed = ledger_state_of(_outcome(state))
        in_doubt = LedgerState.INVERSE_IN_DOUBT if state in (
            VerifiedState.REVERTED,
            VerifiedState.INVERSE_VERIFIED_ABSENT,
        ) else LedgerState.IN_DOUBT
        assert transition(in_doubt, landed) is landed


# ------------------------------------------------------------------ purity


def test_r18_imports_stay_pure() -> None:
    tree = ast.parse(inspect.getsource(contracts))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {
        "__future__",
        "base64",
        "dataclasses",
        "datetime",
        "enum",
        "hashlib",
        "typing",
        "suzka.limits",
        "suzka.r16.contracts",
        "suzka.r16.history_codec",
        "suzka.r16.successor_policy",
        "suzka.r17.contracts",
    }
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
    # Submodules (contracts, and IP-15's target / verifier once imported)
    # are attributes of the package but not part of its value exports.
    assert set(r18.__all__) == {
        name
        for name in dir(r18)
        if not name.startswith("_") and not inspect.ismodule(getattr(r18, name))
    }
