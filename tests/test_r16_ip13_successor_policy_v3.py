"""IP-13: SuccessorPolicy V3 = V2 + PRIOR_INCARNATION_EFFECT_PRESENT (#411).

V2 stays the sealed version (the issuer moves with IP-14a); every V2 byte,
schema string and fingerprint must stay exactly what IP-8b produced.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.contracts import ReasonCode  # noqa: E402
from suzka.r16.history_codec import MAX_POLICY_INPUT_BYTES, PolicyKind  # noqa: E402
from suzka.r16.successor_policy import (  # noqa: E402
    INPUT_SCHEMA,
    INPUT_SCHEMA_V3,
    POLICY_KIND,
    CommitmentState,
    FeedbackState,
    InverseAuthorization,
    LedgerState,
    LineageState,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorRow,
    evaluate,
    evaluate_sealed,
    worst_case_input,
)

from test_r16_ip8b_successor_policy import (  # noqa: E402
    FEEDBACK,
    LEDGER,
    OUTCOME,
    _active,
    _input,
    _verified,
)

V2 = PolicyKind.SUCCESSOR_POLICY_V2
V3 = PolicyKind.SUCCESSOR_POLICY_V3
PRIOR = LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT

# (canonical-bytes sha256, evidence fingerprint) produced by rebuild/develop
# 5afda24, before IP-13. A V2 input must reproduce them exactly.
V2_GOLDEN = {
    "input": (
        "97e3b316636f479acc02302c3a6ad9c0a65123840a5c6dc141ac5d1a1279bcdb",
        "517834561d0d00265e7716542567f59ff21341f0d60352b04bd87ea1bf1fd0da",
    ),
    "active": (
        "9f578f236316741c83fcd1ad5967b267228c8a7be0c8be659806a9920f783944",
        "e6af5f5de48222af6873aad35723521373add84bc4405055277f61090baaf836",
    ),
    "verified": (
        "ce5c50cacea06a16eceb5e9695576b9d32b0513f4f8e9ca7963568d6ada7db87",
        "f0d8bc67a5399c1ef964e393b225aa197b7194d6d69601f44fa2493ad23492a3",
    ),
    "reverted": (
        "bcb741056318d3f35db715060defee6babf31a9ef3875a0e14edcdd6e11a49a8",
        "ef2917c7dff6ac6acfce2ee742c2c478fac214ce424cf10222bb204aecd51884",
    ),
    "worst": (
        "e3182284973e2b03b84944767dbcd3bcc9b6a94f55aeef8c6e4d82d57e65d9ce",
        "e061238ea64cb49638dfc93e93b8ec3922235e8e0294dbc1e251d14a295fd5ab",
    ),
}


def _golden_inputs() -> dict[str, SuccessorPolicyInputV2]:
    return {
        "input": _input(),
        "active": _active(),
        "verified": _verified(),
        "reverted": _verified(
            ledger=LedgerState.REVERTED, inverse_authorization=InverseAuthorization.AUTHORIZED
        ),
        "worst": worst_case_input(),
    }


def _prior(**changes: object) -> SuccessorPolicyInputV2:
    """A prior-incarnation effect seen by a qualified successor."""

    values: dict[str, object] = {
        "version": V3,
        "ledger": PRIOR,
        "ledger_refs": LEDGER,
        "verified_outcome_ref": None,
        "feedback": FeedbackState.ABSENT,
        "feedback_ref": None,
    }
    values.update(changes)
    return _verified(**values)


# ------------------------------------------------------- V2 compatibility


def test_v2_bytes_and_fingerprints_are_unchanged() -> None:
    for name, value in _golden_inputs().items():
        assert value.version is V2
        payload = value.canonical_bytes()
        assert (sha256(payload).hexdigest(), value.evidence_fingerprint()) == V2_GOLDEN[name], name
        assert f'"schema":"{INPUT_SCHEMA}"'.encode() in payload


def test_the_issuer_seals_v3_since_ip14a3() -> None:
    assert POLICY_KIND is V3
    assert INPUT_SCHEMA == "SUCCESSOR_POLICY_INPUT_V2"
    assert INPUT_SCHEMA_V3 == "SUCCESSOR_POLICY_INPUT_V3"


def test_v2_bytes_never_decode_as_v3() -> None:
    for value in _golden_inputs().values():
        payload = value.canonical_bytes()
        decoded = SuccessorPolicyInputV2.from_bytes(payload)
        assert decoded.version is V2 and decoded == value
        assert evaluate_sealed(payload) == evaluate(value)
        # The schema string alone selects the version; relabelling is a
        # different input with a different fingerprint, not a misread.
        relabelled = SuccessorPolicyInputV2.from_bytes(payload.replace(b"INPUT_V2", b"INPUT_V3"))
        assert relabelled.version is V3
        assert relabelled.evidence_fingerprint() != value.evidence_fingerprint()
        for schema in (b"INPUT_V1", b"INPUT_V4", b"input_V2"):
            with pytest.raises(ValueError):
                SuccessorPolicyInputV2.from_bytes(payload.replace(b"INPUT_V2", schema))


def test_v3_only_state_is_refused_in_v2() -> None:
    with pytest.raises(ValueError, match="V3-only"):
        _prior(version=V2)
    v3 = _prior().canonical_bytes()
    with pytest.raises(ValueError):
        SuccessorPolicyInputV2.from_bytes(v3.replace(b"INPUT_V3", b"INPUT_V2"))


def test_version_must_be_v2_or_v3_exactly() -> None:
    for bad in (PolicyKind.SUCCESSOR_POLICY_V1, "SUCCESSOR_POLICY_V2", None):
        with pytest.raises(ValueError):
            _input(version=bad)


def test_v3_evaluates_every_v2_state_exactly_like_v2() -> None:
    """V3 adds one state; the ordered table is otherwise identical."""

    samples = [*_golden_inputs().values()]
    for ledger in LedgerState:
        if ledger is PRIOR:
            continue
        extra: dict[str, object] = (
            {"inverse_authorization": InverseAuthorization.AUTHORIZED}
            if ledger is LedgerState.REVERTED
            else {}
        )
        samples.append(_verified(ledger=ledger, **extra))
        samples.append(_verified(ledger=ledger, verified_outcome_ref=None, **extra))
    for value in samples:
        v3 = replace(value, version=V3)
        assert evaluate(v3) == evaluate(value)
        assert SuccessorPolicyInputV2.from_bytes(v3.canonical_bytes()) == v3
        assert len(v3.canonical_bytes()) == len(value.canonical_bytes())


# ------------------------------------------------------ the new V3 state


def test_prior_incarnation_effect_is_a_row_2_hold() -> None:
    """#373 A2-4 / #378 II-4 / #380 D1 P1-2: HOLD, WRITE=false."""

    for commitment in (CommitmentState.ACTIVE, CommitmentState.ABSENT):
        extra: dict[str, object] = (
            {"commitment_ref": None, "commitment_deadline_utc": None}
            if commitment is CommitmentState.ABSENT
            else {}
        )
        result = evaluate(_prior(commitment=commitment, **extra))
        assert (result.row, result.hold_reason, result.is_hold) == (
            SuccessorRow.ROW_2,
            ReasonCode.INSUFFICIENT_EVIDENCE,
            True,
        )


def test_prior_incarnation_effect_cannot_carry_a_verified_outcome_or_feedback() -> None:
    """No source-side Claim survives, so no Outcome and no Feedback (D6-2)."""

    for changes in (
        {"verified_outcome_ref": OUTCOME},
        {"feedback": FeedbackState.MATCH_SELF, "feedback_ref": FEEDBACK},
        {"ledger_refs": ()},
    ):
        result = evaluate(_prior(**changes))
        assert (result.row, result.hold_reason) == (
            SuccessorRow.ROW_0,
            ReasonCode.CONFLICTING_EVIDENCE,
        ), changes


def test_prior_incarnation_effect_without_predecessor_keeps_the_v2_row_0_rule() -> None:
    """A non-NONE ledger under ABSENT_INITIAL is already a row 0 contradiction
    in V2; V3 does not relax it. Both rows HOLD with WRITE=false."""

    result = evaluate(_prior(predecessor=PredecessorState.ABSENT_INITIAL, predecessor_ref=_input().predecessor_ref))
    assert (result.row, result.is_hold) == (SuccessorRow.ROW_0, True)


def test_earlier_rows_still_win_over_the_new_state() -> None:
    assert evaluate(_prior(lineage=LineageState.MISMATCH)).row is SuccessorRow.ROW_0
    assert evaluate(_prior(required_source=RequiredSourceState.UNAVAILABLE)).row is SuccessorRow.ROW_1


def test_v3_sealed_bytes_round_trip_and_replay() -> None:
    value = _prior()
    payload = value.canonical_bytes()
    assert f'"schema":"{INPUT_SCHEMA_V3}"'.encode() in payload
    assert SuccessorPolicyInputV2.from_bytes(payload) == value
    assert evaluate_sealed(payload) == evaluate(value)


def test_fingerprint_domain_is_per_version() -> None:
    value = _active()
    assert replace(value, version=V3).evidence_fingerprint() != value.evidence_fingerprint()


# -------------------------------------------------------- byte-bound proof


@pytest.mark.parametrize("version", [V2, V3])
def test_worst_case_input_fits_for_each_version(version: PolicyKind) -> None:
    """The single-field size proof for both versions is the parametrized
    IP-8b test; this checks the bound and the round trip."""

    worst = worst_case_input(version)
    assert worst.version is version
    assert len(worst.canonical_bytes()) <= MAX_POLICY_INPUT_BYTES
    assert SuccessorPolicyInputV2.from_bytes(worst.canonical_bytes()) == worst


def test_v3_worst_case_is_no_longer_than_v2() -> None:
    assert len(worst_case_input(V3).canonical_bytes()) == len(worst_case_input(V2).canonical_bytes())


# ------------------------------------------------------- issuer head check


def test_v2_head_still_verifies_and_a_kind_mismatch_does_not(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """#411 IP-13 acceptance: an existing V2 head decodes and replays (the
    V2 head is issued as before IP-14a-3 switched the issuer to V3)."""

    from suzka.r16 import decision_issuer as issuer_module

    from test_r16_ip10_decision_issuer import _app, _issue, _reader_head, _settings
    from test_r16_u3a_plan_preview import _qualified_goal

    monkeypatch.setattr(issuer_module, "POLICY_KIND", V2)
    with TestClient(_app(_settings(tmp_path / "ip13"))) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        issued = _issue(state, gid)
        committed = issued.decision
        assert committed is not None and committed.policy_kind is V2
        head = _reader_head(state, issued)
        issuer = state.decision_issuer
        assert issuer._head_verified(committed, committed.incarnation_id, head.transaction_id)
        relabelled = replace(committed, policy_kind=V3)
        assert not issuer._head_verified(relabelled, committed.incarnation_id, head.transaction_id)
        # The next issue sees the V2 head as an existing, verified predecessor,
        # also with the V3 issuer of IP-14a-3.
        monkeypatch.setattr(issuer_module, "POLICY_KIND", V3)
        again = _issue(state, gid)
        assert again.decision is not None and again.decision.decision_id == committed.decision_id
