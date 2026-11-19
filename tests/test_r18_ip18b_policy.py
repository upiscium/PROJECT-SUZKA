"""IP-18b (policy part): row 2c for the closed forward ledger states, the
V3-only TARGET_UNAVAILABLE member, the shared FENCED predicate and the pure
target-facts mapping (#411 D1 r4 §9-1, §9-2, §9-4, §9-5, §12-2).

The runtime consumers (the sweep, ``prior_claims()``, the Decision issuer's
``_ledger``, the SweepGate) follow IP-18a's merge and are not tested here.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path

import pytest

from suzka.r16.history_codec import PolicyKind, PolicyRow
from suzka.r16.successor_policy import (
    CLOSED_FORWARD_LEDGER,
    CommitmentState,
    FeedbackState,
    Interpretation,
    LedgerState,
    LineageState,
    PolicyOutputKind,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorRow,
    evaluate,
)
from suzka.r18.contracts import Direction, VerifiedState, allowed_transitions, operation_key
from suzka.r18.facts import (
    EMPTY_FACTS,
    TARGET_BLOCKED_REF_KIND,
    TARGET_FENCE_REF_KIND,
    TARGET_UNAVAILABLE_LEDGER,
    TargetFact,
    TargetFacts,
    classify_forward,
    compute_target_facts,
    facts_ledger,
    target_blocked_ref,
    with_target_blocked,
)
from suzka.r18.ledger import NO_LEDGER, LedgerRead
from suzka.r18.target import LedgerKind, LedgerRow
from suzka.r18.verifier import fenced_without_artifact

from test_r16_ip8b_successor_policy import _consistent, _ledger_states
from test_r18_ip15_target_verifier import (
    GROUP,
    _forward,
    _outside,
    _store,
    _verify,
)

V2 = PolicyKind.SUCCESSOR_POLICY_V2
V3 = PolicyKind.SUCCESSOR_POLICY_V3
IP17A = {LedgerState.POLICY_DENIED, LedgerState.APPROVAL_EXPIRED_UNCLAIMED}
IP18B = {
    LedgerState.UNAUTHORIZED_EFFECT,
    LedgerState.VERIFIED_ABSENT,
    LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT,
}
NEW_STATES = IP18B | {LedgerState.TARGET_UNAVAILABLE}


def _digest(version: PolicyKind, wanted) -> tuple[str, dict[str, int]]:  # type: ignore[no-untyped-def]
    hashed, rows = sha256(), Counter()
    for states in product(
        RequiredSourceState, PredecessorState, LineageState, CommitmentState,
        _ledger_states(version), FeedbackState, Interpretation,
    ):
        if not wanted(states[4]):
            continue
        row = evaluate(_consistent(*states, version=version)).row
        rows[row.value] += 1
        hashed.update(("|".join(s.value for s in states) + "=" + row.value + "\n").encode())
    return hashed.hexdigest(), dict(sorted(rows.items()))


# Pinned from the code at 671ec11 (before IP-18b) and from this code.
_V2_ALL = "5098da70263a8f3bfcbb9ec259caaf06d9110a6dc9c1cd006703afa790579c7c"
_V3_KEEP = "7fb412985a95a103129ab94c1b8d1d12a7d56dd0306a4c328cc31bda830ab106"
_V3_IP17A = "1d7678c29e819f991067b777715f0de6053cf567b564e0067333b442b20ce77c"
_V3_IP18B_OLD = "e619b250c1ab274e2eb050b326d626c9c0791431570da2452c1eeaa1e9bebe8e"
_V3_IP18B_NEW = "70546eb0dd9f3b4cf6a3ac31b88884d46f861e0742fab082c5fd961c2f05ffd6"


def test_replay_safety_only_the_intended_tuples_change() -> None:
    """§7 / §12-2.  Over every state tuple of the ip8b builder:

    - V2 (all 91800 tuples): unchanged, digest ``5098da70...``.
    - V3 tuples whose ledger is not one of the closed forward states
      (70200 tuples): unchanged, digest pinned from 671ec11.
    - V3 tuples with POLICY_DENIED / APPROVAL_EXPIRED_UNCLAIMED: IP-17a's, unchanged
      by IP-18b.
    - V3 tuples with UNAUTHORIZED_EFFECT / VERIFIED_ABSENT /
      PRIOR_INCARNATION_EFFECT_PRESENT (16200 tuples) change, and only they:
      row 2 (48) leaves to 2b / 2c, and row 0's ABSENT_INITIAL rule exempts
      them like the IP-17a states (row 0 15240 -> 14760, row 1 912 -> 1344,
      2b +24, 2c +72).  TARGET_UNAVAILABLE did not exist.
    - No persisted Decision holds any of these ledger states (R18 off)."""

    assert _digest(V2, lambda _l: True)[0] == _V2_ALL
    assert _digest(V3, lambda led: led not in CLOSED_FORWARD_LEDGER)[0] == _V3_KEEP
    assert _digest(V3, lambda led: led in IP17A)[0] == _V3_IP17A
    new_digest, new_rows = _digest(V3, lambda led: led in IP18B)
    assert new_digest == _V3_IP18B_NEW != _V3_IP18B_OLD
    assert new_rows == {"0": 14760, "1": 1344, "2b": 24, "2c": 72}
    # No row-2 HOLD is left for the three states (the inverse states stay row 2).
    assert "2" not in new_rows
    assert CLOSED_FORWARD_LEDGER == IP17A | NEW_STATES


def _value(version: PolicyKind, ledger: LedgerState, **changes):  # type: ignore[no-untyped-def]
    states = dict(
        required=RequiredSourceState.QUALIFIED,
        predecessor=PredecessorState.QUALIFIED,
        lineage=LineageState.CONSISTENT,
        commitment=CommitmentState.ACTIVE,
        ledger=ledger,
        feedback=FeedbackState.ABSENT,
        interpretation=Interpretation.SETTLED,
    )
    states.update(changes)
    return _consistent(*states.values(), version=version)


@pytest.mark.parametrize("ledger", sorted(NEW_STATES, key=lambda led: led.value))
def test_new_closed_states_are_row_2c_plan_a_under_v3_only(ledger: LedgerState) -> None:
    result = evaluate(_value(V3, ledger))
    assert (result.row, result.output, result.write_eligible) == (
        SuccessorRow.ROW_2C, PolicyOutputKind.PLAN_A, False,
    )
    assert result.history_row is PolicyRow.ROW_2C
    assert evaluate(_value(V3, ledger, interpretation=Interpretation.CONTRADICTORY)).row is SuccessorRow.ROW_2B
    assert evaluate(_value(V3, ledger, commitment=CommitmentState.ABSENT)).row is SuccessorRow.ROW_2C
    # A new lineage on a closed group re-decides (row 0's exemption).
    assert evaluate(_value(V3, ledger, predecessor=PredecessorState.ABSENT_INITIAL)).row is SuccessorRow.ROW_2C
    assert evaluate(_value(V3, ledger, lineage=LineageState.UNAVAILABLE)).row is SuccessorRow.ROW_1
    if ledger in _ledger_states(V2):
        assert evaluate(_value(V2, ledger)).row is SuccessorRow.ROW_2  # V2 unchanged
        assert evaluate(_value(V2, ledger, predecessor=PredecessorState.ABSENT_INITIAL)).row is SuccessorRow.ROW_0


def test_inverse_states_stay_row_2_and_claimed_is_not_closed() -> None:
    for ledger in LedgerState:
        if ledger.name.startswith("INVERSE_") or ledger in (
            LedgerState.REVERTED, LedgerState.CLAIMED, LedgerState.IN_DOUBT, LedgerState.INTENT_ISSUED
        ):
            assert evaluate(_value(V3, ledger)).row is SuccessorRow.ROW_2, ledger


def test_unauthorized_feedback_is_intercepted_by_row_2c_before_rows_5_and_6() -> None:
    """A Feedback with UNAUTHORIZED_EFFECT: no positive Feedback may accompany
    it (row 0); any other Feedback is row 2c, never 5 / 6."""

    for feedback in FeedbackState:
        if feedback in (FeedbackState.ABSENT, FeedbackState.INVALID):
            continue
        row = evaluate(_value(V3, LedgerState.UNAUTHORIZED_EFFECT, feedback=feedback)).row
        expected = SuccessorRow.ROW_0 if feedback is FeedbackState.MATCH_SELF else SuccessorRow.ROW_2C
        assert row is expected, feedback


def test_target_unavailable_is_v3_only_code_only_and_outcome_free() -> None:
    assert LedgerState.TARGET_UNAVAILABLE.value == "TARGET_UNAVAILABLE"
    with pytest.raises(ValueError, match="V3-only"):
        _value(V2, LedgerState.TARGET_UNAVAILABLE)
    # An Outcome ref is refused with it; Feedback is refused with it (row 0).
    base = _value(V3, LedgerState.TARGET_UNAVAILABLE)
    ref = PolicyRef("r18-outcome", "o" * 8, 1, "c" * 64)
    from dataclasses import replace

    assert evaluate(replace(base, verified_outcome_ref=ref)).row is SuccessorRow.ROW_0
    assert evaluate(replace(base, feedback=FeedbackState.UNKNOWN, feedback_ref=PolicyRef("fb", "f" * 8, 1, "d" * 64))).row is SuccessorRow.ROW_0
    # The canonical bytes round-trip, and V3 bytes name the state.
    again = SuccessorPolicyInputV2.from_bytes(base.canonical_bytes())
    assert again == base and b"TARGET_UNAVAILABLE" in base.canonical_bytes()
    # Not a state of the operation's own machine: nothing follows it.
    assert allowed_transitions(LedgerState.TARGET_UNAVAILABLE) == {LedgerState.CORRUPT}


def test_target_unavailable_ref_is_constant() -> None:
    ref = target_blocked_ref()
    assert (ref.kind, ref.ref_id, ref.revision) == (TARGET_BLOCKED_REF_KIND, "target", 1)
    assert ref == target_blocked_ref() and TARGET_UNAVAILABLE_LEDGER.refs == (ref,)
    assert TARGET_UNAVAILABLE_LEDGER.verified_outcome_ref is None


# ------------------------------------------------- the ledger witness (§9-5)


def test_substitution_is_last_and_only_over_an_empty_ledger() -> None:
    kw = dict(blocked=True, has_obligation=True, has_prior_claim=False)
    assert with_target_blocked(NO_LEDGER, **kw) is TARGET_UNAVAILABLE_LEDGER
    assert with_target_blocked(NO_LEDGER, **{**kw, "blocked": False}) is NO_LEDGER
    assert with_target_blocked(NO_LEDGER, **{**kw, "has_obligation": False}) is NO_LEDGER
    assert with_target_blocked(NO_LEDGER, **{**kw, "has_prior_claim": True}) is NO_LEDGER
    other = LedgerRead(LedgerState.CLAIMED, (PolicyRef("r18-claim", "c" * 8, 1, "a" * 64),), None, None)
    assert with_target_blocked(other, **kw) is other
    assert with_target_blocked(TARGET_UNAVAILABLE_LEDGER, **kw) is TARGET_UNAVAILABLE_LEDGER
    # The witness evaluates to Plan A (row 2c) with an obligation.
    base = _value(V3, LedgerState.TARGET_UNAVAILABLE)
    from dataclasses import replace

    sealed = replace(base, ledger_refs=TARGET_UNAVAILABLE_LEDGER.refs)
    assert evaluate(sealed).row is SuccessorRow.ROW_2C


# ------------------------------------------------------- the shared predicate


def _states(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Every target state of §3-3 as (name, intent, challenge, store)."""

    from test_r18_ip13_contracts import NONCE  # noqa: F401

    out = []

    def new(name: str):  # type: ignore[no-untyped-def]
        sub = tmp_path / name
        sub.mkdir()
        store = _store(sub)
        intent, challenge, claim = _forward(store)
        return store, intent, challenge, claim

    store, intent, challenge, _ = new("none")
    out.append(("none", store, intent, challenge))
    store, intent, challenge, _ = new("fenced")
    store.fence(GROUP, Direction.FORWARD)
    out.append(("fenced", store, intent, challenge))
    store, intent, challenge, claim = new("effect")
    store.forward_write(intent, challenge, claim, now=lambda: _dt.datetime(2026, 11, 16, 0, 6, tzinfo=_dt.timezone.utc))
    out.append(("effect", store, intent, challenge))
    store, intent, challenge, claim = new("mixed")
    _outside(
        store,
        "INSERT INTO target_artifacts (obligation_group_key, state, revision, payload) VALUES (?, 'ACTIVE', 1, ?)",
        parameters=(GROUP, intent.artifact.payload_bytes()),
    )
    store.fence(GROUP, Direction.FORWARD)
    out.append(("fenced_with_artifact", store, intent, challenge))
    store, intent, challenge, claim = new("gone")
    store.forward_write(intent, challenge, claim, now=lambda: _dt.datetime(2026, 11, 16, 0, 6, tzinfo=_dt.timezone.utc))
    _outside(store, "DELETE FROM target_artifacts")
    out.append(("effect_artifact_gone", store, intent, challenge))
    store, intent, challenge, claim = new("tampered")
    store.forward_write(intent, challenge, claim, now=lambda: _dt.datetime(2026, 11, 16, 0, 6, tzinfo=_dt.timezone.utc))
    _outside(store, "UPDATE target_artifacts SET payload = x'00'")
    out.append(("effect_foreign", store, intent, challenge))
    return out


def test_verifier_and_facts_agree_on_every_target_state(tmp_path: Path) -> None:
    expected = {
        "none": (False, None, None),
        "fenced": (True, VerifiedState.VERIFIED_ABSENT, TargetFact.FENCED_NO_CLAIM),
        "effect": (False, VerifiedState.ACTIVE_VERIFIED_AUTHORIZED, TargetFact.EFFECT_NO_CLAIM),
        "fenced_with_artifact": (False, None, None),
        "effect_artifact_gone": (False, None, None),
        "effect_foreign": (False, VerifiedState.UNAUTHORIZED_EFFECT, TargetFact.EFFECT_NO_CLAIM),
    }
    for name, store, intent, challenge in _states(tmp_path):
        status = store.status(GROUP)
        shared = fenced_without_artifact(status.forward, status.artifact)
        outcome = _verify(store, intent, challenge)
        absent = outcome is not None and outcome.outcome_state is VerifiedState.VERIFIED_ABSENT
        fact = classify_forward(status.forward, status.artifact)
        want_shared, want_state, want_fact = expected[name]
        assert shared is want_shared is absent, name
        assert (None if outcome is None else outcome.outcome_state) is want_state, name
        assert fact is want_fact, name
        # The shared predicate and the fact never disagree.
        assert (fact is TargetFact.FENCED_NO_CLAIM) is shared, name
        # The facts for a snapshot of this one group.
        facts = compute_target_facts(
            [(status.forward, status.artifact)] if status.forward is not None else [], ()
        )
        assert facts.get(intent.forward_key) is want_fact, name


# ----------------------------------------------------- facts (pure mapping)


def _row(group: str, kind: LedgerKind, *, key: str | None = None) -> LedgerRow:
    return LedgerRow(
        key or operation_key(group, Direction.FORWARD), group, Direction.FORWARD, kind,
        None, None, None, None,
    )


def test_compute_target_facts_is_a_pure_function_of_rows_and_claimed_keys(tmp_path: Path) -> None:
    states = {name: (store, intent) for name, store, intent, _c in _states(tmp_path)}
    rows = []
    keys = {}
    for name, (store, intent) in states.items():
        status = store.status(GROUP)
        if status.forward is not None:
            rows.append((name, status.forward, status.artifact))
            keys[name] = intent.forward_key
    # The same group key appears in every sub-store, so classify one at a time.
    for name, forward, artifact in rows:
        facts = compute_target_facts([(forward, artifact)], ())
        again = compute_target_facts([(forward, artifact)], frozenset())
        assert facts == again
        # A key with a Claim in this incarnation is never a fact (not prior).
        assert not compute_target_facts([(forward, artifact)], [forward.operation_key])
    # Many groups in one snapshot, order independent, claimed keys excluded.
    groups = [f"{n:064x}" for n in range(1, 6)]
    snapshot = [
        (_row(groups[0], LedgerKind.FENCED), None),
        (_row(groups[1], LedgerKind.FENCED), object()),  # mixed: no fact
        (_row(groups[2], LedgerKind.EFFECT), None),  # effect, artifact gone: no fact
        (_row(groups[3], LedgerKind.FENCED), None),
    ]
    claimed = [operation_key(groups[3], Direction.FORWARD)]
    facts = compute_target_facts(snapshot, claimed)  # type: ignore[arg-type]
    assert dict(facts.by_key) == {operation_key(groups[0], Direction.FORWARD): TargetFact.FENCED_NO_CLAIM}
    assert compute_target_facts(list(reversed(snapshot)), claimed) == facts  # type: ignore[arg-type]
    # A foreign row (key not derived from its group) yields nothing.
    assert not compute_target_facts([(_row(groups[0], LedgerKind.FENCED, key="a" * 64), None)], ())
    # An inverse row is not a forward fact.
    inverse = LedgerRow(
        operation_key(groups[0], Direction.INVERSE), groups[0], Direction.INVERSE,
        LedgerKind.FENCED, None, None, None, None,
    )
    assert not compute_target_facts([(inverse, None)], ())
    assert not EMPTY_FACTS and len(EMPTY_FACTS) == 0 and TargetFacts({}) == EMPTY_FACTS


def test_facts_ledger_states_and_rows() -> None:
    group = "1" * 64
    key = operation_key(group, Direction.FORWARD)
    assert facts_ledger(EMPTY_FACTS, key) is None
    fenced = facts_ledger(TargetFacts({key: TargetFact.FENCED_NO_CLAIM}), key)
    effect = facts_ledger(TargetFacts({key: TargetFact.EFFECT_NO_CLAIM}), key)
    assert fenced is not None and effect is not None
    assert fenced.state is LedgerState.VERIFIED_ABSENT
    assert effect.state is LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT
    assert fenced.refs[0].kind == TARGET_FENCE_REF_KIND
    assert fenced.verified_outcome_ref is None and effect.verified_outcome_ref is None
    # Deterministic: the same facts, the same bytes (identical across restarts).
    assert fenced == facts_ledger(TargetFacts({key: TargetFact.FENCED_NO_CLAIM}), key)
    for read in (fenced, effect):
        sealed = _value(V3, read.state)
        from dataclasses import replace

        sealed = replace(sealed, ledger_refs=read.refs)
        assert evaluate(sealed).row is SuccessorRow.ROW_2C  # no Outcome needed
        assert SuccessorPolicyInputV2.from_bytes(sealed.canonical_bytes()) == sealed
