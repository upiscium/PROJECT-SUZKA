"""IP-8b: SuccessorPolicyV2 / SuccessorPolicyInputV2 (#374 F1 D3 §3, #371 D5/D6)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.contracts import ReasonCode  # noqa: E402
from suzka.r16.history_codec import (  # noqa: E402
    MAX_POLICY_INPUT_BYTES,
    PolicyKind,
    PolicyRow,
)
from suzka.r16.successor_policy import (  # noqa: E402
    MAX_LEDGER_REFS,
    NO_PREDECESSOR_HORIZON_KIND,
    POLICY_KIND,
    CommitmentState,
    FeedbackState,
    Interpretation,
    InterpretationStage,
    InverseAuthorization,
    LedgerState,
    LineageRef,
    LineageState,
    PolicyOutputKind,
    PolicyRef,
    PredecessorState,
    RequiredSourceState,
    SuccessorPolicyInputV2,
    SuccessorPolicyResult,
    SuccessorRow,
    cross_reference_conflicts,
    evaluate,
    evaluate_sealed,
    interpretation_from_r15,
    utc_timestamp,
    worst_case_input,
)

D = "a" * 64
NOW = "2026-11-14T00:00:00Z"
DEADLINE = "2026-11-21T00:00:00Z"
SOURCES = tuple(PolicyRef(kind, f"{kind}-1", 1, D) for kind in ("context", "experience", "goal", "value"))
LINEAGE = LineageRef("lineage-1", "baseline-1", D, None, "b" * 64)
COMMITMENT = PolicyRef("commitment", "commitment-1", 1, D)
PREDECESSOR = PolicyRef("decision", "decision-1", 1, D)
HORIZON = PolicyRef(NO_PREDECESSOR_HORIZON_KIND, "horizon-9", 9, D)
LEDGER = (PolicyRef("ledger", "ledger-1", 1, D),)
OUTCOME = PolicyRef("outcome", "outcome-1", 1, D)
FEEDBACK = PolicyRef("feedback", "feedback-1", 1, D)
STAGE_OF = {
    Interpretation.SETTLED: InterpretationStage.S2,
    Interpretation.UNCERTAIN: InterpretationStage.S2,
    Interpretation.CONTRADICTORY: InterpretationStage.S2,
    Interpretation.NOT_REVIEWED: InterpretationStage.NONE,
    Interpretation.UNAVAILABLE: InterpretationStage.NONE,
}


def _input(**changes: object) -> SuccessorPolicyInputV2:
    """Row 3 by default: initial, qualified, no Commitment, not reviewed."""

    base = SuccessorPolicyInputV2(
        required_source=RequiredSourceState.QUALIFIED,
        source_refs=SOURCES,
        predecessor=PredecessorState.ABSENT_INITIAL,
        predecessor_ref=HORIZON,
        lineage=LineageState.CONSISTENT,
        lineage_ref=LINEAGE,
        commitment=CommitmentState.ABSENT,
        commitment_ref=None,
        commitment_deadline_utc=None,
        ledger=LedgerState.NONE,
        ledger_refs=(),
        inverse_authorization=None,
        feedback=FeedbackState.ABSENT,
        feedback_ref=None,
        verified_outcome_ref=None,
        interpretation=Interpretation.NOT_REVIEWED,
        interpretation_stage=InterpretationStage.NONE,
        as_of_r07_sequence=9,
        as_of_snapshot_hash="c" * 64,
        event_id="event-10",
        event_sequence=10,
        now_utc=NOW,
    )
    if "interpretation" in changes and "interpretation_stage" not in changes:
        changes["interpretation_stage"] = STAGE_OF[changes["interpretation"]]  # type: ignore[index]
    return replace(base, **changes)  # type: ignore[arg-type]


def _active(**changes: object) -> SuccessorPolicyInputV2:
    values: dict[str, object] = {
        "commitment": CommitmentState.ACTIVE,
        "commitment_ref": COMMITMENT,
        "commitment_deadline_utc": DEADLINE,
    }
    values.update(changes)
    return _input(**values)


def _verified(**changes: object) -> SuccessorPolicyInputV2:
    values: dict[str, object] = {
        "predecessor": PredecessorState.QUALIFIED,
        "predecessor_ref": PREDECESSOR,
        "ledger": LedgerState.ACTIVE_VERIFIED_AUTHORIZED,
        "ledger_refs": LEDGER,
        "verified_outcome_ref": OUTCOME,
        "feedback": FeedbackState.MATCH_SELF,
        "feedback_ref": FEEDBACK,
    }
    values.update(changes)
    return _active(**values)


def _row(value: SuccessorPolicyInputV2) -> SuccessorRow:
    return evaluate(value).row


# ------------------------------------------------------------- the table


@pytest.mark.parametrize(
    ("value", "row", "output", "reason"),
    [
        (_input(lineage=LineageState.MISMATCH), "0", PolicyOutputKind.HOLD, ReasonCode.CONFLICTING_EVIDENCE),
        (_input(required_source=RequiredSourceState.UNAVAILABLE), "1", PolicyOutputKind.HOLD, ReasonCode.INSUFFICIENT_EVIDENCE),
        (_input(interpretation=Interpretation.UNAVAILABLE), "1", PolicyOutputKind.HOLD, ReasonCode.INSUFFICIENT_EVIDENCE),
        (_verified(ledger=LedgerState.IN_DOUBT, verified_outcome_ref=None, feedback=FeedbackState.ABSENT, feedback_ref=None), "2", PolicyOutputKind.HOLD, ReasonCode.INSUFFICIENT_EVIDENCE),
        (_input(interpretation=Interpretation.CONTRADICTORY), "2b", PolicyOutputKind.HOLD, ReasonCode.CONFLICTING_EVIDENCE),
        (_input(), "3", PolicyOutputKind.PLAN_A, None),
        (_input(interpretation=Interpretation.UNCERTAIN), "3", PolicyOutputKind.PLAN_A, None),
        (_active(), "4", PolicyOutputKind.PLAN_B, None),
        (_active(interpretation=Interpretation.SETTLED), "4", PolicyOutputKind.PLAN_B, None),
        (_active(interpretation=Interpretation.UNCERTAIN), "4b", PolicyOutputKind.PLAN_A, None),
        (_verified(), "5", PolicyOutputKind.NO_OP, None),
        (_verified(interpretation=Interpretation.SETTLED), "5", PolicyOutputKind.NO_OP, None),
        (_verified(interpretation=Interpretation.UNCERTAIN), "5b", PolicyOutputKind.PLAN_A, None),
        (_verified(feedback=FeedbackState.MISMATCH), "6", PolicyOutputKind.PLAN_A, None),
        (_verified(feedback=FeedbackState.ABSENT, feedback_ref=None), "6", PolicyOutputKind.PLAN_A, None),
        (_verified(predecessor=PredecessorState.QUALIFIED, ledger=LedgerState.ACTIVE_VERIFIED_AUTHORIZED, commitment=CommitmentState.ABSENT, commitment_ref=None, commitment_deadline_utc=None), "7", PolicyOutputKind.HOLD, ReasonCode.INSUFFICIENT_EVIDENCE),
    ],
)
def test_each_row_and_its_output(value, row, output, reason) -> None:  # type: ignore[no-untyped-def]
    result = evaluate(value)
    assert (result.row.value, result.output, result.hold_reason) == (row, output, reason)
    assert result.write_eligible is False
    if result.is_hold:
        assert result.hold_assessment().reason is reason
        with pytest.raises(ValueError):
            result.history_row
    else:
        assert result.history_row is PolicyRow(row)
        with pytest.raises(ValueError):
            result.hold_assessment()


def test_row_order_conflict_before_insufficient_before_contradictory() -> None:
    # Row 0 beats row 1; row 1 (interpretation UNAVAILABLE) beats row 2;
    # row 2 (pending ledger) beats row 2b.
    assert _row(_input(lineage=LineageState.MISMATCH, required_source=RequiredSourceState.UNAVAILABLE)) is SuccessorRow.ROW_0
    pending = dict(
        predecessor=PredecessorState.QUALIFIED,
        predecessor_ref=PREDECESSOR,
        ledger=LedgerState.CLAIMED,
        ledger_refs=LEDGER,
    )
    assert _row(_active(interpretation=Interpretation.UNAVAILABLE, **pending)) is SuccessorRow.ROW_1
    assert _row(_active(interpretation=Interpretation.CONTRADICTORY, **pending)) is SuccessorRow.ROW_2


def test_self_origin_goal_is_ineligible_row_1() -> None:
    """#379 D2: a self-origin Goal's commitment arrives INELIGIBLE."""

    value = _input(
        commitment=CommitmentState.INELIGIBLE,
        commitment_ref=COMMITMENT,
        commitment_deadline_utc=DEADLINE,
    )
    assert _row(value) is SuccessorRow.ROW_1


def _consistent(
    required: RequiredSourceState,
    predecessor: PredecessorState,
    lineage: LineageState,
    commitment: CommitmentState,
    ledger: LedgerState,
    feedback: FeedbackState,
    interpretation: Interpretation,
) -> SuccessorPolicyInputV2:
    """Refs filled exactly as each state requires, so only states vary."""

    return _input(
        required_source=required,
        predecessor=predecessor,
        predecessor_ref={
            PredecessorState.QUALIFIED: PREDECESSOR,
            PredecessorState.ABSENT_INITIAL: HORIZON,
        }.get(predecessor),
        lineage=lineage,
        commitment=commitment,
        commitment_ref=None if commitment is CommitmentState.ABSENT else COMMITMENT,
        commitment_deadline_utc=None if commitment is CommitmentState.ABSENT else DEADLINE,
        ledger=ledger,
        ledger_refs=() if ledger is LedgerState.NONE else LEDGER,
        inverse_authorization=InverseAuthorization.UNPROVEN
        if ledger is LedgerState.REVERTED
        else None,
        verified_outcome_ref=None if ledger in (
            LedgerState.NONE,
            LedgerState.INTENT_ISSUED,
            LedgerState.POLICY_DENIED,
            LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
        ) else OUTCOME,
        feedback=feedback,
        feedback_ref=None if feedback in (FeedbackState.ABSENT, FeedbackState.INVALID) else FEEDBACK,
        interpretation=interpretation,
    )


def test_policy_is_total_and_never_writes_over_every_state_tuple() -> None:
    """All 3*4*3*5*17*6*5 state tuples: total, closed, no WRITE path."""

    seen: set[SuccessorRow] = set()
    for states in product(
        RequiredSourceState,
        PredecessorState,
        LineageState,
        CommitmentState,
        LedgerState,
        FeedbackState,
        Interpretation,
    ):
        value = _consistent(*states)
        result = evaluate(value)
        seen.add(result.row)
        interpretation = states[-1]
        assert result.write_eligible is False
        if result.output is PolicyOutputKind.NO_OP:
            assert result.row is SuccessorRow.ROW_5
            assert states[5] is FeedbackState.MATCH_SELF
        if result.output is PolicyOutputKind.PLAN_B:
            assert result.row is SuccessorRow.ROW_4
            assert states[4] is LedgerState.NONE
        if interpretation is Interpretation.CONTRADICTORY or interpretation is Interpretation.UNAVAILABLE:
            assert result.is_hold
        if interpretation is Interpretation.UNCERTAIN:
            # UNCERTAIN never reaches Plan B or NO_OP.
            assert result.output not in (PolicyOutputKind.PLAN_B, PolicyOutputKind.NO_OP)
    assert seen == set(SuccessorRow)


def test_uncertain_only_downgrades_rows_4_and_5_to_plan_a() -> None:
    """Changing only interpretation never moves HOLD <-> Decision except 2b/1."""

    for base in (_input(), _active(), _verified(), _verified(feedback=FeedbackState.UNKNOWN)):
        settled = evaluate(replace(base, interpretation=Interpretation.SETTLED, interpretation_stage=InterpretationStage.S2))
        uncertain = evaluate(replace(base, interpretation=Interpretation.UNCERTAIN, interpretation_stage=InterpretationStage.S2))
        not_reviewed = evaluate(base)
        assert settled.output is not_reviewed.output
        if settled.output in (PolicyOutputKind.PLAN_B, PolicyOutputKind.NO_OP):
            assert uncertain.output is PolicyOutputKind.PLAN_A
        else:
            assert uncertain.row is settled.row


# ------------------------------------------------- D6-2 cross references


@pytest.mark.parametrize(
    ("value", "hit"),
    [
        (_input(source_refs=SOURCES[:3]), "required_source_refs"),
        (_input(predecessor_ref=PREDECESSOR), "predecessor_ref"),
        (_input(predecessor_ref=None), "predecessor_ref"),
        (_input(predecessor=PredecessorState.QUALIFIED), "predecessor_ref"),
        (_input(lineage_ref=None), "lineage_ref"),
        (_input(commitment_ref=COMMITMENT), "absent_commitment_has_refs"),
        (_active(commitment_deadline_utc=None), "commitment_ref"),
        (_input(ledger_refs=LEDGER), "ledger_none_has_refs"),
        (_verified(ledger_refs=()), "ledger_refs"),
        (_input(verified_outcome_ref=OUTCOME), "outcome_without_effect"),
        (_verified(verified_outcome_ref=None), "verified_effect_without_outcome"),
        (_input(feedback_ref=FEEDBACK), "absent_feedback_has_ref"),
        (_verified(feedback_ref=None), "feedback_ref"),
        (_verified(ledger=LedgerState.CLAIMED, verified_outcome_ref=None), "feedback_without_verified_effect"),
        (_verified(ledger=LedgerState.UNAUTHORIZED_EFFECT), "positive_feedback_without_authorized_effect"),
        (_input(as_of_r07_sequence=11), "as_of_not_before_event"),
        (_input(as_of_r07_sequence=10), "as_of_not_before_event"),
        (_verified(verified_outcome_ref=PolicyRef("goal", "goal-1", 1, D)), "reference_identity_reused"),
        (_verified(feedback_ref=COMMITMENT), "reference_identity_reused"),
        (_verified(ledger=LedgerState.REVERTED), "inverse_authorization"),
        (_verified(inverse_authorization=InverseAuthorization.AUTHORIZED), "inverse_authorization"),
        (_active(now_utc=DEADLINE), "active_commitment_past_deadline"),
    ],
)
def test_cross_reference_inconsistency_is_row_0(value, hit) -> None:  # type: ignore[no-untyped-def]
    assert hit in cross_reference_conflicts(value)
    result = evaluate(value)
    assert (result.row, result.hold_reason) == (SuccessorRow.ROW_0, ReasonCode.CONFLICTING_EVIDENCE)


def test_state_level_contradictions_are_row_0() -> None:
    assert _row(_input(feedback=FeedbackState.INVALID)) is SuccessorRow.ROW_0
    assert _row(_input(feedback=FeedbackState.UNKNOWN, feedback_ref=FEEDBACK)) is SuccessorRow.ROW_0
    # ABSENT_INITIAL with any ledger entry is a contradiction.
    assert _row(_input(ledger=LedgerState.INTENT_ISSUED, ledger_refs=LEDGER)) is SuccessorRow.ROW_0
    assert _row(_input(ledger=LedgerState.CORRUPT)) is SuccessorRow.ROW_0


# --------------------------------------------------- input construction


def test_closed_stage_interpretation_pairs() -> None:
    with pytest.raises(ValueError):
        _input(interpretation=Interpretation.SETTLED, interpretation_stage=InterpretationStage.NONE)
    with pytest.raises(ValueError):
        _input(interpretation=Interpretation.CONTRADICTORY, interpretation_stage=InterpretationStage.S1)
    with pytest.raises(ValueError):
        _input(interpretation=Interpretation.NOT_REVIEWED, interpretation_stage=InterpretationStage.S2)
    assert _input(interpretation=Interpretation.UNCERTAIN, interpretation_stage=InterpretationStage.S1)


def test_construction_rejects_non_exact_or_unbounded_values() -> None:
    with pytest.raises(TypeError):
        _input(ledger="NONE")
    with pytest.raises(ValueError):
        _input(now_utc="2026-11-14T00:00:00+00:00")
    with pytest.raises(ValueError):
        _input(event_sequence=0)
    with pytest.raises(ValueError):
        _input(source_refs=tuple(reversed(SOURCES)))
    with pytest.raises(ValueError):
        _verified(ledger_refs=tuple(PolicyRef("ledger", f"l-{i}", 1, D) for i in range(MAX_LEDGER_REFS + 1)))
    with pytest.raises(TypeError):
        evaluate(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SuccessorPolicyResult(SuccessorRow.ROW_4, PolicyOutputKind.PLAN_B, None, write_eligible=True)
    with pytest.raises(ValueError):
        SuccessorPolicyResult(SuccessorRow.ROW_4, PolicyOutputKind.PLAN_A, None)


def test_utc_timestamp_is_one_trusted_clock_read() -> None:
    moment = datetime(2026, 11, 14, 9, 0, 0, tzinfo=timezone(timedelta(hours=9)))
    assert utc_timestamp(moment) == NOW
    with pytest.raises(ValueError):
        utc_timestamp(datetime(2026, 11, 14))


# ------------------------------------------------------- sealed bytes


def test_worst_case_input_fits_the_8_kib_bound() -> None:
    """D6-3 / D3 P2-4: the legal-bound proof for ``policy_input_bytes``."""

    worst = worst_case_input()
    assert len(worst.canonical_bytes()) <= MAX_POLICY_INPUT_BYTES
    # Every bounded collection is at its maximum.
    assert len(worst.source_refs) == 4 and len(worst.ledger_refs) == MAX_LEDGER_REFS
    assert SuccessorPolicyInputV2.from_bytes(worst.canonical_bytes()) == worst


def test_no_single_field_change_makes_the_worst_case_longer() -> None:
    """P2-1: every enum member, every allowed pair and None are no longer."""

    from enum import Enum

    worst = worst_case_input()
    size = len(worst.canonical_bytes())
    for field, kind in (
        ("required_source", RequiredSourceState),
        ("predecessor", PredecessorState),
        ("lineage", LineageState),
        ("commitment", CommitmentState),
        ("ledger", LedgerState),
        ("feedback", FeedbackState),
        ("inverse_authorization", InverseAuthorization),
    ):
        member: Enum
        for member in kind:
            assert len(replace(worst, **{field: member}).canonical_bytes()) <= size
    for interpretation, stage in STAGE_PAIRS:
        changed = replace(worst, interpretation=interpretation, interpretation_stage=stage)
        assert len(changed.canonical_bytes()) <= size
    for field in (
        "predecessor_ref",
        "lineage_ref",
        "commitment_ref",
        "commitment_deadline_utc",
        "inverse_authorization",
        "feedback_ref",
        "verified_outcome_ref",
    ):
        assert len(replace(worst, **{field: None}).canonical_bytes()) <= size
    # Refs already use maximal kind, ID, revision and digest.
    for item in (*worst.source_refs, *worst.ledger_refs):
        assert (len(item.kind), len(item.ref_id)) == (128, 128)


STAGE_PAIRS = (
    (Interpretation.NOT_REVIEWED, InterpretationStage.NONE),
    (Interpretation.UNAVAILABLE, InterpretationStage.NONE),
    (Interpretation.SETTLED, InterpretationStage.S1),
    (Interpretation.UNCERTAIN, InterpretationStage.S1),
    (Interpretation.SETTLED, InterpretationStage.S2),
    (Interpretation.UNCERTAIN, InterpretationStage.S2),
    (Interpretation.CONTRADICTORY, InterpretationStage.S2),
)


def test_inverse_and_reverted_states_hold_on_row_2() -> None:
    """P1-1 / P2-2: A2-2 inverse states are row 2, and a MATCH_SELF admitted
    before the inverse (A2-3) is retained and held, not a conflict."""

    for ledger in (
        LedgerState.INVERSE_INTENT_ISSUED,
        LedgerState.INVERSE_CLAIMED,
        LedgerState.INVERSE_IN_DOUBT,
        LedgerState.INVERSE_POLICY_DENIED,
        LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
        LedgerState.INVERSE_VERIFIED_ABSENT,
        LedgerState.REVERTED,
    ):
        extra = (
            {"inverse_authorization": InverseAuthorization.AUTHORIZED}
            if ledger is LedgerState.REVERTED
            else {}
        )
        for feedback, ref in (
            (FeedbackState.MATCH_SELF, FEEDBACK),
            (FeedbackState.ABSENT, None),
            (FeedbackState.UNKNOWN, FEEDBACK),
        ):
            value = _verified(ledger=ledger, feedback=feedback, feedback_ref=ref, **extra)
            result = evaluate(value)
            assert (result.row, result.hold_reason) == (
                SuccessorRow.ROW_2,
                ReasonCode.INSUFFICIENT_EVIDENCE,
            ), (ledger, feedback, cross_reference_conflicts(value))


def test_policy_result_requires_exact_enums() -> None:
    with pytest.raises(TypeError):
        SuccessorPolicyResult("3", "PLAN_A", None)  # type: ignore[arg-type]


def test_from_bytes_fails_with_one_error_type() -> None:
    for payload in (b"\xff", b"[]", b"{}", b'{"a":'):
        with pytest.raises(ValueError):
            SuccessorPolicyInputV2.from_bytes(payload)


def test_sealed_bytes_round_trip_and_replay() -> None:
    for value in (_input(), _active(), _verified(), _verified(interpretation=Interpretation.UNCERTAIN)):
        payload = value.canonical_bytes()
        assert SuccessorPolicyInputV2.from_bytes(payload) == value
        assert evaluate_sealed(payload) == evaluate(value)


def test_sealed_bytes_are_strict() -> None:
    payload = _input().canonical_bytes()
    with pytest.raises(ValueError):
        SuccessorPolicyInputV2.from_bytes(payload.replace(b"_V2", b"_V1"))
    with pytest.raises(ValueError):
        SuccessorPolicyInputV2.from_bytes(payload.replace(b'"as_of":{', b'"as_of":{"extra":1,'))
    with pytest.raises(ValueError):
        SuccessorPolicyInputV2.from_bytes(b" " + payload)  # not canonical
    with pytest.raises(ValueError):
        SuccessorPolicyInputV2.from_bytes(b"{" + b" " * MAX_POLICY_INPUT_BYTES + b"}")


def test_sealed_input_fits_a_committed_decision() -> None:
    """The V2 kind and every Decision row are valid in SubjectHistory."""

    from test_r16_ip9_subject_history import _committed, _event

    assert POLICY_KIND is PolicyKind.SUCCESSOR_POLICY_V2
    for value in (_input(), _active(), _active(interpretation=Interpretation.UNCERTAIN), _verified(), _verified(interpretation=Interpretation.UNCERTAIN), _verified(feedback=FeedbackState.UNKNOWN)):
        result = evaluate(value)
        decision = _committed(
            _event(f"row-{result.row.value}", 3),
            row=result.history_row,
            kind=POLICY_KIND,
            policy_input=value.canonical_bytes(),
        )
        assert decision.policy_row is result.history_row
        assert evaluate_sealed(decision.policy_input_bytes) == result


def test_evidence_fingerprint_includes_interpretation_but_not_the_event() -> None:
    """#369 / D3 §3-3."""

    base = _active(interpretation=Interpretation.SETTLED)
    later = replace(base, event_id="event-20", event_sequence=20, as_of_r07_sequence=19, now_utc=DEADLINE)
    assert later.evidence_fingerprint() == base.evidence_fingerprint()
    assert replace(base, interpretation=Interpretation.UNCERTAIN).evidence_fingerprint() != base.evidence_fingerprint()
    assert replace(base, interpretation_stage=InterpretationStage.S1).evidence_fingerprint() != base.evidence_fingerprint()


# ------------------------------------------------- R15 interpretation map


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        (("EPISODE", "S1", "PRESENT"), (Interpretation.SETTLED, InterpretationStage.S1)),
        (("EPISODE", "S1", "UNKNOWN"), (Interpretation.UNCERTAIN, InterpretationStage.S1)),
        (("EPISODE", "S2", "PRESENT"), (Interpretation.SETTLED, InterpretationStage.S2)),
        (("EPISODE", "S2", "CONTESTED"), (Interpretation.UNCERTAIN, InterpretationStage.S2)),
        (("EPISODE", "S2", "UNKNOWN"), (Interpretation.UNCERTAIN, InterpretationStage.S2)),
        (("EPISODE", "S2", "CONTRADICTORY"), (Interpretation.CONTRADICTORY, InterpretationStage.S2)),
        (("NOT_ELIGIBLE", "NONE", None), (Interpretation.NOT_REVIEWED, InterpretationStage.NONE)),
        (("UNAVAILABLE", "NONE", None), (Interpretation.UNAVAILABLE, InterpretationStage.NONE)),
        # Outside the closed set: fail closed.
        (("EPISODE", "S1", "CONTESTED"), (Interpretation.UNAVAILABLE, InterpretationStage.NONE)),
        (("EPISODE", "NONE", None), (Interpretation.UNAVAILABLE, InterpretationStage.NONE)),
        (("NOT_ELIGIBLE", "S1", "PRESENT"), (Interpretation.UNAVAILABLE, InterpretationStage.NONE)),
        (("episode", "S2", "PRESENT"), (Interpretation.UNAVAILABLE, InterpretationStage.NONE)),
        ((None, None, None), (Interpretation.UNAVAILABLE, InterpretationStage.NONE)),
    ],
)
def test_interpretation_from_r15(read, expected) -> None:  # type: ignore[no-untyped-def]
    assert interpretation_from_r15(*read) == expected
    interpretation, stage = expected
    # Every mapped value is a constructible policy input.
    assert _input(interpretation=interpretation, interpretation_stage=stage)


def test_every_closed_r15_read_maps_without_unavailable() -> None:
    """IP-8a's closed R15InterpretationRead combinations all map exactly."""

    from suzka.r15.interpretation_reader import InterpretationOutcome, R15InterpretationRead

    valid = []
    for outcome, stage, status in product(
        InterpretationOutcome,
        ("NONE", "S1", "S2"),
        (None, "PRESENT", "CONTESTED", "CONTRADICTORY", "UNKNOWN"),
    ):
        try:
            valid.append(R15InterpretationRead(outcome, stage, status))
        except ValueError:
            continue
    assert valid
    for read in valid:
        mapped = interpretation_from_r15(read.outcome.value, read.stage, read.status)
        unavailable = read.outcome is InterpretationOutcome.UNAVAILABLE
        assert (mapped[0] is Interpretation.UNAVAILABLE) is unavailable
