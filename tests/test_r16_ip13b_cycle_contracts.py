"""IP-13b: R16-owned cycle results (D-R19 §2 / §4). FIXTURES ONLY."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from typing import Any

import pytest

from suzka.r16 import cycle_contracts
from suzka.r16.cycle_contracts import (
    MAX_RECORD_BYTES,
    ExistingDecisionReceiptV1,
    FrozenPredecessor,
    HoldReason,
    HoldReceiptV1,
    LogicalCycleResolutionV1,
    ResultKind,
    wake_review_intent_key,
)
from suzka.r16.history_codec import PolicyKind, PolicyRow
from suzka.r16.successor_policy import SuccessorRow

H = "a" * 64
CYCLE = "c" * 64
FP = "f" * 64
CLAIM = "wake-claim-" + "1" * 64
EVENT_C = "00000000-0000-4000-8000-000000000001"
EVENT_H = "00000000-0000-4000-8000-000000000002"
DECISION = "decision-" + "d" * 64
AT = "2026-11-16T00:00:00Z"
V2 = PolicyKind.SUCCESSOR_POLICY_V2
V3 = PolicyKind.SUCCESSOR_POLICY_V3
KEY = wake_review_intent_key(CYCLE, FrozenPredecessor.absence(9), V2, FP)


def _hold(**changes: Any) -> HoldReceiptV1:
    base = HoldReceiptV1(
        logical_cycle_id=CYCLE,
        review_intent_key=KEY,
        claim_id=CLAIM,
        claim_digest=H,
        claim_event_id=EVENT_C,
        hold_event_id=EVENT_H,
        hold_sequence=30,
        reason=HoldReason.STALE_SOURCE,
        policy_row=None,
        current_fingerprint="9" * 64,
        sealed_input_digest=H,
        prepare_checked_at_utc=AT,
    )
    return replace(base, **changes)


def _existing(**changes: Any) -> ExistingDecisionReceiptV1:
    base = ExistingDecisionReceiptV1(
        logical_cycle_id=CYCLE,
        review_intent_key=KEY,
        claim_id=CLAIM,
        claim_digest=H,
        event_id=EVENT_H,
        event_sequence=30,
        prepare_checked_at_utc=AT,
        current_fingerprint=FP,
        head_decision_id=DECISION,
        head_committed_digest=H,
        reproduced_policy_row=PolicyRow.ROW_3,
    )
    return replace(base, **changes)


def _tamper(payload: bytes, **changes: object) -> bytes:
    row = json.loads(payload)
    row.update(changes)
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


# ------------------------------------------------------- review intent key


def test_review_intent_key_is_independent_of_claim_event_and_time() -> None:
    """#369 A2-1: only cycle, frozen predecessor, policy identity and
    frozen fingerprint go in."""

    assert tuple(inspect.signature(wake_review_intent_key).parameters) == (
        "logical_cycle_id",
        "predecessor",
        "policy_kind",
        "evidence_fingerprint",
    )
    assert wake_review_intent_key(CYCLE, FrozenPredecessor.absence(9), V2, FP) == KEY
    changed = (
        wake_review_intent_key("e" * 64, FrozenPredecessor.absence(9), V2, FP),
        wake_review_intent_key(CYCLE, FrozenPredecessor.absence(10), V2, FP),
        wake_review_intent_key(CYCLE, FrozenPredecessor.decision(DECISION, H), V2, FP),
        wake_review_intent_key(CYCLE, FrozenPredecessor.absence(9), V3, FP),
        wake_review_intent_key(CYCLE, FrozenPredecessor.absence(9), V2, "e" * 64),
    )
    assert len({KEY, *changed}) == 6
    with pytest.raises(ValueError):
        wake_review_intent_key(CYCLE, FrozenPredecessor.absence(9), PolicyKind.SUCCESSOR_POLICY_V1, FP)


def test_frozen_predecessor_is_a_decision_or_a_witnessed_absence() -> None:
    for value in (FrozenPredecessor.absence(9), FrozenPredecessor.decision(DECISION, H)):
        assert FrozenPredecessor.from_value(value.canonical_value()) == value
    for bad in (
        (None, None, None),
        (DECISION, H, 9),
        (DECISION, None, None),
        (None, H, 9),
        ("decision-x", H, None),
        (None, None, 0),
    ):
        with pytest.raises(ValueError):
            FrozenPredecessor(*bad)


# ------------------------------------------------------------ hold receipt


def test_hold_receipt_round_trips_with_stable_identity() -> None:
    receipt = _hold()
    assert HoldReceiptV1.from_json(receipt.canonical_bytes()) == receipt
    assert receipt.hold_id.startswith("cycle-hold-")
    assert receipt.hold_id != _hold(hold_sequence=31).hold_id
    assert receipt.hold_digest != receipt.hold_id[len("cycle-hold-") :]
    assert len(receipt.canonical_bytes()) <= MAX_RECORD_BYTES


def test_policy_row_exactly_for_a_policy_hold_row() -> None:
    for row in (SuccessorRow.ROW_0, SuccessorRow.ROW_1, SuccessorRow.ROW_2, SuccessorRow.ROW_2B, SuccessorRow.ROW_7):
        receipt = _hold(reason=HoldReason.POLICY_HOLD, policy_row=row)
        assert HoldReceiptV1.from_json(receipt.canonical_bytes()) == receipt
    for row in (SuccessorRow.ROW_3, SuccessorRow.ROW_4, SuccessorRow.ROW_5, SuccessorRow.ROW_6):
        with pytest.raises(ValueError, match="HOLD row"):
            _hold(reason=HoldReason.POLICY_HOLD, policy_row=row)
    with pytest.raises(ValueError, match="exactly"):
        _hold(reason=HoldReason.POLICY_HOLD)
    for reason in HoldReason:
        if reason is not HoldReason.POLICY_HOLD:
            with pytest.raises(ValueError, match="exactly"):
                _hold(reason=reason, policy_row=SuccessorRow.ROW_0)


def test_hold_receipt_refuses_bad_bindings() -> None:
    with pytest.raises(ValueError):
        _hold(claim_id="claim-" + "1" * 64)
    with pytest.raises(ValueError, match="later event"):
        _hold(hold_event_id=EVENT_C)
    with pytest.raises(ValueError):
        _hold(prepare_checked_at_utc="2026-11-16 00:00:00")
    with pytest.raises(TypeError):
        _hold(reason="STALE_SOURCE")


def test_hold_receipt_bytes_are_strict() -> None:
    payload = _hold().canonical_bytes()
    for changes in ({"hold_id": "cycle-hold-" + "0" * 64}, {"schema_version": 2}, {"reason": "REFUSED"}):
        with pytest.raises((ValueError, TypeError)):
            HoldReceiptV1.from_json(_tamper(payload, **changes))
    with pytest.raises(ValueError):
        HoldReceiptV1.from_json(b" " + payload)
    with pytest.raises(ValueError):
        HoldReceiptV1.from_json(b"{" + b" " * MAX_RECORD_BYTES + b"}")


# ------------------------------------------------- existing-decision receipt


def test_existing_receipt_round_trips_and_needs_a_decision_row() -> None:
    receipt = _existing()
    assert ExistingDecisionReceiptV1.from_json(receipt.canonical_bytes()) == receipt
    assert receipt.receipt_id.startswith("cycle-existing-")
    with pytest.raises(TypeError):
        _existing(reproduced_policy_row=SuccessorRow.ROW_0)
    with pytest.raises(ValueError):
        _existing(head_decision_id="decision-x")
    with pytest.raises(ValueError):
        ExistingDecisionReceiptV1.from_json(_tamper(receipt.canonical_bytes(), reproduced_policy_row="2"))


# --------------------------------------------------------------- resolution


def test_resolution_points_at_exactly_one_result() -> None:
    hold = LogicalCycleResolutionV1.of_hold(H, _hold())
    existing = LogicalCycleResolutionV1.of_existing(H, _existing())
    assert (hold.result_kind, hold.result_id, hold.result_digest) == (
        ResultKind.HOLD,
        _hold().hold_id,
        _hold().hold_digest,
    )
    assert (existing.result_kind, existing.result_id) == (ResultKind.EXISTING_DECISION, _existing().receipt_id)
    for value in (hold, existing):
        assert LogicalCycleResolutionV1.from_json(value.canonical_bytes()) == value
    # One resolution slot per (incarnation, cycle), whatever the result.
    assert hold.resolution_id == existing.resolution_id
    assert hold.resolution_digest != existing.resolution_digest
    assert replace(hold, incarnation_id="b" * 64).resolution_id != hold.resolution_id


def test_resolution_result_id_must_match_its_kind() -> None:
    for kind, wrong in (
        (ResultKind.DECISION_ISSUED, _hold().hold_id),
        (ResultKind.HOLD, DECISION),
        (ResultKind.EXISTING_DECISION, _hold().hold_id),
    ):
        with pytest.raises(ValueError):
            LogicalCycleResolutionV1(H, CYCLE, KEY, CLAIM, H, kind, wrong, H)
    LogicalCycleResolutionV1(H, CYCLE, KEY, CLAIM, H, ResultKind.DECISION_ISSUED, DECISION, H)


def test_resolution_of_a_decision_takes_its_identity() -> None:
    from test_r16_ip9_subject_history import _committed, _event

    decision = _committed(_event("wake", 3))
    resolution = LogicalCycleResolutionV1.of_decision(CYCLE, CLAIM, H, decision)
    assert (
        resolution.incarnation_id,
        resolution.review_intent_key,
        resolution.result_id,
        resolution.result_digest,
    ) == (decision.incarnation_id, decision.review_intent_key, decision.decision_id, decision.content_digest)
    with pytest.raises(TypeError):
        LogicalCycleResolutionV1.of_decision(CYCLE, CLAIM, H, _hold())  # type: ignore[arg-type]


# ------------------------------------------------------------------ purity


def test_cycle_contracts_stay_pure_and_do_not_import_r19() -> None:
    tree = ast.parse(inspect.getsource(cycle_contracts))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {
        "__future__",
        "dataclasses",
        "datetime",
        "enum",
        "typing",
        "suzka.r16.contracts",
        "suzka.r16.history_codec",
        "suzka.r16.successor_policy",
    }
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))


@pytest.mark.parametrize(
    "decode",
    [HoldReceiptV1.from_json, ExistingDecisionReceiptV1.from_json, LogicalCycleResolutionV1.from_json],
)
def test_every_decoder_refuses_oversize_and_non_bytes(decode: Any) -> None:
    with pytest.raises(ValueError, match="byte bound"):
        decode(b"{" + b" " * MAX_RECORD_BYTES + b"}")
    with pytest.raises(TypeError):
        decode("{}")


def test_r16_ids_and_digests_are_distinct() -> None:
    hold, existing = _hold(), _existing()
    resolution = LogicalCycleResolutionV1.of_hold(H, hold)
    values = [
        KEY, hold.hold_id, hold.hold_digest, existing.receipt_id, existing.receipt_digest,
        resolution.resolution_id, resolution.resolution_digest,
    ]
    tails = [value[-64:] for value in values]
    assert len(set(tails)) == len(tails)
