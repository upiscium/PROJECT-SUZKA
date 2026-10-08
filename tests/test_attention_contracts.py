"""Contract closure, source binding, and continuity evidence tests for R14 U1."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import sys

import pytest

from suzka.attention.common import (
    ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_RENDERED_ITEM_BYTES,
    ATTENTION_PROMPT_BUDGET_BYTES,
    AttentionRevisionReason,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
    attention_focus_capacity,
    prompt_budget_accepts,
)
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionCandidateProjection,
    AttentionContinuity,
    AttentionEvent,
    AttentionEventReceipt,
    AttentionRevisionEvidence,
    AttentionSignalVector,
    AttentionSourceWitness,
    AttentionTarget,
    attention_state_digest,
)


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _target(reference: str = "a" * 64) -> AttentionTarget:
    return AttentionTarget(AttentionTargetKind.MOTIVATION, reference)


def _event(
    event_id: str = "event-1",
    sequence: int = 1,
    occurred_at: datetime = NOW,
) -> AttentionEvent:
    return AttentionEvent(event_id, sequence, occurred_at)


def _source(
    target: AttentionTarget | None = None,
    *,
    event: AttentionEvent | None = None,
) -> AttentionSourceWitness:
    candidate = target or _target()
    current_event = event or _event()
    source_kind = {
        AttentionTargetKind.WORKING_MEMORY: AttentionSourceKind.WORKING_MEMORY,
        AttentionTargetKind.MOTIVATION: AttentionSourceKind.MOTIVATION,
        AttentionTargetKind.GOAL: AttentionSourceKind.GOAL,
        AttentionTargetKind.COMMITMENT: AttentionSourceKind.COMMITMENT,
    }[candidate.kind]
    digest_kind = (
        SourceDigestKind.ATTENTION_PROJECTION
        if candidate.kind is AttentionTargetKind.WORKING_MEMORY
        else SourceDigestKind.UPSTREAM_AUTHORITY
    )
    return AttentionSourceWitness(
        kind=source_kind,
        reference=candidate.reference,
        revision=1,
        digest="b" * 64,
        digest_kind=digest_kind,
        target_kind=candidate.kind,
        target_reference=candidate.reference,
        source_event_id=None
        if candidate.kind is AttentionTargetKind.WORKING_MEMORY
        else current_event.event_id,
        source_event_sequence=None
        if candidate.kind is AttentionTargetKind.WORKING_MEMORY
        else current_event.event_sequence,
        source_occurred_at=None
        if candidate.kind is AttentionTargetKind.WORKING_MEMORY
        else current_event.occurred_at,
    )


def _continuity() -> AttentionContinuity:
    bootstrap = AttentionContinuity.bootstrap()
    event = _event()
    target = _target()
    source = _source(target, event=event)
    candidate = AttentionCandidateContinuity(
        target=target,
        source=source,
        availability=CandidateAvailability.ELIGIBLE,
        habituation=125,
        inhibition=0,
        focused_event_count=1,
        unattended_event_count=0,
    )
    focused_ids = (candidate.candidate_id,)
    state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=1,
        last_event=event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        unfinished_ids=(),
    )
    history = AttentionRevisionEvidence(
        revision=1,
        event=event,
        previous_state_digest=bootstrap.state_digest,
        state_digest=state_digest,
        previous_revision_digest=None,
        focused_ids=focused_ids,
        unfinished_ids=(),
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    receipt = AttentionEventReceipt(
        event=event,
        input_digest="c" * 64,
        result_state_digest=state_digest,
    )
    return AttentionContinuity(
        revision=1,
        last_event=event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        revision_history=(history,),
        receipts=(receipt,),
    )


def test_projection_preserves_known_wm_metadata_without_resolved_content() -> None:
    target = AttentionTarget(AttentionTargetKind.WORKING_MEMORY, "wm-" + "f" * 64)
    projection = AttentionCandidateProjection._create(
        target=target,
        source=_source(target),
        signals=AttentionSignalVector(activation=0.0, salience=0.5),
        event=_event(),
        availability=CandidateAvailability.UNAVAILABLE,
        rendered_bytes=None,
        rendered_digest=None,
    )
    assert projection.signals.activation == 0.0
    assert projection.signals.salience == 0.5
    assert projection.signals.context_compatibility is None
    assert projection.rendered_bytes is None


def test_projection_preserves_oversized_byte_witness_for_policy_omission() -> None:
    target = AttentionTarget(AttentionTargetKind.WORKING_MEMORY, "wm-" + "f" * 64)
    arguments = {
        "target": target,
        "source": _source(target),
        "signals": AttentionSignalVector(activation=1.0, salience=1.0),
        "event": _event(),
        "availability": CandidateAvailability.ELIGIBLE,
        "rendered_digest": "c" * 64,
    }
    oversized = AttentionCandidateProjection._create(
        **arguments, rendered_bytes=ATTENTION_PROMPT_BUDGET_BYTES + 1
    )
    assert oversized.rendered_bytes == ATTENTION_PROMPT_BUDGET_BYTES + 1
    maximum = AttentionCandidateProjection._create(
        **arguments, rendered_bytes=ATTENTION_MAX_RENDERED_ITEM_BYTES
    )
    assert maximum.rendered_bytes == ATTENTION_MAX_RENDERED_ITEM_BYTES
    with pytest.raises(ValueError, match="bounded rendered byte accounting"):
        AttentionCandidateProjection._create(
            **arguments, rendered_bytes=ATTENTION_MAX_RENDERED_ITEM_BYTES + 1
        )


def _continuity_with_revisions(
    *,
    attention_revision: int,
    source_revision: int,
) -> AttentionContinuity:
    bootstrap = AttentionContinuity.bootstrap()
    events = tuple(
        _event(
            f"attention-event-{sequence}",
            sequence,
            NOW + timedelta(seconds=sequence),
        )
        for sequence in range(1, attention_revision + 1)
    )
    current_event = events[-1]
    target = _target()
    source = AttentionSourceWitness(
        kind=AttentionSourceKind.MOTIVATION,
        reference=target.reference,
        revision=source_revision,
        digest="b" * 64,
        digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
        target_kind=target.kind,
        target_reference=target.reference,
    )
    candidate = AttentionCandidateContinuity(
        target=target,
        source=source,
        availability=CandidateAvailability.ELIGIBLE,
        focused_event_count=1,
    )
    focused_ids = (candidate.candidate_id,)
    current_state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=attention_revision,
        last_event=current_event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        unfinished_ids=(),
    )
    history: list[AttentionRevisionEvidence] = []
    receipts: list[AttentionEventReceipt] = []
    previous_state_digest = bootstrap.state_digest
    previous_revision_digest: str | None = None
    previous_receipt_digest: str | None = None
    for revision, event in enumerate(events, start=1):
        revision_state_digest = (
            current_state_digest
            if revision == attention_revision
            else f"{revision:064x}"
        )
        evidence = AttentionRevisionEvidence(
            revision=revision,
            event=event,
            previous_state_digest=previous_state_digest,
            state_digest=revision_state_digest,
            previous_revision_digest=previous_revision_digest,
            focused_ids=focused_ids,
            unfinished_ids=(),
            reason=AttentionRevisionReason.STATE_UPDATE,
        )
        receipt = AttentionEventReceipt(
            event=event,
            input_digest="c" * 64,
            result_state_digest=revision_state_digest,
            previous_receipt_digest=previous_receipt_digest,
        )
        history.append(evidence)
        receipts.append(receipt)
        previous_state_digest = revision_state_digest
        previous_revision_digest = evidence.record_digest
        previous_receipt_digest = receipt.receipt_digest
    return AttentionContinuity(
        revision=attention_revision,
        last_event=current_event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        revision_history=tuple(history),
        receipts=tuple(receipts),
    )


def _continuity_with_two_candidate_streaks() -> AttentionContinuity:
    bootstrap = AttentionContinuity.bootstrap()
    event = _event()
    focused_target = _target()
    unattended_target = AttentionTarget(AttentionTargetKind.GOAL, "b" * 64)
    focused_candidate = AttentionCandidateContinuity(
        target=focused_target,
        source=_source(focused_target, event=event),
        availability=CandidateAvailability.ELIGIBLE,
        focused_event_count=1,
    )
    unattended_candidate = AttentionCandidateContinuity(
        target=unattended_target,
        source=_source(unattended_target, event=event),
        availability=CandidateAvailability.UNAVAILABLE,
        unattended_event_count=1,
    )
    candidates = tuple(
        sorted(
            (focused_candidate, unattended_candidate),
            key=lambda candidate: candidate.candidate_id,
        )
    )
    focused_ids = (focused_candidate.candidate_id,)
    state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=1,
        last_event=event,
        candidates=candidates,
        focused_ids=focused_ids,
        unfinished_ids=(),
    )
    history = AttentionRevisionEvidence(
        revision=1,
        event=event,
        previous_state_digest=bootstrap.state_digest,
        state_digest=state_digest,
        previous_revision_digest=None,
        focused_ids=focused_ids,
        unfinished_ids=(),
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    receipt = AttentionEventReceipt(
        event=event,
        input_digest="c" * 64,
        result_state_digest=state_digest,
    )
    return AttentionContinuity(
        revision=1,
        last_event=event,
        candidates=candidates,
        focused_ids=focused_ids,
        revision_history=(history,),
        receipts=(receipt,),
    )


def test_target_ids_are_typed_and_source_references_are_closed() -> None:
    motivation = AttentionTarget(AttentionTargetKind.MOTIVATION, "a" * 64)
    goal = AttentionTarget(AttentionTargetKind.GOAL, "a" * 64)
    working_memory = AttentionTarget(
        AttentionTargetKind.WORKING_MEMORY,
        "wm-" + "a" * 64,
    )

    assert motivation.candidate_id != goal.candidate_id
    assert len(motivation.candidate_id) == 64
    assert len(working_memory.candidate_id) == 64
    with pytest.raises(ValueError, match="wm-"):
        AttentionTarget(AttentionTargetKind.WORKING_MEMORY, "a" * 64)
    with pytest.raises(ValueError, match="digest"):
        AttentionTarget(AttentionTargetKind.GOAL, "G" * 64)
    with pytest.raises(TypeError, match="AttentionTargetKind"):
        AttentionTarget("motivation", "a" * 64)  # type: ignore[arg-type]


def test_source_witness_binds_target_and_enforces_digest_provenance() -> None:
    target = _target()
    witness = _source(target)
    witness.validate_for(target, _event())
    assert AttentionSourceWitness.from_canonical_value(witness.canonical_value()) == witness
    tampered_witness = witness.canonical_value()
    tampered_witness["digest"] = "c" * 64
    with pytest.raises(ValueError, match="witness digest"):
        AttentionSourceWitness.from_canonical_value(tampered_witness)

    wrong_target = _target("d" * 64)
    with pytest.raises(ValueError, match="primary source reference must match"):
        AttentionCandidateProjection._create(
            target=wrong_target,
            source=witness,
            signals=AttentionSignalVector(),
            event=_event(),
            availability=CandidateAvailability.UNAVAILABLE,
            rendered_bytes=None,
            rendered_digest=None,
        )

    with pytest.raises(ValueError, match="digest kind"):
        AttentionSourceWitness(
            kind=AttentionSourceKind.MOTIVATION,
            reference=target.reference,
            revision=1,
            digest="b" * 64,
            digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
            target_kind=target.kind,
            target_reference=target.reference,
        )
    secondary = AttentionSourceWitness(
        kind=AttentionSourceKind.EXPERIENCE,
        reference="experience-1",
        revision=1,
        digest="c" * 64,
        digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
        target_kind=target.kind,
        target_reference=target.reference,
    )
    with pytest.raises(ValueError, match="primary source kind"):
        AttentionCandidateContinuity(
            target=target,
            source=secondary,
            availability=CandidateAvailability.ELIGIBLE,
        )
    wm_target = AttentionTarget(AttentionTargetKind.WORKING_MEMORY, "wm-" + "c" * 64)
    wm_witness = AttentionSourceWitness(
        kind=AttentionSourceKind.WORKING_MEMORY,
        reference=wm_target.reference,
        revision=0,
        digest="e" * 64,
        digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
        target_kind=wm_target.kind,
        target_reference=wm_target.reference,
    )
    wm_witness.validate_for(wm_target, _event())
    fabricated_wm_event = AttentionSourceWitness(
        kind=AttentionSourceKind.WORKING_MEMORY,
        reference=wm_target.reference,
        revision=0,
        digest="e" * 64,
        digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
        target_kind=wm_target.kind,
        target_reference=wm_target.reference,
        source_event_id="event-1",
        source_event_sequence=1,
        source_occurred_at=NOW,
    )
    with pytest.raises(ValueError, match="no upstream event identity"):
        fabricated_wm_event.validate_primary_for_target(wm_target)


def test_source_event_triple_is_atomic_and_cannot_be_future_evidence() -> None:
    target = _target()
    with pytest.raises(ValueError, match="supplied together"):
        AttentionSourceWitness(
            kind=AttentionSourceKind.MOTIVATION,
            reference=target.reference,
            revision=0,
            digest="b" * 64,
            digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
            target_kind=target.kind,
            target_reference=target.reference,
            source_event_id="event-future",
        )

    future = AttentionEvent("event-2", 2, NOW + timedelta(seconds=1))
    witness = _source(target, event=future)
    with pytest.raises(ValueError, match="future"):
        AttentionCandidateProjection._create(
            target=target,
            source=witness,
            signals=AttentionSignalVector(),
            event=_event(),
            availability=CandidateAvailability.UNAVAILABLE,
            rendered_bytes=None,
            rendered_digest=None,
        )


def test_unknown_signals_remain_distinct_from_measured_zero() -> None:
    unknown = AttentionSignalVector()
    measured = AttentionSignalVector(activation=0.0)
    negative_zero = AttentionSignalVector(activation=-0.0)

    assert unknown.activation is None
    assert measured.activation == 0.0
    assert unknown.canonical_value()["activation"] is None
    assert measured.canonical_value()["activation"] == 0.0.hex()
    assert negative_zero == measured
    assert unknown.canonical_value() != measured.canonical_value()
    assert AttentionSignalVector.from_canonical_value(
        measured.canonical_value()
    ) == measured
    with pytest.raises(TypeError, match="finite number"):
        AttentionSignalVector(activation=True)
    with pytest.raises(ValueError, match="finite"):
        AttentionSignalVector(activation=float("nan"))


def test_projection_is_closed_reference_only_and_preserves_signal_missingness() -> None:
    target = AttentionTarget(AttentionTargetKind.WORKING_MEMORY, "wm-" + "f" * 64)
    event = _event()
    projection = AttentionCandidateProjection._create(
        target=target,
        source=_source(target, event=event),
        signals=AttentionSignalVector(activation=0.0, salience=0.5),
        event=event,
        availability=CandidateAvailability.ELIGIBLE,
        rendered_bytes=12,
        rendered_digest="f" * 64,
    )
    value = projection.canonical_value()
    assert value["candidate_id"] == target.candidate_id
    assert value["rendered_bytes"] == 12
    assert value["rendered_digest"] == "f" * 64
    assert "rendered_text" not in value
    assert not hasattr(projection, "rendered_text")
    assert projection.validated_copy() == projection

    corrupted_signals = projection
    original_projection_digest = corrupted_signals.projection_digest
    object.__setattr__(corrupted_signals.signals, "activation", 0.25)
    with pytest.raises(ValueError, match="projection digest"):
        corrupted_signals.__post_init__()
    with pytest.raises(ValueError, match="projection digest"):
        corrupted_signals.validated_copy()
    assert corrupted_signals.signals.activation == 0.25
    assert corrupted_signals.projection_digest == original_projection_digest

    corrupted_bytes = AttentionCandidateProjection._create(
        target=target,
        source=_source(target, event=event),
        signals=AttentionSignalVector(activation=0.0, salience=0.5),
        event=event,
        availability=CandidateAvailability.ELIGIBLE,
        rendered_bytes=12,
        rendered_digest="f" * 64,
    )
    object.__setattr__(corrupted_bytes, "rendered_bytes", 13)
    with pytest.raises(ValueError, match="projection digest"):
        corrupted_bytes.validated_copy()
    assert corrupted_bytes.rendered_bytes == 13
    assert corrupted_bytes.projection_digest == original_projection_digest

    corrupted_declared_digest = AttentionCandidateProjection._create(
        target=target,
        source=_source(target, event=event),
        signals=AttentionSignalVector(activation=0.0, salience=0.5),
        event=event,
        availability=CandidateAvailability.ELIGIBLE,
        rendered_bytes=12,
        rendered_digest="f" * 64,
    )
    object.__setattr__(corrupted_declared_digest, "projection_digest", "0" * 64)
    with pytest.raises(ValueError, match="projection digest"):
        corrupted_declared_digest.validated_copy()
    assert corrupted_declared_digest.projection_digest == "0" * 64

    unavailable = AttentionCandidateProjection._create(
        target=target,
        source=_source(target, event=event),
        signals=AttentionSignalVector(activation=0.0),
        event=event,
        availability=CandidateAvailability.UNAVAILABLE,
        rendered_bytes=None,
        rendered_digest=None,
    )
    assert unavailable.signals.activation == 0.0
    assert unavailable.signals.context_compatibility is None
    with pytest.raises(ValueError, match="no reviewed adapter"):
        AttentionCandidateProjection._create(
            target=target,
            source=_source(target, event=event),
            signals=AttentionSignalVector(urgency=0.5),
            event=event,
            availability=CandidateAvailability.ELIGIBLE,
            rendered_bytes=12,
            rendered_digest="f" * 64,
        )
    with pytest.raises(ValueError, match="rendered row"):
        AttentionCandidateProjection._create(
            target=target,
            source=_source(target, event=event),
            signals=AttentionSignalVector(),
            event=event,
            availability=CandidateAvailability.INACTIVE,
            rendered_bytes=12,
            rendered_digest="f" * 64,
        )


def test_focus_and_prompt_budget_policy_use_only_global_emotion_arousal() -> None:
    assert attention_focus_capacity(0.0) == ATTENTION_MAX_FOCUS
    assert attention_focus_capacity(0.749999) == ATTENTION_MAX_FOCUS
    assert attention_focus_capacity(0.75) == ATTENTION_HIGH_AROUSAL_MAX_FOCUS
    assert attention_focus_capacity(1.0) == ATTENTION_HIGH_AROUSAL_MAX_FOCUS
    assert prompt_budget_accepts(ATTENTION_PROMPT_BUDGET_BYTES)
    assert not prompt_budget_accepts(ATTENTION_PROMPT_BUDGET_BYTES + 1)
    with pytest.raises(TypeError):
        attention_focus_capacity(True)
    with pytest.raises(ValueError):
        prompt_budget_accepts(True)


def test_continuity_roundtrip_closure_and_independent_receipt_digest() -> None:
    current = _continuity()
    restored = AttentionContinuity.from_json(current.canonical_bytes())
    assert restored == current
    assert restored.state_digest == current.state_digest
    assert restored.authority_digest == current.authority_digest

    original_receipt = current.receipts[0]
    changed_receipt = replace(original_receipt, input_digest="d" * 64)
    receipt_changed_state = replace(current, receipts=(changed_receipt,))
    assert receipt_changed_state.state_digest == current.state_digest
    assert receipt_changed_state.authority_digest != current.authority_digest

    altered = current.canonical_value()
    altered["unknown"] = "field"
    with pytest.raises(ValueError, match="unknown fields"):
        AttentionContinuity.from_canonical_value(altered)
    tampered = current.canonical_value()
    tampered["authority_digest"] = "0" * 64
    with pytest.raises(ValueError, match="authority digest"):
        AttentionContinuity.from_canonical_value(tampered)
    nested_tamper = current.canonical_value()
    nested_tamper["candidates"][0]["habituation"] = 999  # type: ignore[index]
    with pytest.raises(ValueError, match="record digest"):
        AttentionContinuity.from_canonical_value(nested_tamper)
    with pytest.raises(ValueError, match="canonical form"):
        AttentionContinuity.from_json(current.canonical_bytes() + b" ")
    with pytest.raises(ValueError, match="duplicate"):
        AttentionContinuity.from_json(b'{"schema_version":1,"schema_version":1}')


def test_wire_size_gate_runs_before_json_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    import suzka.attention.contracts as contracts_module
    from suzka.attention.bounds import ATTENTION_STATE_MAX_VALUE_BYTES

    def parser_must_not_run(*args: object, **kwargs: object) -> object:
        pytest.fail("JSON parser ran before the Attention size gate")

    monkeypatch.setattr(contracts_module.json, "loads", parser_must_not_run)
    with pytest.raises(ValueError, match="persisted byte bound"):
        AttentionContinuity.from_json(b" " * (ATTENTION_STATE_MAX_VALUE_BYTES + 1))
    with pytest.raises(ValueError, match="persisted byte bound"):
        AttentionContinuity.from_canonical_value(
            {"oversized": "x" * ATTENTION_STATE_MAX_VALUE_BYTES}
        )


@pytest.mark.parametrize(
    ("field", "limit"),
    (
        ("candidates", None),
        ("revision_history", 16),
        ("receipts", 256),
        ("focused_ids", 16),
        ("unfinished_ids", 16),
    ),
)
def test_wire_array_counts_are_gated_before_nested_constructors(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    limit: int | None,
) -> None:
    from suzka.attention.bounds import attention_candidate_capacity

    count_limit = attention_candidate_capacity() if limit is None else limit
    root: dict[str, object] = {
        "authority_digest": "a" * 64,
        "candidates": [],
        "focused_ids": [],
        "last_event": None,
        "policy_version": 1,
        "receipt_anchor": None,
        "receipts": [],
        "revision": 0,
        "revision_anchor": None,
        "revision_history": [],
        "schema_version": 1,
        "state_digest": "b" * 64,
        "unfinished_ids": [],
    }
    root[field] = [None] * (count_limit + 1)  # type: ignore[operator]

    def nested_constructor_must_not_run(cls: type[object], value: object) -> object:
        pytest.fail("nested contract constructor ran before its array count gate")

    monkeypatch.setattr(
        AttentionCandidateContinuity,
        "from_canonical_value",
        classmethod(nested_constructor_must_not_run),
    )
    monkeypatch.setattr(
        AttentionRevisionEvidence,
        "from_canonical_value",
        classmethod(nested_constructor_must_not_run),
    )
    monkeypatch.setattr(
        AttentionEventReceipt,
        "from_canonical_value",
        classmethod(nested_constructor_must_not_run),
    )
    with pytest.raises(ValueError, match="exceeds its (source-authority bound|retained window|bound)"):
        AttentionContinuity.from_canonical_value(root)


def test_deeply_nested_json_fails_closed_before_recursive_construction() -> None:
    deeply_nested = b"[" * 65 + b"0" + b"]" * 65
    with pytest.raises(ValueError, match="nesting bound"):
        AttentionContinuity.from_json(deeply_nested)


def test_continuity_rejects_future_versions_malformed_counters_and_dangling_focus() -> None:
    current = _continuity()
    with pytest.raises(ValueError, match="schema version"):
        replace(current, schema_version=2)
    with pytest.raises(ValueError, match="policy version"):
        replace(current, policy_version=2)
    with pytest.raises(ValueError, match="fixed-point"):
        replace(
            current.candidates[0],
            habituation=1_000_001,
        )
    with pytest.raises(ValueError, match="bounded"):
        replace(current.candidates[0], focused_event_count=True)
    with pytest.raises(ValueError, match="dangling"):
        replace(current, focused_ids=("0" * 64,))
    with pytest.raises(ValueError, match="eligible"):
        inactive = replace(
            current.candidates[0],
            availability=CandidateAvailability.INACTIVE,
        )
        replace(current, candidates=(inactive,))
    with pytest.raises(ValueError, match="uniquely sorted"):
        replace(
            current,
            candidates=(current.candidates[0], current.candidates[0]),
        )
    noncanonical_ids = current.canonical_value()
    noncanonical_ids["focused_ids"] = [current.focused_ids[0], current.focused_ids[0]]
    with pytest.raises(ValueError, match="sorted and unique"):
        AttentionContinuity.from_canonical_value(noncanonical_ids)


def test_focus_and_unattended_counters_are_current_streaks() -> None:
    current = _continuity()
    candidate = current.candidates[0]
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(candidate, unattended_event_count=1)
    with pytest.raises(ValueError, match="only eligible"):
        replace(candidate, availability=CandidateAvailability.UNAVAILABLE)
    inactive_streak = replace(
        candidate,
        availability=CandidateAvailability.INACTIVE,
        focused_event_count=0,
        unattended_event_count=1,
    )
    assert inactive_streak.unattended_event_count == 1

    no_longer_focused = replace(
        candidate,
        focused_event_count=0,
        unattended_event_count=1,
    )
    with pytest.raises(ValueError, match="exactly match"):
        replace(current, candidates=(no_longer_focused,))

    over_event_count = replace(candidate, focused_event_count=2)
    with pytest.raises(ValueError, match="candidate event streak"):
        replace(current, candidates=(over_event_count,))


def test_attention_and_source_revisions_are_independent_namespaces() -> None:
    attention_ahead = _continuity_with_revisions(
        attention_revision=2,
        source_revision=0,
    )
    source_ahead = _continuity_with_revisions(
        attention_revision=1,
        source_revision=5,
    )

    assert attention_ahead.revision == 2
    assert attention_ahead.candidates[0].source.revision == 0
    assert source_ahead.revision == 1
    assert source_ahead.candidates[0].source.revision == 5
    for state, source_revision in ((attention_ahead, 0), (source_ahead, 5)):
        target = state.candidates[0].target
        assert state.candidates[0].source == AttentionSourceWitness(
            kind=AttentionSourceKind.MOTIVATION,
            reference=target.reference,
            revision=source_revision,
            digest="b" * 64,
            digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
            target_kind=target.kind,
            target_reference=target.reference,
        )


def test_streak_limit_is_per_candidate_not_aggregate_event_count() -> None:
    current = _continuity_with_two_candidate_streaks()
    assert current.last_event is not None
    assert sum(
        candidate.focused_event_count + candidate.unattended_event_count
        for candidate in current.candidates
    ) > current.last_event.event_sequence

    focused_candidate = next(
        candidate
        for candidate in current.candidates
        if candidate.candidate_id == current.focused_ids[0]
    )
    over_bound_candidate = replace(
        focused_candidate,
        focused_event_count=2,
        unattended_event_count=0,
    )
    candidates = tuple(
        over_bound_candidate if candidate.candidate_id == over_bound_candidate.candidate_id else candidate
        for candidate in current.candidates
    )
    with pytest.raises(ValueError, match="candidate event streak"):
        replace(current, candidates=candidates)


def test_source_event_identity_must_match_retained_upstream_time() -> None:
    bootstrap = AttentionContinuity.bootstrap()
    first_event = _event()
    current_event = _event("event-2", 2, NOW + timedelta(seconds=2))
    target = _target()
    source = AttentionSourceWitness(
        kind=AttentionSourceKind.MOTIVATION,
        reference=target.reference,
        revision=2,
        digest="b" * 64,
        digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
        target_kind=target.kind,
        target_reference=target.reference,
        source_event_id=first_event.event_id,
        source_event_sequence=first_event.event_sequence,
        source_occurred_at=NOW + timedelta(seconds=1),
    )
    candidate = AttentionCandidateContinuity(
        target=target,
        source=source,
        availability=CandidateAvailability.ELIGIBLE,
        focused_event_count=1,
    )
    focused_ids = (candidate.candidate_id,)
    first_history = AttentionRevisionEvidence(
        revision=1,
        event=first_event,
        previous_state_digest=bootstrap.state_digest,
        state_digest="a" * 64,
        previous_revision_digest=None,
        focused_ids=focused_ids,
        unfinished_ids=(),
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    current_state_digest = attention_state_digest(
        schema_version=1,
        policy_version=1,
        revision=2,
        last_event=current_event,
        candidates=(candidate,),
        focused_ids=focused_ids,
        unfinished_ids=(),
    )
    current_history = AttentionRevisionEvidence(
        revision=2,
        event=current_event,
        previous_state_digest=first_history.state_digest,
        state_digest=current_state_digest,
        previous_revision_digest=first_history.record_digest,
        focused_ids=focused_ids,
        unfinished_ids=(),
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    first_receipt = AttentionEventReceipt(
        event=first_event,
        input_digest="c" * 64,
        result_state_digest=first_history.state_digest,
    )
    current_receipt = AttentionEventReceipt(
        event=current_event,
        input_digest="d" * 64,
        result_state_digest=current_state_digest,
        previous_receipt_digest=first_receipt.receipt_digest,
    )

    with pytest.raises(ValueError, match="retained event identities disagree"):
        AttentionContinuity(
            revision=2,
            last_event=current_event,
            candidates=(candidate,),
            focused_ids=focused_ids,
            revision_history=(first_history, current_history),
            receipts=(first_receipt, current_receipt),
        )


def test_bootstrap_is_exactly_empty_and_nonempty_requires_proofs() -> None:
    bootstrap = AttentionContinuity.bootstrap()
    assert bootstrap.revision == 0
    assert bootstrap.last_event is None
    assert bootstrap.candidates == ()
    assert bootstrap.revision_history == ()
    assert bootstrap.receipts == ()

    with pytest.raises(ValueError, match="empty bootstrap"):
        AttentionContinuity(revision=0, candidates=_continuity().candidates)
    with pytest.raises(ValueError, match="every non-genesis revision"):
        replace(_continuity(), revision_history=())


def test_fresh_imports_do_not_load_runtime_or_model_modules() -> None:
    code = """
import importlib
import sys
for module in (
    'suzka.attention.common',
    'suzka.attention.contracts',
    'suzka.attention.bounds',
):
    importlib.import_module(module)
for prefix in ('suzka.runtime', 'suzka.models', 'torch', 'transformers', 'suzka.scheduler'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
