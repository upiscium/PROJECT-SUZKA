"""Immutable event-scoped metacognition shape and provenance tests."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import sys

import pytest

from suzka.attention.common import (
    ATTENTION_MAX_FOCUS,
    AttentionSourceKind,
    SourceDigestKind,
)
from suzka.attention.contracts import AttentionEvent
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE, MAX_PERSISTED_REVISION
from suzka.metacognition.contracts import (
    METACOGNITION_MAX_ASSESSMENT_BYTES,
    METACOGNITION_MAX_EVIDENCE_WITNESSES,
    METACOGNITION_MAX_REASON_CODES,
    EpistemicBoundary,
    EvidenceCondition,
    FocusAssessmentWitness,
    MetacognitiveAssessment,
    MetacognitiveEvidenceWitness,
    MetacognitiveReasonCode,
    SourceEventOrigin,
    derive_metacognition_assessment_max_bytes,
)


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _event(
    event_id: str = "event-1",
    sequence: int = 1,
    occurred_at: datetime = NOW,
) -> AttentionEvent:
    return AttentionEvent(event_id, sequence, occurred_at)


def _focus(
    event: AttentionEvent | None = None,
    *,
    revision: int = 1,
    focused_ids: tuple[str, ...] = (),
) -> FocusAssessmentWitness:
    current_event = event or _event()
    return FocusAssessmentWitness(
        attention_revision=revision,
        attention_state_digest="a" * 64,
        focused_ids=focused_ids,
        event=current_event,
    )


def _witness(
    reference: str = "source-1",
    *,
    source_kind: AttentionSourceKind = AttentionSourceKind.GOAL,
    source_revision: int | None = 0,
    digest_kind: SourceDigestKind = SourceDigestKind.UPSTREAM_AUTHORITY,
    condition: EvidenceCondition = EvidenceCondition.SUPPORTING,
    confidence_ceiling: float | None = 0.8,
    source_event: AttentionEvent | None = None,
    source_event_origin: SourceEventOrigin | None = None,
) -> MetacognitiveEvidenceWitness:
    return MetacognitiveEvidenceWitness(
        source_kind=source_kind,
        reference=reference,
        source_revision=source_revision,
        digest="b" * 64,
        digest_kind=digest_kind,
        condition=condition,
        confidence_ceiling=confidence_ceiling,
        source_event=source_event,
        source_event_origin=source_event_origin,
    )


def _ordered(*items: MetacognitiveEvidenceWitness) -> tuple[MetacognitiveEvidenceWitness, ...]:
    return tuple(sorted(items, key=lambda item: item.witness_digest))


def test_default_is_explicit_unknown_and_round_trips_canonically() -> None:
    current_event = _event()
    value = MetacognitiveAssessment(current_event, _focus(current_event))

    assert value.epistemic_boundary is EpistemicBoundary.UNKNOWN
    assert value.confidence is None
    assert value.evidence == ()
    assert value.reason_codes == (MetacognitiveReasonCode.MISSING_EVIDENCE,)
    assert value.evidence_sufficiency is None
    assert value.cognitive_load is None
    assert value.attention_saturation is None
    assert value.emotion_influence is None
    assert value.cognitive_quality is None
    assert MetacognitiveAssessment.from_json(value.canonical_bytes()) == value

    with pytest.raises(ValueError, match="missing-evidence reason"):
        replace(value, reason_codes=())
    with pytest.raises(ValueError, match="numeric value"):
        replace(value, cognitive_load=0.0)
    with pytest.raises(ValueError, match="unobserved evidence"):
        replace(value, confidence=0.0)


def test_missing_unknown_and_measured_zero_remain_distinct() -> None:
    current_event = _event()
    missing = MetacognitiveAssessment(current_event, _focus(current_event))
    unobserved = _witness(
        condition=EvidenceCondition.UNKNOWN,
        confidence_ceiling=None,
    )
    unknown_witnessed = MetacognitiveAssessment(
        current_event,
        _focus(current_event),
        evidence=(unobserved,),
    )
    assert unknown_witnessed.epistemic_boundary is EpistemicBoundary.UNKNOWN
    assert unknown_witnessed.confidence is None
    assert unknown_witnessed.evidence_sufficiency is None
    with pytest.raises(ValueError, match="unobserved evidence"):
        replace(unknown_witnessed, evidence_sufficiency=0.0)

    supporting = _witness(confidence_ceiling=0.0)
    measured_zero = MetacognitiveAssessment(
        current_event,
        _focus(current_event),
        evidence=(supporting,),
        evidence_sufficiency=0.0,
        epistemic_boundary=EpistemicBoundary.SUFFICIENT,
        confidence=0.0,
        cognitive_load=0.0,
        attention_saturation=0.0,
        emotion_influence=0.0,
        cognitive_quality=0.0,
        reason_codes=(MetacognitiveReasonCode.LOW_CONFIDENCE,),
    )
    assert measured_zero.evidence_sufficiency == 0.0
    assert measured_zero.confidence == 0.0
    assert measured_zero.canonical_value()["confidence"] == 0.0.hex()
    assert missing.canonical_value()["confidence"] is None
    assert missing.assessment_digest != measured_zero.assessment_digest


def test_contradiction_has_a_distinct_boundary_reason_and_bounded_confidence() -> None:
    current_event = _event()
    contradictory = _witness(
        "counter-evidence",
        condition=EvidenceCondition.CONTRADICTORY,
        confidence_ceiling=None,
    )
    unsupported = MetacognitiveAssessment(
        current_event,
        _focus(current_event),
        evidence=(contradictory,),
        epistemic_boundary=EpistemicBoundary.CONTRADICTORY,
        reason_codes=(MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE,),
    )
    assert unsupported.epistemic_boundary is EpistemicBoundary.CONTRADICTORY
    with pytest.raises(ValueError, match="distinct boundary"):
        replace(
            unsupported,
            epistemic_boundary=EpistemicBoundary.UNKNOWN,
            reason_codes=(MetacognitiveReasonCode.MISSING_EVIDENCE,),
        )
    with pytest.raises(ValueError, match="supporting typed confidence ceiling"):
        replace(unsupported, confidence=0.0)
    with pytest.raises(ValueError, match="distinct reason code"):
        replace(
            unsupported,
            reason_codes=(MetacognitiveReasonCode.LOW_CONFIDENCE,),
        )

    supporting = _witness("support-evidence", confidence_ceiling=0.4)
    bounded = MetacognitiveAssessment(
        current_event,
        _focus(current_event),
        evidence=_ordered(supporting, contradictory),
        epistemic_boundary=EpistemicBoundary.CONTRADICTORY,
        confidence=0.4,
        reason_codes=(MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE,),
    )
    assert bounded.confidence == 0.4
    with pytest.raises(ValueError, match="exceeds its supporting evidence ceiling"):
        replace(bounded, confidence=0.400001)


def test_confidence_requires_a_typed_support_ceiling_and_unknown_is_never_confident() -> None:
    current_event = _event()
    no_ceiling = _witness(confidence_ceiling=None)
    with pytest.raises(ValueError, match="supporting typed confidence ceiling"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            evidence=(no_ceiling,),
            epistemic_boundary=EpistemicBoundary.UNCERTAIN,
            confidence=0.9,
            reason_codes=(),
        )
    with pytest.raises(ValueError, match="must not carry confidence"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            evidence=(_witness(),),
            epistemic_boundary=EpistemicBoundary.UNKNOWN,
            confidence=0.9,
            reason_codes=(),
        )
    with pytest.raises(ValueError, match="unknown epistemic boundary cannot carry numeric"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            evidence=(_witness(),),
            epistemic_boundary=EpistemicBoundary.UNKNOWN,
            cognitive_load=0.0,
            reason_codes=(),
        )


def test_focus_is_event_bound_without_a_persisted_attention_commit() -> None:
    current_event = _event("same-current-event", 4, NOW + timedelta(seconds=4))
    focused_id = "c" * 64
    focus = _focus(current_event, revision=3, focused_ids=(focused_id,))
    value = MetacognitiveAssessment(current_event, focus)
    assert value.focus_witness.event == current_event
    assert value.focus_witness.attention_revision == 3
    assert value.focus_witness.focused_ids == (focused_id,)

    different_event = _event("other-event", 4, NOW + timedelta(seconds=4))
    with pytest.raises(ValueError, match="exact assessment event"):
        MetacognitiveAssessment(current_event, _focus(different_event))


def test_focus_and_evidence_reject_future_or_incoherent_events() -> None:
    current_event = _event("current", 2, NOW + timedelta(seconds=2))
    future_event = _event("future", 3, NOW + timedelta(seconds=3))
    future_witness = _witness(
        digest_kind=SourceDigestKind.UPSTREAM_AUTHORITY,
        source_event=future_event,
        source_event_origin=SourceEventOrigin.UPSTREAM_EVENT,
    )
    with pytest.raises(ValueError, match="future"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            evidence=(future_witness,),
            epistemic_boundary=EpistemicBoundary.SUFFICIENT,
            reason_codes=(),
        )

    same_sequence_wrong_identity = _event("different-at-same-sequence", 2, current_event.occurred_at)
    with pytest.raises(ValueError, match="exact same event"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            evidence=(
                _witness(
                    source_event=same_sequence_wrong_identity,
                    source_event_origin=SourceEventOrigin.UPSTREAM_EVENT,
                ),
            ),
            epistemic_boundary=EpistemicBoundary.SUFFICIENT,
            reason_codes=(),
        )

    with pytest.raises(ValueError, match="digest provenance"):
        _witness(
            digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
            source_event=current_event,
            source_event_origin=SourceEventOrigin.UPSTREAM_EVENT,
        )

    attention_derived = _witness(
        source_kind=AttentionSourceKind.WORKING_MEMORY,
        digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
        source_event=current_event,
        source_event_origin=SourceEventOrigin.ATTENTION_EVENT,
    )
    assessment = MetacognitiveAssessment(
        current_event,
        _focus(current_event),
        evidence=(attention_derived,),
        epistemic_boundary=EpistemicBoundary.SUFFICIENT,
        reason_codes=(),
    )
    assert assessment.evidence[0].source_event == current_event
    assert assessment.evidence[0].source_event_origin is SourceEventOrigin.ATTENTION_EVENT


def test_evidence_revisions_references_conditions_and_enums_are_closed() -> None:
    assert _witness(
        source_kind=AttentionSourceKind.EMOTION,
        source_revision=None,
        confidence_ceiling=None,
    ).source_revision is None
    assert _witness(
        source_kind=AttentionSourceKind.APPRAISAL,
        source_revision=None,
        confidence_ceiling=None,
    ).source_revision is None
    with pytest.raises(ValueError, match="required"):
        _witness(source_kind=AttentionSourceKind.GOAL, source_revision=None)
    with pytest.raises(ValueError, match="bounded"):
        _witness(source_revision=MAX_PERSISTED_REVISION + 1)
    with pytest.raises(ValueError, match="identifier"):
        _witness("a" * 129)
    with pytest.raises(TypeError, match="AttentionSourceKind"):
        _witness(source_kind="goal")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EvidenceCondition"):
        _witness(condition="supporting")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confidence ceiling"):
        _witness(
            condition=EvidenceCondition.UNKNOWN,
            confidence_ceiling=0.5,
        )
    with pytest.raises(TypeError, match="finite number"):
        _witness(confidence_ceiling=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        _witness(confidence_ceiling=float("nan"))


def test_values_are_frozen_slot_closed_and_have_no_history_or_metadata_fields() -> None:
    current_event = _event()
    value = MetacognitiveAssessment(current_event, _focus(current_event))

    assert not hasattr(value, "__dict__")
    assert {item.name for item in fields(value)} == {
        "event",
        "focus_witness",
        "evidence",
        "evidence_sufficiency",
        "epistemic_boundary",
        "confidence",
        "cognitive_load",
        "attention_saturation",
        "emotion_influence",
        "cognitive_quality",
        "reason_codes",
        "assessment_id",
        "assessment_digest",
    }
    with pytest.raises(TypeError, match="unexpected keyword"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            history=(),  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            raw_text="untrusted body",  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            _private_state=(),  # type: ignore[call-arg]
        )
    assert not hasattr(value, "history")
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(value, "assessment_history", ())
    with pytest.raises((AttributeError, TypeError)):
        value.reason_codes = ()  # type: ignore[misc]


def test_canonical_identity_digest_and_parser_tamper_checks() -> None:
    current_event = _event()
    missing = MetacognitiveAssessment(current_event, _focus(current_event))
    restored = MetacognitiveAssessment.from_json(missing.canonical_bytes())
    assert restored == missing
    assert restored.assessment_id == missing.assessment_id
    assert restored.assessment_digest == missing.assessment_digest

    changed_same_event = replace(
        missing,
        reason_codes=(
            MetacognitiveReasonCode.MISSING_EVIDENCE,
            MetacognitiveReasonCode.UNOBSERVED_SOURCE,
        ),
    )
    assert changed_same_event.assessment_id == missing.assessment_id
    assert changed_same_event.assessment_digest != missing.assessment_digest
    different_event = _event("event-2", 2, NOW + timedelta(seconds=1))
    different_assessment = MetacognitiveAssessment(different_event, _focus(different_event))
    assert different_assessment.assessment_id != missing.assessment_id

    extra = missing.canonical_value()
    extra["metadata"] = {}
    with pytest.raises(ValueError, match="unknown fields"):
        MetacognitiveAssessment.from_canonical_value(extra)
    bad_id = missing.canonical_value()
    bad_id["assessment_id"] = "0" * 64
    with pytest.raises(ValueError, match="ID"):
        MetacognitiveAssessment.from_canonical_value(bad_id)
    bad_digest = missing.canonical_value()
    bad_digest["assessment_digest"] = "0" * 64
    with pytest.raises(ValueError, match="assessment digest"):
        MetacognitiveAssessment.from_canonical_value(bad_digest)
    changed_focus = missing.canonical_value()
    focus_value = changed_focus["focus_witness"]
    assert isinstance(focus_value, dict)
    focus_value["attention_state_digest"] = "0" * 64
    with pytest.raises(ValueError, match="assessment digest"):
        MetacognitiveAssessment.from_canonical_value(changed_focus)
    with pytest.raises(ValueError, match="canonical form"):
        MetacognitiveAssessment.from_json(missing.canonical_bytes() + b" ")
    with pytest.raises(ValueError, match="duplicate"):
        MetacognitiveAssessment.from_json(b'{"duplicate":1,"duplicate":1}')
    with pytest.raises(ValueError, match="invalid constant"):
        MetacognitiveAssessment.from_json(b"NaN")

    witnessed = MetacognitiveAssessment(
        current_event,
        _focus(current_event),
        evidence=(_witness(),),
        epistemic_boundary=EpistemicBoundary.SUFFICIENT,
        reason_codes=(),
    )
    changed_witness = witnessed.canonical_value()
    changed_evidence = changed_witness["evidence"]
    assert isinstance(changed_evidence, list)
    changed_evidence[0]["digest"] = "c" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="witness digest"):
        MetacognitiveAssessment.from_canonical_value(changed_witness)


def test_full_bounded_shape_and_one_over_resource_counts() -> None:
    current_event = AttentionEvent(
        "e" * 128,
        MAX_PERSISTED_EVENT_SEQUENCE,
        datetime.max.replace(tzinfo=UTC),
    )
    focused_ids = tuple(f"{index:064x}" for index in range(ATTENTION_MAX_FOCUS))
    focus = _focus(
        current_event,
        revision=MAX_PERSISTED_REVISION,
        focused_ids=focused_ids,
    )
    with pytest.raises(ValueError, match="exceeds its bound"):
        _focus(
            current_event,
            revision=MAX_PERSISTED_REVISION,
            focused_ids=tuple(f"{index:064x}" for index in range(ATTENTION_MAX_FOCUS + 1)),
        )
    with pytest.raises(ValueError, match="bounded"):
        _focus(
            current_event,
            revision=MAX_PERSISTED_REVISION + 1,
        )
    with pytest.raises(ValueError, match="positive bounded exact integer"):
        AttentionEvent(
            "event-over-sequence-bound",
            MAX_PERSISTED_EVENT_SEQUENCE + 1,
            NOW,
        )

    smallest_positive = float.fromhex("0x0.0000000000001p-1022")
    evidence_items = [
        _witness(
            f"{index:02x}" + "a" * 126,
            source_kind=AttentionSourceKind.WORKING_MEMORY,
            source_revision=MAX_PERSISTED_REVISION,
            digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
            condition=(
                EvidenceCondition.CONTRADICTORY
                if index == METACOGNITION_MAX_EVIDENCE_WITNESSES - 1
                else EvidenceCondition.SUPPORTING
            ),
            confidence_ceiling=None
            if index == METACOGNITION_MAX_EVIDENCE_WITNESSES - 1
            else smallest_positive,
            source_event=current_event,
            source_event_origin=SourceEventOrigin.ATTENTION_EVENT,
        )
        for index in range(METACOGNITION_MAX_EVIDENCE_WITNESSES)
    ]
    evidence = _ordered(*evidence_items)
    reasons = tuple(sorted(MetacognitiveReasonCode, key=lambda code: code.value))
    full = MetacognitiveAssessment(
        current_event,
        focus,
        evidence=evidence,
        evidence_sufficiency=smallest_positive,
        epistemic_boundary=EpistemicBoundary.CONTRADICTORY,
        confidence=smallest_positive,
        cognitive_load=smallest_positive,
        attention_saturation=smallest_positive,
        emotion_influence=smallest_positive,
        cognitive_quality=smallest_positive,
        reason_codes=reasons,
    )
    assert len(full.evidence) == METACOGNITION_MAX_EVIDENCE_WITNESSES
    assert len(full.reason_codes) == METACOGNITION_MAX_REASON_CODES
    assert len(full.canonical_bytes()) <= METACOGNITION_MAX_ASSESSMENT_BYTES
    assert derive_metacognition_assessment_max_bytes() == METACOGNITION_MAX_ASSESSMENT_BYTES
    assert len(full.canonical_bytes()) > len(
        MetacognitiveAssessment(current_event, _focus(current_event)).canonical_bytes()
    )

    with pytest.raises(ValueError, match="witness bound"):
        MetacognitiveAssessment(
            current_event,
            focus,
            evidence=(*evidence, evidence[0]),
            epistemic_boundary=EpistemicBoundary.CONTRADICTORY,
            reason_codes=(MetacognitiveReasonCode.CONTRADICTORY_EVIDENCE,),
        )
    with pytest.raises(ValueError, match="reason_codes exceeds"):
        replace(full, reason_codes=(*reasons, reasons[0]))


def test_fraction_fields_reject_booleans_and_nonfinite_values() -> None:
    current_event = _event()
    with pytest.raises(TypeError, match="finite number"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            cognitive_quality=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite"):
        MetacognitiveAssessment(
            current_event,
            _focus(current_event),
            attention_saturation=float("inf"),
        )


def test_fresh_import_does_not_load_runtime_or_later_authorities() -> None:
    code = """
import importlib
import sys
importlib.import_module('suzka.metacognition.contracts')
for prefix in (
    'suzka.runtime', 'suzka.memory', 'suzka.cognition', 'suzka.models',
    'suzka.provider', 'suzka.providers', 'suzka.scheduler',
    'torch', 'transformers',
):
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
