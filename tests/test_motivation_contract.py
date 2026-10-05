"""Focused tests for the pure R13 Motivation contract boundary."""

from dataclasses import fields, replace
from datetime import UTC, datetime
from math import nan
from typing import cast

import pytest

from suzka.motivation.common import (
    MotivationModelIdentity,
    R13Reference,
    R13ReferenceKind,
    RevisionCompactionAnchor,
)
from suzka.motivation.motivation import (
    MotivationCandidateClassification,
    MotivationInterpretationCandidate,
    MotivationKind,
    MotivationLifecycle,
    MotivationRecord,
    MotivationRevisionOperation,
    MotivationRevisionReason,
    MotivationRevisionRecord,
    canonical_motivation_candidate_payload,
    canonical_motivation_payload,
    motivation_candidate_digest,
    motivation_id_for_target,
    motivation_record_digest,
    motivation_state_digest,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def ref(kind: R13ReferenceKind, value: str) -> R13Reference:
    return R13Reference(kind, value)


def state_digest_for(values: dict[str, object]) -> str:
    return motivation_state_digest(
        motivation_id=cast(str, values["motivation_id"]),
        kind=cast(MotivationKind, values["kind"]),
        target=cast(R13Reference, values["target"]),
        lifecycle=cast(MotivationLifecycle, values["lifecycle"]),
        source_evidence=cast(R13Reference, values["source_evidence"]),
        evidence_refs=cast(tuple[R13Reference, ...], values["evidence_refs"]),
        strength=cast(float, values.get("strength", 0.0)),
        persistence=cast(float, values.get("persistence", 0.0)),
        satiation=cast(float, values.get("satiation", 0.0)),
        uncertainty=cast(float, values.get("uncertainty", 0.0)),
        conflict_refs=cast(tuple[R13Reference, ...], values.get("conflict_refs", ())),
        related_refs=cast(tuple[R13Reference, ...], values.get("related_refs", ())),
        schema_version=cast(int, values.get("schema_version", 1)),
    )


def make_record(**changes: object) -> MotivationRecord:
    target = changes.pop("target", ref(R13ReferenceKind.VALUE, "value:one"))
    kind = changes.pop("kind", MotivationKind.DESIRE)
    source = changes.pop(
        "source_evidence", ref(R13ReferenceKind.EXPERIENCE, "experience:one")
    )
    values: dict[str, object] = {
        "motivation_id": motivation_id_for_target(kind, target),
        "kind": kind,
        "target": target,
        "lifecycle": MotivationLifecycle.ACTIVE,
        "source_evidence": source,
        "evidence_refs": (source,),
    }
    values.update(changes)
    if "revision_history" not in values:
        state_digest = state_digest_for(values)
        values["revision_history"] = (
            MotivationRevisionRecord(
                values["motivation_id"],
                0,
                MotivationRevisionOperation.CREATE,
                MotivationRevisionReason.CREATION,
                NOW,
                None,
                state_digest,
                event_id="event:create",
                event_sequence=1,
                evidence_refs=(source.reference,),
            ),
        )
    return MotivationRecord(**values)


def make_genesis(
    motivation_id: str, state_digest: str
) -> MotivationRevisionRecord:
    return MotivationRevisionRecord(
        motivation_id,
        0,
        MotivationRevisionOperation.CREATE,
        MotivationRevisionReason.CREATION,
        NOW,
        None,
        state_digest,
        event_id="event:create",
        event_sequence=1,
        evidence_refs=("experience:one",),
    )


def test_identity_excludes_mutable_state_and_record_is_frozen() -> None:
    first = make_record(strength=0.1)
    second = make_record(
        strength=0.9,
        persistence=0.8,
        satiation=0.2,
        uncertainty=0.4,
    )
    assert first.motivation_id == second.motivation_id
    assert motivation_record_digest(first) != motivation_record_digest(second)
    with pytest.raises(ValueError, match="does not bind its current state"):
        replace(first, strength=0.5)
    with pytest.raises((AttributeError, TypeError)):
        first.strength = 0.5  # type: ignore[misc]


def test_only_r13_authorized_sources_can_be_motivation_evidence() -> None:
    external = ref(R13ReferenceKind.USER_REQUEST, "request:one")
    with pytest.raises(ValueError):
        make_record(source_evidence=external, evidence_refs=(external,))
    with pytest.raises(ValueError):
        make_record(
            evidence_refs=(
                ref(R13ReferenceKind.EXPERIENCE, "experience:two"),
                ref(R13ReferenceKind.EXPERIENCE, "experience:one"),
            )
        )


def test_motivation_cannot_use_its_own_identity_as_source_evidence() -> None:
    target = ref(R13ReferenceKind.VALUE, "value:one")
    motivation_id = motivation_id_for_target(MotivationKind.DESIRE, target)
    self_ref = ref(R13ReferenceKind.MOTIVATION, motivation_id)

    with pytest.raises(ValueError, match="cannot reference itself"):
        make_record(source_evidence=self_ref, evidence_refs=(self_ref,))
    with pytest.raises(ValueError):
        make_record(
            evidence_refs=(
                ref(R13ReferenceKind.EXPERIENCE, "experience:one"),
                ref(R13ReferenceKind.EXPERIENCE, "experience:one"),
            )
        )


def test_model_candidate_is_typed_evidence_not_authoritative_motivation() -> None:
    candidate = MotivationInterpretationCandidate(
        source_evidence_refs=(ref(R13ReferenceKind.EXPERIENCE, "experience:one"),),
        event_id="event:interpret",
        event_sequence=2,
        model_identity=MotivationModelIdentity("transformers", "google/gemma-4-E4B"),
        suggested_kind=MotivationKind.DRIVE,
        suggested_target=ref(R13ReferenceKind.STATE, "state:rest"),
        suggested_strength=0.7,
    )
    assert candidate.classification is MotivationCandidateClassification.MODEL_INFERENCE
    assert candidate.candidate_digest
    assert motivation_candidate_digest(candidate) == candidate.candidate_digest
    assert canonical_motivation_candidate_payload(candidate)
    assert candidate.motivation_id != ""
    assert not isinstance(candidate, MotivationRecord)
    assert not {
        item.name.casefold()
        for item in fields(MotivationInterpretationCandidate)
    }.intersection({"prompt", "rationale", "hidden_thought", "transcript"})
    assert candidate.model_key.startswith("model.")
    with pytest.raises(ValueError):
        MotivationInterpretationCandidate(
            source_evidence_refs=(ref(R13ReferenceKind.EXPERIENCE, "experience:one"),),
            event_id="event:interpret",
            event_sequence=2,
            model_identity=MotivationModelIdentity("operator alice", "model"),
            suggested_kind=MotivationKind.DRIVE,
            suggested_target=ref(R13ReferenceKind.STATE, "state:rest"),
        )


def test_numeric_and_count_bounds_fail_closed_without_truncation() -> None:
    source = ref(R13ReferenceKind.EXPERIENCE, "experience:one")
    with pytest.raises(ValueError):
        make_record(strength=nan)
    with pytest.raises(ValueError):
        make_record(
            evidence_refs=tuple(
                ref(R13ReferenceKind.EXPERIENCE, f"experience:{index:02d}")
                for index in range(33)
            )
        )
    with pytest.raises(ValueError):
        make_record(
            evidence_refs=(source,),
            conflict_refs=tuple(
                ref(R13ReferenceKind.STATE, f"state:{index:02d}")
                for index in range(33)
            ),
        )


def test_revision_digest_chain_and_canonical_payload_are_deterministic() -> None:
    motivation_id = motivation_id_for_target(
        MotivationKind.DESIRE,
        ref(R13ReferenceKind.VALUE, "value:one"),
    )
    values: dict[str, object] = {
        "motivation_id": motivation_id,
        "kind": MotivationKind.DESIRE,
        "target": ref(R13ReferenceKind.VALUE, "value:one"),
        "lifecycle": MotivationLifecycle.ACTIVE,
        "source_evidence": ref(R13ReferenceKind.EXPERIENCE, "experience:one"),
        "evidence_refs": (ref(R13ReferenceKind.EXPERIENCE, "experience:one"),),
    }
    active_digest = state_digest_for(values)
    genesis = make_genesis(motivation_id, active_digest)
    revision = MotivationRevisionRecord(
        motivation_id,
        1,
        MotivationRevisionOperation.UPDATE,
        MotivationRevisionReason.DECAY,
        NOW,
        MotivationLifecycle.ACTIVE,
        state_digest_for({**values, "lifecycle": MotivationLifecycle.DORMANT}),
        previous_revision_digest=genesis.record_digest,
        event_id="event:update",
        event_sequence=2,
        evidence_refs=("experience:one",),
    )
    current = make_record(
        lifecycle=MotivationLifecycle.DORMANT,
        revision=1,
        revision_history=(genesis, revision),
    )
    assert canonical_motivation_payload(current) == canonical_motivation_payload(current)
    assert motivation_record_digest(current) == current.record_digest
    with pytest.raises(ValueError):
        replace(current, revision=2)
    with pytest.raises(ValueError):
        MotivationRevisionRecord(
            motivation_id,
            1,
            MotivationRevisionOperation.UPDATE,
            MotivationRevisionReason.DECAY,
            NOW,
            MotivationLifecycle.ACTIVE,
            "0" * 64,
        )


def test_compaction_anchor_binds_the_full_retained_suffix() -> None:
    motivation_id = motivation_id_for_target(
        MotivationKind.DESIRE,
        ref(R13ReferenceKind.VALUE, "value:one"),
    )
    state_values: dict[str, object] = {
        "motivation_id": motivation_id,
        "kind": MotivationKind.DESIRE,
        "target": ref(R13ReferenceKind.VALUE, "value:one"),
        "lifecycle": MotivationLifecycle.ACTIVE,
        "source_evidence": ref(R13ReferenceKind.EXPERIENCE, "experience:one"),
        "evidence_refs": (ref(R13ReferenceKind.EXPERIENCE, "experience:one"),),
    }
    current_state_digest = state_digest_for(state_values)
    revisions: list[MotivationRevisionRecord] = []
    previous: str | None = None
    for number in range(9):
        item = MotivationRevisionRecord(
            motivation_id,
            number,
            MotivationRevisionOperation.CREATE
            if number == 0
            else MotivationRevisionOperation.UPDATE,
            MotivationRevisionReason.CREATION
            if number == 0
            else MotivationRevisionReason.EVIDENCE_UPDATE,
            NOW,
            None if number == 0 else MotivationLifecycle.ACTIVE,
            current_state_digest,
            previous_revision_digest=previous,
            event_id=f"event:update:{number}",
            event_sequence=number + 1,
            evidence_refs=("experience:one",),
        )
        revisions.append(item)
        previous = item.record_digest
    assert revisions[0].event_id is not None
    assert revisions[0].event_sequence is not None
    current = make_record(
        revision=8,
        revision_history=tuple(revisions[1:]),
        history_anchor=RevisionCompactionAnchor(
            authority_id=motivation_id,
            through_revision=0,
            through_digest=revisions[0].record_digest,
            through_created_at=revisions[0].created_at,
            through_evidence_refs=revisions[0].evidence_refs,
            through_previous_revision_digest=None,
            through_state=MotivationLifecycle.ACTIVE.value,
            through_previous_state=None,
            through_operation=MotivationRevisionOperation.CREATE.value,
            through_reason=MotivationRevisionReason.CREATION.value,
            through_event_id=revisions[0].event_id,
            through_event_sequence=revisions[0].event_sequence,
            through_state_digest=revisions[0].state_digest,
        ),
    )
    assert current.history_anchor is not None
    assert current.history_anchor_digest == revisions[0].record_digest
    with pytest.raises(ValueError, match="anchor does not match"):
        replace(
            current,
            history_anchor=replace(
                current.history_anchor,
                through_event_id="event:tampered-anchor",
            ),
        )


def test_retired_motivation_cannot_be_reopened_by_later_update() -> None:
    target = ref(R13ReferenceKind.VALUE, "value:one")
    source = ref(R13ReferenceKind.EXPERIENCE, "experience:one")
    motivation_id = motivation_id_for_target(MotivationKind.DESIRE, target)
    values: dict[str, object] = {
        "motivation_id": motivation_id,
        "kind": MotivationKind.DESIRE,
        "target": target,
        "lifecycle": MotivationLifecycle.ACTIVE,
        "source_evidence": source,
        "evidence_refs": (source,),
    }
    genesis = make_genesis(motivation_id, state_digest_for(values))
    retired_values = {**values, "lifecycle": MotivationLifecycle.RETIRED}
    retired = MotivationRevisionRecord(
        motivation_id,
        1,
        MotivationRevisionOperation.RETIRE,
        MotivationRevisionReason.RETIREMENT,
        NOW,
        MotivationLifecycle.ACTIVE,
        state_digest_for(retired_values),
        event_id="event:retire",
        event_sequence=2,
        evidence_refs=(source.reference,),
        previous_revision_digest=genesis.record_digest,
    )
    reopened = MotivationRevisionRecord(
        motivation_id,
        2,
        MotivationRevisionOperation.UPDATE,
        MotivationRevisionReason.EVIDENCE_UPDATE,
        NOW,
        MotivationLifecycle.RETIRED,
        state_digest_for(values),
        event_id="event:reopen",
        event_sequence=3,
        evidence_refs=(source.reference,),
        previous_revision_digest=retired.record_digest,
    )

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        make_record(
            revision=2,
            revision_history=(genesis, retired, reopened),
        )


def test_compacted_anchor_cannot_hide_an_update_after_retirement() -> None:
    target = ref(R13ReferenceKind.VALUE, "value:one")
    source = ref(R13ReferenceKind.EXPERIENCE, "experience:one")
    motivation_id = motivation_id_for_target(MotivationKind.DESIRE, target)
    active_values: dict[str, object] = {
        "motivation_id": motivation_id,
        "kind": MotivationKind.DESIRE,
        "target": target,
        "lifecycle": MotivationLifecycle.ACTIVE,
        "source_evidence": source,
        "evidence_refs": (source,),
    }
    dormant_values = {**active_values, "lifecycle": MotivationLifecycle.DORMANT}
    genesis = make_genesis(motivation_id, state_digest_for(active_values))
    invalid_compacted_revision = MotivationRevisionRecord(
        motivation_id,
        1,
        MotivationRevisionOperation.UPDATE,
        MotivationRevisionReason.DECAY,
        NOW,
        MotivationLifecycle.RETIRED,
        state_digest_for(dormant_values),
        event_id="event:invalid-compacted-transition",
        event_sequence=2,
        evidence_refs=(source.reference,),
        previous_revision_digest=genesis.record_digest,
    )
    retained: list[MotivationRevisionRecord] = []
    previous_digest = invalid_compacted_revision.record_digest
    previous_state = MotivationLifecycle.DORMANT
    for number in range(2, 10):
        revision = MotivationRevisionRecord(
            motivation_id,
            number,
            MotivationRevisionOperation.UPDATE,
            MotivationRevisionReason.EVIDENCE_UPDATE,
            NOW,
            previous_state,
            state_digest_for(active_values),
            event_id=f"event:retained:{number}",
            event_sequence=number + 1,
            evidence_refs=(source.reference,),
            previous_revision_digest=previous_digest,
        )
        retained.append(revision)
        previous_digest = revision.record_digest
        previous_state = MotivationLifecycle.ACTIVE

    anchor = RevisionCompactionAnchor(
        authority_id=motivation_id,
        through_revision=1,
        through_digest=invalid_compacted_revision.record_digest,
        through_created_at=invalid_compacted_revision.created_at,
        through_evidence_refs=invalid_compacted_revision.evidence_refs,
        through_previous_revision_digest=genesis.record_digest,
        through_state=MotivationLifecycle.DORMANT.value,
        through_previous_state=MotivationLifecycle.RETIRED.value,
        through_operation=MotivationRevisionOperation.UPDATE.value,
        through_reason=MotivationRevisionReason.DECAY.value,
        through_event_id="event:invalid-compacted-transition",
        through_event_sequence=2,
        through_state_digest=invalid_compacted_revision.state_digest,
    )

    with pytest.raises(ValueError, match="hides an invalid lifecycle transition"):
        make_record(
            lifecycle=MotivationLifecycle.ACTIVE,
            revision=9,
            revision_history=tuple(retained),
            history_anchor=anchor,
        )


def test_revision_sequence_must_increase_and_digest_must_match_contents() -> None:
    target = ref(R13ReferenceKind.VALUE, "value:one")
    source = ref(R13ReferenceKind.EXPERIENCE, "experience:one")
    motivation_id = motivation_id_for_target(MotivationKind.DESIRE, target)
    values: dict[str, object] = {
        "motivation_id": motivation_id,
        "kind": MotivationKind.DESIRE,
        "target": target,
        "lifecycle": MotivationLifecycle.ACTIVE,
        "source_evidence": source,
        "evidence_refs": (source,),
    }
    genesis = make_genesis(motivation_id, state_digest_for(values))
    stale_event = MotivationRevisionRecord(
        motivation_id,
        1,
        MotivationRevisionOperation.UPDATE,
        MotivationRevisionReason.EVIDENCE_UPDATE,
        NOW,
        MotivationLifecycle.ACTIVE,
        state_digest_for(values),
        event_id="event:stale",
        event_sequence=1,
        evidence_refs=(source.reference,),
        previous_revision_digest=genesis.record_digest,
    )
    with pytest.raises(ValueError, match="reuse an event sequence"):
        make_record(revision=1, revision_history=(genesis, stale_event))

    tampered = MotivationRevisionRecord(
        motivation_id,
        1,
        MotivationRevisionOperation.UPDATE,
        MotivationRevisionReason.EVIDENCE_UPDATE,
        NOW,
        MotivationLifecycle.ACTIVE,
        state_digest_for(values),
        event_id="event:tampered",
        event_sequence=2,
        evidence_refs=(source.reference,),
        previous_revision_digest=genesis.record_digest,
    )
    object.__setattr__(tampered, "reason", MotivationRevisionReason.DECAY)
    with pytest.raises(ValueError, match="digest does not match"):
        make_record(revision=1, revision_history=(genesis, tampered))


def test_retained_revision_suffix_rejects_non_adjacent_event_id_reuse() -> None:
    target = ref(R13ReferenceKind.VALUE, "value:one")
    source = ref(R13ReferenceKind.EXPERIENCE, "experience:one")
    motivation_id = motivation_id_for_target(MotivationKind.DESIRE, target)
    values: dict[str, object] = {
        "motivation_id": motivation_id,
        "kind": MotivationKind.DESIRE,
        "target": target,
        "lifecycle": MotivationLifecycle.ACTIVE,
        "source_evidence": source,
        "evidence_refs": (source,),
    }
    genesis = make_genesis(motivation_id, state_digest_for(values))
    middle = MotivationRevisionRecord(
        motivation_id,
        1,
        MotivationRevisionOperation.UPDATE,
        MotivationRevisionReason.EVIDENCE_UPDATE,
        NOW,
        MotivationLifecycle.ACTIVE,
        state_digest_for(values),
        event_id="event:middle",
        event_sequence=2,
        evidence_refs=(source.reference,),
        previous_revision_digest=genesis.record_digest,
    )
    repeated = MotivationRevisionRecord(
        motivation_id,
        2,
        MotivationRevisionOperation.UPDATE,
        MotivationRevisionReason.EVIDENCE_UPDATE,
        NOW,
        MotivationLifecycle.ACTIVE,
        state_digest_for(values),
        event_id="event:create",
        event_sequence=3,
        evidence_refs=(source.reference,),
        previous_revision_digest=middle.record_digest,
    )

    with pytest.raises(ValueError, match="event IDs cannot be reused"):
        make_record(revision=2, revision_history=(genesis, middle, repeated))
