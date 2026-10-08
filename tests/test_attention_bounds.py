"""Full-universe Attention persistence sizing and independent fence tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from suzka.attention.bounds import (
    ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES,
    ATTENTION_AGENT_STATE_MAX_BYTES,
    ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES,
    ATTENTION_FIELD_NAME,
    ATTENTION_ROOT_FIELD_OVERHEAD_BYTES,
    ATTENTION_STATE_MAX_VALUE_BYTES,
    attention_candidate_capacity,
    attention_candidate_capacity_by_kind,
    canonical_attention_schema_size_budget,
    derive_attention_schema_size_budget,
    maximum_attention_continuity_fixture,
    validate_attention_schema_size_budget,
)
from suzka.attention.common import (
    ATTENTION_FIXED_POINT_SCALE,
    ATTENTION_MAX_COUNTER,
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_MAX_EVENT_SEQUENCE,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_REVISION,
    ATTENTION_MAX_REVISION_HISTORY,
    AttentionSourceKind,
    AttentionTargetKind,
    CandidateAvailability,
    SourceDigestKind,
    canonical_json,
)
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionContinuity,
    AttentionSourceWitness,
    AttentionTarget,
)
from suzka.working_memory_contracts import MAX_ITEM_CAPACITY
from suzka.motivation.common import R13_MAX_RECORDS_PER_DOMAIN


def test_supported_identity_capacity_is_source_owned_and_complete() -> None:
    per_kind = attention_candidate_capacity_by_kind()
    assert dict(per_kind) == {
        "commitment": R13_MAX_RECORDS_PER_DOMAIN,
        "goal": R13_MAX_RECORDS_PER_DOMAIN,
        "motivation": R13_MAX_RECORDS_PER_DOMAIN,
        "working_memory": MAX_ITEM_CAPACITY,
    }
    assert attention_candidate_capacity() == (
        MAX_ITEM_CAPACITY + 3 * R13_MAX_RECORDS_PER_DOMAIN
    )
    assert attention_candidate_capacity() == 4_192


def test_full_universe_fixture_covers_all_bounded_schema_fields() -> None:
    fixture = maximum_attention_continuity_fixture()
    budget = derive_attention_schema_size_budget()
    serialized = fixture.canonical_bytes()
    assert fixture.last_event is not None

    assert fixture.revision == ATTENTION_MAX_REVISION
    assert len(fixture.candidates) == 4_192
    assert budget.candidate_count == 4_192
    assert budget.candidate_counts_by_kind == {
        "commitment": R13_MAX_RECORDS_PER_DOMAIN,
        "goal": R13_MAX_RECORDS_PER_DOMAIN,
        "motivation": R13_MAX_RECORDS_PER_DOMAIN,
        "working_memory": MAX_ITEM_CAPACITY,
    }
    assert len(fixture.focused_ids) == ATTENTION_MAX_FOCUS
    assert len(fixture.unfinished_ids) == ATTENTION_MAX_FOCUS
    assert len(fixture.revision_history) == ATTENTION_MAX_REVISION_HISTORY
    assert len(fixture.receipts) == ATTENTION_MAX_EVENT_RECEIPTS
    assert fixture.revision_anchor is not None
    assert fixture.receipt_anchor is not None
    assert all(candidate.habituation == ATTENTION_FIXED_POINT_SCALE for candidate in fixture.candidates)
    assert all(candidate.inhibition == ATTENTION_FIXED_POINT_SCALE for candidate in fixture.candidates)
    focused_ids = set(fixture.focused_ids)
    assert all(
        candidate.focused_event_count
        == (ATTENTION_MAX_COUNTER if candidate.candidate_id in focused_ids else 0)
        for candidate in fixture.candidates
    )
    assert all(
        candidate.unattended_event_count
        == (0 if candidate.candidate_id in focused_ids else ATTENTION_MAX_COUNTER)
        for candidate in fixture.candidates
    )
    assert all(candidate.source.revision == ATTENTION_MAX_REVISION for candidate in fixture.candidates)
    assert all(
        candidate.source.kind.value == candidate.target.kind.value
        and candidate.source.reference == candidate.target.reference
        for candidate in fixture.candidates
    )
    assert all(
        candidate.source.source_event_sequence == ATTENTION_MAX_EVENT_SEQUENCE
        for candidate in fixture.candidates
        if candidate.target.kind is not AttentionTargetKind.WORKING_MEMORY
    )
    assert all(
        candidate.source.event() is None
        for candidate in fixture.candidates
        if candidate.target.kind is AttentionTargetKind.WORKING_MEMORY
    )
    assert all(
        candidate.source.event() == fixture.last_event
        and candidate.source.digest_kind is SourceDigestKind.UPSTREAM_AUTHORITY
        for candidate in fixture.candidates
        if candidate.target.kind is not AttentionTargetKind.WORKING_MEMORY
    )
    assert all(
        candidate.source.digest_kind is SourceDigestKind.ATTENTION_PROJECTION
        for candidate in fixture.candidates
        if candidate.target.kind is AttentionTargetKind.WORKING_MEMORY
    )
    assert budget.attention_state_max_bytes == len(serialized)
    assert AttentionContinuity.from_json(serialized).canonical_bytes() == serialized
    assert budget.attention_state_envelope_bytes == budget.attention_state_max_bytes
    assert budget.candidate_array_bytes == budget.candidate_array_envelope_bytes
    assert (
        budget.revision_history_array_bytes
        == budget.revision_history_array_envelope_bytes
    )
    assert budget.receipts_array_bytes == budget.receipts_array_envelope_bytes
    assert len(budget.candidate_row_envelopes) == len(AttentionTargetKind)
    candidate_field_names = set(fixture.candidates[0].canonical_value())
    for envelope in budget.candidate_row_envelopes:
        assert set(envelope.field_maxima_by_name) == candidate_field_names
        assert (
            envelope.maximum_nonfocused_row_bytes
            == envelope.unavailable_nonfocused_row_bytes
        )
    assert set(budget.field_maxima_by_name) == set(fixture.canonical_value())
    assert budget.field_maxima_by_name == {
        name: len(canonical_json(value))
        for name, value in fixture.canonical_value().items()
    }
    assert validate_attention_schema_size_budget(budget) == budget
    assert canonical_attention_schema_size_budget(budget)

    revision_states = {
        (item.event.event_id, item.event.event_sequence): item.state_digest
        for item in fixture.revision_history
    }
    for receipt in fixture.receipts:
        matching_state = revision_states.get(
            (receipt.event.event_id, receipt.event.event_sequence)
        )
        if matching_state is not None:
            assert receipt.result_state_digest == matching_state


def test_attention_state_value_and_exact_v9_reserved_projection_fit() -> None:
    budget = derive_attention_schema_size_budget()
    assert ATTENTION_STATE_MAX_VALUE_BYTES == 7_866_805
    assert ATTENTION_ROOT_FIELD_OVERHEAD_BYTES == len(canonical_json(ATTENTION_FIELD_NAME)) + 2
    assert budget.attention_state_max_bytes <= ATTENTION_STATE_MAX_VALUE_BYTES
    assert budget.margin_bytes == ATTENTION_STATE_MAX_VALUE_BYTES - budget.attention_state_max_bytes

    # Runtime is intentionally imported only here, in the test. Production
    # Attention modules remain dependency-light and do not import AgentState.
    from suzka.runtime.agent_state import project_agent_state_schema_max_bytes

    whole_v9_with_reserve = project_agent_state_schema_max_bytes(
        schema_version=9,
        base_schema_version=8,
        added_field_maxima={"attention_state": budget.attention_state_max_bytes},
    )
    assert whole_v9_with_reserve == (
        ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES
        + ATTENTION_ROOT_FIELD_OVERHEAD_BYTES
        + budget.attention_state_max_bytes
        + ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES
    )
    assert whole_v9_with_reserve <= ATTENTION_AGENT_STATE_MAX_BYTES
    assert ATTENTION_AGENT_STATE_MAX_BYTES - whole_v9_with_reserve == budget.margin_bytes


def test_history_and_receipt_tails_have_independent_exact_fences() -> None:
    fixture = maximum_attention_continuity_fixture()
    assert fixture.revision_anchor is not None
    assert fixture.receipt_anchor is not None
    assert fixture.revision_anchor.through_revision == (
        fixture.revision - ATTENTION_MAX_REVISION_HISTORY
    )
    assert fixture.revision_anchor.through_event.event_sequence != (
        fixture.receipt_anchor.through_event.event_sequence
    )
    assert fixture.receipt_anchor.through_event.event_sequence < (
        fixture.receipts[0].event.event_sequence
    )

    with pytest.raises(ValueError, match="full retained suffix"):
        replace(fixture, revision_history=fixture.revision_history[:-1])
    with pytest.raises(ValueError, match="retained suffix"):
        broken_revision_anchor = replace(
            fixture.revision_anchor,
            through_revision=fixture.revision_anchor.through_revision + 1,
        )
        replace(fixture, revision_anchor=broken_revision_anchor)
    with pytest.raises(ValueError, match="full independent retained suffix"):
        replace(fixture, receipts=fixture.receipts[:-1])
    with pytest.raises(ValueError, match="receipt anchor event fence"):
        overlapping_receipt_anchor = replace(
            fixture.receipt_anchor,
            through_event=replace(
                fixture.receipt_anchor.through_event,
                event_sequence=fixture.receipts[0].event.event_sequence,
            ),
        )
        replace(fixture, receipt_anchor=overlapping_receipt_anchor)

    # A short receipt suffix cannot substitute for its own full 256-entry tail,
    # regardless of the separately valid 16-entry revision suffix.
    short_receipts = fixture.receipts[-2:]
    independent_receipt_anchor = replace(
        fixture.receipt_anchor,
        through_event=replace(
            fixture.receipt_anchor.through_event,
            event_sequence=short_receipts[0].event.event_sequence - 1,
        ),
    )
    with pytest.raises(ValueError, match="full independent retained suffix"):
        replace(
            fixture,
            receipts=short_receipts,
            receipt_anchor=independent_receipt_anchor,
        )


def test_one_over_candidate_and_receipt_limits_are_rejected_not_clipped() -> None:
    fixture = maximum_attention_continuity_fixture()
    extra_target = AttentionTarget(
        AttentionTargetKind.WORKING_MEMORY,
        "wm-" + "f" * 64,
    )
    extra_source = AttentionSourceWitness(
        kind=AttentionSourceKind.WORKING_MEMORY,
        reference=extra_target.reference,
        revision=ATTENTION_MAX_REVISION,
        digest="f" * 64,
        digest_kind=SourceDigestKind.ATTENTION_PROJECTION,
        target_kind=extra_target.kind,
        target_reference=extra_target.reference,
    )
    extra = AttentionCandidateContinuity(
        target=extra_target,
        source=extra_source,
        availability=CandidateAvailability.UNAVAILABLE,
        habituation=ATTENTION_FIXED_POINT_SCALE,
        inhibition=ATTENTION_FIXED_POINT_SCALE,
        focused_event_count=0,
        unattended_event_count=ATTENTION_MAX_COUNTER,
    )
    over_capacity = tuple(sorted((*fixture.candidates, extra), key=lambda item: item.candidate_id))
    with pytest.raises(ValueError, match="source authority bound"):
        replace(fixture, candidates=over_capacity)

    with pytest.raises(ValueError, match="retained window"):
        replace(fixture, receipts=fixture.receipts + (fixture.receipts[-1],))


def test_budget_helpers_do_not_clamp_oversized_value_bounds() -> None:
    budget = derive_attention_schema_size_budget()
    with pytest.raises(ValueError, match="exceeds its reserved"):
        replace(
            budget,
            attention_state_max_bytes=ATTENTION_STATE_MAX_VALUE_BYTES + 1,
        )
