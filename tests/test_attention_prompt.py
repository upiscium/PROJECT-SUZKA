"""U5 same-turn Attention prompt rendering and source-privacy tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json

import pytest

from suzka.attention.adapters import (
    _R13_RENDERED_ROW_DOMAIN,
    _WM_RENDERED_ROW_DOMAIN,
    project_attention_candidates,
)
from suzka.attention.common import (
    ATTENTION_PROMPT_BUDGET_BYTES,
    AttentionTargetKind,
    canonical_json,
    digest_payload,
)
from suzka.attention.contracts import AttentionEvent
from suzka.attention.policy import (
    ATTENTION_PROMPT_AUTHORITY_INTRO,
    ATTENTION_PROMPT_FRAME_BYTES,
    ATTENTION_PROMPT_FRAME_TEXT,
    ATTENTION_PROMPT_SECTION_HEADERS,
    AttentionPromptReason,
)
from suzka.attention.system import AttentionRefreshResult, AttentionSystem
from suzka.body import EmotionState
from suzka.motivation.commitment import CommitmentLifecycle
from suzka.motivation.goal import GoalLifecycle as R13GoalLifecycle
from suzka.motivation.motivation import MotivationLifecycle
from suzka.motivation.commitment_system import CommitmentSystemSnapshot
from suzka.motivation.goal_system import GoalSystemSnapshot
from suzka.motivation.system import MotivationSystemSnapshot
from suzka.motivation.projection import (
    CommitmentPromptEntry,
    GoalPromptEntry,
    MotivationPromptEntry,
    _commitment_value,
    _goal_value,
    _motivation_value,
)
from suzka.persona.attention_prompt import (
    AttentionPromptPayload,
    build_attention_prompt_payload,
)
from suzka.persona.prompt_builder import PromptBuilder
from suzka.working_memory_contracts import (
    MAX_PROJECTION_BYTES,
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryRetentionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
    working_memory_item_id,
)


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
WM_SOURCE_ID = "opaque-private-wm-source-0001"
WM_TEXT = 'unicode: café 𐀀\nquote="x"; control=\x00; line=\u2028'


def _working_memory(
    content: str = WM_TEXT,
    *,
    source_id: str = WM_SOURCE_ID,
    activation: float = 0.8,
    salience: float = 0.6,
) -> tuple[tuple[WorkingMemoryItem, ...], WorkingMemoryView]:
    item_id = working_memory_item_id(WorkingMemorySourceKind.EPISODIC, source_id)
    item = WorkingMemoryItem(
        item_id=item_id,
        source_kind=WorkingMemorySourceKind.EPISODIC,
        source_id=source_id,
        activation=activation,
        salience=salience,
        retention_reason=WorkingMemoryRetentionReason.RECENT,
        created_revision=1,
        last_activated_revision=1,
    )
    score = 0.6 * activation + 0.4 * salience
    selection = WorkingMemorySelection(
        item_id=item_id,
        source_kind=item.source_kind,
        source_id=item.source_id,
        rendered_content=content,
        score=score,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    decision = WorkingMemoryDecision(
        item_id=item_id,
        source_kind=item.source_kind,
        source_id=item.source_id,
        selected=True,
        score=score,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    view = WorkingMemoryView(
        selected=(selection,),
        decisions=(decision,),
        projected_bytes=len(content.encode("utf-8")),
        item_capacity=1,
        projection_max_bytes=MAX_PROJECTION_BYTES,
        revision=1,
    )
    return (item,), view


def _source_snapshots(*, goal_description: str = "goal description 000"):
    from test_r13_projection import (
        _make_commitment_system,
        _make_goal_system,
        _make_motivation_system,
    )

    motivation = _make_motivation_system((MotivationLifecycle.ACTIVE,)).snapshot()
    goal = _make_goal_system(
        (R13GoalLifecycle.ADOPTED,), description=goal_description
    ).snapshot()
    # Keep the active Commitment's latest upstream event at sequence 5. Its
    # witness then does not conflict with the active Motivation (sequence 1)
    # or adopted Goal (sequence 3) under Attention's global retained-event
    # identity fence. The proposed second record is not projected.
    commitment = _make_commitment_system(
        (CommitmentLifecycle.ACTIVE, CommitmentLifecycle.PROPOSED)
    ).snapshot()
    return motivation, goal, commitment


def _capture(
    *,
    content: str = WM_TEXT,
    goal_description: str = "goal description 000",
    source_id: str = WM_SOURCE_ID,
) -> tuple[
    tuple[WorkingMemoryItem, ...],
    WorkingMemoryView,
    MotivationSystemSnapshot,
    GoalSystemSnapshot,
    CommitmentSystemSnapshot,
    AttentionRefreshResult,
]:
    items, wm_view = _working_memory(content, source_id=source_id)
    motivation, goals, commitments = _source_snapshots(
        goal_description=goal_description
    )
    event = AttentionEvent("attention:prompt-turn", 1_000, NOW)
    projections = project_attention_candidates(
        working_memory_items=items,
        working_memory_revision=1,
        working_memory_view=wm_view,
        motivation_snapshot=motivation,
        goal_snapshot=goals,
        commitment_snapshot=commitments,
        event=event,
    )
    source_events = tuple(
        source_event
        for projection in projections
        if (source_event := projection.source.event()) is not None
    )
    source_sequences = tuple(item.event_sequence for item in source_events)
    assert len(source_sequences) == len(set(source_sequences))
    assert all(item.event_sequence < event.event_sequence for item in source_events)
    assert all(item.occurred_at < event.occurred_at for item in source_events)
    refresh = AttentionSystem().refresh(projections, event)
    return items, wm_view, motivation, goals, commitments, refresh


def _payload(
    items: tuple[WorkingMemoryItem, ...],
    view: WorkingMemoryView,
    motivation: MotivationSystemSnapshot,
    goals: GoalSystemSnapshot,
    commitments: CommitmentSystemSnapshot,
    refresh: AttentionRefreshResult,
) -> AttentionPromptPayload:
    return build_attention_prompt_payload(
        refresh,
        working_memory_items=items,
        working_memory_revision=view.revision,
        working_memory_view=view,
        motivation_state=motivation,
        goal_state=goals,
        commitment_state=commitments,
    )


def test_current_selected_rows_are_revalidated_redacted_and_byte_truthful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suzka.attention.policy as attention_policy
    from suzka.motivation.projection import R13PromptView

    items, view, motivation, goals, commitments, refresh = _capture()
    source_memory = items[0]
    accepted_selection = refresh.view.prompt
    assert accepted_selection is not None
    old_sources = (
        source_memory.item_id,
        source_memory.source_id,
        view.selected[0].rendered_content,
        motivation.authority_digest,
        goals.authority_digest,
        commitments.authority_digest,
        refresh.snapshot.canonical_bytes(),
    )

    def whole_r13_render_is_forbidden(self: object) -> str:
        pytest.fail("selected Attention rendering called R13PromptView.render")

    def selection_must_not_run(*args: object, **kwargs: object) -> object:
        pytest.fail("selected Attention rendering invoked policy selection")

    monkeypatch.setattr(R13PromptView, "render", whole_r13_render_is_forbidden)
    monkeypatch.setattr(attention_policy, "compete_attention", selection_must_not_run)
    monkeypatch.setattr(
        attention_policy, "select_attention_prompt", selection_must_not_run
    )
    payload = _payload(items, view, motivation, goals, commitments, refresh)

    assert type(payload) is AttentionPromptPayload
    assert payload.included_candidate_ids == accepted_selection.included_candidate_ids
    assert payload.selection_digest == accepted_selection.result_digest
    assert payload.witnessed_bytes == accepted_selection.total_bytes
    assert payload.rendered_bytes == len(payload.rendered_text.encode("utf-8"))
    assert payload.rendered_bytes <= payload.witnessed_bytes <= ATTENTION_PROMPT_BUDGET_BYTES
    assert payload.rendered_digest != accepted_selection.result_digest
    expected_frame = ATTENTION_PROMPT_AUTHORITY_INTRO + "".join(
        heading for _, heading in ATTENTION_PROMPT_SECTION_HEADERS
    )
    expected_frame_bytes = len(expected_frame.encode("ascii"))
    assert accepted_selection.frame_bytes == expected_frame_bytes
    assert ATTENTION_PROMPT_FRAME_BYTES == expected_frame_bytes
    assert payload.rendered_text.startswith(
        ATTENTION_PROMPT_AUTHORITY_INTRO + ATTENTION_PROMPT_SECTION_HEADERS[0][1]
    )

    lines = payload.rendered_text.splitlines()
    section_headers = tuple(
        heading.rstrip("\n") for _, heading in ATTENTION_PROMPT_SECTION_HEADERS
    )
    observed_headers = tuple(line for line in lines if line in section_headers)
    assert observed_headers == section_headers
    assert all(lines.count(heading) == 1 for heading in section_headers)
    section_rows: dict[str, list[dict[str, object]]] = {}
    for index, heading in enumerate(section_headers):
        start = lines.index(heading) + 1
        end = (
            lines.index(section_headers[index + 1])
            if index + 1 < len(section_headers)
            else len(lines)
        )
        section_rows[heading] = [
            json.loads(line) for line in lines[start:end] if line.startswith("{")
        ]
    rows = [row for current_rows in section_rows.values() for row in current_rows]
    wm_rows = [row for row in rows if set(row) == {"source_kind", "text"}]
    assert len(wm_rows) == 1
    assert wm_rows[0] == {
        "source_kind": "episodic",
        "text": WM_TEXT,
    }
    assert source_memory.item_id not in payload.rendered_text
    assert source_memory.source_id not in payload.rendered_text
    assert all("source_id" not in row and "item_id" not in row for row in wm_rows)

    projections = project_attention_candidates(
        working_memory_items=items,
        working_memory_revision=view.revision,
        working_memory_view=view,
        motivation_snapshot=motivation,
        goal_snapshot=goals,
        commitment_snapshot=commitments,
        event=refresh.receipt.event,
    )
    working_memory_projection = next(
        item
        for item in projections
        if item.target.kind is AttentionTargetKind.WORKING_MEMORY
    )
    source_row = {
        "item_id": source_memory.item_id,
        "source_kind": source_memory.source_kind.value,
        "source_id": source_memory.source_id,
        "text": WM_TEXT,
    }
    assert working_memory_projection.rendered_bytes == len(canonical_json(source_row))
    assert working_memory_projection.rendered_digest == digest_payload(
        _WM_RENDERED_ROW_DOMAIN, source_row
    )
    assert payload.rendered_digest != working_memory_projection.rendered_digest
    expected_r13 = (
        (
            MotivationPromptEntry._from_record(motivation.records[0]),
            _motivation_value,
        ),
        (GoalPromptEntry._from_record(goals.records[0]), _goal_value),
        (
            CommitmentPromptEntry._from_record(
                next(
                    record
                    for record in commitments.records
                    if record.lifecycle is CommitmentLifecycle.ACTIVE
                )
            ),
            _commitment_value,
        ),
    )
    included_kinds = {
        projection.target.kind
        for projection in projections
        if projection.candidate_id in payload.included_candidate_ids
    }
    assert included_kinds == set(AttentionTargetKind)
    assert section_rows[section_headers[0]] == [
        {"source_kind": "episodic", "text": WM_TEXT}
    ]
    for entry, serializer in expected_r13:
        row = serializer(entry)
        expected_reference = next(
            getattr(entry, name)
            for name in ("motivation_id", "goal_id", "commitment_id")
            if hasattr(entry, name)
        )
        projection = next(
            item for item in projections if item.target.reference == expected_reference
        )
        assert projection.rendered_bytes == len(canonical_json(row))
        assert projection.rendered_digest == digest_payload(
            _R13_RENDERED_ROW_DOMAIN, row
        )
        assert canonical_json(row).decode("ascii") in payload.rendered_text
        expected_section = {
            AttentionTargetKind.MOTIVATION: section_headers[1],
            AttentionTargetKind.GOAL: section_headers[2],
            AttentionTargetKind.COMMITMENT: section_headers[3],
        }[projection.target.kind]
        assert section_rows[expected_section] == [row]
    assert old_sources == (
        source_memory.item_id,
        source_memory.source_id,
        view.selected[0].rendered_content,
        motivation.authority_digest,
        goals.authority_digest,
        commitments.authority_digest,
        refresh.snapshot.canonical_bytes(),
    )

    prompt = PromptBuilder().build(
        "same turn user text",
        EmotionState(),
        view,
        attention_payload=payload,
    )
    assert payload.rendered_text in prompt
    assert "Stored episodic evidence:" not in prompt
    assert "Stored semantic evidence:" not in prompt
    assert "A current Motivation is not a Goal" not in prompt
    assert "Retrieved Memory is stored evidence" in prompt
    assert "Emotion:\n- valence: 0.000000" in prompt
    assert "User: same turn user text\nAssistant:" in prompt


def test_empty_focus_renders_the_exact_fixed_frame_and_no_rows() -> None:
    from test_r13_projection import (
        _make_commitment_system,
        _make_goal_system,
        _make_motivation_system,
    )

    items: tuple[WorkingMemoryItem, ...] = ()
    view = WorkingMemoryView(
        selected=(),
        decisions=(),
        projected_bytes=0,
        item_capacity=1,
        projection_max_bytes=MAX_PROJECTION_BYTES,
        revision=0,
    )
    motivation = _make_motivation_system(()).snapshot()
    goals = _make_goal_system(()).snapshot()
    commitments = _make_commitment_system(()).snapshot()
    event = AttentionEvent("attention:idle-turn", 1_001, NOW)
    projections = project_attention_candidates(
        working_memory_items=items,
        working_memory_revision=0,
        working_memory_view=view,
        motivation_snapshot=motivation,
        goal_snapshot=goals,
        commitment_snapshot=commitments,
        event=event,
    )
    refresh = AttentionSystem().refresh(projections, event)
    payload = _payload(items, view, motivation, goals, commitments, refresh)

    assert refresh.view.prompt is not None
    assert refresh.view.prompt.included_candidate_ids == ()
    assert refresh.view.prompt.total_bytes == ATTENTION_PROMPT_FRAME_BYTES
    assert payload.included_candidate_ids == ()
    assert payload.witnessed_bytes == ATTENTION_PROMPT_FRAME_BYTES
    assert payload.rendered_text == ATTENTION_PROMPT_FRAME_TEXT
    assert payload.rendered_bytes == ATTENTION_PROMPT_FRAME_BYTES


def test_working_memory_id_text_occurrences_are_preserved_without_identity_fields() -> None:
    item_id = working_memory_item_id(
        WorkingMemorySourceKind.EPISODIC, WM_SOURCE_ID
    )
    natural_id_text = f"Literal source labels: item={item_id}; source={WM_SOURCE_ID}"
    items, view, motivation, goals, commitments, refresh = _capture(
        content=natural_id_text
    )
    payload = _payload(items, view, motivation, goals, commitments, refresh)

    rendered_rows = [
        json.loads(line)
        for line in payload.rendered_text.splitlines()
        if line.startswith("{")
    ]
    working_memory_rows = [
        row for row in rendered_rows if set(row) == {"source_kind", "text"}
    ]
    assert len(working_memory_rows) == 1
    row = working_memory_rows[0]
    assert row == {"source_kind": "episodic", "text": natural_id_text}
    assert item_id in row["text"]
    assert WM_SOURCE_ID in row["text"]
    assert "item_id" not in row
    assert "source_id" not in row


def test_changed_working_memory_content_cannot_reuse_accepted_selection() -> None:
    items, view, motivation, goals, commitments, refresh = _capture()
    changed_selection = replace(
        view.selected[0], rendered_content="changed after Attention capture"
    )
    changed_view = replace(
        view,
        selected=(changed_selection,),
        projected_bytes=len(changed_selection.rendered_content.encode("utf-8")),
    )

    with pytest.raises(ValueError, match="competition decision differs"):
        _payload(items, changed_view, motivation, goals, commitments, refresh)


def test_changed_goal_source_snapshot_or_tampered_revision_is_rejected() -> None:
    items, view, motivation, goals, commitments, refresh = _capture()
    _, changed_goals, _ = _source_snapshots(goal_description="new current description")

    with pytest.raises(ValueError, match="Attention root witness"):
        _payload(items, view, motivation, changed_goals, commitments, refresh)

    original_revision = goals.records[0].revision
    object.__setattr__(goals.records[0], "revision", original_revision + 1)
    with pytest.raises(ValueError, match="Goal record is invalid"):
        _payload(items, view, motivation, goals, commitments, refresh)
    assert goals.records[0].revision == original_revision + 1


def test_restored_snapshot_without_ephemeral_competition_is_not_a_current_refresh() -> None:
    items, view, motivation, goals, commitments, refresh = _capture()
    restored_view = AttentionSystem(refresh.snapshot).selected_view()
    restored_refresh = AttentionRefreshResult(
        snapshot=refresh.snapshot,
        receipt=refresh.receipt,
        replayed=False,
        view=restored_view,
    )

    with pytest.raises(ValueError, match="competition evidence is required"):
        _payload(items, view, motivation, goals, commitments, restored_refresh)


def test_payload_copy_detects_text_or_digest_tampering_without_repair() -> None:
    items, view, motivation, goals, commitments, refresh = _capture()
    payload = _payload(items, view, motivation, goals, commitments, refresh)
    original_digest = payload.rendered_digest
    original_text = payload.rendered_text
    object.__setattr__(payload, "rendered_text", original_text + "tampered")

    with pytest.raises(ValueError, match="(rendered_bytes|rendered digest)"):
        payload.validated_copy()
    assert payload.rendered_text == original_text + "tampered"
    assert payload.rendered_digest == original_digest
    with pytest.raises(ValueError, match="(rendered_bytes|rendered digest)"):
        payload.render()
    assert payload.rendered_digest == original_digest
    assert payload.rendered_text == original_text + "tampered"

    fresh = _payload(items, view, motivation, goals, commitments, refresh)
    object.__setattr__(fresh, "rendered_digest", "0" * 64)
    with pytest.raises(ValueError, match="rendered digest"):
        fresh.validated_copy()
    assert fresh.rendered_digest == "0" * 64

    wrong_byte_count = _payload(items, view, motivation, goals, commitments, refresh)
    declared_rendered_bytes = wrong_byte_count.rendered_bytes
    object.__setattr__(wrong_byte_count, "rendered_bytes", declared_rendered_bytes + 1)
    with pytest.raises(ValueError, match="rendered_bytes does not match"):
        wrong_byte_count.validated_copy()
    assert wrong_byte_count.rendered_bytes == declared_rendered_bytes + 1

    wrong_witness_type = _payload(items, view, motivation, goals, commitments, refresh)
    object.__setattr__(wrong_witness_type, "witnessed_bytes", True)
    with pytest.raises(ValueError, match="witnessed_bytes"):
        wrong_witness_type.validated_copy()
    assert wrong_witness_type.witnessed_bytes is True


def test_payload_character_bound_precedes_event_copy_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import suzka.persona.attention_prompt as attention_prompt_module

    items, view, motivation, goals, commitments, refresh = _capture()
    payload = _payload(items, view, motivation, goals, commitments, refresh)
    original_digest = payload.rendered_digest
    original_byte_count = payload.rendered_bytes
    oversized_text = "x" * (ATTENTION_PROMPT_BUDGET_BYTES + 1)

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("oversized prompt text reached event copying or digesting")

    monkeypatch.setattr(attention_prompt_module, "_checked_event", forbidden)
    monkeypatch.setattr(attention_prompt_module, "_render_digest", forbidden)
    object.__setattr__(payload, "rendered_text", oversized_text)

    with pytest.raises(ValueError, match="character bound"):
        payload.validated_copy()
    assert payload.rendered_text == oversized_text
    assert payload.rendered_bytes == original_byte_count
    assert payload.rendered_digest == original_digest

    with pytest.raises(ValueError, match="character bound"):
        AttentionPromptPayload._create(
            event=payload.event,
            selection_digest=payload.selection_digest,
            included_candidate_ids=payload.included_candidate_ids,
            witnessed_bytes=payload.witnessed_bytes,
            rendered_text=oversized_text,
        )


def test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation() -> None:
    huge_text = "z" * (ATTENTION_PROMPT_BUDGET_BYTES + 500)
    items, view, motivation, goals, commitments, refresh = _capture(content=huge_text)
    prompt_selection = refresh.view.prompt
    assert prompt_selection is not None
    wm_candidate_id = next(
        candidate.candidate_id
        for candidate in refresh.snapshot.candidates
        if candidate.target.kind is AttentionTargetKind.WORKING_MEMORY
    )
    wm_decision = next(
        decision
        for decision in prompt_selection.decisions
        if decision.candidate_id == wm_candidate_id
    )
    assert wm_decision.reason is AttentionPromptReason.OVER_BUDGET_SINGLE
    payload = _payload(items, view, motivation, goals, commitments, refresh)

    assert wm_candidate_id not in payload.included_candidate_ids
    assert huge_text not in payload.rendered_text
    assert payload.witnessed_bytes == prompt_selection.total_bytes
    assert payload.rendered_bytes <= payload.witnessed_bytes
    assert payload.rendered_bytes <= ATTENTION_PROMPT_BUDGET_BYTES


def test_attention_builder_rejects_combined_legacy_r13_fallback() -> None:
    from suzka.motivation.projection import R13PromptView

    items, view, motivation, goals, commitments, refresh = _capture()
    payload = _payload(items, view, motivation, goals, commitments, refresh)
    r13_view = R13PromptView.from_snapshots(motivation, goals, commitments)

    with pytest.raises(ValueError, match="cannot be combined"):
        PromptBuilder().build(
            "user",
            EmotionState(),
            view,
            r13_view=r13_view,
            attention_payload=payload,
        )


def test_none_payload_preserves_legacy_working_memory_and_r13_rendering() -> None:
    from suzka.motivation.projection import R13PromptView

    _, view, motivation, goals, commitments, _ = _capture()
    r13_view = R13PromptView.from_snapshots(motivation, goals, commitments)
    legacy_r13_text = r13_view.render()

    prompt = PromptBuilder().build(
        "legacy user",
        EmotionState(),
        view,
        r13_view=r13_view,
    )

    assert legacy_r13_text in prompt
    assert f"- {WM_TEXT}" in prompt
    assert "Stored episodic evidence:" in prompt
    assert "Stored semantic evidence:\n- none" in prompt
    assert "User: legacy user\nAssistant:" in prompt
