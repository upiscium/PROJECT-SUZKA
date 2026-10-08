"""Request-scoped rendering of an accepted Attention prompt selection.

The source adapters and Attention policy own all candidate projection and
selection.  This module only revalidates that accepted evidence against the
current typed source capture, verifies each selected source-row witness, and
renders the already-selected rows without their opaque Working Memory IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from suzka.attention.adapters import (
    _R13_RENDERED_ROW_DOMAIN,
    _WM_RENDERED_ROW_DOMAIN,
    project_attention_candidates,
)
from suzka.attention.common import (
    ATTENTION_MAX_FOCUS,
    ATTENTION_PROMPT_BUDGET_BYTES,
    AttentionTargetKind,
    CandidateAvailability,
    canonical_json,
    digest_payload,
    validate_digest,
)
from suzka.attention.contracts import (
    AttentionCandidateProjection,
    AttentionContinuity,
    AttentionEvent,
    AttentionEventReceipt,
    AttentionTarget,
)
from suzka.attention.policy import (
    ATTENTION_PROMPT_AUTHORITY_INTRO,
    ATTENTION_PROMPT_SECTION_HEADERS,
    AttentionCompetitionResult,
    AttentionPromptReason,
    AttentionPromptSelection,
    _ATTENTION_PROMPT_INPUT_DOMAIN,
)
from suzka.attention.system import AttentionRefreshResult, AttentionSelectedView
from suzka.motivation.commitment import CommitmentLifecycle, CommitmentRecord
from suzka.motivation.commitment_system import CommitmentSystemSnapshot
from suzka.motivation.goal import GoalLifecycle, GoalRecord
from suzka.motivation.goal_system import GoalSystemSnapshot
from suzka.motivation.motivation import MotivationLifecycle, MotivationRecord
from suzka.motivation.projection import (
    CommitmentPromptEntry,
    GoalPromptEntry,
    MotivationPromptEntry,
    _commitment_value,
    _goal_value,
    _motivation_value,
)
from suzka.motivation.system import MotivationSystemSnapshot
from suzka.working_memory_contracts import (
    WorkingMemoryItem,
    WorkingMemorySelection,
    WorkingMemoryView,
)


_ATTENTION_PROMPT_RENDER_DOMAIN: Final[bytes] = (
    b"PROJECT-SUZKA:R14:ATTENTION-PROMPT-RENDER:V1\0"
)


def _checked_event(value: object) -> AttentionEvent:
    if type(value) is not AttentionEvent:
        raise TypeError("event must be an exact AttentionEvent")
    try:
        return AttentionEvent.from_canonical_value(value.canonical_value())
    except Exception:
        raise ValueError("Attention event is invalid") from None


def _render_digest(
    *,
    event: AttentionEvent,
    selection_digest: str,
    included_candidate_ids: tuple[str, ...],
    witnessed_bytes: int,
    rendered_bytes: int,
    rendered_text: str,
) -> str:
    return digest_payload(
        _ATTENTION_PROMPT_RENDER_DOMAIN,
        {
            "event": event.canonical_value(),
            "included_candidate_ids": list(included_candidate_ids),
            "rendered_bytes": rendered_bytes,
            "rendered_text": rendered_text,
            "selection_digest": selection_digest,
            "witnessed_bytes": witnessed_bytes,
        },
    )


@dataclass(frozen=True, slots=True, init=False)
class AttentionPromptPayload:
    """Sealed, bounded, ephemeral selected text for the production prompt.

    ``witnessed_bytes`` accounts for the complete selected source rows recorded
    by Attention. ``rendered_bytes`` accounts for the actual fixed frame and
    privacy-redacted Working Memory rows returned by :meth:`render`.  The two
    digests intentionally use different domains and bind different data.
    Payloads contain request-scoped raw text and must not be stored in a root,
    commitment, or read view.
    """

    event: AttentionEvent
    selection_digest: str
    included_candidate_ids: tuple[str, ...]
    witnessed_bytes: int
    rendered_bytes: int
    rendered_text: str
    rendered_digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("AttentionPromptPayload values are created by the source adapter")

    @classmethod
    def _create(
        cls,
        *,
        event: AttentionEvent,
        selection_digest: str,
        included_candidate_ids: tuple[str, ...],
        witnessed_bytes: int,
        rendered_text: str,
    ) -> AttentionPromptPayload:
        if cls is not AttentionPromptPayload:
            raise TypeError("AttentionPromptPayload cannot be subclassed")
        if type(rendered_text) is not str:
            raise TypeError("rendered_text must be an exact string")
        if len(rendered_text) > ATTENTION_PROMPT_BUDGET_BYTES:
            raise ValueError("rendered_text exceeds the Attention prompt character bound")
        checked_event = _checked_event(event)
        checked_selection_digest = validate_digest(selection_digest, "selection_digest")
        if type(included_candidate_ids) is not tuple:
            raise TypeError("included_candidate_ids must be an exact tuple")
        checked_ids = tuple(candidate_id for candidate_id in included_candidate_ids)
        if len(checked_ids) > ATTENTION_MAX_FOCUS:
            raise ValueError("Attention prompt payload exceeds its focused-row bound")
        if checked_ids != tuple(dict.fromkeys(checked_ids)):
            raise ValueError("included candidate IDs must be unique and retain focus order")
        for candidate_id in checked_ids:
            validate_digest(candidate_id, "included candidate ID")
        if (
            type(witnessed_bytes) is not int
            or not 1 <= witnessed_bytes <= ATTENTION_PROMPT_BUDGET_BYTES
        ):
            raise ValueError("witnessed_bytes exceed the accepted Attention prompt budget")
        try:
            rendered_utf8_bytes = rendered_text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("rendered prompt text must be valid UTF-8") from None
        rendered_bytes = len(rendered_utf8_bytes)
        if rendered_bytes > ATTENTION_PROMPT_BUDGET_BYTES:
            raise ValueError("rendered_text exceeds the Attention prompt byte bound")
        if rendered_bytes > witnessed_bytes:
            raise ValueError("rendered prompt bytes exceed the selected source witnesses")
        result = object.__new__(cls)
        object.__setattr__(result, "event", checked_event)
        object.__setattr__(result, "selection_digest", checked_selection_digest)
        object.__setattr__(result, "included_candidate_ids", checked_ids)
        object.__setattr__(result, "witnessed_bytes", witnessed_bytes)
        object.__setattr__(result, "rendered_bytes", rendered_bytes)
        object.__setattr__(result, "rendered_text", rendered_text)
        object.__setattr__(
            result,
            "rendered_digest",
            _render_digest(
                event=checked_event,
                selection_digest=checked_selection_digest,
                included_candidate_ids=checked_ids,
                witnessed_bytes=witnessed_bytes,
                rendered_bytes=rendered_bytes,
                rendered_text=rendered_text,
            ),
        )
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        if type(self.rendered_text) is not str:
            raise TypeError("rendered_text must be an exact string")
        if len(self.rendered_text) > ATTENTION_PROMPT_BUDGET_BYTES:
            raise ValueError("rendered_text exceeds the Attention prompt character bound")
        event = _checked_event(self.event)
        validate_digest(self.selection_digest, "selection_digest")
        if type(self.included_candidate_ids) is not tuple:
            raise TypeError("included_candidate_ids must be an exact tuple")
        if len(self.included_candidate_ids) > ATTENTION_MAX_FOCUS:
            raise ValueError("Attention prompt payload exceeds its focused-row bound")
        if self.included_candidate_ids != tuple(dict.fromkeys(self.included_candidate_ids)):
            raise ValueError("included candidate IDs must be unique and retain focus order")
        for candidate_id in self.included_candidate_ids:
            validate_digest(candidate_id, "included candidate ID")
        if (
            type(self.witnessed_bytes) is not int
            or not 1 <= self.witnessed_bytes <= ATTENTION_PROMPT_BUDGET_BYTES
        ):
            raise ValueError("witnessed_bytes exceed the accepted Attention prompt budget")
        try:
            rendered_utf8_bytes = self.rendered_text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("rendered_text must be valid UTF-8") from None
        if len(rendered_utf8_bytes) > ATTENTION_PROMPT_BUDGET_BYTES:
            raise ValueError("rendered_text exceeds the Attention prompt byte bound")
        if (
            type(self.rendered_bytes) is not int
            or self.rendered_bytes != len(rendered_utf8_bytes)
        ):
            raise ValueError("rendered_bytes does not match the actual UTF-8 payload")
        if self.rendered_bytes > self.witnessed_bytes:
            raise ValueError("rendered prompt bytes exceed the selected source witnesses")
        declared_digest = validate_digest(self.rendered_digest, "rendered_digest")
        if declared_digest != _render_digest(
            event=event,
            selection_digest=self.selection_digest,
            included_candidate_ids=self.included_candidate_ids,
            witnessed_bytes=self.witnessed_bytes,
            rendered_bytes=self.rendered_bytes,
            rendered_text=self.rendered_text,
        ):
            raise ValueError("Attention prompt rendered digest does not match its contents")

    def validated_copy(self) -> AttentionPromptPayload:
        """Return a detached copy, rejecting rather than repairing tampering."""

        if type(self) is not AttentionPromptPayload:
            raise TypeError("value must be an exact AttentionPromptPayload")
        # Validate every published field before constructing anything from it.
        # In particular, _create derives rendered_bytes from the text, so it
        # must never be allowed to silently repair a tampered declared count.
        self.__post_init__()
        declared_digest = validate_digest(self.rendered_digest, "rendered_digest")
        copied = AttentionPromptPayload._create(
            event=self.event,
            selection_digest=self.selection_digest,
            included_candidate_ids=self.included_candidate_ids,
            witnessed_bytes=self.witnessed_bytes,
            rendered_text=self.rendered_text,
        )
        if (
            copied.event != self.event
            or copied.selection_digest != self.selection_digest
            or copied.included_candidate_ids != self.included_candidate_ids
            or copied.witnessed_bytes != self.witnessed_bytes
            or copied.rendered_bytes != self.rendered_bytes
            or copied.rendered_text != self.rendered_text
            or copied.rendered_digest != declared_digest
        ):
            raise ValueError("Attention prompt rendered digest does not match its contents")
        return copied

    def render(self) -> str:
        """Return the actual bounded selected contribution after revalidation."""

        return self.validated_copy().rendered_text


def _validated_refresh(
    refresh: AttentionRefreshResult,
) -> tuple[
    AttentionEvent,
    AttentionContinuity,
    AttentionCompetitionResult,
    AttentionPromptSelection,
]:
    if type(refresh) is not AttentionRefreshResult:
        raise TypeError("refresh must be an exact AttentionRefreshResult")
    if type(refresh.snapshot) is not AttentionContinuity:
        raise TypeError("refresh snapshot must be an exact AttentionContinuity")
    if type(refresh.receipt) is not AttentionEventReceipt:
        raise TypeError("refresh receipt must be an exact AttentionEventReceipt")
    try:
        snapshot = AttentionContinuity.from_canonical_value(
            refresh.snapshot.canonical_value()
        )
        receipt = AttentionEventReceipt.from_canonical_value(
            refresh.receipt.canonical_value()
        )
    except Exception:
        raise ValueError("Attention refresh root or receipt is invalid") from None
    if type(refresh.view) is not AttentionSelectedView:
        raise TypeError("refresh view must be an exact AttentionSelectedView")
    view = refresh.view
    if type(view.competition) is not AttentionCompetitionResult:
        raise ValueError("current Attention competition evidence is required")
    if type(view.prompt) is not AttentionPromptSelection:
        raise ValueError("current Attention prompt selection evidence is required")

    competition = view.competition
    prompt = view.prompt
    try:
        competition.canonical_value()
        prompt.canonical_value()
        if type(view.focused_targets) is not tuple or any(
            type(target) is not AttentionTarget for target in view.focused_targets
        ):
            raise TypeError("focused target rows are invalid")
        if type(view.unfinished_targets) is not tuple or any(
            type(target) is not AttentionTarget for target in view.unfinished_targets
        ):
            raise TypeError("unfinished target rows are invalid")
        checked_view = AttentionSelectedView(
            revision=view.revision,
            event=None if view.event is None else _checked_event(view.event),
            state_digest=view.state_digest,
            authority_digest=view.authority_digest,
            focused_targets=tuple(
                AttentionTarget(target.kind, target.reference)
                for target in view.focused_targets
            ),
            unfinished_targets=tuple(
                AttentionTarget(target.kind, target.reference)
                for target in view.unfinished_targets
            ),
            competition=competition,
            prompt=prompt,
        )
    except Exception:
        raise ValueError("Attention competition or prompt evidence is invalid") from None

    current_event = receipt.event
    if (
        snapshot.last_event is None
        or receipt.event != snapshot.last_event
        or receipt.result_state_digest != snapshot.state_digest
        or not snapshot.receipts
        or snapshot.receipts[-1].receipt_digest != receipt.receipt_digest
    ):
        raise ValueError("refresh receipt must prove the current Attention root event")
    if (
        checked_view.revision != snapshot.revision
        or checked_view.state_digest != snapshot.state_digest
        or checked_view.authority_digest != snapshot.authority_digest
        or checked_view.event != current_event
        or competition.event != current_event
    ):
        raise ValueError("selected view and competition must match the current Attention root")
    if (
        competition.focused_ids != snapshot.focused_ids
        or competition.unfinished_ids != snapshot.unfinished_ids
    ):
        raise ValueError("competition focus evidence differs from the current Attention root")
    if (
        prompt.competition_digest != competition.result_digest
        or prompt.focus_capacity != competition.focus_capacity
    ):
        raise ValueError("prompt selection does not bind the current competition")

    candidates_by_id = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
    if tuple(target.candidate_id for target in checked_view.focused_targets) != snapshot.focused_ids:
        raise ValueError("selected view focused references differ from the current Attention root")
    if tuple(target.candidate_id for target in checked_view.unfinished_targets) != snapshot.unfinished_ids:
        raise ValueError("selected view unfinished references differ from the current Attention root")
    if any(
        target != candidates_by_id[candidate_id].target
        for target, candidate_id in zip(checked_view.focused_targets, snapshot.focused_ids, strict=True)
    ) or any(
        target != candidates_by_id[candidate_id].target
        for target, candidate_id in zip(checked_view.unfinished_targets, snapshot.unfinished_ids, strict=True)
    ):
        raise ValueError("selected view target references differ from the current Attention root")
    return current_event, snapshot, competition, prompt


def _validate_projection_bindings(
    *,
    projections: tuple[AttentionCandidateProjection, ...],
    snapshot: AttentionContinuity,
    competition: AttentionCompetitionResult,
    selection: AttentionPromptSelection,
) -> dict[str, AttentionCandidateProjection]:
    projections_by_id = {item.candidate_id: item for item in projections}
    if tuple(projections_by_id) != tuple(item.candidate_id for item in projections):
        raise ValueError("current source projections must use canonical candidate order")
    if set(projections_by_id) != {candidate.candidate_id for candidate in snapshot.candidates}:
        raise ValueError("current source capture differs from the complete Attention root")

    candidates_by_id = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
    competition_by_id = {item.candidate_id: item for item in competition.decisions}
    prompt_by_id = {item.candidate_id: item for item in selection.decisions}
    expected_ids = set(projections_by_id)
    if set(competition_by_id) != expected_ids or set(prompt_by_id) != expected_ids:
        raise ValueError("competition and prompt must cover the complete current source universe")

    for candidate_id, projection in projections_by_id.items():
        current_candidate = candidates_by_id[candidate_id]
        if (
            current_candidate.target != projection.target
            or current_candidate.source != projection.source
            or current_candidate.availability is not projection.availability
        ):
            raise ValueError("current source projection differs from its Attention root witness")
        decision = competition_by_id[candidate_id]
        if (
            decision.projection_digest != projection.projection_digest
            or decision.availability is not projection.availability
        ):
            raise ValueError("competition decision differs from the current source projection")
        prompt_decision = prompt_by_id[candidate_id]
        if (
            prompt_decision.projection_digest != projection.projection_digest
            or prompt_decision.rendered_bytes != projection.rendered_bytes
            or prompt_decision.rendered_digest != projection.rendered_digest
            or prompt_decision.competition_reason is not decision.reason
        ):
            raise ValueError("prompt decision differs from its current source witnesses")

    expected_prompt_input = digest_payload(
        _ATTENTION_PROMPT_INPUT_DOMAIN,
        {
            "competition_digest": competition.result_digest,
            "projection_digests": [
                projections_by_id[candidate_id].projection_digest
                for candidate_id in sorted(projections_by_id)
            ],
        },
    )
    if selection.input_digest != expected_prompt_input:
        raise ValueError("prompt input digest does not bind the complete current source capture")
    included_from_focus_order = tuple(
        candidate_id
        for candidate_id in competition.focus_order
        if _selection_by_id_reason(selection, candidate_id) is AttentionPromptReason.INCLUDED
    )
    if included_from_focus_order != selection.included_candidate_ids:
        raise ValueError("prompt included IDs do not preserve the accepted focus order")
    return projections_by_id


def _selection_by_id_reason(
    selection: AttentionPromptSelection, candidate_id: str
) -> AttentionPromptReason | None:
    for decision in selection.decisions:
        if decision.candidate_id == candidate_id:
            return decision.reason
    return None


def _selected_working_memory_row(
    *,
    projection: AttentionCandidateProjection,
    item_by_id: dict[str, WorkingMemoryItem],
    selection_by_id: dict[str, WorkingMemorySelection],
) -> tuple[dict[str, object], dict[str, object]]:
    item = item_by_id.get(projection.source.reference)
    selected = selection_by_id.get(projection.source.reference)
    if item is None or selected is None:
        raise ValueError("selected Working Memory source row is absent from the current capture")
    source_row: dict[str, object] = {
        "item_id": item.item_id,
        "source_kind": item.source_kind.value,
        "source_id": item.source_id,
        "text": selected.rendered_content,
    }
    source_bytes = canonical_json(source_row)
    source_digest = digest_payload(_WM_RENDERED_ROW_DOMAIN, source_row)
    if (
        projection.rendered_bytes != len(source_bytes)
        or projection.rendered_digest != source_digest
    ):
        raise ValueError("Working Memory source row differs from its accepted witness")
    visible_row: dict[str, object] = {
        "source_kind": item.source_kind.value,
        "text": selected.rendered_content,
    }
    if set(visible_row) != {"source_kind", "text"} or any(
        visible_row[key] != source_row[key] for key in visible_row
    ):
        raise ValueError("privacy-redacted Working Memory row changed witnessed values")
    if len(canonical_json(visible_row)) > len(source_bytes):
        raise ValueError("privacy-redacted Working Memory row exceeds its source witness")
    return source_row, visible_row


def _selected_r13_row(
    *,
    projection: AttentionCandidateProjection,
    record: MotivationRecord | GoalRecord | CommitmentRecord,
) -> dict[str, object]:
    if projection.target.kind is AttentionTargetKind.MOTIVATION:
        if type(record) is not MotivationRecord:
            raise TypeError("Motivation row requires an exact MotivationRecord")
        motivation_entry = MotivationPromptEntry._from_record(record)
        row = _motivation_value(motivation_entry)
    elif projection.target.kind is AttentionTargetKind.GOAL:
        if type(record) is not GoalRecord:
            raise TypeError("Goal row requires an exact GoalRecord")
        goal_entry = GoalPromptEntry._from_record(record)
        row = _goal_value(goal_entry)
    elif projection.target.kind is AttentionTargetKind.COMMITMENT:
        if type(record) is not CommitmentRecord:
            raise TypeError("Commitment row requires an exact CommitmentRecord")
        commitment_entry = CommitmentPromptEntry._from_record(record)
        row = _commitment_value(commitment_entry)
    else:
        raise ValueError("R13 source row has a non-R13 target kind")
    row_bytes = canonical_json(row)
    if projection.rendered_bytes != len(row_bytes) or projection.rendered_digest != digest_payload(
        _R13_RENDERED_ROW_DOMAIN, row
    ):
        raise ValueError("R13 source row differs from its accepted witness")
    return row


def _render_selected_rows(
    *,
    event: AttentionEvent,
    selection: AttentionPromptSelection,
    projections_by_id: dict[str, AttentionCandidateProjection],
    working_memory_items: tuple[WorkingMemoryItem, ...],
    working_memory_view: WorkingMemoryView,
    motivation_state: MotivationSystemSnapshot,
    goal_state: GoalSystemSnapshot,
    commitment_state: CommitmentSystemSnapshot,
) -> AttentionPromptPayload:
    item_by_id = {item.item_id: item for item in working_memory_items}
    selection_by_id = {item.item_id: item for item in working_memory_view.selected}
    motivation_by_id = {
        record.motivation_id: record
        for record in motivation_state.records
        if record.lifecycle is MotivationLifecycle.ACTIVE
    }
    goal_by_id = {
        record.goal_id: record
        for record in goal_state.records
        if record.lifecycle is GoalLifecycle.ADOPTED
    }
    commitment_by_id = {
        record.commitment_id: record
        for record in commitment_state.records
        if record.lifecycle is CommitmentLifecycle.ACTIVE
    }

    rendered_rows_by_kind: dict[AttentionTargetKind, list[bytes]] = {
        kind: [] for kind, _ in ATTENTION_PROMPT_SECTION_HEADERS
    }
    witnessed_row_bytes = 0
    rendered_row_bytes = 0
    for candidate_id in selection.included_candidate_ids:
        projection = projections_by_id[candidate_id]
        if projection.availability is not CandidateAvailability.ELIGIBLE:
            raise ValueError("included prompt candidates must remain source eligible")
        if projection.target.kind is AttentionTargetKind.WORKING_MEMORY:
            source_row, visible_row = _selected_working_memory_row(
                projection=projection,
                item_by_id=item_by_id,
                selection_by_id=selection_by_id,
            )
            witnessed_row_bytes += len(canonical_json(source_row))
            visible = canonical_json(visible_row)
        elif projection.target.kind is AttentionTargetKind.MOTIVATION:
            motivation_record = motivation_by_id.get(projection.target.reference)
            if motivation_record is None:
                raise ValueError("selected Motivation is absent from the current source snapshot")
            row = _selected_r13_row(
                projection=projection, record=motivation_record
            )
            witnessed_row_bytes += len(canonical_json(row))
            visible = canonical_json(row)
        elif projection.target.kind is AttentionTargetKind.GOAL:
            goal_record = goal_by_id.get(projection.target.reference)
            if goal_record is None:
                raise ValueError("selected Goal is absent from the current source snapshot")
            row = _selected_r13_row(projection=projection, record=goal_record)
            witnessed_row_bytes += len(canonical_json(row))
            visible = canonical_json(row)
        elif projection.target.kind is AttentionTargetKind.COMMITMENT:
            commitment_record = commitment_by_id.get(projection.target.reference)
            if commitment_record is None:
                raise ValueError("selected Commitment is absent from the current source snapshot")
            row = _selected_r13_row(
                projection=projection, record=commitment_record
            )
            witnessed_row_bytes += len(canonical_json(row))
            visible = canonical_json(row)
        else:
            raise ValueError("prompt selection contains an unsupported target kind")

        row_bytes = len(visible)
        decision = next(
            item for item in selection.decisions if item.candidate_id == candidate_id
        )
        if decision.rendered_bytes is None:
            raise ValueError("included prompt row lacks its original byte witness")
        if projection.target.kind is AttentionTargetKind.WORKING_MEMORY:
            if row_bytes > decision.rendered_bytes:
                raise ValueError("redacted Working Memory row exceeds its original witness")
        elif row_bytes != decision.rendered_bytes:
            raise ValueError("R13 visible row must preserve its exact witnessed byte count")
        rendered_row_bytes += row_bytes
        rendered_rows_by_kind[projection.target.kind].append(visible)

    if witnessed_row_bytes != selection.row_bytes:
        raise ValueError("revalidated source rows do not match the Attention row-byte witness")
    frame_text = ATTENTION_PROMPT_AUTHORITY_INTRO + "".join(
        heading for _, heading in ATTENTION_PROMPT_SECTION_HEADERS
    )
    rendered_parts = [ATTENTION_PROMPT_AUTHORITY_INTRO]
    for kind, heading in ATTENTION_PROMPT_SECTION_HEADERS:
        rendered_parts.append(heading)
        for encoded_row in rendered_rows_by_kind[kind]:
            rendered_parts.extend((encoded_row.decode("ascii"), "\n"))
    rendered_text = "".join(rendered_parts)
    rendered_bytes = len(rendered_text.encode("utf-8"))
    expected_rendered_bytes = (
        len(frame_text.encode("utf-8"))
        + rendered_row_bytes
        + len(selection.included_candidate_ids)
    )
    if rendered_bytes != expected_rendered_bytes:
        raise ValueError("rendered Attention prompt byte count differs from its rows")
    if (
        selection.total_bytes > ATTENTION_PROMPT_BUDGET_BYTES
        or rendered_bytes > selection.total_bytes
        or rendered_bytes > ATTENTION_PROMPT_BUDGET_BYTES
    ):
        raise ValueError("rendered Attention prompt exceeds its accepted source budget")
    return AttentionPromptPayload._create(
        event=event,
        selection_digest=selection.result_digest,
        included_candidate_ids=selection.included_candidate_ids,
        witnessed_bytes=selection.total_bytes,
        rendered_text=rendered_text,
    )


def build_attention_prompt_payload(
    refresh: AttentionRefreshResult,
    *,
    working_memory_items: tuple[WorkingMemoryItem, ...],
    working_memory_revision: int,
    working_memory_view: WorkingMemoryView,
    motivation_state: MotivationSystemSnapshot,
    goal_state: GoalSystemSnapshot,
    commitment_state: CommitmentSystemSnapshot,
) -> AttentionPromptPayload:
    """Revalidate and render only the already-accepted current selection.

    The arguments are the typed, same-turn source capture used by the caller;
    they are not model-facing policy inputs.  This function never selects,
    reranks, truncates, repacks, or resolves sources from another authority.
    """

    event, snapshot, competition, selection = _validated_refresh(refresh)
    if type(working_memory_view) is not WorkingMemoryView:
        raise TypeError("working_memory_view must be an exact WorkingMemoryView")
    projections = project_attention_candidates(
        working_memory_items=working_memory_items,
        working_memory_revision=working_memory_revision,
        working_memory_view=working_memory_view,
        motivation_snapshot=motivation_state,
        goal_snapshot=goal_state,
        commitment_snapshot=commitment_state,
        event=event,
    )
    projections_by_id = _validate_projection_bindings(
        projections=projections,
        snapshot=snapshot,
        competition=competition,
        selection=selection,
    )
    return _render_selected_rows(
        event=event,
        selection=selection,
        projections_by_id=projections_by_id,
        working_memory_items=working_memory_items,
        working_memory_view=working_memory_view,
        motivation_state=motivation_state,
        goal_state=goal_state,
        commitment_state=commitment_state,
    )


__all__ = ["AttentionPromptPayload", "build_attention_prompt_payload"]
