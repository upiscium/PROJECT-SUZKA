"""Process-local R14 Attention state transitions.

The system owns only its immutable Attention continuity and the current
selection view.  Callers supply the complete, coherent current projection
tuple; this boundary validates its shape and evidence, but cannot independently
prove that an upstream producer omitted no current source.  It has no source
lookup, clock, model, runtime, persistence, or Goal/Commitment authority.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from threading import RLock

from suzka.attention.adapters import GlobalEmotionProjection
from suzka.attention.bounds import (
    attention_candidate_capacity,
    attention_candidate_capacity_by_kind,
)
from suzka.attention.common import (
    ATTENTION_MAX_EVENT_RECEIPTS,
    ATTENTION_MAX_FOCUS,
    ATTENTION_MAX_REVISION,
    ATTENTION_MAX_REVISION_HISTORY,
    ATTENTION_POLICY_VERSION,
    AttentionRevisionReason,
    AttentionTargetKind,
    digest_payload,
    validate_digest,
)
from suzka.attention.contracts import (
    AttentionCandidateContinuity,
    AttentionCandidateProjection,
    AttentionContinuity,
    AttentionEvent,
    AttentionEventReceipt,
    AttentionReceiptAnchor,
    AttentionRevisionAnchor,
    AttentionRevisionEvidence,
    AttentionTarget,
    attention_state_digest,
)
from suzka.attention.policy import (
    AttentionCompetitionResult,
    AttentionPromptSelection,
    compete_attention,
    next_habituation_units,
    next_inhibition_units,
    next_streak_counts,
    select_attention_prompt,
)


_REFRESH_INPUT_DOMAIN = b"PROJECT-SUZKA:R14:ATTENTION-REFRESH-INPUT:V1\0"


class AttentionSystemError(ValueError):
    """A process-local Attention transition could not be accepted."""


class AttentionReplayConflict(AttentionSystemError):
    """An event identity was repeated without its exact retained input proof."""


class AttentionSourceConflict(AttentionSystemError):
    """A current primary source witness regressed or contradicted its revision."""


@dataclass(frozen=True, slots=True)
class AttentionSelectedView:
    """Current detached target references and ephemeral selection evidence."""

    revision: int
    event: AttentionEvent | None
    state_digest: str
    authority_digest: str
    focused_targets: tuple[AttentionTarget, ...]
    unfinished_targets: tuple[AttentionTarget, ...]
    competition: AttentionCompetitionResult | None
    prompt: AttentionPromptSelection | None

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("view revision must be a non-negative exact integer")
        if self.event is not None and type(self.event) is not AttentionEvent:
            raise TypeError("view event must be an exact AttentionEvent or None")
        validate_digest(self.state_digest, "view state_digest")
        validate_digest(self.authority_digest, "view authority_digest")
        if type(self.focused_targets) is not tuple or type(self.unfinished_targets) is not tuple:
            raise TypeError("view target references must be exact tuples")
        for name, targets in (
            ("focused_targets", self.focused_targets),
            ("unfinished_targets", self.unfinished_targets),
        ):
            if len(targets) > ATTENTION_MAX_FOCUS or any(
                type(target) is not AttentionTarget for target in targets
            ):
                raise ValueError(f"{name} must contain bounded exact AttentionTarget values")
            ids = tuple(target.candidate_id for target in targets)
            if ids != tuple(sorted(set(ids))):
                raise ValueError(f"{name} must use canonical candidate-ID order")
        if (self.competition is None) != (self.prompt is None):
            raise ValueError("competition and prompt evidence must be present or absent together")
        if self.competition is not None:
            if type(self.competition) is not AttentionCompetitionResult:
                raise TypeError("competition must be an exact AttentionCompetitionResult")
            if self.event is None or self.competition.event != self.event:
                raise ValueError("view competition must match its current event")
            if type(self.prompt) is not AttentionPromptSelection:
                raise TypeError("prompt must be an exact AttentionPromptSelection")
            if self.prompt.competition_digest != self.competition.result_digest:
                raise ValueError("view prompt must bind its competition result")


@dataclass(frozen=True, slots=True)
class AttentionRefreshResult:
    """Detached result of one fresh transition or an exact retained replay."""

    snapshot: AttentionContinuity
    receipt: AttentionEventReceipt
    replayed: bool
    view: AttentionSelectedView

    def __post_init__(self) -> None:
        if type(self.snapshot) is not AttentionContinuity:
            raise TypeError("snapshot must be an exact AttentionContinuity")
        if type(self.receipt) is not AttentionEventReceipt:
            raise TypeError("receipt must be an exact AttentionEventReceipt")
        if type(self.replayed) is not bool:
            raise TypeError("replayed must be an exact bool")
        if type(self.view) is not AttentionSelectedView:
            raise TypeError("view must be an exact AttentionSelectedView")


@dataclass(frozen=True, slots=True)
class _Bundle:
    """One immutable publication unit for the snapshot and its current view."""

    snapshot: AttentionContinuity
    view: AttentionSelectedView


def _copy_event(value: object) -> AttentionEvent:
    if type(value) is not AttentionEvent:
        raise TypeError("event must be an exact AttentionEvent")
    return AttentionEvent(value.event_id, value.event_sequence, value.occurred_at)


def _copy_continuity(value: object) -> AttentionContinuity:
    if type(value) is not AttentionContinuity:
        raise TypeError("snapshot must be an exact AttentionContinuity")
    # The closed canonical decoder compares declared hashes rather than
    # recomputing and silently repairing potentially modified source objects.
    return AttentionContinuity.from_canonical_value(value.canonical_value())


def _copy_receipt(value: object) -> AttentionEventReceipt:
    if type(value) is not AttentionEventReceipt:
        raise TypeError("receipt must be an exact AttentionEventReceipt")
    result = AttentionEventReceipt(
        event=_copy_event(value.event),
        input_digest=value.input_digest,
        result_state_digest=value.result_state_digest,
        previous_receipt_digest=value.previous_receipt_digest,
    )
    if result.receipt_digest != value.receipt_digest:
        raise ValueError("receipt digest does not match its declared contents")
    return result


def _copy_competition(
    value: AttentionCompetitionResult | None,
) -> AttentionCompetitionResult | None:
    if value is None:
        return None
    if type(value) is not AttentionCompetitionResult:
        raise TypeError("competition must be an exact AttentionCompetitionResult or None")
    expected = value.canonical_value()
    result = copy.deepcopy(value)
    if result.canonical_value() != expected:
        raise ValueError("competition changed while making a detached copy")
    return result


def _copy_prompt(
    value: AttentionPromptSelection | None,
) -> AttentionPromptSelection | None:
    if value is None:
        return None
    if type(value) is not AttentionPromptSelection:
        raise TypeError("prompt must be an exact AttentionPromptSelection or None")
    expected = value.canonical_value()
    result = copy.deepcopy(value)
    if result.canonical_value() != expected:
        raise ValueError("prompt changed while making a detached copy")
    return result


def _copy_view(value: object) -> AttentionSelectedView:
    if type(value) is not AttentionSelectedView:
        raise TypeError("view must be an exact AttentionSelectedView")
    return AttentionSelectedView(
        revision=value.revision,
        event=None if value.event is None else _copy_event(value.event),
        state_digest=value.state_digest,
        authority_digest=value.authority_digest,
        focused_targets=tuple(
            AttentionTarget(target.kind, target.reference)
            for target in value.focused_targets
        ),
        unfinished_targets=tuple(
            AttentionTarget(target.kind, target.reference)
            for target in value.unfinished_targets
        ),
        competition=_copy_competition(value.competition),
        prompt=_copy_prompt(value.prompt),
    )


def _view_for(
    snapshot: AttentionContinuity,
    *,
    competition: AttentionCompetitionResult | None = None,
    prompt: AttentionPromptSelection | None = None,
) -> AttentionSelectedView:
    candidates = {item.candidate_id: item.target for item in snapshot.candidates}
    return AttentionSelectedView(
        revision=snapshot.revision,
        event=None if snapshot.last_event is None else _copy_event(snapshot.last_event),
        state_digest=snapshot.state_digest,
        authority_digest=snapshot.authority_digest,
        focused_targets=tuple(
            AttentionTarget(candidates[candidate_id].kind, candidates[candidate_id].reference)
            for candidate_id in snapshot.focused_ids
        ),
        unfinished_targets=tuple(
            AttentionTarget(candidates[candidate_id].kind, candidates[candidate_id].reference)
            for candidate_id in snapshot.unfinished_ids
        ),
        competition=competition,
        prompt=prompt,
    )


def _copy_inputs(
    projections: tuple[AttentionCandidateProjection, ...],
    event: AttentionEvent,
    global_emotion: GlobalEmotionProjection | None,
) -> tuple[
    tuple[AttentionCandidateProjection, ...],
    AttentionEvent,
    GlobalEmotionProjection | None,
    str,
]:
    current_event = _copy_event(event)
    if type(projections) is not tuple:
        raise TypeError("projections must be an exact tuple of current source projections")
    maximum = attention_candidate_capacity()
    if len(projections) > maximum:
        raise ValueError("source projection count exceeds the full candidate bound")
    per_kind_limits = dict(attention_candidate_capacity_by_kind())
    counts = {kind: 0 for kind in AttentionTargetKind}
    by_id: dict[str, AttentionCandidateProjection] = {}
    for supplied in projections:
        if type(supplied) is not AttentionCandidateProjection:
            raise TypeError("projections must contain exact sealed AttentionCandidateProjection values")
        projection = supplied.validated_copy()
        if projection.event != current_event:
            raise ValueError("all projection events must exactly match the supplied event")
        if projection.candidate_id in by_id:
            raise ValueError("source projections must have unique canonical candidate IDs")
        by_id[projection.candidate_id] = projection
        counts[projection.target.kind] += 1
    for kind, count in counts.items():
        if count > per_kind_limits[kind.value]:
            raise ValueError(f"{kind.value} projection count exceeds its source authority bound")
    ordered = tuple(by_id[candidate_id] for candidate_id in sorted(by_id))

    checked_emotion: GlobalEmotionProjection | None
    emotion_value: dict[str, object] | None
    if global_emotion is None:
        checked_emotion = None
        emotion_value = None
    else:
        if type(global_emotion) is not GlobalEmotionProjection:
            raise TypeError("global_emotion must be an exact GlobalEmotionProjection or None")
        checked_emotion = global_emotion.validated_copy()
        if checked_emotion.event != current_event:
            raise ValueError("global emotion projection event must exactly match the Attention event")
        emotion_value = {
            "arousal": checked_emotion.arousal.hex(),
            "event": checked_emotion.event.canonical_value(),
            "projection_digest": checked_emotion.projection_digest,
        }

    request_digest = digest_payload(
        _REFRESH_INPUT_DOMAIN,
        {
            "operation": "refresh",
            "policy_version": ATTENTION_POLICY_VERSION,
            "event": current_event.canonical_value(),
            "projections": [projection.canonical_value() for projection in ordered],
            "global_emotion": emotion_value,
        },
    )
    return ordered, current_event, checked_emotion, request_digest


def _event_evidence(snapshot: AttentionContinuity) -> tuple[AttentionEvent, ...]:
    events: list[AttentionEvent] = []
    if snapshot.last_event is not None:
        events.append(snapshot.last_event)
    events.extend(item.event for item in snapshot.revision_history)
    events.extend(item.event for item in snapshot.receipts)
    if snapshot.revision_anchor is not None:
        events.append(snapshot.revision_anchor.through_event)
    if snapshot.receipt_anchor is not None:
        events.append(snapshot.receipt_anchor.through_event)
    unique: dict[tuple[str, int, datetime], AttentionEvent] = {
        (item.event_id, item.event_sequence, item.occurred_at): item for item in events
    }
    return tuple(unique.values())


def _retained_retry(
    snapshot: AttentionContinuity,
    event: AttentionEvent,
    request_digest: str,
) -> AttentionEventReceipt | None:
    # Receipts are the only retained evidence that proves the original input.
    # Test them before freshness checks so an older retained event can be
    # replayed even if its source projections are no longer current.
    for receipt in snapshot.receipts:
        if (
            receipt.event.event_id == event.event_id
            or receipt.event.event_sequence == event.event_sequence
        ):
            if receipt.event != event:
                raise AttentionReplayConflict(
                    "retained Attention event ID or sequence was reused with different UTC evidence"
                )
            if receipt.input_digest != request_digest:
                raise AttentionReplayConflict(
                    "retained Attention event was retried with different refresh inputs"
                )
            return receipt

    for retained in _event_evidence(snapshot):
        if retained.event_id == event.event_id or retained.event_sequence == event.event_sequence:
            if retained != event:
                raise AttentionReplayConflict(
                    "retained Attention event ID or sequence conflicts with the requested event"
                )
            raise AttentionReplayConflict(
                "retained Attention event has no retained receipt proving its original inputs"
            )
    if snapshot.last_event is not None:
        if event.event_sequence <= snapshot.last_event.event_sequence:
            raise AttentionReplayConflict("unknown Attention event is not newer than current continuity")
        if event.occurred_at < snapshot.last_event.occurred_at:
            raise AttentionReplayConflict("Attention event time is older than current continuity")
    if snapshot.revision_anchor is not None and (
        event.event_sequence <= snapshot.revision_anchor.through_event.event_sequence
    ):
        raise AttentionReplayConflict("unknown Attention event is behind the revision-history anchor")
    if snapshot.receipt_anchor is not None and (
        event.event_sequence <= snapshot.receipt_anchor.through_event.event_sequence
    ):
        raise AttentionReplayConflict("unknown Attention event is behind the receipt anchor")
    return None


def _validate_source_continuity(
    snapshot: AttentionContinuity,
    projections: tuple[AttentionCandidateProjection, ...],
) -> dict[str, AttentionCandidateContinuity]:
    previous_by_id = {item.candidate_id: item for item in snapshot.candidates}
    for projection in projections:
        previous = previous_by_id.get(projection.candidate_id)
        if previous is None:
            continue
        old_source = previous.source
        new_source = projection.source
        if previous.target != projection.target or (
            old_source.kind is not new_source.kind
            or old_source.reference != new_source.reference
            or old_source.digest_kind is not new_source.digest_kind
        ):
            raise AttentionSourceConflict("candidate primary source identity changed")
        if new_source.revision < old_source.revision:
            raise AttentionSourceConflict("candidate primary source revision regressed")
        if new_source.revision == old_source.revision and (
            new_source.digest != old_source.digest
            or new_source.source_event_id != old_source.source_event_id
            or new_source.source_event_sequence != old_source.source_event_sequence
            or new_source.source_occurred_at != old_source.source_occurred_at
        ):
            raise AttentionSourceConflict(
                "candidate primary source digest or event changed without a source revision"
            )
    return previous_by_id


def _next_snapshot(
    prior: AttentionContinuity,
    projections: tuple[AttentionCandidateProjection, ...],
    event: AttentionEvent,
    request_digest: str,
    competition: AttentionCompetitionResult,
) -> tuple[AttentionContinuity, AttentionEventReceipt]:
    next_revision = prior.revision + 1
    focused_ids = competition.focused_ids
    unfinished_ids = competition.unfinished_ids
    focused_set = set(focused_ids)
    previous_by_id = {item.candidate_id: item for item in prior.candidates}
    candidates: list[AttentionCandidateContinuity] = []
    for projection in projections:
        previous = previous_by_id.get(projection.candidate_id)
        habituation = 0 if previous is None else previous.habituation
        inhibition = 0 if previous is None else previous.inhibition
        focused_events = 0 if previous is None else previous.focused_event_count
        unattended_events = 0 if previous is None else previous.unattended_event_count
        focused = projection.candidate_id in focused_set
        next_focused, next_unattended = next_streak_counts(
            focused_events,
            unattended_events,
            focused=focused,
        )
        candidates.append(
            AttentionCandidateContinuity(
                target=projection.target,
                source=projection.source,
                availability=projection.availability,
                habituation=next_habituation_units(habituation, focused=focused),
                inhibition=next_inhibition_units(inhibition),
                focused_event_count=next_focused,
                unattended_event_count=next_unattended,
            )
        )
    current_candidates = tuple(candidates)
    current_state_digest = attention_state_digest(
        schema_version=prior.schema_version,
        policy_version=prior.policy_version,
        revision=next_revision,
        last_event=event,
        candidates=current_candidates,
        focused_ids=focused_ids,
        unfinished_ids=unfinished_ids,
    )

    prior_revision_digest = (
        None if prior.revision == 0 else prior.revision_history[-1].record_digest
    )
    revision_evidence = AttentionRevisionEvidence(
        revision=next_revision,
        event=event,
        previous_state_digest=prior.state_digest,
        state_digest=current_state_digest,
        previous_revision_digest=prior_revision_digest,
        focused_ids=focused_ids,
        unfinished_ids=unfinished_ids,
        reason=AttentionRevisionReason.STATE_UPDATE,
    )
    revision_history = prior.revision_history + (revision_evidence,)
    revision_anchor = prior.revision_anchor
    if len(revision_history) > ATTENTION_MAX_REVISION_HISTORY:
        dropped_revision = revision_history[0]
        revision_history = revision_history[1:]
        revision_anchor = AttentionRevisionAnchor(
            through_revision=dropped_revision.revision,
            through_event=dropped_revision.event,
            through_state_digest=dropped_revision.state_digest,
            through_revision_digest=dropped_revision.record_digest,
        )

    previous_receipt_digest = (
        None if not prior.receipts else prior.receipts[-1].receipt_digest
    )
    receipt = AttentionEventReceipt(
        event=event,
        input_digest=request_digest,
        result_state_digest=current_state_digest,
        previous_receipt_digest=previous_receipt_digest,
    )
    receipts = prior.receipts + (receipt,)
    receipt_anchor = prior.receipt_anchor
    if len(receipts) > ATTENTION_MAX_EVENT_RECEIPTS:
        dropped_receipt = receipts[0]
        receipts = receipts[1:]
        receipt_anchor = AttentionReceiptAnchor(
            through_event=dropped_receipt.event,
            through_result_state_digest=dropped_receipt.result_state_digest,
            through_receipt_digest=dropped_receipt.receipt_digest,
        )

    provisional = AttentionContinuity(
        schema_version=prior.schema_version,
        policy_version=prior.policy_version,
        revision=next_revision,
        last_event=event,
        candidates=current_candidates,
        focused_ids=focused_ids,
        unfinished_ids=unfinished_ids,
        revision_history=revision_history,
        receipts=receipts,
        revision_anchor=revision_anchor,
        receipt_anchor=receipt_anchor,
    )
    current = _copy_continuity(provisional)
    from suzka.attention.bounds import derive_attention_schema_size_budget

    maximum_bytes = derive_attention_schema_size_budget().attention_state_max_bytes
    if len(current.canonical_bytes()) > maximum_bytes:
        raise ValueError("current Attention continuity exceeds its derived schema byte bound")
    return current, receipt


class AttentionSystem:
    """Thread-safe process-local owner of Attention continuity and selections.

    Input tuples are treated as a producer's complete coherent capture; this
    system does not prove upstream membership. Event-ID reuse is fenced by the
    bounded retained receipt/history suffixes and their anchors, not by an
    infinite event-ID ledger after old evidence has been compacted.
    """

    def __init__(self, snapshot: AttentionContinuity | None = None) -> None:
        self._lock = RLock()
        with self._lock:
            current = AttentionContinuity.bootstrap() if snapshot is None else _copy_continuity(snapshot)
            self._bundle = _Bundle(current, _view_for(current))

    def snapshot(self) -> AttentionContinuity:
        """Return a validated, detached copy of the current continuity."""

        with self._lock:
            return _copy_continuity(self._bundle.snapshot)

    def selected_view(self) -> AttentionSelectedView:
        """Return the current immutable target-reference view, detached."""

        with self._lock:
            return _copy_view(self._bundle.view)

    def restore_snapshot(self, snapshot: AttentionContinuity) -> None:
        """Replace local state exactly; do not replay policy or resolve sources."""

        with self._lock:
            restored = _copy_continuity(snapshot)
            restored_bundle = _Bundle(restored, _view_for(restored))
            self._bundle = restored_bundle

    def refresh(
        self,
        projections: tuple[AttentionCandidateProjection, ...],
        event: AttentionEvent,
        *,
        global_emotion: GlobalEmotionProjection | None = None,
    ) -> AttentionRefreshResult:
        """Compete a supplied full current tuple and atomically publish one revision."""

        with self._lock:
            ordered, current_event, checked_emotion, request_digest = _copy_inputs(
                projections,
                event,
                global_emotion,
            )
            prior_bundle = self._bundle
            prior = prior_bundle.snapshot

            retained_receipt = _retained_retry(prior, current_event, request_digest)
            if retained_receipt is not None:
                return AttentionRefreshResult(
                    snapshot=_copy_continuity(prior),
                    receipt=_copy_receipt(retained_receipt),
                    replayed=True,
                    view=_copy_view(prior_bundle.view),
                )

            _validate_source_continuity(prior, ordered)
            if prior.revision >= ATTENTION_MAX_REVISION:
                raise ValueError("Attention revision would exceed its persisted bound")

            competition = compete_attention(
                ordered,
                prior,
                current_event,
                checked_emotion,
            )
            prompt = select_attention_prompt(competition, ordered)
            current, receipt = _next_snapshot(
                prior,
                ordered,
                current_event,
                request_digest,
                competition,
            )
            current_view = _view_for(
                current,
                competition=competition,
                prompt=prompt,
            )
            new_bundle = _Bundle(current, current_view)

            # Prepare every detached return value before the single publication
            # assignment.  No validation, copying, digesting, or other fallible
            # work follows that swap.
            result = AttentionRefreshResult(
                snapshot=_copy_continuity(current),
                receipt=_copy_receipt(receipt),
                replayed=False,
                view=_copy_view(current_view),
            )
            self._bundle = new_bundle
            return result


__all__ = [
    "AttentionRefreshResult",
    "AttentionReplayConflict",
    "AttentionSelectedView",
    "AttentionSourceConflict",
    "AttentionSystem",
    "AttentionSystemError",
]
