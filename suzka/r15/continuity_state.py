"""R15 U5 technical continuity only; never a semantic or Source authority.

One bounded v10 reference envelope ties the exact U3 root to the previously
qualified internal event/episode scope. Technical WAL restore is NOT permission
for a model, operator, public API, or expired evidence to assert intrinsic truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS, validate_identifier
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE
from suzka.r15.common import canonical_json
from suzka.r15.contracts import NarrativeLinkKind, NarrativeSelfState
from suzka.r15.qualified_proposal import (
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
    restore_scoped_option,
)

CHECKPOINT_FIELD = "narrative_admission_checkpoint"
MAX_CHECKPOINT_BYTES = 3263
_FIELDS = frozenset(
    (
        "schema_version",
        "stage",
        "root_digest",
        "root_revision",
        "source_scope",
        "context_id",
        "first_sources",
        "revised_sources",
        "last_review_event_id",
        "last_review_event_sequence",
    )
)
_REF_FIELDS = frozenset(("event_id", "event_sequence", "episode_id"))


class ContinuityInvalid(ValueError):
    """Non-sensitive technical restore refusal."""


def _refs(value: object, count: int) -> tuple[QualifiedTurnRef, ...]:
    if type(value) is not list or len(value) != count:
        raise ContinuityInvalid("continuation source count invalid")
    result = []
    for row in value:
        if type(row) is not dict or set(row) != _REF_FIELDS:
            raise ContinuityInvalid("continuation source fields invalid")
        try:
            event = validate_identifier(row["event_id"])
            episode = validate_identifier(row["episode_id"])
        except (ValueError, TypeError) as exc:
            raise ContinuityInvalid("continuation source reference invalid") from exc
        sequence = row["event_sequence"]
        if (
            type(sequence) is not int
            or not 1 <= sequence <= MAX_PERSISTED_EVENT_SEQUENCE
        ):
            raise ContinuityInvalid("continuation event sequence invalid")
        result.append(QualifiedTurnRef(event, sequence, episode))
    order = [item.event_sequence for item in result]
    if order != sorted(set(order)) or len({item.event_id for item in result}) != len(
        result
    ):
        raise ContinuityInvalid("continuation sources out of order")
    return tuple(result)


def _ref_value(item: QualifiedTurnRef) -> dict[str, str | int]:
    if type(item) is not QualifiedTurnRef:
        raise ContinuityInvalid("continuation ref type invalid")
    try:
        event = validate_identifier(item.event_id)
        episode = validate_identifier(item.episode_id)
    except (ValueError, TypeError) as exc:
        raise ContinuityInvalid("continuation ref identity invalid") from exc
    if (
        type(item.event_sequence) is not int
        or not 1 <= item.event_sequence <= MAX_PERSISTED_EVENT_SEQUENCE
    ):
        raise ContinuityInvalid("continuation ref sequence invalid")
    return {
        "event_id": event,
        "event_sequence": item.event_sequence,
        "episode_id": episode,
    }


def _meaning_bound(root: NarrativeSelfState, scope: str) -> None:
    if root.revision not in (1, 2) or len(root.episodes) != root.revision:
        raise ContinuityInvalid("unsupported MVI R15 revision topology")
    choices = []
    for episode in root.episodes:
        try:
            choices.append(
                restore_scoped_option(scope, episode.interpretation.statement_id)
            )
        except QualifiedProposalUnavailable as exc:
            raise ContinuityInvalid("source-bound meaning cannot be decoded") from exc
    if any(item not in ("A", "B") for item in choices):
        raise ContinuityInvalid("source-bound meaning remains UNKNOWN")
    if root.revision == 2:
        if set(choices) != {"A", "B"}:
            raise ContinuityInvalid("reinterpretation did not change referent")
        ids = {ep.episode_id for ep in root.episodes}
        if not any(
            any(
                link.kind is NarrativeLinkKind.EPISODE
                and link.target_id in ids - {episode.episode_id}
                for link in episode.links
            )
            for episode in root.episodes
        ):
            raise ContinuityInvalid("old interpretation is not linked")


@dataclass(frozen=True, slots=True)
class NarrativeContinuation:
    """Detached exact checkpoint; only StateWAL/AgentState may restore it."""

    root_digest: str
    revision: int
    source_scope: str | None
    context_id: str | None
    first_sources: tuple[QualifiedTurnRef, ...]
    revised_sources: tuple[QualifiedTurnRef, ...]
    last_review_event_id: str | None
    last_review_event_sequence: int | None

    @classmethod
    def empty(cls, root: NarrativeSelfState) -> NarrativeContinuation:
        if (
            type(root) is not NarrativeSelfState
            or root.revision != 0
            or root.episodes
            or root.claims
        ):
            raise ContinuityInvalid("empty continuation needs an empty U3 root")
        return cls(
            str(root.canonical_value()["state_digest"]),
            0,
            None,
            None,
            (),
            (),
            None,
            None,
        )

    def _validate_retained_source_witnesses(self, root: NarrativeSelfState) -> None:
        """Bind the actual persisted E/E2 witness events to replay refs.

        U1 currently retains these witnessed events; setup/S1/S2 episode
        references beyond those witnesses remain technical replay hints and
        MUST be requalified against the live source owners before use.
        """
        first = tuple(ep for ep in root.episodes if not ep.links)
        if len(first) != 1:
            raise ContinuityInvalid("original interpretation source is ambiguous")
        original = first[0]
        initial_ref = self.first_sources[1]
        original_event = original.experience.event
        if original_event is None:
            raise ContinuityInvalid("original Experience event is absent")
        if (
            initial_ref.event_id != original_event.event_id
            or initial_ref.event_sequence != original_event.event_sequence
        ):
            raise ContinuityInvalid("original E reference differs from U3 witness")

        if self.revision == 2:
            revised = tuple(ep for ep in root.episodes if ep.links)
            if len(revised) != 1:
                raise ContinuityInvalid("revised interpretation source is ambiguous")
            later = revised[0]
            subsequent_ref = self.revised_sources[3]
            subsequent_event = later.experience.event
            if subsequent_event is None:
                raise ContinuityInvalid("revised Experience event is absent")
            if (
                subsequent_ref.event_id != subsequent_event.event_id
                or subsequent_ref.event_sequence != subsequent_event.event_sequence
            ):
                raise ContinuityInvalid("reported E2 differs from U3 witness")
            if not any(
                item.event is not None
                and item.event.event_id == original_event.event_id
                and item.event.event_sequence == original_event.event_sequence
                for item in later.interpretation.counter
            ):
                raise ContinuityInvalid("revised counterevidence lost original E")

    def canonical_value(self, root: NarrativeSelfState) -> dict[str, object]:
        if type(root) is not NarrativeSelfState or root.revision != self.revision:
            raise ContinuityInvalid("continuation revision/root mismatch")
        if self.root_digest != root.canonical_value()["state_digest"]:
            raise ContinuityInvalid("continuation digest/root mismatch")
        if self.revision == 0:
            if (
                root.episodes
                or root.claims
                or root.revision_history
                or self.source_scope is not None
                or self.context_id is not None
                or self.first_sources
                or self.revised_sources
                or self.last_review_event_id is not None
                or self.last_review_event_sequence is not None
            ):
                raise ContinuityInvalid("empty continuation contains fabricated source")
            stage = "EMPTY"
        elif self.revision in (1, 2):
            if (
                type(self.source_scope) is not str
                or type(self.context_id) is not str
                or len(root.revision_history) != self.revision
            ):
                raise ContinuityInvalid("active continuation missing source scope")
            try:
                scope = validate_identifier(self.source_scope)
                validate_identifier(self.context_id)
            except (TypeError, ValueError) as exc:
                raise ContinuityInvalid(
                    "source/context scope identifier invalid"
                ) from exc
            _meaning_bound(root, scope)
            first = [_ref_value(ref) for ref in self.first_sources]
            revised = [_ref_value(ref) for ref in self.revised_sources]
            _refs(first, 3)
            _refs(revised, 5 if self.revision == 2 else 0)
            if self.first_sources[0].episode_id != scope:
                raise ContinuityInvalid("option meaning scope changed")
            if self.revision == 2 and self.revised_sources[:3] != self.first_sources:
                raise ContinuityInvalid("reinterpreted source lost S1 evidence")
            self._validate_retained_source_witnesses(root)
            last = root.revision_history[-1].event
            if (
                self.last_review_event_id != last.event_id
                or self.last_review_event_sequence != last.event_sequence
                or last.event_sequence <= self.first_sources[-1].event_sequence
                or (
                    revised
                    and last.event_sequence <= self.revised_sources[-1].event_sequence
                )
            ):
                raise ContinuityInvalid(
                    "revision proof event is not the committed review"
                )
            stage = "S1" if self.revision == 1 else "S2"
        else:
            raise ContinuityInvalid("MVI continuation revision unsupported")
        value: dict[str, object] = {
            "schema_version": 1,
            "stage": stage,
            "root_digest": self.root_digest,
            "root_revision": self.revision,
            "source_scope": self.source_scope,
            "context_id": self.context_id,
            "first_sources": [_ref_value(ref) for ref in self.first_sources],
            "revised_sources": [_ref_value(ref) for ref in self.revised_sources],
            "last_review_event_id": self.last_review_event_id,
            "last_review_event_sequence": self.last_review_event_sequence,
        }
        if len(canonical_json(value)) > MAX_CHECKPOINT_BYTES:
            raise ContinuityInvalid("private continuation exceeds legal bound")
        return value

    @classmethod
    def from_value(
        cls, value: object, root: NarrativeSelfState
    ) -> NarrativeContinuation:
        if type(value) is not dict or set(value) != _FIELDS:
            raise ContinuityInvalid("continuation is not a closed technical record")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ContinuityInvalid("continuation schema unknown")
        revision = value["root_revision"]
        if type(revision) is not int or revision not in (0, 1, 2):
            raise ContinuityInvalid("continuation revision invalid")
        first = _refs(value["first_sources"], 3 if revision else 0)
        later = _refs(value["revised_sources"], 5 if revision == 2 else 0)
        if type(value["root_digest"]) is not str:
            raise ContinuityInvalid("continuation digest type invalid")
        technical = cls(
            value["root_digest"],
            revision,
            value["source_scope"],
            value["context_id"],
            first,
            later,
            value["last_review_event_id"],
            value["last_review_event_sequence"],
        )
        if technical.canonical_value(root) != value:
            raise ContinuityInvalid("continuation differs from the canonical root")
        if value["stage"] != (
            "EMPTY" if revision == 0 else "S1" if revision == 1 else "S2"
        ):
            raise ContinuityInvalid("continuation stage does not match root")
        return technical


def checkpoint_max_bytes() -> int:
    """Legal worst-case private technical value, reserved independently of U1."""
    ident = "z" * MAX_IDENTIFIER_CODEPOINTS
    seq = MAX_PERSISTED_EVENT_SEQUENCE
    event = {"event_id": ident, "event_sequence": seq, "episode_id": ident}
    return len(
        canonical_json(
            {
                "schema_version": 1,
                "stage": "S2",
                "root_digest": "f" * 64,
                "root_revision": 2,
                "source_scope": ident,
                "context_id": ident,
                "first_sources": [event] * 3,
                "revised_sources": [event] * 5,
                "last_review_event_id": ident,
                "last_review_event_sequence": seq,
            }
        )
    )
