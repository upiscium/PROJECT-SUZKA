"""TEST-ONLY v10 continuation capacity and restore-envelope proof.

NOT a product AgentStateSnapshotV10 codec, a valid U5 StateWAL recovery
transaction, an authorized source issuer, or an intrinsic-truth setter.
This preflight only derives the minimum private metadata *in addition* to
the full legal R15 U1 roots. No production owner is modified.
"""

from __future__ import annotations

import json
from typing import Any

from suzka.identifiers import MAX_IDENTIFIER_CODEPOINTS, validate_identifier
from suzka.limits import MAX_PERSISTED_EVENT_SEQUENCE
from suzka.r15.bounds import (
    AGENT_STATE_HARD_CAP_BYTES,
    FULL_FUTURE_RESERVE_BYTES,
    derive_r15_schema_budget,
)
from suzka.r15.common import canonical_json
from suzka.r15.contracts import NarrativeLinkKind, NarrativeSelfState
from suzka.r15.qualified_proposal import (
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
    restore_scoped_option,
)
from suzka.runtime.agent_state import project_agent_state_schema_max_bytes


FIELD = "narrative_admission_checkpoint"
FIELDS = {
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
}
SOURCE_FIELDS = {"event_id", "event_sequence", "episode_id"}


class PreflightRefused(ValueError):
    """A state-format mismatch, without private source or prompt content."""


def _ref_data(ref: QualifiedTurnRef) -> dict[str, str | int]:
    if type(ref) is not QualifiedTurnRef:
        raise PreflightRefused("invalid source ref")
    return {
        "event_id": validate_identifier(ref.event_id),
        "event_sequence": ref.event_sequence,
        "episode_id": validate_identifier(ref.episode_id),
    }


def _validate_sequence(items: object, *, count: int) -> tuple[QualifiedTurnRef, ...]:
    if type(items) is not list or len(items) != count:
        raise PreflightRefused("source replay window mismatch")
    result: list[QualifiedTurnRef] = []
    for value in items:
        if type(value) is not dict or set(value) != SOURCE_FIELDS:
            raise PreflightRefused("source ref fields mismatch")
        try:
            eid = validate_identifier(value["event_id"])
            episode = validate_identifier(value["episode_id"])
        except (ValueError, TypeError) as error:
            raise PreflightRefused("source ref identifier invalid") from error
        n = value["event_sequence"]
        if type(n) is not int or not 1 <= n <= MAX_PERSISTED_EVENT_SEQUENCE:
            raise PreflightRefused("source ref sequence invalid")
        result.append(QualifiedTurnRef(eid, n, episode))
    seq = [ref.event_sequence for ref in result]
    if seq != sorted(set(seq)) or len({ref.event_id for ref in result}) != len(result):
        raise PreflightRefused("source chronology is duplicated or reversed")
    return tuple(result)


def _source_scope(root: NarrativeSelfState, scope: str) -> None:
    """Only a syntactic stage check: owner qualification remains external."""
    if type(root) is not NarrativeSelfState:
        raise PreflightRefused("narrative root type unavailable")
    try:
        if root.revision == 1:
            if len(root.episodes) != 1:
                raise PreflightRefused("initial narrative episode count")
            restored = restore_scoped_option(
                scope, root.episodes[0].interpretation.statement_id
            )
            if restored not in ("A", "B"):
                raise PreflightRefused("initial meaning is not qualified")
        elif root.revision == 2:
            if len(root.episodes) != 2:
                raise PreflightRefused("revised narrative episode count")
            ids = {ep.episode_id for ep in root.episodes}
            for ep in root.episodes:
                restored = restore_scoped_option(scope, ep.interpretation.statement_id)
                if restored not in ("A", "B"):
                    raise PreflightRefused("revised meaning is not qualified")
            if {
                restore_scoped_option(scope, ep.interpretation.statement_id)
                for ep in root.episodes
            } != {"A", "B"}:
                raise PreflightRefused("revision did not change referent")
            if not any(
                any(
                    link.kind is NarrativeLinkKind.EPISODE
                    and link.target_id in (ids - {ep.episode_id})
                    for link in ep.links
                )
                for ep in root.episodes
            ):
                raise PreflightRefused("revised narrative history is disconnected")
        else:
            raise PreflightRefused("MVI source checkpoint only supports revisions 1-2")
    except QualifiedProposalUnavailable as error:
        raise PreflightRefused("semantic scope cannot be decoded") from error


def _decode(raw: bytes) -> dict[str, Any]:
    if len(raw) > checkpoint_full_legal_max_bytes():
        raise PreflightRefused("private checkpoint exceeds its bound")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in pairs:
            if key in row:
                raise PreflightRefused("duplicate private checkpoint field")
            row[key] = value
        return row

    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise PreflightRefused("private checkpoint bytes malformed") from error
    if type(data) is not dict or set(data) != FIELDS:
        raise PreflightRefused("unknown private checkpoint fields")
    if canonical_json(data) != raw:
        raise PreflightRefused("private checkpoint must be canonical")
    return data


def continuity_envelope(
    root: NarrativeSelfState,
    *,
    source_scope: str | None,
    context_id: str | None,
    first_sources: tuple[QualifiedTurnRef, ...] = (),
    revised_sources: tuple[QualifiedTurnRef, ...] = (),
) -> bytes:
    """Bounded technical snapshot proposal, NEVER an adoption API."""
    if type(root) is not NarrativeSelfState:
        raise PreflightRefused("root type unavailable")
    root_digest = str(root.canonical_value()["state_digest"])
    if root.revision == 0:
        if (
            root.episodes
            or root.claims
            or root.revision_history
            or source_scope is not None
            or context_id is not None
            or first_sources
            or revised_sources
        ):
            raise PreflightRefused("empty snapshot cannot assert narrative data")
        value = {
            "schema_version": 1,
            "stage": "EMPTY",
            "root_digest": root_digest,
            "root_revision": 0,
            "source_scope": None,
            "context_id": None,
            "first_sources": [],
            "revised_sources": [],
            "last_review_event_id": None,
            "last_review_event_sequence": None,
        }
    else:
        if root.revision not in (1, 2) or len(root.revision_history) != root.revision:
            raise PreflightRefused("unsupported MVI revision history")
        if type(source_scope) is not str or type(context_id) is not str:
            raise PreflightRefused("qualified option scope absent")
        scope = validate_identifier(source_scope)
        context = validate_identifier(context_id)
        _source_scope(root, scope)
        if (
            type(first_sources) is not tuple
            or len(first_sources) != 3
            or type(revised_sources) is not tuple
            or len(revised_sources) != (5 if root.revision == 2 else 0)
        ):
            raise PreflightRefused("source stages disagree with root revision")
        initial = [_ref_data(ref) for ref in first_sources]
        revised = [_ref_data(ref) for ref in revised_sources]
        _validate_sequence(initial, count=3)
        if first_sources[0].episode_id != scope:
            raise PreflightRefused("source scope does not match first episode")
        if root.revision == 2:
            _validate_sequence(revised, count=5)
            if revised_sources[:3] != first_sources:
                raise PreflightRefused("revised source must retain original E/S1")
        last_review = root.revision_history[-1].event
        value = {
            "schema_version": 1,
            "stage": "S1" if root.revision == 1 else "S2",
            "root_digest": root_digest,
            "root_revision": root.revision,
            "source_scope": scope,
            "context_id": context,
            "first_sources": initial,
            "revised_sources": revised,
            "last_review_event_id": validate_identifier(last_review.event_id),
            "last_review_event_sequence": last_review.event_sequence,
        }
        if last_review.event_sequence <= first_sources[-1].event_sequence or (
            revised_sources
            and last_review.event_sequence <= revised_sources[-1].event_sequence
        ):
            raise PreflightRefused("review event did not follow its source")
    encoded = canonical_json(value)
    if len(encoded) > checkpoint_full_legal_max_bytes():
        raise PreflightRefused("checkpoint capacity exceeded")
    return encoded


def validate_continuity_envelope(
    encoded: bytes,
    root: NarrativeSelfState,
) -> dict[str, Any]:
    """Read back only. This never installs an authoritative NarrativeSelf."""
    data = _decode(encoded)
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise PreflightRefused("unsupported technical checkpoint version")
    revision = data["root_revision"]
    if type(revision) is not int or revision not in (0, 1, 2):
        raise PreflightRefused("technical root revision out of bounds")
    if (
        type(root) is not NarrativeSelfState
        or revision != root.revision
        or data["root_digest"] != root.canonical_value()["state_digest"]
    ):
        raise PreflightRefused("technical checkpoint does not match exact U1 root")
    if revision == 0:
        empty = continuity_envelope(root, source_scope=None, context_id=None)
        if encoded != empty:
            raise PreflightRefused("empty continuation contains invented source")
    else:
        sources = _validate_sequence(data["first_sources"], count=3)
        revised = _validate_sequence(
            data["revised_sources"], count=5 if revision == 2 else 0
        )
        expected = continuity_envelope(
            root,
            source_scope=data["source_scope"],
            context_id=data["context_id"],
            first_sources=sources,
            revised_sources=revised,
        )
        if encoded != expected:
            raise PreflightRefused("checkpoint history is not canonical")
        if data["stage"] != ("S1" if revision == 1 else "S2"):
            raise PreflightRefused("checkpoint stage does not match revision")
        last = root.revision_history[-1].event
        if (
            data["last_review_event_id"] != last.event_id
            or data["last_review_event_sequence"] != last.event_sequence
        ):
            raise PreflightRefused("checkpoint last event is not the root review")
    return data


def checkpoint_full_legal_max_bytes() -> int:
    """Conservative per-field envelope of this *MVI technical schema*.

    UUID/reference strings are bounded by the actual 128 ASCII identifier
    grammar, not the width of a small test case. This includes the complete
    3+5 source window. The full U1 domain capacity is accounted separately.
    """
    wide_id = "z" * MAX_IDENTIFIER_CODEPOINTS
    max_turn = {
        "event_id": wide_id,
        "event_sequence": MAX_PERSISTED_EVENT_SEQUENCE,
        "episode_id": wide_id,
    }
    wide = {
        "schema_version": 1,
        "stage": "S2",
        "root_digest": "f" * 64,
        "root_revision": 2,
        "source_scope": wide_id,
        "context_id": wide_id,
        "first_sources": [max_turn] * 3,
        "revised_sources": [max_turn] * 5,
        "last_review_event_id": wide_id,
        "last_review_event_sequence": MAX_PERSISTED_EVENT_SEQUENCE,
    }
    return len(canonical_json(wide))


def projected_v10_with_continuity_bytes() -> int:
    """Full R15 U1 legal maxima + one private MVI checkpoint + 16MiB reserve."""
    budget = derive_r15_schema_budget()
    return project_agent_state_schema_max_bytes(
        schema_version=10,
        base_schema_version=9,
        added_field_maxima={
            "relationship_state": budget.relationship_bytes,
            "narrative_self_state": budget.narrative_bytes,
            "self_model_state": budget.self_model_bytes,
            FIELD: checkpoint_full_legal_max_bytes(),
        },
    )


def require_preflight_capacity() -> dict[str, int]:
    projected = projected_v10_with_continuity_bytes()
    if projected + 0 > AGENT_STATE_HARD_CAP_BYTES:
        raise PreflightRefused("V10 private continuation violates 128MiB bound")
    budget = derive_r15_schema_budget()
    return {
        "full_r15_only_with_reserve_bytes": budget.v10_with_reserve_bytes,
        "private_checkpoint_max_bytes": checkpoint_full_legal_max_bytes(),
        "v10_with_private_checkpoint_and_full_16MiB_reserve_bytes": projected,
        "hard_cap_bytes": AGENT_STATE_HARD_CAP_BYTES,
        "untouched_reserve_bytes": FULL_FUTURE_RESERVE_BYTES,
        "headroom_beyond_reserve_bytes": AGENT_STATE_HARD_CAP_BYTES - projected,
    }
