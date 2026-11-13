"""Bounded, private, source-requalified U6 read of the committed NarrativeSelf.

A read is not a subjective-state mutation, an independent SourceEpoch
attestation, a verified human-intent claim, or an R16 Decision/Action issuer.
Never expose source references, raw utterances, or an unqualified stale root.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from suzka.r15.continuity_state import ContinuityInvalid, NarrativeContinuation
from suzka.r15.common import InterpretationStatus
from suzka.r15.contracts import NarrativeEpisode, NarrativeSelfState
from suzka.r15.narrative_self_system import _source_episode_digest, derive_episode
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
    proposal_from_observed,
)
from suzka.runtime.agent_state import (
    AgentStateLoadError,
    AgentStateSnapshotV10,
    AgentStateStore,
)

if TYPE_CHECKING:
    from suzka.r15.state_port import R15StatePort


class PrivateReadStatus(str, Enum):
    QUALIFIED = "qualified_current_sources"
    HOLD = "hold_or_source_unavailable"


STATUS_CLASSES = {
    InterpretationStatus.PRESENT: "PRESENT",
    InterpretationStatus.CONTESTED: "CONTESTED",
    InterpretationStatus.CONTRADICTORY: "CONTRADICTORY",
    InterpretationStatus.UNKNOWN: "UNKNOWN",
}


@dataclass(frozen=True, slots=True)
class QualifiedMeaning:
    original: str
    current: str
    status_class: str


@dataclass(frozen=True, slots=True)
class PrivateNarrativeProjection:
    """Only finite A/B/UNKNOWN, revision and status-class semantics.

    No raw source, identity, rationale or text.  ``status_class`` is one of
    PRESENT / CONTESTED / CONTRADICTORY / UNKNOWN (IP-8a, #374 D3 §2).
    """

    status: PrivateReadStatus
    revision: int | None
    original_referent: str | None
    current_referent: str | None
    later_user_reported: bool
    status_class: str | None = None

    @property
    def qualified(self) -> bool:
        return self.status is PrivateReadStatus.QUALIFIED


HOLD = PrivateNarrativeProjection(PrivateReadStatus.HOLD, None, None, None, False, None)


def _committed_root(
    checkpoint: AgentStateSnapshotV10, port: R15StatePort
) -> tuple[NarrativeSelfState, NarrativeContinuation]:
    captured = port._capture_for_agent_state()
    root, continuation = captured.narrative, captured.continuation
    if (
        root.canonical_value() != checkpoint.narrative_self_state
        or continuation.canonical_value(root)
        != checkpoint.narrative_admission_checkpoint
    ):
        raise ContinuityInvalid("uncommitted U3 owner root")
    return root, continuation


def _qualify_source_meaning(
    root: NarrativeSelfState,
    continuation: NarrativeContinuation,
    gate: QualifiedNarrativeSourceGate,
    *,
    context_id: str,
    event_sequence: int,
) -> QualifiedMeaning:
    if (
        root.revision not in (1, 2)
        or continuation.revision != root.revision
        or continuation.source_scope is None
        or continuation.context_id != context_id
    ):
        raise ContinuityInvalid("unavailable committed interpretation/context")

    refs = (
        continuation.revised_sources
        if root.revision == 2
        else continuation.first_sources
    )
    if len(refs) != (5 if root.revision == 2 else 3):
        raise ContinuityInvalid("source window incomplete")
    observed = tuple(gate._read(ref) for ref in refs)
    sequences = tuple(row.event.event_sequence for row in observed)
    if (
        len(set(sequences)) != len(sequences)
        or sequences != tuple(sorted(sequences))
        or sequences[-1] >= event_sequence
        or {row.context_id for row in observed} != {context_id}
    ):
        raise ContinuityInvalid("future, reordered, or cross-context sources")

    original_scope = continuation.source_scope
    if observed[0].episode_id != original_scope:
        raise ContinuityInvalid("original option universe unavailable")
    # Same pure derivation, template gate and v2 table as admission.  D5 §2:
    # S1 *and* S2 episodes are re-derived and compared whole (digest), so a
    # root admitted under the v1 table is HOLD, never a stale inference.
    stage = "S2" if root.revision == 2 else "S1"
    proposal = proposal_from_observed(observed, stage)
    first = tuple(episode for episode in root.episodes if not episode.links)
    later = tuple(episode for episode in root.episodes if episode.links)
    if len(first) != 1 or len(later) != root.revision - 1:
        raise ContinuityInvalid("U3 interpretation topology unavailable")
    expected_first, _row = derive_episode(proposal, "S1")
    if _source_episode_digest(first[0]) != _source_episode_digest(expected_first):
        raise ContinuityInvalid("U3 original interpretation differs from re-derivation")
    if root.revision == 1:
        return QualifiedMeaning(
            proposal.original_choice,
            proposal.original_choice,
            _status_class(first[0]),
        )
    expected_later, _row = derive_episode(proposal, "S2")
    if _source_episode_digest(later[0]) != _source_episode_digest(expected_later):
        raise ContinuityInvalid("U3 revised interpretation differs from re-derivation")
    return QualifiedMeaning(
        proposal.original_choice, proposal.selected, _status_class(later[0])
    )


def _status_class(episode: NarrativeEpisode) -> str:
    # A closed string, so main_loop needs no suzka.r15 enum import (D4 §6).
    return STATUS_CLASSES[episode.interpretation.status]


def read_private_committed_narrative(
    store: AgentStateStore,
    port: R15StatePort,
    gate: QualifiedNarrativeSourceGate,
    *,
    context_id: str,
    event_sequence: int,
) -> PrivateNarrativeProjection:
    """Read strictly from current owners; HOLD, never stale inference, on failure.

    The owning R15StatePort enforces the active serialized runtime event and
    locks this read. Its caller must requalify the same projection after model
    generation, *before* making any source-conditioned response visible.
    """
    from suzka.r15.state_port import R15StatePort

    if (
        type(store) is not AgentStateStore
        or type(port) is not R15StatePort
        or type(gate) is not QualifiedNarrativeSourceGate
        or port.narrative._source_gate is not gate
        or type(context_id) is not str
        or type(event_sequence) is not int
        or event_sequence < 1
    ):
        return HOLD
    try:
        before = store.load()
        if not isinstance(before, AgentStateSnapshotV10):
            return HOLD
        root, continuation = _committed_root(before, port)
        meaning = _qualify_source_meaning(
            root,
            continuation,
            gate,
            context_id=context_id,
            event_sequence=event_sequence,
        )
        after = store.load()
        if not isinstance(after, AgentStateSnapshotV10) or after != before:
            return HOLD
        _committed_root(after, port)
        return PrivateNarrativeProjection(
            PrivateReadStatus.QUALIFIED,
            root.revision,
            meaning.original,
            meaning.current,
            root.revision == 2,
            meaning.status_class,
        )
    except (
        AgentStateLoadError,
        ContinuityInvalid,
        QualifiedProposalUnavailable,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        RecursionError,
    ):
        return HOLD
