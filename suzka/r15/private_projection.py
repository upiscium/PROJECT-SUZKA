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
from suzka.r15.contracts import NarrativeSelfState
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
    _LATER,
    _PRIOR,
    _SETUP,
    _parse_candidate,
    restore_scoped_option,
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


@dataclass(frozen=True, slots=True)
class PrivateNarrativeProjection:
    """Only finite A/B and revision semantics; no raw source/identity/text."""

    status: PrivateReadStatus
    revision: int | None
    original_referent: str | None
    current_referent: str | None
    later_user_reported: bool


HOLD = PrivateNarrativeProjection(PrivateReadStatus.HOLD, None, None, None, False)


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
) -> tuple[str, str]:
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
    if observed[0].episode_id != original_scope or not _SETUP.fullmatch(
        observed[0].request
    ):
        raise ContinuityInvalid("original option universe unavailable")
    choices = {
        name
        for pattern in _PRIOR
        for name in pattern.findall(observed[1].response.replace("*", ""))
    }
    if len(choices) != 1:
        raise ContinuityInvalid("original first-person answer ambiguous")
    original = next(iter(choices))
    if _parse_candidate(observed[2].response, "S1") != original:
        raise ContinuityInvalid("S1 no longer matches original answer")

    first = tuple(episode for episode in root.episodes if not episode.links)
    later = tuple(episode for episode in root.episodes if episode.links)
    if (
        len(first) != 1
        or len(later) != root.revision - 1
        or first[0].experience != observed[1].witness
        or restore_scoped_option(original_scope, first[0].interpretation.statement_id)
        != original
    ):
        raise ContinuityInvalid("U3 original meaning/witness unavailable")

    if root.revision == 1:
        return original, original

    report = _LATER.fullmatch(observed[3].request)
    selected = _parse_candidate(observed[4].response, "S2")
    if (
        report is None
        or report.group(1) != selected
        or selected == original
        or later[0].experience != observed[3].witness
        or restore_scoped_option(original_scope, later[0].interpretation.statement_id)
        != selected
    ):
        raise ContinuityInvalid("U3 revised meaning lacks current evidence")
    return original, selected


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
        original, current = _qualify_source_meaning(
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
            original,
            current,
            root.revision == 2,
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
