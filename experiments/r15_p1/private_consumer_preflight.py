"""TEST ONLY: bounded, source-requalified R15 U6 private read projection.

Not a production consumer, model prompt, public endpoint, R16 Decision,
read-only forensic Source guarantee, or a way to author subjective truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
from suzka.r15.state_port import R15StatePort
from suzka.runtime.agent_state import (
    AgentStateLoadError,
    AgentStateSnapshotV10,
    AgentStateStore,
)


class PrivateReadStatus(str, Enum):
    QUALIFIED = "qualified_current_sources"
    HOLD = "hold_or_source_unavailable"


@dataclass(frozen=True, slots=True)
class PrivateNarrativeProjection:
    """No raw conversation, identity key, credential, or public prose."""

    status: PrivateReadStatus
    revision: int | None
    original_referent: str | None
    current_referent: str | None
    later_user_reported_referent: bool
    human_intent_verified: bool = False
    competence_verified: bool = False


_HOLD = PrivateNarrativeProjection(PrivateReadStatus.HOLD, None, None, None, False)


def _current(
    snapshot: AgentStateSnapshotV10,
    port: R15StatePort,
) -> tuple[NarrativeSelfState, NarrativeContinuation]:
    """One coherent source of private U3 bytes must equal stored WAL state."""
    captured = port._capture_for_agent_state()
    root, continuation = captured.narrative, captured.continuation
    if (
        root.canonical_value() != snapshot.narrative_self_state
        or continuation.canonical_value(root) != snapshot.narrative_admission_checkpoint
    ):
        raise ContinuityInvalid("process-local subject root differs from committed WAL")
    return root, continuation


def _source_bound_options(
    root: NarrativeSelfState,
    continuation: NarrativeContinuation,
    gate: QualifiedNarrativeSourceGate,
) -> tuple[str, str]:
    """Actual read of every committed source, no source ID is trusted alone."""
    if root.revision not in (1, 2) or continuation.revision != root.revision:
        raise ContinuityInvalid("unavailable cognitive revision")
    if continuation.source_scope is None or continuation.context_id is None:
        raise ContinuityInvalid("unavailable source/Context meaning")
    scope = continuation.source_scope
    refs = (
        continuation.revised_sources
        if root.revision == 2
        else continuation.first_sources
    )
    if len(refs) != (5 if root.revision == 2 else 3):
        raise ContinuityInvalid("private source window incomplete")
    observed = tuple(gate._read(ref) for ref in refs)
    if len({row.context_id for row in observed}) != 1:
        raise ContinuityInvalid("source Context generations diverge")
    if observed[0].context_id != continuation.context_id:
        raise ContinuityInvalid("event-time Context not currently qualified")
    if observed[0].episode_id != scope or not _SETUP.fullmatch(observed[0].request):
        raise ContinuityInvalid("original option universe unavailable")

    original_response = observed[1].response.replace("*", "")
    choices = {
        name for pattern in _PRIOR for name in pattern.findall(original_response)
    }
    if len(choices) != 1:
        raise ContinuityInvalid("original first-person option ambiguous")
    original_choice = next(iter(choices))
    if _parse_candidate(observed[2].response, "S1") != original_choice:
        raise ContinuityInvalid("S1 candidate differs from original own response")

    first = tuple(item for item in root.episodes if not item.links)
    later = tuple(item for item in root.episodes if item.links)
    if len(first) != 1 or len(later) != root.revision - 1:
        raise ContinuityInvalid("narrative episode linkage incomplete")
    if first[0].experience != observed[1].witness:
        raise ContinuityInvalid("original R12 witness changed")
    if (
        restore_scoped_option(scope, first[0].interpretation.statement_id)
        != original_choice
    ):
        raise ContinuityInvalid("stored S1 and currently observed meaning differ")

    if root.revision == 1:
        return original_choice, original_choice

    reported = _LATER.fullmatch(observed[3].request)
    if reported is None:
        raise ContinuityInvalid("later E2 user statement is unavailable")
    current = _parse_candidate(observed[4].response, "S2")
    if (
        current == original_choice
        or reported.group(1) != current
        or later[0].experience != observed[3].witness
        or restore_scoped_option(scope, later[0].interpretation.statement_id) != current
    ):
        raise ContinuityInvalid("revised subjective meaning lacks current sources")
    return original_choice, current


def read_private_committed_narrative(
    store: AgentStateStore,
    port: R15StatePort,
    gate: QualifiedNarrativeSourceGate,
) -> PrivateNarrativeProjection:
    """Prototype of future INTERNAL U6 use; never mutate or publicly render.

    An absent, retired, incomplete, inconsistent or locally speculative U3
    root is simply not usable as a qualified private projection.
    """
    if (
        type(store) is not AgentStateStore
        or type(port) is not R15StatePort
        or type(gate) is not QualifiedNarrativeSourceGate
        or port.narrative._source_gate is not gate
    ):
        raise TypeError("private read requires exact internally bound owners")
    try:
        before = store.load()
        if not isinstance(before, AgentStateSnapshotV10):
            return _HOLD
        root, continuation = _current(before, port)
        original, latest = _source_bound_options(root, continuation, gate)
        # After the potentially normal-recovering source reads, neither the
        # authoritative WAL nor the U3 owner may have advanced independently.
        after = store.load()
        if not isinstance(after, AgentStateSnapshotV10) or before != after:
            return _HOLD
        _current(after, port)
        return PrivateNarrativeProjection(
            PrivateReadStatus.QUALIFIED,
            root.revision,
            original,
            latest,
            root.revision == 2,
        )
    except (
        AgentStateLoadError,
        ContinuityInvalid,
        QualifiedProposalUnavailable,
        ValueError,
        OSError,
    ):
        return _HOLD
