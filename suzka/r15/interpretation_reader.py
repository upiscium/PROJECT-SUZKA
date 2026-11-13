"""R15 IP-8a: the read-only R15 -> R16 interpretation eligibility read.

#374 F1 D3 §5, the D3 addendum P1-1, D4 §3 and D5 §3.  For one experience X
(the first turn the subject answered), return a closed value:

- ``EPISODE``: X has an admitted, *currently re-validated* episode; carries
  its stage (S1/S2) and status class.
- ``NOT_ELIGIBLE``: X is not R15-shaped (R16 maps this to NOT_REVIEWED).
- ``UNAVAILABLE``: anything that cannot be proven either way (fail-closed).

Order (D4 §3, D5 §3):
1. episode first: if ``continuation.first_sources[1]`` is X, re-validate the
   committed root exactly as the private read does; failure is UNAVAILABLE;
2. otherwise condition 2 first: X's own response must match ``_PRIOR``
   exactly once, else NOT_ELIGIBLE without reading any predecessor;
3. then condition 1: the nearest earlier qualified CHAT turn of the same
   Context, found by a descending scan of the retained Journal, must
   fullmatch ``_SETUP`` (UNAVAILABLE: R15-shaped without an episode) or not
   (NOT_ELIGIBLE).  No predecessor found in the retained window, or any read
   failure, is UNAVAILABLE.

Carries no rationale, source ID or text.  Never mutates anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.experience_store import ExperienceStoreError
from suzka.r15.continuity_state import ContinuityInvalid
from suzka.r15.private_projection import _committed_root, _qualify_source_meaning
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
    _PRIOR,
    _SETUP,
)
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.agent_state import (
    AgentStateLoadError,
    AgentStateSnapshotV10,
    AgentStateStore,
)
from suzka.runtime.event_journal import EventJournalError, EventLifecycle

if TYPE_CHECKING:
    from suzka.r15.state_port import R15StatePort


class InterpretationOutcome(str, Enum):
    EPISODE = "EPISODE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class R15InterpretationRead:
    """Closed values only: outcome, stage in {NONE,S1,S2}, status class."""

    outcome: InterpretationOutcome
    stage: str
    status: str | None

    def __post_init__(self) -> None:
        valid = (
            (self.outcome is InterpretationOutcome.EPISODE)
            == (self.stage in ("S1", "S2"))
            and (self.stage == "NONE") == (self.status is None)
            and self.status
            in (None, "PRESENT", "CONTESTED", "CONTRADICTORY", "UNKNOWN")
            and not (self.stage == "S1" and self.status not in ("PRESENT", "UNKNOWN"))
        )
        if not valid:
            raise ValueError("R15 interpretation read is not a closed combination")


NOT_ELIGIBLE = R15InterpretationRead(InterpretationOutcome.NOT_ELIGIBLE, "NONE", None)
UNAVAILABLE = R15InterpretationRead(InterpretationOutcome.UNAVAILABLE, "NONE", None)

_READ_ERRORS = (
    AgentStateLoadError,
    ContinuityInvalid,
    QualifiedProposalUnavailable,
    ExperienceStoreError,
    EventJournalError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    RecursionError,
)


def _turn_ref(
    gate: QualifiedNarrativeSourceGate, event_id: str, event_sequence: int
) -> QualifiedTurnRef:
    record = gate._experience.load_committed_current(
        experience_id_for_event(event_id, event_sequence)
    ).record
    return QualifiedTurnRef(event_id, event_sequence, record.source_episode_id)


def _prior_matches(response: str) -> int:
    text = response.replace("*", "")
    return sum(len(pattern.findall(text)) for pattern in _PRIOR)


def read_interpretation(
    store: AgentStateStore,
    port: R15StatePort,
    gate: QualifiedNarrativeSourceGate,
    *,
    x_event_id: str,
    x_event_sequence: int,
    event_sequence: int,
) -> R15InterpretationRead:
    try:
        before = store.load()
        if not isinstance(before, AgentStateSnapshotV10):
            return UNAVAILABLE
        root, continuation = _committed_root(before, port)
        first = continuation.first_sources
        if (
            len(first) == 3
            and first[1].event_id == x_event_id
            and first[1].event_sequence == x_event_sequence
        ):
            if continuation.context_id is None:
                return UNAVAILABLE
            meaning = _qualify_source_meaning(
                root,
                continuation,
                gate,
                context_id=continuation.context_id,
                event_sequence=event_sequence,
            )
            after = store.load()
            if not isinstance(after, AgentStateSnapshotV10) or after != before:
                return UNAVAILABLE
            return R15InterpretationRead(
                InterpretationOutcome.EPISODE,
                "S2" if root.revision == 2 else "S1",
                meaning.status_class,
            )
        x = gate._read(_turn_ref(gate, x_event_id, x_event_sequence))
        if x.event.event_sequence >= event_sequence:
            return UNAVAILABLE
        if _prior_matches(x.response) != 1:
            return NOT_ELIGIBLE
        return _predecessor_shape(gate, x.context_id, x_event_sequence)
    except _READ_ERRORS:
        return UNAVAILABLE


def _predecessor_shape(
    gate: QualifiedNarrativeSourceGate, context_id: str, x_event_sequence: int
) -> R15InterpretationRead:
    """Nearest earlier qualified CHAT turn of the same Context (D5 §3, P2-1)."""

    inspection = gate._journal.inspect()
    candidates = sorted(
        {
            (record.processing_sequence, record.event_id)
            for record in inspection.records
            if record.lifecycle is EventLifecycle.COMPLETED
            and record.event_type is AgentEventType.CHAT
            and record.event_id is not None
            and record.processing_sequence is not None
            and record.processing_sequence < x_event_sequence
        },
        reverse=True,
    )
    for sequence, event_id in candidates:
        # The same qualification as admission (_read): ACCEPTED, COMPLETED and
        # TRANSACTION_COMPLETED once each, no FAILED, API_CHAT.  A turn of
        # another Context that cannot be read is UNAVAILABLE (fail-closed).
        turn = gate._read(_turn_ref(gate, event_id, sequence))
        if turn.context_id != context_id:
            continue
        return UNAVAILABLE if _SETUP.fullmatch(turn.request) else NOT_ELIGIBLE
    # No predecessor in the retained window, Context start unprovable.
    return UNAVAILABLE


__all__ = [
    "NOT_ELIGIBLE",
    "UNAVAILABLE",
    "InterpretationOutcome",
    "R15InterpretationRead",
    "read_interpretation",
]
