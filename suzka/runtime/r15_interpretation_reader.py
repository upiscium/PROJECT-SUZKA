"""IP-8a runtime adapter: R15 interpretation read for R16 consumers.

R16 must not import ``suzka.r15`` (#374 D1 §3-1).  This adapter is the only
runtime module allowed to call ``R15StatePort.read_interpretation`` and
returns plain closed strings; mapping them to the R16 interpretation enum
(SETTLED / UNCERTAIN / CONTRADICTORY / NOT_REVIEWED / UNAVAILABLE) belongs to
IP-8b.  Read-only; call it inside the R16 plan-preview review event before
that event opens its R07 transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from suzka.r15.state_port import R15InterpretationRead, R15StatePort
from suzka.runtime.agent_runtime import AgentEvent


@dataclass(frozen=True, slots=True)
class R15InterpretationEvidence:
    """outcome: EPISODE / NOT_ELIGIBLE / UNAVAILABLE; stage: NONE / S1 / S2;
    status: PRESENT / CONTESTED / CONTRADICTORY / UNKNOWN or None."""

    outcome: str
    stage: str
    status: str | None


_UNAVAILABLE = R15InterpretationEvidence("UNAVAILABLE", "NONE", None)


class R15InterpretationReader:
    def __init__(self, port: R15StatePort) -> None:
        if type(port) is not R15StatePort:
            raise TypeError("an exact R15StatePort is required")
        self._port = port

    def read(
        self, event: AgentEvent, x_event_id: str, x_event_sequence: int
    ) -> R15InterpretationEvidence:
        result = self._port.read_interpretation(event, x_event_id, x_event_sequence)
        if type(result) is not R15InterpretationRead:
            return _UNAVAILABLE
        return R15InterpretationEvidence(
            result.outcome.value, result.stage, result.status
        )


__all__ = ["R15InterpretationEvidence", "R15InterpretationReader"]
