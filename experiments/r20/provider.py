"""Provider stand-ins and the provider / approval-actor mode guard (D1 sec. 3-5).

The scripted provider is injected as ``app.state.model_provider`` by the
harness; nothing is added to ``suzka/models``.  ``--provider real`` is refused
unless the Human's own launch is attested: that refusal is a MISUSE GUARD, not
an approval (permission to run a real model is HP-4 and the Human's own start
of the run).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
import hashlib
import json
import threading

from suzka.models.dummy_provider import DummyProvider

from experiments.r20.guards import GuardRefusal

# A large loss makes the appraisal non-neutral (#370 DRV-1 D): X is then not
# skipped as ``neutral_no_eligible_appraisal``.
NON_NEUTRAL_LOSS = 10000.0
HIDDEN_THOUGHT_MARKER = "R20-HIDDEN-THOUGHT-CANARY"
DEFAULT_ANSWER = "synthetic visible answer (fixture only)"
# The S0 case target: the referent announced in the E2 (reveal) turn.  Fixed (static, pre-registered) to A because
# the real model's original choice was B in 6 of 6 observed runs; the production R15 gate needs target != original.
S0_TARGET = "A"


class ProviderMode(str, Enum):
    SCRIPTED = "scripted"
    DUMMY = "dummy"
    REAL = "real"


class ApprovalActor(str, Enum):
    SCRIPTED = "SCRIPTED"
    HUMAN = "HUMAN"


def check_modes(
    provider: ProviderMode, actor: ApprovalActor, *, human_launch: bool
) -> None:
    """HD-22-3 (a): SCRIPTED approval only with the scripted provider.

    A real provider needs the HUMAN actor and the Human's own launch flag.
    """

    if provider is ProviderMode.REAL:
        if actor is not ApprovalActor.HUMAN:
            raise GuardRefusal("SCRIPTED_APPROVAL_WITH_REAL_PROVIDER")
        if not human_launch:
            raise GuardRefusal("REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH")


def r15_answer(phase: str, selected: str, *, no_reveal: bool = False) -> str:
    """The typed R15 S1 / S2 JSON (as ``tests/test_r15_p1_owner_bridge.py``)."""

    return json.dumps(
        {
            "schema": 1,
            "phase": phase,
            "assumed_referent": selected,
            "assessment": (
                "LIKELY_ALIGNED"
                if phase == "S1"
                else "MAY_HAVE_MISREAD"
                if not no_reveal
                else "UNKNOWN"
            ),
            "uncertainty": "LOW" if phase == "S1" else "MEDIUM",
            "evidence_steps": (
                ["E"]
                if phase == "S1"
                else ["E", "S1"]
                if no_reveal
                else ["E", "S1", "E2"]
            ),
            "rationale": "synthetic fixture only, not subject adoption",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def r15_protocol_script(*, no_reveal: bool = False) -> tuple[str, ...]:
    """Answers for the R15 protocol turns, by call index (setup, E answer,
    S1, E2 answer, S2)."""

    s2_index = 3 if no_reveal else 4
    answers = ["synthetic visible answer"] * (s2_index + 1)
    answers[1] = "元の回答では案Bを暫定的に選びました"
    answers[2] = r15_answer("S1", "B", no_reveal=no_reveal)
    if not no_reveal:
        answers[3] = "補足では案Aと言われました"
    answers[s2_index] = r15_answer("S2", "B" if no_reveal else "A", no_reveal=no_reveal)
    return tuple(answers)


class ScriptedProvider(DummyProvider):
    """``generate`` returns the next scripted answer; ``calculate_loss`` is a
    fixed large value so the appraisal is non-neutral."""

    def __init__(
        self,
        script: Sequence[str] = (),
        *,
        default: str = DEFAULT_ANSWER,
        loss_value: float = NON_NEUTRAL_LOSS,
    ) -> None:
        self._script = tuple(script)
        self._default = default
        self.loss_value = loss_value
        self.calls = 0
        self._lock = threading.Lock()

    def generate(self, prompt: str) -> str:
        with self._lock:
            index = self.calls
            self.calls += 1
        return self._script[index] if index < len(self._script) else self._default

    @property
    def identity(self) -> dict[str, object]:
        return {
            "mode": ProviderMode.SCRIPTED.value,
            "script_sha256": script_digest(self._script, self._default, self.loss_value),
            "fixture_only": True,
        }


def script_digest(script: Sequence[str], default: str, loss_value: float) -> str:
    payload = json.dumps(
        {"default": default, "loss": loss_value, "script": list(script)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_identity(mode: ProviderMode, provider: object | None) -> dict[str, object]:
    if isinstance(provider, ScriptedProvider):
        return provider.identity
    return {"mode": mode.value, "fixture_only": mode is not ProviderMode.REAL}


__all__ = [
    "DEFAULT_ANSWER",
    "HIDDEN_THOUGHT_MARKER",
    "NON_NEUTRAL_LOSS",
    "S0_TARGET",
    "ApprovalActor",
    "ProviderMode",
    "ScriptedProvider",
    "check_modes",
    "provider_identity",
    "r15_answer",
    "r15_protocol_script",
    "script_digest",
]
