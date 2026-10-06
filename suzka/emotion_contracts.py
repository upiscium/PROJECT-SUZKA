"""Dependency-neutral R10 Emotion state; computation remains in the body layer."""

from dataclasses import dataclass
import math


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded(value: float, name: str, lower: float, upper: float) -> float:
    result = _number(value, name)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


@dataclass(frozen=True, slots=True)
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    optimal_loss: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", _bounded(self.valence, "valence", -1.0, 1.0))
        object.__setattr__(self, "arousal", _bounded(self.arousal, "arousal", 0.0, 1.0))
        # Preserve the finite legacy scalar, including the raw update path's
        # negative values. R14 does not strengthen R10 authority invariants.
        object.__setattr__(self, "optimal_loss", _number(self.optimal_loss, "optimal_loss"))


__all__ = ["EmotionState"]
