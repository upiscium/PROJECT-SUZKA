"""IP-17a: the ``AdmissionGate`` hook (#411 D1 r7 §2-7, §10 N8).

The R18 forward writers (the Intent issuer, the Policy owner, and later the
approval owner and the CLAIM) ask one gate, inside their own serialized
event, whether new forward work may be admitted.  A closed gate is a HOLD
that writes nothing.  The real gate, closed during IP-18's startup
reconciliation sweep, is IP-18's; until then the server installs
``ALWAYS_OPEN_UNTIL_IP18``.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable


@runtime_checkable
class AdmissionGate(Protocol):
    def is_open(self) -> bool:
        """True when forward R18 work may be admitted now (no I/O)."""


class _AlwaysOpen:
    """The pre-IP-18 gate: nothing sweeps yet, so it is always open."""

    __slots__ = ()

    def is_open(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "ALWAYS_OPEN_UNTIL_IP18"


ALWAYS_OPEN_UNTIL_IP18: Final[AdmissionGate] = _AlwaysOpen()


def gate_open(gate: AdmissionGate) -> bool:
    """Fail closed: anything but an exact ``True`` is a closed gate."""

    try:
        return gate.is_open() is True
    except Exception:
        return False


__all__ = ["ALWAYS_OPEN_UNTIL_IP18", "AdmissionGate", "gate_open"]
