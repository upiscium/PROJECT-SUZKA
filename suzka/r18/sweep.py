"""IP-18a: the real ``AdmissionGate`` and the sweep's S1 rotation
(#411 IP-18 D1 r4 §3-1, §3-2).

``SweepGate`` replaces ``ALWAYS_OPEN_UNTIL_IP18``. It is closed from
construction; the executor opens it after the sweep (S1-S5) and re-closes it
when the target fails. With ``r18.enabled`` false it is ``DISABLED``: closed
for good, never ``BLOCKED_TARGET``, and nothing can open it.

``is_open`` does no I/O. ``subscribe`` callbacks run on every transition into
OPEN and, at subscription, at once when the gate is already OPEN; they only
post a message to their owner's queue (the R16 driver's own ``_post``) and
never touch another thread's state.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import logging
from threading import Lock
import time
from typing import Final

from suzka.r18.target import TargetStore

LOGGER = logging.getLogger(__name__)

# IP-18b (D1 r4 §9-5, R3-4): BLOCKED_TARGET clears only after the gate has
# been OPEN continuously for this long, so a flapping target cannot churn
# Plan A <-> row 4 Decisions faster than one return per dwell.
BLOCKED_MIN_DWELL_SECONDS: Final = 300.0


class GateState(str, Enum):
    DISABLED = "disabled"
    SWEEPING = "sweeping"
    OPEN = "open"
    BLOCKED_TARGET = "blocked_target"


class SweepGate:
    """Closed until the sweep has settled every unresolved key."""

    def __init__(
        self,
        *,
        enabled: bool,
        monotonic: Callable[[], float] = time.monotonic,
        blocked_dwell_seconds: float = BLOCKED_MIN_DWELL_SECONDS,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        self._monotonic = monotonic
        self._dwell = blocked_dwell_seconds
        # The latch: set on entering BLOCKED_TARGET, cleared after the dwell.
        self._latched = False
        self._open_since: float | None = None
        self._unblocked_subscribers: list[Callable[[], None]] = []
        self._lock = Lock()
        self._state = GateState.SWEEPING if enabled else GateState.DISABLED
        self._subscribers: list[Callable[[], None]] = []

    @property
    def state(self) -> GateState:
        return self._state

    def is_open(self) -> bool:
        return self._state is GateState.OPEN

    def blocked_target(self) -> bool:
        """The Decision issuer's view (§9-5): True in BLOCKED_TARGET and, after
        the target returns, until the gate has been OPEN continuously for the
        minimum dwell (hysteresis). Never True in DISABLED / SWEEPING unless
        the latch is still pending. No I/O. When this call is the one that
        observes the dwell elapsed, the unblocked subscribers run."""

        return self._evaluate_latch()

    def _evaluate_latch(self) -> bool:
        with self._lock:
            if not self._latched:
                return False
            if self._state is GateState.OPEN and self._open_since is not None:
                if self._monotonic() - self._open_since >= self._dwell:
                    self._latched = False
                    subscribers = list(self._unblocked_subscribers)
                else:
                    return True
            else:
                return True
        for callback in subscribers:
            self._call(callback)
        return False

    def poll(self) -> None:
        """Let the latch clear on time alone (the executor's reconciler tick
        calls this), so the unblocked subscribers fire without a Decision."""

        self._evaluate_latch()

    def subscribe_unblocked(self, callback: Callable[[], None]) -> None:
        """``callback`` runs once each time BLOCKED_TARGET clears (after the
        dwell); like ``subscribe`` it only posts to its owner's queue."""

        with self._lock:
            self._unblocked_subscribers.append(callback)

    # ------------------------------------------------------------ transitions

    def _move(self, state: GateState) -> None:
        with self._lock:
            if self._state is GateState.DISABLED or self._state is state:
                return
            self._state = state
            if state is GateState.BLOCKED_TARGET:
                self._latched = True
            # The dwell is "OPEN continuously": any leave of OPEN restarts it.
            self._open_since = self._monotonic() if state is GateState.OPEN else None
            subscribers = list(self._subscribers) if state is GateState.OPEN else []
        for callback in subscribers:
            self._call(callback)

    def open(self) -> None:
        self._move(GateState.OPEN)

    def sweeping(self) -> None:
        self._move(GateState.SWEEPING)

    def blocked(self) -> None:
        self._move(GateState.BLOCKED_TARGET)

    def subscribe(self, callback: Callable[[], None]) -> None:
        """``callback`` runs on every transition into OPEN, and now when the
        gate is already OPEN (a late subscriber cannot miss the transition)."""

        with self._lock:
            self._subscribers.append(callback)
            already = self._state is GateState.OPEN
        if already:
            self._call(callback)

    @staticmethod
    def _call(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            LOGGER.warning("R18 gate subscriber failed")

    def __repr__(self) -> str:
        return f"SweepGate({self._state.value})"


def rotate_if_needed(store: TargetStore, runtime_instance_id: str) -> bool:
    """S1, idempotent: ``meta`` first; "already this process's id" is done.

    Returns True when this call rotated. ``rotate_runtime_instance`` raises
    ``TargetBindingError`` for an id that is already there, so a retry after a
    lost response, or a restarted worker, must not call it again."""

    if store.meta().runtime_instance_id == runtime_instance_id:
        return False
    store.rotate_runtime_instance(runtime_instance_id)
    return True


__all__ = ["GateState", "SweepGate", "rotate_if_needed"]
