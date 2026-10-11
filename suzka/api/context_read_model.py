"""Private externally-terminal Context projection for G1-P public reads.

This is a public-read consistency mechanism, NOT the G1-C Context-capture
participant, an event-time source witness, a Memory reader or R15 evidence.
No new durable file or journal schema is introduced by this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from suzka.runtime.context import (
    ContextRegistryState,
    validate_context_registry_state,
)


class ContextPublicationUnavailable(RuntimeError):
    """The latest externally terminal Context cannot currently be projected."""


class ContextPublicationStatus(StrEnum):
    READY = "ready"
    STARTED_CHECKPOINT_PENDING = "started_checkpoint_pending"
    TERMINAL_PUBLICATION_PENDING = "terminal_publication_pending"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class PublishedContextView:
    """One detached immutable Context generation returned by an admitted GET."""

    state: ContextRegistryState
    terminal_sequence: int
    snapshot_hash: str

    def __post_init__(self) -> None:
        if type(self.state) is not ContextRegistryState:
            raise TypeError("Context view requires an exact registry state")
        validate_context_registry_state(self.state)
        if type(self.terminal_sequence) is not int or self.terminal_sequence < 0:
            raise ValueError("Context terminal sequence is invalid")
        if (
            type(self.snapshot_hash) is not str
            or len(self.snapshot_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.snapshot_hash)
        ):
            raise ValueError("Context snapshot hash is invalid")


class CommittedContextReadModel:
    """Admit public reads only against a single *externally terminal* view.

    All GETs linearize at snapshot() and materialize from that returned
    immutable handle. During a terminal append/swap, later admissions get a
    denial, not a stale or speculative Context. The lock never spans model
    generation or a durable write.

    The caller is the trusted API runtime lifecycle only; these methods are
    not public mutation endpoints and never establish event-time B evidence.
    """

    def __init__(self, initial: PublishedContextView, *, ready: bool) -> None:
        if type(initial) is not PublishedContextView or type(ready) is not bool:
            raise TypeError("Context view bootstrap is invalid")
        self._lock = RLock()
        self._view = initial
        self._status = (
            ContextPublicationStatus.READY
            if ready
            else ContextPublicationStatus.DEGRADED
        )
        self._pending: tuple[str, int] | None = None
        self._restoring: PublishedContextView | None = None

    @property
    def status(self) -> ContextPublicationStatus:
        with self._lock:
            return self._status

    def snapshot(self) -> PublishedContextView:
        with self._lock:
            if self._status is not ContextPublicationStatus.READY:
                raise ContextPublicationUnavailable("Committed Context is unavailable")
            return self._view

    def append_started_checkpoint(self, append: Callable[[], None]) -> None:
        """Fence STARTED append without holding a read lock across disk I/O.

        While the durable Journal append runs, NEW Context GETs fail fast
        rather than waiting indefinitely. Once the append finishes, a verified
        success restores the old public READY generation; failure degrades
        before any new GET can acquire a READY handle.
        """
        if not callable(append):
            raise TypeError("STARTED checkpoint must be callable")
        with self._lock:
            if (
                self._status is not ContextPublicationStatus.READY
                or self._pending is not None
            ):
                raise ContextPublicationUnavailable("Context STARTED gate unavailable")
            self._status = ContextPublicationStatus.STARTED_CHECKPOINT_PENDING
        try:
            append()
        except BaseException:
            with self._lock:
                self._pending = None
                self._restoring = None
                self._status = ContextPublicationStatus.DEGRADED
            raise
        with self._lock:
            if self._status is not ContextPublicationStatus.STARTED_CHECKPOINT_PENDING:
                self._status = ContextPublicationStatus.DEGRADED
                raise ContextPublicationUnavailable("Context STARTED gate diverged")
            self._status = ContextPublicationStatus.READY

    def begin_terminal(
        self, *, event_id: str, sequence: int, expected_previous: PublishedContextView
    ) -> None:
        if type(event_id) is not str or not event_id:
            raise ValueError("Context event identity is invalid")
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("Context event sequence is invalid")
        with self._lock:
            if (
                self._status is not ContextPublicationStatus.READY
                or self._pending is not None
                or self._view is not expected_previous
                or sequence <= self._view.terminal_sequence
            ):
                self._status = ContextPublicationStatus.DEGRADED
                raise ContextPublicationUnavailable("Context publication state diverged")
            self._pending = (event_id, sequence)
            self._restoring = None
            self._status = ContextPublicationStatus.TERMINAL_PUBLICATION_PENDING

    def publish_terminal(
        self, *, event_id: str, sequence: int, candidate: PublishedContextView
    ) -> None:
        if type(candidate) is not PublishedContextView:
            raise TypeError("Context publication candidate is invalid")
        with self._lock:
            if (
                self._status is not ContextPublicationStatus.TERMINAL_PUBLICATION_PENDING
                or self._pending != (event_id, sequence)
                or candidate.terminal_sequence != sequence
                or candidate.terminal_sequence <= self._view.terminal_sequence
            ):
                self._status = ContextPublicationStatus.DEGRADED
                raise ContextPublicationUnavailable("Context terminal publication diverged")
            self._view = candidate
            self._pending = None
            self._status = ContextPublicationStatus.READY

    def degrade(self) -> None:
        with self._lock:
            self._pending = None
            self._restoring = None
            self._status = ContextPublicationStatus.DEGRADED

    def begin_handler_restore(self) -> PublishedContextView | None:
        """Hide all reads until runtime restore + failed Journal terminal finish."""

        with self._lock:
            prior = (
                self._view
                if self._status is ContextPublicationStatus.READY
                and self._pending is None
                else None
            )
            self._restoring = prior
            self._status = ContextPublicationStatus.DEGRADED
            return prior

    def complete_handler_restore(self, prior: PublishedContextView | None) -> None:
        """A handler failure, unlike an uncertain commit, may resume old view."""

        with self._lock:
            if (
                prior is not None
                and self._status is ContextPublicationStatus.DEGRADED
                and self._pending is None
                and self._view is prior
                and self._restoring is prior
            ):
                self._status = ContextPublicationStatus.READY
            self._restoring = None
