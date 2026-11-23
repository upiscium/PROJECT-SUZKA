"""Thread liveness accounting for the F15 termination proof (DC-1).

A thread owner's ``stop()`` joins with a timeout and then *forgets* the
handle, so a handle-based check would report "stopped" for a thread that is
still running.  ``ThreadLedger`` therefore remembers every thread the owner
ever started and reports the ones that are still alive, whatever the owner
believes.  ``threads_alive()`` is read-only: it never joins, signals or
changes the owner (#411 F15 slice D1 sec. 5.1).

Pure standard library, no suzka imports: any layer may use it.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class ThreadsAlive(Protocol):
    """``threads_alive()`` returns the names of the owner's threads that are
    still alive (empty when every one has exited)."""

    def threads_alive(self) -> tuple[str, ...]: ...


class ThreadLedger:
    """Remember started threads; report the still-alive ones by name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def register(self, thread: threading.Thread) -> None:
        """Record a thread that has been started successfully.  A thread that
        never started (``start()`` raised) must not be registered."""

        with self._lock:
            self._prune_unlocked()
            self._threads.append(thread)

    def alive(self) -> tuple[str, ...]:
        with self._lock:
            self._prune_unlocked()
            return tuple(thread.name for thread in self._threads)

    def _prune_unlocked(self) -> None:
        self._threads = [thread for thread in self._threads if thread.is_alive()]
