"""T1: one real app per subject, in this process (D1 sec. 4-1, 4-4).

* ``FakeClock`` patches ``suzka.runtime.agent_runtime.datetime`` PROCESS-WIDE
  (the same seam ``tests/r19_wake_fixtures.py`` uses) and the owners' live
  clocks.  Only one may be installed at a time: a no-kill comparison run goes
  in a separate process or sequentially, never concurrently with a patched
  subject.
* ``T1Runtime`` builds ``create_app`` on the factory's config (absolute paths),
  injects the provider and enters a ``TestClient``.  "Restart" is a new
  ``create_app`` on the SAME root.  Fault injection stays the existing
  test-side monkeypatch seams (D1 fact 13); 22c attaches them through
  ``T1Runtime.state``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any, ClassVar

from fastapi.testclient import TestClient

from experiments.r20.factory import GenesisSubject, Tier, verify_prepared
from experiments.r20.guards import GuardRefusal

from suzka.api.server import create_app
from suzka.runtime import agent_runtime

_REAL_DATETIME = datetime
STAMP = "%Y-%m-%dT%H:%M:%SZ"
# Owners whose live clock is replaced when present (as the existing fixtures do).
_OWNER_CLOCKS = (
    "wake_claim_owner",
    "wake_review_owner",
    "decision_issuer",
    "inbox_ack_owner",
    "approval_owner",
)


class FakeClock:
    """A settable clock: forward-only by ``advance``, backward by ``set_back``."""

    _installed: ClassVar[FakeClock | None] = None

    def __init__(self) -> None:
        self.offset = timedelta(0)
        self.skew = timedelta(0)
        self._real = agent_runtime.datetime

    def install(self) -> FakeClock:
        if FakeClock._installed is not None:
            raise GuardRefusal("FAKE_CLOCK_ALREADY_INSTALLED")
        outer = self

        class _Datetime(_REAL_DATETIME):  # type: ignore[misc]
            @classmethod
            def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
                return _REAL_DATETIME.now(tz) + outer.offset

        self._real = agent_runtime.datetime
        agent_runtime.datetime = _Datetime  # type: ignore[assignment,misc]
        FakeClock._installed = self
        return self

    def uninstall(self) -> None:
        if FakeClock._installed is self:
            agent_runtime.datetime = self._real  # type: ignore[misc]
            FakeClock._installed = None

    def __enter__(self) -> FakeClock:
        return self.install()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.uninstall()

    def attach(self, state: Any) -> None:
        """Point the owners' live clocks at this clock."""

        for name in _OWNER_CLOCKS:
            owner = getattr(state, name, None)
            if owner is not None and hasattr(owner, "_clock"):
                owner._clock = self.live

    def now(self) -> datetime:
        return _REAL_DATETIME.now(timezone.utc) + self.offset

    def live(self) -> datetime:
        return self.now() + self.skew

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("advance is forward-only; use set_back")
        self.offset += timedelta(seconds=seconds)

    def set_back(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("set_back takes a positive number of seconds")
        self.offset -= timedelta(seconds=seconds)

    def stamp(self, seconds: float = 0) -> str:
        return (self.now() + timedelta(seconds=seconds)).strftime(STAMP)


class T1Runtime:
    """One in-process app on a T1 subject."""

    def __init__(
        self,
        subject: GenesisSubject,
        provider: Any,
        *,
        executor: bool = True,
        scheduler: bool = True,
        clock: FakeClock | None = None,
    ) -> None:
        if subject.tier is not Tier.T1:
            raise GuardRefusal("NOT_A_T1_SUBJECT", subject.run_id)
        self.subject = subject
        self.provider = provider
        self.executor = executor
        self.scheduler = scheduler
        self.clock = clock
        self.client: TestClient | None = None

    @property
    def state(self) -> Any:
        if self.client is None:
            raise RuntimeError("the T1 runtime is not started")
        return self.client.app.state  # type: ignore[attr-defined]

    def start(self) -> TestClient:
        if self.client is not None:
            raise RuntimeError("already started")
        verify_prepared(self.subject)
        settings = self.subject.settings()
        app = create_app(settings)
        app.state.model_provider = self.provider
        app.state.r18_executor_enabled = self.executor
        app.state.r19_scheduler_enabled = self.scheduler
        client = TestClient(app, base_url=f"http://localhost:{self.subject.port}")
        client.__enter__()
        self.client = client
        if self.clock is not None:
            self.clock.attach(client.app.state)  # type: ignore[attr-defined]
        return client

    def stop(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            client.__exit__(None, None, None)

    def restart(self) -> TestClient:
        """A fresh ``create_app`` on the same root (the F06 restart)."""

        self.stop()
        return self.start()

    def __enter__(self) -> T1Runtime:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


__all__ = ["STAMP", "FakeClock", "T1Runtime"]
