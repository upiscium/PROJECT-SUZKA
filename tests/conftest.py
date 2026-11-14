"""Shared test isolation.

IP-10b: the repository ``config.yaml`` (which the running service reads)
enables SubjectHistory at a repository-relative path.  Tests load that file
for unrelated reasons; they must never adopt or touch a store in the
checkout's ``.suzka``.  A test that loads ``config.yaml`` without relocating
``subject_history.path`` therefore runs as if SubjectHistory were disabled.
Tests of SubjectHistory itself set an absolute temporary path explicitly.

The patch uses its own ``MonkeyPatch`` so a test's ``monkeypatch.undo()``
cannot remove it, and the session fails if a checkout store appears anyway.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_HISTORY = ROOT / ".suzka" / "private" / "subject_history"


@pytest.fixture(autouse=True)
def _isolate_repository_subject_history() -> Iterator[None]:
    import suzka.api.server as server

    real = server._open_subject_history

    def isolated(app_settings, journal):  # type: ignore[no-untyped-def]
        settings = app_settings.subject_history
        if settings.enabled and not settings.path.expanduser().is_absolute():
            disabled = app_settings.model_copy(
                update={"subject_history": settings.model_copy(update={"enabled": False})}
            )
            return real(disabled, journal)
        return real(app_settings, journal)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(server, "_open_subject_history", isolated)
        yield


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config._suzka_checkout_history = CHECKOUT_HISTORY.exists()  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    existed = getattr(session.config, "_suzka_checkout_history", True)
    if not existed and CHECKOUT_HISTORY.exists():
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        raise pytest.UsageError(
            f"a test created the checkout SubjectHistory store at {CHECKOUT_HISTORY}"
        )
