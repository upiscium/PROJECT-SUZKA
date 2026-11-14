"""Shared test isolation.

IP-10b: the repository ``config.yaml`` (which the running service reads)
enables SubjectHistory at a repository-relative path.  Tests load that file
for unrelated reasons; they must never adopt or touch a store in the
checkout's ``.suzka``.  A test that loads ``config.yaml`` without relocating
``subject_history.path`` therefore runs as if SubjectHistory were disabled.
Tests of SubjectHistory itself set an absolute temporary path explicitly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_repository_subject_history(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(server, "_open_subject_history", isolated)
