"""IP-10b: SubjectHistory is enabled in the running service's profile."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from suzka.config import load_settings

from test_r16_u2a_source_owner import _app, settings_in

ROOT = Path(__file__).resolve().parents[1]


def test_repository_profile_enables_subject_history_with_the_approved_cap() -> None:
    settings = load_settings(ROOT / "config.yaml")
    assert settings.subject_history.enabled is True
    assert settings.subject_history.max_page_count == 65_536
    assert not settings.subject_history.path.is_absolute()


def test_tests_never_touch_the_checkout_store(tmp_path, monkeypatch) -> None:
    """tests/conftest.py: a relative store path is never opened in tests,
    even after the test's own ``monkeypatch.undo()`` (review P1)."""

    from conftest import CHECKOUT_HISTORY

    store_file = CHECKOUT_HISTORY / "subject_history.sqlite3"
    monkeypatch.setenv("SUZKA_IP10B_PROBE", "1")
    monkeypatch.undo()
    if CHECKOUT_HISTORY.exists():
        pytest.skip("a checkout store pre-exists outside this test session")
    with TestClient(_app(settings_in(tmp_path / "isolated"))) as client:
        assert client.app.state.subject_history_store is None
    assert not store_file.exists() and not CHECKOUT_HISTORY.exists()
