"""A relative ``r18.target_path`` resolves against the process cwd once, at
``create_app``, so the executor and every owner see the same absolute path.
Symlink custody in ``_trusted_path`` still applies to the cwd-derived path.
Every fixture is synthetic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from suzka.r18.sweep import GateState
from suzka.r18.target_init import initialize

import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_u2a_source_owner import _app
from test_r18_ip18b_facts_sweep import _until


def _relative_settings(root: Path, relative: Path):  # type: ignore[no-untyped-def]
    base = ip16a2._settings(root, r18=True, init=False)
    return base.model_copy(
        update={"r18": base.r18.model_copy(update={"target_path": relative})}
    )


def test_a_relative_target_path_opens_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rel"
    (root / "t").mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root / "t", 0o700)
    initialize(root / "t" / "target.sqlite3")
    monkeypatch.chdir(root)
    settings = _relative_settings(tmp_path / "settings", Path("t/target.sqlite3"))
    app = _app(settings, executor=True)
    with TestClient(app) as client:
        state = client.app.state
        _until(lambda: state.r18_gate.state is GateState.OPEN)
    assert settings.r18.target_path == Path("t/target.sqlite3")  # caller's settings untouched


def test_an_absolute_target_path_is_unchanged(tmp_path: Path) -> None:
    settings = ip16a2._settings(tmp_path / "abs")
    with TestClient(_app(settings, executor=True)) as client:
        state = client.app.state
        _until(lambda: state.r18_gate.state is GateState.OPEN)


def test_a_relative_path_through_a_symlinked_parent_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "sym"
    real = root / "real"
    real.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(real, 0o700)
    initialize(real / "target.sqlite3")
    (root / "link").symlink_to(real)
    monkeypatch.chdir(root)
    settings = _relative_settings(tmp_path / "settings", Path("link/target.sqlite3"))
    caplog.set_level(logging.INFO, logger="suzka.r18.executor")
    with TestClient(_app(settings, executor=True)) as client:
        state = client.app.state
        _until(lambda: "R18_SWEEP step=FAIL" in caplog.text)
        assert not state.r18_gate.is_open()
    line = next(row for row in caplog.text.splitlines() if "R18_SWEEP step=FAIL" in row)
    assert "error=TargetCorrupt" in line
    assert "reason='target path must be a normal path without symlinks'" in line


def test_with_r18_off_a_relative_path_is_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "off"
    root.mkdir(exist_ok=True)
    monkeypatch.chdir(root)
    base = ip16a2._settings(tmp_path / "settings", r18=False, init=False)
    settings = base.model_copy(
        update={"r18": base.r18.model_copy(update={"target_path": Path("none/target.sqlite3")})}
    )
    with TestClient(_app(settings, executor=True)) as client:
        assert client.app.state.r18_executor is None
        assert client.app.state.r18_gate.state is GateState.DISABLED
    assert not (root / "none").exists()
