"""S0a development-only ASGI import seam, tested without service startup.

Every child uses disposable cwd and never enters FastAPI lifespan. The tests
do NOT grant I1a/I2 admission, touch real Source Owners or alter systemd.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config.yaml"


def _child(
    tmp_path: Path,
    snippet: str,
    *,
    config_path: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SUZKA_CONFIG_PATH": str(config_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def _success(child: subprocess.CompletedProcess[str]) -> None:
    assert child.returncode == 0, (child.stdout, child.stderr)
    assert child.stdout.strip() == "PASS", (child.stdout, child.stderr)


def test_plain_server_import_does_not_read_ambient_missing_config(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "does-not-exist.yaml"
    assert not absent.exists()
    result = _child(
        tmp_path,
        "import suzka.api.server as server; "
        "assert callable(server.create_app); "
        "assert not hasattr(server, 'app'); "
        "print('PASS')",
        config_path=absent,
    )
    _success(result)
    assert list(tmp_path.iterdir()) == []


def test_legacy_server_app_import_fails_closed_without_config_read(
    tmp_path: Path,
) -> None:
    result = _child(
        tmp_path,
        "import importlib\n"
        "server = importlib.import_module('suzka.api.server')\n"
        "assert callable(server.create_app)\n"
        "assert 'app' not in vars(server)\n"
        "try:\n"
        " from suzka.api.server import app\n"
        "except ImportError as error:\n"
        " assert \"cannot import name 'app'\" in str(error)\n"
        " print('PASS')\n"
        "else:\n"
        " raise AssertionError('legacy ASGI alias unexpectedly exists')",
        config_path=tmp_path / "absent.yaml",
    )
    _success(result)
    assert list(tmp_path.iterdir()) == []


def test_explicit_dev_asgi_still_requires_development_config(
    tmp_path: Path,
) -> None:
    result = _child(
        tmp_path,
        "try:\n"
        " import suzka.api.dev_asgi\n"
        "except FileNotFoundError:\n"
        " print('PASS')\n"
        "else:\n"
        " raise AssertionError('developer ASGI should load configured YAML')",
        config_path=tmp_path / "absent.yaml",
    )
    _success(result)
    assert list(tmp_path.iterdir()) == []


def test_explicit_dev_asgi_constructs_app_without_running_lifespan(
    tmp_path: Path,
) -> None:
    result = _child(
        tmp_path,
        "from suzka.api.dev_asgi import app; "
        "assert app.title == 'PROJECT-SUZKA'; "
        "assert app.state.settings.project.name == 'PROJECT-SUZKA'; "
        "print('PASS')",
        config_path=BASE_CONFIG,
    )
    _success(result)
    assert list(tmp_path.iterdir()) == []


def test_development_main_uses_dev_target_not_legacy_asgi(
    tmp_path: Path,
) -> None:
    result = _child(
        tmp_path,
        "import suzka.api.server as server; import uvicorn; "
        "calls=[]; "
        "uvicorn.run=lambda target, **kwargs: calls.append((target, kwargs)); "
        "server.main(); "
        "assert len(calls) == 1; "
        "assert calls[0][0] == 'suzka.api.dev_asgi:app'; "
        "assert calls[0][1]['reload'] is False; "
        "assert calls[0][1]['host'] == server.get_settings().api.host; "
        "assert calls[0][1]['port'] == server.get_settings().api.port; "
        "print('PASS')",
        config_path=BASE_CONFIG,
    )
    _success(result)
    assert list(tmp_path.iterdir()) == []


def test_explicit_factory_never_falls_back_to_unavailable_env(
    tmp_path: Path,
) -> None:
    result = _child(
        tmp_path,
        "from pathlib import Path; "
        "from suzka.api.server import create_app; "
        "from suzka.config.settings import load_settings; "
        "settings=load_settings(Path(" + repr(str(BASE_CONFIG)) + ")); "
        "app=create_app(settings); "
        "assert app.state.settings is settings; "
        "assert app.title == settings.project.name; "
        "print('PASS')",
        config_path=tmp_path / "absent.yaml",
    )
    _success(result)
    assert list(tmp_path.iterdir()) == []
