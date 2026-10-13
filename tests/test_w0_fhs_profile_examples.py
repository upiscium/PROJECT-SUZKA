"""W0-I0: validate non-active FHS examples using disposable source fixtures.

No production /etc, /var, /opt, systemd activation, Chroma, or model writes.
The examples do not implement trusted config admission, migration, or G2 proof.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import configparser
import json
import os
import subprocess
import sys

import pytest
import yaml

from suzka.config import Settings
from suzka.learning.eval_sets import load_eval_sets
from suzka.memory.experience_store import ExperienceStore
from suzka.memory.semantic_store import SemanticStore


ROOT = Path(__file__).resolve().parents[1]
CONFIG_EXAMPLE = ROOT / "deploy/config/config.fhs.yaml.example"
ENV_EXAMPLE = ROOT / "deploy/env/backend.env.fhs.example"
UNIT_EXAMPLE = ROOT / "deploy/systemd/suzka-api-fhs.service.example"
DOC = ROOT / "docs/w0-fhs-source-storage.md"

STATE_ROOT = Path("/var/lib/project-suzka")
READ_ONLY_EVAL = Path("/usr/share/project-suzka/eval_sets/default.json")
PATH_FIELDS = (
    ("memory", "persist_directory"),
    ("sleep", "dream_dataset_path"),
    ("qlora", "output_dir"),
    ("adapter_registry", "path"),
    ("adapter_registry", "eval_result_dir"),
    ("adapter_registry", "eval_sets"),
    ("agent_state", "path"),
    ("event_journal", "path"),
    ("state_wal", "directory"),
)


def _example() -> dict:
    return yaml.safe_load(CONFIG_EXAMPLE.read_text(encoding="utf-8"))


def test_example_preserves_base_settings_and_uses_nine_absolute_paths() -> None:
    example = _example()
    original = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    changed = deepcopy(example)

    for section, key in PATH_FIELDS:
        changed[section][key] = original[section][key]
        raw = example[section][key]
        paths = raw if isinstance(raw, list) else [raw]
        assert paths and all(Path(item).is_absolute() for item in paths)

    # I0 alters only source-location examples, not live model or privacy rules.
    assert changed == original
    typed = Settings.model_validate(example)
    assert typed.memory.persist_directory == STATE_ROOT / "chroma"
    assert typed.event_journal.path == STATE_ROOT / "journal/event_journal.jsonl"
    assert typed.state_wal.directory == STATE_ROOT / "state_wal"
    assert typed.agent_state.path == STATE_ROOT / "agent_state.json"
    assert typed.adapter_registry.eval_sets == [READ_ONLY_EVAL]
    assert typed.sleep.dream_dataset_path == STATE_ROOT / "dreams/dream_dataset.jsonl"
    assert typed.qlora.output_dir == STATE_ROOT / "adapters"
    assert typed.adapter_registry.path == STATE_ROOT / "adapter_registry.json"
    assert typed.adapter_registry.eval_result_dir == STATE_ROOT / "eval_results"
    assert (
        typed.memory.persist_directory.parent / "experience"
        == STATE_ROOT / "experience"
    )
    assert typed.memory.persist_directory.parent / "semantic" == STATE_ROOT / "semantic"
    assert ExperienceStore.from_memory_root(typed.memory.persist_directory).root == (
        STATE_ROOT / "experience"
    )
    assert SemanticStore.from_memory_root(typed.memory.persist_directory).root == (
        STATE_ROOT / "semantic"
    )
    assert "NOT DEPLOYABLE" in CONFIG_EXAMPLE.read_text(encoding="utf-8")


def test_env_example_is_explicit_and_contains_no_credentials() -> None:
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    active = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert active == ["SUZKA_CONFIG_PATH=/etc/project-suzka/config.yaml"]
    assert "I1/I2" in content
    assert not any(
        line.startswith(("SUZKA_ADMIN_TOKEN=", "HF_TOKEN=")) for line in active
    )


def _unit() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.read_string(UNIT_EXAMPLE.read_text(encoding="utf-8"))
    return parser


def test_unit_is_absolute_nonactive_and_does_not_mutate_state_directory() -> None:
    unit = _unit()
    assert unit.has_section("Unit")
    assert unit.has_section("Service")
    service = dict(unit.items("Service"))

    # Inspect actual parsed directives, not comments mentioning StateDirectory.
    assert "statedirectory" not in service
    assert "workingdirectory" not in service
    assert service["execstart"] == (
        "/opt/project-suzka/venv/bin/python3 -m suzka.api.server"
    )
    assert "uv run" not in service["execstart"]
    assert service["environmentfile"] == "/etc/project-suzka/backend.env"
    assert service["user"] == "suzka"
    assert service["group"] == "suzka"
    assert service["umask"] == "0077"
    assert service["runtimedirectory"] == "project-suzka"
    assert service["runtimedirectorymode"] == "0700"
    assert "DO NOT INSTALL/ACTIVATE" in UNIT_EXAMPLE.read_text(encoding="utf-8")


def test_missing_eval_input_is_an_explicit_future_activation_blocker(
    tmp_path: Path,
) -> None:
    # This characterizes existing behavior; I0 must not change the loader.
    assert load_eval_sets([tmp_path / "not-installed.json"]) == []
    docs = DOC.read_text(encoding="utf-8")
    assert "not activation-ready" in docs
    assert "W0-I1" in docs and "W0-I2" in docs
    assert "missing files" in docs


def test_example_unit_has_no_required_live_installation_dependencies() -> None:
    # The example names /opt and /etc, but unit parsing must not execute it.
    unit = _unit()
    assert unit.get("Service", "ExecStart").startswith("/")
    assert "StateDirectory=" in DOC.read_text(encoding="utf-8")


def _staged_config(tmp_path: Path) -> Path:
    config = deepcopy(_example())
    state_root = tmp_path / "isolated-state"
    state_root.mkdir(mode=0o700)

    for section, key in PATH_FIELDS:
        if (section, key) == ("adapter_registry", "eval_sets"):
            continue
        raw = str(config[section][key])
        assert raw.startswith(str(STATE_ROOT) + "/")
        config[section][key] = raw.replace(str(STATE_ROOT), str(state_root), 1)

    eval_file = tmp_path / "operator" / "eval.json"
    eval_file.parent.mkdir(mode=0o700)
    eval_file.write_text(
        json.dumps({"cases": [{"prompt": "synthetic", "expected": "ok"}]}),
        encoding="utf-8",
    )
    config["adapter_registry"]["eval_sets"] = [str(eval_file)]

    conf_dir = tmp_path / "isolated-etc"
    conf_dir.mkdir(mode=0o700)
    conf = conf_dir / "config.yaml"
    conf.write_text(yaml.safe_dump(config), encoding="utf-8")
    return conf


def test_synthetic_journal_reopens_same_inode_from_two_unrelated_cwds(
    tmp_path: Path,
) -> None:
    config_path = _staged_config(tmp_path)
    assert config_path.is_absolute()

    # Using the source tree on PYTHONPATH is a test-only bootstrap. A later
    # deployment gate must independently prove non-editable wheel packaging.
    child = """
import json
from suzka.config.settings import load_settings
from suzka.runtime.event_journal import EventJournal

settings = load_settings()
assert settings.event_journal.path.is_absolute()
journal = EventJournal(settings.event_journal.path, 1048576, 4)
try:
    inspection = journal.inspect()
    if not inspection.records:
        journal.verify_and_reconcile(0, "a" * 64)
    assert journal.inspect().snapshot_hash == "a" * 64
    st = settings.event_journal.path.stat()
    print(json.dumps({"path": str(settings.event_journal.path), "device": st.st_dev, "inode": st.st_ino}))
finally:
    journal.close()
"""

    results: list[dict[str, int | str]] = []
    for dirname in ("unrelated-cwd-1", "unrelated-cwd-2"):
        work_dir = tmp_path / dirname
        work_dir.mkdir()
        env = os.environ.copy()
        env["SUZKA_CONFIG_PATH"] = str(config_path)
        env["PYTHONPATH"] = str(ROOT)
        # Child imports must not create __pycache__ outside tmp_path.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        run = subprocess.run(
            [sys.executable, "-c", child],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        results.append(json.loads(run.stdout))
        assert not (work_dir / ".suzka").exists()

    expected_path = tmp_path / "isolated-state/journal/event_journal.jsonl"
    assert len(results) == 2 and results[0] == results[1]
    assert all(result["path"] == str(expected_path) for result in results)


def test_systemd_example_can_be_verified_with_test_interpreter(
    tmp_path: Path,
) -> None:
    # This is optional: CI need not have systemd-analyze. It only validates
    # unit syntax and never starts or installs a service.
    import shutil

    verifier = shutil.which("systemd-analyze")
    if verifier is None:
        pytest.skip("systemd-analyze is not installed")

    unit = UNIT_EXAMPLE.read_text(encoding="utf-8")
    unit = unit.replace(
        "/opt/project-suzka/venv/bin/python3 -m suzka.api.server",
        f"{Path(sys.executable).absolute()} -m suzka.api.server",
    )
    # Use a test-only identity, not the unavailable production service user.
    unit = unit.replace(
        "User=suzka\nGroup=suzka",
        f"User={os.getuid()}\nGroup={os.getgid()}",
    )
    unit_path = tmp_path / "suzka-api-fhs.service"
    unit_path.write_text(unit, encoding="utf-8")
    verified = subprocess.run(
        [verifier, "verify", str(unit_path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
