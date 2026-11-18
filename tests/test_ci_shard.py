"""The CI shard planner covers every test file exactly once, for any table."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ci_shard", Path(__file__).resolve().parents[1] / "scripts" / "ci_shard.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ci_shard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_shard)


@pytest.mark.parametrize("shards", [1, 2, 5, 8, 40])
def test_every_file_lands_in_exactly_one_shard(shards: int) -> None:
    files = ci_shard.test_files()
    groups = ci_shard.plan(files, ci_shard.load_table(), shards)
    flat = [name for group in groups for name in group]
    assert sorted(flat) == files
    assert len(flat) == len(set(flat))


def test_a_missing_or_stale_table_still_covers_every_file() -> None:
    files = ci_shard.test_files()
    for table in ({}, {files[0]: 900.0}, {"tests/test_gone.py": 5.0}):
        groups = ci_shard.plan(files, table, 4)
        assert sorted(name for group in groups for name in group) == files


def test_the_plan_is_deterministic_and_balanced() -> None:
    files = [f"tests/test_{index}.py" for index in range(20)]
    table = {name: float(index + 1) for index, name in enumerate(files)}
    first = ci_shard.plan(files, table, 4)
    assert first == ci_shard.plan(files, table, 4)
    loads = [sum(table[name] for name in group) for group in first]
    assert max(loads) - min(loads) <= max(table.values())


def test_zero_shards_is_refused() -> None:
    with pytest.raises(ValueError):
        ci_shard.plan(["tests/test_a.py"], {}, 0)


def test_the_plan_covers_exactly_the_files_pytest_collects() -> None:
    """A test file pytest collects but no shard runs would escape CI."""

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = {
        line.split("::", 1)[0] for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line
    }
    assert collected
    assert collected == set(ci_shard.test_files())


def test_the_duration_log_is_parsed_into_the_table(tmp_path: Path) -> None:
    log = tmp_path / "durations.log"
    log.write_text(
        "43.19s call     tests/test_ci_shard.py::test_a\n"
        "2.50s setup    tests/test_ci_shard.py::test_a[param-1]\n"
        "0.60s call     tests/conftest.py::not_a_test_file_but_matches\n"
        "noise line\n"
    )
    table = ci_shard.update_table(log, tmp_path / "table.json")
    assert table["tests/test_ci_shard.py"] == pytest.approx(45.7, abs=0.1)
    assert (tmp_path / "table.json").exists()
    assert ci_shard.load_table(tmp_path / "table.json") == table
    assert ci_shard.load_table(tmp_path / "missing.json") == {}


def test_the_index_must_be_inside_the_shard_count() -> None:
    with pytest.raises(SystemExit):
        ci_shard.main(["--shards", "4", "--index", "4"])
