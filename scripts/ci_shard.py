"""Split the backend test files into balanced shards for CI.

``python scripts/ci_shard.py --shards 6 --index 2`` prints the test files of
shard 2 (0-based), one per line, for ``pytest $(...)``.

Files are assigned longest first to the currently lightest shard, using the
measured seconds in ``scripts/ci_test_durations.json``.  A test file that is
not in that file (a new one, or one that grew past the table) gets the median
of the known durations, so a stale table only makes the balance worse, never
drops or duplicates a file: every file under ``tests/`` that pytest collects
lands in exactly one shard.

``--update-from-log LOG`` rewrites the table from a ``pytest --durations=0``
log (see the module's last lines for how to produce one).

Pure standard library.  No test code is imported.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TABLE = Path(__file__).with_name("ci_test_durations.json")
# ``--durations`` lines: "12.34s call     tests/test_x.py::test_y".
_DURATION = re.compile(r"^\s*([0-9.]+)s\s+(?:setup|call|teardown)\s+(tests/\S+?)::")


def test_files() -> list[str]:
    """Every file pytest would collect from ``tests/`` by default, sorted.

    pytest's default ``python_files`` is ``test_*.py`` and ``*_test.py``; the
    repository has no pytest configuration that changes it.
    """

    found = {path for pattern in ("test_*.py", "*_test.py") for path in TESTS.rglob(pattern)}
    return sorted(path.relative_to(ROOT).as_posix() for path in found)


def load_table(path: Path = TABLE) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    return {str(key): float(value) for key, value in raw.items()}


def plan(files: list[str], table: dict[str, float], shards: int) -> list[list[str]]:
    """Longest-processing-time-first assignment; deterministic."""

    if shards < 1:
        raise ValueError("shards must be at least 1")
    known = [table[name] for name in files if name in table]
    default = statistics.median(known) if known else 1.0
    weight = {name: table.get(name, default) for name in files}
    bins: list[list[str]] = [[] for _ in range(shards)]
    load = [0.0] * shards
    for name in sorted(files, key=lambda item: (-weight[item], item)):
        target = min(range(shards), key=lambda index: (load[index], index))
        bins[target].append(name)
        load[target] += weight[name]
    return [sorted(group) for group in bins]


def update_table(log: Path, path: Path = TABLE) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for line in log.read_text().splitlines():
        match = _DURATION.match(line)
        if match:
            totals[match.group(2)] += float(match.group(1))
    # Files whose tests are all under --durations-min are small: keep a floor.
    table = {name: round(max(totals.get(name, 0.0), 1.0), 1) for name in test_files()}
    path.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n")
    return table


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--update-from-log", type=Path)
    parser.add_argument("--summary", action="store_true", help="print each shard's estimated seconds")
    args = parser.parse_args(argv)
    if args.update_from_log is not None:
        table = update_table(args.update_from_log)
        print(f"{len(table)} files, {sum(table.values()):.0f} s in total")
        return 0
    files = test_files()
    groups = plan(files, load_table(), args.shards)
    if args.summary:
        table = load_table()
        known = [table[name] for name in files if name in table]
        default = statistics.median(known) if known else 1.0
        for index, group in enumerate(groups):
            seconds = sum(table.get(name, default) for name in group)
            print(f"shard {index}: {len(group)} files, ~{seconds:.0f} s")
        return 0
    if not 0 <= args.index < args.shards:
        parser.error("index must be in [0, shards)")
    print("\n".join(groups[args.index]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# To refresh the table (about 20 minutes, on a quiet machine):
#   uv run pytest -q -p no:cacheprovider --durations=0 --durations-min=0 > durations.log
#   python scripts/ci_shard.py --update-from-log durations.log
