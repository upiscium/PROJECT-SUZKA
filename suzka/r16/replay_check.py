"""IP-17a §12-2: the pre-migration replay check of persisted Decisions.

Before the v4 migration is proposed for a production store, every persisted
CommittedDecision's sealed SuccessorPolicy input is replayed with *this*
code's ``evaluate`` and compared with the row the Decision was issued with.
Row 2c (IP-17a) must not move any existing Decision: the check must report
zero row changes on a copy of the real store.

Read-only: the store is opened with SQLite ``mode=ro`` and only
``history_decisions`` is read (its columns are the same in schema v1..v4).
Nothing is migrated, written or created.  Run it on a copy::

    python -m suzka.r16.replay_check /path/to/copy/subject_history.sqlite3

The report is JSON on stdout: counts per policy kind, the replayed and
unchanged counts, and for each change or error only the Decision id, its
kind, the stored row and the replayed row (no policy input, no source).  The
exit status is 0 only when nothing changed and nothing failed to replay.

``SUCCESSOR_POLICY_V1`` Decisions are counted but not replayed: V1 inputs are
not SuccessorPolicy V2 / V3 bytes, and IP-17a changes V3 only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import sys
from typing import Final

from suzka.r16.history_codec import PolicyKind, PolicyRow
from suzka.r16.successor_policy import SuccessorPolicyInputV2, evaluate

_HOLD: Final = "HOLD"


@dataclass(frozen=True, slots=True)
class ReplayChange:
    decision_id: str
    policy_kind: str
    stored_row: str
    replayed_row: str


@dataclass(frozen=True, slots=True)
class ReplayReport:
    total: int
    by_kind: dict[str, int]
    replayed: int
    unchanged: int
    not_replayed_v1: int
    changes: tuple[ReplayChange, ...] = field(default=())
    errors: tuple[ReplayChange, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.changes and not self.errors

    def to_json(self) -> str:
        return json.dumps(
            {
                "by_kind": dict(sorted(self.by_kind.items())),
                "changes": [_change(item) for item in self.changes],
                "errors": [_change(item) for item in self.errors],
                "not_replayed_v1": self.not_replayed_v1,
                "ok": self.ok,
                "replayed": self.replayed,
                "total": self.total,
                "unchanged": self.unchanged,
            },
            sort_keys=True,
        )


def _change(item: ReplayChange) -> dict[str, str]:
    return {
        "decision_id": item.decision_id,
        "policy_kind": item.policy_kind,
        "replayed_row": item.replayed_row,
        "stored_row": item.stored_row,
    }


def replay_rows(rows: list[tuple[object, object, object, object]]) -> ReplayReport:
    """Replay ``(decision_id, policy_kind, policy_row, policy_input_bytes)``."""

    by_kind: Counter[str] = Counter()
    changes: list[ReplayChange] = []
    errors: list[ReplayChange] = []
    replayed = unchanged = v1 = 0
    for decision_id, kind, row, payload in rows:
        name = str(decision_id)
        kind_name = str(kind)
        stored = str(row)
        by_kind[kind_name] += 1
        if kind_name == PolicyKind.SUCCESSOR_POLICY_V1.value:
            v1 += 1
            continue
        try:
            if type(payload) is not bytes:
                raise ValueError("policy input is not a blob")
            sealed = SuccessorPolicyInputV2.from_bytes(payload)
            if sealed.version.value != kind_name:
                raise ValueError("sealed version is not the stored policy kind")
            result = evaluate(sealed)
            PolicyRow(stored)
        except (TypeError, ValueError) as error:
            errors.append(ReplayChange(name, kind_name, stored, f"ERROR:{type(error).__name__}"))
            continue
        replayed += 1
        now = _HOLD if result.is_hold else result.history_row.value
        if now == stored:
            unchanged += 1
        else:
            changes.append(ReplayChange(name, kind_name, stored, now))
    return ReplayReport(
        total=sum(by_kind.values()),
        by_kind=dict(by_kind),
        replayed=replayed,
        unchanged=unchanged,
        not_replayed_v1=v1,
        changes=tuple(changes),
        errors=tuple(errors),
    )


def replay_store(path: Path) -> ReplayReport:
    """Open ``path`` read-only and replay every ``history_decisions`` row."""

    resolved = path.expanduser().resolve(strict=True)
    uri = f"{resolved.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT decision_id, policy_kind, policy_row, policy_input_bytes "
            "FROM history_decisions ORDER BY issuer_sequence, decision_id"
        ).fetchall()
    finally:
        connection.close()
    return replay_rows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m suzka.r16.replay_check",
        description="Replay every persisted Decision's sealed policy input (read-only).",
    )
    parser.add_argument("store", type=Path, help="a COPY of subject_history.sqlite3")
    args = parser.parse_args(argv)
    report = replay_store(args.store)
    sys.stdout.write(report.to_json() + "\n")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["ReplayChange", "ReplayReport", "main", "replay_rows", "replay_store"]
