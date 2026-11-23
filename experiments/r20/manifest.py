"""The evidence manifest (D1 sec. 4-6).

``evidence/manifest.json`` lives in the SIBLING evidence directory, never in
the subject root and never in SubjectHistory.  Identifiers, digests, row names
and counts only; no raw private text.  The key set is closed.

``plan_hold_receipts``: the receipt writer (IP-14b-3) is a separate unit.  A HOLD
the harness observed is recorded ``HOLD_UNSEALED`` unless the caller names the
id of a receipt READ FROM SubjectHistory.  The manifest cannot verify that id
itself: the caller (22d) must read it from the store; a manifest copy of a log
is never a receipt (IP-14b HD-3 (a)).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments.r20.guards import GuardRefusal

ADMISSION_PATH = "HARNESS_BRIDGE"
HOLD_UNSEALED = "HOLD_UNSEALED"
HOLD_SEALED = "SEALED"
KEYS = (
    "run_id", "profile", "tier", "code_head", "config_sha256", "provider", "identity",
    "approval_actor", "admission_path", "fixture_only", "timeline", "policy_inputs",
    "plan_hold_receipts", "logical_cycle_resolutions", "ablation", "kill_points",
    "restarts", "journal", "results", "recorder", "destination",
)
_LISTS = frozenset(
    {
        "timeline", "policy_inputs", "plan_hold_receipts", "logical_cycle_resolutions",
        "ablation", "kill_points", "restarts", "recorder",
    }
)


class Manifest:
    def __init__(self, evidence_dir: Path, run_id: str, *, subject_root: Path | None = None) -> None:
        resolved = Path(os.path.realpath(evidence_dir))
        if subject_root is not None:
            root = Path(os.path.realpath(subject_root))
            if resolved == root or root in resolved.parents:
                raise GuardRefusal("EVIDENCE_INSIDE_SUBJECT")
        self.path = resolved / "manifest.json"
        self._data: dict[str, Any] = {key: [] if key in _LISTS else None for key in KEYS}
        self._data.update(
            {
                "run_id": run_id,
                "admission_path": ADMISSION_PATH,
                "fixture_only": True,
                "journal": {},
                "results": {},
            }
        )

    def set(self, key: str, value: Any) -> None:
        if key not in KEYS or key in _LISTS:
            raise KeyError(key)
        self._data[key] = value

    def append(self, key: str, entry: Any) -> None:
        if key not in _LISTS:
            raise KeyError(key)
        self._data[key].append(entry)

    def result(self, check: str, label: str) -> None:
        self._data["results"][check] = label

    def record_hold(self, row: str, *, receipt_id: str | None = None) -> dict[str, Any]:
        """A HOLD the harness observed.  Sealed only by a store receipt id."""

        entry = {
            "row": row,
            "outcome": HOLD_UNSEALED if receipt_id is None else HOLD_SEALED,
            "receipt_id": receipt_id,
        }
        self.append("plan_hold_receipts", entry)
        return entry

    def record_journal_bytes(self, *paths: Path) -> int:
        total = sum(Path(path).stat().st_size for path in paths if Path(path).exists())
        self._data["journal"]["live_bytes"] = total
        return total

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def write(self) -> Path:
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self._data, sort_keys=True, indent=1), encoding="utf-8")
        os.replace(temporary, self.path)
        return self.path


__all__ = ["ADMISSION_PATH", "HOLD_SEALED", "HOLD_UNSEALED", "KEYS", "Manifest"]
