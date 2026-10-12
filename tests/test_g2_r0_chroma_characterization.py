"""G2-R0 *test-only* ChromaDB source-effects characterization.

All Chroma files are created below pytest's private tmp_path. These tests do
NOT construct a G2 source owner, a ContextCaptureParticipant, an epoch issuer
or SourceSetVerified. They cannot establish production DB1 source authority.
The path marker check prevents accidental non-fixture selection only; it does
NOT resist an adversarial same-UID process swapping filesystem paths between
validation and Chroma open (outside the selected R15 threat model).
The acceptance contract is #288/#294; physical file writes are diagnostics,
not an automatic failure when the complete relevant logical source is stable.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import chromadb
from chromadb.config import Settings
import pytest

_COLLECTION = "g2r0_synthetic_two_source_records"
_ROOT = "g2r0-synthetic-chroma"
_MARKER = ".g2r0-synthetic-fixture"
_TARGET = "synthetic-episode-E"
_SENTINEL = "synthetic-episode-E2"

_RECORDS: dict[str, dict[str, Any]] = {
    _TARGET: {
        "document": "Synthetic episode E: no personal information",
        "metadata": {"kind": "synthetic-original-E", "epoch": 0},
        "embedding": [0.125, 0.25, -0.5],
    },
    _SENTINEL: {
        "document": "Synthetic sentinel E2: no personal information",
        "metadata": {"kind": "synthetic-original-E2", "epoch": 0},
        "embedding": [0.75, -0.25, 0.5],
    },
}


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / _ROOT
    root.mkdir()
    (root / _MARKER).write_text("G2-R0 synthetic-only source fixture", encoding="utf-8")
    return root


def _require_fixture(root: Path) -> None:
    """Best-effort accidental-path guard, NOT a concurrent filesystem boundary.

    Requires pytest-owned tmp_path custody. Chroma accepts a pathname, not an
    O_NOFOLLOW directory descriptor; hostile same-UID rename/symlink swaps
    between this check and the backend open are explicitly NOT prevented.
    """
    if (
        root.name != _ROOT
        or root.is_symlink()
        or not root.is_dir()
        or not (root / _MARKER).is_file()
        or (root / _MARKER).is_symlink()
    ):
        raise RuntimeError("G2-R0 may access synthetic pytest fixtures only")


def _collection(root: Path, *, seed: bool = False) -> Any:
    _require_fixture(root)
    client = chromadb.PersistentClient(
        path=str(root),
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )
    collection = (
        client.get_or_create_collection(_COLLECTION)
        if seed
        else client.get_collection(_COLLECTION)
    )
    if seed:
        collection.add(
            ids=list(_RECORDS),
            documents=[item["document"] for item in _RECORDS.values()],
            metadatas=[item["metadata"] for item in _RECORDS.values()],
            embeddings=[item["embedding"] for item in _RECORDS.values()],
        )
    return collection


def _projection(collection: Any, *, ids: list[str] | None = None) -> dict[str, Any]:
    """Compare the entire synthetic logical source, not only a selected row."""
    result = collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
    vectors = result["embeddings"]
    if vectors is None:
        raise AssertionError("synthetic Chroma embeddings were not returned")
    rows = {
        record_id: {
            "document": result["documents"][index],
            "metadata": result["metadatas"][index],
            # Chroma stores floats in its own representation: use the actual
            # stored numbers, not the Python binary64 inputs.
            "embedding": [float(value) for value in vectors[index]],
        }
        for index, record_id in enumerate(result["ids"])
    }
    if ids is None:
        return {"count": collection.count(), "rows": rows}
    return {"count": len(rows), "rows": rows}


def _digest(data: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            data, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _child(root: Path, action: str) -> dict[str, Any]:
    _require_fixture(root)
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--g2r0-worker",
            action,
            str(root),
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    lines = [
        line.removeprefix("G2R0_SYNTHETIC_RESULT=")
        for line in child.stdout.splitlines()
        if line.startswith("G2R0_SYNTHETIC_RESULT=")
    ]
    if len(lines) != 1:
        raise AssertionError("synthetic Chroma child returned no unambiguous result")
    return json.loads(lines[0])


def _worker(action: str, root: Path) -> dict[str, Any]:
    if action not in {"seed", "reopen"}:
        raise RuntimeError("unsupported synthetic fixture action")
    collection = _collection(root, seed=action == "seed")
    if action == "reopen":
        # Deliberately trigger an ordinary point lookup on a fresh process.
        assert _projection(collection, ids=[_TARGET])["rows"][_TARGET]["document"]
    return _projection(collection)


def test_two_row_cold_reopens_preserve_full_synthetic_source(tmp_path: Path) -> None:
    """Whole-fixture comparison does not reject solely on file hash changes."""
    root = _fixture_root(tmp_path)
    baseline = _child(root, "seed")
    assert baseline["count"] == 2
    assert set(baseline["rows"]) == {_TARGET, _SENTINEL}
    assert all("Synthetic" in row["document"] for row in baseline["rows"].values())
    assert _child(root, "reopen") == baseline
    assert _child(root, "reopen") == baseline


def test_non_target_same_count_change_is_detected(tmp_path: Path) -> None:
    """A target-only or count-only check misses collateral source mutations."""
    root = _fixture_root(tmp_path)
    before = _child(root, "seed")
    collection = _collection(root)
    collection.update(
        ids=[_SENTINEL],
        documents=["Synthetic sentinel E2: explicitly mutated by fixture owner"],
        metadatas=[_RECORDS[_SENTINEL]["metadata"]],
        embeddings=[_RECORDS[_SENTINEL]["embedding"]],
    )
    after = _projection(collection)
    assert before["count"] == after["count"] == 2
    assert before["rows"][_TARGET] == after["rows"][_TARGET]
    assert before["rows"][_SENTINEL] != after["rows"][_SENTINEL]
    assert _digest(before) != _digest(after)


def test_interleaved_gets_do_not_provide_a_snapshot_epoch(tmp_path: Path) -> None:
    """Mixed old/new separately read rows need not form a coherent source set."""
    root = _fixture_root(tmp_path)
    before = _child(root, "seed")
    collection = _collection(root)
    old_target = _projection(collection, ids=[_TARGET])["rows"][_TARGET]
    collection.update(
        ids=list(_RECORDS),
        documents=["Synthetic E after update", "Synthetic E2 after update"],
        metadatas=[
            {"kind": "synthetic-original-E", "epoch": 1},
            {"kind": "synthetic-original-E2", "epoch": 1},
        ],
        embeddings=[item["embedding"] for item in _RECORDS.values()],
    )
    new_sentinel = _projection(collection, ids=[_SENTINEL])["rows"][_SENTINEL]
    after = _projection(collection)
    mixed = {"count": 2, "rows": {_TARGET: old_target, _SENTINEL: new_sentinel}}
    assert mixed != before
    assert mixed != after
    assert mixed["rows"][_TARGET]["metadata"]["epoch"] == 0
    assert mixed["rows"][_SENTINEL]["metadata"]["epoch"] == 1


def test_restoration_and_same_id_recreation_require_an_owner_incarnation(
    tmp_path: Path,
) -> None:
    """Identical logical bytes cannot certify no mutation/retraction occurred."""
    root = _fixture_root(tmp_path)
    before = _child(root, "seed")
    collection = _collection(root)
    collection.update(
        ids=[_TARGET],
        documents=["Synthetic E: changed, then restored"],
        metadatas=[{"kind": "synthetic-original-E", "epoch": 1}],
        embeddings=[_RECORDS[_TARGET]["embedding"]],
    )
    assert _projection(collection) != before
    collection.update(
        ids=[_TARGET],
        documents=[_RECORDS[_TARGET]["document"]],
        metadatas=[_RECORDS[_TARGET]["metadata"]],
        embeddings=[_RECORDS[_TARGET]["embedding"]],
    )
    assert _projection(collection) == before
    collection.delete(ids=[_TARGET])
    collection.add(
        ids=[_TARGET],
        documents=[_RECORDS[_TARGET]["document"]],
        metadatas=[_RECORDS[_TARGET]["metadata"]],
        embeddings=[_RECORDS[_TARGET]["embedding"]],
    )
    assert _projection(collection) == before


def test_unmarked_directory_cannot_become_a_test_chroma_owner(tmp_path: Path) -> None:
    """Catch accidental path selection, not malicious concurrent path swaps."""
    root = tmp_path / _ROOT
    root.mkdir()
    with pytest.raises(RuntimeError, match="synthetic pytest fixtures only"):
        _collection(root, seed=True)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("bad_id", ["", "..", "../outside", "not/opaque", "日本語"])
def test_existing_identifier_validation_rejects_unsafe_source_ids(bad_id: str) -> None:
    """Existing owner-neutral grammar can be reused only after approved G2 design."""
    from suzka.identifiers import validate_identifier

    with pytest.raises((TypeError, ValueError)):
        validate_identifier(bad_id)


if __name__ == "__main__" and sys.argv[1:2] == ["--g2r0-worker"]:
    if len(sys.argv) != 4:
        raise SystemExit("invalid synthetic-only worker invocation")
    _, _, action, root_arg = sys.argv
    print("G2R0_SYNTHETIC_RESULT=" + json.dumps(_worker(action, Path(root_arg))))
