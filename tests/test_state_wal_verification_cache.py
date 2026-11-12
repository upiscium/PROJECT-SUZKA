"""IP-1 (#383 / #382 C): in-process incremental verification of StateWAL.

The cache must be an optimisation only: every inspection must be identical to
a fresh full verification, any change to already-verified bytes must fall back
to full verification, and a fresh instance always verifies everything.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from uuid import uuid4

import pytest

from suzka.runtime.state_wal import RecoveryReason, StateWAL, StateWALIntegrityError

from tests.test_state_wal import make_snapshot


def _bootstrap(tmp_path: Path, sequence: int = 10) -> StateWAL:
    wal = StateWAL(tmp_path / "wal")
    wal.bootstrap(make_snapshot(sequence), sequence)
    return wal


def _append(wal: StateWAL, before: int, after: int, *, value: float = 0.1) -> None:
    wal.append_transition(
        event_id=uuid4(),
        event_type="state.transition",
        event_source="test",
        processing_sequence=after,
        prior_snapshot=make_snapshot(before, value=value),
        candidate_snapshot=make_snapshot(after, value=value),
    )


def _generation_path(wal: StateWAL) -> Path:
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    return wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"


def _fresh(wal: StateWAL) -> StateWAL:
    """A new instance has no cache, i.e. it performs a full verification."""
    return StateWAL(wal.root)


def test_cached_inspection_equals_full_verification_under_random_appends(
    tmp_path: Path,
) -> None:
    rng = random.Random(20261110)
    wal = _bootstrap(tmp_path)
    sequence = 10
    for _ in range(40):
        step = rng.randint(1, 3)
        _append(wal, sequence, sequence + step)
        sequence += step
        if rng.random() < 0.5:
            assert wal.inspect() == _fresh(wal).inspect()
    assert wal.inspect() == _fresh(wal).inspect()
    assert wal.inspect().latest_snapshot_sequence == sequence


def test_cache_verifies_only_appended_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wal = _bootstrap(tmp_path)
    for before in range(10, 20):
        _append(wal, before, before + 1)
    wal.inspect()
    verified: list[int] = []
    original = StateWAL._verify_generation_lines

    def counting(self, manifest, lines, scan):  # type: ignore[no-untyped-def]
        verified.append(len(lines))
        return original(self, manifest, lines, scan)

    monkeypatch.setattr(StateWAL, "_verify_generation_lines", counting)
    wal.inspect()
    assert verified == [0]
    _append(wal, 20, 21)
    verified.clear()
    wal.inspect()
    assert verified == [1]


def test_fresh_instance_always_verifies_the_whole_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wal = _bootstrap(tmp_path)
    for before in range(10, 15):
        _append(wal, before, before + 1)
    verified: list[int] = []
    original = StateWAL._verify_generation_lines

    def counting(self, manifest, lines, scan):  # type: ignore[no-untyped-def]
        verified.append(len(lines))
        return original(self, manifest, lines, scan)

    monkeypatch.setattr(StateWAL, "_verify_generation_lines", counting)
    _fresh(wal).inspect()
    assert verified == [6]


def _rewrite_line_same_size(path: Path, index: int) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[index])
    record["candidate_snapshot"]["emotion_state"]["valence"] = 0.2
    tampered = (
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    assert len(tampered) == len(lines[index])
    with path.open("r+b") as handle:  # same inode, same size
        handle.seek(sum(len(line) for line in lines[:index]))
        handle.write(tampered)


def test_rewrite_of_last_verified_line_is_detected_by_cached_path(
    tmp_path: Path,
) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12, value=0.1)
    _append(wal, 12, 13, value=0.1)
    wal.inspect()
    _rewrite_line_same_size(_generation_path(wal), 2)
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_deep_same_size_rewrite_is_detected_by_full_verification(
    tmp_path: Path,
) -> None:
    """Documented threat model: single cooperative writer.

    A same-size rewrite of an earlier verified line that keeps the last line
    intact is not re-read by the cached path; any full verification (process
    start, explicit invalidation) detects it.
    """
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12, value=0.1)
    _append(wal, 12, 13, value=0.1)
    wal.inspect()
    _rewrite_line_same_size(_generation_path(wal), 1)
    with pytest.raises(StateWALIntegrityError):
        _fresh(wal).inspect()
    wal.invalidate_verification_cache()
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_tampered_appended_suffix_is_detected(tmp_path: Path) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    wal.inspect()
    path = _generation_path(wal)
    last = json.loads(path.read_bytes().splitlines()[-1])
    forged = dict(last)
    forged["record_id"] = str(uuid4())
    forged["event_id"] = str(uuid4())
    with path.open("ab") as handle:
        handle.write(json.dumps(forged, sort_keys=True).encode() + b"\n")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()


def test_truncation_falls_back_to_full_verification(tmp_path: Path) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    _append(wal, 12, 13)
    wal.inspect()
    path = _generation_path(wal)
    data = path.read_bytes()
    keep = b"".join(data.splitlines(keepends=True)[:2])
    path.write_bytes(keep)
    inspection = wal.inspect()
    assert inspection == _fresh(wal).inspect()
    assert inspection.latest_snapshot_sequence == 12


def test_partial_tail_is_rejected_and_does_not_advance_cache(tmp_path: Path) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    wal.inspect()
    path = _generation_path(wal)
    good = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"record_type":"transition"')
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()
    path.write_bytes(good)
    assert wal.inspect() == _fresh(wal).inspect()


def test_replaced_inode_falls_back_to_full_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    wal.inspect()
    path = _generation_path(wal)
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(path.read_bytes())
    os.chmod(replacement, 0o600)
    os.replace(replacement, path)
    verified: list[int] = []
    original = StateWAL._verify_generation_lines

    def counting(self, manifest, lines, scan):  # type: ignore[no-untyped-def]
        verified.append(len(lines))
        return original(self, manifest, lines, scan)

    monkeypatch.setattr(StateWAL, "_verify_generation_lines", counting)
    wal.inspect()
    assert verified == [2]


def test_generation_switch_is_observed_through_the_manifest(tmp_path: Path) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    first = wal.inspect()
    second_manifest = wal.begin_generation(
        make_snapshot(5), 5, reason=RecoveryReason.TRUE_ROLLBACK
    )
    second = wal.inspect()
    assert second.active_manifest == second_manifest
    assert second.active_manifest != first.active_manifest
    assert second.latest_snapshot_sequence == 5
    assert second == _fresh(wal).inspect()


def test_explicit_invalidation_forces_full_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    wal.inspect()
    wal.invalidate_verification_cache()
    verified: list[int] = []
    original = StateWAL._verify_generation_lines

    def counting(self, manifest, lines, scan):  # type: ignore[no-untyped-def]
        verified.append(len(lines))
        return original(self, manifest, lines, scan)

    monkeypatch.setattr(StateWAL, "_verify_generation_lines", counting)
    wal.inspect()
    assert verified == [2]


def test_failed_verification_clears_cache(tmp_path: Path) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    wal.inspect()
    path = _generation_path(wal)
    good = path.read_bytes()
    lines = good.splitlines(keepends=True)
    path.write_bytes(lines[0] + b"garbage\n")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()
    assert wal._verified is None
    path.write_bytes(good)
    assert wal.inspect() == _fresh(wal).inspect()


def test_reconstruct_always_verifies_fully_and_detects_deep_rewrite(
    tmp_path: Path,
) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12, value=0.1)
    _append(wal, 12, 13, value=0.1)
    wal.inspect()  # warm the cache
    assert wal.reconstruct(sequence=12) == make_snapshot(12, value=0.1)
    _rewrite_line_same_size(_generation_path(wal), 1)
    with pytest.raises(StateWALIntegrityError):
        wal.reconstruct(sequence=12)


def test_custom_snapshot_hash_matches_full_verification(tmp_path: Path) -> None:
    def custom(snapshot):  # type: ignore[no-untyped-def]
        return "c" * 63 + str(snapshot.last_processed_event_sequence % 10)

    wal = StateWAL(tmp_path / "wal", snapshot_hash=custom)
    wal.bootstrap(make_snapshot(10), 10)
    _append(wal, 10, 12)
    _append(wal, 12, 13)
    cached = wal.inspect()
    full = StateWAL(tmp_path / "wal", snapshot_hash=custom).inspect()
    assert cached == full
    assert cached.latest_snapshot_hash == custom(make_snapshot(13))


def test_valid_line_followed_by_broken_line_does_not_advance_cache(
    tmp_path: Path,
) -> None:
    wal = _bootstrap(tmp_path)
    _append(wal, 10, 12)
    wal.inspect()
    path = _generation_path(wal)
    good = path.read_bytes()
    other = StateWAL(wal.root)
    _append(other, 12, 13)  # a valid appended line written by another handle
    valid_suffix = path.read_bytes()[len(good) :]
    with path.open("ab") as handle:
        handle.write(b"garbage\n")
    with pytest.raises(StateWALIntegrityError):
        wal.inspect()
    assert wal._verified is None
    path.write_bytes(good + valid_suffix)
    assert wal.inspect() == _fresh(wal).inspect()
    assert wal.inspect().latest_snapshot_sequence == 13
