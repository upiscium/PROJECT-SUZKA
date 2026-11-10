"""IP-2 (#383 / #381 D2 P1-1): in-process incremental segment verification.

The cache must be an optimisation only: every read must be identical to a
fresh full verification, any change to already-verified bytes of a segment
must fall back to a full re-read, rotation (rename) reuses the verified parse
of the renamed segment, and a fresh EventJournal always verifies everything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalIntegrityError,
    EventJournalLoadError,
)

from tests.test_event_journal import HASH_0, HASH_1, NOW, append_success, event


def _journal(path: Path, max_bytes: int = 100_000, retained: int = 50) -> EventJournal:
    return EventJournal(path, max_bytes, retained, clock=lambda: NOW)


def _populate(value: EventJournal, start: int, count: int) -> None:
    hashes = (HASH_0, HASH_1)
    for number in range(start, start + count):
        append_success(
            value,
            event(f"cache-{number}", number),
            hashes[(number - 1) % 2],
            hashes[number % 2],
        )


def _fresh_records(path: Path, value: EventJournal, **kwargs):  # type: ignore[no-untyped-def]
    value.close()
    reopened = _journal(path, **kwargs)
    try:
        return reopened.records
    finally:
        reopened.close()


def _counting_parser(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    parsed: list[int] = []
    original = EventJournal._parse_segment_bytes

    def counting(self, raw):  # type: ignore[no-untyped-def]
        parsed.append(len(raw.splitlines()))
        return original(self, raw)

    monkeypatch.setattr(EventJournal, "_parse_segment_bytes", counting)
    return parsed


def test_cached_reads_equal_fresh_full_verification(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 12)
    cached = value.records
    assert cached == value.records
    assert cached == _fresh_records(path, value)


def test_only_appended_lines_are_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 5)
    value.records
    parsed = _counting_parser(monkeypatch)
    value.records
    assert parsed == []
    # Appends read the Journal internally; across the whole append each new
    # line (accepted, started, prepared, completed) is parsed exactly once.
    _populate(value, 6, 1)
    value.records
    assert sum(parsed) == 4
    value.close()


def test_fresh_instance_parses_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 3)
    total = len(value.records)
    value.close()
    parsed = _counting_parser(monkeypatch)
    reopened = _journal(path)
    assert sum(parsed) == total
    reopened.close()


def test_rotation_reuses_the_renamed_segment_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path, max_bytes=4_000, retained=50)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 6)
    before_rotation = len(value.records)
    parsed = _counting_parser(monkeypatch)
    _populate(value, 7, 6)
    assert sorted(tmp_path.glob("events.jsonl.[0-9]*"))
    records = value.records
    # Re-reading after rotation must not re-parse the renamed segments in full.
    assert sum(parsed) < len(records)
    assert len(records) > before_rotation
    monkeypatch.undo()
    assert records == _fresh_records(path, value, max_bytes=4_000, retained=50)


def _rewrite_line_same_size(path: Path, index: int) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[index])
    original = record["event_id"]
    record["event_id"] = original[:-1] + ("0" if original[-1] != "0" else "1")
    tampered = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    tampered += b"\n"
    if len(tampered) != len(lines[index]):
        pytest.skip("canonical re-encoding changed the line length")
    with path.open("r+b") as handle:
        handle.seek(sum(len(line) for line in lines[:index]))
        handle.write(tampered)


def test_rewrite_of_last_verified_line_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 2)
    value.records
    count = len(path.read_bytes().splitlines())
    _rewrite_line_same_size(path, count - 1)
    with pytest.raises((EventJournalIntegrityError, EventJournalLoadError)):
        value.records
    value.close()


def test_deep_same_size_rewrite_is_detected_by_full_verification(
    tmp_path: Path,
) -> None:
    """Documented threat model: single cooperative writer."""
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 2)
    value.records
    _rewrite_line_same_size(path, 2)
    value.invalidate_verification_cache()
    with pytest.raises((EventJournalIntegrityError, EventJournalLoadError)):
        value.records
    value.close()
    with pytest.raises((EventJournalIntegrityError, EventJournalLoadError)):
        _journal(path)


def test_truncation_falls_back_to_full_read(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 2)
    value.records
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:5]))
    records = value.records
    assert len(records) == 5
    assert records == _fresh_records(path, value)


def test_partial_tail_is_rejected_and_cache_is_not_advanced(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 2)
    expected = value.records
    good = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"schema_version":3')
    with pytest.raises(EventJournalLoadError):
        value.records
    path.write_bytes(good)
    assert value.records == expected
    value.close()


def test_tampered_appended_record_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 1)
    value.records
    last = json.loads(path.read_bytes().splitlines()[-1])
    last["record_hash"] = "f" * 64
    with path.open("ab") as handle:
        handle.write(json.dumps(last, sort_keys=True).encode() + b"\n")
    with pytest.raises((EventJournalIntegrityError, EventJournalLoadError)):
        value.records
    value.close()


def test_replaced_inode_is_read_in_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 2)
    total = len(value.records)
    replacement = path.with_name("replacement.tmp")
    replacement.write_bytes(path.read_bytes())
    os.chmod(replacement, 0o600)
    os.replace(replacement, path)
    parsed = _counting_parser(monkeypatch)
    assert len(value.records) == total
    assert sum(parsed) == total
    value.close()
