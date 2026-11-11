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


def test_random_appends_and_rotations_match_fresh_verification(
    tmp_path: Path,
) -> None:
    """Seeded property test: after every step the cached read equals a fresh one."""
    import random

    rng = random.Random(20261110)
    path = tmp_path / "events.jsonl"
    value = _journal(path, max_bytes=6_000, retained=200)
    value.verify_and_reconcile(0, HASH_0)
    number = 1
    for _ in range(25):
        count = rng.randint(1, 3)
        _populate(value, number, count)
        number += count
        cached = value.records
        observer = EventJournal.__new__(EventJournal)
        # Full verification of the same bytes without taking the lease: reuse
        # the journal's own reader with an empty cache.
        observer.path = value.path
        observer._lock = value._lock
        observer._segment_cache = {}
        with value._lock:
            fresh = observer._read_records_unlocked()
        assert cached == fresh
        assert value._verify_records(cached) == _fresh_verification(fresh)
    assert sorted(tmp_path.glob("events.jsonl.[0-9]*"))
    value.close()


# --- IP-2b: resumable _verify_records state -------------------------------


def _fresh_verification(records):  # type: ignore[no-untyped-def]
    """The reference: a full verification with no cached verifier state."""
    return EventJournal._verify_records_from(records, None)[0]


def _outcome(call):  # type: ignore[no-untyped-def]
    try:
        return ("ok", call())
    except EventJournalIntegrityError as error:
        return ("error", str(error))


def _observer(value: EventJournal) -> EventJournal:
    observer = EventJournal.__new__(EventJournal)
    observer._lock = value._lock
    observer._verifier_state = None
    return observer


def _transaction_journal(path: Path) -> EventJournal:
    from tests.test_event_journal import (
        HASH_2,
        PARTICIPANTS,
        TX_ID,
        bootstrap_v3,
        prepared_transaction,
    )
    from suzka.runtime.event_journal import ParticipantOutcome

    value = bootstrap_v3(path)
    item = prepared_transaction(value)
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[0], HASH_1, ParticipantOutcome.FINALIZED
    )
    value.append_participant_finalized(
        item, TX_ID, PARTICIPANTS[1], HASH_2, ParticipantOutcome.ALREADY_CONSISTENT
    )
    value.append_transaction_completed(item, TX_ID)
    anchor = value.records[0]
    value.append_completed(
        item,
        1,
        HASH_1,
        anchor.wal_generation_id,
        anchor.wal_record_id,
        anchor.wal_record_hash,
    )
    return value


def _gate_clear_journal(path: Path) -> EventJournal:
    from tests.test_event_journal import (
        append_u5_clear,
        append_u5_reconciliation,
        startup_reconciliation_journal,
    )

    value = startup_reconciliation_journal(path)
    append_u5_reconciliation(value)
    append_u5_clear(value)
    append_u5_clear(value, terminal=True)
    return value


@pytest.mark.parametrize(
    "build", [_transaction_journal, _gate_clear_journal], ids=["tx", "gate"]
)
def test_every_prefix_matches_fresh_verification(tmp_path: Path, build) -> None:  # type: ignore[no-untyped-def]
    """Walk prefixes in order, then backwards: each result equals a full one."""
    value = build(tmp_path / "events.jsonl")
    records = value.records
    observer = _observer(value)
    order = [*range(1, len(records) + 1), *range(len(records), 0, -1)]
    for length in order:
        prefix = records[:length]
        cached = _outcome(lambda: observer._verify_records(prefix))
        assert cached == _outcome(lambda: _fresh_verification(prefix)), length
    value.close()


def test_verifier_processes_only_the_new_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 5)
    value._verify_records(value.records)
    started: list[int] = []
    original = EventJournal._initial_verifier_state

    def counting(records):  # type: ignore[no-untyped-def]
        started.append(len(records))
        return original(records)

    monkeypatch.setattr(EventJournal, "_initial_verifier_state", staticmethod(counting))
    _populate(value, 6, 3)
    result = value._verify_records(value.records)
    assert started == []
    assert result == _fresh_verification(value.records)
    value.close()


def test_rejected_candidate_clears_verifier_state(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 2)
    records = value.records
    value._verify_records(records)
    duplicate = records[-1]
    with pytest.raises(EventJournalIntegrityError):
        value._verify_records((*records, duplicate))
    assert value._verifier_state is None
    assert value._verify_records(records) == _fresh_verification(records)
    value.close()


def test_shorter_or_divergent_input_never_reuses_a_longer_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 3)
    records = value.records
    full = value._verify_records(records)
    shorter = records[:-2]
    assert value._verify_records(shorter) == _fresh_verification(shorter)
    assert value._verify_records(shorter) != full
    divergent = (*records[:-1], records[-1].model_copy(update={"record_id": "x"}))
    assert _outcome(lambda: value._verify_records(divergent)) == _outcome(
        lambda: _fresh_verification(divergent)
    )
    value.close()


def test_earlier_results_are_not_mutated_by_later_verification(
    tmp_path: Path,
) -> None:
    value = _transaction_journal(tmp_path / "events.jsonl")
    records = value.records
    observer = _observer(value)
    early = observer._verify_records(records[:-2])
    snapshot = _fresh_verification(records[:-2])
    observer._verify_records(records)
    assert early == snapshot
    value.close()


def test_invalidation_also_drops_verifier_state(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = _journal(path)
    value.verify_and_reconcile(0, HASH_0)
    _populate(value, 1, 2)
    value._verify_records(value.records)
    assert value._verifier_state is not None
    value.invalidate_verification_cache()
    assert value._verifier_state is None
    value.close()
