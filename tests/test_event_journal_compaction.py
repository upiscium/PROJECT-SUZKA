"""IP-4a (#383 / #381 D1, D2): EventJournal COMPACTION and archive.

Old segments past the retention window are archived (compressed, never
deleted), and the latest COMPACTION declares the verifier state at the oldest
retained checkpoint. Archives are not authority: a missing or damaged archive
raises an alert, never a fail-stop.
"""

from __future__ import annotations

from datetime import timedelta
import gzip
import os
from pathlib import Path
import stat
from uuid import uuid4

import pytest

from suzka.runtime.event_journal import (
    EventJournal,
    EventJournalIntegrityError,
    EventLifecycle,
)

from tests.test_event_journal import HASH_0, HASH_1, HASH_2, NOW, event
from tests.test_event_journal_v4_rollover import _commit


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self):  # type: ignore[no-untyped-def]
        return self.now

    def advance(self, **delta: float) -> None:
        self.now = self.now + timedelta(**delta)


def _journal(path: Path, clock: _Clock, max_bytes: int = 3_000) -> EventJournal:
    from uuid import NAMESPACE_URL, uuid5

    value = EventJournal(path, max_bytes, 2, clock=clock)
    if not value.has_history:
        value.append_v2_bootstrap_checkpoint(
            0,
            HASH_0,
            str(uuid5(NAMESPACE_URL, "compaction-generation")),
            str(uuid5(NAMESPACE_URL, "compaction-wal")),
            HASH_1,
        )
        value.append_v3_migration_checkpoint()
        value.append_v4_migration_checkpoint()
    return value


def _fill(value: EventJournal, start: int, count: int) -> int:
    hashes = (HASH_0, HASH_1, HASH_2)
    for number in range(start, start + count):
        _commit(
            value, f"e{number}", number, hashes[(number - 1) % 3], hashes[number % 3]
        )
    return start + count


def _segments(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.[0-9]*"))


def _compacted(tmp_path: Path):  # type: ignore[no-untyped-def]
    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    following = _fill(value, 1, 8)
    assert len(_segments(path)) >= 2
    clock.advance(days=31)
    return path, clock, value, following


# --- the happy path ------------------------------------------------------------


def test_compaction_archives_old_segments_and_keeps_state(tmp_path: Path) -> None:
    path, clock, value, following = _compacted(tmp_path)
    before = value.inspect()
    segments_before = _segments(path)
    outcome = value.compact(retention_days=30)
    assert outcome.performed, outcome.reason
    assert outcome.archived_segments == len(segments_before)
    after = value.inspect()
    assert after.processing_high_water == before.processing_high_water
    assert after.snapshot_hash == before.snapshot_hash
    assert after.wal_generation_id == before.wal_generation_id
    assert after.latest_compaction is not None
    assert value.records[0].record_hash == (
        after.latest_compaction.boundary_checkpoint_hash
    )
    archive = value.archive_dir / outcome.archive_file  # type: ignore[operator]
    assert stat.S_IMODE(archive.stat().st_mode) == 0o400
    assert gzip.decompress(archive.read_bytes())
    assert after.latest_compaction.archived_record_count == outcome.archived_records
    # The Journal keeps working, and a fresh process verifies it fully.
    _fill(value, following, 2)
    value.close()
    reopened = _journal(path, clock)
    assert reopened.archive_alerts == ()
    assert reopened.inspect().processing_high_water == following + 1
    assert reopened.admission_status().available
    reopened.close()


def test_archive_holds_exactly_the_archived_bytes(tmp_path: Path) -> None:
    path, clock, value, _ = _compacted(tmp_path)
    expected = b"".join(segment.read_bytes() for segment in _segments(path))
    outcome = value.compact(retention_days=30)
    archive = value.archive_dir / outcome.archive_file  # type: ignore[operator]
    assert gzip.decompress(archive.read_bytes()) == expected
    value.close()


def test_segments_inside_the_window_are_not_archived(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    _fill(value, 1, 8)
    clock.advance(days=29)
    assert not value.compaction_due(retention_days=30)
    outcome = value.compact(retention_days=30)
    assert not outcome.performed
    assert outcome.reason == "nothing_due"
    value.close()


def test_compaction_requires_a_clean_boundary(tmp_path: Path) -> None:
    path, clock, value, following = _compacted(tmp_path)
    value.append_accepted(event(f"e{following}"))
    outcome = value.compact(retention_days=30)
    assert not outcome.performed
    assert outcome.reason == "boundary_not_clean"
    assert not value.archive_dir.exists() or not list(value.archive_dir.iterdir())
    value.close()


def test_guard_defers_without_writing(tmp_path: Path) -> None:
    path, clock, value, _ = _compacted(tmp_path)
    seen: list[int] = []

    def blocked(high_water: int) -> bool:
        seen.append(high_water)
        return True

    segments = _segments(path)
    outcome = value.compact(retention_days=30, guard=blocked)
    assert not outcome.performed
    assert outcome.reason == "deferred_semantic"
    assert seen and seen[0] > 0
    assert _segments(path) == segments
    assert value.inspect().latest_compaction is None
    value.close()


def test_non_monotonic_timestamps_refuse_compaction(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    following = _fill(value, 1, 6)
    clock.advance(days=-1)
    _fill(value, following, 2)
    clock.advance(days=40)
    outcome = value.compact(retention_days=30)
    assert not outcome.performed
    assert outcome.reason == "clock_not_monotonic"
    value.close()


# --- repeated compaction, chain and carried state -------------------------------


def test_second_compaction_chains_and_can_archive_the_first(tmp_path: Path) -> None:
    """C7: a segment holding a COMPACTION may itself be archived later."""
    path, clock, value, following = _compacted(tmp_path)
    first = value.compact(retention_days=30)
    assert first.performed
    first_hash = next(
        record.record_hash
        for record in value.records
        if record.lifecycle is EventLifecycle.COMPACTION
    )
    following = _fill(value, following, 10)
    clock.advance(days=31)
    second = value.compact(retention_days=30)
    assert second.performed
    latest = value.inspect().latest_compaction
    assert latest is not None
    assert latest.previous_compaction_hash == first_hash
    assert not any(record.record_hash == first_hash for record in value.records), (
        "the first COMPACTION was archived with its segment"
    )
    _fill(value, following, 1)
    value.close()
    reopened = _journal(path, clock)
    assert reopened.archive_alerts == ()
    assert reopened.inspect().latest_compaction == latest
    reopened.close()


def test_carried_tail_survives_archiving(tmp_path: Path) -> None:
    """C8: the Semantic checkpoint tail is carried verbatim."""
    path, clock, value, _ = _compacted(tmp_path)
    tail = next(
        record
        for record in value.records
        if record.lifecycle is EventLifecycle.COMPLETED
    )
    outcome = value.compact(
        retention_days=30, carried_tail=(tail.record_id, tail.record_hash)
    )
    assert outcome.performed
    inspection = value.inspect()
    assert tail not in inspection.records
    assert tail in inspection.carried_records
    value.close()


def test_baseline_is_carried_and_continues(tmp_path: Path) -> None:
    from tests.test_event_journal import append_u5_baseline

    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    append_u5_baseline(value)
    baseline = value.inspect().baselines[-1]
    _fill(value, 1, 8)
    clock.advance(days=31)
    assert value.compact(retention_days=30).performed
    inspection = value.inspect()
    assert inspection.baselines[-1] == baseline
    assert all(
        record.lifecycle is not EventLifecycle.PARTICIPANT_BASELINE_ESTABLISHED
        for record in inspection.records
    )
    value.close()
    reopened = _journal(path, clock)
    assert reopened.inspect().baselines[-1] == baseline
    reopened.close()


# --- crash and archive damage ---------------------------------------------------


def test_crash_before_unlink_rolls_forward_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, clock, value, following = _compacted(tmp_path)
    segments = _segments(path)

    def crash(self, paths):  # type: ignore[no-untyped-def]
        raise OSError("crash before unlink")

    monkeypatch.setattr(EventJournal, "_unlink_archived_segments_unlocked", crash)
    with pytest.raises(OSError):
        value.compact(retention_days=30)
    # In process the archived segments are still present and still verify.
    assert _segments(path) == segments
    assert value.inspect().latest_compaction is not None
    value.close()
    monkeypatch.undo()
    reopened = _journal(path, clock)
    assert _segments(path) != segments
    latest = reopened.inspect().latest_compaction
    assert latest is not None
    assert reopened.records[0].record_hash == latest.boundary_checkpoint_hash
    _fill(reopened, following, 1)
    reopened.close()


def test_missing_archive_is_an_alert_not_a_failure(tmp_path: Path) -> None:
    path, clock, value, _ = _compacted(tmp_path)
    outcome = value.compact(retention_days=30)
    value.close()
    archive = value.archive_dir / outcome.archive_file  # type: ignore[operator]
    os.chmod(archive, 0o600)
    archive.unlink()
    reopened = _journal(path, clock)
    assert [alert.reason for alert in reopened.archive_alerts] == ["missing"]
    assert reopened.admission_status().available
    reopened.close()


def test_damaged_archive_is_an_alert_not_a_failure(tmp_path: Path) -> None:
    path, clock, value, _ = _compacted(tmp_path)
    outcome = value.compact(retention_days=30)
    value.close()
    archive = value.archive_dir / outcome.archive_file  # type: ignore[operator]
    os.chmod(archive, 0o600)
    data = bytearray(archive.read_bytes())
    data[-1] ^= 0xFF
    archive.write_bytes(bytes(data))
    reopened = _journal(path, clock)
    assert [alert.reason for alert in reopened.archive_alerts] == ["digest_mismatch"]
    reopened.close()


def test_missing_archive_keeps_segments_during_roll_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, clock, value, _ = _compacted(tmp_path)
    segments = _segments(path)

    def crash(self, paths):  # type: ignore[no-untyped-def]
        raise OSError("crash before unlink")

    monkeypatch.setattr(EventJournal, "_unlink_archived_segments_unlocked", crash)
    with pytest.raises(OSError):
        value.compact(retention_days=30)
    value.close()
    monkeypatch.undo()
    for archive in value.archive_dir.glob("*.jsonl.gz"):
        os.chmod(archive, 0o600)
        archive.unlink()
    reopened = _journal(path, clock)
    # Nothing is lost: the segments stay and replay normally.
    assert _segments(path) == segments
    assert [alert.reason for alert in reopened.archive_alerts] == ["missing"]
    assert reopened.admission_status().available
    reopened.close()


def test_unreferenced_archive_and_temporary_files_are_cleaned_up(
    tmp_path: Path,
) -> None:
    path, clock, value, _ = _compacted(tmp_path)
    assert value.compact(retention_days=30).performed
    value.close()
    leftover = value.archive_dir / f"999990-999999-{uuid4()}.jsonl.gz"
    leftover.write_bytes(gzip.compress(b"x"))
    temporary = value.archive_dir / f".compaction-{uuid4()}.tmp"
    temporary.write_bytes(b"partial")
    reopened = _journal(path, clock)
    assert not leftover.exists()
    assert leftover.with_name(f".{leftover.name}.unreferenced").exists()
    assert not temporary.exists()
    reopened.close()


# --- verification rules ----------------------------------------------------------


def test_chain_must_connect_to_the_archived_range(tmp_path: Path) -> None:
    path, clock, value, _ = _compacted(tmp_path)
    assert value.compact(retention_days=30).performed
    records = value.records
    position, compaction = next(
        (index, record)
        for index, record in enumerate(records)
        if record.lifecycle is EventLifecycle.COMPACTION
    )
    assert compaction.compaction is not None
    forged = compaction.model_copy(
        update={
            "compaction": compaction.compaction.model_copy(
                update={"archived_last_record_hash": "e" * 64}
            )
        }
    )
    tampered = (*records[:position], forged, *records[position + 1 :])
    with pytest.raises(EventJournalIntegrityError, match="not connected"):
        EventJournal._verify_records_from(tampered, None)
    value.close()


def test_declared_state_must_match_the_archived_records(tmp_path: Path) -> None:
    from tests.test_event_journal import append_u5_baseline

    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    append_u5_baseline(value)
    _fill(value, 1, 8)
    clock.advance(days=31)

    def crash(self, paths):  # type: ignore[no-untyped-def]
        raise OSError("keep the archived segments present")

    original = EventJournal._unlink_archived_segments_unlocked
    EventJournal._unlink_archived_segments_unlocked = crash  # type: ignore[method-assign]
    try:
        with pytest.raises(OSError):
            value.compact(retention_days=30)
    finally:
        EventJournal._unlink_archived_segments_unlocked = original  # type: ignore[method-assign]
    records = value.records
    position, compaction = next(
        (index, record)
        for index, record in enumerate(records)
        if record.lifecycle is EventLifecycle.COMPACTION
    )
    assert compaction.compaction is not None
    forged = compaction.model_copy(
        update={
            "compaction": compaction.compaction.model_copy(
                update={"baseline_record": None}
            )
        }
    )
    tampered = (*records[:position], forged, *records[position + 1 :])
    with pytest.raises(EventJournalIntegrityError, match="declared state"):
        EventJournal._verify_records_from(tampered, None)
    value.close()


def test_every_prefix_after_compaction_matches_fresh_verification(
    tmp_path: Path,
) -> None:
    from tests.test_event_journal_verification_cache import _observer, _outcome

    path, clock, value, following = _compacted(tmp_path)
    assert value.compact(retention_days=30).performed
    _fill(value, following, 3)
    records = value.records
    observer = _observer(value)
    count = len(records)
    for length in [*range(1, count + 1), *range(count, 0, -1)]:
        prefix = records[:length]
        cached = _outcome(lambda: observer._verify_records(prefix))
        fresh = _outcome(lambda: EventJournal._verify_records_from(prefix, None)[0])
        assert cached == fresh, length
    value.close()


# --- end to end through the real chat path ------------------------------------


def test_chat_keeps_admitting_with_small_segments_and_compaction(
    tmp_path: Path,
) -> None:
    """IP-4a acceptance shape: 64 KiB segments used to stop after ~23 turns."""
    from suzka.runtime.journal_compaction import JournalCompactionPolicy

    from tests.test_fastapi_backend import _client, _settings

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "event_journal": settings.event_journal.model_copy(
                update={"max_bytes": 64 * 1024, "retained_files": 4}
            )
        }
    )
    client = _client(tmp_path, settings=settings)
    client.app.state.journal_compaction_policy = JournalCompactionPolicy(0)
    with client:
        for number in range(40):
            response = client.post(
                "/api/chat", json={"message": f"turn {number}", "attachments": []}
            )
            assert response.status_code == 200, (number, response.text)
        journal = client.app.state.event_journal
        inspection = journal.inspect()
        assert inspection.latest_compaction is not None
        archives = list(journal.archive_dir.glob("*.jsonl.gz"))
        # Compaction keeps working, not just once (IP-4a review P0-1): old
        # segments do not pile up and archives keep being written.
        assert len(archives) >= 3
        assert len(_segments(journal.path)) <= 2
        high_water = inspection.processing_high_water
    with _client(tmp_path, settings=settings) as restarted:
        assert restarted.app.state.event_journal.archive_alerts == ()
        assert (
            restarted.app.state.event_journal.inspect().processing_high_water
            == high_water
        )
        response = restarted.post(
            "/api/chat", json={"message": "after restart", "attachments": []}
        )
        assert response.status_code == 200, response.text


# --- independent review follow-ups ---------------------------------------------


def test_repeated_compactions_with_baseline_and_carried_tail(tmp_path: Path) -> None:
    """Review P0-1: the state of the compaction IN EFFECT is the prior."""
    from tests.test_event_journal import append_u5_baseline

    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    append_u5_baseline(value)
    baseline = value.inspect().baselines[-1]
    following = _fill(value, 1, 8)
    tail = next(r for r in value.records if r.lifecycle is EventLifecycle.COMPLETED)
    for _ in range(3):
        clock.advance(days=31)
        outcome = value.compact(
            retention_days=30, carried_tail=(tail.record_id, tail.record_hash)
        )
        assert outcome.performed, outcome.reason
        inspection = value.inspect()
        assert inspection.baselines[-1] == baseline
        assert tail in inspection.carried_records
        following = _fill(value, following, 8)
    value.close()
    reopened = _journal(path, clock)
    inspection = reopened.inspect()
    assert inspection.baselines[-1] == baseline
    assert tail in inspection.carried_records
    assert reopened.archive_alerts == ()
    reopened.close()


def test_chain_starts_at_the_oldest_retained_compaction(tmp_path: Path) -> None:
    """Review P0-2: C0 archived, C1 retained, C2 in effect -> still opens."""
    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock)
    following = _fill(value, 1, 8)
    clock.advance(days=31)
    assert value.compact(retention_days=30).performed  # C0
    following = _fill(value, following, 8)
    clock.advance(days=31)
    assert value.compact(retention_days=30).performed  # C1 archives C0's segment
    following = _fill(value, following, 3)
    clock.advance(days=31)
    third = value.compact(retention_days=30)  # C2, while C1 may stay retained
    assert third.performed, third.reason
    compactions = [r for r in value.records if r.lifecycle is EventLifecycle.COMPACTION]
    assert compactions
    value.inspect()
    _fill(value, following, 2)
    value.close()
    reopened = _journal(path, clock)
    assert reopened.inspect().latest_compaction is not None
    reopened.close()


def test_pending_unlinks_are_finished_before_the_next_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review P2-3: never re-archive records already archived."""
    path, clock, value, following = _compacted(tmp_path)

    def crash(self, paths):  # type: ignore[no-untyped-def]
        raise OSError("crash before unlink")

    monkeypatch.setattr(EventJournal, "_unlink_archived_segments_unlocked", crash)
    with pytest.raises(OSError):
        value.compact(retention_days=30)
    monkeypatch.undo()
    following = _fill(value, following, 8)
    clock.advance(days=31)
    second = value.compact(retention_days=30)
    assert second.performed, second.reason
    ranges = sorted(
        tuple(int(part) for part in archive.name.split("-")[:2])
        for archive in value.archive_dir.glob("*.jsonl.gz")
    )
    assert len(ranges) == 2
    assert ranges[0][1] <= ranges[1][0], ranges
    value.close()

