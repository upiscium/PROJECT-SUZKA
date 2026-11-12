"""IP-3a (#383 / #382 D1): EventJournal schema v4 and WAL rollover evidence.

A completed rollover is a state update of the verifier's WAL anchor: the same
snapshot gets a new anchor (the new generation's baseline). A prepared
rollover must be resolved before anything else is recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from pydantic import ValidationError

from suzka.runtime.event_journal import (
    CURRENT_EVENT_JOURNAL_SCHEMA_VERSION,
    EventJournal,
    EventJournalAppendError,
    EventJournalIntegrityError,
    EventJournalRecord,
    EventLifecycle,
)

from tests.test_event_journal import (
    HASH_0,
    HASH_1,
    HASH_2,
    NOW,
    bootstrap_v3,
    event,
    journal,
)

GENERATION_0 = str(uuid5(NAMESPACE_URL, "v2-generation"))
WAL_0 = str(uuid5(NAMESPACE_URL, "v2-wal"))
WAL_HASH_0 = HASH_1


def _commit(value: EventJournal, name: str, sequence: int, before: str, after: str):  # type: ignore[no-untyped-def]
    """One successful event, bound to the Journal's current WAL generation."""
    item = event(name, sequence)
    inspection = value.inspect()
    generation = inspection.wal_generation_id
    value.append_accepted(event(name))
    value.append_started(item)
    value.append_prepared(item, before, after, generation)
    wal_record_id = str(uuid4())
    wal_record_hash = (f"{sequence:x}" * 64)[:64]
    value.append_completed(
        item, sequence, after, generation, wal_record_id, wal_record_hash
    )
    return wal_record_id, wal_record_hash


def bootstrap_v4(path: Path, **kwargs) -> EventJournal:  # type: ignore[no-untyped-def]
    value = journal(path, **kwargs) if kwargs else None
    if value is None:
        value = bootstrap_v3(path)
    else:
        value.append_v2_bootstrap_checkpoint(0, HASH_0, GENERATION_0, WAL_0, WAL_HASH_0)
        value.append_v3_migration_checkpoint()
    value.append_v4_migration_checkpoint()
    return value


def _prepare(value: EventJournal, **overrides):  # type: ignore[no-untyped-def]
    inspection = value.inspect()
    fields = {
        "rollover_id": str(uuid4()),
        "wal_generation_id": str(uuid4()),
        "wal_record_id": str(uuid4()),
        "wal_record_hash": HASH_2,
        "wal_predecessor_generation_id": inspection.wal_generation_id,
        "wal_predecessor_record_hash": inspection.wal_record_hash,
    }
    fields.update(overrides)
    value.append_wal_rollover_prepared(**fields)
    return fields


def _reopen(path: Path, value: EventJournal, **kwargs) -> EventJournal:  # type: ignore[no-untyped-def]
    value.close()
    return journal(path, **kwargs)


# --- v3 -> v4 migration -------------------------------------------------------


def test_current_schema_is_v4() -> None:
    assert CURRENT_EVENT_JOURNAL_SCHEMA_VERSION == 4


def test_v4_migration_keeps_v3_bytes_and_carries_both_authorities(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap_v3(path)
    _commit(value, "before-v4", 1, HASH_0, HASH_1)
    v3_bytes = path.read_bytes()
    v3_tail = value.records[-1]
    value.append_v4_migration_checkpoint()
    assert path.read_bytes().startswith(v3_bytes)
    checkpoint = value.records[-1]
    assert checkpoint.schema_version == 4
    assert checkpoint.lifecycle is EventLifecycle.CHECKPOINT
    assert checkpoint.migration_previous_v3_hash == v3_tail.record_hash
    assert checkpoint.v4_migration_anchor_hash == v3_tail.record_hash
    assert checkpoint.v3_migration_anchor_hash is not None
    # v3 lines must not gain v4 keys.
    for line in v3_bytes.splitlines():
        assert "rollover_id" not in json.loads(line)
    reopened = _reopen(path, value)
    assert reopened.inspect().schema_version == 4
    # Ordinary evidence continues at schema 4.
    _commit(reopened, "after-v4", 2, HASH_1, HASH_2)
    assert {record.schema_version for record in reopened.records[-4:]} == {4}
    reopened.close()


def test_v4_migration_requires_a_closed_v3_journal(tmp_path: Path) -> None:
    value = bootstrap_v3(tmp_path / "events.jsonl")
    item = event("open", 1)
    value.append_accepted(event("open"))
    value.append_started(item)
    with pytest.raises(EventJournalIntegrityError):
        value.append_v4_migration_checkpoint()
    value.close()


def test_v4_migration_is_not_repeated(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    with pytest.raises(EventJournalIntegrityError):
        value.append_v4_migration_checkpoint()
    value.close()


def test_older_records_cannot_carry_v4_fields() -> None:
    with pytest.raises(ValidationError):
        EventJournalRecord(
            schema_version=3,
            record_id=str(uuid4()),
            timestamp=NOW,
            lifecycle=EventLifecycle.ACCEPTED,
            record_hash=HASH_0,
            rollover_id=str(uuid4()),
        )


def test_v3_record_after_v4_is_a_downgrade(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap_v4(path)
    records = value.records
    forged = records[-2].model_copy(update={"record_id": str(uuid4())})
    with pytest.raises(EventJournalIntegrityError, match="downgrade"):
        EventJournal._verify_records_from((*records, forged), None)
    value.close()


def test_rotation_in_v4_carries_both_migration_authorities(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap_v4(path, max_bytes=3_000, retained=50)
    for number in range(1, 8):
        _commit(value, f"rotate-{number}", number, HASH_0, HASH_0)
    assert sorted(tmp_path.glob("events.jsonl.[0-9]*"))
    head = value.records
    checkpoints = [
        r
        for r in head
        if r.lifecycle is EventLifecycle.CHECKPOINT and r.schema_version == 4
    ]
    assert len(checkpoints) >= 2  # the migration and at least one rotation
    assert {c.v4_migration_anchor_hash for c in checkpoints} == {
        checkpoints[0].v4_migration_anchor_hash
    }
    assert {c.v3_migration_anchor_hash for c in checkpoints} == {
        checkpoints[0].v3_migration_anchor_hash
    }
    reopened = _reopen(path, value, max_bytes=3_000, retained=50)
    assert reopened.inspect().schema_version == 4
    reopened.close()


# --- rollover evidence -----------------------------------------------------------


def test_completed_rollover_moves_the_wal_anchor_but_not_the_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap_v4(path)
    _commit(value, "e1", 1, HASH_0, HASH_1)
    before = value.inspect()
    fields = _prepare(value)
    assert value.inspect().open_wal_rollover is not None
    value.append_wal_rollover_completed(fields["rollover_id"])
    after = value.inspect()
    assert after.open_wal_rollover is None
    assert (after.snapshot_sequence, after.snapshot_hash) == (
        before.snapshot_sequence,
        before.snapshot_hash,
    )
    assert after.processing_high_water == before.processing_high_water
    assert (after.wal_generation_id, after.wal_record_id, after.wal_record_hash) == (
        fields["wal_generation_id"],
        fields["wal_record_id"],
        fields["wal_record_hash"],
    )
    # The next event binds the new generation.
    _commit(value, "e2", 2, HASH_1, HASH_2)
    reopened = _reopen(path, value)
    assert reopened.inspect().snapshot_hash == HASH_2
    reopened.close()


def test_aborted_rollover_keeps_the_previous_anchor(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    _commit(value, "e1", 1, HASH_0, HASH_1)
    before = value.inspect()
    fields = _prepare(value)
    value.append_wal_rollover_aborted(fields["rollover_id"])
    after = value.inspect()
    assert after.open_wal_rollover is None
    assert (after.wal_generation_id, after.wal_record_id, after.wal_record_hash) == (
        before.wal_generation_id,
        before.wal_record_id,
        before.wal_record_hash,
    )
    value.close()


def test_open_rollover_admits_nothing_but_its_resolution(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    _commit(value, "e1", 1, HASH_0, HASH_1)
    fields = _prepare(value)
    with pytest.raises(EventJournalAppendError):
        value.append_accepted(event("blocked"))
    with pytest.raises(EventJournalAppendError):
        _prepare(value)
    records = value.records
    migration = next(
        r
        for r in records
        if r.lifecycle is EventLifecycle.CHECKPOINT and r.schema_version == 4
    )
    forged_checkpoint = migration.model_copy(
        update={"record_id": str(uuid4()), "migration_previous_v3_hash": None}
    )
    started = next(r for r in records if r.lifecycle is EventLifecycle.STARTED)
    forged_started = started.model_copy(update={"record_id": str(uuid4())})
    for forged in (forged_checkpoint, forged_started):
        with pytest.raises(EventJournalIntegrityError, match="rollover is open"):
            EventJournal._verify_records_from((*records, forged), None)
    with pytest.raises(EventJournalIntegrityError):
        value.append_wal_rollover_completed(str(uuid4()))
    value.append_wal_rollover_completed(fields["rollover_id"])
    _commit(value, "e2", 2, HASH_1, HASH_2)
    value.close()


def test_open_rollover_blocks_rotation(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl", max_bytes=1, retained=50)
    _prepare(value)
    status = value.admission_status()
    assert status.lifecycle_blocks_rotation
    value.close()


def test_rollover_requires_a_clean_boundary(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    item = event("in-flight", 1)
    value.append_accepted(event("in-flight"))
    with value.wal_rollover_boundary() as boundary:
        # ACCEPTED-only events have not touched the WAL.
        assert boundary.clean, boundary.reason
    value.append_started(item)
    with value.wal_rollover_boundary() as boundary:
        assert not boundary.clean
        assert boundary.reason == "event_in_progress"
    with pytest.raises(EventJournalAppendError):
        _prepare(value)
    value.close()


def test_accepted_only_event_may_wait_across_a_rollover(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    queued = event("queued", 1)
    value.append_accepted(event("queued"))
    fields = _prepare(value)
    value.append_wal_rollover_completed(fields["rollover_id"])
    value.append_started(queued)
    value.append_prepared(queued, HASH_0, HASH_1, fields["wal_generation_id"])
    value.append_completed(
        queued, 1, HASH_1, fields["wal_generation_id"], str(uuid4()), HASH_2
    )
    assert value.inspect().snapshot_hash == HASH_1
    value.close()


@pytest.mark.parametrize(
    "override",
    [
        {"wal_predecessor_generation_id": str(uuid4())},
        {"wal_predecessor_record_hash": "e" * 64},
    ],
    ids=["predecessor-generation", "predecessor-hash"],
)
def test_rollover_must_chain_to_the_current_anchor(
    tmp_path: Path, override: dict[str, str]
) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    with pytest.raises(EventJournalAppendError):
        _prepare(value, **override)
    value.close()


def test_rollover_must_change_generation(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    current = value.inspect().wal_generation_id
    with pytest.raises(EventJournalAppendError):
        _prepare(value, wal_generation_id=current)
    value.close()


def test_rollover_identifier_is_never_reused(tmp_path: Path) -> None:
    value = bootstrap_v4(tmp_path / "events.jsonl")
    fields = _prepare(value)
    value.append_wal_rollover_aborted(fields["rollover_id"])
    with pytest.raises(EventJournalAppendError):
        _prepare(value, rollover_id=fields["rollover_id"])
    value.close()


def test_rollover_is_unavailable_before_v4(tmp_path: Path) -> None:
    value = bootstrap_v3(tmp_path / "events.jsonl")
    with value.wal_rollover_boundary() as boundary:
        assert not boundary.clean
        assert boundary.reason == "journal_schema_below_4"
    with pytest.raises(EventJournalIntegrityError):
        _prepare(value)
    value.close()


def test_open_rollover_survives_reopen_and_can_be_resolved(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap_v4(path)
    fields = _prepare(value)
    reopened = _reopen(path, value)
    pending = reopened.inspect().open_wal_rollover
    assert pending is not None
    assert pending.rollover_id == fields["rollover_id"]
    assert pending.wal_generation_id == fields["wal_generation_id"]
    reopened.append_wal_rollover_completed(fields["rollover_id"])
    assert reopened.inspect().wal_generation_id == fields["wal_generation_id"]
    reopened.close()


def test_rotation_after_rollover_checkpoints_the_new_anchor(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap_v4(path, max_bytes=2_500, retained=50)
    _commit(value, "e1", 1, HASH_0, HASH_1)
    fields = _prepare(value)
    value.append_wal_rollover_completed(fields["rollover_id"])
    # Force a rotation: the new checkpoint must carry the new anchor.
    number = 2
    while not sorted(tmp_path.glob("events.jsonl.[0-9]*")):
        _commit(value, f"fill-{number}", number, HASH_1, HASH_1)
        number += 1
    checkpoint = [
        r for r in value.records if r.lifecycle is EventLifecycle.CHECKPOINT
    ][-1]  # the head of the active segment
    assert checkpoint.wal_generation_id == fields["wal_generation_id"]
    assert checkpoint.wal_record_id == fields["wal_record_id"]
    reopened = _reopen(path, value, max_bytes=2_500, retained=50)
    assert reopened.inspect().wal_generation_id == fields["wal_generation_id"]
    reopened.close()


def test_checkpoint_after_rollover_with_old_anchor_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    value = bootstrap_v4(path)
    _commit(value, "e1", 1, HASH_0, HASH_1)
    before = value.inspect()
    fields = _prepare(value)
    value.append_wal_rollover_completed(fields["rollover_id"])
    with pytest.raises(EventJournalIntegrityError):
        value.append_v2_current_checkpoint(
            before.snapshot_sequence,
            before.snapshot_hash,
            before.wal_generation_id,
            before.wal_record_id,
            before.wal_record_hash,
        )
    value.append_v2_current_checkpoint(
        before.snapshot_sequence,
        before.snapshot_hash,
        fields["wal_generation_id"],
        fields["wal_record_id"],
        fields["wal_record_hash"],
    )
    value.close()


def test_every_prefix_of_a_rollover_journal_matches_fresh_verification(
    tmp_path: Path,
) -> None:
    from tests.test_event_journal_verification_cache import _observer, _outcome

    value = bootstrap_v4(tmp_path / "events.jsonl")
    _commit(value, "e1", 1, HASH_0, HASH_1)
    fields = _prepare(value)
    value.append_wal_rollover_completed(fields["rollover_id"])
    aborted = _prepare(value)
    value.append_wal_rollover_aborted(aborted["rollover_id"])
    _commit(value, "e2", 2, HASH_1, HASH_2)
    records = value.records
    observer = _observer(value)
    count = len(records)
    for length in [*range(1, count + 1), *range(count, 0, -1)]:
        prefix = records[:length]
        cached = _outcome(lambda: observer._verify_records(prefix))
        fresh = _outcome(lambda: EventJournal._verify_records_from(prefix, None)[0])
        assert cached == fresh, length
    value.close()
