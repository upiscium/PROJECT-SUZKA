"""G2-R0 test-only characterization of *existing* R07/R12 read effects.

This file verifies why current source-owner read paths are NOT acceptable as
G2 verification-only APIs without explicit recovery separation. The real
EventJournal/ExperienceStore are used only with synthetic pytest tmp_path data.

These assertions DO NOT add no-recovery readers, a source epoch, Context B,
G2 SourceSetVerified, or any R15 subject/memory admission capability.
"""

from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
from uuid import uuid4

import pytest

from suzka.experience import (
    ExperienceAppraisalEvidence,
    ExperienceAppraisalReasonCode,
    ExperienceEmotionContributions,
    ExperienceEmotionProjection,
    ExperienceEmotionUpdateReasonCode,
    ExperienceLifecycle,
    ExperienceMeasurementEvidence,
    ExperienceRecord,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    ExperienceRevisionRecord,
)
from suzka.memory.experience_store import ExperienceStore
from suzka.runtime.event_journal import EventJournal, EventJournalIntegrityError

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_HASH = "a" * 64
_OPERATION = "1" * 64
_EPISODIC = "2" * 64


def _fresh_journal(path: Path) -> EventJournal:
    return EventJournal(path, max_bytes=100_000, retained_files=4, clock=lambda: _NOW)


def _seed_journal(path: Path) -> EventJournal:
    journal = _fresh_journal(path)
    journal.verify_and_reconcile(0, _HASH)
    assert journal.inspect().records
    assert path.is_file()
    return journal


def _r12_original() -> ExperienceRecord:
    """Small, synthetic R12 CREATE record; no user text or real Memory."""
    experience_id = "experience:synthetic-r0"
    revision = ExperienceRevisionRecord(
        experience_id,
        0,
        ExperienceRevisionOperation.CREATE,
        ExperienceRevisionReason.CREATION,
        _NOW,
        event_id="event:synthetic-r0",
        event_sequence=1,
        evidence_refs=("event:synthetic-r0",),
    )
    return ExperienceRecord(
        experience_id=experience_id,
        revision=0,
        lifecycle=ExperienceLifecycle.ACTIVE,
        source_event_id="event:synthetic-r0",
        source_event_sequence=1,
        source_episode_id="episode:synthetic-r0",
        context_id="context:synthetic-r0",
        measurement=ExperienceMeasurementEvidence(
            "model." + "a" * 64, True, calibrated_novelty=0.25
        ),
        appraisal=ExperienceAppraisalEvidence(
            novelty=0.25,
            novelty_valid=True,
            reason_codes=(ExperienceAppraisalReasonCode.NOVELTY_MEASURED,),
        ),
        pre_appraisal_emotion=ExperienceEmotionProjection(0.0, 0.0),
        temporal_update_reasons=(
            ExperienceEmotionUpdateReasonCode.TIMELINE_INITIALIZED,
        ),
        post_appraisal_emotion=ExperienceEmotionProjection(0.5, 0.25),
        emotion_contributions=ExperienceEmotionContributions(),
        emotion_update_reasons=(ExperienceEmotionUpdateReasonCode.APPRAISAL_APPLIED,),
        subjective_salience=0.25,
        created_at=_NOW,
        revision_history=(revision,),
    )


def test_existing_r07_inspect_can_repair_interrupted_rotation(tmp_path: Path) -> None:
    """G2 must NOT treat current EventJournal.inspect as non-repairing proof."""
    path = tmp_path / "events.jsonl"
    journal = _seed_journal(path)
    try:
        before = path.read_bytes()
        interrupted = tmp_path / ".events.jsonl.rotation.tmp"
        os.replace(path, interrupted)
        assert not path.exists() and interrupted.is_file()
        assert interrupted.stat().st_mode & 0o777 == 0o600

        # The nominally read-only inspect() promotes the temp into active source.
        result = journal.inspect()
        assert result.records
        assert path.read_bytes() == before
        assert not interrupted.exists()
    finally:
        journal.close()


def test_existing_r07_constructor_can_repair_before_first_inspect(
    tmp_path: Path,
) -> None:
    """A new EventJournal constructor also reads and may recover source."""
    path = tmp_path / "events.jsonl"
    existing = _seed_journal(path)
    old_bytes = path.read_bytes()
    existing.close()
    interrupted = tmp_path / ".events.jsonl.rotation.tmp"
    os.replace(path, interrupted)
    assert not path.exists() and interrupted.exists()
    new_owner = _fresh_journal(path)
    try:
        assert new_owner.inspect().records
        assert path.read_bytes() == old_bytes
        assert not interrupted.exists()
    finally:
        new_owner.close()


def test_existing_r07_inspect_rejects_conflicting_rotation_marker(
    tmp_path: Path,
) -> None:
    """No hidden backfill is permitted when active and temp both exist."""
    path = tmp_path / "events.jsonl"
    journal = _seed_journal(path)
    marker = tmp_path / ".events.jsonl.rotation.tmp"
    marker.write_bytes(b"synthetic conflicting marker")
    marker.chmod(0o600)
    original = path.read_bytes()
    try:
        with pytest.raises(EventJournalIntegrityError):
            journal.inspect()
        assert path.read_bytes() == original
        assert marker.read_bytes() == b"synthetic conflicting marker"
    finally:
        journal.close()


@pytest.mark.parametrize(
    "reader", ["load_current", "load_committed_current", "load_revision"]
)
def test_existing_r12_readers_recover_publication_temp(
    tmp_path: Path,
    reader: str,
) -> None:
    """Even historical load_revision can delete an unfinished temp artifact."""
    store = ExperienceStore(tmp_path / "synthetic-experience-store")
    record = _r12_original()
    store.publish_create(record, _OPERATION, _EPISODIC)
    final_path = store.record_path(record.experience_id, 0)
    final_bytes = final_path.read_bytes()
    temporary = final_path.with_name(f".{final_path.name}.publish-{uuid4()}.tmp")
    temporary.write_bytes(final_bytes)
    temporary.chmod(0o600)
    assert temporary.is_file()

    # Existing load methods are allowed to reconcile a typed temp. A future
    # G2 verifier must instead return RECOVERY_REQUIRED without calling them.
    if reader == "load_current":
        entry = store.load_current(record.experience_id)
    elif reader == "load_committed_current":
        entry = store.load_committed_current(record.experience_id)
    else:
        entry = store.load_revision(record.experience_id, 0)
    assert entry is not None and entry.record == record
    assert not temporary.exists()
    assert final_path.read_bytes() == final_bytes


def test_existing_r12_historical_read_hardens_directory_mode(tmp_path: Path) -> None:
    """Existing load_revision may chmod parent; classify separately from source."""
    store = ExperienceStore(tmp_path / "synthetic-experience-store")
    record = _r12_original()
    store.publish_create(record, _OPERATION, _EPISODIC)
    directory = store.record_path(record.experience_id, 0).parent
    directory.chmod(0o755)
    assert directory.stat().st_mode & 0o777 == 0o755

    entry = store.load_revision(record.experience_id, 0)
    assert entry is not None and entry.record == record
    assert directory.stat().st_mode & 0o777 == 0o700
