"""IP-9 U4-3: one-way adoption of ``subject.history`` and its incarnation (#376)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.history_adoption import (  # noqa: E402
    SUBJECT_HISTORY_BASELINE,
    HistoryAdoptionError,
    activate_adopted_store,
    adopted_history_baseline,
    baseline_digest,
    current_incarnation,
    genesis_incarnation_id,
    history_registry,
    open_or_create_store,
    rollback_incarnation_id,
)
from suzka.r16.history_codec import SUBJECT_HISTORY_PARTICIPANT_ID  # noqa: E402
from suzka.r16.history_participant import (  # noqa: E402
    JournalCommitProofSource,
    SubjectHistoryParticipant,
)
from suzka.r16.history_store import (  # noqa: E402
    HistoryMetaPhase,
    HistoryPhase,
    SubjectHistoryStore,
)
from suzka.runtime.event_journal import EventJournalGateClear  # noqa: E402
from suzka.runtime.startup_reconciliation import (  # noqa: E402
    StartupReconciliationCoordinator,
    StartupReconciliationError,
)
from test_r16_ip9_subject_history import (  # noqa: E402
    _committed,
    _event,
    _prepared,
    _snapshot,
)
from test_startup_reconciliation import _graph  # noqa: E402


PAGES = 4096


def _history_path(tmp_path: Path) -> Path:
    directory = tmp_path / "subject_history"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    return directory / "subject_history.sqlite3"


def _adopt(tmp_path: Path):
    memory, state, journal, wal, recovery = _graph(tmp_path)
    store = open_or_create_store(
        journal, _history_path(tmp_path), subject_id="suzka", max_page_count=PAGES
    )
    assert store.meta().phase is HistoryMetaPhase.PROVISIONAL
    coordinator = StartupReconciliationCoordinator(
        journal, recovery, memory, history_store=store
    )
    baseline = coordinator.ensure_adoption_baseline(recovery.prepare_startup())
    return memory, state, journal, recovery, store, coordinator, baseline


def test_registry_adds_history_as_a_sorted_durable_participant() -> None:
    base = StartupReconciliationCoordinator.participant_registry
    registry = history_registry(base)
    assert len(registry) == len(base) + 1
    assert registry[-1] == SUBJECT_HISTORY_BASELINE
    assert [item.participant_id for item in registry] == sorted(
        item.participant_id for item in registry
    )
    with pytest.raises(HistoryAdoptionError):
        history_registry(registry)


def test_clean_startup_adopts_five_participants_then_activates(tmp_path: Path) -> None:
    _memory, _state, journal, _recovery, store, coordinator, baseline = _adopt(tmp_path)

    baselines = journal.inspect().baselines
    assert len(baselines) == 2
    assert baselines[0].participant_registry == StartupReconciliationCoordinator.participant_registry
    assert baseline == baselines[1]
    assert baseline.adoption_epoch == baselines[0].adoption_epoch + 1
    assert SUBJECT_HISTORY_BASELINE in baseline.participant_registry
    assert adopted_history_baseline(journal.inspect()) == baseline

    meta = store.meta()
    assert meta.phase is HistoryMetaPhase.ACTIVE
    assert meta.adopted_baseline_id == baseline.baseline_id
    assert meta.adopted_baseline_hash == baseline_digest(baseline)
    assert meta.genesis_incarnation_id == genesis_incarnation_id(baseline)
    assert coordinator.history_incarnation is not None
    assert coordinator.history_incarnation() == meta.genesis_incarnation_id
    assert current_incarnation(journal, store) == meta.genesis_incarnation_id


def test_restart_is_idempotent_and_reopens_the_same_store(tmp_path: Path) -> None:
    memory, _state, journal, recovery, store, _coordinator, baseline = _adopt(tmp_path)
    path = store.path
    store.close()
    before = journal.path.read_bytes()

    reopened = open_or_create_store(journal, path, subject_id="suzka", max_page_count=PAGES)
    again = StartupReconciliationCoordinator(
        journal, recovery, memory, history_store=reopened
    ).ensure_adoption_baseline(recovery.prepare_startup())
    assert again == baseline
    assert journal.path.read_bytes() == before
    assert reopened.meta().phase is HistoryMetaPhase.ACTIVE


def test_provisional_store_resumes_activation_after_a_crash(tmp_path: Path) -> None:
    memory, _state, journal, _wal, recovery = _graph(tmp_path)
    store = open_or_create_store(
        journal, _history_path(tmp_path), subject_id="suzka", max_page_count=PAGES
    )
    # Crash after the five-participant baseline but before ACTIVE.
    recovery.establish_participant_baseline(
        recovery.prepare_startup(),
        history_registry(StartupReconciliationCoordinator.participant_registry),
    )
    assert store.meta().phase is HistoryMetaPhase.PROVISIONAL
    path = store.path
    store.close()

    reopened = open_or_create_store(journal, path, subject_id="suzka", max_page_count=PAGES)
    StartupReconciliationCoordinator(
        journal, recovery, memory, history_store=reopened
    ).ensure_adoption_baseline(recovery.prepare_startup())
    assert reopened.meta().phase is HistoryMetaPhase.ACTIVE


def test_lost_or_foreign_store_after_adoption_fails_closed(tmp_path: Path) -> None:
    memory, _state, journal, recovery, store, _coordinator, _baseline = _adopt(tmp_path)
    path = store.path
    store.close()
    path.unlink()
    # A missing store after adoption is lost authority, never recreated.
    with pytest.raises(HistoryAdoptionError, match="unavailable"):
        open_or_create_store(journal, path, subject_id="suzka", max_page_count=PAGES)
    assert not path.exists()

    foreign = SubjectHistoryStore.create(
        path, subject_id="suzka", journal_lineage_id="another-lineage", max_page_count=PAGES
    )
    with pytest.raises(StartupReconciliationError, match="another Journal"):
        StartupReconciliationCoordinator(
            journal, recovery, memory, history_store=foreign
        ).ensure_adoption_baseline(recovery.prepare_startup())


def test_store_adopted_for_another_baseline_fails_closed(tmp_path: Path) -> None:
    _memory, _state, journal, _recovery, store, _coordinator, baseline = _adopt(tmp_path)
    inspection = journal.inspect()
    altered = replace(baseline, processing_high_water=baseline.processing_high_water + 1)
    forged = replace(inspection, baselines=(*inspection.baselines[:-1], altered))
    with pytest.raises(HistoryAdoptionError, match="another baseline"):
        activate_adopted_store(store, forged)


def test_disabling_after_adoption_keeps_the_registry_and_fails_closed(
    tmp_path: Path,
) -> None:
    memory, _state, journal, recovery, _store, _coordinator, _baseline = _adopt(tmp_path)
    from suzka.runtime.state_recovery import StateRecoveryError

    # Startup without the store keeps four participants: R07 refuses a smaller
    # registry, so the adopted participant can never be silently dropped.
    with pytest.raises(StateRecoveryError, match="authority changed"):
        StartupReconciliationCoordinator(
            journal, recovery, memory
        ).ensure_adoption_baseline(recovery.prepare_startup())


def test_incarnation_changes_only_after_an_adopted_true_rollback(tmp_path: Path) -> None:
    _memory, _state, journal, _recovery, store, _coordinator, baseline = _adopt(tmp_path)
    inspection = journal.inspect()
    genesis = genesis_incarnation_id(baseline)

    def clear(baseline_id: str) -> EventJournalGateClear:
        return EventJournalGateClear(
            baseline_id=baseline_id,
            reconciliation_id="reconciliation-1",
            recovery_id="recovery-1",
            snapshot_sequence=inspection.snapshot_sequence,
            snapshot_hash=inspection.snapshot_hash or "0" * 64,
            processing_high_water=inspection.processing_high_water,
            wal_generation_id=inspection.wal_generation_id or "",
            wal_record_id=inspection.wal_record_id or "",
            wal_record_hash=inspection.wal_record_hash or "",
            journal_lineage_id=inspection.journal_lineage_id or "",
            terminal=True,
        )

    class FakeJournal:
        def __init__(self, value):
            self.value = value

        def inspect(self):
            return self.value

    before = clear(inspection.baselines[0].baseline_id)
    assert (
        current_incarnation(FakeJournal(replace(inspection, terminal_gate_clear=before)), store)
        == genesis
    )
    after = clear(baseline.baseline_id)
    rolled = current_incarnation(
        FakeJournal(replace(inspection, terminal_gate_clear=after)), store
    )
    assert rolled == rollback_incarnation_id(after) != genesis
    with pytest.raises(HistoryAdoptionError, match="open"):
        current_incarnation(FakeJournal(replace(inspection, open_gate_clear=after)), store)


def test_adopted_participant_commits_through_the_derived_incarnation(
    tmp_path: Path,
) -> None:
    memory, state, journal, recovery, store, coordinator, _baseline = _adopt(tmp_path)
    assert coordinator.history_incarnation is not None
    incarnation = coordinator.history_incarnation()
    event = _event("adopted-commit")
    decision = _committed(event, incarnation=incarnation)
    participant = SubjectHistoryParticipant(
        store, incarnation, decision, JournalCommitProofSource(journal, recovery)
    )
    _prepared(journal, event, participant)
    recovery.commit_internal_candidate(event, state.load(), _snapshot(1))

    assert coordinator.reconcile_open_transactions() == (True, None)
    completed = journal.inspect().completed_transactions[0]
    row = store.operation(incarnation, completed.transaction_id)
    assert row is not None and row.phase is HistoryPhase.FINALIZED
    assert completed.participant_outcomes[0][0] == SUBJECT_HISTORY_PARTICIPANT_ID


def test_server_opens_creates_and_refuses_disabling(tmp_path: Path) -> None:
    from suzka.api.server import _open_subject_history
    from suzka.config.schema import SubjectHistorySettings
    from suzka.runtime.state_recovery import StateRecoveryError
    from test_startup_reconciliation import _settings

    memory, _state, journal, _wal, recovery = _graph(tmp_path)
    settings = _settings(tmp_path)
    assert settings.subject_history.enabled is False
    assert _open_subject_history(settings, journal) is None

    enabled = settings.model_copy(
        update={
            "subject_history": SubjectHistorySettings(
                enabled=True,
                path=tmp_path / "private" / "history" / "subject_history.sqlite3",
                max_page_count=PAGES,
            )
        }
    )
    store = _open_subject_history(enabled, journal)
    assert store is not None
    assert (store.path.parent.stat().st_mode & 0o777) == 0o700
    StartupReconciliationCoordinator(
        journal, recovery, memory, history_store=store
    ).ensure_adoption_baseline(recovery.prepare_startup())
    store.close()

    with pytest.raises(StateRecoveryError, match="cannot be disabled"):
        _open_subject_history(settings, journal)
    reopened = _open_subject_history(enabled, journal)
    assert reopened is not None and reopened.meta().phase is HistoryMetaPhase.ACTIVE


@pytest.mark.parametrize("case", ["foreign_lineage", "active_without_adoption", "subject"])
def test_unadoptable_store_fails_before_any_journal_write(
    tmp_path: Path, case: str
) -> None:
    memory, _state, journal, _wal, recovery = _graph(tmp_path)
    path = _history_path(tmp_path)
    lineage = journal.inspect().journal_lineage_id
    assert lineage is not None
    store = SubjectHistoryStore.create(
        path,
        subject_id="suzka",
        journal_lineage_id="another-lineage" if case == "foreign_lineage" else lineage,
        max_page_count=PAGES,
    )
    if case == "active_without_adoption":
        store.activate(
            adopted_baseline_id="baseline-elsewhere",
            adopted_baseline_hash="9" * 64,
            genesis_incarnation_id="1" * 64,
        )
    store.close()
    before = journal.path.read_bytes()

    subject = "someone-else" if case == "subject" else "suzka"
    with pytest.raises(HistoryAdoptionError):
        open_or_create_store(journal, path, subject_id=subject, max_page_count=PAGES)
    # An injected store is checked again right before the R07 adoption.
    if case != "subject":
        injected = SubjectHistoryStore.open(path, max_page_count=PAGES)
        with pytest.raises(StartupReconciliationError):
            StartupReconciliationCoordinator(
                journal, recovery, memory, history_store=injected
            ).ensure_adoption_baseline(recovery.prepare_startup())
    assert journal.path.read_bytes() == before
    assert adopted_history_baseline(journal.inspect()) is None


def test_server_refuses_a_symlinked_store_directory(tmp_path: Path) -> None:
    from suzka.api.server import _open_subject_history
    from suzka.config.schema import SubjectHistorySettings
    from suzka.runtime.state_recovery import StateRecoveryError
    from test_startup_reconciliation import _settings

    _memory, _state, journal, _wal, _recovery = _graph(tmp_path)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    settings = _settings(tmp_path).model_copy(
        update={
            "subject_history": SubjectHistorySettings(
                enabled=True,
                path=tmp_path / "linked" / "subject_history.sqlite3",
                max_page_count=PAGES,
            )
        }
    )
    with pytest.raises(StateRecoveryError, match="unavailable"):
        _open_subject_history(settings, journal)
    assert list(real.iterdir()) == []
