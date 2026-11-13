"""IP-3b (#383 / #382 A, B, D1): ordinary StateWAL generation rollover.

A rollover starts a new generation for the same state at a clean boundary,
retains at most three generations, and is resolved at startup if it was
interrupted between the Journal preparation and its completion.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from suzka.runtime.event_journal import EventLifecycle
from suzka.runtime.state_wal import (
    BaselineRecord,
    RecoveryReason,
    StateWAL,
    StateWALConflictError,
    TransitionRecord,
)
from suzka.runtime.wal_rollover import (
    StateWALRolloverCoordinator,
    StateWALRolloverPolicy,
)

from tests.test_fastapi_backend import _client, _settings
from tests.test_state_wal import make_snapshot


def _chat(client, text: str = "hello") -> None:  # type: ignore[no-untyped-def]
    response = client.post("/api/chat", json={"message": text, "attachments": []})
    assert response.status_code == 200, response.text


def _generations(settings) -> set[str]:  # type: ignore[no-untyped-def]
    directory = Path(settings.state_wal.directory) / "generations"
    return {path.name for path in directory.glob("*.jsonl")}


def _state(client) -> tuple[int, str]:  # type: ignore[no-untyped-def]
    journal = client.app.state.event_journal.inspect()
    return journal.snapshot_sequence, journal.snapshot_hash


def _rolled(client) -> list:  # type: ignore[no-untyped-def]
    return [
        record
        for record in client.app.state.event_journal.records
        if record.lifecycle
        in {
            EventLifecycle.WAL_GENERATION_ROLLED_PREPARED,
            EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED,
            EventLifecycle.WAL_GENERATION_ROLLED_ABORTED,
        }
    ]


def _force(client, transitions: int = 1):  # type: ignore[no-untyped-def]
    coordinator = client.app.state.state_wal_rollover
    coordinator.policy = StateWALRolloverPolicy(max_transitions=transitions)
    return coordinator.maybe_rollover()


# --- end to end through the real chat path -----------------------------------


def test_chat_rolls_over_bounds_generations_and_survives_restart(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    client.app.state.state_wal_rollover_policy = StateWALRolloverPolicy(
        max_transitions=2
    )
    with client:
        anchor = client.app.state.state_wal.inspect_boot_anchor_optional()
        for number in range(8):
            _chat(client, f"turn {number}")
            assert len(_generations(settings)) <= 3
        completed = [
            record
            for record in _rolled(client)
            if record.lifecycle is EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED
        ]
        assert len(completed) >= 2
        inspection = client.app.state.state_wal.inspect()
        assert (
            len([r for r in inspection.records if isinstance(r, TransitionRecord)]) <= 2
        )
        assert inspection.active_manifest is not None
        assert inspection.active_manifest.predecessor_generation_id is not None
        # The boot anchor's generation is never pruned.
        if anchor is not None:
            assert f"{anchor.generation_id}.jsonl" in _generations(settings)
        # Restart right after a rollover: the Journal anchor is the ROLLED
        # record, which startup cross-authority validation must accept.
        client.app.state.state_wal_rollover.policy = StateWALRolloverPolicy(
            max_transitions=10_000
        )
        _chat(client, "one more turn without rollover")
        outcome = _force(client)
        assert outcome.performed
        before = _state(client)
    with _client(tmp_path, settings=settings) as restarted:
        assert _state(restarted) == before
        _chat(restarted, "after restart")
        assert restarted.app.state.event_journal.inspect().snapshot_sequence > before[0]


def test_rollover_never_changes_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        _chat(client)
        before = _state(client)
        wal_before = client.app.state.state_wal.inspect()
        outcome = _force(client)
        assert outcome.performed
        assert _state(client) == before
        wal_after = client.app.state.state_wal.inspect()
        assert wal_after.latest_snapshot_hash == wal_before.latest_snapshot_hash
        assert wal_after.latest_snapshot_sequence == wal_before.latest_snapshot_sequence
        baseline = wal_after.records[0]
        assert isinstance(baseline, BaselineRecord)
        assert baseline.reason is RecoveryReason.ROLLOVER
        assert len(wal_after.records) == 1
        assert wal_after.active_manifest is not None
        assert wal_before.active_manifest is not None
        assert (
            wal_after.active_manifest.predecessor_generation_id
            == wal_before.active_manifest.active_generation_id
        )
        assert (
            wal_after.active_manifest.predecessor_generation_hash
            == (wal_before.record_hashes[-1])
        )
        journal = client.app.state.event_journal.inspect()
        assert journal.wal_generation_id == str(
            wal_after.active_manifest.active_generation_id
        )
        _chat(client, "next")


def test_below_threshold_does_not_roll_over(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _chat(client)
        outcome = _force(client, transitions=10_000)
        assert not outcome.performed
        assert outcome.reason == "below_threshold"
        assert _rolled(client) == []


# --- crash matrix: interrupt the protocol, then restart ------------------------


def _prepare_only(client):  # type: ignore[no-untyped-def]
    """Steps 1-2 of the protocol (written + Journal PREPARED), then stop."""
    journal = client.app.state.event_journal
    wal = client.app.state.state_wal
    generation_id = uuid4()
    rollover_id = uuid4()
    with journal.wal_rollover_boundary() as boundary:
        assert boundary.clean, boundary.reason
        baseline, baseline_hash = wal.prepare_rollover_generation(
            generation_id=generation_id,
            journal_processing_high_water=boundary.processing_high_water,
        )
        journal.append_wal_rollover_prepared(
            rollover_id=str(rollover_id),
            wal_generation_id=str(generation_id),
            wal_record_id=str(baseline.record_id),
            wal_record_hash=baseline_hash,
            wal_predecessor_generation_id=boundary.wal_generation_id,
            wal_predecessor_record_hash=boundary.wal_record_hash,
        )
    return generation_id, rollover_id, baseline_hash


def _publish(client, generation_id, baseline_hash) -> None:  # type: ignore[no-untyped-def]
    client.app.state.state_wal.publish_rollover_generation(
        generation_id=generation_id,
        baseline_record_hash=baseline_hash,
        journal_processing_high_water=(
            client.app.state.event_journal.inspect().processing_high_water
        ),
    )


def test_crash_after_prepared_before_publish_is_aborted_at_startup(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        _chat(client)
        before = _state(client)
        old_generation = client.app.state.event_journal.inspect().wal_generation_id
        generation_id, rollover_id, _ = _prepare_only(client)
    with _client(tmp_path, settings=settings) as restarted:
        journal = restarted.app.state.event_journal.inspect()
        assert journal.open_wal_rollover is None
        assert _rolled(restarted)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_ABORTED
        )
        assert _rolled(restarted)[-1].rollover_id == str(rollover_id)
        assert _state(restarted) == before
        assert journal.wal_generation_id == old_generation
        assert f"{generation_id}.jsonl" not in _generations(settings)
        preserved = list(
            (Path(settings.state_wal.directory) / "generations").glob(
                f".{generation_id}.jsonl.*.invalid"
            )
        )
        assert len(preserved) == 1
        _chat(restarted, "after abort")


def test_crash_after_publish_before_completed_rolls_forward_at_startup(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        _chat(client)
        before = _state(client)
        generation_id, rollover_id, baseline_hash = _prepare_only(client)
        _publish(client, generation_id, baseline_hash)
    with _client(tmp_path, settings=settings) as restarted:
        journal = restarted.app.state.event_journal.inspect()
        assert journal.open_wal_rollover is None
        assert _rolled(restarted)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED
        )
        assert journal.wal_generation_id == str(generation_id)
        assert _state(restarted) == before
        _chat(restarted, "after roll forward")


def test_published_but_corrupt_generation_is_aborted_and_recovered(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        _chat(client)
        before = _state(client)
        generation_id, _, baseline_hash = _prepare_only(client)
        _publish(client, generation_id, baseline_hash)
    path = Path(settings.state_wal.directory) / "generations" / f"{generation_id}.jsonl"
    path.write_bytes(b"garbage\n")
    with _client(tmp_path, settings=settings) as restarted:
        assert restarted.app.state.event_journal.inspect().open_wal_rollover is None
        assert _rolled(restarted)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_ABORTED
        )
        # Existing R06 recovery repairs the active WAL from confirmed state.
        assert _state(restarted) == before
        _chat(restarted, "after repair")


def test_crash_after_completed_before_prune_is_pruned_by_the_next_rollover(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        for number in range(3):
            _chat(client, f"turn {number}")
            generation_id, rollover_id, baseline_hash = _prepare_only(client)
            _publish(client, generation_id, baseline_hash)
            client.app.state.event_journal.append_wal_rollover_completed(
                str(rollover_id)
            )
        # No pruning happened: more than three generations exist.
        assert len(_generations(settings)) > 3
    with _client(tmp_path, settings=settings) as restarted:
        _chat(restarted, "after restart")
        outcome = _force(restarted)
        assert outcome.performed
        assert outcome.pruned
        assert len(_generations(settings)) <= 3


def test_publish_failure_is_compensated_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        _chat(client)
        before = _state(client)
        wal = client.app.state.state_wal

        def failing(**_kwargs):  # type: ignore[no-untyped-def]
            raise OSError("injected")

        monkeypatch.setattr(wal, "publish_rollover_generation", failing)
        outcome = _force(client)
        assert not outcome.performed
        assert outcome.reason == "publish_failed"
        assert client.app.state.event_journal.inspect().open_wal_rollover is None
        assert _rolled(client)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_ABORTED
        )
        monkeypatch.undo()
        assert _state(client) == before
        _chat(client, "after compensation")


def test_prepare_failure_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        _chat(client)
        wal = client.app.state.state_wal

        def failing(**_kwargs):  # type: ignore[no-untyped-def]
            raise OSError("injected")

        monkeypatch.setattr(wal, "prepare_rollover_generation", failing)
        outcome = _force(client)
        assert not outcome.performed
        assert outcome.reason == "rollover_failed"
        assert _rolled(client) == []


# --- StateWAL unit behaviour ---------------------------------------------------


def _wal(tmp_path: Path) -> StateWAL:
    wal = StateWAL(tmp_path / "wal")
    wal.bootstrap(make_snapshot(10), 10)
    wal.append_transition(
        event_id=uuid4(),
        event_type="state.transition",
        event_source="test",
        processing_sequence=11,
        prior_snapshot=make_snapshot(10),
        candidate_snapshot=make_snapshot(11),
    )
    return wal


def test_prepared_generation_is_not_published(tmp_path: Path) -> None:
    wal = _wal(tmp_path)
    before = wal.inspect()
    generation_id = uuid4()
    baseline, _ = wal.prepare_rollover_generation(
        generation_id=generation_id, journal_processing_high_water=11
    )
    assert wal.inspect() == before
    assert baseline.baseline_snapshot == make_snapshot(11)
    assert baseline.predecessor_generation_hash == before.record_hashes[-1]


def test_publish_requires_the_baseline_to_chain_to_the_current_tail(
    tmp_path: Path,
) -> None:
    wal = _wal(tmp_path)
    generation_id = uuid4()
    _, baseline_hash = wal.prepare_rollover_generation(
        generation_id=generation_id, journal_processing_high_water=11
    )
    # The active generation advanced after preparation: the prepared baseline
    # no longer describes the latest state.
    wal.append_transition(
        event_id=uuid4(),
        event_type="state.transition",
        event_source="test",
        processing_sequence=12,
        prior_snapshot=make_snapshot(11),
        candidate_snapshot=make_snapshot(12),
    )
    with pytest.raises(StateWALConflictError):
        wal.publish_rollover_generation(
            generation_id=generation_id,
            baseline_record_hash=baseline_hash,
            journal_processing_high_water=11,
        )
    with pytest.raises(StateWALConflictError):
        wal.publish_rollover_generation(
            generation_id=generation_id,
            baseline_record_hash="0" * 64,
            journal_processing_high_water=11,
        )


def test_active_generation_cannot_be_quarantined(tmp_path: Path) -> None:
    wal = _wal(tmp_path)
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    with pytest.raises(StateWALConflictError):
        wal.quarantine_generation(manifest.active_generation_id, uuid4())


def _roll(wal: StateWAL, sequence: int) -> UUID:
    generation_id = uuid4()
    _, baseline_hash = wal.prepare_rollover_generation(
        generation_id=generation_id, journal_processing_high_water=sequence
    )
    wal.publish_rollover_generation(
        generation_id=generation_id,
        baseline_record_hash=baseline_hash,
        journal_processing_high_water=sequence,
    )
    return generation_id


def test_prune_keeps_active_predecessor_and_boot_anchor(tmp_path: Path) -> None:
    wal = _wal(tmp_path)
    first = wal.inspect().active_manifest
    assert first is not None
    anchored = first.active_generation_id
    record = wal.inspect().records[-1]
    wal.publish_boot_anchor(
        snapshot_sequence=11,
        snapshot_hash=wal.inspect().latest_snapshot_hash,
        generation_id=anchored,
        anchored_record_id=record.record_id,
        anchored_record_hash=wal.inspect().record_hashes[-1],
        journal_processing_high_water=11,
        journal_tail_record_id=uuid4(),
        journal_tail_record_hash="a" * 64,
        journal_lineage_id=uuid4(),
    )
    generations = [_roll(wal, 11) for _ in range(4)]
    removed = set(wal.prune_generations())
    remaining = {UUID(path.stem) for path in (wal.root / "generations").glob("*.jsonl")}
    assert remaining == {anchored, generations[-2], generations[-1]}
    assert removed == set(generations[:-2])
    assert wal.inspect().latest_snapshot_sequence == 11


def test_coordinator_skips_when_wal_tail_is_not_journal_anchored(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        _chat(client)
        wal = client.app.state.state_wal
        latest = wal.inspect()
        snapshot = latest.records[-1]
        state = (
            snapshot.baseline_snapshot
            if isinstance(snapshot, BaselineRecord)
            else snapshot.candidate_snapshot
        )
        # An uncommitted WAL tail (no Journal completion) is not a clean
        # boundary for a rollover.
        coordinator = StateWALRolloverCoordinator(
            client.app.state.event_journal,
            wal,
            StateWALRolloverPolicy(max_transitions=1),
        )
        original = wal.inspect

        def tail_moved():  # type: ignore[no-untyped-def]
            inspection = original()
            return inspection.model_copy(
                update={"record_hashes": (*inspection.record_hashes[:-1], "f" * 64)}
            )

        wal.inspect = tail_moved  # type: ignore[method-assign]
        try:
            outcome = coordinator.maybe_rollover()
        finally:
            del wal.inspect
        assert not outcome.performed
        assert outcome.reason == "wal_tail_not_anchored"
        assert state is not None


# --- independent review follow-ups -------------------------------------------


def test_transient_manifest_error_never_aborts_a_published_rollover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review P1-1: an unreadable manifest fails startup; it does not abort."""
    from suzka.runtime.state_recovery import StateRecoveryError
    from suzka.runtime.state_wal import StateWALIntegrityError

    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        _chat(client)
        before = _state(client)
        generation_id, _, baseline_hash = _prepare_only(client)
        _publish(client, generation_id, baseline_hash)

    def unreadable(self):  # type: ignore[no-untyped-def]
        raise StateWALIntegrityError("transient")

    monkeypatch.setattr(StateWAL, "read_manifest_optional", unreadable)
    with pytest.raises(StateRecoveryError):
        with _client(tmp_path, settings=settings):
            pass
    monkeypatch.undo()
    with _client(tmp_path, settings=settings) as restarted:
        assert _rolled(restarted)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED
        )
        assert restarted.app.state.event_journal.inspect().wal_generation_id == str(
            generation_id
        )
        assert _state(restarted) == before
        _chat(restarted, "after transient manifest error")


def test_unverifiable_published_generation_is_aborted_then_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review P1-1: abort of a published generation routes to R06 repair."""
    settings = _settings(tmp_path)
    with _client(tmp_path, settings=settings) as client:
        _chat(client)
        before = _state(client)
        generation_id, _, baseline_hash = _prepare_only(client)
        _publish(client, generation_id, baseline_hash)
    monkeypatch.setattr(
        StateWAL, "inspect_rollover_generation", lambda self, generation_id: None
    )
    with _client(tmp_path, settings=settings) as restarted:
        monkeypatch.undo()
        assert _rolled(restarted)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_ABORTED
        )
        manifest = restarted.app.state.state_wal.inspect().active_manifest
        assert manifest is not None
        assert manifest.active_generation_id != generation_id
        assert _state(restarted) == before
        _chat(restarted, "after repair")
    with _client(tmp_path, settings=settings) as again:
        _chat(again, "second restart")


class _Hook:
    def __init__(self) -> None:
        self.armed: str | None = None
        self.fired: list[str] = []

    def __call__(self, stage: str) -> None:
        if stage == self.armed:
            self.armed = None
            self.fired.append(stage)
            raise OSError(f"injected at {stage}")


def _hooked_client(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path)
    hook = _Hook()
    client = _client(tmp_path, settings=settings)
    client.app.state.state_wal = StateWAL(
        settings.state_wal.directory, failure_hook=hook
    )
    client.app.state.state_wal_rollover_policy = StateWALRolloverPolicy(
        max_transitions=1
    )
    return client, hook, settings


@pytest.mark.parametrize("stage", ["generation_write", "generation_fsync"])
def test_failure_before_preparation_skips_without_failing_the_turn(
    tmp_path: Path, stage: str
) -> None:
    client, hook, settings = _hooked_client(tmp_path)
    with client:
        _chat(client, "first")  # bootstrap transitions
        rolled_before = len(_rolled(client))
        hook.armed = stage
        _chat(client, "turn whose rollover fails")
        assert hook.fired == [stage]
        assert len(_rolled(client)) == rolled_before
        assert client.app.state.event_journal.inspect().open_wal_rollover is None
        _chat(client, "next turn rolls over normally")
        assert _rolled(client)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_COMPLETED
        )
    with _client(tmp_path, settings=settings) as restarted:
        _chat(restarted, "after restart")


def test_publish_stage_failure_is_aborted_in_process(tmp_path: Path) -> None:
    client, hook, _ = _hooked_client(tmp_path)
    with client:
        _chat(client, "first")
        hook.armed = "rollover_publish"
        _chat(client, "turn whose publish fails")
        assert hook.fired == ["rollover_publish"]
        assert _rolled(client)[-1].lifecycle is (
            EventLifecycle.WAL_GENERATION_ROLLED_ABORTED
        )
        _chat(client, "next turn")


def test_prune_stage_failure_is_retried_by_the_next_rollover(tmp_path: Path) -> None:
    client, hook, settings = _hooked_client(tmp_path)
    with client:
        for number in range(4):
            _chat(client, f"turn {number}")
        hook.armed = "generation_prune"
        _chat(client, "turn whose prune fails")
        assert hook.fired == ["generation_prune"]
        assert len(_generations(settings)) > 3
        _chat(client, "next turn prunes")
        assert len(_generations(settings)) <= 3


def test_completed_append_failure_is_resolved_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        _chat(client)
        journal = client.app.state.event_journal
        original = journal.append_wal_rollover_completed
        calls: list[str] = []

        def flaky(rollover_id: str) -> None:
            calls.append(rollover_id)
            if len(calls) == 1:
                raise OSError("injected")
            original(rollover_id)

        monkeypatch.setattr(journal, "append_wal_rollover_completed", flaky)
        outcome = _force(client)
        assert outcome.performed
        assert len(calls) == 2
        assert journal.inspect().open_wal_rollover is None
        _chat(client, "next")


def test_boot_anchor_generation_survives_many_rollovers_and_restart(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    client = _client(tmp_path, settings=settings)
    client.app.state.state_wal_rollover_policy = StateWALRolloverPolicy(
        max_transitions=1
    )
    with client:
        anchor = client.app.state.state_wal.inspect_boot_anchor()
        for number in range(6):
            _chat(client, f"turn {number}")
        assert f"{anchor.generation_id}.jsonl" in _generations(settings)
        assert len(_generations(settings)) <= 3
    with _client(tmp_path, settings=settings) as restarted:
        republished = restarted.app.state.state_wal.inspect_boot_anchor()
        manifest = restarted.app.state.state_wal.inspect().active_manifest
        assert manifest is not None
        assert republished.generation_id == manifest.active_generation_id
        _chat(restarted, "after restart")
