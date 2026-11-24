"""F15 DC-4: quarantine of the Experience / Semantic participants (D1 r3 sec. 4.2,
5.2, 6, 7, 10).

Real graph (Journal, StateWAL, AgentState, Chroma, the two stores); the damaged
store is a real directory and the coordinator is the real
``StartupReconciliationCoordinator``.  See ``dc4_kit.Rig``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from dc4_kit import Prepared, Rig, vanish
from dc_kit import QuarantineAccessSpy, assert_trees_equal, tree_snapshot
from suzka.memory.experience_store import ExperienceStore
from suzka.memory.semantic_store import SemanticStore
from suzka.runtime import degraded_continuation as dc
from suzka.runtime.degraded_continuation import (
    CycleRestart,
    IntegrityVerdict,
    ManifestState,
    classify_participant_error,
)
from suzka.runtime.domain_quarantine import (
    DomainQuarantinePolicy,
    ParticipantResolutionPolicy,
    create_empty_root,
    participant_floor,
    read_epoch_marker,
)
from suzka.runtime.event_journal import (
    AbortOutcome,
    EventJournalAppendError,
    EventLifecycle,
    ParticipantOutcome,
    ReconciliationReason,
    StartupParticipantOutcome,
)
from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator
from suzka.runtime.transaction_coordinator import (
    ParticipantDivergedError,
    ParticipantEvidenceAbsentError,
    ParticipantUnavailableError,
    UnsupportedParticipantReconciliationError,
)

EXPERIENCE = "memory.experience"
SEMANTIC = "memory.semantic"
LEGACY_REASON = "external_participant_reconciliation_required"


# ----------------------------------------------------------------- helpers


def cycle(rig: Rig, **policy_options: Any) -> tuple[bool, str | None]:
    """One cycle as the server runs it: ``start_cycle`` first (fresh policy and
    fresh store objects), then ``reconcile_open_transactions``."""

    policy = rig.policy(**policy_options)
    policy.start_cycle()
    rig.experience_store = ExperienceStore.from_memory_root(
        rig.settings.memory.persist_directory
    )
    rig.semantic_store = SemanticStore.from_memory_root(
        rig.settings.memory.persist_directory
    )
    return rig.coordinator(policy).reconcile_open_transactions()


def damage_experience(rig: Rig, prepared: Prepared) -> None:
    rig.experience_store.pending_path(prepared.transaction_id).write_text(
        json.dumps({"conflict": 1})
    )


def damage_semantic(rig: Rig, prepared: Prepared) -> None:
    rig.semantic_store.pending_path(prepared.transaction_id).write_text(
        json.dumps({"conflict": 1})
    )


def records_of(rig: Rig, transaction_id: str | None = None) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for record in rig.journal.inspect().records:
        if transaction_id is not None and record.transaction_id != transaction_id:
            continue
        if record.transaction_id is None and record.reconciliation_id is None:
            continue
        outcome = (
            record.participant_outcome
            or record.abort_outcome
            or record.startup_participant_outcome
        )
        rows.append(
            (
                record.lifecycle.value,
                record.participant_id or "",
                getattr(outcome, "value", ""),
            )
        )
    return rows


def terminal_of(rig: Rig, transaction_id: str) -> str | None:
    inspection = rig.journal.inspect()
    for group, name in (
        (inspection.completed_transactions, "completed"),
        (inspection.reconciled_transactions, "reconciled"),
        (inspection.aborted_transactions, "aborted"),
    ):
        if any(t.transaction_id == transaction_id for t in group):
            return name
    return None


def appended_since(rig: Rig, count: int) -> list[tuple[str, ...]]:
    """(lifecycle, participant, outcome) of the Journal records after the first
    ``count`` records."""

    rows = []
    for record in rig.journal.inspect().records[count:]:
        outcome = (
            record.participant_outcome
            or record.abort_outcome
            or record.startup_participant_outcome
        )
        rows.append(
            (record.lifecycle.value, record.participant_id or "", getattr(outcome, "value", ""))
        )
    return rows


def only_manifest(rig: Rig):  # type: ignore[no-untyped-def]
    writer = rig.policy().writer
    (epoch,) = writer.list_epochs()
    return writer.read_manifest(epoch)


# ------------------------------------------- 1. evidence absent vs resolver absent


def test_evidence_absent_is_a_subclass_and_resolver_absent_stays_plain() -> None:
    assert issubclass(ParticipantEvidenceAbsentError, UnsupportedParticipantReconciliationError)
    absent = ParticipantEvidenceAbsentError("x")
    plain = UnsupportedParticipantReconciliationError("x")
    for participant in (EXPERIENCE, SEMANTIC):
        assert classify_participant_error(participant, absent) is IntegrityVerdict.DIVERGED
        assert classify_participant_error(participant, plain) is IntegrityVerdict.PROTECTED_LOST
    for participant in ("memory.episodic", "subject.history", "session.turn"):
        assert (
            classify_participant_error(participant, absent) is IntegrityVerdict.PROTECTED_LOST
        )


def test_the_ten_evidence_absent_sites_raise_the_subclass_and_the_resolver_ones_do_not() -> None:
    root = Path(__file__).resolve().parents[1] / "suzka"
    for relative, expected in (
        ("memory/experience_participant.py", 5),
        ("memory/semantic_participant.py", 5),
    ):
        text = (root / relative).read_text()
        assert text.count("raise ParticipantEvidenceAbsentError(") == expected
        assert "raise UnsupportedParticipantReconciliationError(" not in text
    text = (root / "runtime/startup_reconciliation.py").read_text()
    assert text.count("raise UnsupportedParticipantReconciliationError(") == 8
    assert "ParticipantEvidenceAbsentError" not in text


def test_an_experience_directory_that_is_gone_raises_evidence_absent(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("gone", 1, experience=True)
    vanish(rig.experience_store.root)
    coordinator = rig.coordinator()
    with pytest.raises(ParticipantEvidenceAbsentError):
        coordinator._roll_forward_transaction(rig.journal.inspect().open_transactions[0])
    assert prepared.experience_id is not None


def test_a_semantic_directory_that_is_gone_raises_evidence_absent(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.prepare("gone-s", 1, semantic=True)
    vanish(rig.semantic_store.root)
    with pytest.raises(ParticipantEvidenceAbsentError):
        rig.coordinator()._roll_forward_transaction(
            rig.journal.inspect().open_transactions[0]
        )


# ------------------------------------- 2. the live (finalize) path, Experience


def test_experience_diverged_finalize_path_quarantines_and_ends_reconciled(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("a", 1, experience=True)
    damage_experience(rig, prepared)
    root = rig.experience_store.root
    before = tree_snapshot(root)

    with pytest.raises(CycleRestart) as restart:
        cycle(rig)
    assert restart.value.reason == "participant_quarantined"

    # the damaged tree is byte-identical, beside its origin, and the live path is gone
    quarantined = root.with_name(root.name + ".quarantine-" + only_manifest(rig).epoch_id)
    assert_trees_equal(before, tree_snapshot(quarantined))
    assert not root.exists()
    manifest = only_manifest(rig)
    assert manifest.state is ManifestState.JOURNALED
    assert manifest.artifacts[0].verified is True
    assert manifest.transaction_ids == (prepared.transaction_id,)

    # the Journal went through the EXISTING reconciliation-required path
    rows = records_of(rig, prepared.transaction_id)
    assert ("transaction_reconciliation_required", "", "") in rows
    assert ("participant_finalized", EXPERIENCE, "quarantined") in rows
    assert not any(row[0] == "transaction_completed" for row in rows)
    # every participant is resolved in the quarantining cycle, so the transaction is
    # terminal there (RECONCILED, never COMPLETED); the store objects of that cycle are
    # stale, so the cycle ended with CycleRestart instead of serving
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    (transaction,) = rig.journal.inspect().reconciled_transactions
    assert transaction.reconciliation_reason is ReconciliationReason.PARTICIPANT_DIVERGED
    assert dict(transaction.participant_outcomes)[EXPERIENCE] is ParticipantOutcome.QUARANTINED
    assert not root.exists()

    # the NEXT cycle creates the empty root + marker FIRST, then has nothing to resolve
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    assert not any(
        t.transaction_id == prepared.transaction_id
        for t in rig.journal.inspect().completed_transactions
    )
    assert sorted(p.name for p in root.iterdir()) == [dc.EPOCH_MARKER_NAME]
    assert read_epoch_marker(root) == manifest.epoch_id
    assert only_manifest(rig).state is ManifestState.COMMITTED
    # the episodic participant of the same transaction was finalized normally
    assert ("participant_finalized", "memory.episodic", "finalized") in rows or (
        "participant_finalized",
        "memory.episodic",
        "already_consistent",
    ) in rows


def test_the_quarantine_is_visible_in_health_for_the_rest_of_the_incarnation(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("a", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    cycle(rig)
    snapshot = rig.status.snapshot()
    assert snapshot["dc_reason"] == "domain_quarantined"
    assert snapshot["dc_participants"] == EXPERIENCE
    assert snapshot["dc_epoch_id"] == only_manifest(rig).epoch_id
    assert rig.status.health_fields()["dc_reason"] == "domain_quarantined"
    # a later publish (SERVING) cannot hide it, and a new process reads the manifest dir
    rig.status.publish(dc.DcState.SERVING)
    assert rig.status.snapshot()["dc_reason"] == "domain_quarantined"
    fresh = Rig.__new__(Rig)
    fresh.__dict__.update(rig.__dict__)
    fresh.status = dc.DegradedContinuationStatus()
    policy = fresh.policy()
    policy.start_cycle()
    assert fresh.status.snapshot()["dc_participants"] == EXPERIENCE


def test_experience_directory_gone_is_evidence_absent_and_quarantined_with_no_artifact(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("gone", 1, experience=True)
    root = rig.experience_store.root
    vanish(root)
    with pytest.raises(CycleRestart):
        cycle(rig)
    manifest = only_manifest(rig)
    assert manifest.artifacts == ()  # nothing existed to move
    assert manifest.state is ManifestState.JOURNALED
    assert ("participant_finalized", EXPERIENCE, "quarantined") in records_of(
        rig, prepared.transaction_id
    )
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    assert read_epoch_marker(root) == manifest.epoch_id


# ------------------------------------------------------- 3. Semantic


def test_semantic_diverged_is_quarantined_like_experience(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("s", 1, semantic=True)
    damage_semantic(rig, prepared)
    root = rig.semantic_store.root
    before = tree_snapshot(root)
    with pytest.raises(CycleRestart):
        cycle(rig)
    manifest = only_manifest(rig)
    assert manifest.participant_id == SEMANTIC
    assert_trees_equal(
        before,
        tree_snapshot(root.with_name(root.name + ".quarantine-" + manifest.epoch_id)),
    )
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    assert ("participant_finalized", SEMANTIC, "quarantined") in records_of(
        rig, prepared.transaction_id
    )
    assert read_epoch_marker(root) == manifest.epoch_id


def test_semantic_directory_gone_is_quarantined(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("s-gone", 1, semantic=True)
    vanish(rig.semantic_store.root)
    with pytest.raises(CycleRestart):
        cycle(rig)
    assert only_manifest(rig).artifacts == ()
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"


def test_both_domains_in_one_transaction_end_in_one_restart(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("both", 1, experience=True, semantic=True)
    damage_experience(rig, prepared)
    damage_semantic(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    writer = rig.policy().writer
    states = {
        writer.read_manifest(e).participant_id: writer.read_manifest(e).state
        for e in writer.list_epochs()
    }
    assert states == {
        EXPERIENCE: ManifestState.JOURNALED,
        SEMANTIC: ManifestState.JOURNALED,
    }
    rows = records_of(rig, prepared.transaction_id)
    assert ("participant_finalized", EXPERIENCE, "quarantined") in rows
    assert ("participant_finalized", SEMANTIC, "quarantined") in rows
    # only ONE reconciliation-required record: the second quarantine joins it
    assert sum(r[0] == "transaction_reconciliation_required" for r in rows) == 1
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"


# --------------------------- 4. TRANSIENT is never quarantined (HD-new-2 (a))


def test_a_transient_participant_fault_is_never_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suzka.memory.experience_store import ExperienceStoreUnavailable

    rig = Rig(tmp_path)
    prepared = rig.prepare("t", 1, experience=True)
    root = rig.experience_store.root
    before = tree_snapshot(root)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise ExperienceStoreUnavailable("EACCES for a minute")

    monkeypatch.setattr(ExperienceStore, "load_pending", unavailable)
    result = cycle(rig)
    assert result == (False, LEGACY_REASON)
    # it was PROBED inside one T_retry (60 s of sleeps, never more) and then left
    assert sum(rig.clock.slept) == pytest.approx(60.0)
    assert rig.policy().writer.list_epochs() == ()  # no manifest at all
    assert_trees_equal(before, tree_snapshot(root))  # nothing moved
    assert not any("quarantined" in row for row in records_of(rig, prepared.transaction_id))
    assert terminal_of(rig, prepared.transaction_id) is None  # still open, re-evaluated
    monkeypatch.undo()
    assert cycle(rig) == (True, None)  # the fault cleared: it FINALIZES, no quarantine
    assert terminal_of(rig, prepared.transaction_id) == "completed"
    assert ("participant_finalized", EXPERIENCE, "finalized") in records_of(
        rig, prepared.transaction_id
    )


def test_a_divergence_that_clears_on_the_confirming_re_read_is_not_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("flaky", 1, experience=True)
    real = ExperienceStore.load_pending
    calls = {"n": 0}

    def flaky(self: ExperienceStore, transaction_id: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"conflict": "once"}
        return real(self, transaction_id)

    monkeypatch.setattr(ExperienceStore, "load_pending", flaky)
    assert cycle(rig) == (True, None)
    assert rig.policy().writer.list_epochs() == ()
    assert terminal_of(rig, prepared.transaction_id) == "completed"


# ------------------------------------------------ 5. the abort path (PRE_INTERNAL)


def test_abort_path_records_abort_outcome_quarantined_and_ends_aborted(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("abort", 1, experience=True, commit_internal=False)
    damage_experience(rig, prepared)
    root = rig.experience_store.root
    before = tree_snapshot(root)
    with pytest.raises(CycleRestart):
        cycle(rig)
    manifest = only_manifest(rig)
    assert_trees_equal(
        before,
        tree_snapshot(root.with_name(root.name + ".quarantine-" + manifest.epoch_id)),
    )
    rows = records_of(rig, prepared.transaction_id)
    assert ("participant_aborted", EXPERIENCE, "quarantined") in rows
    assert terminal_of(rig, prepared.transaction_id) == "aborted"
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "aborted"
    (aborted,) = rig.journal.inspect().aborted_transactions
    assert dict(aborted.abort_outcomes)[EXPERIENCE] is AbortOutcome.QUARANTINED


def test_semantic_abort_with_an_absent_pending_artifact_is_already_absent_not_a_quarantine(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("abort-s", 1, semantic=True, commit_internal=False)
    rig.semantic_store.pending_path(prepared.transaction_id).unlink()
    assert cycle(rig) == (True, None)
    assert rig.policy().writer.list_epochs() == ()
    (aborted,) = rig.journal.inspect().aborted_transactions
    assert dict(aborted.abort_outcomes)[SEMANTIC] is AbortOutcome.ALREADY_ABSENT


# ------------------------------- 6. every open transaction needing P at once


def test_the_attach_covers_every_open_transaction_and_startup_reconciliation_it_finds(
    tmp_path: Path,
) -> None:
    """A valid Journal admits ONE open transaction at a time (an event cannot be
    terminal with an open transaction, and only one event is processed), so the
    "attach to every open transaction that needs P" rule is exercised here with
    one transaction; the code iterates ``open_transactions`` and
    ``open_startup_reconciliations`` generally (see the aggregate tests for R)."""

    rig = Rig(tmp_path)
    prepared = rig.prepare("one", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    assert only_manifest(rig).transaction_ids == (prepared.transaction_id,)
    assert not rig.journal.inspect().open_transactions


# --------------------------------------- 7. nothing but the two memory domains


def test_a_diverged_episodic_participant_is_never_quarantined(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("epi", 1, experience=True)
    assert prepared.episode is not None
    pending = prepared.episode.pending_path(
        rig.coordinator()._binding(
            rig.journal.inspect().open_transactions[0],
            next(
                r
                for r in rig.journal.inspect().open_transactions[0].required_participants
                if r.participant_id == "memory.episodic"
            ),
        )
    )
    pending.write_text(json.dumps({"conflict": 1}))
    before = tree_snapshot(rig.experience_store.root)
    assert cycle(rig) == (False, LEGACY_REASON)
    assert rig.policy().writer.list_epochs() == ()
    assert_trees_equal(before, tree_snapshot(rig.experience_store.root))


def test_the_policy_refuses_every_other_participant_and_every_other_class(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    coordinator = rig.coordinator(rig.policy())
    policy = coordinator.resolution_policy
    for participant in ("subject.history", "memory.episodic", "session.turn", None):
        for error in (
            ParticipantDivergedError("x"),
            ParticipantEvidenceAbsentError("x"),
        ):
            assert not policy.resolve_transaction_fault(coordinator, participant, error)
    for error in (
        ParticipantUnavailableError("x"),
        UnsupportedParticipantReconciliationError("resolver"),
        OSError("x"),
        KeyError("x"),
    ):
        assert not policy.resolve_transaction_fault(coordinator, EXPERIENCE, error)
    assert policy.writer.list_epochs() == ()


# ------------------------------- 8. the Journal accepts QUARANTINED only here


def test_the_journal_validator_rejects_a_quarantined_outcome_elsewhere(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("val", 1, experience=True)
    event = prepared.event
    for participant in ("memory.episodic", "subject.history", "session.turn"):
        with pytest.raises(Exception):
            rig.journal.append_participant_finalized(
                event, prepared.transaction_id, participant, "a" * 64,
                ParticipantOutcome.QUARANTINED,
            )
        with pytest.raises(Exception):
            rig.journal.append_participant_aborted(
                event, prepared.transaction_id, participant, "a" * 64,
                AbortOutcome.QUARANTINED,
            )
    with pytest.raises(Exception):
        rig.journal.append_startup_participant_reconciled(
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
            0, "0" * 64, 0,
            "00000000-0000-4000-8000-000000000003",
            "00000000-0000-4000-8000-000000000004",
            "0" * 64,
            "00000000-0000-4000-8000-000000000005",
            "memory.episodic", "a" * 64, StartupParticipantOutcome.QUARANTINED,
        )


def test_a_quarantined_outcome_cannot_precede_a_reconciliation_reason(
    tmp_path: Path,
) -> None:
    """It always goes through RECONCILIATION_REQUIRED, so the transaction can end
    only RECONCILED, never COMPLETED (R15 reads COMPLETED as 'the data exists')."""

    rig = Rig(tmp_path)
    prepared = rig.prepare("order", 1, experience=True)
    requirement = next(
        r
        for r in rig.journal.inspect().open_transactions[0].required_participants
        if r.participant_id == EXPERIENCE
    )
    with pytest.raises(EventJournalAppendError):
        rig.journal.append_participant_finalized(
            prepared.event,
            prepared.transaction_id,
            EXPERIENCE,
            requirement.operation_digest,
            ParticipantOutcome.QUARANTINED,
        )
    # the SAME append is valid once the reconciliation reason is recorded
    open_transaction = rig.journal.inspect().open_transactions[0]
    unresolved = tuple(
        sorted(
            {r.participant_id for r in open_transaction.required_participants}
            - {pid for pid, _ in open_transaction.participant_outcomes}
        )
    )
    rig.journal.append_transaction_reconciliation_required(
        prepared.event,
        prepared.transaction_id,
        ReconciliationReason.PARTICIPANT_DIVERGED,
        unresolved,
    )
    rig.journal.append_participant_finalized(
        prepared.event,
        prepared.transaction_id,
        EXPERIENCE,
        requirement.operation_digest,
        ParticipantOutcome.QUARANTINED,
    )


def test_a_transaction_that_carries_quarantined_cannot_be_completed(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("nocomplete", 1, semantic=True)
    (requirement,) = rig.journal.inspect().open_transactions[0].required_participants
    rig.journal.append_transaction_reconciliation_required(
        prepared.event,
        prepared.transaction_id,
        ReconciliationReason.PARTICIPANT_DIVERGED,
        (SEMANTIC,),
    )
    rig.journal.append_participant_finalized(
        prepared.event,
        prepared.transaction_id,
        SEMANTIC,
        requirement.operation_digest,
        ParticipantOutcome.QUARANTINED,
    )
    with pytest.raises(EventJournalAppendError):
        rig.journal.append_transaction_completed(prepared.event, prepared.transaction_id)
    rig.journal.append_transaction_reconciled(prepared.event, prepared.transaction_id)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"


# ----------------------------- 9. legacy equivalence (no policy / mode refuse)


@pytest.mark.parametrize("shape", ["diverged", "gone"])
def test_without_the_policy_the_behaviour_is_the_legacy_one(
    tmp_path: Path, shape: str
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("legacy", 1, experience=True)
    if shape == "diverged":
        damage_experience(rig, prepared)
    else:
        vanish(rig.experience_store.root)
    count_before = len(rig.journal.inspect().records)
    names_before = sorted(p.name for p in tmp_path.iterdir())
    coordinator = rig.coordinator()  # default policy
    assert isinstance(coordinator.resolution_policy, ParticipantResolutionPolicy)
    assert not isinstance(coordinator.resolution_policy, DomainQuarantinePolicy)
    assert coordinator.reconcile_open_transactions() == (False, LEGACY_REASON)
    # legacy progress only (the episodic participant finalizes before the fault):
    # no reconciliation-required, no quarantine, no file
    assert {row[0] for row in appended_since(rig, count_before)} <= {"participant_finalized"}
    assert all(row[1] == "memory.episodic" for row in appended_since(rig, count_before))
    assert sorted(p.name for p in tmp_path.iterdir()) == names_before
    assert not any(
        "quarantined" in row for row in records_of(rig, prepared.transaction_id)
    )
    assert terminal_of(rig, prepared.transaction_id) is None
    assert not (tmp_path / "quarantine").exists()


# ------------------------------------------- 10. rename failure, authority


def test_a_rename_failure_moves_nothing_and_writes_no_journal_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("eacces", 1, experience=True)
    damage_experience(rig, prepared)
    root = rig.experience_store.root
    before = tree_snapshot(root)
    count_before = len(rig.journal.inspect().records)

    def denied(source: Path, target: Path) -> None:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(dc, "_rename_directory", denied)
    assert cycle(rig) == (False, LEGACY_REASON)
    assert_trees_equal(before, tree_snapshot(root))
    # no quarantine record: only the legacy episodic progress
    assert not any(
        row[0] == "transaction_reconciliation_required" or row[2] == "quarantined"
        for row in appended_since(rig, count_before)
    )
    assert terminal_of(rig, prepared.transaction_id) is None
    # a later cycle with a working rename resumes from the PENDING manifest
    monkeypatch.undo()
    with pytest.raises(CycleRestart):
        cycle(rig)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"


def test_without_the_journal_lease_nothing_is_done(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("lease", 1, experience=True)
    damage_experience(rig, prepared)
    root = rig.experience_store.root
    before = tree_snapshot(root)
    count_before = len(rig.journal.inspect().records)
    policy = rig.policy(has_exclusive_authority=lambda: False)
    result = rig.coordinator(policy).reconcile_open_transactions()
    assert result == (False, LEGACY_REASON)
    assert_trees_equal(before, tree_snapshot(root))
    assert not any(
        row[0] == "transaction_reconciliation_required" or row[2] == "quarantined"
        for row in appended_since(rig, count_before)
    )
    assert policy.writer.list_epochs() == ()


# ------------------------------------------------------- 11. the empty root


def test_the_empty_root_is_created_with_its_marker_and_never_over_an_existing_path(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "p"
    parent.mkdir(mode=0o700)
    root = parent / "experience"
    create_empty_root(root, epoch_id="e1", participant_id=EXPERIENCE)
    assert [p.name for p in root.iterdir()] == [dc.EPOCH_MARKER_NAME]
    assert read_epoch_marker(root) == "e1"
    assert oct(root.stat().st_mode & 0o777) == "0o700"
    with pytest.raises(dc.QuarantineError):
        create_empty_root(root, epoch_id="e2", participant_id=EXPERIENCE)
    assert read_epoch_marker(root) == "e1"  # untouched
    # a leftover temporary of THIS epoch that is exactly the marker is re-used
    other = parent / "semantic"
    leftover = parent / "semantic.dc-new-e3"
    leftover.mkdir(mode=0o700)
    (leftover / dc.EPOCH_MARKER_NAME).write_text(json.dumps({"epoch_id": "e3"}))
    create_empty_root(other, epoch_id="e3", participant_id=SEMANTIC)
    assert read_epoch_marker(other) == "e3"
    # a leftover that is not the marker only is refused, not deleted
    bad = parent / "third.dc-new-e4"
    bad.mkdir(mode=0o700)
    (bad / "stray").write_text("x")
    with pytest.raises(dc.QuarantineError):
        create_empty_root(parent / "third", epoch_id="e4", participant_id=EXPERIENCE)
    assert (bad / "stray").exists()


def test_the_store_works_on_the_empty_root_and_a_sibling_quarantine_name(
    tmp_path: Path,
) -> None:
    """S0-d: a marker file in the root and a ``*.quarantine-*`` sibling do not
    disturb either store."""

    rig = Rig(tmp_path)
    prepared = rig.prepare("d", 1, experience=True, semantic=True)
    damage_experience(rig, prepared)
    damage_semantic(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    assert cycle(rig) == (True, None)
    rig.recovery.prepare_startup()  # the event of the first transaction ends
    assert any(".quarantine-" in p.name for p in tmp_path.iterdir())
    # a NEW transaction on the new epoch works end to end
    second = rig.prepare("d2", 2, experience=True, semantic=True)
    assert cycle(rig) == (True, None)
    assert terminal_of(rig, second.transaction_id) == "completed"
    assert dc.EPOCH_MARKER_NAME in {p.name for p in rig.experience_store.root.iterdir()}


# -------------------------------- 12. the four-state manifest crash matrix


class _Crash(Exception):
    pass


def _first_cycle_crashing_at(rig: Rig, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    """Run the quarantining cycle and die right AFTER the manifest reached
    ``state`` (the next manifest write raises)."""

    writer_cls = dc.QuarantineWriter
    real_write = writer_cls._write_manifest
    seen: list[str] = []

    def write(self: dc.QuarantineWriter, manifest: dc.QuarantineManifest) -> None:
        if state in seen:  # the state already reached disk: die on the next write
            raise _Crash
        real_write(self, manifest)
        seen.append(manifest.state.value)

    monkeypatch.setattr(writer_cls, "_write_manifest", write)
    try:
        # _Crash is raised inside the policy's guarded section, so it surfaces
        # as the legacy result or as a plain exception; both are a dead cycle.
        try:
            cycle(rig)
        except (CycleRestart, _Crash):
            pass
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("state", ["PENDING", "MOVED", "JOURNALED"])
def test_crash_after_each_manifest_state_resumes_to_one_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("crash", 1, experience=True)
    damage_experience(rig, prepared)
    root = rig.experience_store.root
    before = tree_snapshot(root)
    _first_cycle_crashing_at(rig, monkeypatch, state)
    (epoch,) = rig.policy().writer.list_epochs()
    assert rig.policy().writer.read_manifest(epoch).state.value == state
    # a restart: cycles until nothing is left to do (bounded)
    for _ in range(4):
        try:
            result = cycle(rig)
        except CycleRestart:
            continue
        if result == (True, None):
            break
    else:  # pragma: no cover
        pytest.fail("the quarantine did not resume")
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    assert len(rig.policy().writer.list_epochs()) == 1  # ONE quarantine, never two
    manifest = rig.policy().writer.read_manifest(epoch)
    assert manifest.state is ManifestState.COMMITTED
    assert_trees_equal(
        before,
        tree_snapshot(root.with_name(root.name + ".quarantine-" + epoch)),
    )
    rows = records_of(rig, prepared.transaction_id)
    assert sum(r == ("participant_finalized", EXPERIENCE, "quarantined") for r in rows) == 1
    assert read_epoch_marker(root) == epoch


def test_a_crash_inside_step_three_between_two_quarantines_resumes_without_a_double(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("two-domains", 1, experience=True, semantic=True)
    damage_experience(rig, prepared)
    damage_semantic(rig, prepared)
    real = type(rig.journal).append_participant_finalized

    def dying(self: Any, *args: Any, **kwargs: Any) -> None:
        if args[2] == SEMANTIC and args[4] is ParticipantOutcome.QUARANTINED:
            raise OSError(5, "EIO")  # the Journal fails inside the SEMANTIC step 3
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(rig.journal), "append_participant_finalized", dying)
    with pytest.raises(CycleRestart):  # the Experience quarantine happened: restart
        cycle(rig)
    monkeypatch.undo()
    writer = rig.policy().writer
    states = {
        writer.read_manifest(e).participant_id: writer.read_manifest(e).state
        for e in writer.list_epochs()
    }
    assert states == {
        EXPERIENCE: ManifestState.JOURNALED,
        SEMANTIC: ManifestState.MOVED,  # resumable
    }
    with pytest.raises(CycleRestart):  # begin() continues the Semantic step 3
        cycle(rig)
    assert cycle(rig) == (True, None)
    rows = records_of(rig, prepared.transaction_id)
    for participant in (EXPERIENCE, SEMANTIC):
        assert sum(r == ("participant_finalized", participant, "quarantined") for r in rows) == 1
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    assert len(writer.list_epochs()) == 2


def test_the_journaled_resume_creates_the_empty_root_before_any_store_exists(
    tmp_path: Path,
) -> None:
    """A JOURNALED manifest is the trigger: ``start_cycle`` makes the empty root
    and the marker with NO store object and NO Journal reconciliation."""

    rig = Rig(tmp_path)
    prepared = rig.prepare("jr", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    root = rig.experience_store.root
    assert not root.exists()
    policy = rig.policy()
    policy.start_cycle()  # nothing else has run
    assert policy.start_block is None
    assert read_epoch_marker(root) == only_manifest(rig).epoch_id
    assert only_manifest(rig).state is ManifestState.COMMITTED
    # idempotent
    policy.start_cycle()
    assert policy.start_block is None


# ------------------------------------------- 13. the hash-mismatch exit


def test_a_hash_mismatch_after_the_move_has_a_bounded_human_free_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("mm", 1, experience=True)
    damage_experience(rig, prepared)
    root = rig.experience_store.root
    real_rename = dc._rename_directory

    def rename_then_change(source: Path, target: Path) -> None:
        real_rename(source, target)
        # a straggler changes one byte of the moved tree: same size, other bytes
        victim = next(target.rglob("*.json"))
        victim.write_bytes(b"X" * victim.stat().st_size)

    monkeypatch.setattr(dc, "_rename_directory", rename_then_change)
    assert cycle(rig) == (False, LEGACY_REASON)
    monkeypatch.undo()
    manifest = only_manifest(rig)
    assert manifest.state is ManifestState.PENDING  # not accepted yet
    assert manifest.hash_mismatch is not None and manifest.hash_mismatch.accepted_at is None
    quarantined = root.with_name(root.name + ".quarantine-" + manifest.epoch_id)
    after_change = tree_snapshot(quarantined)

    # inside the bound: still blocked, nothing else happens
    assert cycle(rig) == (False, "domain_quarantine_blocked")
    assert not root.exists()
    assert only_manifest(rig).state is ManifestState.PENDING

    # past the bound (here: the observation count / a zero bound), the exit is taken
    outcome: tuple[bool, str | None] | None = None
    for _ in range(dc.HASH_MISMATCH_MAX_OBSERVATIONS + 2):
        try:
            outcome = cycle(rig)
        except CycleRestart:
            continue
        if outcome == (True, None):
            break
    assert outcome == (True, None)
    final = only_manifest(rig)
    assert final.state is ManifestState.COMMITTED
    assert final.hash_mismatch is not None and final.hash_mismatch.accepted_at is not None
    assert final.hash_mismatch.differing  # the manifest names the differing paths
    assert final.artifacts[0].verified is None  # NOT claimed verified
    # the damaged artifact is byte-preserved exactly as it was found (never repaired)
    assert_trees_equal(after_change, tree_snapshot(quarantined))
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"
    assert read_epoch_marker(root) == final.epoch_id


def test_the_hash_mismatch_bound_also_ends_by_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta, timezone

    rig = Rig(tmp_path)
    prepared = rig.prepare("mm2", 1, experience=True)
    damage_experience(rig, prepared)
    real_rename = dc._rename_directory

    def rename_then_change(source: Path, target: Path) -> None:
        real_rename(source, target)
        victim = next(target.rglob("*.json"))
        victim.write_bytes(b"Y" * victim.stat().st_size)

    clock = {"now": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    monkeypatch.setattr(dc, "_rename_directory", rename_then_change)
    cycle(rig, wall_clock=lambda: clock["now"])
    monkeypatch.undo()
    assert cycle(rig, wall_clock=lambda: clock["now"]) == (False, "domain_quarantine_blocked")
    clock["now"] += timedelta(seconds=dc.HASH_MISMATCH_BOUND_S + 1)
    with pytest.raises(CycleRestart):  # the exit is taken: step 3 runs
        cycle(rig, wall_clock=lambda: clock["now"])
    assert only_manifest(rig).hash_mismatch is not None
    assert cycle(rig, wall_clock=lambda: clock["now"]) == (True, None)
    assert terminal_of(rig, prepared.transaction_id) == "reconciled"


# ----------------------------------------- 14. a restored store (FOREIGN_RESTORE)


def test_a_restored_directory_over_a_quarantined_store_is_detected_and_blocks(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("fr", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    assert cycle(rig) == (True, None)
    root = rig.experience_store.root
    epoch = only_manifest(rig).epoch_id
    quarantined = root.with_name(root.name + ".quarantine-" + epoch)
    # the Human puts a copy of the quarantined tree back over the live root
    live = root.with_name(root.name + ".live-aside")
    root.rename(live)
    import shutil

    shutil.copytree(quarantined, root)
    snapshot_root = tree_snapshot(root)
    snapshot_q = tree_snapshot(quarantined)
    journal_before = rig.journal.path.read_bytes()
    result = cycle(rig)
    assert result == (False, "domain_store_replaced")
    # no automatic action at all
    assert_trees_equal(snapshot_root, tree_snapshot(root))
    assert_trees_equal(snapshot_q, tree_snapshot(quarantined))
    assert rig.journal.path.read_bytes() == journal_before
    assert len(rig.policy().writer.list_epochs()) == 1


# ------------------------------------------- 15. never reads a quarantined tree


def test_the_live_process_never_opens_a_quarantined_path_after_the_quarantine(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("spy", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)  # the quarantine action itself reads the tree it moves
    rig.recovery.prepare_startup()
    with QuarantineAccessSpy() as refused:
        assert cycle(rig) == (True, None)
        second = rig.prepare("spy2", 2, experience=True)
        assert cycle(rig) == (True, None)
        assert terminal_of(rig, second.transaction_id) == "completed"
    assert refused == []


def test_no_row_continues_an_identifier_that_exists_only_in_the_quarantined_tree(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    first = rig.prepare("old", 1, experience=True)
    damage_experience(rig, first)
    with pytest.raises(CycleRestart):
        cycle(rig)
    cycle(rig)
    rig.recovery.prepare_startup()
    second = rig.prepare("new", 2, experience=True)
    assert cycle(rig) == (True, None)
    assert first.experience_id is not None and second.experience_id is not None
    # the identifiers that exist only in the quarantined artifact
    root = rig.experience_store.root
    quarantined = root.with_name(root.name + ".quarantine-" + only_manifest(rig).epoch_id)
    old_only = {
        p.relative_to(quarantined).as_posix()
        for p in quarantined.rglob("*")
        if first.experience_id in p.as_posix() or first.transaction_id in p.as_posix()
    }
    assert old_only
    new_paths = {p.relative_to(root).as_posix() for p in root.rglob("*")}
    assert not (new_paths & old_only)
    # the live store holds no byte of the old record's identity
    for path in root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert first.experience_id.encode() not in data
    # the manifest holds hashes, never payload bytes
    manifest_text = (
        rig.qdir / only_manifest(rig).epoch_id / "manifest.json"
    ).read_text()
    assert '"conflict"' not in manifest_text


def test_no_lineage_or_baseline_record_is_appended_by_a_quarantine(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("lin", 1, experience=True)
    damage_experience(rig, prepared)
    before = rig.journal.inspect()
    with pytest.raises(CycleRestart):
        cycle(rig)
    cycle(rig)
    after = rig.journal.inspect()
    assert after.journal_lineage_id == before.journal_lineage_id
    assert after.baselines == before.baselines
    new = {r.lifecycle for r in after.records[len(before.records):]}
    assert new <= {
        EventLifecycle.TRANSACTION_RECONCILIATION_REQUIRED,
        EventLifecycle.PARTICIPANT_FINALIZED,
        EventLifecycle.TRANSACTION_RECONCILED,
    }
    assert os.path.exists(rig.journal.path)


# ============================================ 16. the aggregate (true-rollback) path


def _true_rollback(rig: Rig, name: str = "agg") -> tuple[Prepared, Any]:
    """A completed Experience transaction, then a true rollback of AgentState
    (the same fault the existing gate tests inject)."""

    boot = rig.recovery.prepare_startup()
    rig.recovery.publish_boot_anchor(boot)
    prepared = rig.prepare(name, 1, experience=True)
    rig.transactions.finalize_event(prepared.event, prepared.evidence)  # type: ignore[arg-type]
    rig.recovery.complete_committed_event(prepared.event, prepared.evidence)  # type: ignore[arg-type]
    manifest = rig.wal.inspect().active_manifest
    assert manifest is not None
    generation = rig.wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    rig.state.path.unlink()
    from suzka.runtime.state_recovery import StateRecoveryCoordinator

    rolled_back = StateRecoveryCoordinator(
        rig.state, rig.journal, rig.wal
    ).prepare_startup()
    assert rolled_back.true_rollback_performed
    assert rolled_back.external_reconciliation_required
    return prepared, rolled_back


def _damage_committed_experience(rig: Rig, prepared: Prepared) -> None:
    record = next(
        p
        for p in (rig.experience_store.records_root / str(prepared.experience_id)).glob("*.json")
    )
    record.write_bytes(b'{"damaged": true}')


def gate(rig: Rig, rolled_back: Any, **policy_options: Any) -> Any:
    policy = rig.policy(**policy_options)
    policy.start_cycle()
    rig.experience_store = ExperienceStore.from_memory_root(
        rig.settings.memory.persist_directory
    )
    rig.semantic_store = SemanticStore.from_memory_root(
        rig.settings.memory.persist_directory
    )
    return rig.coordinator(policy).reconcile_recovery_gate(rolled_back)


def test_aggregate_path_quarantines_restarts_and_the_same_reconciliation_resumes(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared, rolled_back = _true_rollback(rig)
    _damage_committed_experience(rig, prepared)
    root = rig.experience_store.root
    before = tree_snapshot(root)

    with pytest.raises(CycleRestart):
        gate(rig, rolled_back)
    (open_reconciliation,) = rig.journal.inspect().open_startup_reconciliations
    reconciliation_id = open_reconciliation.reconciliation_id
    digests = {
        item.participant_id: item.operation_digest
        for item in open_reconciliation.required_participants
    }
    outcomes = {pid: outcome for pid, _digest, outcome in open_reconciliation.participant_outcomes}
    assert outcomes[EXPERIENCE] is StartupParticipantOutcome.QUARANTINED
    manifest = only_manifest(rig)
    assert manifest.state is ManifestState.JOURNALED
    assert_trees_equal(
        before,
        tree_snapshot(root.with_name(root.name + ".quarantine-" + manifest.epoch_id)),
    )

    # the NEXT cycle: the SAME R (same id, unchanged requirement digests) resumes
    result = gate(rig, rolled_back)
    assert result.participants_consistent
    assert not result.recovery.external_reconciliation_required
    inspection = rig.journal.inspect()
    assert not inspection.open_startup_reconciliations
    (completed,) = inspection.completed_startup_reconciliations
    assert completed.reconciliation_id == reconciliation_id
    assert {i.participant_id: i.operation_digest for i in completed.required_participants} == digests
    assert dict((p, o) for p, _d, o in completed.participant_outcomes)[EXPERIENCE] is (
        StartupParticipantOutcome.QUARANTINED
    )
    assert only_manifest(rig).state is ManifestState.COMMITTED
    assert read_epoch_marker(root) == manifest.epoch_id


def test_the_floor_comes_from_journal_records_only_and_a_later_reconciliation_is_not_stuck(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared, rolled_back = _true_rollback(rig)
    _damage_committed_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        gate(rig, rolled_back)
    result = gate(rig, rolled_back)
    assert result.participants_consistent
    inspection = rig.journal.inspect()
    assert participant_floor(inspection, EXPERIENCE) == 1  # R's recovery high-water
    assert participant_floor(inspection, SEMANTIC) == 0
    assert participant_floor(inspection, "memory.episodic") == 0
    # a sidecar can never narrow verification: with every manifest gone the floor
    # is unchanged, because it is a function of the Journal alone
    import shutil

    shutil.rmtree(rig.qdir)
    assert participant_floor(rig.journal.inspect(), EXPERIENCE) == 1
    # a LATER reconciliation (a second true rollback) over the same history and the
    # new empty store: verified, never re-quarantined, never stuck
    rig.recovery.prepare_startup()
    second = rig.prepare("agg2", 2, experience=True)
    rig.transactions.finalize_event(second.event, second.evidence)  # type: ignore[arg-type]
    rig.recovery.complete_committed_event(second.event, second.evidence)  # type: ignore[arg-type]
    # the floor excludes only transactions at or below the quarantined high-water
    transactions = rig.journal.inspect().completed_transactions
    assert {t.processing_sequence for t in transactions} >= {1, 2}
    coordinator = rig.coordinator()
    verdict = coordinator._reconcile_aggregate_participant(EXPERIENCE, tuple(transactions))
    assert verdict.value == "verified_consistent"  # tx 1 skipped, tx 2 verified


def test_the_floor_is_not_an_input_of_the_requirement_digest_or_the_transaction_list(
    tmp_path: Path,
) -> None:
    """P1-A: ``participant_floor`` is used ONLY inside
    ``_reconcile_aggregate_participant`` (AST), so a quarantine cannot change the
    requirement digest of an open reconciliation."""

    import ast

    source = (Path(__file__).resolve().parents[1] / "suzka" / "runtime" / "startup_reconciliation.py")
    tree = ast.parse(source.read_text())
    users: set[str] = set()
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                if isinstance(node, ast.Name) and node.id == "participant_floor":
                    users.add(function.name)
    assert users == {"_reconcile_aggregate_participant"}
    # and the module that defines it is read by nothing else in suzka/
    root = Path(__file__).resolve().parents[1] / "suzka"
    for path in root.rglob("*.py"):
        if path.name in {"domain_quarantine.py", "startup_reconciliation.py"}:
            continue
        assert "participant_floor" not in path.read_text(), path


def test_the_aggregate_requirements_do_not_change_across_the_quarantine(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared, rolled_back = _true_rollback(rig)
    before = rig.journal.inspect()
    _damage_committed_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        gate(rig, rolled_back)
    (opened,) = rig.journal.inspect().open_startup_reconciliations
    baseline = before.baselines[-1]
    transactions = tuple(
        sorted(
            rig.journal.inspect().completed_transactions,
            key=lambda item: item.transaction_id,
        )
    )
    recomputed = StartupReconciliationCoordinator._aggregate_requirements(
        transactions, baseline.participant_registry
    )
    assert opened.required_participants == recomputed


def test_the_gate_with_the_legacy_policy_is_unchanged_for_a_damaged_store(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    prepared, rolled_back = _true_rollback(rig)
    _damage_committed_experience(rig, prepared)
    result = rig.coordinator().reconcile_recovery_gate(rolled_back)
    assert not result.participants_consistent
    assert result.degraded_reason == LEGACY_REASON
    assert rig.policy().writer.list_epochs() == ()
    assert rig.experience_store.root.exists()


# ===================================================== 17. S0-c: Semantic is OFF


def test_the_default_policy_quarantines_experience_only(tmp_path: Path) -> None:
    from suzka.runtime.domain_quarantine import DEFAULT_QUARANTINE_PARTICIPANTS

    rig = Rig(tmp_path)
    assert DEFAULT_QUARANTINE_PARTICIPANTS == {EXPERIENCE}
    policy = rig.policy(participants=None)  # the production default
    assert policy.enabled_participants == {EXPERIENCE}
    coordinator = rig.coordinator(policy)
    error = ParticipantDivergedError("x")
    assert not policy.wants_confirmation(SEMANTIC, error)
    assert not policy.resolve_transaction_fault(coordinator, SEMANTIC, error)
    assert policy.wants_confirmation(EXPERIENCE, error)
    with pytest.raises(ValueError):
        rig.policy(participants=frozenset({"subject.history"}))


def test_with_the_default_policy_a_semantic_fault_is_left_as_it_was(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("s-off", 1, semantic=True)
    damage_semantic(rig, prepared)
    before = tree_snapshot(rig.semantic_store.root)
    assert cycle(rig, participants=None) == (False, LEGACY_REASON)
    assert_trees_equal(before, tree_snapshot(rig.semantic_store.root))
    assert rig.policy().writer.list_epochs() == ()
    assert terminal_of(rig, prepared.transaction_id) is None


def test_an_existing_manifest_of_a_disabled_participant_is_still_finished(
    tmp_path: Path,
) -> None:
    """Disabling a participant never strands a quarantine that already happened."""

    rig = Rig(tmp_path)
    prepared = rig.prepare("s-on", 1, semantic=True)
    damage_semantic(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)  # enabled: the quarantine happens
    assert cycle(rig, participants=None) == (True, None)  # default policy finishes it
    assert only_manifest(rig).state is ManifestState.COMMITTED


# ============================================ 18. S0-b: what an unknown value does


def test_an_unknown_outcome_value_fails_closed_and_the_journal_is_not_touched(
    tmp_path: Path,
) -> None:
    """S0-b, the mechanism.  An OLD binary has no ``QUARANTINED`` member, which is
    exactly an unknown outcome value to it.  The reader is pydantic validation of
    each record line: an unknown enum value is an invalid record and the load
    raises ``EventJournalLoadError`` (fail closed); nothing is truncated,
    repaired or rewritten.  (The same check was run against the real code of
    40e0f59 on a real Journal that contains the value: see the DC-4 report.)"""

    from suzka.runtime.event_journal import EventJournal, EventJournalLoadError

    rig = Rig(tmp_path)
    prepared = rig.prepare("old-binary", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    path = rig.journal.path
    rig.journal.close()
    lines = path.read_bytes().splitlines(keepends=True)
    assert any(b'"participant_outcome":"quarantined"' in line for line in lines)
    forged = path.read_bytes().replace(b'"quarantined"', b'"quarantined_from_the_future"')
    path.write_bytes(forged)
    with pytest.raises(EventJournalLoadError, match="invalid record"):
        EventJournal(path, 1_000_000, 4).inspect()
    assert path.read_bytes() == forged  # fail closed: not one byte changed


def test_a_forged_quarantined_record_for_another_participant_is_not_loadable(
    tmp_path: Path,
) -> None:
    """The validator is on the READ side too: a Journal line carrying QUARANTINED
    for ``memory.episodic`` is an invalid record."""

    from suzka.runtime.event_journal import EventJournal, EventJournalLoadError

    rig = Rig(tmp_path)
    prepared = rig.prepare("forged", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    path = rig.journal.path
    rig.journal.close()
    text = path.read_text().splitlines(keepends=True)
    out = []
    for line in text:
        record = json.loads(line)
        if record.get("participant_outcome") == "quarantined":
            record["participant_id"] = "memory.episodic"
        out.append(json.dumps(record, separators=(",", ":")) + "\n")
    path.write_text("".join(out))
    with pytest.raises(EventJournalLoadError):
        EventJournal(path, 1_000_000, 4).inspect()


# ============================================ 19. readers of the outcome lists


def test_a_quarantined_outcome_is_never_read_as_the_data_exists(tmp_path: Path) -> None:
    """S0-b audit: the two readers that asked "does a Semantic outcome exist" now
    ask "is it FINALIZED / ALREADY_CONSISTENT"."""

    from types import SimpleNamespace

    from suzka.runtime.event_journal import (
        EventJournalTransaction,
        ParticipantCapability,
        ParticipantRequirement,
        TransactionKind,
    )
    from suzka.runtime.semantic_receipt_retention import SemanticReceiptRetentionCoordinator
    from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType

    requirement = ParticipantRequirement(
        participant_id=SEMANTIC,
        operation_digest="a" * 64,
        capabilities=(
            ParticipantCapability.IDEMPOTENT_FINALIZE,
            ParticipantCapability.PREPARE,
        ),
    )

    def transaction(outcome: ParticipantOutcome) -> EventJournalTransaction:
        return EventJournalTransaction(
            transaction_id="00000000-0000-4000-8000-000000000001",
            event_id="e",
            event_type=AgentEventType.SLEEP,
            source=AgentEventSource.API_SLEEP_RUN,
            processing_sequence=1,
            kind=TransactionKind.EVENT_MUTATION,
            required_participants=(requirement,),
            participant_outcomes=((SEMANTIC, outcome),),
        )

    def proofs(outcome: ParticipantOutcome) -> dict[str, str]:
        inspection = SimpleNamespace(
            completed_transactions=(),
            reconciled_transactions=(transaction(outcome),),
            baselines=(SimpleNamespace(processing_high_water=10),),
        )
        coordinator = SemanticReceiptRetentionCoordinator.__new__(
            SemanticReceiptRetentionCoordinator
        )
        coordinator.journal = SimpleNamespace(inspect=lambda: inspection)  # type: ignore[assignment]
        coordinator._verified_checkpoint_horizon = lambda _i: None  # type: ignore[method-assign]
        return coordinator._retired_transaction_proofs()

    assert proofs(ParticipantOutcome.FINALIZED) == {
        "00000000-0000-4000-8000-000000000001": "a" * 64
    }
    assert proofs(ParticipantOutcome.QUARANTINED) == {}

    # and the DB2 projection reconciliation
    entry = SimpleNamespace(
        revision=SimpleNamespace(event_id="e", event_sequence=1, semantic_id="s"),
        operation_digest="a" * 64,
    )

    def projection(outcome: ParticipantOutcome) -> tuple[bool, str | None]:
        inspection = SimpleNamespace(
            completed_transactions=(),
            reconciled_transactions=(transaction(outcome),),
        )
        coordinator = StartupReconciliationCoordinator.__new__(StartupReconciliationCoordinator)
        coordinator.semantic_store = SimpleNamespace(iter_current=lambda: (entry,))  # type: ignore[assignment]
        coordinator.journal = SimpleNamespace(inspect=lambda: inspection)  # type: ignore[assignment]
        coordinator.memory = SimpleNamespace(  # type: ignore[assignment]
            inspect_semantic_projection=lambda *_a: SimpleNamespace(status=None),
            project_semantic_revision=lambda *_a: None,
        )
        return coordinator.reconcile_terminal_semantic_projections()

    assert projection(ParticipantOutcome.FINALIZED) == (True, None)
    assert projection(ParticipantOutcome.QUARANTINED) == (
        False,
        "semantic_terminal_proof_unavailable",
    )


# ===================================================== 20. T_retry probes (TRANSIENT)


def test_a_transient_fault_that_clears_inside_t_retry_finalizes_in_the_same_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suzka.memory.experience_store import ExperienceStoreUnavailable

    rig = Rig(tmp_path)
    prepared = rig.prepare("clears", 1, experience=True)
    real = ExperienceStore.load_pending
    state = {"calls": 0}

    def flaky(self: ExperienceStore, transaction_id: str) -> Any:
        state["calls"] += 1
        if state["calls"] <= 3:
            raise ExperienceStoreUnavailable("EIO for a moment")
        return real(self, transaction_id)

    monkeypatch.setattr(ExperienceStore, "load_pending", flaky)
    assert cycle(rig) == (True, None)  # NO restart, NO quarantine
    assert rig.policy().writer.list_epochs() == ()
    assert terminal_of(rig, prepared.transaction_id) == "completed"
    assert ("participant_finalized", EXPERIENCE, "finalized") in records_of(
        rig, prepared.transaction_id
    )
    assert rig.clock.slept[:3] == [1.0, 2.0, 4.0]  # the schedule 1, 2, 4, 8, 16, 29


def test_the_probes_share_one_deadline_and_a_disabled_participant_is_not_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from suzka.memory.experience_store import ExperienceStoreUnavailable
    from suzka.memory.semantic_store import SemanticStoreUnavailable

    rig = Rig(tmp_path)
    rig.prepare("shared", 1, experience=True)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise ExperienceStoreUnavailable("down")

    monkeypatch.setattr(ExperienceStore, "load_pending", unavailable)
    policy = rig.policy(t_retry_s=10.0)
    policy.start_cycle()
    coordinator = rig.coordinator(policy)
    assert coordinator.reconcile_open_transactions() == (False, LEGACY_REASON)
    assert sum(rig.clock.slept) == pytest.approx(10.0)
    # the deadline is spent: a second pass in the same attempt does NOT probe again
    slept = len(rig.clock.slept)
    assert coordinator.reconcile_open_transactions() == (False, LEGACY_REASON)
    assert len(rig.clock.slept) == slept
    # a participant the policy does not cover (Semantic, off by default) is not probed
    default = rig.policy(participants=None)
    assert not default.wants_retry(SEMANTIC, ParticipantUnavailableError("x"))
    assert default.wants_retry(EXPERIENCE, ParticipantUnavailableError("x"))
    assert not default.wants_retry(EXPERIENCE, ParticipantDivergedError("x"))
    assert not default.wants_retry("subject.history", ParticipantUnavailableError("x"))
    assert SemanticStoreUnavailable  # imported for the class-name classification


# ============================================ 21. review fixes


def test_a_journaled_manifest_is_not_acted_on_without_the_lease(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("lease2", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    policy = rig.policy(has_exclusive_authority=lambda: False)
    policy.start_cycle()
    assert policy.start_block == "domain_quarantine_resume_failed"
    assert not rig.experience_store.root.exists()  # no empty root without the lease
    assert only_manifest(rig).state is ManifestState.JOURNALED


def test_a_missing_quarantined_tree_has_its_own_reason(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    prepared = rig.prepare("gone-tree", 1, experience=True)
    damage_experience(rig, prepared)
    with pytest.raises(CycleRestart):
        cycle(rig)
    root = rig.experience_store.root
    root.with_name(root.name + ".quarantine-" + only_manifest(rig).epoch_id).rename(
        tmp_path / "moved-by-hand"
    )
    policy = rig.policy()
    policy.start_cycle()
    assert policy.start_block == "domain_quarantine_tree_missing"
    assert not root.exists()  # nothing created, nothing guessed


def test_start_cycle_resets_the_per_cycle_quarantine_record(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    policy = rig.policy()
    policy.quarantined_this_cycle.append(EXPERIENCE)
    policy.start_cycle()
    assert policy.quarantined_this_cycle == [] and not policy.should_restart()
