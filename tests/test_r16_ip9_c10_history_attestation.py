"""IP-9 C10: SubjectHistory attestation across Journal compaction (#376 C10).

Compaction of a Journal that registers ``subject.history`` requires an
attestation from the provider; afterwards the head reader proves rows at or
below the archived high-water from the attestation (SQLite layer) and rows
above it from the retained R07 records, so adoption survives archiving.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
import shutil
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parent))

from suzka.r16.history_attestation import (  # noqa: E402
    SubjectHistoryAttestationProvider,
)
from suzka.r16.history_codec import GENESIS  # noqa: E402
from suzka.r16.history_reader import HeadBlockReason, HeadStatus  # noqa: E402
from suzka.r16.history_store import SubjectHistoryStore  # noqa: E402
from suzka.runtime.event_journal import (  # noqa: E402
    COMPACTION_CARRY_VERSION,
    EventJournal,
    EventJournalHistoryAttestation,
    HistoryAttestationDefer,
)
from suzka.runtime.journal_compaction import (  # noqa: E402
    JournalCompactionCoordinator,
    JournalCompactionPolicy,
)
from test_r16_ip9_history_adoption import PAGES  # noqa: E402
from test_r16_ip9_history_reader import SCOPE, _World  # noqa: E402
from test_r16_ip9_subject_history import _committed, _event  # noqa: E402

from tests.test_event_journal import NOW  # noqa: E402

OTHER_SCOPE = "goal-lineage-2"
LATER = NOW + timedelta(days=2)


def _small(world: _World) -> None:
    world.journal.append_v4_migration_checkpoint()
    world.journal.max_bytes = 4_000


def _commit_scope(world: _World, name: str, *, intent: str, scope: str, parent=None):
    """Like ``_World.commit`` but in another lineage scope."""

    world.sequence += 1
    event = _event(name, world.sequence)
    predecessor = (
        (GENESIS, None) if parent is None else (parent.decision_id, parent.content_digest)
    )
    decision = replace(
        _committed(
            event, incarnation=world.incarnation, intent=intent, predecessor=predecessor
        ),
        lineage_scope=scope,
    )
    from test_r16_ip9_subject_history import _prepared, _snapshot

    coordinator = _prepared(world.journal, event, world.participant(decision))
    evidence = world.recovery.commit_internal_candidate(
        event, world.state.load(), _snapshot(world.sequence)
    )
    coordinator.finalize_event(event, evidence)
    world.recovery.complete_committed_event(event, evidence)
    return decision


def _chain(world: _World, count: int, *, prefix: str = "d", parent=None):
    decisions = []
    for index in range(count):
        _event_, parent = world.commit(
            f"{prefix}-{index}",
            intent=sha256(f"{prefix}-{index}".encode()).hexdigest(),
            parent=parent,
        )
        decisions.append(parent)
    return decisions


def _provider(world: _World) -> SubjectHistoryAttestationProvider:
    return SubjectHistoryAttestationProvider(world.journal, world.store)


def _compact(world: _World, provider=None):  # type: ignore[no-untyped-def]
    return world.journal.compact(
        retention_days=1, now=LATER, history_attestation=provider
    )


def _rotated(world: _World) -> int:
    path = world.journal.path
    return len(list(path.parent.glob(f"{path.name}.[0-9]*")))


def test_compaction_without_a_provider_is_refused(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    _chain(world, 6)
    assert _rotated(world) >= 1
    before = world.journal.inspect().records

    outcome = _compact(world)
    assert (outcome.performed, outcome.reason) == (False, "history_attestation_required")
    assert world.journal.inspect().records == before
    # The coordinator without the provider is refused the same way (no fail-open).
    coordinator = JournalCompactionCoordinator(world.journal, JournalCompactionPolicy(1))
    world.journal._clock = lambda: LATER  # type: ignore[method-assign]
    assert coordinator.maybe_compact().reason == "history_attestation_required"


def test_attested_compaction_keeps_the_head_adopted(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    decisions = _chain(world, 6)
    assert world.reader().head(SCOPE).depth == 6

    outcome = _compact(world, _provider(world))
    assert outcome.performed, outcome.reason
    inspection = world.journal.inspect()
    compaction = inspection.latest_compaction
    assert compaction is not None
    assert compaction.carry_version == COMPACTION_CARRY_VERSION
    attestation = compaction.history_attestation
    assert attestation is not None
    assert attestation.incarnation_id == world.incarnation
    assert attestation.previous_attestation_hash is None
    assert not attestation.poisoned_all and attestation.poisoned_scopes == ()
    assert attestation.operations_count >= 1 and attestation.scope_count == 1
    assert compaction.baseline_history_records  # lineage baselines carried

    head = world.reader().head(SCOPE)
    assert (head.status, head.decision, head.depth) == (
        HeadStatus.ADOPTED,
        decisions[-1],
        6,
    )
    # A child of the (partly archived) chain is adopted normally.
    _event_, child = world.commit("after", intent="f" * 64, parent=decisions[-1])
    head = world.reader().head(SCOPE)
    assert (head.decision, head.depth) == (child, 7)

    # A fresh Journal over the same files verifies the C10 declaration.
    world.journal.close()
    reopened = EventJournal(
        world.journal.path, world.journal.max_bytes, 4, clock=lambda: LATER
    )
    assert reopened.inspect().latest_compaction == compaction
    reopened.close()


def test_repeated_compactions_chain_their_attestations(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    decisions = _chain(world, 6)
    assert _compact(world, _provider(world)).performed
    first = world.journal.inspect().latest_compaction
    assert first is not None and first.history_attestation is not None
    decisions += _chain(world, 8, prefix="e", parent=decisions[-1])
    outcome = world.journal.compact(
        retention_days=1,
        now=LATER + timedelta(days=3),
        history_attestation=_provider(world),
    )
    assert outcome.performed, outcome.reason
    second = world.journal.inspect().latest_compaction
    assert second is not None and second.history_attestation is not None
    assert (
        second.history_attestation.previous_attestation_hash
        == first.history_attestation.attestation_hash
    )
    assert second.history_attestation.operations_count > (
        first.history_attestation.operations_count
    )
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision, head.depth) == (
        HeadStatus.ADOPTED,
        decisions[-1],
        14,
    )


def test_permanently_blocked_scope_is_attested_as_poisoned(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    orphan_parent = _committed(
        _event("never-committed", 999), incarnation=world.incarnation, intent="a" * 64
    )
    world.commit("orphan", intent="b" * 64, parent=orphan_parent)
    assert world.reader().head(SCOPE).reason is HeadBlockReason.LINEAGE_GAP
    root = _commit_scope(world, "other-root", intent="c" * 64, scope=OTHER_SCOPE)
    _chain(world, 5, prefix="filler", parent=None)  # rotate segments

    outcome = _compact(world, _provider(world))
    assert outcome.performed, outcome.reason
    attestation = world.journal.inspect().latest_compaction.history_attestation
    assert attestation is not None
    assert attestation.poisoned_scopes == (SCOPE,) and not attestation.poisoned_all

    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (HeadStatus.BLOCKED, HeadBlockReason.POISONED)
    other = world.reader().head(OTHER_SCOPE)
    assert (other.status, other.decision) == (HeadStatus.ADOPTED, root)


def test_store_rolled_back_below_the_attestation_diverges(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    (root,) = _chain(world, 1)
    backup = tmp_path / "older.sqlite3"
    shutil.copyfile(world.store.path, backup)
    _chain(world, 5, prefix="later", parent=root)
    assert _compact(world, _provider(world)).performed

    path = world.store.path
    world.store.close()
    shutil.copyfile(backup, path)
    world.store = SubjectHistoryStore.open(path, max_page_count=PAGES)
    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (HeadStatus.BLOCKED, HeadBlockReason.DIVERGED)


def test_unavailable_store_defers_without_writing(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    _chain(world, 6)
    before = world.journal.inspect().records
    provider = _provider(world)
    world.store.close()

    outcome = _compact(world, provider)
    assert (outcome.performed, outcome.reason) == (False, "deferred_history")
    assert world.journal.inspect().records == before
    deferred = provider(_request(world))
    assert isinstance(deferred, HistoryAttestationDefer)


def test_attestation_with_a_wrong_previous_hash_is_refused(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    _chain(world, 6)
    real = _provider(world)

    def forged(request):  # type: ignore[no-untyped-def]
        attested = real(request)
        assert isinstance(attested, EventJournalHistoryAttestation)
        return attested.model_copy(update={"previous_attestation_hash": "0" * 64})

    outcome = _compact(world, forged)
    assert (outcome.performed, outcome.reason) == (False, "history_attestation_required")


def test_tampered_attested_row_diverges(tmp_path: Path) -> None:
    """A row below H whose SQLite content changes no longer matches."""

    import sqlite3

    world = _World(tmp_path)
    _small(world)
    _chain(world, 6)
    assert _compact(world, _provider(world)).performed
    attestation = world.journal.inspect().latest_compaction.history_attestation
    assert attestation is not None
    first = min(
        world.store.operations(world.incarnation), key=lambda row: row.processing_sequence
    )
    path = world.store.path
    world.store.close()
    raw = sqlite3.connect(path)
    triggers = list(
        raw.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
    )
    for name, _sql in triggers:
        raw.execute(f'DROP TRIGGER "{name}"')
    raw.execute(
        "DELETE FROM history_operations WHERE transaction_id = ?", (first.transaction_id,)
    )
    for _name, sql in triggers:
        raw.execute(sql)  # schema verification sees the original triggers
    raw.commit()
    raw.close()
    world.store = SubjectHistoryStore.open(path, max_page_count=PAGES)
    head = world.reader().head(SCOPE)
    assert (head.status, head.reason) == (HeadStatus.BLOCKED, HeadBlockReason.DIVERGED)


def _request(world: _World):  # type: ignore[no-untyped-def]
    from suzka.runtime.event_journal import HistoryAttestationRequest

    return HistoryAttestationRequest(
        world.journal.inspect().processing_high_water, (), None
    )


def test_globally_diverged_incarnation_is_poisoned_whole(tmp_path: Path) -> None:
    """A permanent global block never starves compaction (C10 decision)."""

    world = _World(tmp_path)
    _small(world)
    (root,) = _chain(world, 1)
    backup = tmp_path / "older.sqlite3"
    shutil.copyfile(world.store.path, backup)
    _chain(world, 5, prefix="later", parent=root)
    path = world.store.path
    world.store.close()
    shutil.copyfile(backup, path)
    world.store = SubjectHistoryStore.open(path, max_page_count=PAGES)
    assert world.reader().head(SCOPE).reason is HeadBlockReason.DIVERGED

    outcome = _compact(world, _provider(world))
    assert outcome.performed, outcome.reason
    attestation = world.journal.inspect().latest_compaction.history_attestation
    assert attestation is not None and attestation.poisoned_all
    assert attestation.poisoned_scopes == () and attestation.scope_count == 0
    for scope in (SCOPE, OTHER_SCOPE):
        head = world.reader().head(scope)
        assert (head.status, head.reason) == (HeadStatus.BLOCKED, HeadBlockReason.POISONED)


def test_coordinator_passes_the_provider(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    decisions = _chain(world, 6)
    world.journal._clock = lambda: LATER  # type: ignore[method-assign]
    coordinator = JournalCompactionCoordinator(
        world.journal, JournalCompactionPolicy(1), history_attestation=_provider(world)
    )
    outcome = coordinator.maybe_compact()
    assert outcome.performed, outcome.reason
    assert world.reader().head(SCOPE).decision == decisions[-1]


def test_poisoned_scope_stays_poisoned_after_a_second_compaction(tmp_path: Path) -> None:
    world = _World(tmp_path)
    _small(world)
    orphan_parent = _committed(
        _event("never-committed", 999), incarnation=world.incarnation, intent="a" * 64
    )
    world.commit("orphan", intent="b" * 64, parent=orphan_parent)
    root = _commit_scope(world, "other-root", intent="c" * 64, scope=OTHER_SCOPE)
    assert _compact(world, _provider(world)).reason in (None, "nothing_due")
    _chain(world, 8, prefix="filler")
    outcome = world.journal.compact(
        retention_days=1,
        now=LATER + timedelta(days=3),
        history_attestation=_provider(world),
    )
    assert outcome.performed, outcome.reason
    attestation = world.journal.inspect().latest_compaction.history_attestation
    assert attestation is not None and SCOPE in attestation.poisoned_scopes
    assert world.reader().head(SCOPE).reason is HeadBlockReason.POISONED
    assert world.reader().head(OTHER_SCOPE).decision == root


def test_permanent_adoption_mismatch_is_refused_not_deferred(tmp_path: Path) -> None:
    from suzka.r16 import history_attestation as module
    from suzka.r16.history_adoption import HistoryAdoptionError

    world = _World(tmp_path)
    _small(world)
    _chain(world, 6)

    def mismatch(_inspection, _store):  # type: ignore[no-untyped-def]
        raise HistoryAdoptionError("SubjectHistory adoption evidence is not exact")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "resolve_incarnation", mismatch)
        outcome = _compact(world, _provider(world))
    assert (outcome.performed, outcome.reason, outcome.detail) == (
        False,
        "history_attestation_refused",
        "history_adoption_invalid",
    )


def test_compaction_after_a_recent_true_rollback(tmp_path: Path) -> None:
    """Review P1-1: a horizon above the archived high-water is valid."""

    import json

    from suzka.r16.history_adoption import resolve_incarnation
    from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator
    from suzka.runtime.state_recovery import StateRecoveryCoordinator

    world = _World(tmp_path)
    world.recovery.publish_boot_anchor(world.recovery.prepare_startup())
    _small(world)
    decisions = _chain(world, 6)
    old_incarnation = world.incarnation
    # Everything from here on is recent: only the older segments are due, so
    # the rollback horizon ends above the archived high-water.
    world.journal._clock = lambda: NOW + timedelta(days=1)  # type: ignore[method-assign]
    _chain(world, 2, prefix="recent", parent=decisions[-1])

    wal = world.recovery.wal
    manifest = wal.inspect().active_manifest
    assert manifest is not None
    generation = wal.root / "generations" / f"{manifest.active_generation_id}.jsonl"
    lines = generation.read_bytes().splitlines(keepends=True)
    transition = json.loads(lines[1])
    transition["record_hash"] = "0" * 64
    lines[1] = json.dumps(transition, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    world.state.path.unlink()
    rolled_back = StateRecoveryCoordinator(world.state, world.journal, wal).prepare_startup()
    assert rolled_back.true_rollback_performed
    assert StartupReconciliationCoordinator(
        world.journal, world.recovery, world.memory, history_store=world.store
    ).reconcile_recovery_gate(rolled_back).participants_consistent
    new_incarnation, horizon = resolve_incarnation(world.journal.inspect(), world.store)
    assert new_incarnation != old_incarnation

    outcome = world.journal.compact(
        retention_days=1,
        now=NOW + timedelta(days=1, hours=12),
        history_attestation=_provider(world),
    )
    assert outcome.performed, (outcome.reason, outcome.detail)
    compaction = world.journal.inspect().latest_compaction
    assert compaction is not None and compaction.history_attestation is not None
    attestation = compaction.history_attestation
    assert attestation.incarnation_id == new_incarnation
    assert attestation.horizon == horizon > compaction.archived_processing_high_water
    assert attestation.operations_count == 0
    archives = list(world.journal.archive_dir.glob("*.jsonl.gz"))
    assert len(archives) == 1  # nothing orphaned

    # The new incarnation starts empty and adopts a fresh GENESIS root.
    world.incarnation = new_incarnation
    world.sequence = world.journal.inspect().processing_high_water
    assert world.reader().head(SCOPE).status is HeadStatus.EMPTY
    _event_, root = world.commit("after-rollback", intent="9" * 64)
    head = world.reader().head(SCOPE)
    assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, root, 1)
