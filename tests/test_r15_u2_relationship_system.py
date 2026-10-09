"""Process-local UNKNOWN Relationship owner, not a subject-admission producer."""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import suzka.r15.relationship_system as relationship_module
from suzka.emotion_contracts import EmotionState
from suzka.r15.bounds import derive_r15_schema_budget
from suzka.r15.common import (
    ClaimedOrigin,
    Concept,
    EventRef,
    InterpretationStatus,
    SourceDisposition,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
    source_disposition,
)
from suzka.r15.contracts import RelationshipAxisKind, RelationshipState
from suzka.r15.relationship_system import (
    MAX_OBSERVATION_RECEIPTS,
    RelationshipAuthority,
    RelationshipCapacityExceeded,
    RelationshipEventConflict,
    RelationshipSourceUnavailable,
    RelationshipSystem,
    relationship_authority_status,
)
from suzka.runtime.context import ContextRegistry, ContextStatus, ContextType


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def registry(*participants: str) -> ContextRegistry:
    result = ContextRegistry(clock=lambda: NOW)
    result.create("context-a", ContextType.CONVERSATION, "chat", participant_refs=participants)
    result.set_current("context-a")
    return result


def event(sequence: int, *, identifier: str | None = None) -> EventRef:
    return EventRef(identifier or f"event-{sequence}", sequence, NOW + timedelta(seconds=sequence))


def assert_unknown(root: RelationshipState) -> None:
    for record in root.records:
        assert record.claimed_identity_key is None
        assert len(record.axes) == len(RelationshipAxisKind) == 6
        assert tuple(axis.kind for axis in record.axes) == tuple(sorted(RelationshipAxisKind, key=lambda item: item.value))
        assert len({axis.interpretation.statement_id for axis in record.axes}) == 6
        for axis in record.axes:
            interpretation = axis.interpretation
            assert interpretation.status is InterpretationStatus.UNKNOWN
            assert interpretation.estimate is interpretation.confidence is None
            assert interpretation.support == interpretation.counter == ()


def test_r15_u2_exact_r09_current_observation_creates_only_six_unknown_axes() -> None:
    context = registry("ref-a", "ref-b")
    before_context = context.state
    owner = RelationshipSystem(context)
    empty = owner.snapshot()
    assert empty == RelationshipState(1, 0, (), None, ())
    assert owner.selected_view("ref-a") is None

    first = owner.observe_current_interlocutor(event(1), "ref-a")

    assert not first.replayed and first.receipt.created
    assert first.snapshot.revision == 1 and len(first.snapshot.records) == 1
    assert_unknown(first.snapshot)
    assert context.state == before_context
    assert first.view.root_revision == 1
    assert {axis.kind for axis in first.view.axes} == set(RelationshipAxisKind)
    assert all(axis.status is InterpretationStatus.UNKNOWN and axis.confidence is None for axis in first.view.axes)
    assert first.receipt.context_id == "context-a"
    assert first.receipt.context_revision == before_context.revision
    assert not hasattr(first.receipt, "registry_digest")
    assert owner._last_registry_digest == relationship_module._r09_authority_digest(before_context)
    assert first.receipt.result_state_digest == first.view.root_digest
    proof = first.snapshot.revision_history[0]
    assert proof.revision == 1 and proof.event == event(1)
    assert proof.target_id == first.snapshot.records[0].relationship_id
    assert proof.before_digest == relationship_module._projection_digest(empty.revision, empty.records)
    assert proof.after_digest == relationship_module._projection_digest(first.snapshot.revision, first.snapshot.records)
    assert proof.after_digest != first.receipt.result_state_digest  # proof/root circularity is forbidden
    assert proof.support_digests == proof.counter_digests == ()


def test_r15_u2_equal_r09_claimed_identity_keys_never_merge_distinct_refs() -> None:
    context = registry("ref-a")
    context.upsert_interlocutor_binding("ref-a", "same-person", 1.0, ("asserted",))
    owner = RelationshipSystem(context)
    left = owner.observe_current_interlocutor(event(1), "ref-a")
    context.create("context-b", ContextType.CONVERSATION, "chat", participant_refs=("ref-b",))
    context.upsert_interlocutor_binding("ref-b", "same-person", 1.0, ("asserted",))
    context.set_current("context-b")
    right = owner.observe_current_interlocutor(event(2), "ref-b")
    assert len(right.snapshot.records) == 2
    assert {record.interlocutor_key for record in right.snapshot.records} == {"ref-a", "ref-b"}
    assert len({record.relationship_id for record in right.snapshot.records}) == 2
    assert all(record.claimed_identity_key is None for record in right.snapshot.records)
    assert owner.selected_view("ref-a") is not None
    assert owner.selected_view("ref-b") is not None
    assert right.snapshot.records[0].relationship_id != left.snapshot.records[0].relationship_id or right.snapshot.records[1].relationship_id != left.snapshot.records[0].relationship_id
    assert_unknown(right.snapshot)


def test_r15_u2_duplicate_exact_retry_and_historical_retry_do_not_inflate_root() -> None:
    owner = RelationshipSystem(registry("ref-a", "ref-b"))
    first = owner.observe_current_interlocutor(event(1), "ref-a")
    second = owner.observe_current_interlocutor(event(2), "ref-b")
    replay = owner.observe_current_interlocutor(event(1), "ref-a")
    current_replay = owner.observe_current_interlocutor(event(2), "ref-b")
    assert replay.replayed and current_replay.replayed
    assert replay.receipt.receipt_digest == first.receipt.receipt_digest
    assert current_replay.receipt.receipt_digest == second.receipt.receipt_digest
    assert replay.snapshot == second.snapshot == current_replay.snapshot == owner.snapshot()
    assert replay.receipt.result_revision == 1 and replay.snapshot.revision == 2
    assert len(owner.snapshot().revision_history) == 2
    third = owner.observe_current_interlocutor(event(3), "ref-a")
    assert not third.receipt.created and third.snapshot.revision == 2
    assert third.receipt.previous_receipt_digest == second.receipt.receipt_digest
    assert third.snapshot == owner.snapshot()
    assert_unknown(third.snapshot)


def test_r15_u2_conflicting_replay_stale_event_and_time_reject_without_mutation() -> None:
    owner = RelationshipSystem(registry("ref-a", "ref-b"))
    first = owner.observe_current_interlocutor(event(2), "ref-a")
    failures = (
        (event(2), "ref-b"),
        (event(2, identifier="spoofed-event"), "ref-a"),
        (EventRef("event-2", 3, NOW + timedelta(seconds=3)), "ref-a"),
        (event(1), "ref-a"),
        (EventRef("event-3", 3, NOW), "ref-a"),
    )
    for invalid_event, ref in failures:
        with pytest.raises(RelationshipEventConflict):
            owner.observe_current_interlocutor(invalid_event, ref)
        assert owner.snapshot() == first.snapshot
    accepted = owner.observe_current_interlocutor(event(3), "ref-b")
    assert accepted.snapshot.revision == 2


def test_r15_u2_unavailable_r09_selection_membership_lifecycle_and_future_time_fail_closed() -> None:
    context = ContextRegistry(clock=lambda: NOW)
    owner = RelationshipSystem(context)
    with pytest.raises(RelationshipSourceUnavailable):
        owner.observe_current_interlocutor(event(1), "ref-a")
    context.create("context-a", ContextType.CONVERSATION, "chat", participant_refs=("ref-a",))
    context.set_current("context-a")
    with pytest.raises(RelationshipSourceUnavailable):
        owner.observe_current_interlocutor(event(1), "not-a-participant")
    with pytest.raises(RelationshipSourceUnavailable):
        owner.observe_current_interlocutor(EventRef("too-early", 1, NOW - timedelta(seconds=1)), "ref-a")
    context.suspend("context-a")
    assert context.get("context-a").status is ContextStatus.SUSPENDED
    with pytest.raises(RelationshipSourceUnavailable):
        owner.observe_current_interlocutor(event(1), "ref-a")
    assert owner.snapshot().revision == 0
    context.resume("context-a")
    context.set_current("context-a")
    assert owner.observe_current_interlocutor(event(1), "ref-a").snapshot.revision == 1


def test_r15_u2_claimed_experience_and_all_positive_authorities_remain_unavailable() -> None:
    owner = RelationshipSystem(registry("ref-a"))
    for lifecycle in (SourceLifecycle.ACTIVE, SourceLifecycle.RETRACTED, SourceLifecycle.SUPERSEDED):
        forged = SourceWitness(
            SourceKind.EXPERIENCE, "experience-claimed", 0, "a" * 64, "b" * 64,
            lifecycle, ClaimedOrigin.SUBJECT_OBSERVATION, event(1),
        )
        expected = (SourceDisposition.REQUIRES_TRUSTED_ROOT if lifecycle is SourceLifecycle.ACTIVE
                    else SourceDisposition.UNAVAILABLE)
        assert source_disposition(Concept.RELATIONSHIP, forged) is expected
    assert all(
        relationship_authority_status(authority) is SourceDisposition.UNAVAILABLE
        for authority in RelationshipAuthority
    )
    assert not any(hasattr(owner, name) for name in (
        "revise", "set_axis", "set_trust", "adopt", "merge_persons", "ingest_experience", "restore_snapshot"
    ))
    assert owner.observe_current_interlocutor(event(1), "ref-a").snapshot.records[0].axes[0].interpretation.confidence is None
    assert EmotionState(valence=-1.0, arousal=1.0, optimal_loss=-0.1).valence < 0.0
    assert all(axis.interpretation.status is InterpretationStatus.UNKNOWN for axis in owner.snapshot().records[0].axes)
    with pytest.raises(TypeError):
        relationship_authority_status("positive_axis_interpretation")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RelationshipSystem(object())  # type: ignore[arg-type]


def test_r15_u2_detached_reads_cannot_overwrite_six_axes_or_expose_raw_content() -> None:
    owner = RelationshipSystem(registry("ref-a"))
    owner.observe_current_interlocutor(event(1), "ref-a")
    detached = owner.snapshot()
    view = owner.selected_view("ref-a")
    assert view is not None
    with pytest.raises(FrozenInstanceError):
        view.root_revision = 99  # type: ignore[misc]
    object.__setattr__(detached.records[0].axes[0].interpretation, "status", InterpretationStatus.PRESENT)
    object.__setattr__(view.axes[0], "confidence", 1_000_000)
    assert_unknown(owner.snapshot())
    assert owner.selected_view("ref-a").axes[0].confidence is None  # type: ignore[union-attr]
    assert all(token not in str(view).lower() for token in ("transcript", "private", "raw_prompt", "identity_key"))


def test_r15_u2_all_64_legal_unknown_relationships_and_proofs_fit_without_compaction() -> None:
    context = registry(*(f"ref-{index:02d}" for index in range(32)))
    owner = RelationshipSystem(context)
    for index in range(32):
        owner.observe_current_interlocutor(event(index + 1), f"ref-{index:02d}")
    context.create("context-b", ContextType.CONVERSATION, "chat", participant_refs=tuple(f"ref-{index:02d}" for index in range(32, 64)))
    context.set_current("context-b")
    for index in range(32, 64):
        owner.observe_current_interlocutor(event(index + 1), f"ref-{index:02d}")
    full = owner.snapshot()
    assert len(full.records) == len(full.revision_history) == 64
    assert full.history_anchor is None
    assert_unknown(full)
    before_context = context.state
    context.create("context-c", ContextType.CONVERSATION, "chat", participant_refs=("new-ref",))
    context.set_current("context-c")
    with pytest.raises(RelationshipCapacityExceeded):
        owner.observe_current_interlocutor(event(65), "new-ref")
    assert owner.snapshot() == full
    assert context.state.revision > before_context.revision  # R09 changed independently, never by U2


def test_r15_u2_receipt_bound_does_not_silently_evict_old_replay_proof() -> None:
    owner = RelationshipSystem(registry("ref-a"))
    first = owner.observe_current_interlocutor(event(1), "ref-a")
    for number in range(2, MAX_OBSERVATION_RECEIPTS + 1):
        current = owner.observe_current_interlocutor(event(number), "ref-a")
        assert not current.receipt.created
    before = owner.snapshot()
    with pytest.raises(RelationshipCapacityExceeded):
        owner.observe_current_interlocutor(event(MAX_OBSERVATION_RECEIPTS + 1), "ref-a")
    replay = owner.observe_current_interlocutor(event(1), "ref-a")
    assert replay.replayed and replay.receipt.receipt_digest == first.receipt.receipt_digest
    assert owner.snapshot() == before
    assert len(before.revision_history) == 1


def test_r15_u2_forced_reentrancy_and_r09_change_between_reads_abort_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    context = registry("ref-a")
    owner = RelationshipSystem(context)
    original_getter = ContextRegistry.state.fget
    assert original_getter is not None
    attempts = 0

    def reentrant_state(registry_instance: ContextRegistry) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            with pytest.raises(RelationshipEventConflict):
                owner.observe_current_interlocutor(event(1), "ref-a")
        return original_getter(registry_instance)

    monkeypatch.setattr(ContextRegistry, "state", property(reentrant_state))
    with pytest.raises(RelationshipEventConflict):
        owner.observe_current_interlocutor(event(1), "ref-a")
    monkeypatch.setattr(ContextRegistry, "state", property(original_getter))
    assert owner.snapshot().revision == 0
    assert owner.observe_current_interlocutor(event(1), "ref-a").snapshot.revision == 1

    second = RelationshipSystem(context)
    reads = 0

    def changing_state(registry_instance: ContextRegistry) -> object:
        nonlocal reads
        reads += 1
        result = original_getter(registry_instance)
        if reads == 1:
            registry_instance.upsert_interlocutor_binding("ref-a", "claimed", 1.0, ("operator-claim",))
        return result

    monkeypatch.setattr(ContextRegistry, "state", property(changing_state))
    with pytest.raises(RelationshipSourceUnavailable):
        second.observe_current_interlocutor(event(2), "ref-a")
    monkeypatch.setattr(ContextRegistry, "state", property(original_getter))
    assert second.snapshot().revision == 0
    assert second.observe_current_interlocutor(event(2), "ref-a").snapshot.revision == 1


def test_r15_u2_r09_restore_regression_or_same_revision_fork_rejects_without_root_change() -> None:
    context = registry("ref-a")
    old_r09 = context.state
    context.add_participant_ref("context-a", "ref-b")
    owner = RelationshipSystem(context)
    first = owner.observe_current_interlocutor(event(1), "ref-b")

    context.restore_exact(old_r09)
    with pytest.raises(RelationshipSourceUnavailable, match="regressed"):
        owner.observe_current_interlocutor(event(2), "ref-a")
    assert owner.snapshot() == first.snapshot

    context.add_participant_ref("context-a", "ref-c")
    assert context.state.revision == first.receipt.context_revision
    with pytest.raises(RelationshipSourceUnavailable, match="diverged"):
        owner.observe_current_interlocutor(event(2), "ref-c")
    assert owner.snapshot() == first.snapshot
    assert_unknown(owner.snapshot())


def test_r15_u2_competing_exact_events_are_serialized_without_confidence_inflation() -> None:
    owner = RelationshipSystem(registry("ref-a"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner.observe_current_interlocutor, event(1), "ref-a") for _ in range(2)]
        results = [future.result() for future in futures]
    assert sorted(result.replayed for result in results) == [False, True]
    assert owner.snapshot().revision == 1
    assert_unknown(owner.snapshot())


def test_r15_u2_module_is_not_a_runtime_or_source_truth_consumer() -> None:
    module = ROOT / "suzka" / "r15" / "relationship_system.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert set(imports).intersection({"suzka.runtime.context"}) == {"suzka.runtime.context"}
    assert not any(name.startswith((
        "suzka.runtime.main_loop", "suzka.runtime.agent_state", "suzka.memory",
        "suzka.cognition", "suzka.api", "suzka.models", "suzka.motivation",
        "suzka.attention", "suzka.metacognition", "suzka.identity", "suzka.emotion_contracts",
    )) for name in imports)
    assert not any(isinstance(node, ast.Name) and node.id in {
        "uuid4", "datetime", "AgentStateSnapshotV10", "GoalSystem", "Decision"
    } for node in ast.walk(tree))
    assert "suzka.r15" not in (ROOT / "suzka" / "runtime" / "main_loop.py").read_text(encoding="utf-8")


def test_r15_u2_documented_unavailability_and_full_reserve_match_pure_u1() -> None:
    text = (ROOT / "docs" / "r15-u2-relationship-system.md").read_text(encoding="utf-8")
    budget = derive_r15_schema_budget()
    for phrase in (
        "**Owns:**", "**May:**", "**Must not:**", "**Depends on:**", "**Used by:**",
        "UNAVAILABLE", "R07", "R09", "R10", "R12", "U5", "U6",
        "not a production source", "no present production consumer", "history compaction",
        f"**{budget.v10_with_reserve_bytes:,} B**",
        f"**{budget.remaining_beyond_reserve_bytes:,} B**",
        f"**{MAX_OBSERVATION_RECEIPTS}**",
    ):
        assert phrase.lower() in text.lower(), phrase


def test_r15_u2_r09_read_error_is_private_free_and_preserves_prior_view(monkeypatch: pytest.MonkeyPatch) -> None:
    context = registry("ref-a")
    owner = RelationshipSystem(context)
    def failing_state(_: ContextRegistry) -> object:
        raise RuntimeError("PRIVATE-R09-SECRET")
    monkeypatch.setattr(ContextRegistry, "state", property(failing_state))
    with pytest.raises(RelationshipSourceUnavailable) as captured:
        owner.observe_current_interlocutor(event(1), "ref-a")
    assert "PRIVATE-R09-SECRET" not in str(captured.value)
    assert owner.snapshot().revision == 0
