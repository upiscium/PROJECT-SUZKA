"""U4 closed UNKNOWN SelfModel owner; synthetic claims are not subject proof."""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import suzka.r15.self_model_system as self_module
from suzka.r15.bounds import derive_r15_schema_budget
from suzka.r15.common import (
    Accessibility,
    ClaimedOrigin,
    EventRef,
    Inertia,
    Interpretation,
    InterpretationStatus,
    RevisionProof,
    RevisionReason,
    SCORE_SCALE,
    SourceDisposition,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
)
from suzka.r15.contracts import (
    CapabilityHypothesis,
    ClaimMeaning,
    PropositionPolarity,
    SelfClaim,
    SelfClaimKind,
    SelfModelState,
    SemanticCode,
    SemanticReferentKind,
    VerifiedCompetence,
    current_semantic_producer,
)
from suzka.r15.self_model_system import (
    MAX_SELF_BOUNDARY_RECEIPTS,
    SelfAvailability,
    SelfBoundaryResult,
    SelfCapability,
    SelfHistoricalAcknowledgement,
    SelfModelCapacityExceeded,
    SelfModelEventConflict,
    SelfModelSystem,
    SelfTaskView,
    self_availability,
)


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def event(sequence: int, name: str | None = None) -> EventRef:
    return EventRef(name or f"self-event-{sequence}", sequence, NOW + timedelta(seconds=sequence))


def alleged_experience(index: int, *, origin: ClaimedOrigin = ClaimedOrigin.SUBJECT_OBSERVATION,
                       lifecycle: SourceLifecycle = SourceLifecycle.ACTIVE) -> SourceWitness:
    return SourceWitness(
        SourceKind.EXPERIENCE, f"experience-{index}", 0, "a" * 64, "b" * 64,
        lifecycle, origin, event(index),
    )


def test_r15_u4_empty_owner_read_keeps_task_unknown_distinct_from_measured_zero() -> None:
    owner = SelfModelSystem()
    root = SelfModelState(1, 0, (), (), None, ())
    assert owner.snapshot() == root
    view = owner.selected_view()
    assert view.revision == 0 and view.claim_ids == view.task_class_keys == ()
    assert view.root_digest == root.canonical_value()["state_digest"]
    task = owner.task_view("task:python")
    assert task.task_class_key == "task:python" and task.tracked is False
    assert task.attempted_status is InterpretationStatus.UNKNOWN and task.attempted_confidence is None
    assert task.verified_competence is VerifiedCompetence.UNKNOWN
    assert owner.snapshot() == root  # the query cannot create a capability row
    with pytest.raises(FrozenInstanceError):
        task.tracked = True  # type: ignore[misc]
    object.__setattr__(view, "root_digest", "c" * 64)
    assert owner.selected_view().root_digest == root.canonical_value()["state_digest"]
    with pytest.raises(ValueError):
        owner.task_view("task/invalid")
    with pytest.raises(ValueError):
        SelfTaskView("task:python", True, InterpretationStatus.UNKNOWN, None, VerifiedCompetence.UNKNOWN)
    with pytest.raises(ValueError):
        SelfTaskView("task:python", False, InterpretationStatus.PRESENT, 0, VerifiedCompetence.UNKNOWN)


def test_r15_u4_typed_source_vs_semantic_vs_verified_competence_are_all_separate() -> None:
    owner = SelfModelSystem()
    source = alleged_experience(1)
    meaning = ClaimMeaning(
        SemanticCode.SELF_TRAIT_CAUTIOUS, SemanticReferentKind.EXPERIENCE,
        source.reference, PropositionPolarity.AFFIRMS,
    )
    for kind in SelfCapability:
        result = self_availability(kind, source, meaning)
        expected = (SourceDisposition.UNAVAILABLE if kind is SelfCapability.VERIFIED_COMPETENCE
                    else SourceDisposition.REQUIRES_TRUSTED_ROOT)
        assert result.source is expected
        assert result.semantic is result.admission is SourceDisposition.UNAVAILABLE
        assert result.verified is VerifiedCompetence.UNKNOWN
    assert current_semantic_producer(meaning) is SourceDisposition.UNAVAILABLE
    for unqualified in (
        alleged_experience(2, origin=ClaimedOrigin.OPERATOR_ASSERTION),
        alleged_experience(2, lifecycle=SourceLifecycle.RETRACTED),
        alleged_experience(2, lifecycle=SourceLifecycle.SUPERSEDED),
        SourceWitness(SourceKind.MODEL_TEXT, "experience-2", 0, "a" * 64,
                      "b" * 64, SourceLifecycle.ACTIVE, ClaimedOrigin.SUBJECT_OBSERVATION, event(2)),
        SourceWitness(SourceKind.WEB_TEXT, "experience-2", 0, "a" * 64,
                      "b" * 64, SourceLifecycle.ACTIVE, ClaimedOrigin.SUBJECT_OBSERVATION, event(2)),
    ):
        assert self_availability(SelfCapability.POSSIBLE_TRAIT, unqualified, meaning).source is SourceDisposition.UNAVAILABLE
    assert self_availability(SelfCapability.ATTEMPTED_TASK).source is SourceDisposition.UNAVAILABLE
    assert owner.snapshot() == SelfModelState(1, 0, (), (), None, ())
    with pytest.raises(ValueError):
        SelfAvailability(SelfCapability.VERIFIED_COMPETENCE, SourceDisposition.UNAVAILABLE,
                         SourceDisposition.UNAVAILABLE, SourceDisposition.REQUIRES_TRUSTED_ROOT,
                         VerifiedCompetence.UNKNOWN)
    with pytest.raises(ValueError):
        SelfAvailability(SelfCapability.VERIFIED_COMPETENCE, SourceDisposition.REQUIRES_TRUSTED_ROOT,
                         SourceDisposition.UNAVAILABLE, SourceDisposition.UNAVAILABLE,
                         VerifiedCompetence.UNKNOWN)
    with pytest.raises(TypeError):
        self_availability("verified_competence")  # type: ignore[arg-type]


def test_r15_u4_no_change_receipt_is_event_bound_and_exact_replay_is_idempotent() -> None:
    owner = SelfModelSystem()
    first = owner.review_boundary(event(1))
    second = owner.review_boundary(event(2))
    replay = owner.review_boundary(event(1))
    assert isinstance(first, SelfBoundaryResult)
    assert isinstance(second, SelfBoundaryResult) and isinstance(replay, SelfBoundaryResult)
    assert not first.replayed and replay.replayed
    assert replay.receipt.receipt_digest == first.receipt.receipt_digest
    assert second.receipt.previous_receipt_digest == first.receipt.receipt_digest
    assert replay.receipt.result_revision == replay.snapshot.revision == replay.view.revision == 0
    assert replay.receipt.result_state_digest == replay.snapshot.canonical_value()["state_digest"] == replay.view.root_digest
    assert owner.snapshot().claims == owner.snapshot().capabilities == owner.snapshot().revision_history == ()
    assert len(owner._receipts) == 2


def test_r15_u4_conflicting_stale_reentrant_and_capacity_requests_fail_without_root_update(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = SelfModelSystem()
    first = owner.review_boundary(event(2))
    for invalid in (
        event(2, "other-id"), EventRef(first.receipt.event.event_id, 3, NOW + timedelta(seconds=3)),
        event(1), EventRef("self-event-3", 3, NOW),
    ):
        with pytest.raises(SelfModelEventConflict):
            owner.review_boundary(invalid)
    original = self_module._result_for
    attempted = False

    def reenter(root: SelfModelState, receipt: self_module.SelfBoundaryReceipt,
                *, replayed: bool) -> SelfBoundaryResult | SelfHistoricalAcknowledgement:
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(SelfModelEventConflict):
                owner.review_boundary(event(3))
        return original(root, receipt, replayed=replayed)

    monkeypatch.setattr(self_module, "_result_for", reenter)
    with pytest.raises(SelfModelEventConflict):
        owner.review_boundary(event(3))
    monkeypatch.setattr(self_module, "_result_for", original)
    assert len(owner._receipts) == 1 and owner.snapshot().revision == 0
    for sequence in range(3, MAX_SELF_BOUNDARY_RECEIPTS + 2):
        owner.review_boundary(event(sequence))
    assert len(owner._receipts) == MAX_SELF_BOUNDARY_RECEIPTS
    with pytest.raises(SelfModelCapacityExceeded):
        owner.review_boundary(event(MAX_SELF_BOUNDARY_RECEIPTS + 2))
    assert owner.review_boundary(event(2)).replayed
    assert owner.snapshot().revision_history == ()


def test_r15_u4_caught_reentrant_exact_replay_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = SelfModelSystem()
    first = owner.review_boundary(event(1))
    original = self_module._result_for
    attempted = False

    def reenter(root: SelfModelState, receipt: self_module.SelfBoundaryReceipt,
                *, replayed: bool) -> SelfBoundaryResult | SelfHistoricalAcknowledgement:
        nonlocal attempted
        if replayed and not attempted:
            attempted = True
            with pytest.raises(SelfModelEventConflict):
                owner.review_boundary(event(1))
        return original(root, receipt, replayed=replayed)

    monkeypatch.setattr(self_module, "_result_for", reenter)
    with pytest.raises(SelfModelEventConflict, match="reentrant replay"):
        owner.review_boundary(event(1))
    monkeypatch.setattr(self_module, "_result_for", original)
    assert owner.snapshot() == first.snapshot and len(owner._receipts) == 1
    assert owner.review_boundary(event(1)).replayed


def test_r15_u4_event_and_returned_receipt_mutation_cannot_rewrite_internal_proof() -> None:
    owner = SelfModelSystem()
    supplied = event(1)
    result = owner.review_boundary(supplied)
    original_digest = result.receipt.receipt_digest
    object.__setattr__(supplied, "event_id", "forged-later")
    object.__setattr__(supplied, "event_sequence", 99)
    object.__setattr__(result.receipt, "result_revision", 99)
    replay = owner.review_boundary(event(1))
    assert isinstance(replay, SelfBoundaryResult)
    assert replay.receipt.receipt_digest == original_digest
    assert owner._receipts[0].event == event(1)
    assert len(owner._receipts) == 1 and owner.snapshot().revision == 0


def test_r15_u4_concurrent_same_event_has_one_receipt_and_no_hypothesis() -> None:
    owner = SelfModelSystem()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner.review_boundary, event(1)) for _ in range(2)]
        outcomes = [future.result() for future in futures]
    assert sorted(item.replayed for item in outcomes) == [False, True]
    assert len(owner._receipts) == 1 and owner.snapshot().capabilities == ()


def _synthetic_self_root() -> SelfModelState:
    """TEST ONLY: task class is not a verified attempt or outcome."""

    attempt = Interpretation("c" * 64, InterpretationStatus.UNKNOWN, None, None,
                             (), (), Accessibility.CURRENT, Inertia.TENTATIVE)
    capability = CapabilityHypothesis("task:python", attempt, VerifiedCompetence.UNKNOWN)
    proof = RevisionProof(1, event(2), "d" * 64, "e" * 64, "f" * 64,
                          (), (), RevisionReason.INITIAL_INTERPRETATION, None)
    return SelfModelState(1, 1, (), (capability,), None, (proof,))


def test_r15_u4_test_only_historical_ack_never_pairs_old_receipt_with_new_root() -> None:
    owner = SelfModelSystem()
    current = owner.review_boundary(event(1))
    later = _synthetic_self_root()  # U4 has no way to install this as its root
    with pytest.raises(ValueError):
        SelfBoundaryResult(later, self_module._view(later), current.receipt, True)
    historic = self_module._result_for(later, current.receipt, replayed=True)
    assert isinstance(historic, SelfHistoricalAcknowledgement)
    assert historic.event_result_digest == current.snapshot.canonical_value()["state_digest"]
    assert not hasattr(historic, "snapshot") and not hasattr(historic, "view")
    assert owner.snapshot() == current.snapshot and len(owner._receipts) == 1
    with pytest.raises(ValueError):
        self_module._result_for(later, current.receipt, replayed=False)


def test_r15_u4_test_only_hypothesis_conflict_accessibility_and_attempt_not_competence() -> None:
    """Closed U1 shape simulation, never Suzka's admitted self-truth."""

    first, opposed = alleged_experience(1), alleged_experience(2)
    claims: list[SelfClaim] = []
    for polarity in PropositionPolarity:
        meaning = ClaimMeaning(SemanticCode.SELF_TRAIT_CAUTIOUS,
                               SemanticReferentKind.EXPERIENCE, first.reference, polarity)
        claims.append(SelfClaim(
            SelfClaimKind.POSSIBLE_TRAIT,
            Interpretation(meaning.statement_id(domain="self_model", kind=SelfClaimKind.POSSIBLE_TRAIT.value),
                           InterpretationStatus.CONTESTED, SCORE_SCALE, SCORE_SCALE,
                           (first,), (opposed,), Accessibility.LATENT, Inertia.REVISABLE), meaning,
        ))
    attempt = Interpretation("c" * 64, InterpretationStatus.PRESENT, SCORE_SCALE, SCORE_SCALE,
                             (first,), (), Accessibility.REACTIVATABLE, Inertia.TENTATIVE)
    capability = CapabilityHypothesis("task:python", attempt, VerifiedCompetence.UNKNOWN)
    root = SelfModelState(1, 0, tuple(sorted(claims, key=lambda item: item.interpretation.statement_id)),
                          (capability,), None, ())
    restored = SelfModelState.from_json(root.canonical_bytes())
    assert {item.meaning.polarity for item in restored.claims} == set(PropositionPolarity)
    assert all(item.interpretation.counter and item.interpretation.accessibility is Accessibility.LATENT
               for item in restored.claims)
    assert restored.capabilities[0].attempted.confidence == SCORE_SCALE
    assert restored.capabilities[0].verified_competence is VerifiedCompetence.UNKNOWN
    assert all(current_semantic_producer(item.meaning) is SourceDisposition.UNAVAILABLE for item in restored.claims)
    assert SelfModelSystem().snapshot() == SelfModelState(1, 0, (), (), None, ())
    with pytest.raises(ValueError):
        replace(capability, verified_competence="verified")
    with pytest.raises(ValueError):
        replace(claims[0], interpretation=replace(claims[0].interpretation,
                                                   statement_id=claims[1].interpretation.statement_id))
    with pytest.raises(ValueError):
        replace(claims[0], interpretation=replace(claims[0].interpretation,
                                                   support=(replace(first, lifecycle=SourceLifecycle.RETRACTED),)))


def test_r15_u4_scope_does_not_import_sources_later_owners_or_modify_accepted_bounds() -> None:
    module = ROOT / "suzka" / "r15" / "self_model_system.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(item.startswith((
        "suzka.runtime", "suzka.memory", "suzka.api", "suzka.models", "suzka.cognition",
        "suzka.motivation", "suzka.attention", "suzka.metacognition", "suzka.identity",
        "suzka.r15.relationship_system", "suzka.r15.narrative_self_system",
    )) for item in imports)
    assert not any(isinstance(node, ast.Name) and node.id in {
        "uuid4", "datetime", "AgentStateSnapshotV10", "Decision", "GoalSystem",
    } for node in ast.walk(tree))
    assert not any(hasattr(SelfModelSystem(), name) for name in (
        "adopt_trait", "set_identity", "add_attempt", "verify_competence", "restore_snapshot"
    ))
    assert "suzka.r15.self_model_system" not in (ROOT / "suzka" / "runtime" / "main_loop.py").read_text(encoding="utf-8")
    budget = derive_r15_schema_budget()
    assert budget.v10_with_reserve_bytes == 133_726_597
    assert budget.remaining_beyond_reserve_bytes == 491_131


def test_r15_u4_documented_design_hold_and_distinct_unknowns_match_budget() -> None:
    text = (ROOT / "docs" / "r15-u4-self-model-system.md").read_text(encoding="utf-8")
    budget = derive_r15_schema_budget()
    for phrase in (
        "**Owns:**", "**May:**", "**Must not:**", "**Depends on:**", "**Used by:**",
        "#288", "DESIGN REQUIRED", "REQUIRES_TRUSTED_ROOT", "VerifiedCompetence.UNKNOWN",
        "current_semantic_producer", "attempted.statement_id", "task_class_key",
        "SelfHistoricalAcknowledgement", "test-only synthetic", "D9 absolute MUST NOT",
        f"**{budget.v10_with_reserve_bytes:,} B**",
        f"**{budget.remaining_beyond_reserve_bytes:,} B**",
    ):
        assert phrase.lower() in text.lower(), phrase
