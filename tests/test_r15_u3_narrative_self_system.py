"""U3 fail-closed process-local narrative boundary, not a source producer."""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import suzka.r15.narrative_self_system as narrative_module
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
    ClaimMeaning,
    NarrativeClaim,
    NarrativeClaimKind,
    NarrativeEpisode,
    NarrativeSelfState,
    PropositionPolarity,
    SemanticCode,
    SemanticReferentKind,
    current_semantic_producer,
)
from suzka.r15.narrative_self_system import (
    MAX_NARRATIVE_BOUNDARY_RECEIPTS,
    NarrativeAvailability,
    NarrativeBoundaryResult,
    NarrativeCapability,
    NarrativeCapacityExceeded,
    NarrativeEventConflict,
    NarrativeHistoricalAcknowledgement,
    NarrativeSelfSystem,
    narrative_availability,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


def event(sequence: int, name: str | None = None) -> EventRef:
    return EventRef(name or f"narrative-event-{sequence}", sequence, NOW + timedelta(seconds=sequence))


def claimed_experience(index: int, *, lifecycle: SourceLifecycle = SourceLifecycle.ACTIVE,
                       origin: ClaimedOrigin = ClaimedOrigin.SUBJECT_OBSERVATION) -> SourceWitness:
    return SourceWitness(
        SourceKind.EXPERIENCE, f"claimed-experience-{index}", 0,
        "a" * 64, "b" * 64, lifecycle, origin, event(index),
    )


def test_r15_u3_empty_root_and_detached_private_view_have_no_hidden_biography() -> None:
    owner = NarrativeSelfSystem()
    expected = NarrativeSelfState(1, 0, (), (), None, ())
    assert owner.snapshot() == expected
    first = owner.selected_view()
    assert first.revision == 0 and first.episode_ids == first.claim_ids == ()
    assert first.root_digest == expected.canonical_value()["state_digest"]
    with pytest.raises(FrozenInstanceError):
        first.revision = 3  # type: ignore[misc]
    object.__setattr__(first, "root_digest", "f" * 64)
    assert owner.selected_view().root_digest == expected.canonical_value()["state_digest"]
    assert not hasattr(owner, "restore_snapshot")
    assert not any(hasattr(owner, name) for name in ("add_episode", "adopt_claim", "revise", "reactivate", "import_biography"))


def test_r15_u3_separates_declared_source_from_unavailable_semantics_and_admission() -> None:
    owner = NarrativeSelfSystem()
    active = claimed_experience(1)
    meaning = ClaimMeaning(
        SemanticCode.NARRATIVE_REINTERPRETATION, SemanticReferentKind.EPISODE,
        "a" * 64, PropositionPolarity.AFFIRMS,
    )
    declared = narrative_availability(NarrativeCapability.EPISODE_INDEX, active)
    assert declared.source is SourceDisposition.REQUIRES_TRUSTED_ROOT
    assert declared.semantic is declared.admission is SourceDisposition.UNAVAILABLE
    for operation in NarrativeCapability:
        result = narrative_availability(operation, active, meaning)
        assert result.admission is result.semantic is SourceDisposition.UNAVAILABLE
        assert current_semantic_producer(meaning) is SourceDisposition.UNAVAILABLE
    for invalid in (
        claimed_experience(2, lifecycle=SourceLifecycle.SUPERSEDED),
        claimed_experience(2, lifecycle=SourceLifecycle.RETRACTED),
        claimed_experience(2, origin=ClaimedOrigin.OPERATOR_ASSERTION),
        SourceWitness(SourceKind.MODEL_TEXT, "claimed-experience-2", 0, "a" * 64,
                      "b" * 64, SourceLifecycle.ACTIVE, ClaimedOrigin.SUBJECT_OBSERVATION, event(2)),
    ):
        assert narrative_availability(NarrativeCapability.CLAIM_ADMISSION, invalid, meaning).source is SourceDisposition.UNAVAILABLE
    assert narrative_availability(NarrativeCapability.CLAIM_ADMISSION, None, meaning).source is SourceDisposition.UNAVAILABLE
    assert owner.snapshot() == NarrativeSelfState(1, 0, (), (), None, ())
    with pytest.raises((TypeError, ValueError)):
        NarrativeAvailability(NarrativeCapability.CLAIM_ADMISSION, SourceDisposition.REQUIRES_TRUSTED_ROOT,
                              SourceDisposition.UNAVAILABLE, SourceDisposition.REQUIRES_TRUSTED_ROOT)
    with pytest.raises(TypeError):
        narrative_availability("claim_admission")  # type: ignore[arg-type]


def test_r15_u3_no_change_event_acknowledgement_replays_exactly_without_source_claim() -> None:
    owner = NarrativeSelfSystem()
    first = owner.review_boundary(event(1))
    second = owner.review_boundary(event(2))
    repeated = owner.review_boundary(event(1))
    assert isinstance(first, NarrativeBoundaryResult)
    assert isinstance(second, NarrativeBoundaryResult)
    assert isinstance(repeated, NarrativeBoundaryResult)
    assert not first.replayed and repeated.replayed
    assert repeated.receipt.receipt_digest == first.receipt.receipt_digest
    assert second.receipt.previous_receipt_digest == first.receipt.receipt_digest
    assert repeated.receipt.result_revision == repeated.snapshot.revision == repeated.view.revision == 0
    assert repeated.receipt.result_state_digest == repeated.snapshot.canonical_value()["state_digest"] == repeated.view.root_digest
    assert owner.snapshot().episodes == owner.snapshot().claims == owner.snapshot().revision_history == ()
    assert len(owner._receipts) == 2  # process-local request receipts, not Experience evidence
    assert narrative_availability(NarrativeCapability.EPISODE_INDEX).admission is SourceDisposition.UNAVAILABLE


def test_r15_u3_conflicting_events_reentrancy_and_receipt_overflow_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = NarrativeSelfSystem()
    first = owner.review_boundary(event(2))
    for invalid in (
        event(2, "other-id"), EventRef(first.receipt.event.event_id, 3, NOW + timedelta(seconds=3)),
        event(1), EventRef("narrative-event-3", 3, NOW),
    ):
        with pytest.raises(NarrativeEventConflict):
            owner.review_boundary(invalid)
    assert owner.snapshot().revision == 0 and len(owner._receipts) == 1
    original = narrative_module._result_for
    attempted = False

    def reenter(root: NarrativeSelfState, receipt: narrative_module.NarrativeBoundaryReceipt,
                *, replayed: bool) -> NarrativeBoundaryResult | NarrativeHistoricalAcknowledgement:
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(NarrativeEventConflict):
                owner.review_boundary(event(3))
        return original(root, receipt, replayed=replayed)

    monkeypatch.setattr(narrative_module, "_result_for", reenter)
    with pytest.raises(NarrativeEventConflict):
        owner.review_boundary(event(3))
    monkeypatch.setattr(narrative_module, "_result_for", original)
    assert owner.snapshot().revision == 0 and len(owner._receipts) == 1
    for number in range(3, MAX_NARRATIVE_BOUNDARY_RECEIPTS + 2):
        owner.review_boundary(event(number))
    assert len(owner._receipts) == MAX_NARRATIVE_BOUNDARY_RECEIPTS
    with pytest.raises(NarrativeCapacityExceeded):
        owner.review_boundary(event(MAX_NARRATIVE_BOUNDARY_RECEIPTS + 2))
    assert owner.review_boundary(event(2)).replayed
    assert len(owner._receipts) == MAX_NARRATIVE_BOUNDARY_RECEIPTS
    assert owner.snapshot().revision_history == ()


def test_r15_u3_caught_reentrant_exact_replay_still_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = NarrativeSelfSystem()
    first = owner.review_boundary(event(1))
    original = narrative_module._result_for
    attempted = False

    def reenter(root: NarrativeSelfState, receipt: narrative_module.NarrativeBoundaryReceipt,
                *, replayed: bool) -> NarrativeBoundaryResult | NarrativeHistoricalAcknowledgement:
        nonlocal attempted
        if replayed and not attempted:
            attempted = True
            with pytest.raises(NarrativeEventConflict):
                owner.review_boundary(event(1))
        return original(root, receipt, replayed=replayed)

    monkeypatch.setattr(narrative_module, "_result_for", reenter)
    with pytest.raises(NarrativeEventConflict, match="reentrant replay"):
        owner.review_boundary(event(1))
    monkeypatch.setattr(narrative_module, "_result_for", original)
    assert owner.snapshot() == first.snapshot
    assert len(owner._receipts) == 1
    assert owner.review_boundary(event(1)).replayed


def test_r15_u3_caller_event_and_returned_receipt_are_detached_from_owner() -> None:
    owner = NarrativeSelfSystem()
    supplied = event(1)
    first = owner.review_boundary(supplied)
    original_digest = first.receipt.receipt_digest
    object.__setattr__(supplied, "event_id", "modified-later")
    object.__setattr__(supplied, "event_sequence", 80)
    object.__setattr__(first.receipt, "result_revision", 99)
    replay = owner.review_boundary(event(1))
    assert isinstance(replay, NarrativeBoundaryResult)
    assert replay.receipt.receipt_digest == original_digest
    assert owner._receipts[0].event == event(1)
    assert len(owner._receipts) == 1 and owner.snapshot().revision == 0


def test_r15_u3_concurrent_exact_requests_are_serialized_as_no_change() -> None:
    owner = NarrativeSelfSystem()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner.review_boundary, event(1)) for _ in range(2)]
        outcomes = [future.result() for future in futures]
    assert sorted(result.replayed for result in outcomes) == [False, True]
    assert len(owner._receipts) == 1 and owner.snapshot().revision_history == ()


def _synthetic_root() -> NarrativeSelfState:
    """TEST ONLY. An ACTIVE-looking R12 tuple is NOT a finalized source."""

    source = claimed_experience(1)
    episode = NarrativeEpisode(
        "a" * 64, source,
        Interpretation("c" * 64, InterpretationStatus.UNKNOWN, None, None, (), (),
                       Accessibility.CURRENT, Inertia.TENTATIVE), (),
    )
    proof = RevisionProof(
        1, event(2), episode.episode_id, "d" * 64, "e" * 64,
        (), (), RevisionReason.INITIAL_INTERPRETATION, None,
    )
    return NarrativeSelfState(1, 1, (episode,), (), None, (proof,))


def test_r15_u3_test_only_old_result_never_pairs_historic_receipt_with_new_root() -> None:
    owner = NarrativeSelfSystem()
    accepted = owner.review_boundary(event(1))
    future = _synthetic_root()  # no method exists to install this root in U3
    assert isinstance(accepted, NarrativeBoundaryResult)
    with pytest.raises(ValueError):
        NarrativeBoundaryResult(future, narrative_module._view(future), accepted.receipt, True)
    historic = narrative_module._result_for(future, accepted.receipt, replayed=True)
    assert isinstance(historic, NarrativeHistoricalAcknowledgement)
    assert historic.event_result_digest == accepted.snapshot.canonical_value()["state_digest"]
    assert not hasattr(historic, "snapshot") and not hasattr(historic, "view")
    assert owner.snapshot() == accepted.snapshot
    assert len(owner._receipts) == 1 and owner.snapshot().revision_history == ()
    with pytest.raises(ValueError):
        narrative_module._result_for(future, accepted.receipt, replayed=False)


def test_r15_u3_test_only_conflict_and_accessibility_representations_do_not_admit_sources() -> None:
    """Pure U1 values only; NEVER a trusted Episode/Claim producer fixture."""

    first = claimed_experience(1)
    opposite = claimed_experience(2)
    episodes = tuple(
        NarrativeEpisode(
            f"{position + 1:064x}", source,
            Interpretation(f"{position + 11:064x}", InterpretationStatus.CONTESTED,
                           SCORE_SCALE, SCORE_SCALE, (source,), (opposite if position == 0 else first,),
                           Accessibility.LATENT if position == 0 else Accessibility.REACTIVATABLE,
                           Inertia.REVISABLE), (),
        )
        for position, source in enumerate((first, opposite))
    )
    ids = tuple(item.episode_id for item in episodes)
    claims: list[NarrativeClaim] = []
    for polarity in PropositionPolarity:
        meaning = ClaimMeaning(SemanticCode.NARRATIVE_CONTINUITY,
                               SemanticReferentKind.EPISODE, ids[0], polarity)
        claims.append(NarrativeClaim(
            NarrativeClaimKind.CONTINUITY,
            Interpretation(meaning.statement_id(domain="narrative", kind="continuity", episode_ids=ids),
                           InterpretationStatus.CONTRADICTORY, None, None,
                           (first,), (opposite,), Accessibility.LATENT, Inertia.REVISABLE),
            ids, meaning,
        ))
    snapshot = NarrativeSelfState(1, 0, episodes,
                                  tuple(sorted(claims, key=lambda item: item.interpretation.statement_id)), None, ())
    parsed = NarrativeSelfState.from_json(snapshot.canonical_bytes())
    assert len(parsed.claims) == 2
    assert {claim.meaning.polarity for claim in parsed.claims} == set(PropositionPolarity)
    assert parsed.episodes[0].interpretation.accessibility is Accessibility.LATENT
    assert parsed.episodes[1].interpretation.accessibility is Accessibility.REACTIVATABLE
    assert parsed.episodes[0].experience == first  # accessibility did not delete R12 source
    assert all(current_semantic_producer(claim.meaning) is SourceDisposition.UNAVAILABLE for claim in parsed.claims)
    assert NarrativeSelfSystem().snapshot() == NarrativeSelfState(1, 0, (), (), None, ())
    with pytest.raises(ValueError):
        replace(episodes[0], experience=replace(first, lifecycle=SourceLifecycle.RETRACTED))
    with pytest.raises(ValueError):
        replace(claims[0], meaning=replace(
            claims[0].meaning,
            polarity=(PropositionPolarity.AFFIRMS if claims[0].meaning.polarity is PropositionPolarity.DENIES
                      else PropositionPolarity.DENIES),
        ))


def test_r15_u3_scope_excludes_effectful_sources_and_later_owner_writes() -> None:
    module = ROOT / "suzka" / "r15" / "narrative_self_system.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith((
        "suzka.runtime", "suzka.memory", "suzka.api", "suzka.models",
        "suzka.cognition", "suzka.motivation", "suzka.attention",
        "suzka.metacognition", "suzka.identity", "suzka.r15.relationship_system",
    )) for name in imports)
    assert not any(isinstance(node, ast.Name) and node.id in {
        "uuid4", "datetime", "AgentStateSnapshotV10", "GoalSystem", "Decision",
    } for node in ast.walk(tree))
    assert not any(hasattr(NarrativeSelfSystem(), name) for name in (
        "adopt_claim", "create_episode", "restore_snapshot", "set_importance", "change_accessibility"
    ))
    assert "suzka.r15.narrative_self_system" not in (ROOT / "suzka" / "runtime" / "main_loop.py").read_text(encoding="utf-8")
    budget = derive_r15_schema_budget()
    assert budget.v10_with_reserve_bytes == 133_726_597
    assert budget.remaining_beyond_reserve_bytes == 491_131


def test_r15_u3_documented_design_hold_and_exact_reserve_match_u1() -> None:
    text = (ROOT / "docs" / "r15-u3-narrative-self-system.md").read_text(encoding="utf-8")
    budget = derive_r15_schema_budget()
    for phrase in (
        "**Owns:**", "**May:**", "**Must not:**", "**Depends on:**", "**Used by:**",
        "positive-admission DESIGN HOLD", "REQUIRES_TRUSTED_ROOT", "SEMANTIC_ADMISSION_UNAVAILABLE",
        "NarrativeHistoricalAcknowledgement", "test-only synthetic", "no source/claim writer",
        "R07", "R09", "R12", "U5", "U6", "D9 MUST NOT",
        f"**{budget.v10_with_reserve_bytes:,} B**",
        f"**{budget.remaining_beyond_reserve_bytes:,} B**",
    ):
        assert phrase.lower() in text.lower(), phrase
