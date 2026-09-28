"""U6 whole-R12 integration evidence and cross-boundary regressions.

This suite indexes the responsibility-oriented tests that already own each
authority and adds only small read-only composition checks.  It must not become
a second Experience, Semantic, Belief, Value, or recovery authority.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from suzka.body import EmotionState
from suzka.belief import (
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefMutationEvidence,
    BeliefProposition,
    BeliefSystem,
    build_conflict_candidate,
)
from suzka.identity import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
    ValueMutationReason,
    ValueSelfAdmission,
)
from suzka.persona import PromptBuilder
from suzka.runtime import (
    WorkingMemoryDecisionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
)
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType


NOW = datetime(2026, 1, 3, tzinfo=UTC)


R12_F_MATRIX: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "F1",
        (
            "tests/test_belief_contract.py::test_external_evidence_does_not_adopt_a_belief",
            "tests/test_r12_whole_integration.py::test_repeated_semantic_retrieval_does_not_mutate_belief",
        ),
    ),
    (
        "F2",
        (
            "tests/test_r12_whole_integration.py::test_repeated_semantic_retrieval_does_not_mutate_belief",
        ),
    ),
    (
        "F3",
        (
            "tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak",
            "tests/test_experience_participant.py::test_first_create_publication_crash_restarts_from_pending_to_exact_one_revision",
        ),
    ),
    (
        "F4",
        (
            "tests/test_experience_contract.py::test_experience_revision_evidence_and_event_binding_are_strict",
        ),
    ),
    (
        "F5",
        (
            "tests/test_fastapi_backend.py::test_direct_runtime_submit_uses_public_chat_live_authority",
            "tests/test_fastapi_backend.py::test_chat_and_emotion_tick_share_fifo_durable_order",
            "tests/test_emotion_timer.py::test_timer_does_not_create_duplicate_producers_or_cancel_accepted_work",
            "tests/test_r12_whole_integration.py::test_no_cross_boundary_producer_or_direct_adoption_api",
        ),
    ),
    (
        "F6",
        (
            "tests/test_semantic_context_projection.py::test_new_format_projection_distinguishes_context_shapes_and_uses_mean",
        ),
    ),
    (
        "F7",
        (
            "tests/test_semantic_context_projection.py::test_new_format_projection_distinguishes_context_shapes_and_uses_mean",
            "tests/test_semantic_context_projection.py::test_projection_preserves_incomplete_source_statuses",
        ),
    ),
    (
        "F8",
        (
            "tests/test_semantic_context_projection.py::test_projection_is_canonical_for_duplicate_order_and_preserves_status",
        ),
    ),
    (
        "F9",
        (
            "tests/test_semantic_context_projection.py::test_projection_preserves_incomplete_source_statuses",
        ),
    ),
    (
        "F10",
        (
            "tests/test_semantic_lifecycle_contract.py::test_revision_digest_is_order_independent_and_immutable",
        ),
    ),
    (
        "F11",
        (
            "tests/test_semantic_context_projection.py::test_legacy_contextual_resolver_never_reads_db1",
        ),
    ),
    (
        "F12",
        (
            "tests/test_semantic_context_projection.py::test_persisted_new_format_repeated_reads_are_pure_and_never_read_db1",
        ),
    ),
    (
        "F13",
        (
            "tests/test_belief_contract.py::test_conflict_candidate_is_a_pure_hint_only",
            "tests/test_r12_whole_integration.py::test_conflict_candidates_do_not_mutate_belief",
        ),
    ),
    (
        "F14",
        (
            "tests/test_belief_system.py::test_successful_terminal_mutations_leave_the_ordinary_active_view",
        ),
    ),
    (
        "F15",
        (
            "tests/test_agent_state.py::test_v6_round_trip_preserves_intrinsic_belief_authority_without_replay",
            "tests/test_state_wal.py::test_v6_wal_reconstructs_nonempty_belief_without_replay",
        ),
    ),
    (
        "F16",
        (
            "tests/test_experience_participant.py::test_revision_prepare_writes_pending_and_restart_rolls_forward",
            "tests/test_semantic_participant.py::test_partial_lifecycle_publication_restarts_from_pending_batch",
            "tests/test_startup_reconciliation.py::test_internal_commit_rolls_forward_without_state_replay",
        ),
    ),
    (
        "F17",
        (
            "tests/test_r12_whole_integration.py::test_experience_reference_cannot_self_endorse_value",
            "tests/test_identity_origin.py::test_external_actors_cannot_construct_active_admissions",
        ),
    ),
    (
        "F18",
        (
            "tests/test_experience_contract.py::test_experience_record_is_immutable_bounded_and_rejects_raw_content_fields",
            "tests/test_agent_state.py::test_canonical_snapshot_contains_no_private_or_independent_store_data",
            "tests/test_main_loop.py::test_debug_trace_exposes_private_thought_only_ephemerally",
        ),
    ),
    (
        "F19",
        (
            "tests/test_belief_system.py::test_total_record_capacity_fails_closed_without_eviction",
            "tests/test_agent_state.py::test_belief_schema_budget_is_derived_from_all_bounded_fields",
            "tests/test_belief_system.py::test_restore_rejects_missing_or_extra_latest_revision_witnesses",
        ),
    ),
    (
        "F20",
        (
            "tests/test_main_loop.py::test_ordinary_and_debug_chat_use_working_memory_without_prompt_mutation",
            "tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak",
            "tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences",
        ),
    ),
)


RECOVERY_CRASH_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "CHAT before internal commit",
        "tests/test_agent_runtime.py::test_handler_failure_skips_checkpoint",
    ),
    (
        "CHAT pending Experience publication",
        "tests/test_experience_participant.py::test_revision_prepare_writes_pending_and_restart_rolls_forward",
    ),
    (
        "CHAT Experience exactly once",
        "tests/test_experience_participant.py::test_first_create_publication_crash_restarts_from_pending_to_exact_one_revision",
    ),
    (
        "DEBUG_CHAT and EMOTION_TICK ordering",
        "tests/test_fastapi_backend.py::test_chat_and_emotion_tick_share_fifo_durable_order",
    ),
    (
        "SLEEP visible semantic persistence",
        "tests/test_r12_semantic_integration.py::test_sleep_persists_visible_semantic_only_and_never_reruns_model",
    ),
    (
        "Semantic lifecycle pending",
        "tests/test_semantic_participant.py::test_partial_lifecycle_publication_restarts_from_pending_batch",
    ),
    (
        "DB2 projection repair",
        "tests/test_r12_semantic_integration.py::test_terminal_startup_reconciles_missing_projection_from_authority",
    ),
    (
        "Divergent DB2",
        "tests/test_semantic_participant.py::test_divergent_legacy_projection_is_not_overwritten",
    ),
    (
        "Receipt without lifecycle authority",
        "tests/test_experience_participant.py::test_receipt_without_committed_record_fails_closed",
    ),
    (
        "AgentState v6 Belief restart",
        "tests/test_state_wal.py::test_v6_wal_reconstructs_nonempty_belief_without_replay",
    ),
    (
        "True rollback non-destructive external history",
        "tests/test_startup_reconciliation.py::test_true_rollback_restores_working_memory_only_and_preserves_newer_episodic",
    ),
    (
        "Malformed Belief state",
        "tests/test_agent_state.py::test_v6_nonempty_belief_restore_requires_intrinsic_authority",
    ),
)


PRIVACY_SENTINEL_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "Experience raw content",
        "tests/test_experience_contract.py::test_experience_record_is_immutable_bounded_and_rejects_raw_content_fields",
    ),
    (
        "DB1 private metadata",
        "tests/test_dual_memory_system.py::test_new_memory_metadata_rejects_private_fields",
    ),
    (
        "AgentState private fields",
        "tests/test_agent_state.py::test_canonical_snapshot_contains_no_private_or_independent_store_data",
    ),
    (
        "Debug/private response",
        "tests/test_main_loop.py::test_visible_response_does_not_contain_think_tags_or_private_sentinel",
    ),
    (
        "Provider exception",
        "tests/test_surprisal_calculator.py::test_provider_exception_becomes_bounded_private_free_evidence",
    ),
    (
        "Visible Semantic content distinction",
        "tests/test_r12_semantic_integration.py::test_sleep_persists_visible_semantic_only_and_never_reruns_model",
    ),
    (
        "StateWAL private payload rejection",
        "tests/test_state_wal.py::test_private_sentinel_and_bounded_errors",
    ),
    (
        "EventJournal private payload rejection",
        "tests/test_event_journal.py::test_u1_f11_invalid_transaction_fields_are_bounded_and_private_free",
    ),
)


COMPATIBILITY_NON_INVASION_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "R03 AgentRuntime",
        "tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences",
    ),
    (
        "R04-R06 state/WAL",
        "tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable",
    ),
    (
        "R07 transaction participants",
        "tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value",
    ),
    (
        "R08 WorkingMemory",
        "tests/test_working_memory.py::test_projection_never_exceeds_utf8_byte_budget",
    ),
    (
        "R09 Context",
        "tests/test_chat_context.py::test_contextual_working_memory_uses_compatibility_without_mutating_authority",
    ),
    (
        "R10 appraisal/emotion",
        "tests/test_appraisal.py::test_appraisal_is_pure_deterministic_and_does_not_consume_private_text",
    ),
    (
        "R11 Value",
        "tests/test_value_system.py::test_prompt_projection_is_immutable_and_excludes_authority_metadata",
    ),
    (
        "Public chat",
        "tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak",
    ),
    (
        "Sleep/training",
        "tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active",
    ),
)


def _memory_selection(source_kind: WorkingMemorySourceKind, content: str) -> WorkingMemorySelection:
    return WorkingMemorySelection(
        item_id=f"u6-{source_kind.value}-item",
        source_kind=source_kind,
        source_id=f"u6-{source_kind.value}-source",
        rendered_content=content,
        score=0.5,
        reason=WorkingMemoryDecisionReason.SELECTED,
    )


def _memory_view(*selections: WorkingMemorySelection) -> WorkingMemoryView:
    return WorkingMemoryView(
        selected=selections,
        decisions=(),
        projected_bytes=0,
        item_capacity=4,
        projection_max_bytes=4096,
        revision=1,
    )


def _belief_with_proposal() -> BeliefSystem:
    active_event: AgentEvent | None = None

    def set_event() -> BeliefMutationEvidence:
        nonlocal active_event
        active_event = AgentEvent(
            event_id="u6-belief-event",
            event_type=AgentEventType.CHAT,
            source=AgentEventSource.API_CHAT,
            requested_at=NOW,
            processing_sequence=1,
        )
        return BeliefMutationEvidence("u6-belief-event", 1, NOW)

    system = BeliefSystem(event_provider=lambda: active_event)
    system.create_proposal(
        BeliefProposition("stored evidence is not adopted belief"),
        set_event(),
        evidence=(
            BeliefEvidence("experience:u6", BeliefEvidenceType.EXPERIENCE),
        ),
    )
    return system


def test_repeated_semantic_retrieval_does_not_mutate_belief() -> None:
    system = _belief_with_proposal()
    before = system.snapshot()
    memory_view = _memory_view(
        _memory_selection(
            WorkingMemorySourceKind.SEMANTIC,
            "The subject lives in a stored semantic record.",
        )
    )

    prompts = tuple(
        PromptBuilder().build("hello", EmotionState(), memory_view)
        for _ in range(3)
    )

    assert prompts[0] == prompts[1] == prompts[2]
    assert "Stored semantic evidence:" in prompts[0]
    assert system.snapshot() == before


def test_repeated_experience_reads_do_not_mutate_belief() -> None:
    system = _belief_with_proposal()
    before = system.snapshot()
    memory_view = _memory_view(
        _memory_selection(
            WorkingMemorySourceKind.EPISODIC,
            "A stored Experience evidence reference was retrieved.",
        )
    )

    for _ in range(3):
        PromptBuilder().build("hello", EmotionState(), memory_view)

    assert system.snapshot() == before


def test_experience_reference_cannot_self_endorse_value() -> None:
    with pytest.raises(ValueError, match="self_endorsed"):
        ValueSelfAdmission(
            target_value_id="value-u6",
            subject_origin=IdentityOrigin(
                actor=OriginActor.EXTERNAL_SOURCE,
                input_kind=OriginInputKind.EVIDENCE,
                admission=ValueAdmissionStatus.SELF_ENDORSED,
                source_ref="experience:u6",
                event_id="u6-event",
                event_sequence=1,
            ),
            evidence_refs=("experience:u6",),
            requested_delta=0.1,
            confidence=1.0,
            reason=ValueMutationReason.ADMITTED_UPDATE,
        )


def test_conflict_candidates_do_not_mutate_belief() -> None:
    system = _belief_with_proposal()
    before = system.snapshot()
    candidate = build_conflict_candidate(
        BeliefProposition("Alice likes tea", "Alice", "likes", "tea"),
        BeliefProposition("Alice likes coffee", "Alice", "likes", "coffee"),
        ("context:u6",),
        ("context:u6",),
    )

    assert candidate is not None
    assert system.snapshot() == before


def test_no_cross_boundary_producer_or_direct_adoption_api() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "suzka/experience",
        "suzka/memory/experience_participant.py",
        "suzka/memory/semantic_participant.py",
    ):
        path = root / relative
        paths = path.rglob("*.py") if path.is_dir() else (path,)
        for source_path in paths:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("suzka.belief")
                elif isinstance(node, ast.Import):
                    assert all(alias.name != "suzka.belief" for alias in node.names)
    assert "Experience" not in (
        root / "suzka" / "runtime" / "emotion_timer.py"
    ).read_text(encoding="utf-8")
    assert not any((root / "suzka" / "api" / "routes").glob("*belief*.py"))


def _assert_test_reference(reference: str, repository_root: Path) -> None:
    test_path, test_name = reference.split("::", 1)
    path = repository_root / test_path
    assert path.is_file(), f"missing evidence file: {test_path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert test_name in names, f"missing evidence test: {reference}"


def test_r12_f_matrix_is_complete_and_references_existing_tests() -> None:
    assert tuple(item[0] for item in R12_F_MATRIX) == tuple(
        f"F{index}" for index in range(1, 21)
    )
    root = Path(__file__).resolve().parents[1]
    for _requirement, references in R12_F_MATRIX:
        assert references
        for reference in references:
            _assert_test_reference(reference, root)


def test_r12_recovery_privacy_and_compatibility_matrices_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    for matrix in (
        RECOVERY_CRASH_MATRIX,
        PRIVACY_SENTINEL_MATRIX,
        COMPATIBILITY_NON_INVASION_MATRIX,
    ):
        assert matrix
        for _boundary, reference in matrix:
            _assert_test_reference(reference, root)
    assert len(RECOVERY_CRASH_MATRIX) >= 12
    assert len(PRIVACY_SENTINEL_MATRIX) >= 5
    assert len(COMPATIBILITY_NON_INVASION_MATRIX) >= 8
