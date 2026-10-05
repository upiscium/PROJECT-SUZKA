"""U6 F1–F22 evidence-map consistency and R13 boundary checks."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from suzka.body import EmotionState
from suzka.config import load_settings
from suzka.identity import ValuePromptView
from suzka.memory import DualMemorySystem
from suzka.motivation.commitment_system import CommitmentSystem
from suzka.motivation.goal_system import GoalSystem
from suzka.motivation.projection import (
    R13_PROMPT_MAX_RECORDS_PER_DOMAIN,
    R13_PROMPT_MAX_RENDERED_BYTES,
    R13_PROMPT_MAX_SERIALIZED_BYTES,
)
from suzka.motivation.system import MotivationSystem
from suzka.persona import ContextPromptView
from suzka.runtime import SuzkaMainLoop, WorkingMemoryView
from suzka.models import DummyProvider
from test_r13_codec import _commitment_snapshot, _goal_snapshot, _motivation_snapshot


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DOCUMENT = ROOT / "docs" / "r13-evidence.md"


R13_F_MATRIX: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "F1",
        "DIRECT_R13_PASS",
        "—",
        ("tests/test_goal_system.py::test_repeated_external_request_does_not_turn_proposal_into_adoption",),
    ),
    (
        "F2",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_goal_system.py::test_request_operator_external_and_motivation_origins_never_auto_adopt",
            "tests/test_goal_system.py::test_ingestion_retains_only_proposal_and_exact_replay_is_idempotent",
        ),
    ),
    (
        "F3",
        "REPRESENTATIONAL_NO_PRODUCER",
        "—",
        (
            "tests/test_motivation_system.py::test_experience_factory_uses_bounded_appraisal_and_keeps_event_origin",
            "tests/test_motivation_system.py::test_candidate_alone_has_no_motivation_authority",
            "tests/test_goal_system.py::test_request_operator_external_and_motivation_origins_never_auto_adopt",
        ),
    ),
    (
        "F4",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_motivation_system.py::test_goal_proposal_uses_the_exact_first_threshold_eligibility_event",
            "tests/test_motivation_system.py::test_goal_proposal_is_bounded_deterministic_proposed_and_admission_free",
            "tests/test_motivation_system.py::test_high_salience_evidence_stays_within_per_event_record_and_goal_budgets",
        ),
    ),
    (
        "F5",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_motivation_system.py::test_external_origin_is_retained_and_cannot_be_relabelled_on_replay",
            "tests/test_goal_system.py::test_request_operator_external_and_motivation_origins_never_auto_adopt",
        ),
    ),
    (
        "F6",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_motivation_system.py::test_same_opaque_id_under_different_source_kinds_remains_distinct",
            "tests/test_motivation_system.py::test_snapshot_rejects_evidence_provenance_not_bound_by_its_receipt",
            "tests/test_r13_codec.py::test_motivation_table_keeps_candidate_evidence_eligibility_and_typed_refs",
        ),
    ),
    (
        "F7",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_motivation_system.py::test_conflicting_motivations_coexist_without_winner_selection",
            "tests/test_goal_system.py::test_conflicting_adopted_goals_coexist_without_ranking_or_suspension",
        ),
    ),
    (
        "F8",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_motivation_system.py::test_decay_review_and_satiation_are_deterministic_and_event_bound",
            "tests/test_motivation_system.py::test_motivation_history_compacts_without_losing_evidence_idempotency",
            "tests/test_goal_system.py::test_subject_proofs_and_receipts_survive_bounded_revision_compaction",
        ),
    ),
    (
        "F9",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_goal_system.py::test_ingestion_retains_only_proposal_and_exact_replay_is_idempotent",
            "tests/test_goal_system.py::test_only_exact_caller_supplied_admission_can_adopt",
        ),
    ),
    (
        "F10",
        "DEFERRED",
        "R16",
        (
            "tests/test_motivation_system.py::test_conflicting_motivations_coexist_without_winner_selection",
            "tests/test_goal_system.py::test_conflicting_adopted_goals_coexist_without_ranking_or_suspension",
            "tests/test_goal_system.py::test_graph_conflict_facts_and_snapshot_export_are_exact_and_side_effect_free",
        ),
    ),
    (
        "F11",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_goal_system.py::test_conflicting_adopted_goals_coexist_without_ranking_or_suspension",
            "tests/test_goal_system.py::test_missing_dependency_and_dependency_cycle_fail_closed",
        ),
    ),
    (
        "F12",
        "DIRECT_R13_PASS",
        "—",
        ("tests/test_commitment_system.py::test_proposal_sources_cannot_activate_responsibility",),
    ),
    (
        "F13",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_commitment_system.py::test_accept_consumes_exact_supplied_proof_without_producing_one",
            "tests/test_commitment_system.py::test_wrong_admission_shape_or_event_fails_atomically",
        ),
    ),
    (
        "F14",
        "REPRESENTATIONAL_NO_PRODUCER",
        "—",
        ("tests/test_commitment_system.py::test_desire_goal_and_deadline_are_references_not_automatic_authority",),
    ),
    (
        "F15",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_commitment_system.py::test_fresh_transition_proofs_release_or_renegotiate_without_reopening",
            "tests/test_commitment_system.py::test_transition_rejects_wrong_authority_terms_operation_evidence_and_event",
        ),
    ),
    (
        "F16",
        "DEFERRED",
        "R17, R18",
        (
            "tests/test_commitment_system.py::test_unreachable_compaction_and_unverified_outcome_records_are_rejected",
            "tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods",
            "tests/test_goal_system.py::test_ingestion_retains_only_proposal_and_exact_replay_is_idempotent",
        ),
    ),
    (
        "F17",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_r13_wal_recovery.py::test_committed_v8_crash_reconstructs_without_policy_replay",
            "tests/test_agent_state_v8.py::test_v8_round_trip_preserves_complete_empty_and_nontrivial_r13_graphs",
            "tests/test_r13_restore_ports.py::test_motivation_restore_preserves_exact_graph_and_event_replay",
            "tests/test_r13_restore_ports.py::test_goal_restore_preserves_ingestion_admission_replay_and_compaction",
            "tests/test_r13_restore_ports.py::test_commitment_restore_preserves_proposal_and_terminal_proof_replay",
            "tests/test_r13_integration.py::test_populated_v8_restarts_exactly_and_model_response_never_changes_r13",
            "tests/test_r13_integration.py::test_successful_injected_commit_publishes_one_bundle_and_retry_is_idempotent",
        ),
    ),
    (
        "F18",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_r13_projection.py::test_projection_excludes_each_noncurrent_lifecycle_and_all_proofs",
            "tests/test_r13_evidence_matrix.py::test_main_loop_r13_ports_are_private_and_prompt_projection_is_pure",
            "tests/test_r13_codec.py::test_state_snapshot_requires_all_domains_and_forbids_private_extra_fields",
            "tests/test_r13_integration.py::test_standard_prompt_renders_only_minimal_current_r13_authority",
            "tests/test_r13_integration.py::test_production_ports_upgrade_retained_v7_on_first_commit_without_feedback",
        ),
    ),
    (
        "F19",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_r13_contract_bounds.py::test_schema_budget_is_reproducible_and_preserves_the_v8_future_reserve",
            "tests/test_r13_codec.py::test_capacity_gate_uses_derived_codec_maxima_and_utf8_domain_bytes",
            "tests/test_r13_projection.py::test_projection_includes_all_active_records_and_derives_exact_size",
            "tests/test_r13_projection.py::test_extreme_unicode_is_escaped_and_output_never_shortens_records",
            "tests/test_r13_evidence_matrix.py::test_documented_projection_bounds_match_derived_schema",
            "tests/test_motivation_system.py::test_record_capacity_event_budget_and_candidate_capacity_fail_atomically",
            "tests/test_goal_system.py::test_full_u1_goal_record_bound_is_preserved_and_next_record_fails",
            "tests/test_commitment_system.py::test_full_record_and_receipt_bounds_accept_maximum_reject_one_over",
        ),
    ),
    (
        "F20",
        "DIRECT_R13_PASS",
        "—",
        (
            "tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences",
            "tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable",
            "tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value",
            "tests/test_working_memory.py::test_projection_never_exceeds_utf8_byte_budget",
            "tests/test_chat_context.py::test_contextual_working_memory_uses_compatibility_without_mutating_authority",
            "tests/test_appraisal.py::test_appraisal_is_pure_deterministic_and_does_not_consume_private_text",
            "tests/test_value_system.py::test_prompt_projection_is_immutable_and_excludes_authority_metadata",
            "tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak",
            "tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active",
            "tests/test_agent_state_v8.py::test_v7_belief_capture_stays_v7_and_r13_capture_upgrades_to_v8",
            "tests/test_main_loop.py::test_failed_agent_state_restore_keeps_all_previous_committed_views",
            "tests/test_fastapi_backend.py::test_server_commit_does_not_call_legacy_view_publishers_after_durability",
        ),
    ),
    (
        "F21",
        "DEFERRED",
        "R19",
        (
            "tests/test_goal_system.py::test_graph_conflict_facts_and_snapshot_export_are_exact_and_side_effect_free",
            "tests/test_commitment_system.py::test_desire_goal_and_deadline_are_references_not_automatic_authority",
            "tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods",
        ),
    ),
    (
        "F22",
        "REPRESENTATIONAL_NO_PRODUCER",
        "—",
        (
            "tests/test_r13_evidence_matrix.py::test_main_loop_r13_surface_has_no_producer_or_later_authority_calls",
            "tests/test_r13_integration.py::test_production_mainloop_exposes_no_mutable_r13_read_or_later_producer",
            "tests/test_r13_integration.py::test_populated_v8_restarts_exactly_and_model_response_never_changes_r13",
            "tests/test_motivation_system.py::test_system_api_import_does_not_load_models_runtime_or_timer_modules",
            "tests/test_goal_system.py::test_goal_system_module_has_no_later_runtime_model_or_scheduler_imports",
            "tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods",
        ),
    ),
)


RECOVERY_CRASH_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "production v7 startup and first ordinary v8 commit",
        "tests/test_r13_integration.py::test_production_ports_upgrade_retained_v7_on_first_commit_without_feedback",
    ),
    (
        "v7 bootstrap and first ordinary v8 commit",
        "tests/test_r13_wal_recovery.py::test_first_normal_v8_commit_after_v7_is_lazy_and_preserves_v7_lineage",
    ),
    (
        "failed first v8 publication",
        "tests/test_r13_wal_recovery.py::test_failed_first_v8_publication_keeps_canonical_v7_bytes",
    ),
    (
        "committed-v8 restart without policy replay",
        "tests/test_r13_wal_recovery.py::test_committed_v8_crash_reconstructs_without_policy_replay",
    ),
    (
        "malformed persisted v8",
        "tests/test_r13_wal_recovery.py::test_malformed_persisted_v8_snapshot_is_rejected",
    ),
    (
        "ordinary WAL v8 downgrade rejected",
        "tests/test_r13_wal_recovery.py::test_ordinary_v8_to_v7_wal_transition_is_rejected",
    ),
    (
        "internal v8 downgrade rejected before Journal prepare",
        "tests/test_r13_wal_recovery.py::test_internal_commit_rejects_v8_to_v7_before_prepare",
    ),
    (
        "true rollback to retained v7 clears R13",
        "tests/test_r13_wal_recovery.py::test_true_rollback_from_v8_to_v7_clears_r13_restore_ports",
    ),
    (
        "true rollback to older v8 restores exact R13",
        "tests/test_r13_wal_recovery.py::test_true_rollback_to_older_v8_restores_exact_r13_authority",
    ),
    (
        "unpublished production v8 failure and exact restart",
        "tests/test_r13_integration.py::test_failed_v8_publication_keeps_committed_r13_view_and_restart_authority",
    ),
    (
        "published unconfirmed v8 failure fail-stops and restarts exactly",
        "tests/test_r13_integration.py::test_published_v8_failure_fail_stops_and_restart_recovers_exact_authority",
    ),
    (
        "projection preflight rejects before durable publication",
        "tests/test_r13_integration.py::test_projection_preflight_failure_precedes_durable_publication",
    ),
    (
        "whole restore keeps previous public views during failure",
        "tests/test_main_loop.py::test_failed_agent_state_restore_keeps_all_previous_committed_views",
    ),
)

RESTART_NO_REPLAY_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "populated production MainLoop restart and no model feedback",
        "tests/test_r13_integration.py::test_populated_v8_restarts_exactly_and_model_response_never_changes_r13",
    ),
    (
        "empty and populated v8 exact round-trip",
        "tests/test_agent_state_v8.py::test_v8_round_trip_preserves_complete_empty_and_nontrivial_r13_graphs",
    ),
    (
        "mixed retained-v7 and v8 WAL snapshots",
        "tests/test_r13_wal_recovery.py::test_mixed_retained_v7_then_v8_wal_reconstructs_exact_snapshots_and_hash_chain",
    ),
    (
        "Motivation restore and replay",
        "tests/test_r13_restore_ports.py::test_motivation_restore_preserves_exact_graph_and_event_replay",
    ),
    (
        "Goal restore and replay",
        "tests/test_r13_restore_ports.py::test_goal_restore_preserves_ingestion_admission_replay_and_compaction",
    ),
    (
        "Commitment restore and replay",
        "tests/test_r13_restore_ports.py::test_commitment_restore_preserves_proposal_and_terminal_proof_replay",
    ),
)

PRIVACY_SENTINEL_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "standard prompt omits retained admissions proofs and receipts",
        "tests/test_r13_integration.py::test_standard_prompt_renders_only_minimal_current_r13_authority",
    ),
    (
        "ordinary and debug chat have no model-output feedback",
        "tests/test_r13_integration.py::test_production_ports_upgrade_retained_v7_on_first_commit_without_feedback",
    ),
    (
        "projection excludes subject and proofs",
        "tests/test_r13_projection.py::test_projection_excludes_each_noncurrent_lifecycle_and_all_proofs",
    ),
    (
        "private MainLoop projection read",
        "tests/test_r13_evidence_matrix.py::test_main_loop_r13_ports_are_private_and_prompt_projection_is_pure",
    ),
    (
        "persistence rejects private extra fields",
        "tests/test_r13_codec.py::test_state_snapshot_requires_all_domains_and_forbids_private_extra_fields",
    ),
    (
        "private thought remains ephemeral",
        "tests/test_main_loop.py::test_debug_trace_exposes_private_thought_only_ephemerally",
    ),
)

SIZE_CAPACITY_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "32 active entries per prompt domain and exact derived size",
        "tests/test_r13_projection.py::test_projection_includes_all_active_records_and_derives_exact_size",
    ),
    (
        "extreme Unicode and no shortening",
        "tests/test_r13_projection.py::test_extreme_unicode_is_escaped_and_output_never_shortens_records",
    ),
    (
        "documented count and byte limits match derived prompt schema",
        "tests/test_r13_evidence_matrix.py::test_documented_projection_bounds_match_derived_schema",
    ),
    (
        "R13 schema budget and future reserve",
        "tests/test_r13_contract_bounds.py::test_schema_budget_is_reproducible_and_preserves_the_v8_future_reserve",
    ),
    (
        "codec aggregate and UTF-8 capacity",
        "tests/test_r13_codec.py::test_capacity_gate_uses_derived_codec_maxima_and_utf8_domain_bytes",
    ),
    (
        "AgentState v8 126350904-byte schema maximum",
        "tests/test_agent_state_capacity.py::test_v8_schema_maxima_cover_all_r13_domains_and_keep_future_reserve",
    ),
    (
        "Motivation atomic capacity failure",
        "tests/test_motivation_system.py::test_record_capacity_event_budget_and_candidate_capacity_fail_atomically",
    ),
    (
        "Goal record maximum and overflow",
        "tests/test_goal_system.py::test_full_u1_goal_record_bound_is_preserved_and_next_record_fails",
    ),
    (
        "Commitment record maximum and overflow",
        "tests/test_commitment_system.py::test_full_record_and_receipt_bounds_accept_maximum_reject_one_over",
    ),
)

COMPATIBILITY_NON_INVASION_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "R03 runtime FIFO",
        "tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences",
    ),
    (
        "R04–R06 state and WAL durability",
        "tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable",
    ),
    (
        "R07 transaction participants",
        "tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value",
    ),
    (
        "R08 WorkingMemory projection",
        "tests/test_working_memory.py::test_projection_never_exceeds_utf8_byte_budget",
    ),
    (
        "R09 Context",
        "tests/test_chat_context.py::test_contextual_working_memory_uses_compatibility_without_mutating_authority",
    ),
    (
        "R10 appraisal",
        "tests/test_appraisal.py::test_appraisal_is_pure_deterministic_and_does_not_consume_private_text",
    ),
    (
        "R11 Value projection",
        "tests/test_value_system.py::test_prompt_projection_is_immutable_and_excludes_authority_metadata",
    ),
    (
        "public chat",
        "tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak",
    ),
    (
        "sleep/training candidate boundary",
        "tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active",
    ),
    (
        "v7 capture and v8 upgrade behavior",
        "tests/test_agent_state_v8.py::test_v7_belief_capture_stays_v7_and_r13_capture_upgrades_to_v8",
    ),
    (
        "whole restore keeps Value Belief and R13 public views unchanged on failure",
        "tests/test_main_loop.py::test_failed_agent_state_restore_keeps_all_previous_committed_views",
    ),
    (
        "server commit does no fallible per-domain view publication after durability",
        "tests/test_fastapi_backend.py::test_server_commit_does_not_call_legacy_view_publishers_after_durability",
    ),
)

PRODUCER_AUTHORITY_BOUNDARY_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "private MainLoop authority injection",
        "tests/test_main_loop.py::test_main_loop_clones_injected_r13_authorities_and_publishes_their_view",
    ),
    (
        "committed projection publishes only after staging",
        "tests/test_main_loop.py::test_main_loop_r13_view_changes_only_after_staged_publication",
    ),
    (
        "direct restore does not publish prompt view",
        "tests/test_main_loop.py::test_main_loop_direct_r13_restore_ports_do_not_publish_prompt_view",
    ),
    (
        "injected handler or Goal ingestion failure rolls back without view leakage",
        "tests/test_r13_integration.py::test_failed_injected_domain_operation_rolls_back_without_speculative_view",
    ),
    (
        "successful injected commit and exact domain retry",
        "tests/test_r13_integration.py::test_successful_injected_commit_publishes_one_bundle_and_retry_is_idempotent",
    ),
    (
        "production has no mutable public R13 system or later producer",
        "tests/test_r13_integration.py::test_production_mainloop_exposes_no_mutable_r13_read_or_later_producer",
    ),
    (
        "no MainLoop producer or later-authority calls",
        "tests/test_r13_evidence_matrix.py::test_main_loop_r13_surface_has_no_producer_or_later_authority_calls",
    ),
    (
        "Motivation model/runtime import boundary",
        "tests/test_motivation_system.py::test_system_api_import_does_not_load_models_runtime_or_timer_modules",
    ),
    (
        "Goal later-authority import boundary",
        "tests/test_goal_system.py::test_goal_system_module_has_no_later_runtime_model_or_scheduler_imports",
    ),
    (
        "Commitment later-authority import boundary",
        "tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods",
    ),
)

R13_CROSSCUTTING_MATRIX: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("recovery_crash", RECOVERY_CRASH_MATRIX),
    ("restart_no_replay", RESTART_NO_REPLAY_MATRIX),
    ("privacy", PRIVACY_SENTINEL_MATRIX),
    ("size_capacity", SIZE_CAPACITY_MATRIX),
    ("compatibility_non_invasion", COMPATIBILITY_NON_INVASION_MATRIX),
    ("producer_authority_boundary", PRODUCER_AUTHORITY_BOUNDARY_MATRIX),
)


def _assert_test_reference(reference: str, repository_root: Path) -> None:
    test_path, test_name = reference.split("::", maxsplit=1)
    assert test_path.startswith("tests/"), f"evidence must be a test path: {reference}"
    path = repository_root / test_path
    assert path.is_file(), f"missing evidence file: {test_path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert test_name in names, f"missing executable evidence test: {reference}"


def _document_f_rows(
    document: str,
) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    rows: list[tuple[str, str, str, tuple[str, ...]]] = []
    for line in document.splitlines():
        if not line.startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if not cells or re.fullmatch(r"F\d+", cells[0]) is None:
            continue
        references = tuple(re.findall(r"`(tests/[^`]+::test_[^`]+)`", line))
        rows.append((cells[0], cells[1], cells[2], references))
    return tuple(rows)


def _document_crosscutting_rows(
    document: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    expected_keys = {key for key, _matrix in R13_CROSSCUTTING_MATRIX}
    rows: list[tuple[str, tuple[str, ...]]] = []
    for line in document.splitlines():
        if not line.startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if not cells or cells[0].strip("`") not in expected_keys:
            continue
        references = tuple(re.findall(r"`(tests/[^`]+::test_[^`]+)`", line))
        rows.append((cells[0].strip("`"), references))
    return tuple(rows)


def test_main_loop_r13_ports_are_private_and_prompt_projection_is_pure(
    tmp_path: Path,
) -> None:
    constructor_parameters = inspect.signature(SuzkaMainLoop).parameters
    assert {
        "motivation_system",
        "goal_system",
        "commitment_system",
    } <= constructor_parameters.keys()

    settings = load_settings(ROOT / "config.yaml")
    settings = settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "memory",
                    "db1_collection": "r13_evidence_episodic",
                    "db2_collection": "r13_evidence_semantic",
                }
            )
        }
    )

    injected_motivation = MotivationSystem()
    injected_motivation.restore_motivation_state(_motivation_snapshot())
    injected_goal = GoalSystem()
    injected_goal.restore_goal_state(_goal_snapshot())
    injected_commitment = CommitmentSystem()
    injected_commitment.restore_commitment_state(_commitment_snapshot())
    loop = SuzkaMainLoop(
        settings,
        DummyProvider(),
        DualMemorySystem(settings),
        motivation_system=injected_motivation,
        goal_system=injected_goal,
        commitment_system=injected_commitment,
    )

    assert loop._motivation_system is not injected_motivation
    assert loop._goal_system is not injected_goal
    assert loop._commitment_system is not injected_commitment
    assert not hasattr(loop, "motivation_system")
    assert not hasattr(loop, "goal_system")
    assert not hasattr(loop, "commitment_system")

    before = (
        loop.export_motivation_state(),
        loop.export_goal_state(),
        loop.export_commitment_state(),
    )
    view = loop.r13_view()
    assert loop.r13_view() is view
    working_memory_view = WorkingMemoryView(
        selected=(),
        decisions=(),
        projected_bytes=0,
        item_capacity=loop.working_memory.item_capacity,
        projection_max_bytes=loop.working_memory.projection_max_bytes,
        revision=loop.working_memory.revision,
    )
    prompt = loop._build_prompt(
        "read-only R13 prompt",
        EmotionState(),
        working_memory_view,
        ContextPromptView(
            context_id="conversation-default",
            context_type="conversation",
            source_channel="chat",
            source_session_id=None,
            participant_refs=(),
        ),
        ValuePromptView(()),
    )

    assert view.render() in prompt
    assert "Current Motivations:" in prompt
    assert "Adopted Goals:" in prompt
    assert "Active Commitments:" in prompt
    assert all(record.subject not in prompt for record in before[2].records)
    assert "event:proposal:" not in prompt
    assert before == (
        loop.export_motivation_state(),
        loop.export_goal_state(),
        loop.export_commitment_state(),
    )


def test_main_loop_r13_surface_has_no_producer_or_later_authority_calls() -> None:
    source_path = ROOT / "suzka" / "runtime" / "main_loop.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    main_loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SuzkaMainLoop"
    )
    r13_owners = {
        "self._motivation_system",
        "self._goal_system",
        "self._commitment_system",
    }
    forbidden_operations = {
        "apply_evidence",
        "ingest_candidate",
        "propose_goal",
        "ingest_proposal",
        "adopt",
        "defer",
        "resume",
        "abandon",
        "accept",
        "release",
        "renegotiate",
        "complete",
        "fail",
        "fulfill",
        "breach",
        "expire",
        "tick",
        "schedule",
    }
    calls = (
        node
        for node in ast.walk(main_loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    r13_calls = {
        (ast.unparse(node.func.value), node.func.attr)
        for node in calls
        if ast.unparse(node.func.value) in r13_owners
    }
    assert not {
        call for call in r13_calls if call[1] in forbidden_operations
    }
    assert {
        "_motivation_system",
        "_goal_system",
        "_commitment_system",
    } <= {
        node.attr
        for node in ast.walk(main_loop)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def test_r13_f_matrix_is_complete_classified_and_matches_documented_evidence() -> None:
    assert tuple(row[0] for row in R13_F_MATRIX) == tuple(
        f"F{number}" for number in range(1, 23)
    )
    allowed_classifications = {
        "DIRECT_R13_PASS",
        "REPRESENTATIONAL_NO_PRODUCER",
        "DEFERRED",
    }
    for requirement, classification, owner, references in R13_F_MATRIX:
        assert classification in allowed_classifications, requirement
        assert references, requirement
        if classification == "DEFERRED":
            assert re.findall(r"R1[4-9]", owner), (requirement, owner)
        else:
            assert owner == "—", (requirement, owner)
        for reference in references:
            _assert_test_reference(reference, ROOT)

    documented = _document_f_rows(EVIDENCE_DOCUMENT.read_text(encoding="utf-8"))
    assert documented == R13_F_MATRIX


def test_documented_projection_bounds_match_derived_schema() -> None:
    document = EVIDENCE_DOCUMENT.read_text(encoding="utf-8")
    assert f"| Records per domain | {R13_PROMPT_MAX_RECORDS_PER_DOMAIN} |" in document
    assert f"| Canonical JSON | {R13_PROMPT_MAX_SERIALIZED_BYTES:,} |" in document
    assert f"| Rendered UTF-8 | {R13_PROMPT_MAX_RENDERED_BYTES:,} |" in document


def test_r13_crosscutting_matrices_have_explicit_keys_and_match_document() -> None:
    expected_keys = (
        "recovery_crash",
        "restart_no_replay",
        "privacy",
        "size_capacity",
        "compatibility_non_invasion",
        "producer_authority_boundary",
    )
    assert tuple(key for key, _matrix in R13_CROSSCUTTING_MATRIX) == expected_keys

    expected_document_rows: list[tuple[str, tuple[str, ...]]] = []
    for key, matrix in R13_CROSSCUTTING_MATRIX:
        assert matrix, key
        references = tuple(reference for _case, reference in matrix)
        expected_document_rows.append((key, references))
        for reference in references:
            _assert_test_reference(reference, ROOT)

    document = EVIDENCE_DOCUMENT.read_text(encoding="utf-8")
    assert _document_crosscutting_rows(document) == tuple(expected_document_rows)
