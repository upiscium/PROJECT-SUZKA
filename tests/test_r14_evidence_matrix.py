"""U6 R14 evidence-map consistency and closed-source-surface checks."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DOCUMENT = ROOT / "docs" / "r14-evidence.md"
INLINE_CODE_PATTERN = re.compile(r"`([^`]*)`")
TEST_REFERENCE_PATTERN = re.compile(
    r"tests/[A-Za-z0-9_./-]+\.py::test_[A-Za-z0-9_]+"
)


R14_F_MATRIX: tuple[
    tuple[str, str, str, str, str, tuple[str, ...]], ...
] = (
    (
        "F1",
        "DIRECT_R14_EVIDENCE",
        "—",
        "High salience transient can't permanently evict persistent Goal/Commitment without bounded switch/persistence.",
        "Real R08 transient challengers displace current Goal/Commitment focus with fixed switch cost and bounded unfinished evidence; removal restores the unchanged current R13 focus at the next event. This is bounded continuity, not eternal focus or absolute source-kind priority.",
        (
            "tests/test_attention_policy.py::test_continuity_bonus_switch_cost_hysteresis_and_displaced_unfinished",
            "tests/test_attention_system.py::test_salient_challenger_eventually_switches_after_repeat_focus_habituation",
            "tests/test_r14_focus_regressions.py::test_transient_r08_challengers_displace_then_restore_real_r13_focus",
        ),
    ),
    (
        "F2",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Multiple high-arousal candidates can't monopolize all finite capacity.",
        "One global focus ceiling falls from 16 to 8 at arousal >= 0.75 without changing candidate scores. There are no per-kind quotas, reserved slots, or fairness guarantee; a single kind can occupy the bounded set.",
        (
            "tests/test_attention_system.py::test_high_global_arousal_only_caps_focus_and_does_not_change_candidate_scores",
            "tests/test_attention_policy.py::test_unknown_global_emotion_is_explicit_and_emotion_only_changes_capacity",
        ),
    ),
    (
        "F3",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Idle when no eligible candidate exists.",
        "An empty, unavailable-only, or below-threshold candidate set yields an explicit idle outcome; no source is promoted to make a winner.",
        (
            "tests/test_attention_policy.py::test_policy_is_idle_when_no_candidate_is_eligible_or_above_threshold",
            "tests/test_attention_system.py::test_bootstrap_empty_refresh_and_current_exact_retry_are_receipted",
        ),
    ),
    (
        "F4",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Typed provenance; no raw content persisted.",
        "Attention continuity retains closed typed references, source witnesses, digests, and bounded counters, not rendered source text or signal vectors. Hashes bind declared values and do not authenticate producers.",
        (
            "tests/test_attention_contracts.py::test_projection_is_closed_reference_only_and_preserves_signal_missingness",
            "tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out",
            "tests/test_attention_state_codec.py::test_full_legal_candidate_universe_is_preserved_without_truncation",
        ),
    ),
    (
        "F5",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Selection does not mutate source authorities.",
        "Adapters and selection are read-only projections over caller-supplied current snapshots/views. Attention does not mutate, admit, resolve, or revise Motivation, Goal, Commitment, Working Memory, or Emotion truth.",
        (
            "tests/test_attention_system.py::test_real_source_adapters_feed_exact_current_targets_without_mutating_sources",
            "tests/test_attention_policy.py::test_policy_and_prompt_are_source_immutable_and_revalidate_event_bindings",
            "tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful",
        ),
    ),
    (
        "F6",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Deterministic from prior authority, typed projections, and explicit event.",
        "Canonical typed inputs and event bind deterministic competition and continuity. The separate refresh-operation receipt digest binds projections/event/Emotion for retry identity and intentionally excludes prior authority; competition itself binds the prior root.",
        (
            "tests/test_attention_policy.py::test_ties_and_input_permutations_have_canonical_total_order_and_digest",
            "tests/test_attention_system.py::test_tied_scores_and_input_permutations_produce_identical_state_and_views",
            "tests/test_attention_system_boundaries.py::test_input_digest_binds_event_sorted_projections_and_emotion_not_prior_authority",
            "tests/test_attention_system.py::test_same_primary_revision_must_keep_its_digest_and_source_revision_is_independent",
        ),
    ),
    (
        "F7",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Retry is idempotent and does not inflate habituation.",
        "An exact retained event plus operation input returns a replay receipt/current view without scoring or continuity arithmetic. Older unverifiable retries fail closed; the bounded receipt window is not an infinite replay ledger.",
        (
            "tests/test_attention_system.py::test_bootstrap_empty_refresh_and_current_exact_retry_are_receipted",
            "tests/test_attention_system.py::test_historical_exact_retry_returns_current_state_and_original_accepted_receipt",
            "tests/test_attention_system.py::test_constructor_and_restore_retry_without_reranking_or_habituation_helpers",
        ),
    ),
    (
        "F8",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Source removal or inactivation is explicit.",
        "The trusted complete current capture determines membership. Absent candidates leave current continuity; unavailable/inactive rows lose focus and unfinished references; no shadow cache restores removed state.",
        (
            "tests/test_attention_system.py::test_unavailable_or_inactive_rows_remain_but_lose_references_and_no_cache",
            "tests/test_attention_system.py::test_unfinished_references_survive_only_while_the_complete_current_set_has_them",
            "tests/test_attention_adapters.py::test_snapshot_batch_requires_complete_working_memory_membership_coverage",
        ),
    ),
    (
        "F9",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Unfinished evidence and switch cost are bounded and explicit.",
        "A displaced focused item can remain unfinished only while current and eligible. Its union with prior unfinished IDs is bounded at 16; overflow rejects with explicit evidence rather than truncation. Switching has a fixed cost, not an absolute veto.",
        (
            "tests/test_attention_policy.py::test_continuity_bonus_switch_cost_hysteresis_and_displaced_unfinished",
            "tests/test_attention_policy.py::test_unfinished_overflow_returns_bounded_fail_closed_evidence_without_slicing",
            "tests/test_attention_system_boundaries.py::test_unfinished_reference_overflow_is_atomic_without_truncation",
        ),
    ),
    (
        "F10",
        "DIRECT_R14_EVIDENCE",
        "—",
        "No truncation outside Attention.",
        "Attention omits whole oversized selected rows and never shortens source records. The 131072-byte bound applies only to the Attention-selected contribution, not the entire model prompt or other source projections.",
        (
            "tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation",
            "tests/test_attention_prompt.py::test_attention_builder_rejects_combined_legacy_r13_fallback",
            "tests/test_r14_u5_scope.py::test_u5_document_limits_are_selected_contribution_not_whole_prompt",
        ),
    ),
    (
        "F11",
        "EXCLUDED_DEFERRED",
        "R16",
        "No Goal/Action winner is selected by Attention.",
        "R14 ranks bounded attention resources only. The frozen prompt frame disclaims Goal winners/actions; the later Decision owner is not implemented by this evidence scope.",
        (
            "tests/test_attention_policy.py::test_prompt_frame_uses_parent_frozen_authority_intro_and_literal_headings",
            "tests/test_attention_system.py::test_source_kind_never_overrides_fixed_numeric_competition",
            "tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local",
        ),
    ),
    (
        "F12",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Confidence is evidence-bound, not model certainty.",
        "U3 confidence requires typed supporting recorded-confidence bounds, verified link coverage, and observed quality. No model self-report, source similarity, normative Value confidence, or raw text creates confidence; U5 does not wire optional U3 assessment into production.",
        (
            "tests/test_metacognition_contracts.py::test_confidence_requires_a_typed_support_ceiling_and_unknown_is_never_confident",
            "tests/test_metacognition_assessment_boundaries.py::test_source_confidence_uses_only_current_active_belief_records",
            "tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters",
        ),
    ),
    (
        "F13",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Missing evidence is not contradiction and is not overload.",
        "U3 preserves unknown versus measured zero, requires typed current structured conflicts for CONTRADICTORY, and never emits SUFFICIENT. Load/saturation cannot turn absent links into contradictions or complete knowledge.",
        (
            "tests/test_metacognition_contracts.py::test_missing_unknown_and_measured_zero_remain_distinct",
            "tests/test_metacognition_assessment.py::test_only_typed_current_structured_belief_conflicts_yield_contradiction",
            "tests/test_metacognition_assessment_boundaries.py::test_all_unknown_nonempty_source_witnesses_assess_to_all_unknown_metrics",
            "tests/test_metacognition_assessment_boundaries.py::test_source_confidence_uses_only_current_active_belief_records",
        ),
    ),
    (
        "F14",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Load, saturation, and Emotion quality cannot rewrite source truth.",
        "U3 resource heuristics affect event-scoped assessment quality only. Attention/Metacognition reads and assessment operations do not mutate source authorities, Attention continuity, Belief truth, or persisted source state. The live-turn integration assessment is invoked only by its test harness; it does not wire U3 into production.",
        (
            "tests/test_metacognition_assessment_boundaries.py::test_source_text_is_not_retained_or_scored_and_assessment_is_pure",
            "tests/test_metacognition_assessment_boundaries.py::test_maximum_attention_root_and_restore_remain_unchanged_by_explicit_assessment",
            "tests/test_attention_system.py::test_real_source_adapters_feed_exact_current_targets_without_mutating_sources",
            "tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only",
        ),
    ),
    (
        "F15",
        "EXCLUDED_DEFERRED",
        "R15",
        "No capability authority before R15.",
        "R14 adds no capability, trait, bias, or self-model producer. The closed U3 assessment shape has no such fields; no claim is made about a later SelfModel implementation.",
        (
            "tests/test_r14_u1_scope.py::test_metacognition_is_neither_persisted_history_nor_a_decision_contract",
            "tests/test_r14_u3_scope.py::test_u3_modules_have_no_runtime_or_mutation_authority",
            "tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence",
        ),
    ),
    (
        "F16",
        "EXCLUDED_DEFERRED",
        "R16",
        "No Decision recommendations before R16.",
        "No R14 Attention or U3 assessment field/API supplies a Decision recommendation or Action. R16 owns any later Decision authority; this is an absence boundary, not a delivered recommendation feature.",
        (
            "tests/test_metacognition_contracts.py::test_values_are_frozen_slot_closed_and_have_no_history_or_metadata_fields",
            "tests/test_r14_u1_scope.py::test_metacognition_is_neither_persisted_history_nor_a_decision_contract",
            "tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters",
        ),
    ),
    (
        "F17",
        "EXCLUDED_DEFERRED",
        "R17, R18",
        "No unverified outcome calibration before R17/R18.",
        "R14 persists no MetacognitiveAssessment history/calibration and creates no outcome learner. R17/R18 own later verified-outcome authority; checksum integrity is not producer authentication.",
        (
            "tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence",
            "tests/test_r14_u4_scope.py::test_u4_snapshot_has_no_assessment_or_rendered_source_fields",
            "tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay",
        ),
    ),
    (
        "F18",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Restore is cognition-free and exact.",
        "U4 restores the exact persisted Attention continuity through its owner port without reranking, source lookup, prompt reconstruction, or U3 assessment. Transient competition/prompt witnesses remain absent after restore; true rollback to retained v8 bootstraps Attention.",
        (
            "tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy",
            "tests/test_agent_state_v9.py::test_v9_alias_round_trip_uses_canonical_attention_and_requires_owner",
            "tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay",
            "tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay",
        ),
    ),
    (
        "F19",
        "DIRECT_R14_EVIDENCE",
        "—",
        "No raw prompt/transcript/hidden/model/provider state is durable R14 state.",
        "Attention continuity and committed read views contain bounded typed references/witnesses, not raw text or provider state. U5's request-scoped payload may render selected source text but is not placed in the durable root; this is not a claim about other application storage.",
        (
            "tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out",
            "tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful",
            "tests/test_r14_u5_scope.py::test_u5_payload_is_ephemeral_sealed_and_has_distinct_byte_witnesses",
        ),
    ),
    (
        "F20",
        "DIRECT_R14_EVIDENCE",
        "—",
        "Bounds, future/malformed inputs, and revision errors fail closed; never truncate.",
        "Source, focus, unfinished, revision, receipt, byte, nesting, schema, checksum, and root-event fences reject invalid/over-bound input before publication or repair. Only Attention may omit a whole row under its own byte budget. The restored-root integration case exercises U5 producer-only refresh after restore; it is not full HTTP rollback or automatic migration.",
        (
            "tests/test_attention_contracts.py::test_continuity_rejects_future_versions_malformed_counters_and_dangling_focus",
            "tests/test_attention_contracts.py::test_deeply_nested_json_fails_closed_before_recursive_construction",
            "tests/test_attention_bounds.py::test_one_over_candidate_and_receipt_limits_are_rejected_not_clipped",
            "tests/test_attention_state_codec.py::test_v9_raw_attention_byte_limit_precedes_attention_decoder",
            "tests/test_agent_state_v9.py::test_v9_requires_a_closed_attention_field_and_root_event_fences",
            "tests/test_r14_wal_recovery.py::test_raw_v9_attention_tampering_is_rejected_before_wal_repair_or_truncation",
            "tests/test_r14_integration.py::test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation",
            "tests/test_r14_wal_recovery.py::test_ordinary_v9_to_legacy_transition_is_rejected_by_wal_without_mutation",
            "tests/test_r14_wal_recovery.py::test_ordinary_v9_internal_commit_rejects_legacy_before_journal_prepare",
        ),
    ),
    (
        "F21",
        "DIRECT_R14_EVIDENCE",
        "—",
        "R03–R13 contracts remain non-invaded and compatible through recovery.",
        "R14 is additive to the v8 base and preserves tested runtime/WAL/transaction, context, Value, appraisal, API, and Sleep candidate boundaries. V8 remains lazy-compatible; explicit true rollback to v8 retains R13 and bootstraps empty Attention by the accepted legacy rule.",
        (
            "tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences",
            "tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable",
            "tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value",
            "tests/test_agent_state_v9.py::test_v8_schema_and_capture_remain_explicitly_legacy",
            "tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_to_retained_v8_bootstraps_attention_and_keeps_r13",
        ),
    ),
    (
        "F22",
        "EXCLUDED_DEFERRED",
        "R19",
        "No scheduler, wake, or Outbox authority before R19.",
        "AttentionSystem exposes only refresh, restore_snapshot, selected_view, and snapshot; no scheduler, wake, Outbox, timer, or R19 mode is delivered. Existing finite explicit Sleep/QLoRA candidate work is not an Attention scheduler or R19#279 mode.",
        (
            "tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local",
            "tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities",
            "tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active",
        ),
    ),
)


R14_A_MATRIX: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "A1",
        "Attention and Metacognition are distinct: source focus is not winner selection, and confidence is not truth.",
        ("F11", "F12", "F13"),
        (
            "tests/test_attention_policy.py::test_prompt_frame_uses_parent_frozen_authority_intro_and_literal_headings",
            "tests/test_metacognition_contracts.py::test_confidence_requires_a_typed_support_ceiling_and_unknown_is_never_confident",
            "tests/test_metacognition_assessment.py::test_resource_only_assessment_never_claims_factual_confidence",
        ),
    ),
    (
        "A2",
        "Attention owns persisted intrinsic continuity, distinct from source truth and transient selection witnesses.",
        ("F4", "F18"),
        (
            "tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out",
            "tests/test_agent_state_v9.py::test_v9_alias_round_trip_uses_canonical_attention_and_requires_owner",
            "tests/test_r14_wal_recovery.py::test_mixed_v8_v9_v9_wal_reconstructs_exact_attention_roots_and_receipts",
        ),
    ),
    (
        "A3",
        "Metacognition is event-scoped and non-durable; it has no production wiring or assessment history.",
        ("F12", "F17", "F19"),
        (
            "tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence",
            "tests/test_r14_u4_scope.py::test_u4_snapshot_has_no_assessment_or_rendered_source_fields",
            "tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only",
        ),
    ),
    (
        "A4",
        "The current source universe is 4096 WM plus ACTIVE Motivation, ADOPTED Goal, and ACTIVE Commitment bounds for 4192; do not scan historical records.",
        ("F8", "F20"),
        (
            "tests/test_attention_bounds.py::test_supported_identity_capacity_is_source_owned_and_complete",
            "tests/test_attention_adapters.py::test_snapshot_batch_covers_full_4192_source_universe_canonically",
            "tests/test_attention_main_loop.py::test_real_main_loop_projects_the_complete_4192_candidate_universe",
        ),
    ),
    (
        "A5",
        "Other source signals require exact typed links; free-text similarity is not link or confidence authority.",
        ("F5", "F12", "F13"),
        (
            "tests/test_metacognition_evidence.py::test_linked_belief_uses_only_current_typed_r13_references_and_scope_gate",
            "tests/test_metacognition_evidence.py::test_belief_links_are_taken_from_motivation_related_goal_and_commitment_refs",
            "tests/test_metacognition_assessment.py::test_only_typed_current_structured_belief_conflicts_yield_contradiction",
        ),
    ),
    (
        "A6",
        "Raw user/model text and model self-reports do not become Attention or Metacognition authority.",
        ("F4", "F12", "F19"),
        (
            "tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters",
            "tests/test_metacognition_assessment_boundaries.py::test_source_text_is_not_retained_or_scored_and_assessment_is_pure",
            "tests/test_r14_u5_scope.py::test_u5_producer_rejects_caller_forged_context_even_on_actual_worker",
        ),
    ),
    (
        "A7",
        "The four-stage Attention pipeline is fixed and deterministic over typed current inputs, prior continuity, and the event.",
        ("F6", "F7", "F10"),
        (
            "tests/test_attention_policy.py::test_ties_and_input_permutations_have_canonical_total_order_and_digest",
            "tests/test_attention_system.py::test_tied_scores_and_input_permutations_produce_identical_state_and_views",
            "tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt",
        ),
    ),
    (
        "A8",
        "Attention owns focus only: it does not mutate source truth, select Actions, or schedule work.",
        ("F5", "F11", "F22"),
        (
            "tests/test_attention_system.py::test_real_source_adapters_feed_exact_current_targets_without_mutating_sources",
            "tests/test_attention_system.py::test_source_kind_never_overrides_fixed_numeric_competition",
            "tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local",
        ),
    ),
    (
        "A9",
        "R08 owns Working Memory membership and resolution eligibility; R14 performs final focus and prompt selection without changing R08 selection.",
        ("F5", "F8", "F10"),
        (
            "tests/test_attention_adapters.py::test_selected_working_memory_row_is_exact_and_may_exceed_prompt_budget",
            "tests/test_attention_policy.py::test_unavailable_working_memory_keeps_measured_metadata_but_does_not_compete",
            "tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation",
        ),
    ),
    (
        "A10",
        "Prompt omission is explicit and bounded; rows are omitted whole and never truncated outside Attention.",
        ("F10", "F20"),
        (
            "tests/test_attention_policy.py::test_single_oversized_prompt_row_is_explicitly_omitted_not_truncated",
            "tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation",
            "tests/test_r14_u5_scope.py::test_u5_document_limits_are_selected_contribution_not_whole_prompt",
        ),
    ),
    (
        "A11",
        "The production Attention refresh and selected prompt are bound to the same serialized runtime event.",
        ("F6", "F10", "F21"),
        (
            "tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt",
            "tests/test_agent_runtime_ordered_time.py::test_real_chat_attention_belief_and_goal_keep_runtime_event_identity",
            "tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only",
        ),
    ),
    (
        "A12",
        "Current uncommitted Experience is not captured as current source evidence for its own response.",
        ("F5", "F8"),
        (
            "tests/test_main_loop.py::test_current_future_episode_is_absent_from_its_own_working_memory_view",
            "tests/test_r14_evidence_matrix.py::test_source_inventory_and_unwired_assessment_boundary_are_closed",
        ),
    ),
    (
        "A13",
        "Restart restores reference-first Attention continuity without source text, reranking, or prompt reconstruction.",
        ("F4", "F18", "F19"),
        (
            "tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay",
            "tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy",
            "tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out",
        ),
    ),
    (
        "A14",
        "Invalid, unavailable, inactive, removed, future, or stale sources follow explicit fail-closed transitions without repair.",
        ("F8", "F20"),
        (
            "tests/test_attention_system.py::test_unavailable_or_inactive_rows_remain_but_lose_references_and_no_cache",
            "tests/test_attention_adapters.py::test_r13_adapters_reject_future_events_and_ineligible_lifecycles",
            "tests/test_r14_integration.py::test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation",
        ),
    ),
    (
        "A15",
        "No direct operator refocus or producer API is added; AttentionSystem retains its closed process-local surface.",
        ("F11", "F22"),
        (
            "tests/test_attention_main_loop.py::test_main_loop_owns_an_isolated_attention_system_and_concrete_state_port",
            "tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local",
            "tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities",
        ),
    ),
    (
        "A16",
        "Metacognition reports only bounded current event-scoped resource, evidence, and confidence metrics.",
        ("F12", "F13", "F14"),
        (
            "tests/test_r14_u3_scope.py::test_u3_documented_policy_and_ephemeral_envelope_match_derivation",
            "tests/test_metacognition_assessment.py::test_fixed_integer_quality_and_ceiling_coverage_confidence_are_order_invariant",
            "tests/test_metacognition_evidence.py::test_current_working_memory_join_derives_load_without_retaining_rendered_content",
        ),
    ),
    (
        "A17",
        "Confidence is deterministic and evidence-bound; UNKNOWN, CONTRADICTORY, and overload are distinct, and missing evidence is not contradiction.",
        ("F12", "F13"),
        (
            "tests/test_metacognition_contracts.py::test_confidence_requires_a_typed_support_ceiling_and_unknown_is_never_confident",
            "tests/test_metacognition_contracts.py::test_missing_unknown_and_measured_zero_remain_distinct",
            "tests/test_metacognition_assessment_boundaries.py::test_all_unknown_nonempty_source_witnesses_assess_to_all_unknown_metrics",
            "tests/test_metacognition_assessment.py::test_only_typed_current_structured_belief_conflicts_yield_contradiction",
        ),
    ),
    (
        "A18",
        "R14 adds no Decision, autonomy, Action, or Metacognition feedback authority.",
        ("F11", "F16", "F17"),
        (
            "tests/test_r14_u1_scope.py::test_metacognition_is_neither_persisted_history_nor_a_decision_contract",
            "tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters",
            "tests/test_r14_u3_scope.py::test_u3_modules_have_no_runtime_or_mutation_authority",
        ),
    ),
    (
        "A19",
        "Metacognition does not create durable traits, bias, capability, or assessment history.",
        ("F15", "F17"),
        (
            "tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence",
            "tests/test_r14_u4_scope.py::test_u4_snapshot_has_no_assessment_or_rendered_source_fields",
            "tests/test_metacognition_contracts.py::test_values_are_frozen_slot_closed_and_have_no_history_or_metadata_fields",
        ),
    ),
    (
        "A20",
        "Capacity stays under hard cap 134217728 with reserve 16777216, v8 base 109573688, R14 maximum addition 7866824, and v9 plus reserve 130544419.",
        ("F20", "F21"),
        (
            "tests/test_attention_bounds.py::test_attention_state_value_and_exact_v9_reserved_projection_fit",
            "tests/test_agent_state_v9.py::test_first_v9_capacity_is_derived_from_the_exact_attention_field_bound",
            "tests/test_r14_u1_scope.py::test_u1_documented_capacity_and_resources_match_executable_derivation",
        ),
    ),
    (
        "A21",
        "Recovery is exact and cognition-free; restart does not replay Attention or Metacognition.",
        ("F18", "F20", "F21"),
        (
            "tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy",
            "tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay",
            "tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay",
        ),
    ),
    (
        "A22",
        "No raw prompt, transcript, hidden thought, model, or provider state is durable R14 state.",
        ("F4", "F19"),
        (
            "tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out",
            "tests/test_r14_u5_scope.py::test_u5_payload_is_ephemeral_sealed_and_has_distinct_byte_witnesses",
            "tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful",
        ),
    ),
    (
        "A23",
        "No scheduler/wake/Outbox is added; finite explicit Sleep is not R19#279 scheduling authority.",
        ("F22",),
        (
            "tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local",
            "tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active",
            "tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities",
        ),
    ),
    (
        "A24",
        "R14 adds no R15-R19 authority or producer.",
        ("F15", "F16", "F17", "F22"),
        (
            "tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities",
            "tests/test_r14_u3_scope.py::test_u3_modules_have_no_runtime_or_mutation_authority",
            "tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local",
        ),
    ),
)


RECOVERY_CRASH_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "ordinary v9 downgrade is rejected before WAL mutation",
        "tests/test_r14_wal_recovery.py::test_ordinary_v9_to_legacy_transition_is_rejected_by_wal_without_mutation",
    ),
    (
        "ordinary v9 downgrade is rejected before Journal prepare",
        "tests/test_r14_wal_recovery.py::test_ordinary_v9_internal_commit_rejects_legacy_before_journal_prepare",
    ),
    (
        "lazy v8-to-v9 and first normal commit",
        "tests/test_r14_wal_recovery.py::test_first_normal_v8_to_v9_commit_is_lazy_then_publishes_canonical_attention",
    ),
    (
        "pre-WAL and pre-canonical crash boundaries",
        "tests/test_r14_wal_recovery.py::test_v9_crash_before_internal_commit_recovers_prior_v8_without_replay",
    ),
    (
        "canonical publication before Journal completion",
        "tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay",
    ),
    (
        "true rollback to retained v9 and v8",
        "tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_restores_exact_attention_history_and_r13_ports",
    ),
    (
        "true rollback to v8 bootstraps Attention but keeps R13",
        "tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_to_retained_v8_bootstraps_attention_and_keeps_r13",
    ),
    (
        "malformed Attention fails before WAL repair or truncation",
        "tests/test_r14_wal_recovery.py::test_raw_v9_attention_tampering_is_rejected_before_wal_repair_or_truncation",
    ),
    (
        "persisted root event chronology fences survive restart",
        "tests/test_agent_state_v9.py::test_v9_requires_a_closed_attention_field_and_root_event_fences",
    ),
    (
        "production preflight failure leaves prior published views",
        "tests/test_attention_production.py::test_read_view_preflight_failure_precedes_internal_journal_and_wal_commit",
    ),
    (
        "published failure restarts from durable v9 without speculative view",
        "tests/test_attention_production.py::test_published_snapshot_failure_keeps_prior_view_and_restarts_from_v9_focus",
    ),
)

RESTART_NO_REPLAY_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "mixed v8/v9 WAL roots and receipts",
        "tests/test_r14_wal_recovery.py::test_mixed_v8_v9_v9_wal_reconstructs_exact_attention_roots_and_receipts",
    ),
    (
        "production v9 restart forbids Attention and cognition replay",
        "tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay",
    ),
    (
        "recovery after canonical save",
        "tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay",
    ),
    (
        "exact cognition-free local restore",
        "tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy",
    ),
    (
        "v9 AgentState restore requires and restores exact owner state",
        "tests/test_agent_state_v9.py::test_v9_alias_round_trip_uses_canonical_attention_and_requires_owner",
    ),
)

PRIVACY_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "Attention durable root excludes raw source payloads",
        "tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out",
    ),
    (
        "selected prompt row redacts structural identity fields",
        "tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful",
    ),
    (
        "request-scoped payload remains ephemeral",
        "tests/test_r14_u5_scope.py::test_u5_payload_is_ephemeral_sealed_and_has_distinct_byte_witnesses",
    ),
    (
        "U3 source text is not retained or scored",
        "tests/test_metacognition_assessment_boundaries.py::test_source_text_is_not_retained_or_scored_and_assessment_is_pure",
    ),
    (
        "production prompt selects from current source authority",
        "tests/test_attention_production.py::test_production_chat_refreshes_and_selects_from_current_sources_before_prompt",
    ),
)

SIZE_CAPACITY_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "all 4192 source identities are in the derived continuity envelope",
        "tests/test_attention_bounds.py::test_full_universe_fixture_covers_all_bounded_schema_fields",
    ),
    (
        "v9 capacity preserves full future reserve and hard-cap margin",
        "tests/test_attention_bounds.py::test_attention_state_value_and_exact_v9_reserved_projection_fit",
    ),
    (
        "wire and whole-state decoders gate capacity before construction",
        "tests/test_attention_state_codec.py::test_full_legal_candidate_universe_is_preserved_without_truncation",
    ),
    (
        "AgentState v9 exact derived capacity",
        "tests/test_agent_state_v9.py::test_first_v9_capacity_is_derived_from_the_exact_attention_field_bound",
    ),
    (
        "full 4192-source production capture",
        "tests/test_attention_main_loop.py::test_real_main_loop_projects_the_complete_4192_candidate_universe",
    ),
    (
        "whole selected prompt row omitted at Attention budget",
        "tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation",
    ),
)

COMPATIBILITY_NON_INVASION_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "R03 FIFO admission",
        "tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences",
    ),
    (
        "R04-R06 state and WAL durability",
        "tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable",
    ),
    (
        "R07 transaction participant order",
        "tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value",
    ),
    (
        "R09 Context is read-only with respect to authority",
        "tests/test_chat_context.py::test_contextual_working_memory_uses_compatibility_without_mutating_authority",
    ),
    (
        "R10 appraisal purity",
        "tests/test_appraisal.py::test_appraisal_is_pure_deterministic_and_does_not_consume_private_text",
    ),
    (
        "R11 Value projection boundary",
        "tests/test_value_system.py::test_prompt_projection_is_immutable_and_excludes_authority_metadata",
    ),
    (
        "R12 public chat privacy",
        "tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak",
    ),
    (
        "R13 Sleep candidate is not active",
        "tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active",
    ),
    (
        "retained v8 schema and capture remain legacy",
        "tests/test_agent_state_v9.py::test_v8_schema_and_capture_remain_explicitly_legacy",
    ),
    (
        "true rollback to v8 keeps R13 and bootstraps empty Attention explicitly",
        "tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_to_retained_v8_bootstraps_attention_and_keeps_r13",
    ),
)

PRODUCER_AUTHORITY_BOUNDARY_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "private cloned Attention owner and concrete persistence port",
        "tests/test_attention_main_loop.py::test_main_loop_owns_an_isolated_attention_system_and_concrete_state_port",
    ),
    (
        "serialized same-turn refresh before prompt construction",
        "tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt",
    ),
    (
        "no raw assessment authority parameters and no optional U3 call",
        "tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters",
    ),
    (
        "caller-forged current Context is rejected",
        "tests/test_r14_u5_scope.py::test_u5_producer_rejects_caller_forged_context_even_on_actual_worker",
    ),
    (
        "production source capture and selected prompt",
        "tests/test_attention_production.py::test_production_chat_refreshes_and_selects_from_current_sources_before_prompt",
    ),
    (
        "no production subject or optional Metacognition producer in source inventory",
        "tests/test_r14_evidence_matrix.py::test_source_inventory_and_unwired_assessment_boundary_are_closed",
    ),
)

SAME_TURN_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "ordinary and debug handlers share event-bound refresh-before-prompt order",
        "tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt",
    ),
    (
        "production chat refreshes from actual sources before model prompt",
        "tests/test_attention_production.py::test_production_chat_refreshes_and_selects_from_current_sources_before_prompt",
    ),
    (
        "Attention read-view preflight precedes durability",
        "tests/test_attention_production.py::test_read_view_preflight_failure_precedes_internal_journal_and_wal_commit",
    ),
    (
        "runtime event identity is carried through Attention and R12 authority",
        "tests/test_agent_runtime_ordered_time.py::test_real_chat_attention_belief_and_goal_keep_runtime_event_identity",
    ),
)

DETERMINISTIC_FOCUS_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "numeric tie order is independent of candidate input permutation",
        "tests/test_attention_policy.py::test_ties_and_input_permutations_have_canonical_total_order_and_digest",
    ),
    (
        "process owner publishes identical continuity for equivalent input order",
        "tests/test_attention_system.py::test_tied_scores_and_input_permutations_produce_identical_state_and_views",
    ),
    (
        "arousal changes global capacity but not candidate scores",
        "tests/test_attention_system.py::test_high_global_arousal_only_caps_focus_and_does_not_change_candidate_scores",
    ),
    (
        "bounded habituation eventually permits a challenger to switch focus",
        "tests/test_attention_system.py::test_salient_challenger_eventually_switches_after_repeat_focus_habituation",
    ),
)

PROMPT_OMISSION_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "oversized row is omitted whole without repacking or truncation",
        "tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation",
    ),
    (
        "empty focus emits fixed frame and no source rows",
        "tests/test_attention_prompt.py::test_empty_focus_renders_the_exact_fixed_frame_and_no_rows",
    ),
    (
        "selected-contribution byte limit is not the entire model prompt",
        "tests/test_r14_u5_scope.py::test_u5_document_limits_are_selected_contribution_not_whole_prompt",
    ),
    (
        "combined legacy R13 fallback cannot bypass selected Attention boundary",
        "tests/test_attention_prompt.py::test_attention_builder_rejects_combined_legacy_r13_fallback",
    ),
)

NO_SCHEDULER_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "AttentionSystem has a closed process-local API and no scheduler import",
        "tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local",
    ),
    (
        "U1 source modules do not import later authorities",
        "tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities",
    ),
    (
        "explicit Sleep registers a training candidate, not an active source",
        "tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active",
    ),
)

RUNTIME_TIME_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "reentrant admission cannot invert FIFO event time",
        "tests/test_agent_runtime_ordered_time.py::test_reentrant_admission_is_refused_before_it_can_invert_fifo_time",
    ),
    (
        "admission watermark keeps FIFO request times nondecreasing",
        "tests/test_agent_runtime_ordered_time.py::test_runtime_fifo_admission_has_nondecreasing_request_times",
    ),
    (
        "backward UTC sample is refused without advancing sequence",
        "tests/test_agent_runtime_ordered_time.py::test_backward_clock_refusal_preserves_admission_boundary_and_sequence",
    ),
    (
        "Attention/Belief/Goal use the actual runtime event identity",
        "tests/test_agent_runtime_ordered_time.py::test_real_chat_attention_belief_and_goal_keep_runtime_event_identity",
    ),
)

WHOLE_STAGE_INTEGRATION_MATRIX: tuple[tuple[str, str], ...] = (
    (
        "test-only Metacognition is bound to the exact live R14 turn and does not enter production prompts or durable state",
        "tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only",
    ),
    (
        "v9 restart restores exact Attention without replaying Attention or cognition",
        "tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay",
    ),
    (
        "U5 producer-only refresh on an already restored root rejects earlier runtime time without mutation",
        "tests/test_r14_integration.py::test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation",
    ),
    (
        "transient R08 challengers leave current R13 truth intact and permit focus restoration",
        "tests/test_r14_focus_regressions.py::test_transient_r08_challengers_displace_then_restore_real_r13_focus",
    ),
)

R14_CROSSCUTTING_MATRIX: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("recovery_crash", RECOVERY_CRASH_MATRIX),
    ("restart_no_replay", RESTART_NO_REPLAY_MATRIX),
    ("privacy", PRIVACY_MATRIX),
    ("size_capacity", SIZE_CAPACITY_MATRIX),
    ("compatibility_non_invasion", COMPATIBILITY_NON_INVASION_MATRIX),
    ("producer_authority_boundary", PRODUCER_AUTHORITY_BOUNDARY_MATRIX),
    ("same_turn", SAME_TURN_MATRIX),
    ("deterministic_focus", DETERMINISTIC_FOCUS_MATRIX),
    ("prompt_omission", PROMPT_OMISSION_MATRIX),
    ("no_scheduler", NO_SCHEDULER_MATRIX),
    ("runtime_time", RUNTIME_TIME_MATRIX),
    ("whole_stage_integration", WHOLE_STAGE_INTEGRATION_MATRIX),
)


def _assert_test_reference(reference: str) -> None:
    test_path, test_name = reference.split("::", maxsplit=1)
    assert test_path.startswith("tests/"), reference
    path = ROOT / test_path
    assert path.is_file(), f"missing evidence file: {test_path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    top_level_tests = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    assert test_name in top_level_tests, f"missing executable test node: {reference}"


def _document_references(cell: str) -> tuple[str, ...]:
    code_spans = INLINE_CODE_PATTERN.findall(cell)
    references = tuple(token for token in code_spans if "tests/" in token)
    assert cell.count("tests/") == len(references), (
        "test references must be fully backticked",
        cell,
    )
    for reference in references:
        assert TEST_REFERENCE_PATTERN.fullmatch(reference), (
            "malformed backticked test reference",
            reference,
        )
    return references


def _normalized_prose(text: str) -> str:
    return " ".join(text.split()).casefold()


def _table_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _document_section(document: str, heading: str, next_heading: str) -> str:
    assert heading in document, f"missing section: {heading}"
    section = document.split(heading, maxsplit=1)[1]
    assert next_heading in section, f"missing section after {heading}: {next_heading}"
    return section.split(next_heading, maxsplit=1)[0]


def _document_f_rows(
    document: str,
) -> tuple[tuple[str, str, str, str, str, tuple[str, ...]], ...]:
    rows: list[tuple[str, str, str, str, str, tuple[str, ...]]] = []
    for line in document.splitlines():
        if not line.startswith("|"):
            continue
        cells = _table_cells(line)
        if not cells or re.fullmatch(r"F\d+", cells[0]) is None:
            continue
        assert len(cells) == 6, f"malformed F row: {line}"
        rows.append(
            (
                cells[0],
                cells[1],
                cells[2],
                cells[3],
                cells[4],
                _document_references(cells[5]),
            )
        )
    return tuple(rows)


def _document_a_rows(
    document: str,
) -> tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...]:
    rows: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
    for line in document.splitlines():
        if not line.startswith("|"):
            continue
        cells = _table_cells(line)
        if not cells or re.fullmatch(r"A\d+", cells[0]) is None:
            continue
        assert len(cells) == 4, f"malformed A row: {line}"
        related_f_rows = tuple(part.strip() for part in cells[2].split(","))
        rows.append(
            (
                cells[0],
                cells[1],
                related_f_rows,
                _document_references(cells[3]),
            )
        )
    return tuple(rows)


def _document_crosscutting_rows(
    document: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    expected_keys = {key for key, _matrix in R14_CROSSCUTTING_MATRIX}
    rows: list[tuple[str, tuple[str, ...]]] = []
    for line in document.splitlines():
        if not line.startswith("|"):
            continue
        cells = _table_cells(line)
        if not cells or cells[0].strip("`") not in expected_keys:
            continue
        assert len(cells) == 3, f"malformed cross-cutting row: {line}"
        rows.append((cells[0].strip("`"), _document_references(cells[2])))
    return tuple(rows)


def test_r14_f_matrix_is_complete_classified_and_matches_document() -> None:
    expected_requirements = tuple(f"F{number}" for number in range(1, 23))
    assert tuple(row[0] for row in R14_F_MATRIX) == expected_requirements

    allowed_classifications = {"DIRECT_R14_EVIDENCE", "EXCLUDED_DEFERRED"}
    deferred_owners = {"R15", "R16", "R17, R18", "R19"}
    for requirement, classification, owner, _definition, _claim, references in (
        R14_F_MATRIX
    ):
        assert classification in allowed_classifications, requirement
        assert references, requirement
        if classification == "EXCLUDED_DEFERRED":
            assert owner in deferred_owners, (requirement, owner)
        else:
            assert owner == "—", (requirement, owner)
        for reference in references:
            _assert_test_reference(reference)

    document = EVIDENCE_DOCUMENT.read_text(encoding="utf-8")
    f_section = _document_section(
        document,
        "## F1–F22 classification and evidence",
        "## Crosscutting evidence matrices",
    )
    assert _document_f_rows(f_section) == R14_F_MATRIX
    assert "not an automatic pass" in _normalized_prose(document)
    assert "DIRECT_R14_EVIDENCE" in document
    assert "EXCLUDED_DEFERRED" in document


def test_r14_a_matrix_is_complete_classified_by_f_rows_and_matches_document() -> None:
    expected_clauses = tuple(f"A{number}" for number in range(1, 25))
    assert tuple(row[0] for row in R14_A_MATRIX) == expected_clauses
    valid_f_rows = {row[0] for row in R14_F_MATRIX}
    for clause, _summary, related_f_rows, references in R14_A_MATRIX:
        assert related_f_rows, clause
        assert all(reference in valid_f_rows for reference in related_f_rows), clause
        assert references, clause
        for reference in references:
            _assert_test_reference(reference)

    document = EVIDENCE_DOCUMENT.read_text(encoding="utf-8")
    a_section = _document_section(
        document,
        "## A1–A24 accepted-contract index",
        "## F1–F22 classification and evidence",
    )
    assert _document_a_rows(a_section) == R14_A_MATRIX
    normalized = _normalized_prose(a_section)
    assert "concise accepted-contract summaries, not verbatim quotations" in normalized
    assert "authoritative #281 issue and final acceptance" in normalized


def test_r14_crosscutting_matrices_match_document_and_resolve_by_ast() -> None:
    expected_keys = (
        "recovery_crash",
        "restart_no_replay",
        "privacy",
        "size_capacity",
        "compatibility_non_invasion",
        "producer_authority_boundary",
        "same_turn",
        "deterministic_focus",
        "prompt_omission",
        "no_scheduler",
        "runtime_time",
        "whole_stage_integration",
    )
    assert tuple(key for key, _matrix in R14_CROSSCUTTING_MATRIX) == expected_keys

    expected_document_rows: list[tuple[str, tuple[str, ...]]] = []
    for key, matrix in R14_CROSSCUTTING_MATRIX:
        assert matrix, key
        references = tuple(reference for _case, reference in matrix)
        expected_document_rows.append((key, references))
        for reference in references:
            _assert_test_reference(reference)

    document = EVIDENCE_DOCUMENT.read_text(encoding="utf-8")
    crosscutting_section = _document_section(
        document,
        "## Crosscutting evidence matrices",
        "## Bounded caveats and out-of-scope claims",
    )
    assert _document_crosscutting_rows(crosscutting_section) == tuple(
        expected_document_rows
    )
    for malformed in (
        "`tests/test_example.py:test_example`",
        "`tests/test_example.py::not_a_test_node`",
        "tests/test_example.py::test_unbackticked",
    ):
        with pytest.raises(AssertionError):
            _document_references(malformed)


def test_source_inventory_and_unwired_assessment_boundary_are_closed() -> None:
    common_path = ROOT / "suzka" / "attention" / "common.py"
    common_tree = ast.parse(common_path.read_text(encoding="utf-8"))
    target_enum = next(
        node
        for node in common_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AttentionTargetKind"
    )
    target_members = {
        node.targets[0].id: node.value.value
        for node in target_enum.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert target_members == {
        "WORKING_MEMORY": "working_memory",
        "MOTIVATION": "motivation",
        "GOAL": "goal",
        "COMMITMENT": "commitment",
    }

    adapters_path = ROOT / "suzka" / "attention" / "adapters.py"
    adapters_tree = ast.parse(adapters_path.read_text(encoding="utf-8"))
    candidate_projectors = {
        node.name
        for node in adapters_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("project_")
    }
    assert candidate_projectors == {
        "project_working_memory",
        "project_motivation",
        "project_goal",
        "project_commitment",
        "project_attention_candidates",
        "project_global_emotion",
    }

    main_loop_path = ROOT / "suzka" / "runtime" / "main_loop.py"
    main_loop_tree = ast.parse(main_loop_path.read_text(encoding="utf-8"))
    refresh = next(
        node
        for node in ast.walk(main_loop_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_refresh_attention_prompt"
    )
    called_names = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else ""
        for node in ast.walk(refresh)
        if isinstance(node, ast.Call)
    }
    assert not {"observe_metacognition", "assess_metacognition"} & called_names

    document = _normalized_prose(EVIDENCE_DOCUMENT.read_text(encoding="utf-8"))
    assert "subject-source producer: unavailable" in document
    assert "optional u3 metacognition production: unavailable" in document
    assert "not a claim of feature delivery" in document


def test_documented_capacity_is_derived_and_scope_limits_are_explicit() -> None:
    from suzka.attention.bounds import (
        ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES,
        ATTENTION_AGENT_STATE_MAX_BYTES,
        ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES,
        ATTENTION_ROOT_FIELD_OVERHEAD_BYTES,
        ATTENTION_STATE_MAX_VALUE_BYTES,
        attention_candidate_capacity,
        derive_attention_schema_size_budget,
    )
    from suzka.attention.common import (
        ATTENTION_HIGH_AROUSAL_MAX_FOCUS,
        ATTENTION_HIGH_AROUSAL_THRESHOLD,
        ATTENTION_MAX_FOCUS,
        ATTENTION_PROMPT_BUDGET_BYTES,
    )

    document = _normalized_prose(EVIDENCE_DOCUMENT.read_text(encoding="utf-8"))
    budget = derive_attention_schema_size_budget()
    v9_with_reserve = (
        ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES
        + ATTENTION_ROOT_FIELD_OVERHEAD_BYTES
        + budget.attention_state_max_bytes
        + ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES
    )
    assert attention_candidate_capacity() == 4_192
    assert budget.attention_state_max_bytes == 4_193_496
    assert ATTENTION_STATE_MAX_VALUE_BYTES == 7_866_805
    assert ATTENTION_AGENT_STATE_V8_BASE_MAX_BYTES == 109_573_688
    assert (
        ATTENTION_STATE_MAX_VALUE_BYTES + ATTENTION_ROOT_FIELD_OVERHEAD_BYTES
        == 7_866_824
    )
    assert v9_with_reserve == 130_544_419
    assert ATTENTION_AGENT_STATE_MAX_BYTES == 134_217_728
    assert ATTENTION_AGENT_STATE_MAX_BYTES - v9_with_reserve == 3_673_309
    assert ATTENTION_AGENT_STATE_FUTURE_RESERVE_BYTES == 16_777_216
    assert ATTENTION_PROMPT_BUDGET_BYTES == 131_072
    assert (ATTENTION_MAX_FOCUS, ATTENTION_HIGH_AROUSAL_MAX_FOCUS) == (16, 8)
    assert ATTENTION_HIGH_AROUSAL_THRESHOLD == 0.75

    for expected in (
        "4,192 (4,096 Working Memory + 32 each Motivation, Goal, Commitment)",
        "16 / 8 at global arousal >= 0.75",
        "131,072 bytes",
        "4,193,496 bytes",
        "109,573,688 bytes",
        "7,866,824 bytes",
        "130,544,419 bytes",
        "16,777,216-byte reserve",
        "134,217,728-byte hard cap",
        "3,673,309-byte margin beyond reserve",
        "The 131,072-byte limit is the Attention-selected contribution, not the entire model prompt.",
        "source-local revisions",
        "U3 never emits `SUFFICIENT`",
        "unknown, contradiction, and overload remain distinct",
    ):
        assert _normalized_prose(expected) in document, expected


def test_lineage_responsibility_and_parent_owned_review_state_are_documented() -> None:
    document = _normalized_prose(EVIDENCE_DOCUMENT.read_text(encoding="utf-8"))
    for expected in (
        "Human U5 PASS/COMPLETE",
        "8c63d21501d09fe843694f6cc8cd5cfe40087892",
        "6051359998",
        "U6 is authorized only for whole-stage integration, adversarial/regression",
        "testing, bounded evidence, and final review",
        "no redesign is authorized",
        "Draft PR #282",
        "Human U6/whole-R14 review is pending",
        "U6 -> whole R14 -> Human exact-head review -> HumanReadyMerge",
        "## Parent-owned whole-stage execution addendum",
        "NOT RUN by this evidence author",
    ):
        assert _normalized_prose(expected) in document, expected
