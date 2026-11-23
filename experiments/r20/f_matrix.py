"""The F01-F15 matrix of R20 (IP-22c; D1 ``d-ip22`` sec. 6 / sec. 11) as DATA,
its checker and its runner: ``python -m experiments.r20.f_matrix run``.

Reading of a row:

* ``merged``: tests that already run the case on a real app or a real store
  (cited by FUNCTION NAME, ``tests/<file>::<function>``, so a rename fails the
  meta-test).  ``new``: tests IP-22c adds, all T1 (in-process).  ``smoke``:
  T2 (real process / SIGKILL) tests; they are smoke evidence and are NEVER
  counted as covered (the kill window is not controlled).
* ``determinism``: ``DET_T1`` = deterministic inputs and a T1 decision;
  ``DET_T1_PLUS_T2_SMOKE_KILL_WINDOW_NONDET`` = the same plus a T2 smoke whose
  kill window is not deterministic; ``DET_T1_MERGED_ONLY`` = merged T1 tests only.
* ``status``: ``COVERED`` (>= 1 T1 test, nothing pending), ``TRANSITIONAL``
  (covered only as a guard that must not be read as final behaviour) or
  ``PENDING`` (the acceptance is being redesigned; nothing is claimed).

F14 (HD-22-5) is a TRANSITIONAL guard (tracked in issue #463): the Human's final
design lets a subject revise its interpretation, human-like; today's refusal of
a second R15 interpretation (``narrative_self_system.py`` ~509-537) exists only
because no revision stage exists yet, and is NOT pinned as final behaviour.

F15 is a PENDING row: its acceptance was rejected and is being redesigned
(degraded-continuation slice, #411 comment 7103463226).  No F15 / F15-x test
exists and no F15 wording is fixed here.

Test-side only; nothing under ``suzka/`` imports this module.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from experiments.r20 import guards
from experiments.r20.manifest import Manifest

TESTS = "tests"
COVERED = "COVERED"
TRANSITIONAL = "TRANSITIONAL"
PENDING = "PENDING"
STATUSES = (COVERED, TRANSITIONAL, PENDING)
DETERMINISM = (
    "DET_T1",
    "DET_T1_PLUS_T2_SMOKE_KILL_WINDOW_NONDET",
    "DET_T1_MERGED_ONLY",
)
F15_REDESIGN = "#411 comment 7103463226"
F14_ISSUE = "#463"
TRANSITIONAL_MARK = "TRANSITIONAL"

# result labels of the per-F manifest entries
PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
SMOKE_NOT_COUNTED = "SMOKE_NOT_COUNTED"


@dataclass(frozen=True, slots=True)
class TestRef:
    """One test cited by file and FUNCTION name (never by line)."""

    __test__ = False  # not a pytest class

    file: str
    name: str

    @property
    def node_id(self) -> str:
        return f"{TESTS}/{self.file}::{self.name}"


def _t(file: str, name: str) -> TestRef:
    return TestRef(file, name)


@dataclass(frozen=True, slots=True)
class FCase:
    case: str
    title: str
    owning_unit: str
    determinism: str
    status: str
    merged: tuple[TestRef, ...]
    new: tuple[TestRef, ...]
    smoke: tuple[TestRef, ...]
    # F14 only: what is transitional and where the final design is tracked
    transitional: str | None = None
    # F15 only: why nothing is claimed
    pending_reason: str | None = None
    # what a reader must not over-read
    caveat: str | None = None

    @property
    def counted(self) -> tuple[TestRef, ...]:
        """The T1 tests that count as coverage; T2 smoke never does."""

        return (*self.merged, *self.new)

    @property
    def all_ids(self) -> tuple[TestRef, ...]:
        return (*self.counted, *self.smoke)


_RAW: tuple[FCase, ...] = (
    FCase(
        case='F01',
        title='source authority injected by caller / model',
        owning_unit='R16 U2a / U2b / U2c owners, IP-6b driver, R15 P1b / IP-8a',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r16_u2a_source_owner.py', 'test_internal_r16_event_without_qualified_owner_fails_before_wal'),
            _t('test_r16_u2a_source_owner.py', 'test_no_http_route_schema_field_or_public_u1_export'),
            _t('test_r16_u2a_source_owner.py', 'test_selection_shape_rejects_forged_content_before_admission'),
            _t('test_r16_u2a_source_owner.py', 'test_worker_pair_and_private_nonce_are_both_required'),
            _t('test_r16_u2b_goal_owner.py', 'test_unbound_owner_or_forged_selection_does_not_mutate'),
            _t('test_r16_ip6b_driver.py', 'test_commands_are_recognized_by_exact_fullmatch_only'),
            _t('test_r15_p1b_positive.py', 'test_direct_claim_shapes_are_not_authoritative'),
            _t('test_r15_p1b_positive.py', 'test_missing_source_and_forged_refs_refuse_without_mutation'),
            _t('test_r15_p1b_positive.py', 'test_process_local_u3_refuses_external_pseudo_admission'),
        ),
        new=(
            _t('test_r20_ip22c_f01_f02.py', 'test_f01_m_a_model_answer_or_a_caller_cannot_create_goal_commitment_or_value'),
        ),
        smoke=(),
    ),
    FCase(
        case='F02',
        title='wrong / suspended Context, cross-context token',
        owning_unit='IP-5 / IP-9 owners, IP-16a-2 issuer, IP-21c',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r16_u2a_source_owner.py', 'test_same_token_in_different_contexts_never_authenticates_actor'),
            _t('test_r16_u3a_plan_preview.py', 'test_real_context_transition_during_preview_fails_closed_before_wal'),
            _t('test_r16_ip6b_driver.py', 'test_unobserved_or_other_context_targets_are_indistinguishable'),
            _t('test_r18_ip16a2_intent_issuer.py', 'test_a_suspended_and_resumed_context_is_unavailable'),
            _t('test_r19_ip21c_inbox_api.py', 'test_a_closed_context_is_refused_by_the_handler_and_a_missing_registry_skips_it'),
            _t('test_r19_ip21c_inbox_api.py', 'test_get_is_200_and_empty_for_every_unreadable_context'),
            _t('test_r19_ip21c_inbox_api.py', 'test_wrong_context_and_unknown_message_are_the_same_404'),
            _t('test_r16_ip14b1_history_v4.py', 'test_trigger_matrix_r1_claims_closure_outbox_and_ack'),
            _t('test_r16_ip14b1_history_v4.py', 'test_wake_and_ack_keys_are_reserved_at_prepare'),
        ),
        new=(
            _t('test_r20_ip22c_f01_f02.py', 'test_f02_m_a_command_from_another_or_a_suspended_context_never_acts'),
            _t('test_r20_ip22c_f01_f02.py', 'test_f02_m_an_ack_from_the_wrong_or_a_suspended_context_writes_nothing'),
        ),
        smoke=(),
    ),
    FCase(
        case='F03',
        title='invalid / missing / stale Goal, digest, Decision proof',
        owning_unit='IP-10 issuer, IP-16a-2, IP-17a',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r16_ip10_decision_issuer.py', 'test_head_without_finalized_proof_is_a_conflict'),
            _t('test_r16_ip10_decision_issuer.py', 'test_source_drift_without_new_witness_holds'),
            _t('test_r16_ip10_decision_issuer.py', 'test_unqualified_goal_still_yields_a_policy_row'),
            _t('test_r16_ip10_decision_issuer.py', 'test_unverifiable_head_is_a_conflicting_predecessor'),
            _t('test_r18_ip16a2_intent_issuer.py', 'test_a_real_value_revision_move_is_source_drift'),
            _t('test_r18_ip16a2_intent_issuer.py', 'test_an_unverifiable_head_is_a_head_conflict'),
            _t('test_r18_ip16a2_intent_issuer.py', 'test_drifted_sources_hold'),
            _t('test_r18_ip17a_policy.py', 'test_a_superseded_decision_is_denied_even_with_the_clock_set_back'),
            _t('test_r18_ip17a_policy.py', 'test_source_drift_is_deny_then_row_2c_then_a_new_request_is_2c'),
        ),
        new=(),
        smoke=(),
    ),
    FCase(
        case='F04',
        title='forecast only after the effect',
        owning_unit='IP-12 / IP-16a-3 / IP-19',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r17_ip12_contracts.py', 'test_forecast_for_another_intent_or_after_the_claim_is_invalid'),
            _t('test_r17_ip19_feedback_owner.py', 'test_a_forecast_after_the_claim_is_not_assessed'),
            _t('test_r17_ip19_feedback_owner.py', 'test_a_missing_forecast_writes_no_assessment_or_feedback'),
            _t('test_r17_ip16a3_expectation_issuer.py', 'test_terminal_intent_gets_exactly_one_bound_forecast'),
        ),
        new=(),
        smoke=(),
    ),
    FCase(
        case='F05',
        title='user / model Tool, missing / reused / stale approval',
        owning_unit='IP-16a, IP-17a / b / c',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r18_ip17c_claim.py', 'test_n3_a_second_confirm_after_a_claim_is_refused_and_one_claim_exists'),
            _t('test_r18_ip17c_claim.py', 'test_n4_ttl_and_window_expiry_refuse_the_claim'),
            _t('test_r18_ip17c_claim.py', 'test_n5_each_bound_field_mutated_holds_the_claim'),
            _t('test_r18_ip17c_claim.py', 'test_n8_a_closed_admission_gate_holds_the_claim'),
            _t('test_r18_ip17c_api.py', 'test_both_transports_are_accepted_and_the_flow_claims_once'),
            _t('test_r18_ip17c_api.py', 'test_n1_only_the_approval_api_calls_the_challenge_and_claim_submit_methods'),
            _t('test_r18_ip17c_api.py', 'test_n2_every_refusal_writes_nothing_and_submits_no_event'),
            _t('test_r18_ip18a_flow.py', 'test_f05_nothing_is_enqueued_for_a_held_claim'),
        ),
        new=(
            _t('test_r20_ip22c_f05.py', 'test_f05_w_an_unapproved_intent_closes_as_row_2c_with_no_claim_and_the_subject_goes_on'),
            _t('test_r20_ip22c_f05.py', 'test_f05_w_a_replayed_nonce_over_http_is_refused_one_claim_one_state'),
            _t('test_r20_ip22c_f05.py', 'test_f05_w_the_stand_in_is_a_guarded_client_with_and_without_fetch_metadata'),
        ),
        smoke=(),
    ),
    FCase(
        case='F06',
        title='crash before / after PREPARED, WAL, finalize',
        owning_unit='IP-9 (crash matrix), IP-16a-2 / 3, IP-17b / c, IP-18a, IP-19, IP-20d, IP-21',
        determinism='DET_T1_PLUS_T2_SMOKE_KILL_WINDOW_NONDET',
        status='COVERED',
        merged=(
            _t('test_r16_ip9_crash_matrix.py', 'test_crash_after_ambiguous_state_stays_blocked'),
            _t('test_r16_ip9_crash_matrix.py', 'test_crash_after_participant_finalized_before_transaction_completed'),
            _t('test_r16_ip9_crash_matrix.py', 'test_crash_after_r07_prepared_before_sqlite_prepare'),
            _t('test_r16_ip9_crash_matrix.py', 'test_crash_inside_the_sqlite_finalize_rolls_back_then_rolls_forward'),
            _t('test_r16_ip9_crash_matrix.py', 'test_interrupted_create_is_completed_on_open'),
            _t('test_r16_ip9_crash_matrix.py', 'test_tombstone_and_prepare_race_has_one_winner'),
            _t('test_r16_ip9_crash_matrix.py', 'test_true_rollback_starts_a_new_incarnation'),
            _t('test_r18_ip17c_claim.py', 'test_n6_after_a_claim_a_restart_is_still_claimed_with_no_second_claim'),
            _t('test_r18_ip18a_flow.py', 'test_p5_a_crash_inside_the_outcome_op_tombstones_then_exactly_one_outcome_exists'),
            _t('test_r17_ip19_feedback_owner.py', 'test_crash_after_internal_commit_rolls_forward_then_existing'),
            _t('test_r17_ip19_feedback_owner.py', 'test_crash_after_prepare_is_tombstoned_then_recommitted'),
            _t('test_r19_ip20d_scheduler.py', 'test_a_crash_before_the_wal_aborts_and_the_next_tick_converges'),
            _t('test_startup_reconciliation.py', 'test_pre_internal_aborts_and_records_terminal_evidence'),
        ),
        new=(),
        smoke=(
            _t('test_r20_ip22c_f06c.py', 'test_f06_c_a_real_sigkill_during_the_r16_chain_finishes_with_one_decision_per_stage'),
        ),
    ),
    FCase(
        case='F07',
        title='crash after claim, before receipt / Outcome',
        owning_unit='IP-15 ledger, IP-18a',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r18_ip18a_flow.py', 'test_a_flapping_gate_holds_the_claim_consumes_nothing_and_writes_once'),
            _t('test_r18_ip18a_flow.py', 'test_p1_claim_adopted_executor_never_ran_the_sweep_settles_absent_no_write'),
            _t('test_r18_ip18a_flow.py', 'test_p3_p4_effect_committed_but_the_outcome_never_ran_the_sweep_reads_it_back'),
            _t('test_r18_ip18a_executor.py', 'test_a_claim_of_an_earlier_process_is_never_written_and_settles_absent'),
            _t('test_r18_ip18a_executor.py', 'test_an_unknown_failure_is_never_resent_and_the_store_is_reopened'),
            _t('test_r18_ip18a_executor.py', 'test_when_in_doubt_do_not_write'),
            _t('test_r18_ip15_target_verifier.py', 'test_double_execution_is_suppressed_across_reopen'),
            _t('test_r18_ip15_target_verifier.py', 'test_fence_makes_a_late_cas_fail_and_verifies_absent'),
        ),
        new=(),
        smoke=(),
    ),
    FCase(
        case='F08',
        title='receipt says success, independent read disagrees',
        owning_unit='IP-15 verifier, IP-18a, IP-19',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r18_ip15_target_verifier.py', 'test_receipt_says_applied_but_the_effect_is_gone'),
            _t('test_r18_ip15_target_verifier.py', 'test_the_written_bytes_must_be_the_approved_bytes'),
            _t('test_r18_ip18a_executor.py', 'test_a_lying_executor_cannot_make_an_effect_appear'),
            _t('test_r18_ip18a_flow.py', 'test_a_target_changed_before_the_verification_is_an_unauthorized_effect'),
            _t('test_r17_ip19_feedback_owner.py', 'test_an_unauthorized_or_absent_outcome_gets_an_assessment_and_never_feedback'),
        ),
        new=(),
        smoke=(),
    ),
    FCase(
        case='F09',
        title='external change, unexpected result',
        owning_unit='**IP-15 and IP-19**',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r18_ip15_target_verifier.py', 'test_an_outside_change_is_foreign_and_unauthorized'),
            _t('test_r17_ip19_feedback_owner.py', 'test_f09_a_foreign_change_is_environment_and_never_row_5'),
            _t('test_r18_ip18a_flow.py', 'test_a_target_changed_before_the_verification_is_an_unauthorized_effect'),
            _t('test_r17_ip19_feedback_read.py', 'test_a_newer_outcome_revision_never_reuses_the_older_feedback'),
        ),
        new=(),
        smoke=(),
    ),
    FCase(
        case='F10',
        title='R19 restart, duplicate Wake, backward clock, budget',
        owning_unit='IP-20a-d',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r19_ip20d_scheduler.py', 'test_a_duplicate_scan_and_a_second_wake_for_one_fingerprint_are_nothing'),
            _t('test_r19_ip20d_scheduler.py', 'test_a_forward_clock_jump_retires_the_wake_once_and_never_re_wakes'),
            _t('test_r19_ip20d_scheduler.py', 'test_a_no_row_hold_storm_is_bounded_over_seven_days'),
            _t('test_r19_ip20d_scheduler.py', 'test_a_review_below_its_claims_floor_is_a_recorded_refusal_that_counts'),
            _t('test_r19_ip20d_scheduler.py', 'test_a_rolled_back_clock_waits_without_a_claim_or_a_terminal_or_a_double_write'),
            _t('test_r19_ip20d_scheduler.py', 'test_budget_exhaustion_ends_after_three_leases_and_never_calls_a_model'),
            _t('test_r19_ip20d_scheduler.py', 'test_claim_after_restart_a_live_lease_without_a_started_review_gets_one'),
            _t('test_r19_ip20d_scheduler.py', 'test_existing_and_drift_close_once_and_are_never_failed_or_claimed_again'),
            _t('test_r19_ip20b_materializer.py', 'test_the_same_fingerprint_never_makes_a_second_wake_even_after_the_cycle_ended'),
        ),
        new=(),
        smoke=(),
    ),
    FCase(
        case='F11',
        title='duplicate / expired / acked message, replayed ACK, wrong Context',
        owning_unit='IP-14b-1, IP-21b / c',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r19_ip21b_emitter.py', 'test_dedupe_comes_first_even_for_a_message_that_has_since_aged_out'),
            _t('test_r19_ip21b_emitter.py', 'test_one_message_per_closed_cycle_a_second_emission_is_existing'),
            _t('test_r19_ip21c_inbox_api.py', 'test_ack_of_a_pending_message_writes_one_row_and_one_event'),
            _t('test_r19_ip21c_inbox_api.py', 'test_an_emitted_decision_message_is_read_and_acked'),
            _t('test_r16_ip14b1_history_v4.py', 'test_trigger_matrix_r1_claims_closure_outbox_and_ack'),
            _t('test_r16_ip14b1_history_v4.py', 'test_wake_and_ack_keys_are_reserved_at_prepare'),
        ),
        new=(),
        smoke=(),
    ),
    FCase(
        case='F12',
        title='true rollback with surviving Decision / Outbox / Action',
        owning_unit='**IP-13 and IP-18** (+ IP-9 incarnation)',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r16_ip13_successor_policy_v3.py', 'test_prior_incarnation_effect_is_a_row_2c_plan_a'),
            _t('test_r18_ip18b_facts_sweep.py', 'test_f12_a_prior_incarnation_claim_is_fenced_by_the_restart_sweep_and_the_subject_re_decides_plan_a'),
            _t('test_r16_ip9_history_adoption.py', 'test_incarnation_changes_only_after_an_adopted_true_rollback'),
            _t('test_r16_ip9_history_reader.py', 'test_store_rollback_under_a_terminal_transaction_diverges'),
            _t('test_r16_ip9_crash_matrix.py', 'test_true_rollback_starts_a_new_incarnation'),
            _t('test_r16_ip9_c10_history_attestation.py', 'test_compaction_after_a_recent_true_rollback'),
            _t('test_startup_reconciliation.py', 'test_true_rollback_over_an_archived_range_fails_closed'),
            _t('test_startup_reconciliation.py', 'test_true_rollback_reconciles_aggregate_and_clears_gate'),
        ),
        new=(
            _t('test_r20_ip22c_f12.py', 'test_f12_r_a_true_rollback_on_a_realistic_s0_root_leaves_no_actionable_old_branch'),
        ),
        smoke=(),
    ),
    FCase(
        case='F13',
        title='public API / Cockpit / Outbox disclose private roots, hidden thought, credentials',
        owning_unit='IP-17c, IP-18a, IP-21c, R15 private projection',
        determinism='DET_T1',
        status='COVERED',
        merged=(
            _t('test_r19_ip21c_inbox_api.py', 'test_a_sentinel_secret_reaches_no_message_response_or_log'),
            _t('test_r19_ip21c_inbox_api.py', 'test_every_response_body_is_closed'),
            _t('test_r18_ip17c_api.py', 'test_n1_model_and_chat_paths_cannot_reach_the_approval_api'),
            _t('test_r18_ip17c_api.py', 'test_the_page_script_has_no_html_sink_and_computes_what_the_server_does'),
            _t('test_r18_ip18a_executor.py', 'test_the_log_line_field_set_is_pinned'),
            _t('test_r18_ip18a_executor.py', 'test_the_log_lines_carry_no_request_text'),
        ),
        new=(
            _t('test_r20_ip22c_f13.py', 'test_f13_c_no_canary_reaches_any_response_the_inbox_the_page_or_the_log'),
        ),
        smoke=(),
    ),
    FCase(
        case='F14',
        title='post-restart new interpretation differs from identical retained source',
        owning_unit='R15 (IP-8a, P1bb restore); **no new product event**',
        determinism='DET_T1',
        status='TRANSITIONAL',
        merged=(
            _t('test_r15_p1bb_wal.py', 'test_authoritative_r15_revision_persists_in_real_wal_and_restarts'),
            _t('test_r15_p1b_positive.py', 'test_a_new_owner_cannot_pretend_to_restore_unpersisted_u3'),
            _t('test_r15_p1b_positive.py', 'test_missing_committed_e2_fails_before_owner_revision'),
            _t('test_r15_p1b_positive.py', 'test_repeated_s1_without_e2_does_not_create_second_revision'),
            _t('test_r15_u3_narrative_self_system.py', 'test_r15_u3_conflicting_events_reentrancy_and_receipt_overflow_fail_closed'),
            _t('test_r15_u3_narrative_self_system.py', 'test_r15_u3_documented_design_hold_and_exact_reserve_match_u1'),
        ),
        new=(
            _t('test_r20_ip22c_f14.py', 'test_f14_committed_revisions_are_never_rewritten_and_a_changed_interpretation_is_a_new_event'),
        ),
        smoke=(),
    ),
    FCase(
        case='F15',
        title='partial / missing / corrupt DB / Journal / WAL / source',
        owning_unit='R06 / R07, IP-9 adoption, IP-15, IP-18a',
        determinism='DET_T1_MERGED_ONLY',
        status='PENDING',
        merged=(
            _t('test_state_wal.py', 'test_boot_anchor_and_corrupt_tail_prefix'),
            _t('test_state_wal.py', 'test_corrupt_existing_wal_cannot_be_replaced_by_ordinary_generation'),
            _t('test_state_wal.py', 'test_tampered_embedded_snapshot_is_rejected'),
            _t('test_state_wal.py', 'test_tampered_manifest_is_rejected'),
            _t('test_state_recovery.py', 'test_corrupt_active_wal_rebaselines_valid_canonical_current_without_gate'),
            _t('test_state_recovery.py', 'test_corrupt_before_journal_bound_record_fails_closed_when_anchor_is_invalid'),
            _t('test_state_recovery.py', 'test_corrupt_journal_bound_record_falls_back_to_older_boot_anchor'),
            _t('test_startup_reconciliation.py', 'test_ambiguous_commit_proof_fails_closed'),
            _t('test_startup_reconciliation.py', 'test_participant_missing_or_conflicting_evidence_degrades'),
            _t('test_r16_ip9_history_adoption.py', 'test_disabling_after_adoption_keeps_the_registry_and_fails_closed'),
            _t('test_r16_ip9_history_adoption.py', 'test_lost_or_foreign_store_after_adoption_fails_closed'),
            _t('test_r16_ip9_history_adoption.py', 'test_store_adopted_for_another_baseline_fails_closed'),
            _t('test_r16_ip9_subject_history.py', 'test_tampered_schema_or_pragmas_fail_closed'),
            _t('test_r18_ip15_target_verifier.py', 'test_a_dropped_guard_is_corruption'),
            _t('test_r18_ip15_target_verifier.py', 'test_custody_and_schema_are_checked'),
            _t('test_r18_ip18a_executor.py', 'test_a_missing_target_blocks_after_60_seconds_and_the_probe_reopens_by_itself'),
        ),
        new=(),
        smoke=(),
    ),
)

_ANNOTATIONS: dict[str, dict[str, str]] = {
    "F06": {
        "caveat": (
            "F06-c is a T2 smoke at the R16 level and is not counted; K2-K6 on the R18 / R19 "
            "chain need a T2 root on which the R18 executor starts (it does not today: see the "
            "F06-c test docstring), a finding for IP-22d.  The exact windows are the merged T1 tests."
        )
    },
    "F12": {
        "caveat": (
            "F12-r ran on an S0-CONFIG root (R18 + R19 on, driver activated, scripted provider) "
            "with the R15 admission NOT performed; the R15 part and the SIGKILL variant belong to "
            "IP-22d.  The feasibility spike passed: startup was not refused."
        )
    },
    "F14": {
        "transitional": (
            f"{TRANSITIONAL_MARK}: today's refusal of a second R15 interpretation "
            "(narrative_self_system.py ~509-537) exists only because no revision stage exists yet; "
            f"the final design lets a subject revise its interpretation (issue {F14_ISSUE}).  "
            "Only the invariants that stay true are asserted: committed S1 / S2 revisions are never "
            "silently rewritten and restore byte-identically, a replay of the S2 admission with the "
            "same sources is changed = False, a changed interpretation is a new attributed event."
        )
    },
    "F15": {
        "pending_reason": (
            f"The F15 acceptance was rejected and is being redesigned ({F15_REDESIGN}: "
            "degraded-continuation slice).  No F15 / F15-x test exists and no F15 wording is fixed "
            "here; the merged tests cited are listed for reference only and are not claimed as "
            "F15 coverage."
        )
    },
}
CASES: tuple[FCase, ...] = tuple(
    FCase(**{**{field: getattr(case, field) for field in FCase.__dataclass_fields__}, **_ANNOTATIONS.get(case.case, {})})
    for case in _RAW
)
EXPECTED_CASES = tuple(f"F{number:02d}" for number in range(1, 16))


def case_for(case_id: str) -> FCase:
    (found,) = [case for case in CASES if case.case == case_id]
    return found


# ------------------------------------------------------------------ checking


def _test_functions(path: Path) -> dict[str, ast.AST]:
    """Every ``def`` (module level or in a class) whose name starts with ``test``."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found[node.name] = node
    return found


def _is_disabled(node: ast.AST) -> bool:
    """A skip / xfail decorator makes a "covered" claim vacuous."""

    for decorator in getattr(node, "decorator_list", []):
        text = ast.unparse(decorator)
        if "skip" in text or "xfail" in text:
            return True
    return False


def uses_process_tier(node: ast.AST) -> bool:
    text = ast.unparse(node)
    return "Tier.T2" in text or "ChildProcess" in text


def check_table(root: Path | None = None) -> list[str]:
    """Every problem of the table; an empty list is a sound matrix."""

    base = root or guards.repo_root()
    problems: list[str] = []
    if tuple(case.case for case in CASES) != EXPECTED_CASES:
        problems.append("the table is not exactly F01..F15 in order")
    cache: dict[str, dict[str, ast.AST]] = {}

    def functions(file: str) -> dict[str, ast.AST] | None:
        path = base / TESTS / file
        if not path.is_file():
            return None
        if file not in cache:
            cache[file] = _test_functions(path)
        return cache[file]

    for case in CASES:
        label = case.case
        if not case.owning_unit.strip():
            problems.append(f"{label}: no owning unit")
        if case.determinism not in DETERMINISM:
            problems.append(f"{label}: determinism flag {case.determinism!r} is not one of {DETERMINISM}")
        if case.status not in STATUSES:
            problems.append(f"{label}: status {case.status!r}")
        if not case.all_ids:
            problems.append(f"{label}: no test id")
        seen: set[str] = set()
        for ref in case.all_ids:
            if ref.node_id in seen:
                problems.append(f"{label}: {ref.node_id} cited twice")
            seen.add(ref.node_id)
            found = functions(ref.file)
            if found is None:
                problems.append(f"{label}: {ref.file} does not exist")
            elif ref.name not in found:
                problems.append(f"{label}: {ref.node_id} does not exist")
            elif _is_disabled(found[ref.name]):
                problems.append(f"{label}: {ref.node_id} is skipped or xfail (vacuous)")
        # only T1 results count: no counted test may use the process tier, and a smoke test must
        for ref in case.counted:
            found = functions(ref.file) or {}
            if ref.name in found and uses_process_tier(found[ref.name]):
                problems.append(f"{label}: counted test {ref.node_id} uses the process tier (T2)")
        for ref in case.smoke:
            found = functions(ref.file) or {}
            if ref.name in found and not uses_process_tier(found[ref.name]):
                problems.append(f"{label}: smoke test {ref.node_id} is not a T2 test")
        if case.status == COVERED:
            if not case.counted:
                problems.append(f"{label}: COVERED without a counted T1 test")
            if case.transitional or case.pending_reason:
                problems.append(f"{label}: COVERED but annotated transitional / pending")
        if case.status == TRANSITIONAL and (not case.transitional or TRANSITIONAL_MARK not in case.transitional):
            problems.append(f"{label}: TRANSITIONAL without the TRANSITIONAL label")
        if case.status == PENDING:
            if not case.pending_reason:
                problems.append(f"{label}: PENDING without a reason")
            if case.new or case.smoke:
                problems.append(f"{label}: PENDING row must not carry new / smoke tests")
        if case.determinism == "DET_T1_PLUS_T2_SMOKE_KILL_WINDOW_NONDET" and not case.smoke:
            problems.append(f"{label}: a T2 determinism flag without a smoke test")
    return problems


def covered_cases() -> tuple[str, ...]:
    """Only T1 results count; TRANSITIONAL and PENDING rows are not 'covered'."""

    return tuple(case.case for case in CASES if case.status == COVERED and case.counted)


# -------------------------------------------------------------------- runner

Runner = Callable[[Sequence[str]], Mapping[str, str]]


def pytest_runner(node_ids: Sequence[str]) -> dict[str, str]:
    """Run the node ids through ``pytest.main`` and return node id -> outcome
    (``passed`` / ``failed`` / ``skipped`` / ``error``)."""

    import pytest

    outcomes: dict[str, str] = {}

    class Collector:
        def pytest_runtest_logreport(self, report: Any) -> None:
            if report.when == "call":
                outcomes[report.nodeid] = report.outcome
            elif report.outcome != "passed":  # setup / teardown trouble
                outcomes[report.nodeid] = "skipped" if report.outcome == "skipped" else "error"

    root = guards.repo_root()
    previous = Path.cwd()
    try:
        os.chdir(root)
        pytest.main(["-q", "-p", "no:cacheprovider", *node_ids], plugins=[Collector()])
    finally:
        os.chdir(previous)
    return outcomes


def aggregate_outcome(outcomes: Mapping[str, str], node_id: str) -> str:
    """The outcome of one cited test.  A parametrized test reports one node id
    per parameter (``id[...]``): all of them must have passed; any other state
    wins; nothing reported is ``NOT_RUN``."""

    found = [state for key, state in outcomes.items() if key == node_id or key.startswith(node_id + "[")]
    if not found:
        return NOT_RUN
    if all(state == "passed" for state in found):
        return "passed"
    return next(state for state in found if state != "passed")


def run_matrix(
    runner: Runner,
    evidence_dir: Path,
    *,
    only: Sequence[str] | None = None,
    include_smoke: bool = False,
    run_id: str = "fmatrix",
) -> tuple[Manifest, dict[str, str]]:
    """Run the counted tests of every non-pending row (and the T2 smoke on
    request), and write one result label per F to the manifest.

    Returns the manifest and ``{F: label}``.  A PENDING row is never run and
    never PASS; a TRANSITIONAL row passes as ``PASS_TRANSITIONAL``; a smoke
    result is labelled ``SMOKE_NOT_COUNTED`` and never raises a row to PASS.
    """

    selected = [case for case in CASES if only is None or case.case in only]
    wanted: list[str] = []
    for case in selected:
        if case.status == PENDING:
            continue
        wanted += [ref.node_id for ref in case.counted]
        if include_smoke:
            wanted += [ref.node_id for ref in case.smoke]
    outcomes = dict(runner(list(dict.fromkeys(wanted)))) if wanted else {}

    def outcome(ref: TestRef) -> str:
        return aggregate_outcome(outcomes, ref.node_id)

    manifest = Manifest(evidence_dir, run_id)
    manifest.set("profile", "F-MATRIX")
    manifest.set("tier", "T1")
    labels: dict[str, str] = {}
    for case in selected:
        if case.status == PENDING:
            label = "PENDING"
        else:
            states = [outcome(ref) for ref in case.counted]
            if any(state == NOT_RUN for state in states):
                label = NOT_RUN
            elif all(state == "passed" for state in states):
                label = PASS if case.status == COVERED else "PASS_TRANSITIONAL"
            else:
                label = FAIL
        labels[case.case] = label
        manifest.result(case.case, label)
        if include_smoke and case.smoke:
            smoke_states = [outcome(ref) for ref in case.smoke]
            manifest.result(
                f"{case.case}.smoke",
                SMOKE_NOT_COUNTED if all(s == "passed" for s in smoke_states) else f"{SMOKE_NOT_COUNTED}:{FAIL}",
            )
        if case.status == TRANSITIONAL:
            manifest.result(f"{case.case}.second_interpretation", TRANSITIONAL_MARK)
    problems = check_table()
    manifest.result("table", PASS if not problems else FAIL)
    return manifest, labels


def exit_code(labels: Mapping[str, str], *, strict: bool) -> int:
    if any(label in {FAIL, NOT_RUN} for label in labels.values()):
        return 1
    if strict and any(label == "PENDING" for label in labels.values()):
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.r20.f_matrix")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check the table only (no test is run)")
    run = commands.add_parser("run", help="run the counted tests and write the per-F manifest")
    run.add_argument("--only", help="comma-separated F ids, e.g. F01,F05")
    run.add_argument("--smoke", action="store_true", help="also run the T2 smoke tests (never counted)")
    run.add_argument("--evidence-dir", help="an existing empty directory outside any subject root")
    run.add_argument("--strict", action="store_true", help="a PENDING row makes the exit code 3")
    args = parser.parse_args(argv)

    problems = check_table()
    for problem in problems:
        print(f"TABLE: {problem}", file=sys.stderr)
    if args.command == "check":
        return 1 if problems else 0
    if problems:
        return 1
    evidence = Path(args.evidence_dir) if args.evidence_dir else Path(tempfile.mkdtemp(prefix="r20-fmatrix-", dir="/tmp"))
    only = args.only.split(",") if args.only else None
    manifest, labels = run_matrix(pytest_runner, evidence, only=only, include_smoke=args.smoke)
    path = manifest.write()
    for case_id, label in labels.items():
        print(f"{case_id}: {label}")
    pending = [case_id for case_id, label in labels.items() if label == "PENDING"]
    if pending:
        print(f"PENDING (nothing claimed): {', '.join(pending)}")
    print(f"manifest: {path}")
    return exit_code(labels, strict=bool(args.strict))


if __name__ == "__main__":
    raise SystemExit(main())
