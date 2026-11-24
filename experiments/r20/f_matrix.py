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

F15 is bound to the rewritten acceptance of the degraded-continuation slice
(F15 slice D1 r3 FINAL, sec. 10; it REPLACES IP-22 D1's old F15 text and HD-22-7 (a)):
a serving state within stated bounds for the COVERED fault classes, and an explicit
list of known violations that are reported, never hidden.  Each known violation is a
``KnownViolation`` row with the D1 text and, where a deterministic T1 test pins the
boundary, that test; its manifest label is ``NOT_COVERED_KNOWN_VIOLATION``.  The T2
F15-x tests are smoke (a real SIGKILL; the kill window is not controlled) and never
count as covered.

``--strict`` passes only when every F is ``PASS`` (DONE) and every known violation is
explicitly ``NOT_COVERED_KNOWN_VIOLATION`` with its D1 text; ``PENDING`` and
``PASS_TRANSITIONAL`` (F14, #463) rows make it exit 3.

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
F15_SLICE = "F15 slice D1 r3 FINAL (#411 comments 7104976285, 7104977032; decisions 7104977597)"
F14_ISSUE = "#463"
TRANSITIONAL_MARK = "TRANSITIONAL"

# result labels of the per-F manifest entries
PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
SMOKE_NOT_COUNTED = "SMOKE_NOT_COUNTED"
NOT_COVERED_KNOWN_VIOLATION = "NOT_COVERED_KNOWN_VIOLATION"
PASS_TRANSITIONAL = "PASS_TRANSITIONAL"


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
class KnownViolation:
    """A fault class F15 does NOT cover (a known violation of "a safety stop must
    never stop the subject"), with the D1 text.  ``evidence`` are deterministic T1
    tests that pin the boundary (it stays refused / re-evaluated / off); a class with
    no such test says so (empty tuple) and is still listed."""

    id: str
    title: str
    text: str
    source: str
    evidence: tuple[TestRef, ...] = ()


@dataclass(frozen=True, slots=True)
class CoveredClass:
    """A fault class F15 covers, the setting it holds under, and the T1 tests."""

    name: str
    setting: str
    bound: str
    evidence: tuple[TestRef, ...]


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
    # F15 only: the acceptance wording, what it covers (with its setting) and what it does not
    acceptance: str | None = None
    covered_classes: tuple[CoveredClass, ...] = ()
    known_violations: tuple[KnownViolation, ...] = ()

    @property
    def counted(self) -> tuple[TestRef, ...]:
        """The T1 tests that count as coverage; T2 smoke never does."""

        return (*self.merged, *self.new)

    @property
    def all_ids(self) -> tuple[TestRef, ...]:
        return (*self.counted, *self.smoke)

    @property
    def boundary_ids(self) -> tuple[TestRef, ...]:
        """T1 tests that pin a known-violation boundary (not coverage)."""

        seen: dict[str, TestRef] = {}
        for violation in self.known_violations:
            for ref in violation.evidence:
                seen.setdefault(ref.node_id, ref)
        return tuple(seen.values())


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
        new=(
            # IP-22d: K1 in T1 only (through the real approval route) and the K-point predicates
            _t('test_r20_ip22d_kill.py', 'test_k1_t1_a_failure_between_the_challenge_write_and_the_claim_voids_it_and_needs_a_new_approval'),
            _t('test_r20_ip22d_kill.py', 'test_each_kill_predicate_holds_in_exactly_its_own_window_of_the_chain'),
        ),
        smoke=(
            _t('test_r20_ip22c_f06c.py', 'test_f06_c_a_real_sigkill_during_the_r16_chain_finishes_with_one_decision_per_stage'),
            # IP-22d: K2..K6 on the R18 / R19 chain of a T2 S0 root (K1 has no T2 form)
            _t('test_r20_ip22d_kill.py', 'test_f06_c_a_real_sigkill_at_each_t2_point_keeps_the_invariants_after_the_restart'),
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
        owning_unit='F15 slice DC-1 / DC-4 / DC-5 / DC-6 (degraded continuation) over R06 / R07, IP-9 adoption, IP-15, IP-18a',
        determinism='DET_T1_PLUS_T2_SMOKE_KILL_WINDOW_NONDET',
        status='COVERED',
        merged=(
            # (a) a Runtime durability failure: supervised restart within T_failed_to_serving_max
            _t('test_dc5_supervisor.py', 'test_real_app_serves_restarts_after_a_runtime_failure_with_fresh_identities'),
            _t('test_dc5_supervisor.py', 'test_a_failure_the_poll_sees_without_a_callback_also_restarts'),
            _t('test_dc5_supervisor.py', 'test_a_teardown_of_140_virtual_seconds_is_not_threads_not_stopped'),
            _t('test_dc5_supervisor.py', 'test_the_stated_bounds_add_up'),
            # (b) an Experience participant divergent / evidence absent with an open transaction
            _t('test_dc4_real_app.py', 'test_real_app_quarantines_the_experience_store_and_keeps_serving'),
            _t('test_dc4_real_app.py', 'test_s0c_r15_with_an_emptied_experience_store_does_not_loop_or_stop_the_subject'),
            _t('test_dc4_real_app.py', 'test_s0c_r16_with_an_emptied_experience_store_does_not_loop_or_stop_the_subject'),
            _t('test_dc4_participant_quarantine.py', 'test_experience_diverged_finalize_path_quarantines_and_ends_reconciled'),
            _t('test_dc4_participant_quarantine.py', 'test_experience_directory_gone_is_evidence_absent_and_quarantined_with_no_artifact'),
            _t('test_dc4_participant_quarantine.py', 'test_no_row_continues_an_identifier_that_exists_only_in_the_quarantined_tree'),
            _t('test_dc4_participant_quarantine.py', 'test_the_live_process_never_opens_a_quarantined_path_after_the_quarantine'),
            _t('test_dc4_participant_quarantine.py', 'test_a_rename_failure_moves_nothing_and_writes_no_journal_record'),
            _t('test_dc4_participant_quarantine.py', 'test_aggregate_path_quarantines_restarts_and_the_same_reconciliation_resumes'),
            # (c) a Journal over its hard cap with archivable segments: the retention ladder
            _t('test_dc6_hard_band.py', 'test_real_app_hard_at_startup_runs_the_ladder_then_starts_the_runtime'),
            _t('test_dc6_hard_band.py', 'test_real_app_hard_while_serving_runs_the_ladder_at_idle_without_a_restart'),
            _t('test_dc6_hard_band.py', 'test_the_ladder_frees_a_real_journal_at_the_first_sufficient_rung'),
            _t('test_dc6_hard_band.py', 'test_the_window_never_returns_and_archives_stay_byte_identical'),
        ),
        new=(),
        smoke=(
            _t('test_r20_ip22d_f15.py', 'test_f15_x_1_a_covered_experience_fault_after_a_kill_is_quarantined_and_the_subject_serves'),
            _t('test_r20_ip22d_f15.py', 'test_f15_x_2_a_restored_copy_of_the_quarantined_tree_is_detected_blocks_and_is_never_used'),
            _t('test_r20_ip22d_f15.py', 'test_f15_x_3_an_uncovered_fault_stays_refused_with_the_process_alive_and_nothing_repaired'),
        ),
    ),
)

# The F15 acceptance wording of the F15 slice D1 r3 FINAL (sec. 10, "F15 acceptance
# wording"; replaces IP-22 D1 sec. 6 F15 and HD-22-7 (a)), verbatim.
F15_ACCEPTANCE = (
    "F15 (partial / missing / corrupt DB / Journal / WAL / source). For each fault class in the covered list "
    "(a Runtime durability failure; an Experience / Semantic participant whose store is divergent or whose "
    "evidence is absent while an open transaction or a true-rollback gate needs it; a Journal over its hard cap "
    "with archivable segments), a restart or an in-process failure reaches a serving state (Runtime ACCEPTING, "
    "chat answered) within the stated bound without Human action: for a Runtime durability failure "
    "T_failed_to_serving_max 400 s measured from the instant the failure is observed and including the teardown "
    "of the failed cycle; for a start with a participant fault T_boot_max 240 s from the start of the first "
    "build (teardown is not part of a cold start); for the HARD band T_hard (600 s) + at most 5 rungs of 60 s "
    "from the first HARD observation. In every such run: (1) each damaged artifact is byte-preserved (tree "
    "SHA-256 equal to the pre-fault value; files the bounded hash budget did not reach are recorded unhashed and "
    "are still renamed, never rewritten) beside its origin and is never modified, deleted or read again by the "
    "process; (2) no row or record copies content or continues an identifier whose only source is a quarantined "
    "artifact (the Journal outcome and the manifest carry transaction ids and hashes only); (3) no Action is "
    "admitted and no Intent / Policy / Approval / Claim is issued on a Decision whose evidence is unproven in "
    "the current process; R18 reopens only through the existing sweep and dwell (the foreign-target latch is the "
    "separate PR); (4) every stop in the run ends in a terminal or re-evaluable state within its bound. Not "
    "covered (a known violation of 'a safety stop must never stop the subject', reported as such): Journal "
    "damage (including a torn tail left by our own failed append), SubjectHistory loss / corruption / "
    "disablement / capacity, AMBIGUOUS commit evidence, memory.episodic and subject.history failures, an "
    "Experience / Semantic store that is merely Unavailable (not divergent), DB2 projection and "
    "semantic_receipt_retention failures, true-rollback gate refusals (participant_baseline_unavailable, "
    "participant_history_not_provable, participant_history_archived), a memory store damaged with no open "
    "transaction, a stale restore over a quarantined store, HARD where compaction cannot free enough, threads "
    "that do not stop, a process that dies. For those the start or the incarnation remains in a re-evaluated "
    "state (RECONCILING, REFUSED_RUNNING, RESTART_LIMIT, HARD_STUCK), and the report lists them."
)
_D1 = "F15 slice D1 r3 FINAL"
_ACC = "F15 acceptance 'Not covered'"


def _kv(id_: str, title: str, text: str, source: str, *evidence: TestRef) -> KnownViolation:
    return KnownViolation(id=id_, title=title, text=text, source=source, evidence=tuple(evidence))


# Every entry carries the D1 text.  An entry with no ``evidence`` is listed anyway
# (no deterministic test pins that boundary: stated, not hidden).
F15_KNOWN_VIOLATIONS: tuple[KnownViolation, ...] = (
    _kv(
        "KV-JOURNAL", "Journal damage",
        "Journal damage (including a torn tail left by our own failed append): the start stays REFUSED, now as "
        "REFUSED_RUNNING (process alive, re-evaluated); only a new epoch (DC-2) could continue safely.",
        f"{_ACC}; {_D1} K1, F-1, F-3",
        _t('test_dc5_supervisor.py', 'test_real_app_corrupt_journal_refuses_and_keeps_answering'),
    ),
    _kv(
        "KV-SUBJECT-HISTORY", "SubjectHistory loss / corruption / disablement / capacity",
        "SubjectHistory lost / corrupt / foreign / disabled-by-config / older schema, capacity, POISONED / "
        "DIVERGED: unchanged IP-9 refusals and sticky HOLDs; the slice never creates, replaces, quarantines or "
        "re-adopts a SubjectHistory file.",
        f"{_ACC}; {_D1} K2, K9, F-8",
        _t('test_r16_ip9_history_adoption.py', 'test_lost_or_foreign_store_after_adoption_fails_closed'),
        _t('test_r16_ip9_history_adoption.py', 'test_disabling_after_adoption_keeps_the_registry_and_fails_closed'),
    ),
    _kv(
        "KV-AMBIGUOUS", "AMBIGUOUS commit evidence",
        "AMBIGUOUS commit evidence is re-evaluated with the same evidence, never resolved to a branch.",
        f"{_ACC}; {_D1} K3, F-4",
        _t('test_startup_reconciliation.py', 'test_ambiguous_commit_proof_fails_closed'),
    ),
    _kv(
        "KV-EPISODIC-AND-HISTORY-PARTICIPANTS", "memory.episodic and subject.history failures",
        "memory.episodic (Chroma DB1) and subject.history participants are not reconcilable: they end in "
        "RECONCILING, re-evaluated, not acted on, and are never quarantined.",
        f"{_ACC}; {_D1} K4, F-6",
        _t('test_dc4_participant_quarantine.py', 'test_a_diverged_episodic_participant_is_never_quarantined'),
        _t('test_dc4_participant_quarantine.py', 'test_the_policy_refuses_every_other_participant_and_every_other_class'),
    ),
    _kv(
        "KV-UNAVAILABLE", "an Experience / Semantic store that is merely Unavailable",
        "An Experience or Semantic store that stays merely Unavailable (unreadable, not broken) is not quarantined "
        "under HD-new-2 (a): RECONCILING forever while it lasts.",
        f"{_ACC}; {_D1} K12, F-17",
        _t('test_dc4_participant_quarantine.py', 'test_a_transient_participant_fault_is_never_quarantined'),
    ),
    _kv(
        "KV-DB2-AND-RECEIPT-RETENTION", "DB2 projection and semantic_receipt_retention failures",
        "Chroma DB2 semantic projection diverged / unavailable and semantic_receipt_retention_unavailable end in "
        "RECONCILING, re-evaluated, not acted on.",
        f"{_ACC}; {_D1} K4, F-13",
    ),
    _kv(
        "KV-TRUE-ROLLBACK-GATE", "true-rollback gate refusals",
        "participant_baseline_unavailable, participant_history_not_provable and participant_history_archived end in "
        "RECONCILING, re-evaluated, not acted on; a shorter retention window (the ladder) makes "
        "participant_history_archived more likely after a later true rollback.",
        f"{_ACC}; {_D1} K4, F-12",
        _t('test_startup_reconciliation.py', 'test_true_rollback_over_an_archived_range_fails_closed'),
        _t('test_dc4_participant_quarantine.py', 'test_the_gate_with_the_legacy_policy_is_unchanged_for_a_damaged_store'),
    ),
    _kv(
        "KV-NO-OPEN-TRANSACTION", "a memory store damaged with no open transaction",
        "A damaged Experience record that no open transaction touches is not noticed at start and is not handled "
        "(DC-8, not in the slice).",
        f"{_ACC}; {_D1} F-16",
    ),
    _kv(
        "KV-STALE-RESTORE", "a stale restore over a quarantined store",
        "A restored or hand-replaced memory directory cannot be re-quarantined safely without a Journal-side epoch "
        "marker: detection and block only (RECONCILING, domain_store_replaced); no automatic action.",
        f"{_ACC}; {_D1} K11, F-15",
        _t('test_dc4_participant_quarantine.py', 'test_a_restored_directory_over_a_quarantined_store_is_detected_and_blocks'),
    ),
    _kv(
        "KV-HARD-STUCK", "HARD where compaction cannot free enough",
        "HARD where archiving cannot free enough (the active segment, carried records and the segments inside the "
        "smallest rung already exceed max_total_bytes; or deferred_history / history_attestation_required / "
        "boundary_not_clean never clears): HARD_STUCK.",
        f"{_ACC}; {_D1} F-5",
        _t('test_dc6_hard_band.py', 'test_an_overflow_inside_the_active_segment_stays_hard_f5'),
        _t('test_dc6_hard_band.py', 'test_a_real_semantic_guard_stops_the_real_ladder'),
    ),
    _kv(
        "KV-THREADS-NOT-STOPPED", "threads that do not stop",
        "Threads of a previous cycle that do not stop (a worker blocked forever, an unbounded shutdown() / "
        "EmotionTimer.stop() join): terminal REFUSED_RUNNING(threads_not_stopped), no further in-process cycle; "
        "R18 stays closed only as far as the old executor honours its stop flag.",
        f"{_ACC}; {_D1} K10, F-14",
        _t('test_dc5_supervisor.py', 'test_dc5_stuck_thread_ends_refused_running'),
        _t('test_dc5_supervisor.py', 'test_zombie_executor_means_no_in_process_restart_and_a_terminal_refusal'),
    ),
    _kv(
        "KV-PROCESS-DIES", "a process that dies",
        "A process that dies: no host restarter is assumed (HD-F15-10 (a)), but one is used if present.",
        f"{_ACC}; {_D1} K7, F-9",
    ),
    # --- two boundaries the setting-dependence and the implementation make explicit
    _kv(
        "KV-SEMANTIC-QUARANTINE-OFF", "memory.semantic quarantine is OFF by default",
        "The D1 acceptance names an Experience / Semantic participant; DC-4 implemented memory.semantic but left it "
        "OFF by default: after a Semantic quarantine every chat fails while a DB2 projection row of the old store "
        "exists (S0-c). D1 F-7: 'If unsafe, DC-4 narrows to Experience only'. A Semantic fault therefore ends "
        "RECONCILING as before: a known violation, not covered.",
        f"{_D1} F-7; suzka/runtime/domain_quarantine.py (DEFAULT_QUARANTINE_PARTICIPANTS)",
        _t('test_dc4_real_app.py', 'test_s0c_semantic_is_off_by_default_so_a_semantic_fault_is_left_as_it_was'),
        _t('test_dc4_real_app.py', 'test_s0c_a_semantic_quarantine_would_fail_every_chat_which_is_why_it_is_off'),
        _t('test_dc4_participant_quarantine.py', 'test_with_the_default_policy_a_semantic_fault_is_left_as_it_was'),
    ),
    _kv(
        "KV-HARD-SHRINK-OFF", "HARD with hard_band_auto_shrink = false",
        "hard_band_auto_shrink: false restores the old behaviour: the HARD band blocks admission for the whole "
        "incarnation (HD-F15-8 (a): the Human may switch the ladder off). The HARD class is covered only with the "
        "ladder on.",
        f"{_D1} sec. 2 A-3, sec. 8",
        _t('test_dc6_hard_band.py', 'test_flag_off_auto_shrink_disabled_the_hard_band_behaves_as_before'),
    ),
    _kv(
        "KV-MODE-REFUSE", "degraded_continuation.mode = refuse",
        "mode = 'refuse' is the exact legacy behaviour (no supervisor, no quarantine, no ladder; the first start "
        "raises, DEGRADED-UP and FAILED are permanent): none of the covered classes holds under it.",
        f"{_D1} sec. 8",
        _t('test_dc4_real_app.py', 'test_mode_refuse_never_quarantines_and_never_creates_a_file'),
        _t('test_dc6_hard_band.py', 'test_flag_off_mode_refuse_the_hard_band_behaves_as_before'),
        _t('test_dc5_supervisor.py', 'test_the_default_mode_still_raises_at_a_refused_start'),
    ),
)

_ANNOTATIONS: dict[str, dict[str, Any]] = {
    "F06": {
        "caveat": (
            "F06-c is T2 smoke and is not counted (provider deterministic, kill window non-deterministic): "
            "the R16-level SIGKILL (IP-22c) and K2..K6 on the R18 / R19 chain (IP-22d; the relative-path R18 "
            "bug is fixed, the chain runs under T2).  K1 has no T2 form and is a T1 test.  The exact windows "
            "are the merged T1 tests."
        )
    },
    "F12": {
        "caveat": (
            "F12-r ran on an S0-CONFIG root (R18 + R19 on, driver activated, scripted provider) "
            "with the R15 admission NOT performed.  The feasibility spike passed: startup was not refused.  "
            "The R15-admitted variant and the SIGKILL variant are not claimed here."
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
        "acceptance": F15_ACCEPTANCE,
        "covered_classes": (
            CoveredClass(
                name="Runtime durability failure",
                setting="degraded_continuation.mode = continue (the supervised restart)",
                bound="T_failed_to_serving_max 400 s, measured from the instant the failure is observed, teardown included",
                evidence=(
                    _t('test_dc5_supervisor.py', 'test_real_app_serves_restarts_after_a_runtime_failure_with_fresh_identities'),
                    _t('test_dc5_supervisor.py', 'test_a_teardown_of_140_virtual_seconds_is_not_threads_not_stopped'),
                    _t('test_dc5_supervisor.py', 'test_the_stated_bounds_add_up'),
                ),
            ),
            CoveredClass(
                name="Experience participant divergent or evidence absent, an open transaction needs it",
                setting="degraded_continuation.mode = continue; memory.semantic is NOT quarantined (OFF by default, "
                        "see the known violation KV-SEMANTIC-QUARANTINE-OFF)",
                bound="T_boot_max 240 s from the start of the first build (teardown is not part of a cold start)",
                evidence=(
                    _t('test_dc4_real_app.py', 'test_real_app_quarantines_the_experience_store_and_keeps_serving'),
                    _t('test_dc4_participant_quarantine.py', 'test_experience_diverged_finalize_path_quarantines_and_ends_reconciled'),
                    _t('test_dc4_participant_quarantine.py', 'test_experience_directory_gone_is_evidence_absent_and_quarantined_with_no_artifact'),
                    _t('test_dc4_participant_quarantine.py', 'test_no_row_continues_an_identifier_that_exists_only_in_the_quarantined_tree'),
                ),
            ),
            CoveredClass(
                name="Journal over its hard cap with archivable segments",
                setting="degraded_continuation.mode = continue AND hard_band_auto_shrink = true (the default); the "
                        "ladder is exercised only with hard_band_auto_shrink, never with it off",
                bound="T_hard 600 s of continuous HARD, then at most 5 rungs of 60 s",
                evidence=(
                    _t('test_dc6_hard_band.py', 'test_real_app_hard_at_startup_runs_the_ladder_then_starts_the_runtime'),
                    _t('test_dc6_hard_band.py', 'test_real_app_hard_while_serving_runs_the_ladder_at_idle_without_a_restart'),
                    _t('test_dc6_hard_band.py', 'test_the_ladder_frees_a_real_journal_at_the_first_sufficient_rung'),
                ),
            ),
        ),
        "known_violations": F15_KNOWN_VIOLATIONS,
        "caveat": (
            "Counted coverage is the deterministic T1 DC tests.  F15-x-1 / -2 / -3 are T2 smoke (a real "
            "SIGKILL; provider deterministic, kill window non-deterministic) and are not counted.  F15 holds "
            "only with degraded_continuation.mode = continue; with 'refuse' none of the covered classes holds "
            "(KV-MODE-REFUSE).  Semantic quarantine is OFF by default, so a Semantic fault is a known violation, "
            "not covered (a narrowing of the D1 wording, see KV-SEMANTIC-QUARANTINE-OFF)."
        ),
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
    return any(marker in text for marker in ("Tier.T2", "ChildProcess", "S0Session", "run_kill_point", "_t2_quarantined_subject("))


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
        problems += _check_f15_binding(case, functions)
    return problems


def _check_f15_binding(case: FCase, functions: Callable[[str], dict[str, ast.AST] | None]) -> list[str]:
    """A row that names known violations or covered classes carries the acceptance
    wording, every violation carries its D1 text, and every test they cite exists,
    is not skipped and is a T1 test."""

    problems: list[str] = []
    label = case.case
    if not (case.known_violations or case.covered_classes or case.acceptance):
        return problems
    if not case.acceptance or "Not covered" not in case.acceptance:
        problems.append(f"{label}: no acceptance wording with its 'Not covered' list")
    if not case.known_violations:
        problems.append(f"{label}: an acceptance wording without a known-violation list")
    if not case.covered_classes:
        problems.append(f"{label}: an acceptance wording without covered classes")
    ids = [violation.id for violation in case.known_violations]
    if len(ids) != len(set(ids)):
        problems.append(f"{label}: a known-violation id is used twice")
    counted = {ref.node_id for ref in case.counted}

    def check(ref: TestRef, owner: str) -> None:
        found = functions(ref.file)
        if found is None or ref.name not in found:
            problems.append(f"{label}: {owner} cites {ref.node_id}, which does not exist")
        elif _is_disabled(found[ref.name]):
            problems.append(f"{label}: {owner} cites {ref.node_id}, which is skipped or xfail (vacuous)")
        elif uses_process_tier(found[ref.name]):
            problems.append(f"{label}: {owner} cites {ref.node_id}, a T2 test (only T1 pins a boundary)")

    for violation in case.known_violations:
        if not violation.title.strip() or len(violation.text.split()) < 8 or not violation.source.strip():
            problems.append(f"{label}: {violation.id} has no D1 text or no source")
        for ref in violation.evidence:
            check(ref, violation.id)
    for covered in case.covered_classes:
        if not covered.setting.strip() or not covered.bound.strip() or not covered.evidence:
            problems.append(f"{label}: covered class {covered.name!r} has no setting, bound or evidence")
        for ref in covered.evidence:
            check(ref, covered.name)
            if ref.node_id not in counted:
                problems.append(f"{label}: covered class {covered.name!r} cites {ref.node_id}, which is not a counted test")
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
        wanted += [ref.node_id for ref in case.boundary_ids]  # the boundary tests are T1 and always run
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
                label = PASS if case.status == COVERED else PASS_TRANSITIONAL
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
        if case.status != PENDING:
            for covered in case.covered_classes:
                states = [outcome(ref) for ref in covered.evidence]
                manifest.result(
                    f"{case.case}.covered.{covered.name}",
                    PASS if all(state == "passed" for state in states) else (NOT_RUN if NOT_RUN in states else FAIL),
                )
            for violation in case.known_violations:
                # the D1 text is the label's reason; a failing boundary test means the boundary moved
                states = [outcome(ref) for ref in violation.evidence]
                if any(state == NOT_RUN for state in states):
                    result_label = NOT_RUN
                elif all(state == "passed" for state in states):
                    result_label = NOT_COVERED_KNOWN_VIOLATION
                else:
                    result_label = FAIL
                manifest.result(f"{case.case}.known_violation.{violation.id}", result_label)
                manifest.append(
                    "timeline",
                    {
                        "kind": "known_violation",
                        "case": case.case,
                        "id": violation.id,
                        "label": result_label,
                        "text": violation.text,
                        "source": violation.source,
                        "boundary_tests": [ref.node_id for ref in violation.evidence],
                    },
                )
    problems = check_table()
    manifest.result("table", PASS if not problems else FAIL)
    return manifest, labels


def exit_code(
    labels: Mapping[str, str], *, strict: bool, results: Mapping[str, str] | None = None
) -> int:
    """0 = nothing failed; 1 = a row or a known-violation boundary FAILed or was
    NOT_RUN; 3 (``--strict`` only) = something is not DONE: a PENDING row, a
    PASS_TRANSITIONAL row (F14, #463), or a known violation that is not explicitly
    ``NOT_COVERED_KNOWN_VIOLATION``.  ``--strict`` therefore passes only when every F
    is ``PASS`` and every uncovered class is a named known violation with its D1 text."""

    violation_labels = {
        key: label for key, label in (results or {}).items() if ".known_violation." in key
    }
    covered_labels = {key: label for key, label in (results or {}).items() if ".covered." in key}
    every = [*labels.values(), *violation_labels.values(), *covered_labels.values()]
    if any(label in {FAIL, NOT_RUN} for label in every):
        return 1
    if strict:
        if any(label in {"PENDING", PASS_TRANSITIONAL} for label in labels.values()):
            return 3
        if any(label != NOT_COVERED_KNOWN_VIOLATION for label in violation_labels.values()):
            return 3
        if any(label != PASS for label in covered_labels.values()):
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
    run.add_argument(
        "--strict",
        action="store_true",
        help="exit 3 unless every F is PASS and every uncovered class is NOT_COVERED_KNOWN_VIOLATION",
    )
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
    results = manifest.to_dict()["results"]
    for key, label in results.items():
        if ".known_violation." in key:
            print(f"{key}: {label}")
    transitional = [case_id for case_id, label in labels.items() if label == PASS_TRANSITIONAL]
    if transitional:
        print(f"TRANSITIONAL (not DONE, --strict fails): {', '.join(transitional)}")
    print(f"manifest: {path}")
    return exit_code(labels, strict=bool(args.strict), results=results)


if __name__ == "__main__":
    raise SystemExit(main())
