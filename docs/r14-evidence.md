# R14 U6 bounded evidence map

> **Evidence index, not an automatic PASS or readiness declaration.** The
> classifications below tell the reader whether R14 has direct executable
> evidence for a bounded claim or whether the capability is explicitly deferred.
> This file does not assert that its linked tests were run in this session, does
> not deliver a deferred producer, and is not Human U6 acceptance.

## Scope, authority, and evidence boundary

U6 is authorized only for whole-stage integration, adversarial/regression
testing, bounded evidence, and final review of the accepted R14 U1–U5 work. U6
does not redesign policy, add a source, change an upstream authority, or deliver
an R15–R19 capability; no redesign is authorized. The evidence matrix is a
compact map of the existing implementation and tests, not a feature-completion
checklist.

The bounded source inventory is four candidate target kinds: 4,096 current
Working Memory members plus at most 32 ACTIVE Motivations, 32 ADOPTED Goals, and
32 ACTIVE Commitments, for **4,192** total identities. Source adapters preserve
typed references, source-local revision/event/digest witnesses and explicit
availability; Attention has its own independent revision namespace. Context
and current Emotion are bounded evidence inputs, with Emotion changing the
global capacity only. A checksum binds declared data; it does not authenticate
the producer or prove that separately read authorities were atomically
captured. U5's actual serialized runtime event is the trusted capture boundary.

The **131,072-byte** ceiling limits only the Attention-selected contribution,
not the whole model prompt. Attention either includes a complete witnessed row
or explicitly omits that row. It does not truncate a source record or repack
additional rows after U5 removes structural Working Memory identity fields for
display. Existing Context, Values, Emotion, instruction, and User prompt content
remain outside that contribution budget. Natural identifiers present inside
source text are preserved; structural item/source IDs are not rendered.

| Bounded implementation value | Accepted bound |
| --- | ---: |
| Supported universe | 4,192 (4,096 Working Memory + 32 each Motivation, Goal, Commitment) |
| Focus ceiling | 16 / 8 at global arousal >= 0.75 |
| Attention-selected contribution | 131,072 bytes |
| Derived Attention continuity maximum | 4,193,496 bytes |
| Frozen v8 base maximum | 109,573,688 bytes |
| Maximum R14 addition including root key | 7,866,824 bytes |
| v9 including reserve | 130,544,419 bytes |
| Future-state reserve | 16,777,216-byte reserve |
| AgentState hard cap | 134,217,728-byte hard cap |
| Margin beyond reserve | 3,673,309-byte margin beyond reserve |

No U6 change spends capacity or alters the v8/v9 codecs. The 131,072-byte
limit is the Attention-selected contribution, not the entire model prompt.

### Deliberately unavailable producers (not feature delivery)

- **Subject-source producer: unavailable.** No Attention subject/Goal admission
  or Commitment acceptance producer is introduced by R14. The current candidate
  target inventory remains Working Memory, Motivation, Goal, and Commitment;
  upstream owners remain authoritative for creating/adopting/accepting their
  records.
- **Optional U3 Metacognition production: unavailable.** U3 is an ephemeral pure
  observation/assessment capability and U5 does not call or persist it. No
  production assessment, assessment prompt feedback, calibration history,
  Decision recommendation, or outcome learner is delivered here.
- The accepted `AttentionSourceKind` vocabulary can name typed source categories
  that do not have a current candidate producer. A vocabulary entry is not an
  adapter, producer, or claim that Value, Belief, Experience, Appraisal, or
  subject-derived candidate projection is available.
- These declared absences are boundaries, not a claim of feature delivery.

### R14 responsibility contract for U6

- **Owns:** the bounded whole-R14 evidence index, executable map consistency,
  crosscutting reference inventory, and review caveats for accepted U1–U5.
- **May:** map existing exact tests; check rows/references against the local AST;
  summarize bounded integration, recovery, privacy, and compatibility claims.
- **Must not:** redesign or edit `suzka/**`, mutate upstream source truth, create
  subject or optional-Metacognition production, introduce capability/Decision/
  outcome/scheduler authority, alter persistence, or claim security/performance
  properties not tested here.
- **Depends on:** accepted #281 contract and U1–U5; R03–R13 state, runtime,
  transaction, privacy, and recovery contracts.
- **Used by:** the U6 whole-stage review, whole-R14 review, and subsequent Human
  exact-head review. U6 does not grant Ready or Merge authority.

Compatibility remains additive: retained v8 loads lazily without eager rewrite;
the first normal successful Attention-authoritative U5 commit publishes v9.
Explicit true rollback to retained v8 keeps R13 and bootstraps empty Attention;
rollback to an older v9 restores its exact persisted Attention continuity.
Malformed/future state and ordinary v9 downgrade fail closed. Restore/recovery
does not rerank, search Memory, invoke a model, recompute Metacognition, or rebuild
transient competition/prompt witnesses. These are tested boundaries, not a claim
about arbitrary filesystem or hardware failures.

## Accepted lineage and review state

| Unit | Accepted lineage recorded for this evidence map | Boundary |
| --- | --- | --- |
| U1 | `cdced8c2148d4fd07b47ae92c6a538ac762618a6`, review comment `6012724848` | Immutable contracts, adapters, deterministic policy, and capacity projection. |
| U2 | `cfd98aab41aa4b5a4c86afd0c4b4de0dcf846d8a`, review comment `6027481363` | Process-local Attention owner only. |
| U3 | `41023f05abc03784e68b93b5b70be81c77977506`, Human review comment `6031506761` | Pure event-scoped assessment; no production wiring/history. |
| U4 | `ee7e9f849e0b6f0f3ccbd455a7acae9800a8a056`, Human review comment `6035601273` | Additive v9 persistence and cognition-free restore/recovery. |
| U5 | Human U5 PASS/COMPLETE at `8c63d21501d09fe843694f6cc8cd5cfe40087892`, comment `6051359998` | Same-turn current-source capture, selected prompt, and committed read views. |

The U5 PASS is lineage only; it is not U6/whole-R14 approval. Draft PR #282
remains Draft. Human U6/whole-R14 review is pending. Review sequence:
**U6 -> whole R14 -> Human exact-head review -> HumanReadyMerge**. Publication
is authorized through the normal branch process; this evidence map does not
assert that it has occurred.

## A1–A24 accepted-contract index

The clause descriptions below are concise accepted-contract summaries, not
verbatim quotations. They index the [authoritative #281 issue and final
acceptance](https://github.com/upiscium/PROJECT-SUZKA/issues/281#issuecomment-5996825681); that final
acceptance and the accepted U1–U5 policy govern any compression here. The
related F rows and executable nodes are indexed together, and the matrix test
requires complete ordered A1–A24 coverage and resolves every test node by AST.
Primary checked the authoritative final Acceptance Contract at comment
`5996825681` and U5 acceptance/U6 authorization at comment `6051359998`.

| Accepted clause | Concise accepted-contract summary | Related F rows | Executable evidence |
| --- | --- | --- | --- |
| A1 | Attention and Metacognition are distinct: source focus is not winner selection, and confidence is not truth. | F11, F12, F13 | `tests/test_attention_policy.py::test_prompt_frame_uses_parent_frozen_authority_intro_and_literal_headings`; `tests/test_metacognition_contracts.py::test_confidence_requires_a_typed_support_ceiling_and_unknown_is_never_confident`; `tests/test_metacognition_assessment.py::test_resource_only_assessment_never_claims_factual_confidence` |
| A2 | Attention owns persisted intrinsic continuity, distinct from source truth and transient selection witnesses. | F4, F18 | `tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out`; `tests/test_agent_state_v9.py::test_v9_alias_round_trip_uses_canonical_attention_and_requires_owner`; `tests/test_r14_wal_recovery.py::test_mixed_v8_v9_v9_wal_reconstructs_exact_attention_roots_and_receipts` |
| A3 | Metacognition is event-scoped and non-durable; it has no production wiring or assessment history. | F12, F17, F19 | `tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence`; `tests/test_r14_u4_scope.py::test_u4_snapshot_has_no_assessment_or_rendered_source_fields`; `tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only` |
| A4 | The current source universe is 4096 WM plus ACTIVE Motivation, ADOPTED Goal, and ACTIVE Commitment bounds for 4192; do not scan historical records. | F8, F20 | `tests/test_attention_bounds.py::test_supported_identity_capacity_is_source_owned_and_complete`; `tests/test_attention_adapters.py::test_snapshot_batch_covers_full_4192_source_universe_canonically`; `tests/test_attention_main_loop.py::test_real_main_loop_projects_the_complete_4192_candidate_universe` |
| A5 | Other source signals require exact typed links; free-text similarity is not link or confidence authority. | F5, F12, F13 | `tests/test_metacognition_evidence.py::test_linked_belief_uses_only_current_typed_r13_references_and_scope_gate`; `tests/test_metacognition_evidence.py::test_belief_links_are_taken_from_motivation_related_goal_and_commitment_refs`; `tests/test_metacognition_assessment.py::test_only_typed_current_structured_belief_conflicts_yield_contradiction` |
| A6 | Raw user/model text and model self-reports do not become Attention or Metacognition authority. | F4, F12, F19 | `tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters`; `tests/test_metacognition_assessment_boundaries.py::test_source_text_is_not_retained_or_scored_and_assessment_is_pure`; `tests/test_r14_u5_scope.py::test_u5_producer_rejects_caller_forged_context_even_on_actual_worker` |
| A7 | The four-stage Attention pipeline is fixed and deterministic over typed current inputs, prior continuity, and the event. | F6, F7, F10 | `tests/test_attention_policy.py::test_ties_and_input_permutations_have_canonical_total_order_and_digest`; `tests/test_attention_system.py::test_tied_scores_and_input_permutations_produce_identical_state_and_views`; `tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt` |
| A8 | Attention owns focus only: it does not mutate source truth, select Actions, or schedule work. | F5, F11, F22 | `tests/test_attention_system.py::test_real_source_adapters_feed_exact_current_targets_without_mutating_sources`; `tests/test_attention_system.py::test_source_kind_never_overrides_fixed_numeric_competition`; `tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local` |
| A9 | R08 owns Working Memory membership and resolution eligibility; R14 performs final focus and prompt selection without changing R08 selection. | F5, F8, F10 | `tests/test_attention_adapters.py::test_selected_working_memory_row_is_exact_and_may_exceed_prompt_budget`; `tests/test_attention_policy.py::test_unavailable_working_memory_keeps_measured_metadata_but_does_not_compete`; `tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation` |
| A10 | Prompt omission is explicit and bounded; rows are omitted whole and never truncated outside Attention. | F10, F20 | `tests/test_attention_policy.py::test_single_oversized_prompt_row_is_explicitly_omitted_not_truncated`; `tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation`; `tests/test_r14_u5_scope.py::test_u5_document_limits_are_selected_contribution_not_whole_prompt` |
| A11 | The production Attention refresh and selected prompt are bound to the same serialized runtime event. | F6, F10, F21 | `tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt`; `tests/test_agent_runtime_ordered_time.py::test_real_chat_attention_belief_and_goal_keep_runtime_event_identity`; `tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only` |
| A12 | Current uncommitted Experience is not captured as current source evidence for its own response. | F5, F8 | `tests/test_main_loop.py::test_current_future_episode_is_absent_from_its_own_working_memory_view`; `tests/test_r14_evidence_matrix.py::test_source_inventory_and_unwired_assessment_boundary_are_closed` |
| A13 | Restart restores reference-first Attention continuity without source text, reranking, or prompt reconstruction. | F4, F18, F19 | `tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay`; `tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy`; `tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out` |
| A14 | Invalid, unavailable, inactive, removed, future, or stale sources follow explicit fail-closed transitions without repair. | F8, F20 | `tests/test_attention_system.py::test_unavailable_or_inactive_rows_remain_but_lose_references_and_no_cache`; `tests/test_attention_adapters.py::test_r13_adapters_reject_future_events_and_ineligible_lifecycles`; `tests/test_r14_integration.py::test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation` |
| A15 | No direct operator refocus or producer API is added; AttentionSystem retains its closed process-local surface. | F11, F22 | `tests/test_attention_main_loop.py::test_main_loop_owns_an_isolated_attention_system_and_concrete_state_port`; `tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local`; `tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities` |
| A16 | Metacognition reports only bounded current event-scoped resource, evidence, and confidence metrics. | F12, F13, F14 | `tests/test_r14_u3_scope.py::test_u3_documented_policy_and_ephemeral_envelope_match_derivation`; `tests/test_metacognition_assessment.py::test_fixed_integer_quality_and_ceiling_coverage_confidence_are_order_invariant`; `tests/test_metacognition_evidence.py::test_current_working_memory_join_derives_load_without_retaining_rendered_content` |
| A17 | Confidence is deterministic and evidence-bound; UNKNOWN, CONTRADICTORY, and overload are distinct, and missing evidence is not contradiction. | F12, F13 | `tests/test_metacognition_contracts.py::test_confidence_requires_a_typed_support_ceiling_and_unknown_is_never_confident`; `tests/test_metacognition_contracts.py::test_missing_unknown_and_measured_zero_remain_distinct`; `tests/test_metacognition_assessment_boundaries.py::test_all_unknown_nonempty_source_witnesses_assess_to_all_unknown_metrics`; `tests/test_metacognition_assessment.py::test_only_typed_current_structured_belief_conflicts_yield_contradiction` |
| A18 | R14 adds no Decision, autonomy, Action, or Metacognition feedback authority. | F11, F16, F17 | `tests/test_r14_u1_scope.py::test_metacognition_is_neither_persisted_history_nor_a_decision_contract`; `tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters`; `tests/test_r14_u3_scope.py::test_u3_modules_have_no_runtime_or_mutation_authority` |
| A19 | Metacognition does not create durable traits, bias, capability, or assessment history. | F15, F17 | `tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence`; `tests/test_r14_u4_scope.py::test_u4_snapshot_has_no_assessment_or_rendered_source_fields`; `tests/test_metacognition_contracts.py::test_values_are_frozen_slot_closed_and_have_no_history_or_metadata_fields` |
| A20 | Capacity stays under hard cap 134217728 with reserve 16777216, v8 base 109573688, R14 maximum addition 7866824, and v9 plus reserve 130544419. | F20, F21 | `tests/test_attention_bounds.py::test_attention_state_value_and_exact_v9_reserved_projection_fit`; `tests/test_agent_state_v9.py::test_first_v9_capacity_is_derived_from_the_exact_attention_field_bound`; `tests/test_r14_u1_scope.py::test_u1_documented_capacity_and_resources_match_executable_derivation` |
| A21 | Recovery is exact and cognition-free; restart does not replay Attention or Metacognition. | F18, F20, F21 | `tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy`; `tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay`; `tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay` |
| A22 | No raw prompt, transcript, hidden thought, model, or provider state is durable R14 state. | F4, F19 | `tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out`; `tests/test_r14_u5_scope.py::test_u5_payload_is_ephemeral_sealed_and_has_distinct_byte_witnesses`; `tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful` |
| A23 | No scheduler/wake/Outbox is added; finite explicit Sleep is not R19#279 scheduling authority. | F22 | `tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local`; `tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active`; `tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities` |
| A24 | R14 adds no R15-R19 authority or producer. | F15, F16, F17, F22 | `tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities`; `tests/test_r14_u3_scope.py::test_u3_modules_have_no_runtime_or_mutation_authority`; `tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local` |

## F1–F22 classification and evidence

The `#281 definition` column gives concise bounded summaries, not verbatim
quotations; final A1–A24 and accepted policy govern. `DIRECT_R14_EVIDENCE` means tests directly exercise the listed
R14 boundary; it is not an automatic PASS and does not say that the tests were
executed here. `EXCLUDED_DEFERRED` identifies a later owner and is not evidence
that its capability exists. Every executable reference is resolved by parsing
the referenced test module's AST; the matrix test also enforces exact ordered
agreement between this table and its checked-in row data.

| Requirement | Classification | Deferred owner | #281 definition | Bounded R14 interpretation | Executable evidence |
| --- | --- | --- | --- | --- | --- |
| F1 | DIRECT_R14_EVIDENCE | — | High salience transient can't permanently evict persistent Goal/Commitment without bounded switch/persistence. | Real R08 transient challengers displace current Goal/Commitment focus with fixed switch cost and bounded unfinished evidence; removal restores the unchanged current R13 focus at the next event. This is bounded continuity, not eternal focus or absolute source-kind priority. | `tests/test_attention_policy.py::test_continuity_bonus_switch_cost_hysteresis_and_displaced_unfinished`; `tests/test_attention_system.py::test_salient_challenger_eventually_switches_after_repeat_focus_habituation`; `tests/test_r14_focus_regressions.py::test_transient_r08_challengers_displace_then_restore_real_r13_focus` |
| F2 | DIRECT_R14_EVIDENCE | — | Multiple high-arousal candidates can't monopolize all finite capacity. | One global focus ceiling falls from 16 to 8 at arousal >= 0.75 without changing candidate scores. There are no per-kind quotas, reserved slots, or fairness guarantee; a single kind can occupy the bounded set. | `tests/test_attention_system.py::test_high_global_arousal_only_caps_focus_and_does_not_change_candidate_scores`; `tests/test_attention_policy.py::test_unknown_global_emotion_is_explicit_and_emotion_only_changes_capacity` |
| F3 | DIRECT_R14_EVIDENCE | — | Idle when no eligible candidate exists. | An empty, unavailable-only, or below-threshold candidate set yields an explicit idle outcome; no source is promoted to make a winner. | `tests/test_attention_policy.py::test_policy_is_idle_when_no_candidate_is_eligible_or_above_threshold`; `tests/test_attention_system.py::test_bootstrap_empty_refresh_and_current_exact_retry_are_receipted` |
| F4 | DIRECT_R14_EVIDENCE | — | Typed provenance; no raw content persisted. | Attention continuity retains closed typed references, source witnesses, digests, and bounded counters, not rendered source text or signal vectors. Hashes bind declared values and do not authenticate producers. | `tests/test_attention_contracts.py::test_projection_is_closed_reference_only_and_preserves_signal_missingness`; `tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out`; `tests/test_attention_state_codec.py::test_full_legal_candidate_universe_is_preserved_without_truncation` |
| F5 | DIRECT_R14_EVIDENCE | — | Selection does not mutate source authorities. | Adapters and selection are read-only projections over caller-supplied current snapshots/views. Attention does not mutate, admit, resolve, or revise Motivation, Goal, Commitment, Working Memory, or Emotion truth. | `tests/test_attention_system.py::test_real_source_adapters_feed_exact_current_targets_without_mutating_sources`; `tests/test_attention_policy.py::test_policy_and_prompt_are_source_immutable_and_revalidate_event_bindings`; `tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful` |
| F6 | DIRECT_R14_EVIDENCE | — | Deterministic from prior authority, typed projections, and explicit event. | Canonical typed inputs and event bind deterministic competition and continuity. The separate refresh-operation receipt digest binds projections/event/Emotion for retry identity and intentionally excludes prior authority; competition itself binds the prior root. | `tests/test_attention_policy.py::test_ties_and_input_permutations_have_canonical_total_order_and_digest`; `tests/test_attention_system.py::test_tied_scores_and_input_permutations_produce_identical_state_and_views`; `tests/test_attention_system_boundaries.py::test_input_digest_binds_event_sorted_projections_and_emotion_not_prior_authority`; `tests/test_attention_system.py::test_same_primary_revision_must_keep_its_digest_and_source_revision_is_independent` |
| F7 | DIRECT_R14_EVIDENCE | — | Retry is idempotent and does not inflate habituation. | An exact retained event plus operation input returns a replay receipt/current view without scoring or continuity arithmetic. Older unverifiable retries fail closed; the bounded receipt window is not an infinite replay ledger. | `tests/test_attention_system.py::test_bootstrap_empty_refresh_and_current_exact_retry_are_receipted`; `tests/test_attention_system.py::test_historical_exact_retry_returns_current_state_and_original_accepted_receipt`; `tests/test_attention_system.py::test_constructor_and_restore_retry_without_reranking_or_habituation_helpers` |
| F8 | DIRECT_R14_EVIDENCE | — | Source removal or inactivation is explicit. | The trusted complete current capture determines membership. Absent candidates leave current continuity; unavailable/inactive rows lose focus and unfinished references; no shadow cache restores removed state. | `tests/test_attention_system.py::test_unavailable_or_inactive_rows_remain_but_lose_references_and_no_cache`; `tests/test_attention_system.py::test_unfinished_references_survive_only_while_the_complete_current_set_has_them`; `tests/test_attention_adapters.py::test_snapshot_batch_requires_complete_working_memory_membership_coverage` |
| F9 | DIRECT_R14_EVIDENCE | — | Unfinished evidence and switch cost are bounded and explicit. | A displaced focused item can remain unfinished only while current and eligible. Its union with prior unfinished IDs is bounded at 16; overflow rejects with explicit evidence rather than truncation. Switching has a fixed cost, not an absolute veto. | `tests/test_attention_policy.py::test_continuity_bonus_switch_cost_hysteresis_and_displaced_unfinished`; `tests/test_attention_policy.py::test_unfinished_overflow_returns_bounded_fail_closed_evidence_without_slicing`; `tests/test_attention_system_boundaries.py::test_unfinished_reference_overflow_is_atomic_without_truncation` |
| F10 | DIRECT_R14_EVIDENCE | — | No truncation outside Attention. | Attention omits whole oversized selected rows and never shortens source records. The 131072-byte bound applies only to the Attention-selected contribution, not the entire model prompt or other source projections. | `tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation`; `tests/test_attention_prompt.py::test_attention_builder_rejects_combined_legacy_r13_fallback`; `tests/test_r14_u5_scope.py::test_u5_document_limits_are_selected_contribution_not_whole_prompt` |
| F11 | EXCLUDED_DEFERRED | R16 | No Goal/Action winner is selected by Attention. | R14 ranks bounded attention resources only. The frozen prompt frame disclaims Goal winners/actions; the later Decision owner is not implemented by this evidence scope. | `tests/test_attention_policy.py::test_prompt_frame_uses_parent_frozen_authority_intro_and_literal_headings`; `tests/test_attention_system.py::test_source_kind_never_overrides_fixed_numeric_competition`; `tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local` |
| F12 | DIRECT_R14_EVIDENCE | — | Confidence is evidence-bound, not model certainty. | U3 confidence requires typed supporting recorded-confidence bounds, verified link coverage, and observed quality. No model self-report, source similarity, normative Value confidence, or raw text creates confidence; U5 does not wire optional U3 assessment into production. | `tests/test_metacognition_contracts.py::test_confidence_requires_a_typed_support_ceiling_and_unknown_is_never_confident`; `tests/test_metacognition_assessment_boundaries.py::test_source_confidence_uses_only_current_active_belief_records`; `tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters` |
| F13 | DIRECT_R14_EVIDENCE | — | Missing evidence is not contradiction and is not overload. | U3 preserves unknown versus measured zero, requires typed current structured conflicts for CONTRADICTORY, and never emits SUFFICIENT. Load/saturation cannot turn absent links into contradictions or complete knowledge. | `tests/test_metacognition_contracts.py::test_missing_unknown_and_measured_zero_remain_distinct`; `tests/test_metacognition_assessment.py::test_only_typed_current_structured_belief_conflicts_yield_contradiction`; `tests/test_metacognition_assessment_boundaries.py::test_all_unknown_nonempty_source_witnesses_assess_to_all_unknown_metrics`; `tests/test_metacognition_assessment_boundaries.py::test_source_confidence_uses_only_current_active_belief_records` |
| F14 | DIRECT_R14_EVIDENCE | — | Load, saturation, and Emotion quality cannot rewrite source truth. | U3 resource heuristics affect event-scoped assessment quality only. Attention/Metacognition reads and assessment operations do not mutate source authorities, Attention continuity, Belief truth, or persisted source state. The live-turn integration assessment is invoked only by its test harness; it does not wire U3 into production. | `tests/test_metacognition_assessment_boundaries.py::test_source_text_is_not_retained_or_scored_and_assessment_is_pure`; `tests/test_metacognition_assessment_boundaries.py::test_maximum_attention_root_and_restore_remain_unchanged_by_explicit_assessment`; `tests/test_attention_system.py::test_real_source_adapters_feed_exact_current_targets_without_mutating_sources`; `tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only` |
| F15 | EXCLUDED_DEFERRED | R15 | No capability authority before R15. | R14 adds no capability, trait, bias, or self-model producer. The closed U3 assessment shape has no such fields; no claim is made about a later SelfModel implementation. | `tests/test_r14_u1_scope.py::test_metacognition_is_neither_persisted_history_nor_a_decision_contract`; `tests/test_r14_u3_scope.py::test_u3_modules_have_no_runtime_or_mutation_authority`; `tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence` |
| F16 | EXCLUDED_DEFERRED | R16 | No Decision recommendations before R16. | No R14 Attention or U3 assessment field/API supplies a Decision recommendation or Action. R16 owns any later Decision authority; this is an absence boundary, not a delivered recommendation feature. | `tests/test_metacognition_contracts.py::test_values_are_frozen_slot_closed_and_have_no_history_or_metadata_fields`; `tests/test_r14_u1_scope.py::test_metacognition_is_neither_persisted_history_nor_a_decision_contract`; `tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters` |
| F17 | EXCLUDED_DEFERRED | R17, R18 | No unverified outcome calibration before R17/R18. | R14 persists no MetacognitiveAssessment history/calibration and creates no outcome learner. R17/R18 own later verified-outcome authority; checksum integrity is not producer authentication. | `tests/test_r14_u3_scope.py::test_u3_values_add_no_history_traits_or_attention_persistence`; `tests/test_r14_u4_scope.py::test_u4_snapshot_has_no_assessment_or_rendered_source_fields`; `tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay` |
| F18 | DIRECT_R14_EVIDENCE | — | Restore is cognition-free and exact. | U4 restores the exact persisted Attention continuity through its owner port without reranking, source lookup, prompt reconstruction, or U3 assessment. Transient competition/prompt witnesses remain absent after restore; true rollback to retained v8 bootstraps Attention. | `tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy`; `tests/test_agent_state_v9.py::test_v9_alias_round_trip_uses_canonical_attention_and_requires_owner`; `tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay`; `tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay` |
| F19 | DIRECT_R14_EVIDENCE | — | No raw prompt/transcript/hidden/model/provider state is durable R14 state. | Attention continuity and committed read views contain bounded typed references/witnesses, not raw text or provider state. U5's request-scoped payload may render selected source text but is not placed in the durable root; this is not a claim about other application storage. | `tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out`; `tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful`; `tests/test_r14_u5_scope.py::test_u5_payload_is_ephemeral_sealed_and_has_distinct_byte_witnesses` |
| F20 | DIRECT_R14_EVIDENCE | — | Bounds, future/malformed inputs, and revision errors fail closed; never truncate. | Source, focus, unfinished, revision, receipt, byte, nesting, schema, checksum, and root-event fences reject invalid/over-bound input before publication or repair. Only Attention may omit a whole row under its own byte budget. The restored-root integration case exercises U5 producer-only refresh after restore; it is not full HTTP rollback or automatic migration. | `tests/test_attention_contracts.py::test_continuity_rejects_future_versions_malformed_counters_and_dangling_focus`; `tests/test_attention_contracts.py::test_deeply_nested_json_fails_closed_before_recursive_construction`; `tests/test_attention_bounds.py::test_one_over_candidate_and_receipt_limits_are_rejected_not_clipped`; `tests/test_attention_state_codec.py::test_v9_raw_attention_byte_limit_precedes_attention_decoder`; `tests/test_agent_state_v9.py::test_v9_requires_a_closed_attention_field_and_root_event_fences`; `tests/test_r14_wal_recovery.py::test_raw_v9_attention_tampering_is_rejected_before_wal_repair_or_truncation`; `tests/test_r14_integration.py::test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation`; `tests/test_r14_wal_recovery.py::test_ordinary_v9_to_legacy_transition_is_rejected_by_wal_without_mutation`; `tests/test_r14_wal_recovery.py::test_ordinary_v9_internal_commit_rejects_legacy_before_journal_prepare` |
| F21 | DIRECT_R14_EVIDENCE | — | R03–R13 contracts remain non-invaded and compatible through recovery. | R14 is additive to the v8 base and preserves tested runtime/WAL/transaction, context, Value, appraisal, API, and Sleep candidate boundaries. V8 remains lazy-compatible; explicit true rollback to v8 retains R13 and bootstraps empty Attention by the accepted legacy rule. | `tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences`; `tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable`; `tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value`; `tests/test_agent_state_v9.py::test_v8_schema_and_capture_remain_explicitly_legacy`; `tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_to_retained_v8_bootstraps_attention_and_keeps_r13` |
| F22 | EXCLUDED_DEFERRED | R19 | No scheduler, wake, or Outbox authority before R19. | AttentionSystem exposes only refresh, restore_snapshot, selected_view, and snapshot; no scheduler, wake, Outbox, timer, or R19 mode is delivered. Existing finite explicit Sleep/QLoRA candidate work is not an Attention scheduler or R19#279 mode. | `tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local`; `tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities`; `tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active` |

## Crosscutting evidence matrices

These stable keys index existing executable tests; they do not imply those tests
were run as part of editing this evidence map. The U6 matrix test verifies the
ordered keys, exact references, and AST resolution together. The whole-prompt
budget is not conflated with the Attention-selected budget. Likewise, the
runtime's last-successful-admission watermark is **process-local**: persisted
Attention/AgentState root chronology fences still apply after restart, but no
cross-restart watermark or clock-migration guarantee is introduced.

| Matrix key | Cases indexed | Executable evidence |
| --- | --- | --- |
| `recovery_crash` | Ordinary downgrade refusal is distinct from true rollback; lazy v8-to-v9 publication; crash/root/malformed-state fences; production preflight/published failure. | `tests/test_r14_wal_recovery.py::test_ordinary_v9_to_legacy_transition_is_rejected_by_wal_without_mutation`; `tests/test_r14_wal_recovery.py::test_ordinary_v9_internal_commit_rejects_legacy_before_journal_prepare`; `tests/test_r14_wal_recovery.py::test_first_normal_v8_to_v9_commit_is_lazy_then_publishes_canonical_attention`; `tests/test_r14_wal_recovery.py::test_v9_crash_before_internal_commit_recovers_prior_v8_without_replay`; `tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay`; `tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_restores_exact_attention_history_and_r13_ports`; `tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_to_retained_v8_bootstraps_attention_and_keeps_r13`; `tests/test_r14_wal_recovery.py::test_raw_v9_attention_tampering_is_rejected_before_wal_repair_or_truncation`; `tests/test_agent_state_v9.py::test_v9_requires_a_closed_attention_field_and_root_event_fences`; `tests/test_attention_production.py::test_read_view_preflight_failure_precedes_internal_journal_and_wal_commit`; `tests/test_attention_production.py::test_published_snapshot_failure_keeps_prior_view_and_restarts_from_v9_focus` |
| `restart_no_replay` | Mixed v8/v9 WAL reconstruction; production v9 restart without Attention/Metacognition replay; exact local restore; owner-required v9 AgentState restore. | `tests/test_r14_wal_recovery.py::test_mixed_v8_v9_v9_wal_reconstructs_exact_attention_roots_and_receipts`; `tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay`; `tests/test_r14_wal_recovery.py::test_canonical_v9_save_before_journal_completion_recovers_without_policy_replay`; `tests/test_attention_system_boundaries.py::test_canonical_restore_and_local_restore_do_not_replay_policy`; `tests/test_agent_state_v9.py::test_v9_alias_round_trip_uses_canonical_attention_and_requires_owner` |
| `privacy` | Attention durable root; structural identity redaction; request-scoped payload; U3 text exclusion; production current-source selection. | `tests/test_attention_system.py::test_snapshot_and_selected_view_reads_are_detached_and_raw_payloads_stay_out`; `tests/test_attention_prompt.py::test_current_selected_rows_are_revalidated_redacted_and_byte_truthful`; `tests/test_r14_u5_scope.py::test_u5_payload_is_ephemeral_sealed_and_has_distinct_byte_witnesses`; `tests/test_metacognition_assessment_boundaries.py::test_source_text_is_not_retained_or_scored_and_assessment_is_pure`; `tests/test_attention_production.py::test_production_chat_refreshes_and_selects_from_current_sources_before_prompt` |
| `size_capacity` | Full 4192-source universe; derived schema/reserve; codec full-universe retention; v9 exact bound; production source capture; whole-row Attention omission. | `tests/test_attention_bounds.py::test_full_universe_fixture_covers_all_bounded_schema_fields`; `tests/test_attention_bounds.py::test_attention_state_value_and_exact_v9_reserved_projection_fit`; `tests/test_attention_state_codec.py::test_full_legal_candidate_universe_is_preserved_without_truncation`; `tests/test_agent_state_v9.py::test_first_v9_capacity_is_derived_from_the_exact_attention_field_bound`; `tests/test_attention_main_loop.py::test_real_main_loop_projects_the_complete_4192_candidate_universe`; `tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation` |
| `compatibility_non_invasion` | R03 FIFO; R04-R06 durability; R07 transactions; R09 Context; R10 appraisal; R11 Value; R12 API; R13 Sleep candidate; retained v8; true rollback to v8. | `tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences`; `tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable`; `tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value`; `tests/test_chat_context.py::test_contextual_working_memory_uses_compatibility_without_mutating_authority`; `tests/test_appraisal.py::test_appraisal_is_pure_deterministic_and_does_not_consume_private_text`; `tests/test_value_system.py::test_prompt_projection_is_immutable_and_excludes_authority_metadata`; `tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak`; `tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active`; `tests/test_agent_state_v9.py::test_v8_schema_and_capture_remain_explicitly_legacy`; `tests/test_r14_wal_recovery.py::test_true_rollback_from_v9_to_retained_v8_bootstraps_attention_and_keeps_r13` |
| `producer_authority_boundary` | Private owner; actual serialized same-turn event; no raw U3 assessment input/call; forged Context rejection; production capture; closed source inventory. | `tests/test_attention_main_loop.py::test_main_loop_owns_an_isolated_attention_system_and_concrete_state_port`; `tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt`; `tests/test_r14_u5_scope.py::test_u5_refresh_source_capture_has_no_raw_authority_parameters`; `tests/test_r14_u5_scope.py::test_u5_producer_rejects_caller_forged_context_even_on_actual_worker`; `tests/test_attention_production.py::test_production_chat_refreshes_and_selects_from_current_sources_before_prompt`; `tests/test_r14_evidence_matrix.py::test_source_inventory_and_unwired_assessment_boundary_are_closed` |
| `same_turn` | Ordinary/debug refresh before prompt; production source capture; pre-durability read-view preflight; exact runtime event identity across Attention/Belief/Goal. | `tests/test_attention_main_loop.py::test_ordinary_and_debug_handlers_refresh_from_complete_current_sources_before_prompt`; `tests/test_attention_production.py::test_production_chat_refreshes_and_selects_from_current_sources_before_prompt`; `tests/test_attention_production.py::test_read_view_preflight_failure_precedes_internal_journal_and_wal_commit`; `tests/test_agent_runtime_ordered_time.py::test_real_chat_attention_belief_and_goal_keep_runtime_event_identity` |
| `deterministic_focus` | Total order under input permutation; identical published continuity; global arousal cap without score rewrite; bounded challenger switch. | `tests/test_attention_policy.py::test_ties_and_input_permutations_have_canonical_total_order_and_digest`; `tests/test_attention_system.py::test_tied_scores_and_input_permutations_produce_identical_state_and_views`; `tests/test_attention_system.py::test_high_global_arousal_only_caps_focus_and_does_not_change_candidate_scores`; `tests/test_attention_system.py::test_salient_challenger_eventually_switches_after_repeat_focus_habituation` |
| `prompt_omission` | Whole oversized row omission; empty fixed frame; selected contribution bound; no combined legacy R13 fallback. | `tests/test_attention_prompt.py::test_oversized_selected_row_is_omitted_whole_without_repacking_or_truncation`; `tests/test_attention_prompt.py::test_empty_focus_renders_the_exact_fixed_frame_and_no_rows`; `tests/test_r14_u5_scope.py::test_u5_document_limits_are_selected_contribution_not_whole_prompt`; `tests/test_attention_prompt.py::test_attention_builder_rejects_combined_legacy_r13_fallback` |
| `no_scheduler` | Closed process-local Attention API/no scheduler import; U1 no later authority imports; explicit Sleep candidate does not become active. | `tests/test_attention_system_boundaries.py::test_full_schema_fixture_fits_value_bound_and_system_import_stays_local`; `tests/test_r14_u1_scope.py::test_u1_modules_do_not_import_runtime_providers_or_later_authorities`; `tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active` |
| `runtime_time` | Reentrant/FIFO admission order; nondecreasing request times; backward-clock refusal; actual runtime identity carried through Attention/Belief/Goal. | `tests/test_agent_runtime_ordered_time.py::test_reentrant_admission_is_refused_before_it_can_invert_fifo_time`; `tests/test_agent_runtime_ordered_time.py::test_runtime_fifo_admission_has_nondecreasing_request_times`; `tests/test_agent_runtime_ordered_time.py::test_backward_clock_refusal_preserves_admission_boundary_and_sequence`; `tests/test_agent_runtime_ordered_time.py::test_real_chat_attention_belief_and_goal_keep_runtime_event_identity` |

| `whole_stage_integration` | Test-only live-turn Metacognition boundary; exact v9 restart/no replay; restored-root U5 producer refusal for earlier time without mutation; transient R08 challenge and unchanged R13 focus restoration at normal/high arousal. The time-refusal case is not full HTTP rollback or automatic migration. | `tests/test_r14_integration.py::test_test_harness_metacognition_observes_the_exact_live_r14_turn_only`; `tests/test_r14_integration.py::test_v9_restart_recovers_without_attention_or_cognition_replay`; `tests/test_r14_integration.py::test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation`; `tests/test_r14_focus_regressions.py::test_transient_r08_challengers_displace_then_restore_real_r13_focus` |

## Bounded caveats and out-of-scope claims

- Source completeness/coherence is a trusted producer precondition. U1/U2 pure
  adapters/owners validate the supplied snapshot tuple, but cannot independently
  enumerate separate stores or prove their capture was atomic. U5 is the current
  serialized capture path; the finite runtime watermark does not persist across
  restarts.
- Attention's own revision and source-local revisions occupy independent
  namespaces. At an unchanged source revision, primary source digest and upstream
  event identity must match; event-scoped projection signals such as deadline
  urgency or Context may change independently without mutating source truth.
- U3 assessment is not production-wired. U3 never emits `SUFFICIENT`; all-unknown
  evidence keeps numeric fields unknown, and confidence needs a typed supporting
  ceiling plus observed link/quality facts. The terms unknown, contradiction,
  and overload remain distinct; missingness, structured contradiction, and
  load/saturation are not interchangeable.
- U5's **131072** bytes govern selected contribution, not the whole prompt. A
  selected row is exact or omitted whole; R08/other sources are not truncated to
  satisfy Attention's limit. Structural Working Memory IDs are redacted from
  display while natural identifier strings in source text remain unchanged.
- `test_restored_attention_root_rejects_earlier_new_runtime_time_without_mutation`
  exercises the U5 producer-only refresh path on an already restored root. It
  is not evidence of full HTTP rollback or automatic migration.
- Checksums/digests establish internal binding, not producer authenticity,
  authorization, or a security proof. No security audit, penetration test,
  complete threat proof, latency SLA, or live-model/performance benchmark is
  claimed.
- Explicit finite Sleep/QLoRA candidate registration is not an Attention
  scheduler, wake mode, Outbox, or R19#279 delivery. Frontend issue #280 is
  separate; this work adds no frontend/UI contract.
- Test names below are evidence locators, not counts or results. U6's matrix
  consistency check does not replace the whole-stage integration/adversarial
  review, whole-R14 review, Human exact-head review, or head-associated CI.

## Reproducible verification commands

Run from this R14 worktree. Executed results are recorded separately below:

```sh
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_r14_integration.py tests/test_r14_focus_regressions.py tests/test_r14_evidence_matrix.py -q
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_attention*.py tests/test_metacognition*.py tests/test_r14*.py tests/test_agent_state_v9.py tests/test_agent_runtime_ordered_time.py -q
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_r14_evidence_matrix.py tests/test_r14_focus_regressions.py tests/test_agent_runtime_ordered_time.py -q
env -u SUZKA_CONFIG_PATH just check-all
git diff --check
git diff --exit-code 8c63d21501d09fe843694f6cc8cd5cfe40087892 -- suzka frontend config.yaml pyproject.toml uv.lock AGENTS.md Justfile .github
```

### Evidence author verification record

- Matrix pytest: **NOT RUN by this evidence author**.
- Ruff: **NOT RUN by this evidence author**.
- Full `just check-all`: **NOT RUN by this evidence author**.
- `git diff --check`: **NOT RUN by this evidence author**.
- The delegated evidence author had no command tool. The NOT RUN records above
  describe that leaf only; primary/verifier execution below is separate evidence.
  No pass/count is inferred from existing test names or accepted prior-unit
  reports. The matrix test performs no Git or subprocess operation.

## Parent-owned whole-stage execution addendum

### New U6 evidence and scope

Exactly four new files: this document, `tests/test_r14_evidence_matrix.py`,
`tests/test_r14_integration.py`, and `tests/test_r14_focus_regressions.py`.
No accepted U1–U5 source, existing tests/reports, schema, policy, codec, WAL,
recovery, API, frontend, dependency lock, or validation workflow is changed.
The unchanged-source command above returned exit 0. Primary inspected status;
the independent verifier also found only these four untracked U6 additions.

The new cases are **5 behavioral cases + 6 checked-index/scope cases = 11**:

- Test-harness-only U3 observation/assessment from the actual current serialized
  MainLoop turn: exact Runtime/Attention event, current R09/R12/R13 roots,
  nonempty complete focused-R13 record set, unknown absent Belief link/ceiling,
  forged Context/event/non-focused-record refusal, and no source/root/view/payload
  mutation or assessment persistence/feedback. The fixture's Belief authority is
  empty; populated linked Belief behavior remains exercised by the indexed U3
  tests. This does not deliver a production U3 producer.
- Real production v9 restart with sentinels forbidding Attention competition,
  U3 observer **and** assessor, model generation, prompt building, and retrieval;
  canonical snapshot bytes/focus survive and transient witnesses stay absent.
- Earlier current admission time on a restored later historical v9 Attention
  root: exact U5 producer refresh refuses without state/view/source mutation.
  This proves the residual process-local boundary, not HTTP-wide rollback,
  a persisted watermark, or automatic clock migration.
- Real transient R08 challenger displacement/restoration of unchanged current
  R13 Goal/Commitment, fixed switch cost, bounded unfinished refs, complete
  candidate retention, and exact retry at both normal and high arousal. No
  absolute source-kind priority, eternal focus, or per-kind fairness is asserted.
- The checked index validates all **F1–F22**, all **A1–A24**, **12** crosscutting
  keys, classifications, exact document rows, valid AST test-node references,
  closed target/adapter inventory, unwired U3, and executable capacity constants.

### Executed local checks (primary and independent verifier)

| Check | Executed result |
| --- | --- |
| New U6 integration/focus/matrix suites | **11 PASS** |
| All R14 unit/integration/adversarial/scope suites plus v9 and runtime-time regressions | **327 PASS** in 458.11 s |
| Final matrix after downgrade locator closure | **6 PASS** |
| Independent matrix/focus/runtime-time set | **14 PASS** |
| `env -u SUZKA_CONFIG_PATH just check-all` | **1912 backend PASS**, Ruff PASS, Mypy **118 source files PASS**, Frontend **6 tests/build PASS** |
| Independent `uv run ruff check suzka tests` / `uv run mypy suzka` | **PASS / PASS (118 files)** |
| `git diff --check` and unchanged-source comparison to accepted U5 | **PASS / exit 0** |

Local Python **3.13.12**. Full backend completed in **566.86 s**; durations are
execution records, not a latency SLA or benchmark. The full gate also reruns
R03–R13 compatibility, transactions, legacy recovery, privacy, and finite Sleep
tests. Earlier U1–U5 counts are historical and not substituted for these runs.
Initial U6 matrix failures exposed wording/reference mismatches and were fixed;
no production failure was suppressed or accepted test relaxed.

### Independent review / publication boundary

Source-only architecture review found potentially vacuous focused-R13 coverage;
primary added a nonempty/complete-count assertion and reran the integration case.
Whole-R14/U6 source review found missing ordinary-v9-downgrade locators; F20 and
`recovery_crash` now index both WAL no-mutation and pre-Journal-prepare refusal,
distinct from explicit true rollback. Narrow re-review reported **no findings**.
Source reviews are not executed tests, and neither is Human acceptance.

Exact U6 head, CI run, actual PR checkout/parents, checkout-versus-head file
comparison, and current external secrets-check result will be recorded in
Draft PR #282 after publication. Prior U5 CI is historical only. This committed
document intentionally does not invent its own future commit SHA or CI result.
Frontend advisories remain separately tracked by **#280**; no dependency changes
or whole-system security claim. R19 wake/scheduling remains **#279**.

## Remaining review items

1. Human U6 focused review and whole-R14 exact-head review remain pending. The
   A/F summaries do not supersede the linked Acceptance Contract or accepted
   unit policies; the evidence index alone is not a feature-completion verdict.
2. Confirm new-head CI and actual checkout provenance in the PR handoff before
   the final Human decision; do not substitute previous U5 CI.
3. Draft PR #282 stays Draft. Ready/Merge remain Human-owned and are not approved
   by these implementation, review-preparation, or execution records.
