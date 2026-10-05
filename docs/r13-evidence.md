# R13 F1–F22 evidence map

> **Evidence index, not a readiness declaration.** This U6 map is awaiting
> review. It does not declare R13 complete, and it does not claim any deferred
> producer is implemented.

## Scope and evidence boundary

R13 provides bounded Motivation, Goal, and Commitment domain contracts and
process-local authority systems; the lossless R13 codec and AgentState v8
continuity; and a minimal, immutable prompt projection of current state. The
MainLoop owns private dependency-injected system instances and explicit state
ports, clones injected authorities, and supplies the committed projection to a
compatible prompt builder. The projection is not an admission, selection, or
execution authority.

R13 does **not** add production Motivation evidence updates from model or
Experience events, producer routing for Goal or Commitment proposals, Goal
admission or Commitment acceptance producers, verified outcome producers,
automatic suspension/release, a winner selector, or a scheduler. The focused
system tests exercise domain contracts using explicit inputs; they are not
evidence that an absent production route exists. “Deferred” below means not
implemented in this scope.

The prompt projection includes every current active Motivation, adopted Goal,
and active Commitment, in canonical identity order, up to the projection's
per-domain bound of 32. It does not rank or truncate active records. Its
minimal schema omits evidence, histories, proofs, outcomes, and the Commitment
subject. Projection and rendering are bounded by the schema-derived serialized
and rendered byte limits in `suzka/motivation/projection.py`.

| Projection bound | Maximum |
| --- | ---: |
| Records per domain | 32 |
| Canonical JSON | 3,576,318 |
| Rendered UTF-8 | 3,576,433 |

The byte maxima include every entry field, JSON separators/field names, and
rendered headings/authority labels. They are conservative schema-derived
envelopes, not typical payload measurements. The full-count Unicode test
retains 32 active records in each domain, 1,024 supplementary codepoints per
Goal description, 32 distinct 256-codepoint scope items per Commitment, and
actual maximum UTC deadlines (`9999-12-31T23:59:59.999999Z`). No active entry or
supported field is shortened to satisfy the bound. The documentation bounds
are checked against the executable derivation.

Value, Belief, and the coherent three-domain R13 projection are prepared from
the captured candidate before durable commit, then published with one immutable
read-bundle reference replacement. Whole-AgentState restore likewise stages
the bundle before mutation and publishes only after every restore succeeds.
Individual state ports do not publish speculative reads. This does not make
multiple separate public read calls one cross-call transaction: a reader
spanning a commit may observe different committed generations.

### AgentState v7/v8 behavior and retained capacity

An existing v7 snapshot remains byte-preserving during load/startup and WAL
bootstrap; there is no eager v8 rewrite. Legacy pre-v8-only runtimes may still
capture v7. The production MainLoop now always provides Belief and all three
R13 ports, so its capture is v8 even when R13 is empty. Its first ordinary
successful state commit publishes v8 and retains its v7 lineage. Every v8
restore requires complete R13 ports; an ordinary v8-to-v7 downgrade is rejected.
Explicit true rollback to retained v7 remains legal and clears R13 exactly.

The U5 v8 schema maximum remains **126,350,904 bytes**, consisting of the
109,573,688-byte v8 base maximum plus its **16 MiB future-state reserve**.
R13 does not change this capacity or consume that reserved headroom. The
checked-in capacity test derives the same value from the three R13 codec
maxima and asserts the reserve explicitly.
The hard cap remains 134,217,728 bytes, with 7,866,824 bytes of margin beyond
the future-state reserve. The prompt view is not a new persisted authority.

## F1–F22 classification and evidence

`DIRECT_R13_PASS` records direct tests of the stated R13 contract;
`REPRESENTATIONAL_NO_PRODUCER` records only the existing representation or
explicit absence boundary; `DEFERRED` identifies a later owner and is not a
claim that its producer exists. Every evidence link names an executable test
node. The test suite checks this table against its matrix and resolves every
link to a test function in the checked-in tree.

| Requirement | Classification | Deferred owner | Claim boundary | Executable evidence |
| --- | --- | --- | --- | --- |
| F1 | DIRECT_R13_PASS | — | Repeating an external request leaves the Goal proposed and does not create adoption. | `tests/test_goal_system.py::test_repeated_external_request_does_not_turn_proposal_into_adoption` |
| F2 | DIRECT_R13_PASS | — | Operator proposals are proposals; the Goal authority does not auto-adopt based on origin. | `tests/test_goal_system.py::test_request_operator_external_and_motivation_origins_never_auto_adopt`; `tests/test_goal_system.py::test_ingestion_retains_only_proposal_and_exact_replay_is_idempotent` |
| F3 | REPRESENTATIONAL_NO_PRODUCER | — | An Experience projection can form typed Motivation evidence, but production Experience-to-Motivation routing and any proposal/adoption route are absent. Motivation candidates alone confer no authority; a Motivation-origin Goal stays proposed. | `tests/test_motivation_system.py::test_experience_factory_uses_bounded_appraisal_and_keeps_event_origin`; `tests/test_motivation_system.py::test_candidate_alone_has_no_motivation_authority`; `tests/test_goal_system.py::test_request_operator_external_and_motivation_origins_never_auto_adopt` |
| F4 | DIRECT_R13_PASS | — | Strong-enough Motivation state produces a deterministic, bounded Goal proposal with the exact eligibility event; it remains admission-free. This is a domain operation, not production routing. | `tests/test_motivation_system.py::test_goal_proposal_uses_the_exact_first_threshold_eligibility_event`; `tests/test_motivation_system.py::test_goal_proposal_is_bounded_deterministic_proposed_and_admission_free`; `tests/test_motivation_system.py::test_high_salience_evidence_stays_within_per_event_record_and_goal_budgets` |
| F5 | DIRECT_R13_PASS | — | External origin remains typed and cannot be relabelled on replay; external origin is not an admission. | `tests/test_motivation_system.py::test_external_origin_is_retained_and_cannot_be_relabelled_on_replay`; `tests/test_goal_system.py::test_request_operator_external_and_motivation_origins_never_auto_adopt` |
| F6 | DIRECT_R13_PASS | — | Same opaque identifiers under distinct source kinds remain distinct, and snapshot receipts bind exact typed provenance. | `tests/test_motivation_system.py::test_same_opaque_id_under_different_source_kinds_remains_distinct`; `tests/test_motivation_system.py::test_snapshot_rejects_evidence_provenance_not_bound_by_its_receipt`; `tests/test_r13_codec.py::test_motivation_table_keeps_candidate_evidence_eligibility_and_typed_refs` |
| F7 | DIRECT_R13_PASS | — | Conflicting Motivation kinds and conflicting adopted Goals coexist; these checks do not implement a winner selector. | `tests/test_motivation_system.py::test_conflicting_motivations_coexist_without_winner_selection`; `tests/test_goal_system.py::test_conflicting_adopted_goals_coexist_without_ranking_or_suspension` |
| F8 | DIRECT_R13_PASS | — | Decay, review, satiation, and retained history have deterministic event-bound revisions and bounded compaction without losing replay evidence. | `tests/test_motivation_system.py::test_decay_review_and_satiation_are_deterministic_and_event_bound`; `tests/test_motivation_system.py::test_motivation_history_compacts_without_losing_evidence_idempotency`; `tests/test_goal_system.py::test_subject_proofs_and_receipts_survive_bounded_revision_compaction` |
| F9 | DIRECT_R13_PASS | — | Proposal ingestion and adoption are separate; only an exact caller-supplied Goal admission can adopt. | `tests/test_goal_system.py::test_ingestion_retains_only_proposal_and_exact_replay_is_idempotent`; `tests/test_goal_system.py::test_only_exact_caller_supplied_admission_can_adopt` |
| F10 | DEFERRED | R16 | Winner selection is deferred to R16. Current tests establish coexistence and absence of ranking APIs, not a selection policy. | `tests/test_motivation_system.py::test_conflicting_motivations_coexist_without_winner_selection`; `tests/test_goal_system.py::test_conflicting_adopted_goals_coexist_without_ranking_or_suspension`; `tests/test_goal_system.py::test_graph_conflict_facts_and_snapshot_export_are_exact_and_side_effect_free` |
| F11 | DIRECT_R13_PASS | — | Conflicting adopted Goals remain active together; invalid missing/cyclic dependency facts fail closed rather than triggering auto-suspension. | `tests/test_goal_system.py::test_conflicting_adopted_goals_coexist_without_ranking_or_suspension`; `tests/test_goal_system.py::test_missing_dependency_and_dependency_cycle_fail_closed` |
| F12 | DIRECT_R13_PASS | — | User, operator, external-request, and Goal proposal origins do not activate Commitment responsibility. | `tests/test_commitment_system.py::test_proposal_sources_cannot_activate_responsibility` |
| F13 | DIRECT_R13_PASS | — | Activating responsibility consumes the exact caller-supplied admission and event; the system does not create the admission. | `tests/test_commitment_system.py::test_accept_consumes_exact_supplied_proof_without_producing_one`; `tests/test_commitment_system.py::test_wrong_admission_shape_or_event_fails_atomically` |
| F14 | REPRESENTATIONAL_NO_PRODUCER | — | A disappeared/vanished Desire reference or elapsed deadline is not itself a release proof. The test checks references and absence of expiry/tick/auto-release APIs; it does not simulate a production Desire-disappearance event. | `tests/test_commitment_system.py::test_desire_goal_and_deadline_are_references_not_automatic_authority` |
| F15 | DIRECT_R13_PASS | — | Release and renegotiation require fresh, exact transition proofs and do not reopen the prior Commitment. | `tests/test_commitment_system.py::test_fresh_transition_proofs_release_or_renegotiate_without_reopening`; `tests/test_commitment_system.py::test_transition_rejects_wrong_authority_terms_operation_evidence_and_event` |
| F16 | DEFERRED | R17, R18 | Verified outcome producers are deferred to R17/R18. R13 rejects unverified outcome records and has no Goal completion/failure or Commitment fulfillment/breach producer. | `tests/test_commitment_system.py::test_unreachable_compaction_and_unverified_outcome_records_are_rejected`; `tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods`; `tests/test_goal_system.py::test_ingestion_retains_only_proposal_and_exact_replay_is_idempotent` |
| F17 | DIRECT_R13_PASS | — | v8 snapshots, production restart and WAL recover the exact R13 graph without policy replay; restore-port and injected-commit tests cover exact domain replay identity. | `tests/test_r13_wal_recovery.py::test_committed_v8_crash_reconstructs_without_policy_replay`; `tests/test_agent_state_v8.py::test_v8_round_trip_preserves_complete_empty_and_nontrivial_r13_graphs`; `tests/test_r13_restore_ports.py::test_motivation_restore_preserves_exact_graph_and_event_replay`; `tests/test_r13_restore_ports.py::test_goal_restore_preserves_ingestion_admission_replay_and_compaction`; `tests/test_r13_restore_ports.py::test_commitment_restore_preserves_proposal_and_terminal_proof_replay`; `tests/test_r13_integration.py::test_populated_v8_restarts_exactly_and_model_response_never_changes_r13`; `tests/test_r13_integration.py::test_successful_injected_commit_publishes_one_bundle_and_retry_is_idempotent` |
| F18 | DIRECT_R13_PASS | — | Prompt projection omits Commitment subject and proof/history fields; MainLoop projection reads do not mutate authority. Persistence schemas reject private extra fields; ordinary/debug/model responses do not feed back into R13. | `tests/test_r13_projection.py::test_projection_excludes_each_noncurrent_lifecycle_and_all_proofs`; `tests/test_r13_evidence_matrix.py::test_main_loop_r13_ports_are_private_and_prompt_projection_is_pure`; `tests/test_r13_codec.py::test_state_snapshot_requires_all_domains_and_forbids_private_extra_fields`; `tests/test_r13_integration.py::test_standard_prompt_renders_only_minimal_current_r13_authority`; `tests/test_r13_integration.py::test_production_ports_upgrade_retained_v7_on_first_commit_without_feedback` |
| F19 | DIRECT_R13_PASS | — | Domain, codec, prompt, and AgentState capacity checks reject over-bound inputs without truncation; full-count Unicode/max-deadline projection and checked documentation preserve the derived envelope and v8 future reserve. | `tests/test_r13_contract_bounds.py::test_schema_budget_is_reproducible_and_preserves_the_v8_future_reserve`; `tests/test_r13_codec.py::test_capacity_gate_uses_derived_codec_maxima_and_utf8_domain_bytes`; `tests/test_r13_projection.py::test_projection_includes_all_active_records_and_derives_exact_size`; `tests/test_r13_projection.py::test_extreme_unicode_is_escaped_and_output_never_shortens_records`; `tests/test_r13_evidence_matrix.py::test_documented_projection_bounds_match_derived_schema`; `tests/test_motivation_system.py::test_record_capacity_event_budget_and_candidate_capacity_fail_atomically`; `tests/test_goal_system.py::test_full_u1_goal_record_bound_is_preserved_and_next_record_fails`; `tests/test_commitment_system.py::test_full_record_and_receipt_bounds_accept_maximum_reject_one_over` |
| F20 | DIRECT_R13_PASS | — | Existing R03–R12 contracts remain non-invaded; whole restore keeps Value/Belief/R13 public reads committed, and server publication has no fallible per-domain snapshot step after durability. | `tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences`; `tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable`; `tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value`; `tests/test_working_memory.py::test_projection_never_exceeds_utf8_byte_budget`; `tests/test_chat_context.py::test_contextual_working_memory_uses_compatibility_without_mutating_authority`; `tests/test_appraisal.py::test_appraisal_is_pure_deterministic_and_does_not_consume_private_text`; `tests/test_value_system.py::test_prompt_projection_is_immutable_and_excludes_authority_metadata`; `tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak`; `tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active`; `tests/test_agent_state_v8.py::test_v7_belief_capture_stays_v7_and_r13_capture_upgrades_to_v8`; `tests/test_main_loop.py::test_failed_agent_state_restore_keeps_all_previous_committed_views`; `tests/test_fastapi_backend.py::test_server_commit_does_not_call_legacy_view_publishers_after_durability` |
| F21 | DEFERRED | R19 | Scheduler/tick/expiry orchestration is deferred to R19. Current Goal and Commitment authorities explicitly lack scheduler operations. | `tests/test_goal_system.py::test_graph_conflict_facts_and_snapshot_export_are_exact_and_side_effect_free`; `tests/test_commitment_system.py::test_desire_goal_and_deadline_are_references_not_automatic_authority`; `tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods` |
| F22 | REPRESENTATIONAL_NO_PRODUCER | — | R13 does not import or implement later R14–R19 authorities. Production ownership/restart assertions delimit this layer only; they are not evidence that later authorities or their producers exist. | `tests/test_r13_evidence_matrix.py::test_main_loop_r13_surface_has_no_producer_or_later_authority_calls`; `tests/test_r13_integration.py::test_production_mainloop_exposes_no_mutable_r13_read_or_later_producer`; `tests/test_r13_integration.py::test_populated_v8_restarts_exactly_and_model_response_never_changes_r13`; `tests/test_motivation_system.py::test_system_api_import_does_not_load_models_runtime_or_timer_modules`; `tests/test_goal_system.py::test_goal_system_module_has_no_later_runtime_model_or_scheduler_imports`; `tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods` |

## Cross-cutting evidence matrices

The keys below are stable matrix identifiers. Each reference names an
executable test node; the U6 matrix test verifies the keys, links, and this
document's rows together. These references index existing coverage and do not
imply this evidence-only change ran those tests.

| Matrix key | Cases indexed | Executable evidence |
| --- | --- | --- |
| `recovery_crash` | Production v7 startup/first v8; lazy WAL upgrade; unpublished/published-unconfirmed failure; committed crash; malformed/downgrade rejection; true rollback to v7/older v8; preflight and whole-restore failure. | `tests/test_r13_integration.py::test_production_ports_upgrade_retained_v7_on_first_commit_without_feedback`; `tests/test_r13_wal_recovery.py::test_first_normal_v8_commit_after_v7_is_lazy_and_preserves_v7_lineage`; `tests/test_r13_wal_recovery.py::test_failed_first_v8_publication_keeps_canonical_v7_bytes`; `tests/test_r13_wal_recovery.py::test_committed_v8_crash_reconstructs_without_policy_replay`; `tests/test_r13_wal_recovery.py::test_malformed_persisted_v8_snapshot_is_rejected`; `tests/test_r13_wal_recovery.py::test_ordinary_v8_to_v7_wal_transition_is_rejected`; `tests/test_r13_wal_recovery.py::test_internal_commit_rejects_v8_to_v7_before_prepare`; `tests/test_r13_wal_recovery.py::test_true_rollback_from_v8_to_v7_clears_r13_restore_ports`; `tests/test_r13_wal_recovery.py::test_true_rollback_to_older_v8_restores_exact_r13_authority`; `tests/test_r13_integration.py::test_failed_v8_publication_keeps_committed_r13_view_and_restart_authority`; `tests/test_r13_integration.py::test_published_v8_failure_fail_stops_and_restart_recovers_exact_authority`; `tests/test_r13_integration.py::test_projection_preflight_failure_precedes_durable_publication`; `tests/test_main_loop.py::test_failed_agent_state_restore_keeps_all_previous_committed_views` |
| `restart_no_replay` | Production v8 restart/no feedback; exact graph round-trip and mixed WAL; domain restore/replay identity. | `tests/test_r13_integration.py::test_populated_v8_restarts_exactly_and_model_response_never_changes_r13`; `tests/test_agent_state_v8.py::test_v8_round_trip_preserves_complete_empty_and_nontrivial_r13_graphs`; `tests/test_r13_wal_recovery.py::test_mixed_retained_v7_then_v8_wal_reconstructs_exact_snapshots_and_hash_chain`; `tests/test_r13_restore_ports.py::test_motivation_restore_preserves_exact_graph_and_event_replay`; `tests/test_r13_restore_ports.py::test_goal_restore_preserves_ingestion_admission_replay_and_compaction`; `tests/test_r13_restore_ports.py::test_commitment_restore_preserves_proposal_and_terminal_proof_replay` |
| `privacy` | Standard prompt excludes retained witnesses; ordinary/debug chat has no feedback; projection omits proofs/subject; persistence rejects private extras; thought remains ephemeral. | `tests/test_r13_integration.py::test_standard_prompt_renders_only_minimal_current_r13_authority`; `tests/test_r13_integration.py::test_production_ports_upgrade_retained_v7_on_first_commit_without_feedback`; `tests/test_r13_projection.py::test_projection_excludes_each_noncurrent_lifecycle_and_all_proofs`; `tests/test_r13_evidence_matrix.py::test_main_loop_r13_ports_are_private_and_prompt_projection_is_pure`; `tests/test_r13_codec.py::test_state_snapshot_requires_all_domains_and_forbids_private_extra_fields`; `tests/test_main_loop.py::test_debug_trace_exposes_private_thought_only_ephemerally` |
| `size_capacity` | Full 32/domain including maximum Unicode/deadlines; checked projection limits; derived codec/v8 budgets; atomic per-domain overflow. | `tests/test_r13_projection.py::test_projection_includes_all_active_records_and_derives_exact_size`; `tests/test_r13_projection.py::test_extreme_unicode_is_escaped_and_output_never_shortens_records`; `tests/test_r13_evidence_matrix.py::test_documented_projection_bounds_match_derived_schema`; `tests/test_r13_contract_bounds.py::test_schema_budget_is_reproducible_and_preserves_the_v8_future_reserve`; `tests/test_r13_codec.py::test_capacity_gate_uses_derived_codec_maxima_and_utf8_domain_bytes`; `tests/test_agent_state_capacity.py::test_v8_schema_maxima_cover_all_r13_domains_and_keep_future_reserve`; `tests/test_motivation_system.py::test_record_capacity_event_budget_and_candidate_capacity_fail_atomically`; `tests/test_goal_system.py::test_full_u1_goal_record_bound_is_preserved_and_next_record_fails`; `tests/test_commitment_system.py::test_full_record_and_receipt_bounds_accept_maximum_reject_one_over` |
| `compatibility_non_invasion` | R03–R12 FIFO/durability/transaction/projection/API/sleep contracts; v7 compatibility; coherent restore/commit read publication. | `tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences`; `tests/test_state_recovery.py::test_normal_commit_order_and_artifacts_are_durable`; `tests/test_transaction_coordinator.py::test_prepare_and_finalize_are_sorted_and_publish_only_public_value`; `tests/test_working_memory.py::test_projection_never_exceeds_utf8_byte_budget`; `tests/test_chat_context.py::test_contextual_working_memory_uses_compatibility_without_mutating_authority`; `tests/test_appraisal.py::test_appraisal_is_pure_deterministic_and_does_not_consume_private_text`; `tests/test_value_system.py::test_prompt_projection_is_immutable_and_excludes_authority_metadata`; `tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak`; `tests/test_sleep_qlora.py::test_sleep_cycle_registers_candidate_and_never_active`; `tests/test_agent_state_v8.py::test_v7_belief_capture_stays_v7_and_r13_capture_upgrades_to_v8`; `tests/test_main_loop.py::test_failed_agent_state_restore_keeps_all_previous_committed_views`; `tests/test_fastapi_backend.py::test_server_commit_does_not_call_legacy_view_publishers_after_durability` |
| `producer_authority_boundary` | Private cloned ownership; staged view; direct restore isolation; injected failure/retry only; no production producer/later-authority calls. | `tests/test_main_loop.py::test_main_loop_clones_injected_r13_authorities_and_publishes_their_view`; `tests/test_main_loop.py::test_main_loop_r13_view_changes_only_after_staged_publication`; `tests/test_main_loop.py::test_main_loop_direct_r13_restore_ports_do_not_publish_prompt_view`; `tests/test_r13_integration.py::test_failed_injected_domain_operation_rolls_back_without_speculative_view`; `tests/test_r13_integration.py::test_successful_injected_commit_publishes_one_bundle_and_retry_is_idempotent`; `tests/test_r13_integration.py::test_production_mainloop_exposes_no_mutable_r13_read_or_later_producer`; `tests/test_r13_evidence_matrix.py::test_main_loop_r13_surface_has_no_producer_or_later_authority_calls`; `tests/test_motivation_system.py::test_system_api_import_does_not_load_models_runtime_or_timer_modules`; `tests/test_goal_system.py::test_goal_system_module_has_no_later_runtime_model_or_scheduler_imports`; `tests/test_commitment_system.py::test_no_later_authority_imports_or_production_side_effect_methods` |

## Review caveats

- The prompt projection is a read-only current-state rendering boundary; it
  does not prove that external evidence or model output should cause a state
  mutation.
- F3 and F14 are deliberately classified
  `REPRESENTATIONAL_NO_PRODUCER`: their linked tests establish typed data and
  absence of automatic authority, not a production producer.
- F10, F16, and F21 are explicitly deferred to R16, R17/R18, and R19
  respectively. The later selector, outcome producers, and scheduler are not
  present in R13.
- U5 recovery tests establish v7 byte preservation before the first ordinary
  v8 publication and exact reconstruction at tested crash boundaries; they do
  not establish arbitrary filesystem or hardware failure behavior.
- This evidence map remains subject to R13 review. Do not call the whole R13
  lane ready based on the presence of this table or the listed tests.
