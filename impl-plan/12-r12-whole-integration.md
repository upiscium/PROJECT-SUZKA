# U6 / R12 whole-R12 integration

U6 is a closure and integration unit. It creates no state authority, producer,
transaction protocol, or later-roadmap domain. Existing authorities remain
responsible for their own mutation, persistence, and recovery contracts.

## Responsibility boundary

| Authority | Owns | Must not own |
|---|---|---|
| `AgentRuntime` | event ordering, active event identity, lifecycle | domain history or truth |
| `AgentState` / `StateWAL` | bounded intrinsic state and exact recovery | Experience/Semantic history or adoption |
| `ExperienceStore` | committed first-person Experience evidence | transcript bodies, Belief, or Value truth |
| `SemanticStore` | Semantic content lifecycle, revision, and provenance | Context, Belief, or Value truth |
| DB1 | physical episodic Memory projection | Experience lifecycle authority |
| DB2 | physical Semantic search projection | Semantic lifecycle authority |
| `ContextRegistry` | Context identity and compatibility | Memory provenance or truth |
| `BeliefSystem` | explicitly admitted propositions and revisions | Memory/external-claim truth or automatic adoption |
| `ValueSystem` | Value origin, admission, and mutation | Experience-driven self-admission |
| `EventJournal` | transaction/crash/recovery evidence | R12 payload history or domain authority |

Retrieved Memory is **stored evidence**, not an adopted Belief or guaranteed
current fact. Prompt construction labels episodic and semantic material as
stored evidence and does not add an Active Beliefs section. U6 adds no
automatic Belief producer and no direct-adoption API.

Experience references may be carried as opaque evidence by the existing Value
contract, but an Experience reference, frequency, salience, appraisal, or
emotion field cannot construct a `self_endorsed` Value. ValueSystem remains the
only Value mutation authority.

## F1–F20 evidence matrix

The executable index and AST existence checks are in
`tests/test_r12_whole_integration.py`. Every row below names the concrete
evidence used by the whole-R12 review.

| ID | Invariant | Evidence |
|---|---|---|
| F1 | external/user evidence does not adopt Belief | `tests/test_belief_contract.py::test_external_evidence_does_not_adopt_a_belief`; `tests/test_fastapi_backend.py::test_repeated_real_memory_and_experience_reads_do_not_mutate_authority` |
| F2 | repeated Semantic retrieval and ExperienceStore reads do not mutate authority | `tests/test_fastapi_backend.py::test_repeated_real_memory_and_experience_reads_do_not_mutate_authority` |
| F3 | one committed CHAT produces at most one Experience | `tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak`; `tests/test_experience_participant.py::test_first_create_publication_crash_restarts_from_pending_to_exact_one_revision` |
| F4 | Experience event/episode/Context evidence is exact and bounded | `tests/test_experience_contract.py::test_experience_revision_evidence_and_event_binding_are_strict` |
| F5 | timer/debug paths do not become Experience producers | `tests/test_fastapi_backend.py::test_direct_runtime_submit_uses_public_chat_live_authority`; `tests/test_fastapi_backend.py::test_chat_and_emotion_tick_share_fifo_durable_order`; `tests/test_emotion_timer.py::test_timer_does_not_create_duplicate_producers_or_cancel_accepted_work`; `tests/test_r12_whole_integration.py::test_no_cross_boundary_producer_or_direct_adoption_api` |
| F6 | X/Y provenance retains both source Context lineages | `tests/test_semantic_context_projection.py::test_new_format_projection_distinguishes_context_shapes_and_uses_mean` |
| F7 | X/X, X/Y, X/unknown, all-unknown, source-less remain distinct | `tests/test_semantic_context_projection.py::test_new_format_projection_distinguishes_context_shapes_and_uses_mean`; `tests/test_semantic_context_projection.py::test_projection_preserves_incomplete_source_statuses` |
| F8 | order and duplicate sources do not alter identity/weight | `tests/test_semantic_context_projection.py::test_projection_is_canonical_for_duplicate_order_and_preserves_status` |
| F9 | missing/retracted/superseded evidence remains explicit | `tests/test_semantic_context_projection.py::test_projection_preserves_incomplete_source_statuses` |
| F10 | Semantic correction preserves immutable revision history | `tests/test_semantic_lifecycle_contract.py::test_revision_digest_is_order_independent_and_immutable` |
| F11 | legacy reads do not invent provenance or read DB1 | `tests/test_semantic_context_projection.py::test_legacy_contextual_resolver_never_reads_db1` |
| F12 | multi-Context projection is pure and zero-write | `tests/test_semantic_context_projection.py::test_persisted_new_format_repeated_reads_are_pure_and_never_read_db1` |
| F13 | conflict candidates do not mutate Belief | `tests/test_belief_contract.py::test_conflict_candidate_is_a_pure_hint_only`; `tests/test_r12_whole_integration.py::test_conflict_candidates_do_not_mutate_belief` |
| F14 | non-active Beliefs are excluded from ordinary active view | `tests/test_belief_system.py::test_successful_terminal_mutations_leave_the_ordinary_active_view` |
| F15 | Belief restores without cognition/model replay | `tests/test_agent_state.py::test_v6_round_trip_preserves_intrinsic_belief_authority_without_replay`; `tests/test_state_wal.py::test_v6_wal_reconstructs_nonempty_belief_without_replay` |
| F16 | Experience/Semantic crash boundaries follow R07 | `tests/test_experience_participant.py::test_revision_prepare_writes_pending_and_restart_rolls_forward`; `tests/test_semantic_participant.py::test_partial_lifecycle_publication_restarts_from_pending_batch`; `tests/test_startup_reconciliation.py::test_internal_commit_rolls_forward_without_state_replay` |
| F17 | Experience cannot manufacture or implicitly mutate Value authority | `tests/test_r12_whole_integration.py::test_experience_reference_cannot_self_endorse_value`; `tests/test_fastapi_backend.py::test_opaque_experience_evidence_and_proposal_do_not_mutate_value_authority`; `tests/test_identity_origin.py::test_external_actors_cannot_construct_active_admissions` |
| F18 | private/raw content is rejected or remains ephemeral | `tests/test_experience_contract.py::test_experience_record_is_immutable_bounded_and_rejects_raw_content_fields`; `tests/test_agent_state.py::test_canonical_snapshot_contains_no_private_or_independent_store_data`; `tests/test_main_loop.py::test_debug_trace_exposes_private_thought_only_ephemerally` |
| F19 | malformed/future/overflow state fails closed across Experience, Semantic, and Belief authorities | `tests/test_experience_contract.py::test_experience_record_is_immutable_bounded_and_rejects_raw_content_fields`; `tests/test_experience_store.py::test_store_rejects_malformed_and_symlink_artifacts`; `tests/test_experience_store.py::test_store_rejects_future_schema_artifact`; `tests/test_experience_store.py::test_retention_is_exact_anchored_and_idempotent`; `tests/test_semantic_store.py::test_revision_serialization_rejects_digest_tampering`; `tests/test_semantic_store.py::test_store_rejects_unknown_revision_artifact`; `tests/test_semantic_store.py::test_store_rejects_future_schema_artifact`; `tests/test_semantic_store.py::test_typed_empty_record_directory_requires_pending_create_evidence`; `tests/test_semantic_store.py::test_revision_round_trip_and_exact_current_plus_32_retention`; `tests/test_belief_system.py::test_total_record_capacity_fails_closed_without_eviction`; `tests/test_agent_state.py::test_belief_schema_budget_is_derived_from_all_bounded_fields`; `tests/test_belief_system.py::test_restore_rejects_missing_or_extra_latest_revision_witnesses` |
| F20 | R03–R11 and public chat compatibility remains intact | `tests/test_main_loop.py::test_ordinary_and_debug_chat_use_working_memory_without_prompt_mutation`; `tests/test_fastapi_backend.py::test_api_chat_works_with_dummy_provider_without_debug_leak`; `tests/test_agent_runtime.py::test_fifo_order_and_consumer_sequences` |

## Crash/restart matrix

The consolidated matrix is `RECOVERY_CRASH_MATRIX` in
`tests/test_r12_whole_integration.py`; it indexes these reviewed boundaries:

| Boundary | Required result | Evidence |
|---|---|---|
| CHAT before internal commit | no committed Experience | `tests/test_agent_runtime.py::test_handler_failure_skips_checkpoint` |
| CHAT pending Experience publication | exact roll-forward/reuse | `tests/test_experience_participant.py::test_revision_prepare_writes_pending_and_restart_rolls_forward` |
| CHAT Experience publication crash | exactly one revision | `tests/test_experience_participant.py::test_first_create_publication_crash_restarts_from_pending_to_exact_one_revision` |
| DEBUG_CHAT / EMOTION_TICK | ordered runtime behavior, no new producer | `tests/test_fastapi_backend.py::test_chat_and_emotion_tick_share_fifo_durable_order` |
| SLEEP before semantic commit | visible Semantic only, no model replay | `tests/test_r12_semantic_integration.py::test_sleep_persists_visible_semantic_only_and_never_reruns_model` |
| SLEEP model/handler failure before internal commit | no Semantic authority, receipt, or DB2 projection | `tests/test_fastapi_backend.py::test_sleep_handler_failure_before_internal_commit_leaves_no_semantic_authority` |
| Semantic lifecycle pending | deterministic roll-forward | `tests/test_semantic_participant.py::test_partial_lifecycle_publication_restarts_from_pending_batch` |
| DB2 missing/stale | reconcile from Semantic authority | `tests/test_r12_semantic_integration.py::test_terminal_startup_reconciles_missing_projection_from_authority` |
| divergent DB2 | fail closed/no authority overwrite | `tests/test_semantic_participant.py::test_divergent_legacy_projection_is_not_overwritten` |
| Semantic receipt without lifecycle authority | fail closed without receipt/projection mutation | `tests/test_r12_semantic_integration.py::test_retained_receipt_with_deleted_lifecycle_fails_closed_everywhere` |
| AgentState/Belief v6 | exact state without inference | `tests/test_state_wal.py::test_v6_wal_reconstructs_nonempty_belief_without_replay` |
| true rollback | intrinsic rollback without destructive external rollback; ExperienceStore and new-format SemanticStore history bytes remain unchanged | `tests/test_startup_reconciliation.py::test_true_rollback_restores_working_memory_only_and_preserves_newer_episodic` |
| malformed/future/overflow Belief | fail closed | `tests/test_agent_state.py::test_v6_nonempty_belief_restore_requires_intrinsic_authority` |

## Privacy and compatibility

The `PRIVACY_SENTINEL_MATRIX` and `COMPATIBILITY_NON_INVASION_MATRIX` in the
U6 suite index the following boundaries:

- raw user/assistant/private thought/provider errors are rejected or ephemeral
  in Experience, AgentState, WAL, Journal, and public response projections;
- `tests/test_state_wal.py::test_private_sentinel_and_bounded_errors` and
  `tests/test_event_journal.py::test_u1_f11_invalid_transaction_fields_are_bounded_and_private_free`
  directly exercise WAL and Journal private-payload rejection;
- explicitly selected visible Semantic content is allowed in SemanticStore/DB2
  as Memory content, but raw prompt/private intermediate data is not;
- R03 event ordering, R04–R06 state/WAL recovery, R07 participant protocol,
  R08 WorkingMemory, R09 Context, R10 appraisal/emotion, R11 Value, public
  chat, and sleep/training regressions remain exercised by their owning tests;
- no U6 read path mutates Memory, Context, WorkingMemory, Experience, Semantic,
  Belief, Value, AgentState, WAL, or Journal state.

## D7 and integration bookkeeping

The schema-derived maximum Belief section is **47,541,622 bytes (45.34 MiB)**.
The Belief runtime cap remains **64 MiB** and the whole AgentState cap remains
**128 MiB**. BeliefSystem owns mutation/lifecycle authority; AgentState and
StateWAL own bounded persistence continuity only. The D7 bound remains below
the runtime cap, so no authority extraction is introduced by U6.

Issue **#263** (multi-Context Semantic provenance) and Issue **#265**
(durable appraisal/emotion Experience evidence) remain open for Human close
bookkeeping. Their implementation and focused review obligations are complete;
U6 records their whole-R12 integration evidence but does not close them.

No R13+ authority is introduced: no Goal/Commitment, Attention, Relationship,
SelfModel, Decision, Action, Scheduler/Outbox, training, or adapter authority
is part of U6.
