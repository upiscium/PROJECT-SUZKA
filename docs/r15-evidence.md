# R15 MVI whole-stage evidence index

Run: R15-U7-MATRIX-20261029-01, instruction rev 1. This is a docs/test-only
index at base commit `4b1dc8486a55a2fd2a8a459912f533e7c27b30f7`, Git tree
`4108b9f89d6250f5a6d375949f9978ff0ed0358b`. It supplements, not replaces,
the U1-era index in `docs/r15-u1-contract.md`. Intent summaries are not new
acceptance clauses. The original broader requirements remain visible below.

## Contract and evidence provenance

- Frozen original A1–A27/F1–F22/D9: [#283 contract comment 6058471359](https://github.com/upiscium/PROJECT-SUZKA/issues/283#issuecomment-6058471359),
  [Human acceptance 6058579682](https://github.com/upiscium/PROJECT-SUZKA/issues/283#issuecomment-6058579682).
- [#323 Human MVI U2 amendment](https://github.com/upiscium/PROJECT-SUZKA/issues/323)
  permits bounded scoped interaction instead of positive lasting Relationship;
  [#324 deferred register](https://github.com/upiscium/PROJECT-SUZKA/issues/324)
  retains original positive Relationship, alias/split, advanced accessibility,
  G2 and other broader work. These Human deferrals are not MVI blockers, but
  the original broader positive requirements are **not proven**.
- [#351 seq19](https://github.com/upiscium/PROJECT-SUZKA/issues/351): explicit Human
  acceptance of bounded two-case U7 evidence, **SCIENTIFIC EVIDENCE ONLY**,
  with semantic caveats. [#283 seq114](https://github.com/upiscium/PROJECT-SUZKA/issues/283)
  is the acceptance-gap audit preceding this index. Sequence labels identify
  coordinator-verified records, not invented comment permalinks. These records
  and the CI result below were coordinator-verified against GitHub records;
  this static index itself does not live-query references. The cited CI applies
  to base HEAD, not the current uncommitted files.
- Full generation identity remains separately unproven under
  [#343](https://github.com/upiscium/PROJECT-SUZKA/issues/343). No repeat model trial.

## Responsibility and interpretation

**Owns:** traceable coverage and bounded evidence descriptions, not intrinsic
state or acceptance. **May:** statically resolve repository test callables and
reject incomplete/misclassified rows. **Must not:** mutate product state, confer
source authority, accept R15, authorize deployment, publish or repeat Gemma4.
**Depends on:** the frozen contract, Human MVI amendment/deferred register and
the inspected base source. **Used by:** a still-required independent whole-stage
review and later Human decision; neither is replaced by this document.

Compatibility/migration impact: none; no source/schema/config changes. Failure
and recovery: the deterministic checker refuses malformed coverage/references;
repair this index rather than changing product behavior to make it green.
Privacy: only source names and bounded summaries here, no trial payloads or
external private artifacts. Evidence links name executable tests, not assertions
that those tests ran in this worktree. U1 pure shapes are not production proof.

Controlled MVI classifications (none means whole-clause PASS):

- `MVI_WITNESSED`: named test witnesses the stated finite MVI behavior only.
- `MVI_NEGATIVE_PROVEN`: named adversarial test checks a specific refusal or
  non-mutation; not proof of every possible bypass or a positive producer.
- `UNAVAILABLE_FAIL_CLOSED`: required positive producer is unavailable; named
  tests witness UNKNOWN/refusal only, never original positive completion.
- `HUMAN_DEFERRED`: #323/#324 removed the broader positive requirement from
  the bounded MVI gate; tests constrain representations/refusal, not fulfillment.
- `PARTIAL_OR_UNVERIFIED`: evidence covers only part of the clause, or the
  requested stage-wide validation/review is still outstanding.

## Acceptance matrix

| ID | Intent | Classification | Exact repository evidence | Evidence boundary | Unresolved or deferred condition |
| --- | --- | --- | --- | --- | --- |
| A1 | Three independent authoritative roots | MVI_WITNESSED | `tests/test_r15_u1_contracts.py::test_r15_closed_roots_roundtrip_is_canonical_and_independent`; `tests/test_r15_u4b_attempt_admission.py::test_real_owner_attempt_commit_restart_and_no_duplicate` | Pure root independence plus finite owned attempt/WAL path, not a monolithic self authority. | Arbitrary cross-root mutations and every future producer are not proven. |
| A2 | R09 participant reference is not person proof | MVI_NEGATIVE_PROVEN | `tests/test_r15_p1b_positive.py::test_identity_boundaries_preserve_unknown_identity`; `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input` | Opaque tokens and UNKNOWN axes; live current-event scoping does not authenticate a person. | Positive identity corroboration remains unavailable. |
| A3 | Qualified alias/split and cross-context admission | UNAVAILABLE_FAIL_CLOSED | `tests/test_r15_u1_contracts.py::test_r15_claimed_identity_is_not_person_proof_or_alias_merge`; `tests/test_r15_u2_live_scoped_interaction.py::test_closed_collision_refuses_inherited_member` | Identity producer unavailable; collision refusal is not alias/split production or atomic dependent-ref proof. | Original positive alias/split is not proven; Human-deferred #323/#324. |
| A4 | Closed upstream source graph | PARTIAL_OR_UNVERIFIED | `tests/test_r15_p1b_positive.py::test_missing_source_and_forged_refs_refuse_without_mutation`; `tests/test_r15_u3_narrative_self_system.py::test_r15_u3_scope_excludes_effectful_sources_and_later_owner_writes` | Finite source-gated Narrative path and structural owner guard, not general semantic admission. | All future source types and advanced G2 attestation remain outside this proof. |
| A5 | Validate actual source ownership and provenance | MVI_NEGATIVE_PROVEN | `tests/test_r15_p1b_positive.py::test_missing_source_and_forged_refs_refuse_without_mutation`; `tests/test_r15_p1bb_failures.py::test_private_checkpoint_cannot_rebind_witnessed_original_e` | Missing/forged finite source refs and changed checkpoint bindings refuse without adoption. | General historical SourceEpoch/G2 integrity remains Human-deferred, not established by hashes. |
| A6 | No same-turn uncommitted Experience shortcut | MVI_WITNESSED | `tests/test_r15_p1b_positive.py::test_real_serialized_u3_owner_admits_revision_only_after_committed_e2`; `tests/test_r15_p1b_positive.py::test_missing_committed_e2_fails_before_owner_revision` | Real isolated R07/R12/DB1 sequence qualifies committed E/E2 before finite S1/S2 revision. | Synthetic provider; arbitrary same-turn semantic producers are not proven. |
| A7 | D9 assertions cannot adopt intrinsic truth | MVI_NEGATIVE_PROVEN | `tests/test_r15_u1_contracts.py::test_r15_d9_every_planned_ingress_claim_fails_closed`; `tests/test_r15_u2_live_scoped_interaction.py::test_forged_claim_does_not_adopt_relationship_or_goals`; `tests/test_r15_u4b_attempt_admission.py::test_unauthorized_event_cannot_adopt_and_cannot_be_called_directly` | Pure ingress enumeration plus real finite forged-claim and unauthorized-call refusals. | Exhaustive bypass absence is not proven by enumeration or finite adversarial cases. |
| A8 | Independent subjective Relationship axes | HUMAN_DEFERRED | `tests/test_r15_u1_contracts.py::test_r15_missingness_conflict_and_fixed_point_are_not_implicit_confidence`; `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input` | Six UNKNOWN axes and current scoped observation, not durable positive trust. | Original positive lasting Relationship is not proven; #323/#324 explicitly defer it beyond MVI. |
| A9 | Relationship cannot appropriate upstream truth | MVI_NEGATIVE_PROVEN | `tests/test_r15_u2_live_scoped_interaction.py::test_forged_claim_does_not_adopt_relationship_or_goals` | Tested model/user trust assertion leaves Relationship, Value and Goal unchanged. | Not an exhaustive noninvasion proof for all future R11–R18 consumers. |
| A10 | Source-bound Narrative conflict continuity | MVI_WITNESSED | `tests/test_r15_p1b_positive.py::test_real_serialized_u3_owner_admits_revision_only_after_committed_e2`; `tests/test_r15_p1b_positive.py::test_repeated_s1_without_e2_does_not_create_second_revision` | Finite provisional A/B S1→S2 preserves original interpretation; no E2 control creates no second revision. | General autobiography/semantic mapping is not proven; model statements are not external intent. |
| A11 | SelfModel hypotheses, limitations and unknowns | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u1_contracts.py::test_r15_self_meanings_are_closed_source_bound_and_opposites_stay_distinct`; `tests/test_r15_u4b_attempt_admission.py::test_observed_task_view_cannot_upgrade_verified_competence` | Closed shapes plus one observed attempt; no stable trait/role/identity producer. | Broader positive SelfModel hypotheses remain unproven, not established by model persona. |
| A12 | Attempt is not verified competence | UNAVAILABLE_FAIL_CLOSED | `tests/test_r15_u4b_attempt_admission.py::test_real_owner_attempt_commit_restart_and_no_duplicate`; `tests/test_r15_u4b_attempt_admission.py::test_observed_task_view_cannot_upgrade_verified_competence` | Attempted PRESENT is occurrence evidence only; verified competence remains UNKNOWN. | Qualified R17/R18 outcome producer unavailable; no success probability claim. |
| A13 | Bounded revisions and retained conflicts | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u1_contracts.py::test_r15_revision_suffix_reason_and_technical_restore_are_distinct`; `tests/test_r15_p1b_positive.py::test_real_serialized_u3_owner_admits_revision_only_after_committed_e2` | Pure reason/suffix constraints plus finite source-bound Narrative revision retaining S1. | General proof-preserving compaction and durable Relationship revision are not proven. |
| A14 | Core affect stays R10-owned | MVI_WITNESSED | `tests/test_r15_u1_contracts.py::test_r15_stance_accepts_exact_finite_negative_r10_optimal_loss`; `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input` | Exact current affect sampled in Actor; global affect does not establish actor causation. | Pure negative-loss shape coverage is not a claim about every durable Emotion codec. |
| A15 | Ephemeral event-scoped private stance | MVI_WITNESSED | `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input`; `tests/test_r15_u2_live_scoped_interaction.py::test_restart_does_not_restore_a_live_stance` | Live scope disappears after event/restart, Relationship/Goal remain unchanged. | No durable stance history, scheduler or future Action authority is demonstrated or allowed. |
| A16 | Tentative association is not person blame | UNAVAILABLE_FAIL_CLOSED | `tests/test_r15_u1_contracts.py::test_r15_stance_is_current_event_only_not_actor_blame_or_durable_trust`; `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input` | Actor causation and live association stay UNKNOWN; pure tentative shape is not an issued causal claim. | Qualified actor-specific producer unavailable; named blame/intent remains unproven. |
| A17 | Reviewed bounded safe-expression consumer | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u2_live_scoped_interaction.py::test_qualified_prompt_retains_safe_context_categories_only`; `tests/test_r15_u6b_runtime_use.py::test_postgeneration_race_blocks_source_conditioned_response` | Synthetic-provider prompt allowlisting and requalification, not semantic honesty of real generated answers. | U7 monetary claims and no-E2 confidence caveats remain; general safe style not proven. |
| A18 | Later evidence-bound cognitive revision | MVI_WITNESSED | `tests/test_r15_p1b_positive.py::test_real_serialized_u3_owner_admits_revision_only_after_committed_e2`; `tests/test_r15_p1b_positive.py::test_repeated_s1_without_e2_does_not_create_second_revision` | Normal serialized review with committed E2 revises finite provisional meaning, without rewriting original E. | General subjective cognition and actual external human intent are not established. |
| A19 | Exact recovery differs from cognitive accessibility | PARTIAL_OR_UNVERIFIED | `tests/test_r15_p1bb_wal.py::test_authoritative_r15_revision_persists_in_real_wal_and_restarts`; `tests/test_r15_u1_contracts.py::test_r15_revision_suffix_reason_and_technical_restore_are_distinct` | Bounded WAL restart is witnessed; accessibility enum/shape is not retrieval behavior. | Advanced functional accessibility/reactivation/compaction is not proven; #324 defers it, not an MVI blocker. |
| A20 | No R15 Decision/execution/scheduling authority | MVI_NEGATIVE_PROVEN | `tests/test_r15_u3_narrative_self_system.py::test_r15_u3_scope_excludes_effectful_sources_and_later_owner_writes`; `tests/test_r15_u2_live_scoped_interaction.py::test_forged_claim_does_not_adopt_relationship_or_goals` | Structural guard plus finite live non-mutation; does not implement R16–R19. | Optional R14 assessment and later authority remain unwired/out of scope. |
| A21 | Actual serialized admission, no operator subject edit | MVI_NEGATIVE_PROVEN | `tests/test_r15_p1b_positive.py::test_narrative_update_requires_exact_runtime_type_and_source`; `tests/test_r15_u4b_attempt_admission.py::test_unauthorized_event_cannot_adopt_and_cannot_be_called_directly` | Tested type/source/direct-call failures on finite Narrative and attempt paths. | Not a universal API security proof or grant of an operator edit surface. |
| A22 | Full legal capacity with untouched reserve | MVI_WITNESSED | `tests/test_r15_u1_capacity.py::test_r15_full_legal_fixture_fills_every_window_and_roundtrips`; `tests/test_r15_p1bb_v10_preflight.py::test_full_legal_v10_budget_reserves_all_16_mib` | Full legal v10 projection including continuation is 133,729,894 B with full 16,777,216 B reserve. | Projection is not a WAL artifact-size bound; new vocabulary/topology needs fresh sizing. |
| A23 | Additive v10, legacy, exact recovery and rollback | PARTIAL_OR_UNVERIFIED | `tests/test_r15_p1bb_wal.py::test_authoritative_r15_revision_persists_in_real_wal_and_restarts`; `tests/test_r15_p1bb_wal.py::test_v9_stays_strict_and_has_no_claimed_r15_authority`; `tests/test_r15_p1bb_failures.py::test_capture_pre_wal_fault_rolls_back_owner_to_last_committed_s1` | Actual finite v10 restart/pre-WAL rollback and strict v9; not all legacy mixed-generation/downgrade/true-rollback cases. | Whole v1–v10 lifecycle and classified true-rollback matrix require independent verification. |
| A24 | Publish only consistent durable read views | PARTIAL_OR_UNVERIFIED | `tests/test_r15_p1bb_failures.py::test_capture_pre_wal_fault_rolls_back_owner_to_last_committed_s1`; `tests/test_r15_p1bb_failures.py::test_wal_committed_but_not_terminal_event_recovers_once`; `tests/test_r15_u6b_runtime_use.py::test_current_wal_mismatch_yields_hold_not_uncommitted_root` | Tested pre/post-WAL boundaries and private read HOLD, not interruption-free multi-owner atomicity. | Final source-check-to-HTTP publication interval remains a documented U6b limitation. |
| A25 | Closed privacy/provenance payloads | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u1_contracts.py::test_r15_json_requires_closed_exact_canonical_shape_and_known_version`; `tests/test_r15_u2_live_scoped_interaction.py::test_qualified_prompt_retains_safe_context_categories_only`; `tests/test_r15_u6b_runtime_use.py::test_nonrecall_chat_never_injects_private_interpretation` | Closed payload and scoped prompt tests, not a comprehensive privacy audit of external source stores. | Source-owned content retains its own rules; real-model output privacy not guaranteed here. |
| A26 | Determinism, replay and adversarial failure evidence | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u4_self_model_system.py::test_r15_u4_conflicting_stale_reentrant_and_capacity_requests_fail_without_root_update`; `tests/test_r15_p1bb_failures.py::test_wal_committed_but_not_terminal_event_recovers_once`; `tests/test_r15_u2_live_scoped_interaction.py::test_forged_owner_arguments_and_out_of_actor_replay_hold` | Finite no-change receipt, recovery and live-owner adversarial cases; not every original listed failure. | Full alias/split and original whole-stage adversarial coverage remain not proven; index creation cannot complete this clause. |
| A27 | Explicit unit, whole-stage and Human gates | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u1_scope.py::test_r15_u1_obligations_index_is_complete_and_ast_checked`; `tests/test_r15_evidence_matrix.py::test_r15_evidence_index_is_complete_and_source_checked` | U1 and this whole-stage static index check coverage/references only, never Human acceptance. | Independent whole-stage review still required; exact-head validation of these new files is outstanding. Green CI or document creation cannot grant Ready/Merge. |

## Failure-obligation matrix

F intents follow the R15 U1 obligation index, not another stage's similarly
numbered failures; broader authoritative wording stays with #283.

| ID | Intent | Classification | Exact repository evidence | Evidence boundary | Unresolved or deferred condition |
| --- | --- | --- | --- | --- | --- |
| F1 | No claimed-identity/person false merge | MVI_NEGATIVE_PROVEN | `tests/test_r15_p1b_positive.py::test_identity_boundaries_preserve_unknown_identity` | Finite identity source boundary keeps UNKNOWN, not person corroboration. | Positive identity producer remains unavailable. |
| F2 | Missing/ambiguous binding stays unknown | MVI_NEGATIVE_PROVEN | `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input`; `tests/test_r15_u2_live_scoped_interaction.py::test_closed_collision_refuses_inherited_member` | Repeated/missing token is not reissued as a new speaker; cross-Context collision holds. | No authenticated person identity or lasting association is proven. |
| F3 | Untrusted ingress cannot force adoption | MVI_NEGATIVE_PROVEN | `tests/test_r15_u1_contracts.py::test_r15_d9_every_planned_ingress_claim_fails_closed`; `tests/test_r15_u2_live_scoped_interaction.py::test_forged_claim_does_not_adopt_relationship_or_goals` | Pure ingress enumeration plus actual finite model/operator assertion refusal. | Not exhaustive proof of every privileged ingress implementation. |
| F4 | Independent axes, no confidence from missingness | MVI_WITNESSED | `tests/test_r15_u1_contracts.py::test_r15_missingness_conflict_and_fixed_point_are_not_implicit_confidence`; `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input` | Pure independent interpretation constraints and six live UNKNOWN axes only. | Original durable positive multi-axis updates are Human-deferred #323/#324. |
| F5 | No R11 Value/R12 Belief appropriation | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u2_live_scoped_interaction.py::test_forged_claim_does_not_adopt_relationship_or_goals`; `tests/test_r15_u3_narrative_self_system.py::test_r15_u3_scope_excludes_effectful_sources_and_later_owner_writes` | Value unchanged in finite live attack; structural guard is not exhaustive Belief mutation testing. | Complete upstream noninvasion review remains required. |
| F6 | Evidence/inertia-aware lasting revision | HUMAN_DEFERRED | `tests/test_r15_u1_contracts.py::test_r15_missingness_conflict_and_fixed_point_are_not_implicit_confidence`; `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input` | Representation and UNKNOWN stance, not positive lasting Relationship revision. | Original positive lasting Relationship is not proven; #323/#324 defer it beyond MVI. |
| F7 | Bounds, ordering and committed provenance | MVI_NEGATIVE_PROVEN | `tests/test_r15_u1_contracts.py::test_r15_bounded_identifier_evidence_revision_and_ordering_reject`; `tests/test_r15_p1b_positive.py::test_missing_source_and_forged_refs_refuse_without_mutation` | Pure bounds plus finite real source rejection with unchanged owner. | General source graph/event combinations are not proven. |
| F8 | No automatic Goal/Commitment adoption | MVI_NEGATIVE_PROVEN | `tests/test_r15_u2_live_scoped_interaction.py::test_forged_claim_does_not_adopt_relationship_or_goals` | Forged trust assertion leaves tested Goal root empty/unchanged. | Every future Goal/Commitment integration remains separately owned and unproven. |
| F9 | No competence from attempt/handler/Decision | UNAVAILABLE_FAIL_CLOSED | `tests/test_r15_u1_contracts.py::test_r15_verified_competence_never_from_attempt_handler_or_decision`; `tests/test_r15_u4b_attempt_admission.py::test_observed_task_view_cannot_upgrade_verified_competence` | Verified competence stays UNKNOWN, not measured success or failure. | Independent R17/R18 outcome producer unavailable. |
| F10 | Nonfinal or missing Experience cannot prove a claim | MVI_NEGATIVE_PROVEN | `tests/test_r15_p1b_positive.py::test_missing_committed_e2_fails_before_owner_revision`; `tests/test_r15_u1_contracts.py::test_r15_uncommitted_retracted_superseded_experience_has_no_current_proof` | Actual missing-E2 rejection plus pure lifecycle dispositions; not every compacted/retracted store topology. | General historical eligibility/G2 integrity is not proven. |
| F11 | Affect is not actor blame or stable trait | UNAVAILABLE_FAIL_CLOSED | `tests/test_r15_u1_contracts.py::test_r15_stance_is_current_event_only_not_actor_blame_or_durable_trust`; `tests/test_r15_u2_live_scoped_interaction.py::test_current_actor_observation_and_private_model_input` | Actor causation UNKNOWN; no trait or lasting trust update from current mood. | Qualified actor/stable-trait producer unavailable. |
| F12 | Exact semantic identity and ordered revisions | MVI_WITNESSED | `tests/test_r15_u1_contracts.py::test_r15_semantic_meaning_identity_binds_exact_subject_predicate_polarity_and_links`; `tests/test_r15_p1b_positive.py::test_real_serialized_u3_owner_admits_revision_only_after_committed_e2` | Pure semantic checksum plus finite A/B source-qualified revision, not generic semantic truth. | Full semantic producer and real human intent remain unproven. |
| F13 | Source-bound closed self meanings | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u1_contracts.py::test_r15_self_meanings_are_closed_source_bound_and_opposites_stay_distinct`; `tests/test_r15_p1bb_failures.py::test_private_checkpoint_cannot_rebind_witnessed_later_e2` | Pure self meaning constraint and Narrative checkpoint anti-rebinding; different scopes, not general SelfModel admission. | Stable role/trait/identity production is not proven. |
| F14 | Attempt occurrence distinct from verified outcome | MVI_WITNESSED | `tests/test_r15_u4b_attempt_admission.py::test_real_owner_attempt_commit_restart_and_no_duplicate`; `tests/test_r15_u4b_attempt_admission.py::test_restored_legal_but_unqualified_task_is_not_a_positive_u4_view` | Qualified attempt persists once; legal but unqualified restored task does not create positive view. | Verified competence stays UNKNOWN; future outcome producer not implemented. |
| F15 | Forged subject/provenance cannot adopt capability | MVI_NEGATIVE_PROVEN | `tests/test_r15_u4b_attempt_admission.py::test_unauthorized_event_cannot_adopt_and_cannot_be_called_directly`; `tests/test_r15_p1b_positive.py::test_process_local_u3_refuses_external_pseudo_admission` | Direct/unauthorized finite producer calls are refused, not granted authority by transport. | Exhaustive capability adoption attacks remain outside these cases. |
| F16 | Alias/split requires qualified atomic identity evidence | UNAVAILABLE_FAIL_CLOSED | `tests/test_r15_u1_contracts.py::test_r15_claimed_identity_is_not_person_proof_or_alias_merge`; `tests/test_r15_u2_live_scoped_interaction.py::test_closed_collision_refuses_inherited_member` | Identity producer unavailable; no positive alias/split or atomic dependent-ref implementation. | Original positive alias/split is not proven; #323/#324 Human deferral. |
| F17 | Private durable state and safe projection | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u2_live_scoped_interaction.py::test_qualified_prompt_retains_safe_context_categories_only`; `tests/test_r15_u6b_runtime_use.py::test_nonrecall_chat_never_injects_private_interpretation` | Scoped private prompt gates and redaction, not whole-store/output privacy certification. | Independent privacy review and external source privacy remain separate. |
| F18 | Exact replay/no duplicate; older retries fail closed | MVI_WITNESSED | `tests/test_r15_p1bb_wal.py::test_authoritative_r15_revision_persists_in_real_wal_and_restarts`; `tests/test_r15_u4_self_model_system.py::test_r15_u4_conflicting_stale_reentrant_and_capacity_requests_fail_without_root_update` | Real finite Narrative replay plus process-local no-change receipt rejection; scopes not interchangeable. | Universal historical replay across all admitted domains is not proven. |
| F19 | Malformed/future/over-bound shapes reject | MVI_NEGATIVE_PROVEN | `tests/test_r15_u1_contracts.py::test_r15_json_requires_closed_exact_canonical_shape_and_known_version`; `tests/test_r15_p1bb_v10_preflight.py::test_noncanonical_duplicate_json_keys_and_oversize_are_refused` | Pure canonical parser and continuation preflight refusal, not a general source attestation. | Other codec/recovery paths need their own validation; no silent repair authority. |
| F20 | Exact cognition-free recovery and rollback | PARTIAL_OR_UNVERIFIED | `tests/test_r15_p1bb_failures.py::test_capture_pre_wal_fault_rolls_back_owner_to_last_committed_s1`; `tests/test_r15_p1bb_failures.py::test_wal_committed_but_not_terminal_event_recovers_once`; `tests/test_r15_u4b_attempt_admission.py::test_handler_failure_after_revision_restores_exact_selfmodel` | Actual finite pre/post-WAL and attempt rollback, not every classified true rollback or legacy generation. | Complete cognition-free mixed-schema lifecycle still requires independent review. |
| F21 | Full legal capacity and untouched reserve | MVI_WITNESSED | `tests/test_r15_u1_capacity.py::test_r15_full_legal_fixture_fills_every_window_and_roundtrips`; `tests/test_r15_p1bb_v10_preflight.py::test_full_legal_v10_budget_reserves_all_16_mib` | Adversarial full-window schema projection includes continuation and full reserve, not tiny-fixture sizing. | WAL file-size behavior is separate from snapshot cap. |
| F22 | No later-owner authority from R15 | PARTIAL_OR_UNVERIFIED | `tests/test_r15_u1_scope.py::test_r15_u1_pure_modules_are_not_runtime_producers`; `tests/test_r15_u3_narrative_self_system.py::test_r15_u3_scope_excludes_effectful_sources_and_later_owner_writes`; `tests/test_r15_u2_live_scoped_interaction.py::test_forged_claim_does_not_adopt_relationship_or_goals` | R15-specific U1 structural guard plus later finite boundaries, not another stage's F22 or production proof from pure U1. | Whole-stage noninvasion review required; future R16–R19 authority remains absent. |

## Crosscutting evidence boundaries

### D9 — no assertion-to-authority shortcut

The A7/F3/F15 tests distinguish pure ingress dispositions from actual finite
producer refusal. Model/operator/Web assertions, repeated claims, privileged
transport, checksums and technical restore do not create adoption authority.
These checks do not prove exhaustive absence of every future bypass.

### Capacity — 128 MiB and untouched 16 MiB

`docs/r15-p1bb-v10-wal.md` supersedes U1-only sizing for the continuation:
133,729,894 B including the untouched 16,777,216 B future reserve, below the
134,217,728 B hard cap with 487,834 B spare. A22/F21 link full-legal tests.
This is the snapshot projection; it is not a 128 MiB WAL artifact-size claim.

### Privacy — closed state and bounded private projections

A25/F17 link shape and live scoped prompt tests. No raw prompt, transcript,
hidden thought, model rationale or free-text profile belongs in intrinsic roots.
Source-owned Memory content and real generated outputs need separate privacy
analysis. A private hint is not an operator-facing truth or new diagnostic API.

### Source — actual ownership and event ordering

`docs/r15-p1ba-qualified-admission.md` describes finite qualification against
terminal R07 CHAT, committed active R12/DB1 and matching R09 Context. A6/A18
link E→S1 before E2→S2 and no-E2 controls. These are normal runtime readers,
not certified no-repair forensic reads or SourceSetVerified. Advanced G2
historical SourceEpoch integrity is Human-deferred under #324/Post-R26;
`docs/adr/2026-10-12-r15-g2-source-preserving-reads.md` retains that distinction.

### U2 — same-token cross Context is a hold, not identity

`docs/r15-u2-relationship-system.md` is a historical owner-boundary document,
not the final MVI amendment. Actual live Actor tests in A2/A3/A15/F2 cover
current R09/R10 capture, same raw token across Contexts (including a closed
sibling), inherited membership, and no durable Relationship update. #323/#324
permit this bounded scoped interaction; positive lasting Relationship and
alias/split remain not proven rather than silently passed.

### U3 — positive provisional revision and no E2 control

The earlier `docs/r15-u3-narrative-self-system.md` design hold and pure tests
are not positive production evidence. The later P1b callable in A10/A18
exercises real isolated owners with a synthetic provider: committed E/S1 then
E2/S2, source-bound provisional A/B, retained S1, and no second revision without
E2. UNKNOWN is held/absent, not a positive claim: see
`tests/test_r15_p1bb_failures.py::test_unknown_referent_is_not_admitted_as_source_qualified_positive`.
Neither A nor B proves actual human intent or a general semantic interpreter.

### U4 — attempted PRESENT, verified UNKNOWN

`docs/r15-u4b-attempt-admission.md` owns one closed observed task class.
A12/F14 test the real isolated attempt commit/restart/duplicate path and
rejection of legal but unqualified restored shapes. Attempt confidence is
occurrence of an attempt, not task accuracy, success probability or intent.
R17/R18 verified competence remains UNKNOWN pending its own accepted producer.

### U5 — v10 WAL, recovery and rollback

`docs/r15-p1bb-v10-wal.md` describes actual v10 roots/continuation and lazy
legacy behavior. A23/A24/F20 link finite real restart, pre-WAL owner rollback,
post-WAL/pre-terminal recovery and read mismatch holds. Integrity-only restore
must not replay cognition. This index does not claim every v1–v10 mixed WAL,
ordinary downgrade or classified true rollback case has been revalidated.

### U6 — qualified private read and publication limits

`docs/r15-u6b-private-use.md` describes explicit referent recall in the same
Context, pre-generation qualification and post-generation requalification.
`tests/test_r15_u6b_runtime_use.py::test_next_real_chat_and_restart_use_current_provisional_option`
is the synthetic-provider positive read witness;
`tests/test_r15_u6b_runtime_use.py::test_missing_original_or_later_experience_forces_hold_no_prompt_hint`
and `tests/test_r15_u6b_runtime_use.py::test_outside_chat_cannot_request_private_projection`
bound it. A17/A24 cover races and WAL mismatch. No interruption-free
multi-owner atomicity between final source check and HTTP publication is proven.

### U7 — bounded real-model scientific evidence only

Human accepted the two cases on #351 seq19 as SCIENTIFIC EVIDENCE ONLY.
Gemma4 made unsupported monetary claims in ordinary answers; case B confidence
increased with no E2. Independent semantic reviews were blocked by native
external_directory DENY. These caveats do not become cognitive-state truth,
verified identity, lasting trust, verified competence or safe-expression proof.
No local deterministic callable reproduces the accepted real-model record;
this index cannot attest inaccessible artifacts, fix the claims, or repeat the
trial. Full generation identity #343 remains unproven and separately deferred.

### Authority — no later-stage or acceptance authority

No R16 Decision, R17/R18 execution/outcome or R19 autonomous scheduling is
introduced. Advanced accessibility and original lasting Relationship/alias/split
remain not proven under #323/#324, not bounded MVI blockers. A26/A27 retain
independent whole-stage review and the later Human gate. There is no whole-R15
Human acceptance or deployment authorization here; no publication is requested.

### CI — exact base head only, not these uncommitted files

Coordinator-verified GitHub Actions push run
[40176534738](https://github.com/upiscium/PROJECT-SUZKA/actions/runs/40176534738)
reported Backend and Frontend success at
`4b1dc8486a55a2fd2a8a459912f533e7c27b30f7`. That is exact-head CI evidence
for the immutable base only, not for this uncommitted two-file change. The
checker performs no network access and does not infer CI or Human acceptance
from the filesystem. Native PATH inspection found no python/python3/pytest/uv
or just executable in this session; focused tests and `just check-all` were
not executed. No pass count is claimed. Independent whole-stage review is
still required, including checks of these links against the frozen obligations.

## Focused verification handoff

When native project tooling is available, the bounded commands are:

```sh
just test tests/test_r15_evidence_matrix.py
uv run pytest -q tests/test_r15_u1_scope.py tests/test_r15_u1_capacity.py tests/test_r15_u2_live_scoped_interaction.py tests/test_r15_u4b_attempt_admission.py tests/test_r15_u6b_runtime_use.py
just lint tests/test_r15_evidence_matrix.py
```

These commands are not an execution record. The new checker uses only Python
stdlib plus existing pytest, imports no product/test modules, executes no model,
and checks AST definitions without collecting or running referenced tests.
It deliberately cannot establish semantic correctness, comprehensive adversarial
coverage, independent review, deployment safety or Human acceptance.
