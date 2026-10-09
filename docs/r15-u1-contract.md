# R15 U1 — pure intrinsic-state contracts and capacity gate (#284)

This document is **U1 evidence, not U1 Human PASS, U2 permission, production
admission, or v10 migration**. [A1–A27](https://github.com/upiscium/PROJECT-SUZKA/issues/283#issuecomment-6058471359)
are Human accepted. Three separate read-shaped contract roots live in
`suzka/r15/contracts.py`; source and ingress *obligations* in `common.py`,
event-only `InteractionStance` in `stance.py`, and the full-capacity projector
in `bounds.py`. Constructors/parsers/digests **never confer subject authority**.

## Responsibility

- **Owns:** typed, finite, checksum-bound Relationship, NarrativeSelf, SelfModel
  representation; declarative source/admission requirements; an event-scoped
  stance value; a pure worst-case v10 size calculation.
- **May:** reject malformed/future/over-bound shapes; represent subjective
  UNKNOWN/PRESENT/CONTESTED/CONTRADICTORY and independent axes, conflict proofs,
  tentative association, and unknown verified competence; compute canonical
  checksums and a read-only capacity budget.
- **Must not:** admit any source or subject, merge identities, create a store,
  mutate a runtime/source root, infer actor cause, persist v10/WAL, publish read
  projections or affect prompts, adopt Goal/Decision, or operate a scheduler.
- **Depends on:** accepted R09–R14 contracts and the exact R14 AgentState v9
  capacity projection, *not* their producer ownership or runtime imports.
- **Used by:** proposed U2–U4 owners, U5 persistence and U6 reviewed producer
  after their separate Human gates. There is **no present production consumer**.

### Typed interpretations and independent roots

The Relationship root holds up to 64 separately keyed perceived interlocutors,
each with all six independently represented axes: trust, familiarity,
closeness, caution, reciprocity *expectation*, and unresolved issue.
`claimed_identity_key` is a declarative R09 claim; equal keys do not collapse
distinct interlocutors. Uncertainty is per interpretation; `UNKNOWN` has no
fabricated zero-confidence measurement. `PRESENT` requires support;
`CONTESTED` retains support **and** counterevidence and bounded subjective
estimate/confidence; `CONTRADICTORY` retains both sides with no asserted
numeric resolution. Axes do not assert the other person's actual beliefs or
intent. A single unverified negative global Emotion is not a trust update.

The NarrativeSelf root holds at most 64 explicitly Experience-linked episodes,
32 structured claims, four links per episode/claim, opposing evidence and 64
retained revision proofs. No biography, summary, future event, prompt or source
payload can be inserted into a field. Each present episode has its referenced
Experience among its supporting witnesses; this still does **not** prove that
the Experience was externally committed. A stale/retracted/superseded episode
cannot have a present interpretation. Link IDs are cross-root declarative
references only; future owners verify target existence/revision and atomic
impact of any separately qualified identity split or alias transition.

SelfModel has 32 identity/role/possible-trait/limitation/epistemic-unknown
claims and 16 distinct task-class hypotheses. An attempted-task hypothesis has
separate *subjective* confidence; `verified_competence` has **only `UNKNOWN`**.
Changing that vocabulary requires a later independently accepted verified
R17/R18 outcome producer. Tool/Decision/handler/model fluency cannot populate
verified task success probability. No root can edit another root, R09 person,
R11 Value, R12 Belief/Experience or R13 Goal/Commitment.

#### Closed claim meaning and unresolved semantic producer (HOLD-2 correction)

`NarrativeClaim` and `SelfClaim` now require `ClaimMeaning(code,
referent_kind, referent, polarity)`. Every `SemanticCode` is a *specific finite
proposition*, never a free-form statement, biography, user profile or hash of
unknown text. `AFFIRMS` and `DENIES` are distinct hypotheses, not truth
values; denying unverified competence or independent Action would improperly
imply verified capability and is forbidden. The complete current vocabulary:

| Domain/kind | Exact semantic predicate code(s) | Meaning and bounded target owner |
| --- | --- | --- |
| Narrative continuity | `same_autobiographical_continuity` | Suzka interprets two distinct referenced episodes as continuous, not identical people or verified external events; R15 episode ID(s) backed by R12. |
| Narrative role | `assistant_role_in_episode`, `conversation_partner_role_in_episode` | Claimed *Suzka* role in a particular retained episode; episode ID, not a user/person identity. |
| Narrative reinterpretation | `reinterpret_episode_as_subjectively_uncertain` | Current subjective interpretation of a prior episode is uncertain; original R12 Experience remains unchanged. |
| Narrative contradiction | `conflicting_first_person_interpretations` | Opposing first-person interpretations of the named episode(s) are retained, not resolved as fact. |
| Self identity | `subject_identity_suzka` | First-person identity hypothesis scoped to a specifically referenced R12 Experience. |
| Self role | `subject_role_assistant`, `subject_role_conversation_partner` | Suzka's proposed role scoped to an R12 Experience, not another person's role. |
| Possible self trait | `subject_possibly_cautious`, `subject_possibly_patient` | Two distinct *possible*, revisable traits tied to an R12 Experience; no stable-trait adoption from a single event. |
| Limitation | `subject_lacks_verified_task_competence`, `subject_lacks_independent_external_action` | Typed limits, not proof of failure; denied polarity prohibited because it would assert a presently unverified ability. Experience ID only locates a candidate. |
| Epistemic unknown | `subject_task_competence_is_unknown` | A distinct retained `task_class_key`; claim status must stay `UNKNOWN`, and verified competence stays `UNKNOWN`. |

Narrative referents must match one of that claim's bounded episode links *and*
the retained episode root; a current claim cannot refer to an UNKNOWN,
retracted or superseded episode. Current self-role/trait/identity/limitation
shapes require an exact supporting R12 Experience reference matching the
meaning's bounded target; a missing/retracted source fails structural checks.
Task-class referents must match a retained bounded capability row. **None**
of these checks establishes actual source authenticity or Suzka's subjective
adoption. `current_semantic_producer` returns `UNAVAILABLE` for every code:
the integrated R12 Experience has event lineage and R12 Belief may have
canonical text, but neither owns a deterministic mapping from its state to
these R15 propositions. A future U3/U4 separately reviewed owner must prove
exact qualified first-person interpretation, source-root/revision/digest,
counterevidence, subject admission and semantic mapping before any durable
`PRESENT` claim. Missing/unavailable mapping remains UNKNOWN / unavailable;
operator/model/Web text, matching names and attacker-recomputed hashes cannot
fill the gap. Shapes are capacity reservations, **not admitted assertions**.

`Interpretation.statement_id` for these two claim types is a domain-separated
SHA-256 of the exact owner domain, claim kind, `ClaimMeaning` including
polarity, and for Narrative the full sorted episode link tuple. Reversing a
polarity, role/trait code, target or linked episode while retaining the old ID
fails closed even if an attacker recomputes the outer root checksum. Opposing
meaning IDs remain separately retainable as conflict. The corrected U1 closed
schema is still **pre-production**: no U1 JSON was persisted in AgentState;
older Draft-U1 JSON lacking a required meaning fails closed. It is not a
lazy legacy v10 migration claim. New codes/owners require separate review and
re-derivation of the maximum before implementation.

All persisted IDs are canonical ASCII `[A-Za-z0-9._:-]`, at most 128 codepoints
without `..`; digests are exactly 64 lowercase hex chars. Times are explicit
UTC with six fractional digits and `Z`, event sequences ≤ 2⁶³−1, revisions
≤ 2³¹−1, confidence/estimate exact integers in `[0, 1,000,000]` (null is
UNKNOWN, never coerced to zero). JSON input must have exact closed keys,
ordered unique rows, canonical bytes, no duplicate keys, no invalid floats,
at most 32 levels, and per-root full-size preflight. SHA-256 hashes are
domain-separated checksums of canonical serialized shapes, **not signatures**.
Raw prompt, transcript, user profile, attachment, hidden thought and model
rationale have no R15 field. Roots never call a clock, random source, provider,
runtime handler, R16–R19 or production persistence API.

### Exact source/admission map — `REQUIRES_TRUSTED_ROOT` is NOT acceptance

| R15 concept | Existing actual producer / candidate | Required later validation and availability |
| --- | --- | --- |
| Perceived interlocutor | R09 active Context frame, caller-supplied opaque `interlocutor_key` | U2/U6 must compare exact selected Context ID/revision/participants to the ordered trusted R09 root. This is a participant label, not real-world person identity. |
| Claimed identity | R09 `InterlocutorBinding` holds caller-supplied `identity_key`, confidence and evidence refs | Only a **declarative** reference; U2 may compare exact binding revision and retain ambiguity. Qualified independent same-person alias/split producer does **not** exist; verified person/automatic merge remains UNAVAILABLE. |
| Subjective trust, familiarity, caution, closeness, expectations | R12 first-person Experience may ground a hypothesis **only after actual external commit + R07 reconciliation** | U2 checks current exact stored Experience ID/revision/digest/event/episode/context/lifecycle and independently qualified interpretation and all opposing evidence; active event-local construction and asserted proof tuple are NOT finalized. No R10 actor-specific appraisal producer. Operator/model/Web assertions cannot update these axes. |
| Personal episode / narrative | R12 committed ACTIVE Experience; adopted Belief may contextualize a separate claim | U3 verifies stored Experience and lifecycle/adoption, source-root identity, prior version/counterproof, subject interpretation, event order and links atomically. No current R12 producer maps content to a closed R15 `SemanticCode`; until a reviewed semantic mapping/subject admission exists, claim adoption is UNAVAILABLE. R11 Value/R13 Goal/Commitment refs are declarative context, not R15 self-endorsement. Imagined events and model biographies unavailable as source. |
| Possible trait / limitation / attempted task | Committed qualified Experience may locate a **hypothesis** | U4 independently checks first-person evidence, current root, semantic predicate and exact backing referent, opposing/ambiguous evidence, subject admission and inertia. Current stable-trait/role semantic producer is UNAVAILABLE; synthetic operator endorsement/repeated claims fail. R14 U3 assessment is only pure and **not production-wired**. |
| Verified competence or success probability | R17/R18 qualified outcome producer absent; Decision/Tool/handler success is unverified | UNAVAILABLE; U4 may represent only `UNKNOWN`. Future R17/R18 authority needs its own review. |
| Tentative interaction association | R10 current global valence/arousal/optimal-loss **and** R09 current Context projection at the exact serialized event | U6 must compare both projected checksums, participants and exact event with trusted current snapshots. This is explicitly subjective *provisional* interaction association, not proved person, intent, or lasting trust. |
| Actor-directed emotion/blame/cause | R10 ordinary chat calls `AppraisalSignals()` (no actor-specific producer); R14 U3 has none | UNAVAILABLE; stance actor causation and any named anger label remain UNKNOWN, not inferred from negative valence/arousal or matching names. |

Closed R11 Value, R12 Belief and R13 Motivation/Goal/Commitment are external
truths, not R15 authorization for own intrinsic trait or adoption. R14
Attention focus/Metacognition supply no trust/personality/verified competence
proof. `SourceWitness` records a typed declared source reference, claimed
origin, lifecycle, root checksum, event and revision; a syntactically valid or
attacker-recomputed digest is still **untrusted**. `source_disposition` only
returns `UNAVAILABLE` or `REQUIRES_TRUSTED_ROOT`, never `ADMITTED`.

### D9 and failure boundaries

`UntrustedIngress` enumerates operator API, privileged correction, internal
handler, serialized state port, repeated claim, forged subject admission,
migration, technical restore, model and Web ingress. Direct claims through **all
these paths** are unavailable as intrinsic trust/trait/competence source; an
authenticated transport or serialized event never grants subjective adoption.
Forging `SourceKind.EXPERIENCE`, `SUBJECT_OBSERVATION`, an active lifecycle,
checksum or event tuple yields **only a pending trusted-root check**. Later U2–U4
must fetch actual source ownership, committed state, revision/digest/event and
subject-origin/provenance from an ordered runtime owner, validate complete
counterevidence, then stage and atomically commit or reject **without partial
mutation**. Source inactivation and capacity exhaustion must reject/review,
not silently drop claims or legal proofs. Exact duplicate canonical bytes are
stable; same-event receipts and older retry rejection are deferred to U2–U5.
No mutation/recovery/REST ingress exists in U1.

### D6 fast affect; D11 slow reinterpretation vs recovery

`InteractionStance` is an immutable private value tied to **one exact event**.
It carries a bounded sorted R09 participant projection, exact Context ID/revision
and checksum, a separately checked current R10 global Emotion projection,
binary64-exact hex scalars, optional *declared* prior committed Relationship
digest, and `UNKNOWN` or `TENTATIVE_INTERACTION` with explicit uncertainty.
Both checksums detect internal mismatch, **not producer authenticity**: U6 must
compare with the actual trusted current serialized R09/R10 snapshots, and
ensure the optional R15 digest points to a prior **committed** read view.
`optimal_loss_hex` is any exact canonical *finite* R10 binary64 value,
including negatives and signed negative zero accepted by current R10
`EmotionState`. Valence/arousal retain their distinct bounds. The stricter
nonnegative durable v9 Emotion snapshot rule does not silently narrow the
legal *current-domain* D6 stance; U6 must reconcile trusted snapshot/producer
compatibility before consumption, without changing accepted R10/R14.
There is no actor-specific cause/name, persistent field/history, source
selection, prompt/style wiring or Goal/Action influence in U1. Uncommitted
current Experience cannot serve as its durable successor evidence.

Technical restart/restore/classified true rollback (deferred U5) must rebuild
the exact compatible root from integrity evidence with **no cognition replay**.
An ordinary cognitive change (deferred U2–U4) is separately event/revision-
ordered, source-validated `RevisionReason` and a finite 64-proof per-owner
suffix with a previous-proof digest anchor; current conflicts keep up to four
support and four independent contrary witnesses per interpretation. Retaining
an anchor does **not** itself license truncation: U2–U5 must prove an acceptable
compaction scheme before any history transition. `Accessibility` current,
latent, reactivatable is functional retrieval/salience of *interpretation*,
not R12 record deletion, physical privacy erasure, R08 WorkingMemory mutation,
R14 focus policy or source retraction. Reactivation needs a qualified cue;
reinterpretation preserves original R12 Experience and prior conflicting
claims. R15 does not create a hidden timer or model ranking.

### Full-legal v10 size gate (schema projection only)

The fixture in `maximum_legal_r15_roots` fills **all 64 relationship records ×
6 distinct axes × 4 supporting + 4 contrary witnesses**, **64 episodes × 8
witnesses × 4 links**, **32 narrative claims × 8 witnesses × 4 episode refs**,
**32 self claims × 8 witnesses**, **16 task hypotheses × 8 witnesses**, plus
**64 revision proofs in EACH owner**, max-width IDs, digests, counters and
canonical UTC timestamps. All 32 narrative claims carry closed longest legal
meaning predicates, polarity, and 64-byte retained episode referents; all 32
self claims carry closed longest legal codes and distinct 128-character
first-person Experience referents. Every row is instantiated as a legal immutable
contract, roots round-trip exact full bytes, and each full section has an exact
canonical serializer count. Legal alternatives (Belief for NarrativeClaim,
UNKNOWN, inactive episodes, missing refs, shorter enums) are no larger than
the full-width event-bearing committed-Experience / contested form. All
other compatible semantic code/kind pairs and the UNKNOWN task-class alternative
are tested not to exceed the maximum legal root; increasing this vocabulary
requires rerunning the proof, never clipping the reserved headroom.
Array/key/braces/commas, three new top-level keys and the 9→10 root version
digit are included. Stance is event-only and **not a persisted field**.

| Full-legal future v10 projection | Bytes |
| --- | ---: |
| Integrated v9 before reserve (unchanged) | 113767203 |
| Relationship value | 2159153 |
| NarrativeSelf value | 678461 |
| SelfModel value | 344497 |
| Three new JSON root-key/comma/colon overhead | 66 |
| Schema version `9` → `10` width delta | 1 |
| **v10 before reserve** | **116949381** |
| **Full untouched future reserve** | **16777216** |
| **v10 including reserve** | **133726597** |
| 128 MiB hard cap | 134217728 |
| **Margin beyond full reserve** | **491131** |

64 admitted, evidence-rich simultaneous relationships are distinct from R09's
1024 possible declarative participant bindings; **not** all bindings become
qualified relationships. The other maxima preserve dozens of episodic and
self interpretations with contradictory evidence and retained revisions.
Capacity cannot be increased by hiding evidence or evicting automatically.
If U2–U4 prove the required simultaneous admitted set exceeds any declared
maximum, they must reject and obtain a separate Human capacity decision, NOT
silently clip. The 64-proof suffix and anchor are a *design shape*, not proof
of a safe compaction algorithm or a promise that old conflict history can be
discarded. Full legal v10 sizing passes the snapshot hard-cap prerequisite;
U5 separately proves actual v10 codec/WAL JSONL embedded snapshot bytes,
retained mixed generations, durability/recovery and any WAL artifact caps.
WAL files are separate artifacts; this snapshot projection does not claim
they have a 128 MiB file-size limit or that v10 migration is implemented.

## Checked acceptance and failure-obligation index

`U1_DIRECT` means the named test exercises a *pure shape/availability* only;
`PARTIAL_DEFER_Ux` means later production proof is needed; `DEFER_Ux` means U1
does not claim runtime implementation. The exact references are AST-resolved
by `tests/test_r15_u1_scope.py`; they are not a passing-count claim. A1–A27
and F1–F22 definitions and wording remain governed by #283, not this summary.

| Clause | U1 classification | Executable U1 evidence / later obligation |
| --- | --- | --- |
| A1 | PARTIAL_DEFER_U2_U4 | `tests/test_r15_u1_contracts.py::test_r15_closed_roots_roundtrip_is_canonical_and_independent`; mutation isolation U2–U4. |
| A2 | PARTIAL_DEFER_U2 | `tests/test_r15_u1_contracts.py::test_r15_claimed_identity_is_not_person_proof_or_alias_merge`; trusted R09 binding U2. |
| A3 | DEFER_U2 | Qualified identity admission/alias/split producer absent; atomic dependent-ref handling U2. |
| A4 | PARTIAL_DEFER_U2_U4 | `tests/test_r15_u1_scope.py::test_r15_u1_pure_modules_are_not_runtime_producers`; source root checks U2–U4. |
| A5 | PARTIAL_DEFER_U2_U4 | `tests/test_r15_u1_contracts.py::test_r15_fake_provenance_subject_admission_and_untrusted_text_cannot_adopt`; current root checks U2–U4. |
| A6 | PARTIAL_DEFER_U2_U4 | `tests/test_r15_u1_contracts.py::test_r15_uncommitted_retracted_superseded_experience_has_no_current_proof`; R12 external commit check U2–U4. |
| A7 | PARTIAL_DEFER_U2_U6 | `tests/test_r15_u1_contracts.py::test_r15_d9_every_planned_ingress_claim_fails_closed`; ingress/recovery proof U2–U6. |
| A8 | PARTIAL_DEFER_U2 | `tests/test_r15_u1_contracts.py::test_r15_missingness_conflict_and_fixed_point_are_not_implicit_confidence`; event mutation U2. |
| A9 | PARTIAL_DEFER_U2_U6 | `tests/test_r15_u1_contracts.py::test_r15_claimed_identity_is_not_person_proof_or_alias_merge`; owner noninvasion U2/U6. |
| A10 | PARTIAL_DEFER_U3 | `tests/test_r15_u1_contracts.py::test_r15_semantic_meaning_identity_binds_exact_subject_predicate_polarity_and_links`; qualified semantic producer/committed capture U3. |
| A11 | PARTIAL_DEFER_U4 | `tests/test_r15_u1_contracts.py::test_r15_self_meanings_are_closed_source_bound_and_opposites_stay_distinct`; first-person semantic producer U4. |
| A12 | PARTIAL_DEFER_U4 | `tests/test_r15_u1_contracts.py::test_r15_verified_competence_never_from_attempt_handler_or_decision`; verified producer R17/R18. |
| A13 | PARTIAL_DEFER_U2_U4 | `tests/test_r15_u1_contracts.py::test_r15_revision_suffix_reason_and_technical_restore_are_distinct`; proof-preserving revision U2–U4. |
| A14 | PARTIAL_DEFER_U6 | `tests/test_r15_u1_contracts.py::test_r15_stance_accepts_exact_finite_negative_r10_optimal_loss`; R10 capture U6. |
| A15 | PARTIAL_DEFER_U6 | `tests/test_r15_u1_contracts.py::test_r15_stance_is_current_event_only_not_actor_blame_or_durable_trust`; runtime projection U6. |
| A16 | PARTIAL_DEFER_U6 | `tests/test_r15_u1_contracts.py::test_r15_stance_is_current_event_only_not_actor_blame_or_durable_trust`; qualified actor producer absent. |
| A17 | DEFER_U6 | No prompt/style consumer; separately reviewed safe expression U6. |
| A18 | DEFER_U2_U4 | Qualified ordinary cognitive revision and retained contrary evidence U2–U4. |
| A19 | PARTIAL_DEFER_U5 | `tests/test_r15_u1_contracts.py::test_r15_revision_suffix_reason_and_technical_restore_are_distinct`; exact restore U5 and accessibility U2–U4. |
| A20 | PARTIAL_DEFER_U6 | `tests/test_r15_u1_scope.py::test_r15_u1_pure_modules_are_not_runtime_producers`; Decision/Goal/Action boundary U6. |
| A21 | DEFER_U2_U6 | Serialized runtime admission, no direct operator-edit API U2–U6. |
| A22 | U1_DIRECT | `tests/test_r15_u1_capacity.py::test_r15_v10_projection_matches_integrated_v9_projector_and_keeps_full_reserve`; no persistence claim. |
| A23 | DEFER_U5 | v10 lazy compatibility, WAL and true rollback U5. |
| A24 | DEFER_U5_U6 | Durable atomic publication/rollback of committed views U5–U6. |
| A25 | PARTIAL_DEFER_U6 | `tests/test_r15_u1_contracts.py::test_r15_self_meanings_are_closed_source_bound_and_opposites_stay_distinct`; privacy-safe projections U6. |
| A26 | PARTIAL_DEFER_U2_U6 | `tests/test_r15_u1_contracts.py::test_r15_closed_roots_roundtrip_is_canonical_and_independent`; receipts/crash tests U2–U6. |
| A27 | PARTIAL_DEFER_U7 | `tests/test_r15_u1_scope.py::test_r15_u1_obligations_index_is_complete_and_ast_checked`; whole-stage gate U7/Human. |

| Failure | U1 classification | Executable U1 evidence / later obligation |
| --- | --- | --- |
| F1 | PARTIAL_DEFER_U2 | `tests/test_r15_u1_contracts.py::test_r15_claimed_identity_is_not_person_proof_or_alias_merge`; U2 identity. |
| F2 | PARTIAL_DEFER_U2 | `tests/test_r15_u1_contracts.py::test_r15_claimed_identity_is_not_person_proof_or_alias_merge`; U2 unknown binding. |
| F3 | PARTIAL_DEFER_U2 | `tests/test_r15_u1_contracts.py::test_r15_d9_every_planned_ingress_claim_fails_closed`; U2 ingress. |
| F4 | PARTIAL_DEFER_U2 | `tests/test_r15_u1_contracts.py::test_r15_missingness_conflict_and_fixed_point_are_not_implicit_confidence`; U2 independent axes. |
| F5 | DEFER_U2_U6 | Source-owned R11/R12 Value/Belief noninvasion U2–U6. |
| F6 | DEFER_U2 | Evidence/inertia-aware durable revision U2. |
| F7 | PARTIAL_DEFER_U2_U4 | `tests/test_r15_u1_contracts.py::test_r15_bounded_identifier_evidence_revision_and_ordering_reject`; committed provenance U2–U4. |
| F8 | DEFER_U2_U6 | No Goal/Commitment adoption from relationship U2/U6. |
| F9 | PARTIAL_DEFER_U4 | `tests/test_r15_u1_contracts.py::test_r15_verified_competence_never_from_attempt_handler_or_decision`; verified outcomes R17/R18. |
| F10 | PARTIAL_DEFER_U3 | `tests/test_r15_u1_contracts.py::test_r15_uncommitted_retracted_superseded_experience_has_no_current_proof`; U3 R12 finality. |
| F11 | PARTIAL_DEFER_U4 | `tests/test_r15_u1_contracts.py::test_r15_stance_is_current_event_only_not_actor_blame_or_durable_trust`; U4 trait admission. |
| F12 | PARTIAL_DEFER_U3 | `tests/test_r15_u1_contracts.py::test_r15_semantic_meaning_identity_binds_exact_subject_predicate_polarity_and_links`; U3 revisions. |
| F13 | PARTIAL_DEFER_U3 | `tests/test_r15_u1_contracts.py::test_r15_self_meanings_are_closed_source_bound_and_opposites_stay_distinct`; U3 semantic producer. |
| F14 | PARTIAL_DEFER_U4 | `tests/test_r15_u1_contracts.py::test_r15_verified_competence_never_from_attempt_handler_or_decision`; U4/R17/R18 distinctions. |
| F15 | PARTIAL_DEFER_U4 | `tests/test_r15_u1_contracts.py::test_r15_fake_provenance_subject_admission_and_untrusted_text_cannot_adopt`; verified outcome R17/R18. |
| F16 | DEFER_U2 | Qualified identity alias/split and atomic dependent refs U2. |
| F17 | PARTIAL_DEFER_U6 | `tests/test_r15_u1_scope.py::test_r15_u1_pure_modules_are_not_runtime_producers`; durable privacy U5/U6. |
| F18 | DEFER_U2_U5 | Same-event replay receipts and historical retry rejection U2–U5. |
| F19 | PARTIAL_DEFER_U2_U4 | `tests/test_r15_u1_contracts.py::test_r15_json_requires_closed_exact_canonical_shape_and_known_version`; source preflight U2–U4. |
| F20 | DEFER_U5 | Cognition-free restart/true rollback and failure tests U5. |
| F21 | U1_DIRECT | `tests/test_r15_u1_capacity.py::test_r15_full_legal_fixture_fills_every_window_and_roundtrips`; full reserve crosscheck in v10 projection test. |
| F22 | PARTIAL_DEFER_U6 | `tests/test_r15_u1_scope.py::test_r15_u1_pure_modules_are_not_runtime_producers`; no future owner U6/U7. |
