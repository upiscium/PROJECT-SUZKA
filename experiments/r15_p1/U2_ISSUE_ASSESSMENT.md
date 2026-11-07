# R15 U2 — observed difference, provisional unresolved interaction and admission (TEST-ONLY DESIGN)

Date: 2026-10-21 JST. Parent: [Issue #339](https://github.com/upiscium/PROJECT-SUZKA/issues/339); evidence: [Draft PR #344](https://github.com/upiscium/PROJECT-SUZKA/pull/344), merged [U4b #345](https://github.com/upiscium/PROJECT-SUZKA/pull/345), Post-R27 real speaker/person identification [#346](https://github.com/upiscium/PROJECT-SUZKA/issues/346).

**Decision accepted by Human:** Do not equate A/B positions with an intrinsic Relationship problem. Preserve observation, interpretation and durable adoption as three different authorities. Robust per-message speaker/person identification is deferred Post-R27. **No source code, schema, deployment, Subject state, GitHub Ready or Merge authorization is implied here.**

## Observable three-stage behavior

1. **Observed difference:** The actual serialized R07 CHAT/API_CHAT terminal/transaction and active committed R12 Experience qualify the event and lifecycle; the matching DB1 episode supplies the **actual visible user and Suzka text** for the same fixed task; StateWAL's exact event candidate supplies a newly introduced caller-claimed opaque token (distinct from persistent Context membership). The only claim is that two provisional choices were visibly expressed. The test-only result is bounded; no verified real person, intention or correct answer.
2. **Observation triage — NOT a Subject assessment:** If both *committed texts* express different choices on the same closed cooperative task AND both *state* that the decision remains pending, the fixture may return `PENDING_EXPRESSIONS_ONLY`. The requalification/staleness check also rejects retired Source, altered caller-owned data, later ordinary/debug Chat or Context mutation affecting the original frame. This classification means only **two expressions contain pending-decision language**; user/model output can be prompted to say it. It is not a proof of genuine open coordination, of actual speaker identity, or of SUZKA's own accepted perception. **The second-stage subjective significance/interpretation mechanism is not yet implemented.** If only one side says pending, or if a plain A/B difference exists, never infer an interpersonal problem. The test fixture does not infer one either.
3. **Future subjective assessment then durable admission — BOTH NOT IMPLEMENTED:** An independently reviewed internal `RELATIONSHIP_REVIEW` event must requalify all source owners *again*, obtain a separately justified **SUZKA-owned significance assessment** rather than copying the observed user's or model's "未決定" labels, and only when that assessment genuinely identifies a tentative interaction-level relationship issue may it stage a real `RelationshipSystem` revision. Only it may write one tentative, interaction-scoped `UNRESOLVED_ISSUE` from a real SourceWitness and a real event-bound RevisionProof through existing AgentState v10/StateWAL, while all five other axes stay UNKNOWN. This test-only candidate is **not a license to call an existing generic owner mutation or install a forged U1 root**. `VerifiedPerson`, mutual feeling, trust, causality and verified competence are strictly outside scope.

## Explicit interpretation state machine (test-only)

| Source-qualified observation | Test-only expression triage | Durable Relationship |
|---|---|---|
| Same task A/A | AGREEMENT_ONLY | No change |
| Same task A/B without a pending shared decision | OBSERVED_DIFFERENCE_ONLY | No change |
| Same task A/B and only one side declares an open decision | OBSERVED_DIFFERENCE_ONLY | No change |
| Same task A/B with both sides declaring the cooperative decision undecided and Suzka explicitly unverified | PENDING_EXPRESSIONS_ONLY | No change until separate Subject owner event and Human-approved producer |
| Same task A/B but resolution already asserted in this event | UNKNOWN/refuse closed protocol | No change |
| User Q1 vs Suzka Q2 | UNKNOWN | No change |
| Later CHAT in same Context after source E | STALE_HOLD | No change: later turns lack per-message speaker evidence |
| Different Context's later Chat | Candidate remains scoped to original Context; no cross-person association | No change |
| Missing/retired source or caller-mutated cached observation | STALE_HOLD | No change |
| Repeated, omitted or late-added opaque token without new first-binding | UNKNOWN | No change |

This table describes only an observation-text classifier, **not** SUZKA's subjective assessment or the authoritative RelationshipSystem state machine. Subject-owned significance review must be separately defined and source-qualified before any positive revision.

## Temporal and provenance boundaries

- **First-binding-only MVI:** one new caller-claimed opaque token introduced *in* the observed Chat, identified by the before/after retained WAL Context delta. Merely listing an existing token in the current Context proves neither the current event's speaker nor a previous one. A token is not person authentication. After-R27 #346 owns robust actor identification.
- **Freshness:** the test-only triage is re-derived from actual R07/R12/DB1/WAL instead of trusting a caller-owned tuple. Missing historical transition or retired E refuses. Any later ordinary/debug Chat in the same Context, or a later Context-frame change such as suspend/close, makes this first-event result STALE_HOLD. Other Contexts remain independent. This is a conservative same-generation MVI check, not global SourceEpoch.
- **Claimed words are not cognition:** The Suzka response is model-produced visible text in DB1, not an independent Subject-owned internal position or its decision that a relationship issue exists. The user text is caller-authored; two "未決定" utterances can be elicited by one caller. **A future production U2 may not treat these declarations as the sole admission warrant**, even with matching checksums/event ids.
- **After admission:** if a real future Subject revision exists, a later externally sourced resolution must be separately qualified and admitted with a new revision; do not erase original contrary evidence, silently backdate a participant or reclassify historical context. The first-binding-only gate cannot itself guarantee such a second per-person revision.
- **No verified "open now" state:** even the exact finite same-event declaration of an undecided shared task is **observed expressed uncertainty** at E. It is not independently proven perpetual unresolved status. A separate subject review should label it tentative and time-scoped. An unverified user/model assertion that "SUZKA trusts A" or "A is a real verified person" is never an intrinsic state setter.
- **Privacy/capacity:** only bounded IDs, event order, existing Experience witness and finite owner-side interpretation state may be stored after explicit implementation review. No raw transcript, identity profile, or sidecar speaker database enters U1/v10 under this design. Do not change R09/R12/DB1 writers, WAL format or the 128 MiB cap/full 16 MiB reserve for this MVI.

## Tests and non-goals

Tests in `tests/test_r15_u2_issue_assessment_preflight.py` exercise **real isolated R07/R12/DB1/WAL commits**, a deterministic synthetic provider and one closed Q1 grammar. Positive test only produces `PENDING_EXPRESSIONS_ONLY` in local test code and confirms RelationshipSystem remains revision zero. Negatives cover A/B without a joint open decision, one-sided open, aligned A/A, Q1/Q2 mismatch, resolved wording, raw trust attempt, later same Context Chat, another Context, missing same-event token, retired R12 source and altered cached observation.

No runtime scheduler, `RELATIONSHIP_REVIEW` production event, positive Relationship revision, real-model general semantic competence or automatic conversation-style/InteractionStance consumption is demonstrated by this test-only exercise. A15–A17 transient style use is a separate accepted design boundary, not part of this PR.

## Smallest proposed production gate (Human separately approves exact code)

1. Make an exact U2-only source qualifier that returns an internally sealed, finite event-specific **observed interaction** from actual R07/R12/DB1 + retained WAL; do not reuse U3's anonymous-only issuer or accept injected model/user candidates.
2. Define and independently review one **SUZKA-owned subjective significance test** separate from parsed model/user language. A pending *task decision* does not automatically constitute an interpersonal `UNRESOLVED_ISSUE`. The producer must show its own finite, source-backed reason why **the interaction itself** still needs coordination and why the subject adopts that as a tentative relationship issue, then reject stale/resolved/insignificant meaning. If this independent positive case is not reproducible, keep Relationship UNKNOWN and do not force a positive by relabeling A/B mismatch.
3. Only inside a real serialized `RELATIONSHIP_REVIEW` Actor event, let the one `RelationshipSystem` owner optionally install one `UNRESOLVED_ISSUE` `PRESENT` with `Inertia.TENTATIVE` and SourceWitness/RevisionProof, other axes UNKNOWN. Preserve v10/WAL deterministic commit/restart/idempotence; no public setter and no restored-root impersonation.
4. Keep automatic Chat schedule and multi-turn per-person resolution outside this narrow first positive path. Present explicit separate Human implementation scope for #339, run independent code reviews and require a later separate Merge permission.

**Exit:** this test-only design shows that neither mere A/B difference nor textual declarations of a pending shared task are sufficient to create a durable Relationship issue. Genuine SUZKA-owned subjective significance assessment, unresolved-interaction admission and ordinary dialogue consumption remain **unimplemented and untested** until separately authorized and verified.

## A second test-only significance probe: independently owned joint-decision contract

An additional experiment supplies a **fixed test-owned consensus requirement** (not derived from the two Chat utterances). Within that closed experimental task, two different provisional choices imply an uncompleted **joint selection under the stated task rule**. The fixture therefore distinguishes:

- Generic Q1 A/B opinion divergence with **no trusted cooperative-decision rule**: `OBSERVED_DIFFERENCE_ONLY`. The user or model cannot create the trusted rule by text.
- Q1 A/B under the exact independently supplied *test fixture* consensus contract: `PROTOCOL_COORDINATION_GAP_ONLY`. This says **the test task's joint selection criterion is not met**, NOT that the real-world user caused an interpersonal conflict. This is still local classification, not a U2 owner revision.
- Q1 B/B under the same contract: `AGREEMENT_ONLY`; no gap.
- Stale source or absent initial event-time binding, even with the contract: `STALE_HOLD/UNKNOWN`; no gap.

This result clarifies **what additional evidence would be needed** to distinguish task coordination from harmless difference without making model declarations into Relationship authority: a genuine trusted **subject-side task/coordination contract**, and an independently qualified current source event proving whether that contract's bounded coordination condition was fulfilled.

The current R15/MVI **does not have such a production owner-bound cooperative contract for arbitrary chat**. The test uses a private object identity to model an externally trusted test rule; that is **not** authentication, a persistent Source credential, R13 Goal adoption, or a production authorization design. Whether SUZKA can derive an equivalent genuinely Subject-owned significance condition in a narrowly approved U2 producer, or must leave the Relationship axis UNKNOWN and limit itself to event-scoped expression, is the outstanding #339 human implementation/design gate.

Even after that contract, installing `UNRESOLVED_ISSUE` would need a further independently source-qualified, event-scoped **subjective Relationship review**. A merely unfinished joint task may not be a lasting relationship problem; the positive U2 event must identify and defend precisely the *interaction-level* need, not assume all differences lower trust.
