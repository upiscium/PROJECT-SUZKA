# R15 U2 — observed difference, provisional unresolved interaction and admission (TEST-ONLY DESIGN)

Date: 2026-10-21 JST. Parent: [Issue #339](https://github.com/upiscium/PROJECT-SUZKA/issues/339); evidence: [Draft PR #344](https://github.com/upiscium/PROJECT-SUZKA/pull/344), merged [U4b #345](https://github.com/upiscium/PROJECT-SUZKA/pull/345), Post-R27 real speaker/person identification [#346](https://github.com/upiscium/PROJECT-SUZKA/issues/346).

**Decision accepted by Human:** Do not equate A/B positions with an intrinsic Relationship problem. Preserve observation, interpretation and durable adoption as three different authorities. Robust per-message speaker/person identification is deferred Post-R27. **No source code, schema, deployment, Subject state, GitHub Ready or Merge authorization is implied here.**

## Observable three-stage behavior

1. **Observed difference:** The actual serialized R07 CHAT/API_CHAT terminal/transaction, active committed R12 Experience and DB1 episode carry the visible user and Suzka positions for the **same fixed task**; StateWAL's exact event candidate supplies a newly introduced caller-claimed opaque token (distinct from persistent Context membership). The only claim is that two provisional choices were visibly expressed. The test-only result is bounded; no verified real person, intention or correct answer.
2. **Subject-review candidate, not a decision:** If and only if the event is still source-eligible, both *visible* positions differ, the same narrow cooperative task is at issue, both turns **expressly leave its shared decision undecided**, the Suzka response explicitly declares objective correctness unverified, and no later CHAT in the same Context has displaced the event, it may produce `REVIEW_CANDIDATE_ONLY`. The finite rule is a provisional assessment that the joint choice needs clarification, **not a claim that an interpersonal conflict is objectively established**. It is never a generic regex LLM-competence or human-intent classifier.
3. **Subject-owned durable admission — NOT IMPLEMENTED:** An independently reviewed internal `RELATIONSHIP_REVIEW` event must requalify all source owners *again* and perform its own current subject assessment before staging any genuine `RelationshipSystem` revision. Only it may write one tentative, interaction-scoped `UNRESOLVED_ISSUE` from a real SourceWitness and a real event-bound RevisionProof through existing AgentState v10/StateWAL, while all five other axes stay UNKNOWN. This test-only candidate is **not a license to call an existing generic owner mutation or install a forged U1 root**. `VerifiedPerson`, mutual feeling, trust, causality and verified competence are strictly outside scope.

## Explicit interpretation state machine (test-only)

| Source-qualified observation | Assessment | Durable Relationship |
|---|---|---|
| Same task A/A | AGREEMENT_ONLY | No change |
| Same task A/B without a pending shared decision | OBSERVED_DIFFERENCE_ONLY | No change |
| Same task A/B and only one side declares an open decision | OBSERVED_DIFFERENCE_ONLY | No change |
| Same task A/B with both sides declaring the cooperative decision undecided and Suzka explicitly unverified | REVIEW_CANDIDATE_ONLY | No change until separate Subject owner event and Human-approved producer |
| Same task A/B but resolution already asserted in this event | UNKNOWN/refuse closed protocol | No change |
| User Q1 vs Suzka Q2 | UNKNOWN | No change |
| Later CHAT in same Context after source E | STALE_HOLD | No change: later turns lack per-message speaker evidence |
| Different Context's later Chat | Candidate remains scoped to original Context; no cross-person association | No change |
| Missing/retired source or caller-mutated cached observation | STALE_HOLD | No change |
| Repeated, omitted or late-added opaque token without new first-binding | UNKNOWN | No change |

This state machine is a conservative design/test **eligibility classifier**, not the authoritative RelationshipSystem state machine.

## Temporal and provenance boundaries

- **First-binding-only MVI:** one new caller-claimed opaque token introduced *in* the observed Chat, identified by the before/after retained WAL Context delta. Merely listing an existing token in the current Context proves neither the current event's speaker nor a previous one. A token is not person authentication. After-R27 #346 owns robust actor identification.
- **Freshness:** the test candidate is re-derived from actual R07/R12/DB1/WAL at assessment time, rather than trusting a caller-owned tuple. Missing historical transition or retired E refuses. If a later Chat occurs in the same Context, do not assume whether it resolved the issue or came from the same person; HOLD the earlier candidate.
- **After admission:** if a real future Subject revision exists, a later externally sourced resolution must be separately qualified and admitted with a new revision; do not erase original contrary evidence, silently backdate a participant or reclassify historical context. The first-binding-only gate cannot itself guarantee such a second per-person revision.
- **No verified "open now" state:** even the exact finite same-event declaration of an undecided shared task is **observed expressed uncertainty** at E. It is not independently proven perpetual unresolved status. A separate subject review should label it tentative and time-scoped. An unverified user/model assertion that "SUZKA trusts A" or "A is a real verified person" is never an intrinsic state setter.
- **Privacy/capacity:** only bounded IDs, event order, existing Experience witness and finite owner-side interpretation state may be stored after explicit implementation review. No raw transcript, identity profile, or sidecar speaker database enters U1/v10 under this design. Do not change R09/R12/DB1 writers, WAL format or the 128 MiB cap/full 16 MiB reserve for this MVI.

## Tests and non-goals

Tests in `tests/test_r15_u2_issue_assessment_preflight.py` exercise **real isolated R07/R12/DB1/WAL commits**, a deterministic synthetic provider and one closed Q1 grammar. Positive test only produces `REVIEW_CANDIDATE_ONLY` in local test code and confirms RelationshipSystem remains revision zero. Negatives cover A/B without a joint open decision, one-sided open, aligned A/A, Q1/Q2 mismatch, resolved wording, raw trust attempt, later same Context Chat, another Context, missing same-event token, retired R12 source and altered cached observation.

No runtime scheduler, `RELATIONSHIP_REVIEW` production event, positive Relationship revision, real-model general semantic competence or automatic conversation-style/InteractionStance consumption is demonstrated by this test-only exercise. A15–A17 transient style use is a separate accepted design boundary, not part of this PR.

## Smallest proposed production gate (Human separately approves exact code)

1. Make an exact U2-only source qualifier that returns an internally sealed, finite event-specific **observed interaction** from actual R07/R12/DB1 + retained WAL; do not reuse U3's anonymous-only issuer or accept injected model/user candidates.
2. Make an independently reviewed bounded Subject assessment that separates ordinary difference from **provisionally still-undecided, joint decision** and rejects stale/resolved/incomplete meaning. Implement one positive example plus all negative rows above, no verified person.
3. Only inside a real serialized `RELATIONSHIP_REVIEW` Actor event, let the one `RelationshipSystem` owner optionally install one `UNRESOLVED_ISSUE` `PRESENT` with `Inertia.TENTATIVE` and SourceWitness/RevisionProof, other axes UNKNOWN. Preserve v10/WAL deterministic commit/restart/idempotence; no public setter and no restored-root impersonation.
4. Keep automatic Chat schedule and multi-turn per-person resolution outside this narrow first positive path. Present explicit separate Human implementation scope for #339, run independent code reviews and require a later separate Merge permission.

**Exit:** this test-only design shows that "difference does not imply durable issue" is implementable on a closed example. The unresolved subjective admission/ordinary dialogue behavior is still **untested until a separately authorized U2 producer and consuming runtime path exist**.
