# R15 U2 — shared-referent observation and event-scoped clarification (TEST ONLY)

**Status:** TEST-ONLY feasibility and proposed future Human design/code gate, 2026-10-21 JST. Parent [#339](https://github.com/upiscium/PROJECT-SUZKA/issues/339) and [#283](https://github.com/upiscium/PROJECT-SUZKA/issues/283); first-binding provenance [Draft PR #344](https://github.com/upiscium/PROJECT-SUZKA/pull/344); robust per-event identity deferred [Post-R27 #346](https://github.com/upiscium/PROJECT-SUZKA/issues/346).

## The scientific distinction: "we prefer different choices" versus "we have not grounded the same reference"

Two agents may rationally prefer A and B, respectively, **without** misunderstanding one another and without an interpersonal problem. A lasting `UNRESOLVED_ISSUE` cannot be inferred from an A/B difference or from two utterances saying a joint task is pending.

The narrower experimental case concerns the **same referring expression**, e.g. **「前の案」**. A user *reports* B while the committed, **model-generated visible reply states** that SUZKA interpreted A. This is evidence of two **expressed mappings**, **not** independent evidence of SUZKA's internal interpretation. The user could have instructed the model to emit the entire answer; the synthetic provider in this test does just that. Moreover, the user's same-event explanation of B may already have resolved the ambiguity. The only supported candidate is **ephemeral clarification / checking**, not that an enduring issue exists.

**Owner boundary:** SUZKA's own reinterpretation of an ambiguous referent already belongs to **U3 NarrativeSelf**. U2 cannot claim another enduring relationship state from that same U3 episode merely because the referent mapping differed. A truly U2-owned claim would need *additional, independently qualified interpersonal significance*, not only a task-choice or referent parser output.

## Evidence this repository presently owns

| Owner | Observed / available | What it cannot issue |
|---|---|---|
| R07 + R12 + DB1 + StateWAL | Actual completed Chat lineage, committed visible user/model text, exact event-time R09 Context and first *new* opaque token binding | Authenticated person or each later turn's speaker |
| R10 Appraisal | Real DEBUG_CHAT run uses `AppraisalSignals()`; social_relevance, certainty and goal_progress are currently `None` | A trustworthy interpersonal significance/actor-specific emotional-cause signal |
| R13 GoalSystem | One real isolated Chat showed zero adopted Goal records; `GoalSubjectAdmission` is a proof-shaped contract with an independent R16 future producer | An already-adopted Q1 joint Goal or an authenticated code policy chosen by SUZKA |
| R15 U1 `InteractionStance` | Closed, event-scoped felt-association *type* and explicit actor-causation UNKNOWN; only pure construction/rejection tests | A current qualified producer, runtime style consumer or durable subjective Relationship event |
| R15 RelationshipSystem | UNKNOWN multi-axis root, typed replay/proof and existing v10/WAL owner | A source-qualified positive subjective axis under a true `RELATIONSHIP_REVIEW` event (not yet implemented) |

## Test-only design and exact epistemic levels

The file `tests/test_r15_u2_referent_significance_preflight.py` exercises one **fixed, deliberately narrow** shared-expression grammar against actual isolated R07/R12/DB1/StateWAL using a deterministic synthetic language provider.

1. **Observed:** The user-visible source reports B for expression 「前の案」 while SUZKA's committed visible response interpreted A. The source qualifier also checks that the same serialized Chat newly bound one caller-claimed opaque participant token. The result describes **two expressed mappings**, not a verified user's intention or an authenticated SUZKA inner belief.
2. **Ephemeral private epistemic candidate (TEST-ONLY):** If and only if a hard-coded **research-only, local R15 rule** is present, a pure test function maps the qualified mismatch to `TENTATIVE_CLARIFICATION_ONLY`. It means "a clarification **might** be useful" for this event. The research rule is a **plain importable in-memory object sentinel** whose identity is compared in the test. This is **not access control**, and a test caller with module access can pass the sentinel. It is not a production owner, credential, private Subject assessment, R13 Goal, or durable Source issuer.
3. **Actual SUZKA-owned assessment (MISSING):** A separately reviewed producer would need a **structured owner-produced first-person referent** (or another independently validated Subject observation), not model prose that the user can dictate. It must requalify Source and determine whether the interaction warrants only ephemeral clarification; this fixture does **not** establish a genuinely still-unresolved issue.
4. **Durable `UNRESOLVED_ISSUE` (MISSING, NOT JUSTIFIED BY THIS CASE):** A future `RelationshipSystem` revision would require a **different, independently qualified positive interpersonal case** or proof that an interaction-level concern persists at the authorized review time. The same-event user already described B; an additional next Chat invalidates the historical first-binding candidate. This fixture must **never** be used by itself as a positive durable U2 warrant. Any future approved writer would need the actual Experience witness, internal Actor revision proof, existing v10/StateWAL and all other axes UNKNOWN.

Every experimental case explicitly checks that the actual `RelationshipSystem` revision stays 0. Representative real DEBUG_CHAT and main acceptance fixtures also check that the R13 GoalSystem has **no adopted Goal**; this is not a claim that every test asserts the R13 state. The text grammar is fixed and no general real-model semantic ability is established.

## Negative matrix / finite first attempt

- **Same referent A/A or B/B:** no ambiguity and no concern candidate.
- **Same expression, user-reported A versus visible SUZKA response claiming B:** only an ephemeral clarification **research** candidate, never a durable Relationship issue or an authenticated SUZKA internal belief.
- **Separate subjective preferences A/B:** no observed common referent mismatch; never an issue candidate.
- **Different expressions, incomplete response, or a user asserting "Suzka dislikes me":** UNKNOWN, not a subjective authority.
- **No rule, or a different caller-provided object:** observation can exist, but no Subject interpretation candidate.
- **Missing same-event new-token evidence / later unkeyed Chat with an existing Context member:** UNKNOWN, never retroactive attribution.
- **Later Chat in the same Context:** mark the old observation STALE_HOLD before any new event-level reading; cannot prove whether later chat resolved it or who spoke.
- **Real R10 global novelty/affect, with social relevance missing:** no actor-specific relationship significance inferred.
- **No R13 adopted Goal:** no new Goal is manufactured by this experiment.
- **No actual persisted Subject producer:** no Relationship revision, no verified speaker, trust, intimacy, objective correctness, competence, Goal adoption or Decision change.

## Proposed smallest production code architecture (requires new explicit Human approval)

**Possible follow-up: ephemeral referent-grounding interpretation**, owned by U3/U6 expression where appropriate, *not* a positive durable U2 Relationship producer. No new R13 Goal or R10 actor-specific emotion issuer is authorized.

- A sealed private `QualifiedInteractionEvidence` reader joins actual terminal R07, active committed R12 Experience, matching DB1 visible turn, exact retained StateWAL event transition/Context first binding. Do not loosen U3's anonymous-only Source gate.
- A future finite R15 first-person pragmatic review must reconstruct SUZKA's **actual structured Subject-owned referent state** or a comparably qualified internal producer; visible model text is not such a state. A reviewed ephemeral clarification rule could distinguish a different asserted interpretation from a difference of preference. Unknown owner evidence => no Subject truth admission.
- A later `RELATIONSHIP_REVIEW` may admit a durable hypothesis **only after an independent, positively evidenced interpersonal example has passed the Human design/code gate**; the referent fixture here does not qualify. R07/R12/DB1/WAL requalification, a real internal Actor event and R15 `RelationshipSystem` typed RevisionProof are necessary but **not sufficient**. Until a separate genuine significance source exists, keep all Relationship axes UNKNOWN.
- Restart, same-event duplicate, WAL baseline loss, later Context mutation, source retirement, forged participant/rule, model pressure, handler rollback and exact full reserve constraints must pass.
- A15–A17 event-scoped expression may consume a *fresh* qualified temporary interpretation after a separately reviewed runtime implementation; a model is not authorized to rewrite intrinsic state by generating a phrase. No automatic Chat scheduler or real general semantic understanding is claimed until explicitly tested.

**R15 acceptance consequence:** Keeping this referent concern ephemeral/UNKNOWN is currently the only source-justified result. It does **not** by itself satisfy the R15 MVI requirement for at least one genuinely admitted, positive Relationship change. If no other bounded interpersonal source can be established, report U7/R15 **INCOMPLETE** and request an explicit Human alternative-positive-example or scoped acceptance amendment. Do not force positivity simply to meet the roadmap.

## Decision gate

The referent-grounding case is suitable as **test-only evidence and a possible ephemeral style-consumption example**, **not** yet as a production `RelationshipSystem.UNRESOLVED_ISSUE` positive. The next #339 design choice is to find a genuinely interpersonal Subject-owned significance source, or explicitly adjust the positive acceptance contract. Both choices require a separate Human gate; no U1/v10 owner is silently expanded.

**No production code, Subject assessment/Relationship revision, new U1 schema, Source, deployment, Merge or real-model proof is delivered by this test-only experiment.** The user-approved MVI priority and accepted D9/A1–A27 restrictions remain in force. A separate `#339` Human code gate and subsequent Merge gate are mandatory.


## 2026-10-21 independent reviewer corrective disposition

Two independent source-only reviewers found no P1/P2 **in the test-only scope**, but cautioned against treating this result as a durable U2 positive. The user's *same event* may already disambiguate the referent, the only SUZKA-side evidence is **model-generated text susceptible to caller steering**, and any later Chat forces a new event's first-binding provenance to HOLD. Thus a genuine still-open multi-turn interpersonal issue is **not demonstrated**.

- **First-party U3 overlap:** the past-self referent interpretation and reconstruction of why it changed belong primarily to NarrativeSelf, not Relationship. A real U2 revision needs distinct interaction-level significance.
- **Raw caller-token poisoning:** any client can choose an opaque `interlocutor_key` identical to an existing person's purported token in a new Context. No durable cross-Context person/relationship ID or trust hypothesis may be established from the raw token. Post-R27 #346 owns robust identity; any earlier limited Context-scoped Subject association must first have a separately approved *non-colliding, owner-bound scope* that cannot affect other sessions.
- **Prompt-induced self-report:** a user can cause the model to repeat the exact phrase `Suzkaは…案Aと解釈…`; successful regex does not prove SUZKA had that inner representation. Future positive cognition must come from structured, source-qualified Subject state, not generated prose.
- **Freshness:** an event at which the user explains B is not necessarily unresolved; an E2 in the next Chat makes the E candidate stale under this first-binding-only test. Persistent positive admission is not validated by the present path.
- **Fixed exercise cue:** SUZKA's production U6b has a separate special private NarrativeSelf referent-recall cue. The test text intentionally avoids its other required exercise terms; any future task expansion must test both owner boundaries and avoid accidentally invoking a private U3 interpretation reader for an opaque-tagged Context.
- **Test-limits:** the deleted-record negative simulates a missing R12 physical file, not an `ExperienceLifecycle.RETRACTED/SUPERSEDED` transition. Several unusual WAL/Journal exceptions are now handled as a test-only HOLD, but this is not a production SourceEpoch proof. The research sentinel is importable and is never a security credential.

**Revised recommendation:** retain this exact bounded technical experiment and protect it with negative tests; do NOT use it to implement lasting U2 automatically. Proceed to a separate code-reviewed **R15-owned interpersonal significance** source candidate and re-evaluate the MVI positive requirement before authorizing a U2 Relationship writer.
