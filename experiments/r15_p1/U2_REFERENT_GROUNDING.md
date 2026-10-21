# R15 U2 — bounded shared-referent significance without R13 Goal authority

**Status:** TEST-ONLY feasibility and proposed future Human design/code gate, 2026-10-21 JST. Parent [#339](https://github.com/upiscium/PROJECT-SUZKA/issues/339) and [#283](https://github.com/upiscium/PROJECT-SUZKA/issues/283); first-binding provenance [Draft PR #344](https://github.com/upiscium/PROJECT-SUZKA/pull/344); robust per-event identity deferred [Post-R27 #346](https://github.com/upiscium/PROJECT-SUZKA/issues/346).

## The scientific distinction: "we prefer different choices" versus "we have not grounded the same reference"

Two agents may rationally prefer A and B, respectively, **without** misunderstanding one another and without an interpersonal problem. A lasting `UNRESOLVED_ISSUE` cannot be inferred from an A/B difference or from two utterances saying a joint task is pending.

There is a narrower, task- and interaction-relevant first-person case: the *same referring expression*, e.g. **「前の案」**, is reported by the interlocutor as B but is interpreted in SUZKA's visible response as A. In that event, the system has concrete evidence of **different expressed mappings for a shared expression**. Under a code-reviewed **R15-local pragmatic interpretation rule**, SUZKA could provisionally represent: *"the shared reference may require clarification before I rely on it."* This is an **epistemic interaction uncertainty**, not disagreement about which option is better, an assertion of someone's true mind, or objective actor blame.

This candidate has a better R15-specific meaning than a generic "joint selection requires consensus" rule, which would be a task-level Goal/Decision rather than a Relationship claim.

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
2. **Ephemeral private epistemic candidate (TEST-ONLY):** If and only if a hard-coded **research-only, local R15 rule** is present, a pure test function maps the qualified mismatch to `TENTATIVE_CLARIFICATION_ONLY`. It means "a clarification **might** be useful" for this event. The caller cannot supply a model/user prose value or an arbitrary object as this test criterion. The rule is an **in-memory fixture sentinel**, not a production owner, a credential, a Subject adoption, R13 Goal, or persistent source.
3. **Actual SUZKA-owned assessment (MISSING):** A new, code-reviewed R15 subject producer would need to evaluate the **pragmatic relevance of the mismatch to this interaction**, requalify the original committed evidence, and independently decide whether it warrants *ephemeral caution* or a **genuinely unresolved interaction-level hypothesis**, without taking the model response as self-authenticating private cognition.
4. **Durable `UNRESOLVED_ISSUE` (MISSING):** Only if the approved internal actor review forms a source-qualified tentative *interaction* issue, not simply an option preference, may `RelationshipSystem` adopt one revision. The one Experience witness is mandatory, Source Epoch/identity/alias cannot be invented, 5 unrelated axes stay UNKNOWN, R15 v10/StateWAL owns transaction persistence and restart/rollback.

Every experimental case leaves `RelationshipSystem` revision 0 and R13 GoalSystem with no adopted Goal. The text grammar is fixed; broad real-model semantic generalization has NOT been established.

## Negative matrix / finite first attempt

- **Same referent A/A or B/B:** no ambiguity and no concern candidate.
- **Same expression reported A vs SUZKA interpreted B:** only a provisional clarification *research* candidate under the internal *test* rule, never a durable Relationship issue.
- **Separate subjective preferences A/B:** no observed common referent mismatch; never an issue candidate.
- **Different expressions, incomplete response, or a user asserting "Suzka dislikes me":** UNKNOWN, not a subjective authority.
- **No rule, or a different caller-provided object:** observation can exist, but no Subject interpretation candidate.
- **Missing same-event new-token evidence / later unkeyed Chat with an existing Context member:** UNKNOWN, never retroactive attribution.
- **Later Chat in the same Context:** mark the old observation STALE_HOLD before any new event-level reading; cannot prove whether later chat resolved it or who spoke.
- **Real R10 global novelty/affect, with social relevance missing:** no actor-specific relationship significance inferred.
- **No R13 adopted Goal:** no new Goal is manufactured by this experiment.
- **No actual persisted Subject producer:** no Relationship revision, no verified speaker, trust, intimacy, objective correctness, competence, Goal adoption or Decision change.

## Proposed smallest production code architecture (requires new explicit Human approval)

**Preferred: R15-local referent-grounding interpretation**, not a new R13 Goal or R10 actor-specific emotion issuer.

- A sealed private `QualifiedInteractionEvidence` reader joins actual terminal R07, active committed R12 Experience, matching DB1 visible turn, exact retained StateWAL event transition/Context first binding. Do not loosen U3's anonymous-only Source gate.
- A finite, reviewed **R15 cognitive policy** about *shared-reference uncertainty* assesses the event from SUZKA's first-person perspective. It is code/version owned within R15, **never** settable by HTTP, user text, an operator policy override, model rationale, fabricated `SourceWitness`, or arbitrary task semantics. Observed model output is input evidence, not the subjective adoption. The policy must distinguish **ephemeral clarification** from a *persistently perceived issue*, and support UNKNOWN for unclear current significance.
- One `RELATIONSHIP_REVIEW` internal Actor event first requalifies current Source and checks bound event generation. Only the bounded, separately justified persistent interaction issue creates a true `UNRESOLVED_ISSUE` `PRESENT`/`TENTATIVE` with qualified original Experience witness and typed RevisionProof, through existing v10/WAL. No trust/reciprocity/person assertion and no per-turn speaker inference. Missing issue significance leaves Relationship unchanged.
- Restart, same-event duplicate, WAL baseline loss, later Context mutation, source retirement, forged participant/rule, model pressure, handler rollback and exact full reserve constraints must pass.
- A15–A17 event-scoped expression may consume a *fresh* qualified temporary interpretation after a separately reviewed runtime implementation; a model is not authorized to rewrite intrinsic state by generating a phrase. No automatic Chat scheduler or real general semantic understanding is claimed until explicitly tested.

**Alternative:** Keep the entire interaction concern as ephemeral/UNKNOWN until R16 brings real Subject-admitted Goal/Decision provenance. This is safer but does NOT satisfy the current R15 MVI requirement for at least one **positive** Relationship revision without an explicit Human scope amendment; do not falsely mark U7/R15 PASS.

## Decision gate

An R15-owned finite **shared-referent coordination significance rule** is the next bounded production design candidate, stronger than a generic A/B opinion mismatch but much smaller than a general other-mind/identity model. It must be independently reviewed to ensure that its positive criterion is a *genuine provisional relationship-level subjective issue* rather than merely an unfinished task or parser shape.

**No production code, Subject assessment/Relationship revision, new U1 schema, Source, deployment, Merge or real-model proof is delivered by this test-only experiment.** The user-approved MVI priority and accepted D9/A1–A27 restrictions remain in force. A separate `#339` Human code gate and subsequent Merge gate are mandatory.
