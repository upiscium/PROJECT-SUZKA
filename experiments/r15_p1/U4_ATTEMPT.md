# R15 U4 — source-qualified task-attempt candidate (TEST ONLY)

Parent: Issue #334, R15 #283, MVI delivery #323.

**Purpose:** Check whether the already accepted U1 CapabilityHypothesis can represent one observed attempt to resolve the previous-option referent, while VerifiedCompetence stays UNKNOWN. This is not an evaluation of actual task success, general intelligence, personality or emotion.

**Owns:** Evaluator-side typed candidate and disposable cooperative source reads of terminal R07 Chat, active R12 Experience, actual DB1 question/visible answer, and matching anonymous R09 Context. It reads the actual bound in-process U3 owner (not a caller-created U1 root) and independently reopens E/E2 to verify the observed meaning.

**May:** Produce a DETACHED test-only canonical SelfModelState containing an attempted=PRESENT CapabilityHypothesis for one task class, with confidence in the observed occurrence of that attempt, while verified_competence stays UNKNOWN. Confidence does not measure task accuracy.

**Must not:** Write to SelfModelSystem, AgentState v10, WAL, Journal, Context, Source, public API, Goal/Decision, Relationship, installed service or real user data. No artificial RevisionProof is made for an imagined U4 owner update: the detached candidate intentionally has revision=0. The actual U4 owner remains empty.

**Depends on:** Actual disposable SUZKA E→S1→E2→S2 R07/R12/DB1 owners and existing finite U1 syntax. The task class is limited to task:referent-resolution. E2 is a user-reported referent, not objectively verified hidden intent.

**Used by:** A future narrowly approved production U4 attempted-task owner admission. That requires a distinct serialized Actor review, real RevisionProof, idempotence, U5 WAL internal commit/rollback and restart/replay.

## MVI conclusion

- No new U1 schema field is needed for one source-backed attempted task with unverified competence; existing CapabilityHypothesis, Interpretation and SelfModelState support it.
- A real positive SelfModelSystem adoption does NOT exist yet. Separate exact Human implementation approval is necessary to add one first-person attempt producer and normal WAL capture, without any generic competence setter.
- One mistaken answer does NOT license personality, confirmed limitations of general skill, or failure rate estimates. VERIFIED competence remains UNKNOWN until later R17/R18 independent outcomes.
- U2 Relationship must not infer an identified human from the anonymous U3 experiment. A separate event-time R09 participant reference/source proof is needed for any per-person relationship update; do not create trust, closeness, or reciprocity values merely from a user correction.

## Verification

Use python -m pytest -q tests/test_r15_u4_attempt_preflight.py, Ruff on the new two Python files, and scoped Mypy.

**Status:** TEST ONLY. Neither production U4 mutation nor whole R15 MVI acceptance is claimed. No merge/release approval.
