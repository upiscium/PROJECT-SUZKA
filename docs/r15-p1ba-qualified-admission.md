# R15 P1b-A: bounded, serialized positive NarrativeSelf (MVI)

Human code approval: Issue #326, comment 6245092479.
Prior experiment-only evidence: Draft PR #327.
Roadmap: Issue #323 MVI; Post-R26 advanced verification: Issue #320.

## Responsibilities

**Owns:** a first narrow, **process-local**, source-qualified subject
interpretation update. Only the closed A/B/UNKNOWN referent scheme from
a recorded source-owned comparison is supported, not general autobiography.

**May:** on a distinct, internal serialized AgentRuntime NARRATIVE_REVIEW
event, read and re-qualify the actual source owner's terminal R07 Chat
event, committed R12 Experience, matching DB1 input/visible answer and
the original anonymous R09 Context. Build one new immutable U1
NarrativeEpisode and RevisionProof, and publish a new U3 in-memory
root atomically only after validation and exact serialization checks.

**Must not:** install user/model/operator-supplied SourceWitness,
NarrativeSelfState, trust, personality, capability, identity, or an
unverified real-world intent. Model-generated candidate JSON is NOT
intrinsic truth without matching the subject's recorded first-person
response and independently terminal sources. No generic adoption setter,
R15 public API, imported experiment code, production Source manipulation,
source owner implementation edit, or general semantic ClaimMeaning
producer is introduced.

**Depends on:** existing R07 Journal, R12 ExperienceStore, committed DB1
episodic record, R09 ContextRegistry, and the same process-bound Runtime.
Source owner reads may have normal physical maintenance side effects.
They are NOT certified G2 no-repair reads; that optional advanced
requirement remains Post-R26.

**Used by:** P1b-A isolated process-local positive owner tests only.
A future separately approved U5/U6 integration must capture/persist
the U3 root, restore its source-scoped option mapping, bind it to
StateWAL and real Decision prompt consumption before claiming durable
NarrativeSelf.

## Transition

(1) A new Runtime NARRATIVE_REVIEW event receives only owner-verified
event/episode references to setup, original E and pre-E2 model S1.
The Source gate checks the original response selected one A/B option
and a strictly shaped S1 candidate described that same option.
The R15 owner creates a tentative S1 episode and a distinct
subject-event RevisionProof (revision 1). This is not an operator setter.

(2) After independently terminal E2 and S2 Chat events, another
dedicated Runtime event reads all five sources again. The user E2
statement may establish what the user *said*, not the real-world intent.
If the source-backed referent differs from the S1 choice and the
provisional S2 model candidate agrees with the observed user
statement, the owner creates a CONTESTED S2 revision (revision 2)
linked to the unchanged S1 and retains E as counterevidence.

(3) No later E2: no S2 revision. Duplicate qualified source references
are rechecked and do not create a new revision. Missing source,
wrong event type/source, forged IDs, unqualified participant
identity, out-of-order events and reentrant mutations fail closed.

The private current option can be reconstructed with the original
committed setup episode identity and bounded option list.
A standalone SHA-256 statement ID is NOT self-describing meaning
or a security credential.

## Deliberately NOT complete in P1b-A

- The narrow owner instance is currently wired only by isolated code/tests.
  No automatic Chat/MainLoop production trigger or user-visible API.
- In-process option-source scope and current U3 root are NOT captured by
  AgentState v10 or StateWAL; restart/replay is NOT yet guaranteed.
  A pure U1 JSON roundtrip is not WAL recovery.
- Existing generic U3 narrative_availability remains UNAVAILABLE for
  unrestricted narrative claims, person identity, traits and competence.
- Relationship and SelfModel genuine positive state, complex meaning,
  actual Decision consumption and full-R15 acceptance are later tasks.
- Independently reviewed U5/U6 persistence and state budget gate must
  respect the accepted 128 MiB hard cap and untouched 16 MiB reserve.

## Gates

Source correctness/security review, exact-head CI, negative/replay tests,
Human Ready/Merge and any later code touching R07/R09/R12/U1/U5/U6 require
separate authorization. This document is not installation permission or
a source integrity attestation.

Focused verification: pytest tests/test_r15_p1b_positive.py plus existing
R15 U1/U3 and R15 P1a/P1b experimental regressions; Ruff and mypy.
