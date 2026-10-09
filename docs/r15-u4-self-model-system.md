# R15 U4 — process-local SelfModel UNKNOWN boundary / #288 DESIGN HOLD (#289)

U4 follows [Human-accepted U1](https://github.com/upiscium/PROJECT-SUZKA/pull/285#issuecomment-6073619240),
[U2](https://github.com/upiscium/PROJECT-SUZKA/pull/285#issuecomment-6078285032)
and [bounded U3](https://github.com/upiscium/PROJECT-SUZKA/pull/285#issuecomment-6080241548)
on the same Draft PR. The missing [joined R07/R12/event-time R09 + Suzka-owned
semantic producer, Issue #288](https://github.com/upiscium/PROJECT-SUZKA/issues/288)
remains **DESIGN REQUIRED / positive admission BLOCKED**. U4 does not change
accepted U1–U3 or R09–R14 source contracts, no upstream implementation is
authorized by #288, and no production caller consumes the new U4 owner.

## Responsibility

- **Owns:** finite process-local immutable U1 `SelfModelState`, starting at
  `SelfModelState(1,0,(),(),None,())`; detached private root/view and an
  untracked task query which reports attempted-status `UNKNOWN`, subjective
  confidence **null**, verified competence `UNKNOWN`. Typed source/semantic/
  admission availability; bounded, no-change request receipts and replay
  distinction. No claim/attempt is generated merely to fill a read view.
- **May:** classify a source-shaped witness as requiring an independently
  verified source root; always report actual SelfModel admission and current
  semantic producer `UNAVAILABLE`; acknowledge a source-free boundary-review
  *request* with caller-supplied event identity and unchanged U1 root.
- **Must not:** create `PRESENT` identity, role, trait, limitation or task
  attempt from operator/model/Web text, R10 Emotion, R11 Value, Belief, R13
  Goal/Commitment, R14 Metacognition or an ACTIVE-looking R12 witness; infer
  verified task success or probability from Tool/Decision/handler fluency;
  mutate U2 trust, U3 narrative, R09–R14 truth or R16–R19 authority. No
  `set_identity`, `adopt_trait`, `add_attempt`, `verify_competence`, restore
  port, direct operator-facing edit, v10/WAL/Journal, MainLoop/API/prompt.
- **Depends on:** accepted U1 closed `SelfClaim`/`ClaimMeaning`,
  `CapabilityHypothesis` and `VerifiedCompetence.UNKNOWN`; the separate
  approved source/semantic/subject producer from #288 does **not yet exist**.
  A real ordered AgentRuntime caller must later authenticate event identity.
- **Used by:** possible later reviewed joined source owner and U5/U6 only;
  no production consumer or autonomous model policy exists today.

## Exact availability and D9 proof boundary

| Claimed input | Actual current source/semantics | U4 result |
| --- | --- | --- |
| R11 Value/IdentityOrigin | Admission into Value truth, not intrinsic identity/role/trait or a first-person task report. Seeds may be system-authorized, never automatic Suzka personality. | `UNAVAILABLE` for SelfModel admission; do not rewrite Values. |
| R12 Experience/Belief | Experience retains chat event/episode/Context/appraisal/Emotion, **not a task class, attempt or task outcome**. Belief owns adopted proposition, not R15 trait meaning. `load_committed_current()` may recover files/change permissions; R07 Journal alone lacks the artifact/event-time Context. | A syntactically ACTIVE Experience yields only U1 `REQUIRES_TRUSTED_ROOT` for a self/attempt **candidate**. It is **not** `SOURCE_AVAILABLE`, first-person semantic proof or accepted SelfModel mutation. |
| R13 Motivation/Goal/Commitment | A typed request/Goal lifecycle or handler outcome assertion is not an R17/R18 empirically qualified task-result producer. | No confidence from a Goal completed label, Tool response or Commitment fulfilled claim. |
| R14 Attention/U3 Metacognition; R10 Emotion | Event-scoped focus/resource heuristic and global core affect, not enduring trait or verified performance; R14 U3 is production-unwired. | No Emotion turn becomes a stable trait; no Attention ranking/Metacognition feedback mutation. |
| R17/R18 verified outcomes | Not implemented for accepted R15 baseline. | `VerifiedCompetence.UNKNOWN` **only**, not 0% or proof of failure. |
| Operator/model/Web text, digest, privileged correction, fake subject origin | Neither serialized transport nor syntactically valid `SourceWitness` grants Suzka subject admission (D9 absolute MUST NOT). | No direct/edit API and no positive state mutation. |

`self_availability(SelfCapability, SourceWitness?, ClaimMeaning?)` returns
separate `source`, `semantic`, final `admission` and `verified` fields.
An eligible-looking Experience can be **`REQUIRES_TRUSTED_ROOT` only** for a
declared candidate; rejected/stale/operator/model sources are `UNAVAILABLE`.
U1 `current_semantic_producer(ClaimMeaning)` stays **`UNAVAILABLE` for every
closed identity/role/trait/limitation predicate**. The final admission is
always `UNAVAILABLE`; verified competence is always `UNKNOWN`, even with an
operator-supplied canonical hash, repeated claim or supposed subject
endorsement. Neither absence nor an epistemic unknown silently becomes
measured success probability zero.

An *attempted-task hypothesis* differs from verified competence: U1 can
encode subjective uncertainty in `CapabilityHypothesis.attempted` separately
from `VerifiedCompetence.UNKNOWN`. U4's production owner has **no attempt
writer** because current R12 chat Experience has no typed task-class attempt
or side-effect-free finalized source/subject join. Accepted U1 also does not
bind `attempted.statement_id` to `task_class_key`; a future reviewed producer
must define and prove that deterministic identity before accepting a row,
without silently rewriting frozen U1. Test-only fixtures exercising attempted
`PRESENT`, opposite `SelfClaim` polarity, `CONTESTED` counterevidence and
`LATENT/REACTIVATABLE` accessibility prove **representation only**, not
Suzka's self-truth or observed task success. Functional accessibility does
not delete R12 Experience, change R08 membership/R14 focus, or replay
cognition on technical restore.

## Process-local event and result envelopes

`SelfModelSystem.review_boundary(event)` copies the caller `EventRef` and
records a **no-change availability-review request** with a bounded checksum
and process-local receipt only. It never acts on Experience, a trait, an
action or a task. At most **256** receipts are kept without eviction; exact
duplicate returns the original result without new receipts or root revision,
while conflicting reused ID/sequence, older/out-of-order time, reentrancy
(including caught replay reentrancy) and capacity overflow fail closed.
Cross-thread calls are serialized: an exact concurrent duplicate gets one
accepted receipt and one replay; a conflicting request rejects. No `RevisionProof`, history anchor or
source-owned evidence is fabricated. Returned roots/views/receipts are
detached; caller mutation of nominally frozen event/receipt cannot change
retained evidence.

`SelfBoundaryResult` requires the exact receipt result revision/digest to
match its U1 snapshot and full read view. A private pure `_result_for()`
would instead return receipt-only `SelfHistoricalAcknowledgement` if a later
root differed: no old receipt may be paired with a newer snapshot. Production
U4 **cannot install** a nonempty root, so that historical branch is tested
only using a clearly **test-only synthetic U1 root**, never a mock accepted
source or proof of runtime implementation. Later U5/U6 must establish real
ordered event, source-finality, durable receipt/restore and true rollback
behavior; U4 claims process-local request identity only.

## Capacity, failure/recovery and privacy

No accepted U1 schema/bound is changed. Full-legal projected v10 maximum
remains **133,726,597 B**, with all **16,777,216 B** future reserve intact
and **491,131 B** extra headroom below **134,217,728 B** hard cap. The 256
local request receipts are **not** an AgentState field. No truncation,
compaction, capacity laundering, persisted SelfModel, WAL/restart/rollback,
prompt or public/API view is implemented. Failure before a receipt leaves
the empty root and prior receipts unchanged. Read outputs hold only bounded
IDs, revision, digest and explicit UNKNOWN: no R12 content, user profile,
generated biography, private thought, model rationale or transcript.

## Verification / Human gate

- `nix develop --command uv run pytest tests/test_r15_u4_self_model_system.py -q`
- `nix develop --command uv run pytest tests/test_r15_u4_self_model_system.py tests/test_r15_u3_narrative_self_system.py tests/test_r15_u2_relationship_system.py tests/test_r15_u1_contracts.py tests/test_r15_u1_capacity.py tests/test_context_model.py tests/test_experience_participant.py tests/test_r13_contract_bounds.py tests/test_r14_u1_scope.py -q`
- `nix develop --command uv run ruff check suzka tests`
- `nix develop --command uv run mypy suzka`
- `nix develop --command env -u SUZKA_CONFIG_PATH just check-all`
- `git diff --check origin/rebuild/develop...HEAD` after commit.

These are validation obligations, not PASS claims until actually run.
Independent Human exact-head U4 review is required; U4 boundary PASS would
not lift #288's R15 functional DESIGN HOLD, authorize U5–U7, Ready or Merge.
