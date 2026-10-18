# R15 P1b-B1 — State v10 and source-bound Narrative continuity

Scope authorization: Issue #329, Human "Accept R15 P1b-B1 implementation".
Roadmap: Issue #323 MVI. Base PR stack: #285 → #290 → #322 → #327 → #328 → #330.
Advanced G2 historical SourceEpoch integrity remains Post-R26 (#320).

## Contract

**Owns.** AgentStateStore owns the versioned, strict v10 persisted envelope.
StateRecovery/StateWAL remain the ONLY authority for internal commits, Journal
proof and crash recovery. The actual Relationship, NarrativeSelf and SelfModel
domain owners retain subjective mutation authority. A private R15StatePort
exports an exact technical snapshot from the single MainLoop owner group;
it is not a narrative-writing API.

**May.** Preserve the existing full-legal U1 RelationshipState,
NarrativeSelfState and SelfModelState roots as closed canonical data alongside
the bounded private narrative_admission_checkpoint. Source-scoped A/B/UNKNOWN
meaning may be technically resumed from actual stored episode IDs. A later
positive claim is still provisional: the original E and later E2 were model
or user statements, not verified external human intent.

**Must not.** Let user/model/handler/Web text set trust, identity, verified
competence or an arbitrary autobiographical record. Never use an independent
JSON sidecar, hidden memory rewrite, a new Source-owner permission, or
implicit G2 SourceSetVerified. Do not migrate data on an installed service
without a separate operator gate; this branch was tested with fresh
disposable local Store/Journal/ChromaDB/AgentState paths only.

**Depends on.** Previously approved v9 Attention/Value/Belief/R13 owners,
committed R07 Chat/StateWAL event generation, active R12 Experience/DB1,
anonymous matching R09 Context, and serialized AgentRuntime NARRATIVE_REVIEW.
These readers have their normal crash-recovery semantics; they are not
certified read-only forensic historical attestation.

**Used by.** A later U6 private consumer may read only re-qualified
NarrativeSelf interpretations. This B1 branch does NOT make ordinary Chat
automatically initiate a review or inject subjective narrative into a
Decision. U2/U4 positive subjective experience updates also remain pending.

## Compatibility / persistence

- The strict V9 type stays EXACT schema_version=9; legacy V1-V9 can still
  be read. An attached R15 port upgrades the next actual authorized
  transaction to strict V10. V9 history has empty/UNKNOWN R15 roots; no
  speculative reconstructed beliefs.
- V10 consists of all original V9 fields, three U1 roots and the CLOSED,
  canonical narrative_admission_checkpoint from P1b-B preflight #330.
  It contains identifiers/sequence, anonymous Context and U3 revision
  binding only; no raw prompt, user text, model rationale or hidden thought.
- P1b-A source-gate matches the terminal R07 CHAT to committed R12 and
  DB1 and the original anonymous R09 Context before any new positive
  subjective update. The checkpoint is technical replay lineage, not a
  Source authorization itself.
- Capture reads detached Relationship/NarrativeSelf/SelfModel roots and
  Narrative source continuation from one private MainLoop-owned port.
  No second shadow NarrativeSelfSystem is declared.
- StateRecovery persists the candidate snapshot with the existing normal
  WAL append/manifest transition and v3 R07 completion. Startup constructs
  the R15 state port BEFORE restore; v10 validation checks source-scope
  meaning, event ordering, root revision/digest and canonical serialization.
- A new process recovers the same U3 owned S1/S2 state from WAL and
  revalidates Source on any subsequently requested subjective adoption;
  replay of the same committed episodes cannot generate a second revision.

## Failure and authority boundaries

- **Before WAL mutation begins:** R15 owner may have provisionally advanced
  in process, but failed capture, source check, or read-view preparation
  restores the last committed AgentState INCLUDING all three R15 owners.
- **WAL append/write result unknown:** runtime fails closed, does not guess
  that an old revision is authoritative or spuriously retry. Next startup
  performs existing StateRecovery reconciliation; it must not silently
  double-admit an S1/S2 after WAL publication.
- **After WAL commit before R07 terminal:** the same WAL generation
  determines the recovered revision. Never substitute an operator-approved
  prose claim for a missing source. An incomplete event may require
  reconciliation rather than unconditional re-admission.
- Corrupt or inconsistent checkpoint and v10 root reject publication or
  startup. An injected mismatched Experience-owner path does NOT receive
  positive R15 authority; a nonempty restored root under that topology
  blocks startup.

## Full legal capacity / privacy

Using the full legal U1 maxima, not a tiny fixture:
- Existing R15 v10 projection with full reserve: 133,726,597 bytes
- Private technical continuation maximum: 3,263-byte JSON value
- Top-level field punctuation/quoted key: 34 bytes
- Full v10 with the untouched 16,777,216-byte future-state reserve:
  133,729,894 bytes under the hard 134,217,728-byte cap
- Headroom beyond full reserve: 487,834 bytes

The runtime v10 capacity projector must equal the independent prior
test-only preflight; neither may silently shrink the reserve.

## Verification and remaining gates

The intended tests cover:
- E→S1 before E2→S2 (true R07/R12/DB1 owners, both source-scoped roots)
- No-E2 control, post-restart new MainLoop and identical replay
- Pre-WAL failure rollback, WAL-published-but-no-terminal recovery
- V9 compatibility/UNKNOWN startup, duplicate/forged continuation refusal
- Existing R13/R14/Attention/Chat regressions and no new public setter
- Private bounded metadata and exact-head CI

This is a B1 implementation, not full R15 acceptance. Independent
Correctness/Security review and Human Ready/Merge remain separate.
R15 Relationship/SelfModel actual experience-driven positive paths and
U6 private Decision behavior still need their own MVI gates.
