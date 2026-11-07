# R15 U2 — Event-time R09 participant observation (TEST ONLY)

Parent: GitHub Issue #339. Merged R15 U1–U5: PR #333. MVI policy: #323.

**This is not a verified person, relationship admission or durable Source
witness. No product module, database owner, API, StateWAL or installed service
was changed.**

## Problem

R09 ContextFrame.participant_refs is mutable through the Context registry,
and contains opaque tokens rather than verified humans. R12 ExperienceRecord
persists the context_id but no frozen participant set/revision from when E
was actually observed. The latest Context therefore cannot prove whether
a participant token was present at E, or added only later. Matching names
and InterlocutorBinding identity_key claims do not fix missing causation.

## Scope and responsibilities

**Owns:** evaluator-side EventTimeObservation of the actual immutable
ContextFrame returned by resolve_chat_context() *inside* the real serialized
Chat worker, bound in memory to that event ID and sequence. The observation
is deliberately volatile and unavailable after shutdown.

**May:** test fresh synthetic SUZKA Chat, R07 terminal event, R12 committed
Experience, and matching DB1 episode; check the originally captured frame
and current R09 revision. Exactly one unchanged opaque participant token is
test-only OPAQUE_OBSERVED, never a verified person. All other states HOLD.

**Must not:** install Relationship axes, RevisionProof, source attestations,
a new WAL generation, trust/familiarity/closeness or a verified identity;
grant Tool/Decision/Goal authority; infer real human intent from user/model
text. A supplied interlocutor_key is a caller-scoped token, not independent
proof of who spoke.

**Depends on:** existing R09 ContextRegistry, R07 API_CHAT, R12 Experience,
DB1 and actual v10 MainLoop under a disposable TestClient. The observation
hook exists only via test monkeypatch, not a production source witness.

**Used by:** identifying the smallest event-time participant source that
a separately Human-approved U2 positive owner could rely on.

## Adversarial controls

- One participant already present during E: can observe one *opaque* token.
- Anonymous E then token added: HOLD; never retroactively attribute old E.
- One participant during E then second added: current replay HOLD.
- Multiple opaque participants during E: cannot infer a single other person.
- Two distinct tokens with the same claimed identity_key: never merge
  them or assert a verified same-person relationship.
- Forged R07 event ID or missing R12 Experience: HOLD with no U2 mutation.
- Shutdown/new process: the test-only frame cannot be recovered from the
  R12 context_id alone. A genuine persisted event-time witness is missing.

## Next Human-reviewed U2 positive design

Capture one bounded Context revision and opaque participant reference from
the actual Chat event at the point Context is resolved, commit that witness
under the *same* real R07/R12/StateWAL authority as E, and require it when
later privately qualifying any first-person interaction hypothesis. Do not
allow post-E Context lookup or an arbitrary caller-made ContextFrame to
become source authority.

Only after this independent owner scope exists may a new serialized
Relationship review adopt the minimal source-scoped UNRESOLVED_ISSUE
hypothesis, leave TRUST/FAMILIARITY/CLOSENESS/RECIPROCITY/CAUTION UNKNOWN,
persist through current v10/WAL, and verify rollback/restart, stale-source,
same-name, late-participant and no-E controls.

An explicit Human source/schema/size/compatibility gate is still needed
before R09/R12/Runtime/U2 production code edits. Global G2 SourceEpoch
attestation remains deferred after R26, not a prerequisite here.

## Verification

Run pytest on tests/test_r15_u2_event_time_preflight.py and Ruff on the
new experiment and test files; use only fresh synthetic Source.
