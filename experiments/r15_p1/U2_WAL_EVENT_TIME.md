# R15 U2 — retained event-time R09 participant in existing StateWAL (TEST ONLY)

Parent: Issue #339, R15 #283, MVI #323, prior event-time volatile preflight Draft PR #340. Base: merged #342 on rebuild/develop at 9084333e9c0c6290b68f68e712a7aad523170153.

## Question

Can an actual terminal CHAT/API_CHAT event be joined from independently committed R07 Journal, R12 Experience, and DB1 to an **existing StateWAL transition** that already retains the event-time R09 ContextStateSnapshot and its opaque participant_refs? If so, is a new R09 or R12 production event-time schema field necessary for the *smallest* positive U2 scenario?

## Test-only answer and limits

For a small retained same-generation event, the existing TransitionRecord.candidate_snapshot.context_state contains the Context after that event's serialized Chat. In the isolated acceptance fixture, a single pre-existing opaque token at event E is retained in that candidate; an anonymous E remains anonymous even if the same Context receives a token in a later Chat. The earlier candidate's captured Context revision also remains unchanged. Source qualification requires exact terminal R07, R12 committed Experience, DB1, and a valid active StateWAL generation/transition matching the same R07 event id and processing sequence.

This supports the **feasibility** of reconstructing one event-time opaque participant provenance without changing the upstream production R09/R12 data shape. It does NOT implement a production U2 issuer or the owner, prove a real-world person, identify who authored a message, or establish trust, familiarity, closeness, reciprocity or causal fault. The request's interlocutor_key remains caller-supplied unverified input.

StateWAL is a *technical, non-authoritative* replay record. It is not a new identity oracle: a future production U2 review must recheck current real Source owners and require an actual serialized, subject-admitted Relationship review. After WAL baseline generation rotation or retention expiry, the original E transition may no longer be present. An unavailable prior event must yield HOLD/UNKNOWN rather than retroactive inference from the current Context. No cross-owner historical SourceEpoch assurance is claimed; that remains Post-R26 #320.

## Scope

- NEW test-only tests/test_r15_u2_wal_event_time_preflight.py.
- This document only. NO changes to suzka/, AgentState v10, WAL writer, R07/R09/R12/DB1, installed services, public API, RelationshipSystem or Source authority.
- Data are disposable FastAPI/TestClient synthetic providers over actual isolated R07/R09/R12/DB1/WAL code.

## Verification

Run:

    python -m pytest -q tests/test_r15_u2_wal_event_time_preflight.py
    python -m ruff check tests/test_r15_u2_wal_event_time_preflight.py
    python -m ruff format --check tests/test_r15_u2_wal_event_time_preflight.py

## Next decision

Under Issue #339 and a separate Human production code approval, choose whether the **existing retained WAL transition** can serve as event-time technical participant corroboration for a source-qualified U2 provisional UNRESOLVED_ISSUE axis, avoiding changes to R09/R12 writers. Require evidence of an actual independently qualified interaction-level issue; a participant token by itself is not proof of an unresolved issue. A future U2 owner event must remain UNKNOWN if the historical WAL transition is unavailable, ambiguous or source-inconsistent. Do not broaden SourceEpoch or person identity recognition merely to pass a test.

No Merge, production implementation, deployment or Source migration is authorized by this test-only PR.

## 2026-10-21 — additional negative controls on merged U4b baseline

The test-only branch was advanced by a normal merge of merged U4b baseline 98777c88a0cb3b49f9b94ac91e84d88486a9d507. No production changes, PR merge into rebuild/develop or Source migration were made.

Three added isolated negatives make the event-time boundary more precise.

- Retention: a synthetic omission of a historical Chat transition from a StateWAL inspection refuses event-time attribution instead of using the present Context membership. This is a fault-injection proxy, not a physical StateWAL rebaseline test.
- Same display text: identical fixed Chat text in two different sessions retains distinct opaque interlocutor references. This is not a person-identity verification.
- U3 and U2 gate separation: the current U3 NarrativeSelf qualified source reader deliberately rejects any Context with participant refs. This is correct for anonymous U3 history, but means the U2 positive feature cannot reuse the existing anonymous-only U3 gate as its participant evidence issuer. An actual event-time WAL candidate with a token does not change that fact.

### Exact future U2b design boundary (Human code gate still required)

1. Use a new *U2-scoped* private read in a real serialized RELATIONSHIP_REVIEW event. Join one actual terminal R07 CHAT/API_CHAT, active R12 committed Experience, DB1 episode and the retained StateWAL transition of the same event ID/processing sequence. Verify matching Context ID, session/channel, one opaque participant and WAL candidate history. Do not revise U3's strict anonymous-only read.
2. Independently qualify a very narrow semantic interaction hypothesis before any positive Relationship axis is allowed. Presence of an interlocutor token, a user-reported correction, or a display-name match *alone* is insufficient to assert a genuinely UNRESOLVED_ISSUE. If the reported issue was already resolved or does not actually have an observable unresolved meaning, keep UNKNOWN.
3. Admit only a tentative issue about the interaction with that opaque token, not verified personhood, trust, familiarity, closeness, caution, reciprocity or culpability. RelationshipSystem remains the sole owner. Add a genuine internal Actor revision proof, persisted using the existing AgentState v10/StateWAL owner.
4. A missing historical transition after retention/rebaseline, forged or retired Experience, participant added only after E, ambiguous current Context, contradictory source or unproven semantics must fail closed. Do not use the current Context to retroactively assign a past E.
5. Prefer existing R07/R09/R12/DB1 and StateWAL read interfaces without changing their persisted shape. Do not add a general SourceEpoch framework for this MVI. If a specific failed test demonstrates the need for a new historical owner or compatibility field, return to the Human design gate first.
6. A future U6 consumer of Relationship must requalify current Source when using it; a restored current state projection by itself is not new evidence that an identified person exists.

This is a design and test-only outcome, not production U2 implementation or Merge authorization. Issue #339 remains the separately required Human code gate. Current test matrix: 7 new cases, with the three cases above added to the initial four.
