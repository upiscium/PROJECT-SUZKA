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
