# R15 U2 — retained event-time R09 participant in existing StateWAL (TEST ONLY)

Parent: Issue #339, R15 #283, MVI #323, prior event-time volatile preflight Draft PR #340. **Current base:** merged #345 on rebuild/develop at 98777c88a0cb3b49f9b94ac91e84d88486a9d507. The earlier #342 baseline was advanced by a normal merge into this test-only branch.

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

This is a design and test-only outcome, not production U2 implementation or Merge authorization. Issue #339 remains the separately required Human code gate. The current test-only matrix now includes 16 isolated cases. The initial four cases were expanded with WAL retention, U3 gate isolation, sticky Context membership, newly introduced same-event token deltas and restart, a closed same-scope Q1 finite position mismatch, cross-scope/equal-position/raw-trust negatives, and failure rollback.

## Same-event *new binding* is narrower evidence than Context membership

A further three isolated negative/positive tests compare the **immediately preceding retained WAL candidate/baseline Context** with the current Chat's candidate Context for the **same active transition**. They establish:

- One first-time, caller-declared opaque participant newly attached during a serialized Chat can be recovered as a **new event-time Context binding**, even after restart.
- A later Chat with the key omitted retains Context membership but has **no newly bound participant**. Repeatedly supplying an already bound key also has no delta. Neither event is attributed to that token by this narrow rule.
- A participant added only in a later Chat belongs to that later binding event, never retroactively to earlier anonymous E.

This still does **not** verify the author/speaker or a real-world person. It proves only that the current Chat caused an opaque Context association to be introduced under the existing R09 resolver. The test-only helper does not grant U2 or U3 authority.

### Proposed two-tier U2b MVI decision

**Earlier Option A — superseded as a durable U2 positive:** first-binding evidence plus a visibly unresolved expression was previously considered for a provisional relationship issue. Subsequent tests and reviews found that the expression can be controlled by user/model text, may already be resolved, and the caller-chosen token can collide across Contexts. **Do not admit a durable U2 positive from this evidence alone.** First-binding remains useful as event-time technical provenance.

**Alternative for multi-turn per-token updates:** require a distinct, persisted and independently verified per-event caller-declared interlocutor selector or narrowly bound technical observation within the normal R07/R12/WAL transaction, because the present Context membership and ordinary DB1 text do not retain proof of who was declared for each subsequent message. This would be a separate Human-approved compatibility and ownership change, not an assumption that current WAL membership already suffices.

Neither option alone proves UNRESOLVED_ISSUE: a genuine narrow positive semantic example, source qualification, single serialized RelationshipSystem owner revision and v10 replay/rollback controls remain indispensable. Verified person identity, trust, familiarity, reciprocity and skill must stay UNKNOWN. Issue #339 is the Human implementation/design gate; Post-R26 #320 remains separate.

## Detached finite interaction semantics preflight (no real U2 admission)

Three additional disposable test cases use real terminal R07, active R12, DB1 and event-time WAL provenance plus deterministic synthetic model responses. A deliberately CLOSED finite statement pair is observed:

- Input: the caller-visible user turn declares a provisional A option.
- Response: the actual committed Suzka-visible response declares a provisional B option with accuracy expressly unverified.
- Positive only when A and B differ, the current event newly bound exactly one caller-supplied opaque token and both finite responses match the strict exercised protocol. Return a **DETACHED test-only tuple** of token and two options, never a Relationship root, a user/person identity or competence.

Negative controls refuse an equal A/A pair, a later A/B turn where the Context still lists a token but the turn has no new binding, and a raw subjective trust assertion outside the closed option protocol. The actual production Relationship root remains revision 0 in every case.

This supports a possible narrow *observed interaction discrepancy* hypothesis. It **does not** prove that anyone's hidden intention or the objective answer is A/B, that the same real person authored the turn, or that a durable UNRESOLVED_ISSUE remains after a subsequent clarification. It also does not create or revise a U2 subjective state.

The Human implementation gate must separately decide whether one *newly bound, same-turn, unverified opaque token* plus an independently source-qualified, visibly **unresolved** discrepancy can authorize a first provisional U2 Relationship revision, or whether R15 requires broader event-time per-turn selector attribution. There is no automatic code authority to extend persisted R09/R12 data shapes.

## Final test-only semantic and rollback clarifications

The candidate helper is named *position mismatch*, not *unresolved relationship*: it can only return a detached triple of first-bound opaque token and the two visible A/B choices. Its exact closed protocol includes the **same fixed comparison task Q1 in both committed messages**, with an explicit unverified outcome in Suzka's response. A message about Q1 with a response about Q2 is rejected, even if the letters differ. A disagreement is **not** by itself an authenticated durable UNRESOLVED_ISSUE, objective correctness or real-world other-person claim.

A new negative test forces a keyed first Chat to fail during model generation, after the Context could have been modified in memory but before the transaction commits. The existing R07/StateWAL handler failure callback restores the last committed Context; a subsequent unkeyed Chat remains anonymous and has no new participant-binding delta. The test is limited to this failure phase, not every storage crash/failover path.

These tests demonstrate the smallest closed semantic *candidate* suitable for considering a future U2b positive owner, not a production emitter. A future source-qualified Relationship owner still must distinguish current, truly unresolved subjective interaction evidence from an already resolved or unrelated statement, and must never grant a verified speaker, trust or competence based on the mere presence of an opaque ref.

## 2026-10-21 — Cross-Context same-token collision and real R11 Value inventory (TEST ONLY)

Further negative controls against the real isolated R07/R09/R12/DB1/WAL owners show an **additional U2 lasting-admission prerequisite**:

- An `interlocutor_key` is **caller-chosen, not a globally unique authenticated principal**. Two genuine serialized Chat events in **different conversation sessions/Contexts** can **each** first-bind **the identical opaque token**. Their event-time WAL observations are separate and correct. That does **not** prove both conversations involved the same person.
- A strictly conservative **test-only use-time guard** counts *all retained* R09 Context frames containing a token, including **closed historical Contexts**. With only one active matching Context, the guard regards the raw token as locally **unambiguous across retained frames** (still not verified personhood). When a second Context contains the same token, the guard returns **false for both histories**; closing the first Context does **not** restore uniqueness. Two distinct tokens in separate contexts remain locally unambiguous.
- This is **only a necessary hold condition**, **not sufficient evidence** for a positive Relationship or even proof of one speaker. It cannot prevent a *future* new Context from reusing a formerly unique token. Without a **consumption-time** Context-scoped source check, a persistent Relationship record keyed solely by raw token is unsafe for cross-session/person claims. A global name/token is not an identity. Post-R27 [#346](https://github.com/upiscium/PROJECT-SUZKA/issues/346) owns robust per-event actor and cross-context identification.
- The existing `RelationshipRecord` and its `RelationshipState` are keyed by `interlocutor_key` globally, not a pair of (Context ID, token). Current `RelationshipSystem.observe_current_interlocutor` is a technical UNKNOWN-record observer, **not a verified person/positive-axis issuer**. No production change was made; first-binding positive admission remains unapproved.
- **R11 Subject Value inspection:** the isolated live process has two actually active system-authorized `ValueState` records, `care` and `honesty`. These are legitimate ValueSystem owner records; the test verifies they remain unchanged after Chat. But their names and values do **not** independently establish that a *particular interlocutor* created an unresolved relationship issue, nor do they authenticate a speaker. `relationship_authority_status(POSITIVE_AXIS_INTERPRETATION)` remains `UNAVAILABLE`. Even a structured ValueState is **not** permission to invent a U2 subjective consequence.

### Consequences for the R15 MVI Human decision

The current evidence supports two **distinct** achievements, neither of which is the same as a lasting interpersonal problem:

1. **Technical Relationship awareness / UNKNOWN:** an **explicit call** to the existing `RelationshipSystem.observe_current_interlocutor` may create a raw-token-keyed record with six UNKNOWN axes; **ordinary Chat does not call it today**. The observer currently uses `interlocutor_key` alone as the global relationship key. **Wiring it to Runtime unchanged could conflate unrelated Context histories even when every axis is UNKNOWN.** Such structural state change is not a positive subjective relationship interpretation, personhood proof or satisfaction of the `UNRESOLVED_ISSUE` acceptance claim.
2. **Ephemeral interaction cognition:** one source-qualified event may support a temporary clarification/caution candidate, without using model text or Value-name strings as Subject authority, and without persisting a relationship issue. R15 U1 already specifies the `InteractionStance` *value type*; an actual event-scoped producer / style consumer would need separate implementation review.

The remaining proposed **positive** lasting interpersonal meaning still requires a genuine actor/context-bound, source-qualified **SUZKA-owned significance producer** and a separately defended current unresolved interaction, not a nominal Value seed, task protocol object, A/B text mismatch, or caller-chosen key. If no such positive can be demonstrated in the constrained R15 MVI, do not force it through a trust/familiarity reinterpretation merely to pass U7: request an **explicit Human decision** to either (a) approve another genuinely qualified positive U2 case with extra technical attribution, or (b) adjust the R15 MVI whole-stage acceptance to tested UNKNOWN record + ephemeral stance and defer durable positive relationship claims. In either case, no source/schema/R16/identity/authority change has been silently approved.

Tests newly added to `tests/test_r15_u2_wal_event_time_preflight.py`: 4. They use actual isolated serialized Chat events and the real R11 ValueSystem, but the uniqueness predicate and status assertions exist **only in tests**, not as a new production Source or identity service.
