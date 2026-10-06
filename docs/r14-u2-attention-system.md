# R14 U2 — process-local AttentionSystem

This records U2 implementation evidence for #281 / Draft PR #282, not a Human
focused-review verdict. U1 is accepted at
`cdced8c2148d4fd07b47ae92c6a538ac762618a6`. U2 introduces only the process-local
owner of those contracts. U3–U6 remain unauthorized: no Metacognition producer,
AgentState v9/WAL integration, MainLoop/PromptBuilder change, HTTP route, model,
source mutation, timer, scheduler or R15–R19 authority is included.

## Ownership and API

`AttentionSystem` privately owns one immutable continuity/read-view bundle. A
private `RLock` serializes refresh, local restore and reads. Public snapshots
and views are validated detached copies, not mutable aliases to owned state.

```python
AttentionSystem(snapshot: AttentionContinuity | None = None)
snapshot() -> AttentionContinuity
selected_view() -> AttentionSelectedView
restore_snapshot(snapshot: AttentionContinuity) -> None
refresh(projections, event, *, global_emotion=None) -> AttentionRefreshResult
```

The constructor bootstraps empty or validates an exact U1 snapshot. Local
restore accepts every legal U1 shape, including a short unanchored receipt
chain; it does not repair or invent omitted historical receipts. This is a pure
in-memory replacement, not a production state port or persistence path.

`AttentionSelectedView` carries current revision/event/state/authority digests,
typed focus/unfinished references and optional competition/prompt witnesses.
Those ephemeral witnesses are absent after bootstrap or local restore. Focus
continuity is preserved, but source rows are not presumed resolved and no
competition or prompt selection is replayed to recover them.

## Complete-source and freshness boundary

Refresh consumes an exact sealed projection tuple and optional exact sealed
global-Emotion projection. It validates/copies inputs, canonicalizes stable-ID
order, rejects duplicate/over-bound identities and binds every projection to
the explicit event.

The tuple must be the trusted producer's **complete, coherent current capture**.
U1 adapters check completeness relative to supplied source snapshots; U2 does
not independently enumerate source stores or prove atomic capture. Later U5
must provide that ordered capture. Under this precondition, absent prior IDs
are removed from current Attention, and ineligible current IDs cannot remain
focused/unfinished. Explicit unavailable/inactive rows remain represented.
There is no hidden historical candidate cache narrowing the current universe.

For fresh events, the same typed candidate's primary source revision cannot
decrease. At equal source revision, its primary digest and upstream event
triple must agree. Projection digests are **not** compared this way: urgency,
Context/resolution and event observations can legitimately change without a
source-record mutation. Source revisions and Attention revisions remain
independent namespaces. Checksums bind facts; they do not authenticate producers.

## Exact retry and bounded replay

The domain-separated **refresh-operation input digest** covers:

- operation and fixed policy version;
- exact event ID/sequence/UTC time;
- all canonical source projections, sorted by candidate ID;
- exact global-Emotion observation, or explicit absence.

It deliberately excludes prior Attention authority. The existing receipt's
`input_digest` stores this operation identity, making it recomputable after a
transition or exact local restore. The pure policy's separate input digest
still includes prior authority; it is not used as the retry identity.

Retained receipts are checked before source freshness, scoring or continuity
arithmetic. Exact retained event plus input means **no mutation**: refresh
returns the current snapshot/read view, `replayed=True`, and the original
receipt. The receipt's result-state digest may describe an older accepted
state; this is not historical snapshot/selection reconstruction.

Conflicting retained ID/sequence/time/input fails closed. An older event whose
receipt is no longer retained also fails closed: a compaction anchor is a
fence, not proof of its original input. Bounded suffixes cannot prove reuse of
an ID after all identifying evidence has compacted when it arrives under a
new sequence; no infinite event-ID ledger is claimed.

## Fresh transition and atomic publication

Every accepted fresh event, including an idle/empty event, increments the
Attention revision exactly once. Competition reads **prior** focus/counters;
the same exact projections then produce the prompt-selection witness. Only
after selection does each current candidate advance the accepted U1
habituation, inhibition-recovery and mutually exclusive streak arithmetic
once according to final focus. New IDs start from the U1 zero continuity
baseline; removed IDs do not retain a shadow row.

The next content digest excludes history/receipts, avoiding a digest cycle.
Revision and receipt evidence are appended with that result-state digest.
History and receipt windows compact independently:

- at revision overflow, the actual evicted proof forms the revision anchor;
- at receipt overflow, the actual evicted receipt forms the receipt anchor;
- the newest 16 proofs and 256 receipts retain exact chain links and fences.

Restored short unanchored receipt chains append only their declared evidence;
an anchor is created from an actual eviction, never fabricated to make them
look complete.

All input, source, event, counter, unfinished, prompt, chain and canonical-byte
checks complete before publication. The resulting U1 continuity, coherent
current view and detached return value are staged before **one bundle
reference replacement**. A rejected or injected failing stage leaves both
old state and old read view unchanged. No fallible validation/copying follows
the successful swap. Separate read calls spanning a refresh can observe
different generations; a single view or refresh result is coherent.

## Preserved bounds and later integration

| Contract | Unchanged bound |
| --- | ---: |
| Current target universe | 4192 (4096 WM + 32 each R13 domain) |
| Focus / high global-arousal focus | 16 / 8 |
| Unfinished refs | 16 |
| Retained revisions / receipts | 16 / 256 |
| Attention-selected prompt bytes | 131072 |
| Exact Attention canonical value maximum | 4193496 |
| Future v9 including full 16 MiB reserve | 130544419 |
| Margin beyond reserve under 128 MiB cap | 3673309 |

U2 adds no persisted fields and does not change U1 coefficients or capacity
derivation. Explicit R13 no-deadline remains measured urgency `0.0`; genuinely
unknown signals remain `None` with the accepted neutral fallback. Source
records, rendered text, signal vectors and model rationale are not retained
in continuity. Current read witnesses contain only bounded typed metadata.

The selected view is not an actual model prompt. Its row byte/digest witnesses
must be matched by later U5 rendering; local restore cannot recreate missing
source content by cognition. U4 still owns persisted v9 compatibility,
multi-authority rollback, WAL and crash recovery.

Focused tests cover exact/current/historical replay, coherent full-source
fixtures, freshness and source non-invasion, arithmetic and deterministic
focus, independent compaction, local restore without replay, injected failures,
capacity/overflow, detached reads, locking and absence of production/later
authority. Executed counts and any publication/CI evidence are reported by the
primary session; the presence of this report is not a review PASS.
