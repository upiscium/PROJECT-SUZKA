# R15 U3 — bounded process-local NarrativeSelf / positive-admission DESIGN HOLD (#287)

U3 follows [Human-accepted U1](https://github.com/upiscium/PROJECT-SUZKA/pull/285#issuecomment-6073619240)
and [U2](https://github.com/upiscium/PROJECT-SUZKA/pull/285#issuecomment-6078285032).
The binding [A1–A27 contract](https://github.com/upiscium/PROJECT-SUZKA/issues/283#issuecomment-6058471359)
does **not** authorize invented autobiography. `suzka/r15/narrative_self_system.py`
does not import runtime, R12 Memory, Journal, model providers or another R15
owner and does not modify accepted U1/U2 code or schema. U3 has no production
consumer. The R15 v10 schema is still only a *projection*, not U5 persistence.

## Responsibility

- **Owns:** exact immutable process-local `NarrativeSelfState` root (initially
  empty), detached private `snapshot()`/`selected_view()`, typed availability
  audit, finite **no-change request** receipts and exact event/replay fences.
- **May:** report whether a proposed *shape* needs a trusted R12 source root
  (`REQUIRES_TRUSTED_ROOT`) while returning `UNAVAILABLE` for actual semantic
  admission; acknowledge an explicitly source-free `review_boundary(event)`
  under process-local ordering with **no narrative mutation or proof**.
- **Must not:** create or admit episodes/claims from a typed tuple, model/Web/
  operator narrative, event ID or mock; modify R12 Experience, R09 identity,
  R08 WorkingMemory, R11 Value/Belief, R13 Goal/Commitment, R14 focus or U2
  trust. No arbitrary `create_episode`, `adopt_claim`, `revise`, `reactivate`,
  `restore_snapshot` or direct subject-edit writer is exposed; no U4/U5/U6/U7,
  AgentState v10/WAL, MainLoop/API/prompt, Decision/Action or scheduler.
- **Depends on:** accepted U1 `NarrativeSelfState`/closed `ClaimMeaning`, exact
  caller `EventRef` shape. Actual event authorization, R12 finality, event-time
  Context revision and Suzka semantic/subject admission are *missing* and
  remain independent future U5/U6/U3 source-producer gates.
- **Used by:** only a later reviewed joined R12/R07 event-capture producer,
  U5 durability and U6 ordered runtime. No production consumer can mutate it
  today; test-created R12-shaped witnesses are not production authority.

## Read-only source audit and precise admission stop

| Proposed source | Actual current owner/API | Admissible U3 effect |
| --- | --- | --- |
| R12 Experience | `ExperienceStore.load_committed_current()` calls `_load_current()`, which may recover publication temps/compaction and harden parent permissions. `MemoryExperienceParticipant.inspect_reconciliation(binding)` calls that loader and relies on caller binding. The R12 record has source event/episode/context but no R09 revision or actor identity. | No strictly side-effect-free joined R12/R07/event-time R09 proof. An ACTIVE-looking `SourceWitness` receives *at most* U1 `REQUIRES_TRUSTED_ROOT`, **not** `SOURCE_AVAILABLE` or an indexed autobiography. Episode indexing `UNAVAILABLE`. |
| R07 EventJournal | `inspect()` verifies transaction completion and participant operation digests read-only, but does not contain the R12 artifact or Context payload; creating an inspection owner can acquire a lease and mutate lock permissions. | Journal finality alone does not prove a specific Experience, source episode, Context or subject meaning. No U3 Journal reader. |
| DB1 episodic memory | `DualMemorySystem.get_committed_episodic()` can read one document without repair; it lacks the R12/Journal/Context joined proof. | No story/actor/history construction from a Memory scan. |
| R09 Context | `ContextRegistry.state` is a current snapshot, not an authenticated event-time Context revision/digest history. | No real-person or actor attribution and no cross-context auto-merge. |
| U1 narrative semantics | `current_semantic_producer(ClaimMeaning)` returns `UNAVAILABLE` for **every** current code. R12 Experience stores event measurement/affect, not the actual first-person role/trait/narrative predicate; a Belief text/digest is not that mapping. | `SEMANTIC_ADMISSION_UNAVAILABLE` for claim/interpretation/accessibility reactivation. Closed code, matching statement ID or apparent first-person endorsement never authenticates Suzka. |

**Positive-admission DESIGN HOLD:** A production episode/claim mutation needs
a separately reviewed *strictly read-only* joined source owner that verifies
R07 terminal participant/operation proof against the exact externally
finalized R12 Experience artifact/revision/lifecycle/operation digest and
source episode, at the actual event-time R09 Context/revision. A distinct
Suzka-owned first-person semantic mapping/admission must then qualify bounded
support, independent contrary evidence, candidate links, inertia and
accessibility. No combination of the current APIs provides that joined
authority. U3 does **not** call recovery-capable loaders, wrap a test-only
trusted fixture as a production adapter, or widen U1 `current_semantic_producer`.
Even authenticated transport or AgentRuntime serialization of an operator
claim cannot grant personality/trust/capability truth (D9 MUST NOT). A future
positive producer is a separate Human-reviewed contract, not a TODO silently
implemented by U3. No U3 source path currently yields `SOURCE_AVAILABLE`.

`narrative_availability()` reports separate `source`, `semantic` and final
`admission` fields. A declared ACTIVE Experience can be syntactically
`REQUIRES_TRUSTED_ROOT`; stale/retracted/operator/model/Web inputs are
`UNAVAILABLE`. The semantic and final admission fields are **always
`UNAVAILABLE`**. This is a read-only classification and cannot write the
root. `NarrativeSelfSystem` has **no source/claim writer or state injection**.

## Event receipt and replay without invented cognition

`review_boundary(event)` acknowledges *only* a caller-supplied, no-change
availability-review request, **not** an R12 source or subjective narrative
event. It copies the caller `EventRef`, checks exact identity/monotonic UTC
event sequence and time, stores up to **256** process-local request receipts
without eviction, and rejects conflicting/older/reentrant/over-bound calls
before adding a receipt. A caught nested reentrant call invalidates the outer
fresh **or replay** request; it cannot succeed through a replay fast path.
The immutable U1 root stays exactly
`NarrativeSelfState(1, 0, (), (), None, ())`: no synthetic `RevisionProof`,
history anchor, episode, claim, belief, salience or accessibility change.
Receipt counts and reviews are not a proxy for experience or confidence.
Returned receipts/snapshots/views are detached; a caller mutating nominally
frozen inputs cannot change retained evidence. The receipt status is private
and not persisted across restart; it is not an operator-facing API.

`NarrativeBoundaryResult` validates that a receipt's result revision/digest
matches the **same** returned root and `NarrativeReadView` identifiers. Exact
retry while that root remains current returns that bound result without
adding receipts or rereading a source. Pure private `_result_for()` instead
returns receipt-only `NarrativeHistoricalAcknowledgement` if the historical
event result and a later root differ, never pairing an old receipt with a
current snapshot/view. Production U3 cannot reach that branch: the only
permitted event operation does **not** change its root. The U3 test explicitly
constructs a **test-only synthetic** U1 root to check this future replay
classification; that tuple is not a finalized R12 Experience, cannot be
installed into `NarrativeSelfSystem`, and is no evidence of production
admission. U5/U6 must distinguish historical acknowledgements from current
root reads if and when a real source-qualified transition is authorized.

## Conflict, accessibility, capacity, privacy and recovery

The accepted U1 schema can structurally distinguish `PRESENT`, `CONTESTED`,
`CONTRADICTORY`, `UNKNOWN`, paired support/counterevidence, opposite
`ClaimMeaning` polarities and `CURRENT/LATENT/REACTIVATABLE` accessibility.
U3 tests build these shapes **only outside** the production owner to prove
closed serialization/semantic distinction; a copied/retracted Experience
cannot be admitted as a present fact. No cognitive revision/accessibility
reactivation occurs before the missing source/subject producer exists.
Functional accessibility is not R12 deletion, R08 membership, R14 focus or
U5 technical recovery. Current root/history remain empty and no compaction,
silent clipping, WAL/restore, or cognition replay is performed.

Accepted U1 full-legal v10 maximum remains **133,726,597 B**, including the
full untouched **16,777,216 B future reserve**, with **491,131 B** margin
beyond reserve. The 256 U3 request receipts are local only and do not spend
future v10 state capacity; adding durable receipts later requires fresh U5
size/lineage proof. Detached views reveal only bounded IDs/revisions/digests
and never R12 content, operator biography, hidden thought or raw prompt.
Source/read failures, malformed/future input, capacity/reentrancy conflicts
are fail-closed; no U3 mutation can be partially published. Real restart,
rollback and cross-owner atomic publication are **not** claimed.

## Focused evidence and release gate

- `nix develop --command uv run pytest tests/test_r15_u3_narrative_self_system.py -q`
- `nix develop --command uv run pytest tests/test_r15_u3_narrative_self_system.py tests/test_r15_u2_relationship_system.py tests/test_r15_u1_contracts.py tests/test_r15_u1_capacity.py tests/test_context_model.py tests/test_experience_participant.py tests/test_r13_contract_bounds.py tests/test_r14_u1_scope.py -q`
- `nix develop --command uv run ruff check suzka tests`
- `nix develop --command uv run mypy suzka`
- `nix develop --command env -u SUZKA_CONFIG_PATH just check-all`
- `git diff --check origin/rebuild/develop...HEAD` after commit.

These commands are verification requirements, not claimed executions. U3
Human exact-head focused review is mandatory before U4; CI success, typed
fixtures and this producer audit are **not** a Human PASS or authorization
for v10, runtime integration, Ready, or Merge.
