# R15 U2 — process-local Relationship owner, not a trust producer (#286)

This is a **U2-only** implementation against [Human-accepted A1–A27](https://github.com/upiscium/PROJECT-SUZKA/issues/283#issuecomment-6058471359)
after [U1 Human PASS](https://github.com/upiscium/PROJECT-SUZKA/pull/285#issuecomment-6073619240)
at `794c517db5e09149b31392c11097d36a1e6dc76c`. No code in
`suzka/r15/{common,contracts,bounds,stance}.py` changes. This unit introduces
`suzka/r15/relationship_system.py` and focused adversarial tests. It does
**not** wire production AgentRuntime, R12/Journal, API, prompt, AgentState v10
or WAL, and no user-facing correction/merge/trust writer is available.

## Responsibility contract

- **Owns:** one process-local, immutable U1 `RelationshipState` root; exact
  per-instance event receipts; observed *opaque* R09 participant continuity;
  six independently UNKNOWN axes; private detached snapshot/selected read.
- **May:** read one complete current `ContextRegistry.state`, verify active
  Context/participant membership and event time, stage one UNKNOWN record and
  a chained `RevisionProof`, then atomically publish one process-local root.
  An existing opaque ref can receive an explicit no-change receipt.
- **Must not:** interpret/model/score trust, infer actor blame from R10,
  attribute R12 Experience to a person, accept a valid-looking witness as
  producer proof, adopt an operator/model/Web claim, merge/split aliases or
  take R09/R11–R14/Goal/Decision/Action authority. No generic `revise`,
  `set_trust`, `set_axis`, `ingest_experience`, `merge_persons`, or restore
  writer exists. Positive interpretation/person merge/actor cause status is
  typed `UNAVAILABLE`, not `REQUIRES_TRUSTED_ROOT` treated as acceptance.
- **Depends on:** accepted U1 closed roots and maximum legal schema, actual
  R09 `ContextRegistry.state` snapshot of an already selected Context, and
  externally supplied exact ordered `EventRef`. Caller authentication and
  serializing R09/U2 under an actual AgentRuntime event are *not delivered*
  here; U2's `EventRef` itself does not authenticate its sender.
- **Used by:** a later separately reviewed U5 durable port / U6 ordered
  runtime/read producer. There is no present production consumer or source
  route; test-created Context registries prove only UNKNOWN-state mechanics.

## Source/admission inventory and deliberate UNAVAILABLE results

| Input / alleged source | Verified present R14 behavior | U2 decision |
| --- | --- | --- |
| R09 ContextRegistry current frame | `state` projects one locked complete registry revision, selected ID, ordered frames/participants and declarative bindings. `resolve_chat_context` itself may mutate R09 and is **not** used. | Read only one current `ACTIVE` frame, require exact opaque `interlocutor_key` in its participants, R09 times ≤ event time; verify state again before publishing. Create/observe UNKNOWN only. |
| R09 `InterlocutorBinding.identity_key/confidence` | R09 stores caller-supplied identity hypothesis/evidence refs, even with `confidence=1.0`. Equal `identity_key`s do not merge distinct participant references. | U2 ignores these fields as person truth and writes `claimed_identity_key=None`; no same-person/alias/split producer exists. Separate keys remain separate roots. |
| R12 `ExperienceRecord` + R07 | R12 records lack actor identity and relation interpretation. `EventJournal.inspect()` is read-only but cannot prove the Experience artifact. `ExperienceStore.load_committed_current()` may perform recovery/permission mutations, and participant reconciliation needs binding data. No single strictly side-effect-free joined API proves source finality + context/actor + subject admission. | **No R12 loader or Journal import in U2.** A forged `SourceWitness` (even syntactically ACTIVE, subject-looking, correct 64-hex digest) grants no authority. Positive trust/familiarity/etc. admission is UNAVAILABLE pending separately reviewed joined source, actual external finality and subject semantic producer. Test-only R12 fixtures are not a production source. |
| R10 Emotion/ordinary Appraisal | Global valence/arousal/optimal-loss; ordinary chat passes default `AppraisalSignals()` with no actor-specific source. | UNAVAILABLE for person blame, repair, reciprocity or durable trust; D6 tentative felt association remains event-only U1 stance, never Relationship history. |
| Operator/model/Web, typed event, mock data | Authentication, serialized handler or apparent hashes do not supply Suzka subject admission. | No direct subjective adjustment API. U2's only mutation stores UNKNOWN, never confidence/estimate/trait/competence. |

The missing positive mechanism must independently prove **(a)** selected
source and target identity *at event time*, **(b)** R12 Experience really
externally finalized and lifecycle/revision/digest eligible, **(c)** Suzka's
first-person interpretation plus complete independent support/contrary
evidence, and **(d)** subject admission unforgeable by operator/model/caller.
Until all four are reviewed, the output is **UNAVAILABLE**, not inferred from
`identity_key`, a single global affect, an asserted typed source ID, repeated
requests or the R09 participant label. The same source root copied under new
IDs cannot supply independent corroboration, because no positive admission
operation exists. This is an explicit producer gate, not a test fixture
promoted to runtime authority.

## Deterministic UNKNOWN state and bounded process-local receipts

At construction, root is exactly `RelationshipState(1, 0, (), None, ())`.
`observe_current_interlocutor(event, key)` checks an exact `EventRef` and
current selected R09 participant; derives the relationship ID from the opaque
key and six **distinct** axis statement IDs by domain-separated SHA-256.
No guessed neutral score: every axis has `UNKNOWN`, estimate/confidence
`None`, empty support/counter, and no verified-person key. A new record
increments the root revision once and appends one U1 `RevisionProof` with
`INITIAL_INTERPRETATION` and zero evidence; existing key remains unchanged.

Proof `before_digest`/`after_digest` bind **only** the complete record/revision
data projection, not the `RelationshipState.state_digest` which contains the
proof and would be circular. Proofs chain through their exact prior
`record_digest`. No anchor or history compaction is fabricated. A maximum
of **64** relationships and **64** mutation proofs are retained; the 65th
distinct admission fails without discarding legal state. This is within the
unchanged accepted U1 full-legal budget: v10-with-full-16-MiB-reserve
**133,726,597 B**, margin **491,131 B** below the hard cap *after reserve*.
The **256** additional U2 receipts are **process-local only**, not an added
v10 field/reserve spend; hitting their bound fails without silent eviction.

Each accepted event gets a private immutable receipt with event ID/sequence/
UTC time, exact selected Context ID/revision/projection checksum, operation
input digest, before/after root checksums, result revision, creation flag
and previous receipt digest. **A paired `RelationshipObservationResult` is
valid only if the receipt's `result_revision`/`result_state_digest` bind the
returned snapshot revision/canonical digest AND the selected view's root
revision/digest/axis values.** Its constructor rejects mismatched event-result
pairs. The receipt returned to a caller is a validated **copy**, not the
internally retained instance.
The caller's `EventRef` is copied and validated before it enters any proof;
later mutation of the caller-owned object cannot rewrite a retained event or
its receipt chain.

An identical retained event+input never reruns R09 and never changes root,
receipts, history, counters or trust. If that receipt still describes the
**current** root (including after later no-op events), replay returns
`RelationshipObservationResult(replayed=True)` with its matching current
snapshot/view. If later events changed the root, the return type is instead
`RelationshipHistoricalAcknowledgement`: only the original receipt and its
original `event_result_digest`, with **no** `snapshot` or `view`. This is a
historical acknowledgement, **not** a reconstruction of an unretained event
root. A separate `snapshot()` reads the latest state and must not be paired
with the old event receipt. Tampering with a returned receipt cannot mutate
the internally retained replay proof. Same ID with different event/time/target
and reused sequence conflict;
unretained older/reversed event/time rejects. The owner privately retains a
checksum of the **complete** R09 registry generation (NOT in receipts or read
views, because a whole-registry fingerprint can reveal private equality). A later R09 revision
regression or same-revision divergent restore is rejected before any U2
publication; U5/U6 must coordinate classified true rollback across owners,
not let an old Context generation coexist with newer U2 receipts. Checksums
detect exact local divergence, not producer authentication, and a later
higher-revision fork still needs U6 ordered source-history proof. Reads return freshly detached
root objects or compact axis views without private source or model text.

Transition staging holds an RLock with a reentrancy-invalidating guard;
recursive mutation attempts poison the outer transition even if caught.
Source absence, R09 revision rollback/fork or change between initial/recheck, bad event,
history/receipt/record overflow and construction failures do not change root
or receipts. Every output copy, receipt and view is preflighted before the
process-local root+receipt bundle is published. The second R09 read detects
*observed* concurrent changes, but U2 cannot guarantee an atomic cross-owner
read/commit window: only a later ordered AgentRuntime capture (U6) can
serialize both authorities across that boundary. The current `ContextRegistry`
is caller-provided to this local owner and cannot authenticate the event;
**no production callback uses this owner yet**.

## Migration, restart and privacy impact

No AgentState v10/WAL/Journal/Memory/Context mutation, startup hook, durable
receipt or restoration API exists. Per-instance exactly-once is not a
cross-restart guarantee. U5 must separately prove restore/rollback and retained
receipt evidence without cognition replay; U6 must authenticate actual
serialized events, qualified source/current R09 views, and never publish an
uncommitted view. Diagnostic read routes and prompt formatting are not added.
U5/U6 consumers must distinguish the two U2 replay result types; no future
WAL/read adapter may pair a historic receipt with the separately readable
latest root as if it were the event-result root.
Detached views contain IDs and bounded status/numeric *unknowns*, not
transcripts, private R12 content, model rationale, operator profile or raw
prompt. Existing R09/R12 and R11/R13/R14 privacy/authority are unchanged.

## Verification and focused gate

- `nix develop --command uv run pytest tests/test_r15_u2_relationship_system.py -q`
- `nix develop --command uv run pytest tests/test_r15_u1_contracts.py tests/test_r15_u1_capacity.py tests/test_r15_u1_scope.py tests/test_context_model.py tests/test_experience_participant.py tests/test_r14_u1_scope.py -q`
- `nix develop --command uv run ruff check suzka tests`
- `nix develop --command uv run mypy suzka`
- `nix develop --command env -u SUZKA_CONFIG_PATH just check-all`
- `git diff --check origin/rebuild/develop...HEAD` after commit.

Executable U2 tests cover full unknown axes, distinct keys with equal asserted
identity and 1.0 confidence, complete R09 selection/lifecycle/time check,
typed positive-source unavailability, exact and old replay/conflict, historic
acknowledgement without current view, same-event retry after R09 changes,
receipt mutation isolation and receipt/root/view binding, reentrant
and intervening R09 mutation rollback, full 64-proof and 256-receipt overflow,
detached reads/concurrent same-event calls, and U2-only import boundaries.
Local/CI green is necessary but not a Human PASS; stop for exact-head U2 review.
