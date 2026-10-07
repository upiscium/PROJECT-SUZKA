# R14 U4 — AgentState v9 / Attention continuity recovery

U1/U2/U3 are accepted; U3 exact-head Human PASS is
[comment 6031506761](https://github.com/upiscium/PROJECT-SUZKA/pull/282#issuecomment-6031506761)
at `41023f05abc03784e68b93b5b70be81c77977506`. U4 implements persistence and
cognition-free recovery only. Human U4 focused review is pending. U5–U6,
production refresh/prompt integration, Ready, and Merge remain unauthorized.

## Responsibility contract

- **Owns:** additive AgentState v9 Attention continuity encoding, bounded
  preflight, compatible hashing/load/publication, a narrow owner restore seam,
  and mixed-version StateWAL persistence/recovery.
- **May:** capture an explicitly attached Attention owner; validate and persist
  reference-first continuity; stage exact restore; restore/reset continuity
  through the existing recovery and external-reconciliation gates.
- **Must not:** rerank/refresh Attention, regenerate prompt/competition witnesses,
  search Memory, call models, recompute Metacognition, persist assessments,
  mutate source truth, add source authentication, or acquire Decision/SelfModel,
  route, scheduler, provider, or outcome authority.
- **Depends on:** accepted U1 Attention schema/policy/bounds, U2 locked immutable
  state/read bundle, and existing R03–R13 AgentState/Journal/WAL recovery proofs.
- **Used by:** later authorized U5 current-authority capture and same-turn
  production wiring. No MainLoop or PromptBuilder wiring is included here.

## Additive schema and exact capacity

`AgentStateSnapshotV9` has every required v8 field plus one **required** root
`attention_state: AttentionContinuity`. Missing/None Attention is not converted
to an authoritative empty v9 field. The current schema/alias is v9; explicit
v1–v8 classes and their canonical formats remain available. V8 loading and
capacity use a frozen version-8 constant rather than the moving current version.

The v9 discriminator requires the exact integer **9**. Raw preflight rejects
fractional/exponent tokens which CPython would round to 9 before general JSON
materialization, so rounding cannot hide duplicate Attention keys or a v9 WAL
envelope. Existing legacy numeric discriminator behavior remains unchanged.

Attention uses its accepted canonical serializer/parser, not generic dataclass
serialization. Published typed values are detached and revalidated without
repairing checksums. Attention's last event cannot exceed the AgentState root
sequence or saved UTC time. Existing R13 event fences remain enforced.

| Derived component | Bytes |
| --- | ---: |
| Frozen v8 base maximum | 109,573,688 |
| Attention full-schema value maximum | 4,193,496 |
| New root-field overhead | 19 |
| **v9 base maximum** | **113,767,203** |
| Unchanged mandatory future reserve | 16,777,216 |
| **v9 including full reserve** | **130,544,419** |
| Hard cap | 134,217,728 |
| **Margin beyond reserve** | **3,673,309** |

The field maximum derives from the accepted complete **4192**-candidate schema,
not a small sample or the larger 7,866,805-byte available addition ceiling. V8
still has base **109,573,688 B** and including-reserve maximum **126,350,904 B**.
V9 reuses all v8 field maxima and adds Attention exactly once; no source bounds
or future reserve are reduced. The capacity projector supports explicit v8
accounting and defaults to the new current v9 accounting.

The durable field contains typed refs, availability, policy continuity counters,
habituation/inhibition, focus/unfinished refs, events, bounded histories/receipts,
anchors, and digests. No source text, signal vectors, competition/prompt data,
Metacognition value/history/calibration, or arbitrary metadata is added.

## Lazy compatibility and capability-shaped capture

Loading valid retained v1–v8 does not upgrade or rewrite the canonical file.
`ensure_published` also preserves equal valid physical v1–v8 bytes, including
noncanonical formatting, instead of eager startup rewrite. Physical v0 remains
the existing separate migration case: its explicit stabilization publishes the
migrated v7 snapshot, and must not confuse a semantic v7 result with physical v7.
The absent-file bootstrap remains the existing v7 shape.

Capture selects v9 only when the internal owner exposes an exact concrete
`AttentionStatePort` **and** complete Belief/R13 topology. An attached Attention
port without preceding authority topology fails closed, not a lower-version
capture that silently discards Attention. Without the Attention capability,
existing v5/v7/v8 capture behavior remains unchanged. Thus the first normal
Attention-authoritative commit may publish v9; U4 does not force currently
unwired production MainLoop to claim empty Attention authority.

The existing `AgentStatePorts` typing protocol retains its three R13 members.
`AttentionAgentStatePorts` is a separate extension. Runtime discovery treats a
missing Attention member as absence, not malformed R13-only ownership.

V9 restore requires the Attention port even for an empty persisted continuity.
Restoring v1–v8 into an Attention-aware owner explicitly resets that owner to
`AttentionContinuity.bootstrap()` (revision zero, no event/focus/history/receipt)
without changing the retained snapshot version or inventing historical evidence.

## Bounded JSON preflight and malformed input

The shared codec scans raw snapshot/WAL JSON before general `json.loads` can
materialize an invalid Attention subtree or erase duplicate fields. It recognizes
object keys, escapes, and nested structure—not substrings in source text.
Attention's raw value is bounded by **4,193,496 B** before its strict canonical
parser runs. Programmatic typed/mapping inputs are also count/size preflighted.

The accepted Attention-specific nesting limit is **64**, unchanged from U1.
The outer iterative raw scanner has a separate finite **4096**-level limit;
this does not expand the Attention schema. Deep outer decoding or privacy-walk
recursion becomes bounded AgentState/WAL malformed errors, without rewriting
input bytes. Unknown versions, invalid counts/types/digests, duplicate Attention
or ambiguous v9 schema fields, and noncanonical Attention values fail closed.

WAL baseline and candidate snapshots use the same preflight. Duplicate enclosing
`baseline_snapshot`/`candidate_snapshot` keys reject if **any** occurrence is v9,
including escaped spellings or either v8/v9 order. Last-key-wins cannot hide v9
continuity. Existing legacy-only parsing behavior is not silently replaced by
a blanket new duplicate-key policy.

## Exact restore, rollback, and private compatibility seam

The persistence-only concrete `AttentionStatePort` binds one exact U2
`AttentionSystem`; it does not expose refresh or accept model/operator callbacks.
Its sealed thread/owner-bound transaction holds the actual owner RLock, validates
and stages a replacement bundle before other authority mutation, and checkpoints
the original immutable bundle. `publish()` swaps the reference, `complete()`
marks success only after committed read-view publication, and `close()` always
runs, restoring the original bundle unless complete and releasing the lock once.

This deliberately depends on U2's private `_lock`/`_bundle` in **one runtime
adapter module**. Accepted U2 source/API/policy is unchanged. The dependency is
tested and must be revisited if those internals change. A snapshot-only rollback
would be insufficient: public local restore intentionally removes ephemeral
competition/prompt witnesses, so it could not recover the exact prior read view.

Successful restore preserves persisted focus/counters/history/receipts exactly
and leaves competition/prompt metadata absent. Failure after Attention publication
restores both original continuity **and original ephemeral view** without policy,
source, Memory, model, prompt, or Metacognition calls. Other authority validation
and committed read views are staged before mutation as in the existing restore.

This is **Attention-local exact rollback**, not global linearizability of separately
read domain owners. The existing serialized runtime/restore precondition and
other-domain best-effort compensation remain unchanged; U4 does not claim that
arbitrary failing third-party compensation can be made infallible. Persisted
focus is continuity, not proof that sources remain fresh/renderable; later normal
U5 events must capture and reconcile actual current authority.

## WAL versions, crash boundaries, and true rollback

WAL record/manifest/boot-anchor schema and command versions remain **1**. Embedded
AgentState gains v9 through the compatible union; retained v8 bytes/hashes and
existing record chains retain their format. Reconstruction uses stored snapshots
and proofs, not event handlers or policy replay.

Both WAL append and internal commit permit ordinary **v8→v9**, retain rejection
of v8→older, and reject ordinary **v9→pre-v9** before WAL/canonical mutation or
Journal preparation. Explicit true rollback remains the existing classified
recovery-generation path, preserving external reconciliation gates. Rollback to
older v9 restores its exact continuity; rollback to retained v8 resets Attention
through the legacy initialization rule above.

Internal publication order stays Journal PREPARED → WAL transition → canonical
snapshot → separate Journal completion. Tests cover pre-WAL, WAL-only tail, and
canonical-publication-before-completion boundaries with exact v8/v9 identities.
Plain internal events recover their proper snapshot. Declared open **external**
transactions additionally retain PRE_INTERNAL/INTERNALLY_COMMITTED classifier
proofs and the existing R07 gate: bare startup recovery must not hide or retire
their unreconciled evidence. No synthetic cleanup bypass is introduced.

## Privacy and U5 carry-forward

Durable state remains reference-first: no raw prompt/transcript, hidden thought,
rendered source/proposition text, model rationale, provider errors, credentials,
or metadata escape hatches. Checksums bind values, not producers. U3 remains
event-scoped and non-durable.

**Mandatory U5 condition from Human U3 review:** `current_context_id`, Belief
records, and the coherent source set must come from actual current R09/R12
authorities inside the serialized event, never user/operator/API-controlled
assessment inputs. U4 does not implement or pre-approve that wiring.

## Verification commands

```sh
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_attention_state_codec.py tests/test_attention_state_port.py tests/test_agent_state_v9.py tests/test_r14_wal_recovery.py tests/test_r14_u4_scope.py -q
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_agent_state.py tests/test_agent_state_context.py tests/test_agent_state_v8.py tests/test_agent_state_capacity.py tests/test_state_wal.py tests/test_state_recovery.py tests/test_r13_wal_recovery.py tests/test_r13_restore_ports.py tests/test_r13_integration.py tests/test_main_loop.py -q
env -u SUZKA_CONFIG_PATH just check-all
git diff --check
```

## Executed local verification and review

- Final U4 codec/port/v9/WAL/scope: **102 passed**.
- Accepted U1/U2/U3 and source-import/scope suites: **183 passed**.
- Legacy AgentState/v8/capacity/WAL/recovery/R13/MainLoop subset: **349 passed**.
- Final `env -u SUZKA_CONFIG_PATH just check-all`: **1870 backend tests passed**,
  Ruff **PASS**, Mypy **117 source files PASS**, frontend **6 tests / build PASS**.
- Separate static compatibility check for R13-only and Attention-aware port
  owners: **PASS** (117 package files plus the temporary check, 118 total).
- The originally failing rounded-discriminator probe now rejects the payload;
  its regression is included in checked-in tests. Publication was withheld until
  this fix, supported Pydantic model-level validation, and the final gate passed.
- Existing startup load/restore ordering and explicit v0 stabilization remain
  covered; no test was disabled or error hidden. Capacity and accepted U1/U2/U3
  source/policy contracts are unchanged. Local Python is **3.13.12**.

Independent read-only architecture and raw-codec correctness reviews closed
their findings. Source-only review is separate from executed verification and
does not constitute Human PASS. The primary reviewed/integrated concrete diffs;
the focused verifier ran checks, with final expanded suites run by the primary.

Head-associated CI/provenance will be recorded in Draft PR #282 after publication.
Human U4 acceptance is not self-declared; stop for exact-head focused review.
