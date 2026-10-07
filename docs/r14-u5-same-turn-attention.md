# R14 U5 — production same-turn Attention / prompt integration

U4 is accepted at `ee7e9f849e0b6f0f3ccbd455a7acae9800a8a056`,
[Human review 6035601273](https://github.com/upiscium/PROJECT-SUZKA/pull/282#issuecomment-6035601273).
U5 implements production wiring only. Human U5 focused review is pending.
U6, Ready, and Merge remain unauthorized; PR #282 remains Draft.

## Responsibility contract

- **Owns:** same-turn capture/refresh in the existing serialized chat handler,
  source-certified selected prompt rendering, and committed Attention read-view
  publication through the existing v9/WAL pipeline.
- **May:** consume actual owned current R08/R09/R10/R12/R13 state; refresh the
  private U2 owner once for the bound event; render exactly its accepted selected
  IDs; expose detached committed reference/witness metadata.
- **Must not:** accept user/operator/API focus or score authority, infer links
  from text, use current uncommitted Experience, alter source truth, rank with a
  model, persist raw render payloads/Metacognition, or acquire Decision/SelfModel,
  outcome-learning, scheduler, new event-loop, or direct focus-mutation authority.
- **Depends on:** accepted U1–U4 contracts/policy/bounds/port and existing
  AgentRuntime/Journal/WAL/transaction publication and failure semantics.
- **Used by:** later individually authorized U6 integration evidence and future
  R16/R19 consumers. This unit does not implement those owners.

## Actual source authority and same-turn order

Bound ordinary and debug chat share `_run_chat`:

```text
actual active AgentRuntime handler event
 -> resolve current Context through R09
 -> temporal Emotion / calibrated measurement / Appraisal / Emotion update
 -> committed Memory retrieval
 -> WM advance / admission / contextual selection
 -> capture complete current WM membership/revision/view and owned R13 snapshots
 -> validate actual current R09 frame and snapshot actual R12 Belief authority
 -> accepted deterministic Attention refresh
 -> certified selected prompt construction
 -> model generation
 -> current Experience/participant construction
 -> normal internal durability / external finalization
```

Refresh uses the actual event's ID, processing sequence, and UTC requested time.
The producer checks that the same event remains active on the runtime worker.
There is no synthesized event, clock-based ranking, timer, or API focus command.
The current Context frame must equal the registered ACTIVE/current R09 frame;
request selector strings are not themselves authority. Belief is read from the
owned current R12 system, never a request-supplied tuple. The complete coherent
source set is captured inside the serialized event from the actual owners.

All current WM members, including unavailable/unselected refs, participate in
the **4192**-target supported universe: 4096 WM + 32 per R13 domain. Only R08
already-resolved eligible rows can render. Attention does not resolve/promote
an R08-unselected ref or mutate membership. R13 snapshots are complete owned
current captures, not a caller-created prompt subset. Global current Emotion
controls only the accepted focus cap (16, or 8 at arousal >=0.75), not an
invented candidate-specific emotion signal.

The new Experience is created after generation and is never used circularly as
already-committed evidence for that response. Values/Context retain their existing
projections/authority. Raw user/model text may influence reviewed upstream
authorities but is not a direct candidate/priority/confidence input.

Eventless unbound legacy calls retain their existing prompt-builder compatibility
path and do not manufacture an Attention event. Production-bound calls require
the certified selected-payload path; an unsupported custom builder fails before
generation rather than silently dropping selection and rendering all sources.

## Source proof versus actual model-visible render

The accepted U1 WM canonical source row includes `item_id` and `source_id` for
identity/byte/digest witnessing. Existing model prompts do not expose those
opaque IDs. U5 preserves that privacy boundary without changing U1 adapters,
policy, selections, or persisted capacity.

`build_attention_prompt_payload(...)` revalidates the full source capture,
projection/competition/prompt coverage, current refresh/receipt/root binding,
and each included canonical source row's exact original bytes/domain digest.
No policy selection or competition is rerun during rendering.

For display, only WM's structural `item_id`/`source_id` fields are removed;
`source_kind` and `text` are unchanged. Natural identifier strings within the
source text are preserved, not censored. R13 minimal rows remain byte-identical
to their accepted source-row encodings. All text uses canonical JSON escaping;
newlines/non-ASCII/control characters are not cut or silently shortened.

The sealed request-scoped `AttentionPromptPayload` records **distinct**
`witnessed_bytes` and actual `rendered_bytes`/`rendered_digest`, bound to the
original selection digest and exact selected IDs. A redacted row does **not**
claim the original ID-bearing row's digest. Fixed intro/section headers and one
newline per included row are measured exactly. Section order is fixed and each
kind retains its included focus-order subsequence.

```text
actual selected render bytes <= witnessed selected bytes <= 131072 bytes
```

Removing ID fields never permits repacking extra rows. Omitted/oversized rows
remain explicitly omitted under the original Attention decisions; included IDs,
ordering evidence, and source text are not changed. A stale capture, changed
content/revision, missing ephemeral selection, forged packet, or tampered payload
rejects without checksum repair or hidden fallback. Text length is bounded
before UTF-8 allocation; actual UTF-8 bytes and all published fields are checked.

**131072 bytes bounds the Attention-selected contribution, not the entire model
prompt.** Existing Context, Values, Emotion, instruction, and User sections remain
outside that contribution. Legacy no-payload rendering remains unchanged.
The selected path branches before legacy WM lists or full R13 rendering, so the
all-active ~3.58 MB R13 fallback is not assembled or appended to a governed turn.

## Private owner, v9 persistence, and committed reads

MainLoop owns a private exact `AttentionSystem`. A trusted constructor-supplied
system is cloned, not shared; no public System/refresh/refocus mutation API is
added. The actual owner is attached through the accepted exact U4 persistence
port, making normal complete-topology production captures **v9**. Fake/legacy
owners without that capability retain the accepted capability-shaped formats.
There is no alternate Attention persistence file or schema/capacity change.

The committed read aggregate now contains Value, Belief, complete R13, and only
reference/witness `AttentionSelectedView` metadata. `attention_view()` returns a
detached validated copy of that aggregate, not speculative live owner state.
Compatibility domain publishers preserve its Attention member. Raw render text
stays request-scoped; it is never put in the aggregate, AgentState, or WAL.

The server stages a candidate-bound four-domain view **before** internal commit,
checking that normal live selection metadata matches the captured v9 root.
After durability succeeds, one prepared aggregate reference is published;
no new source lookup, cloning, or hook discovery is deferred until afterward.

Restore supplies the explicit target Attention snapshot (or legacy bootstrap) to
the new preparation hook before live owner mutation. It never stages the old
live speculative root or regenerates ranking/render/assessment. The old three-
argument hook remains available for compatible owners.

Handler/preparation failure restores the prior committed snapshot and republishes
the **same prior committed aggregate handle**, retaining prior public ephemeral
selection metadata. The live owner can have its metadata cleared by successful
cognition-free snapshot restore; that is not a new public commit. Non-chat commits
retain the existing Attention event/provenance rather than inventing fresh rows.
Internal durability failures remain fail-stop: prior public views stay published;
restart reconstructs the exact persisted v9 focus with no competition/prompt
witness reconstruction. Existing reconciliation gates remain intact.

The U4 private `_lock`/`_bundle` bridge remains isolated in its one adapter.
U5 uses U2 public snapshot/view methods and does not widen that dependency.

## Metacognition and later ownership

U5 does **not** wire the optional U3 assessment calculation into production.
This avoids making optional evidence overflow a chat failure and introduces no
assessment prompt feedback, decision effect, calibration history, or storage.
If later authorized, its Context/Belief/coherent capture inputs must still come
from actual current R09/R12 authority inside the serialized event, never direct
user/operator/API payloads. U3's pure contracts and uncertainty semantics are
unchanged. U6's whole-R14 F-map/review remains a separate, unapproved unit.

## Preserved capacity, privacy, and verification

All accepted contracts remain: **4192** targets; **4,193,496 B** Attention value;
v9 + full **16,777,216 B** reserve **130,544,419 B**; remaining **3,673,309 B**.
Only v9 production ownership/wiring changes. Codec, WAL versions, policy weights,
focus/history/receipt bounds, and Metacognition non-durability are unchanged.

Tests use coherent global R13 source-event identities, not stripped witnesses.
Production capture assertions now expect v9 while explicit retained-v8 inputs and
eventless rendering remain legacy-compatible. Retrieved WM JSON rows are decoded
for content assertions rather than assuming legacy raw-newline formatting.

Full-96-R13 stress fixtures repeatedly exercise existing R13 decoding/validation.
Their new future waits are finite **60-second test bounds**, not a production
timeout or latency SLA. A serial reproduction showed active R13 decode/validation
and eventual completion, not a persistent lock deadlock; no assertion or accepted
codec was weakened to disguise that cost. No live-model/hardware benchmark is
claimed by these deterministic checks.

```sh
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_attention_prompt.py tests/test_attention_main_loop.py tests/test_attention_production.py tests/test_r14_u5_scope.py -q
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_main_loop.py tests/test_r13_integration.py tests/test_fastapi_backend.py tests/test_prompt_builder.py tests/test_agent_state_v9.py tests/test_r13_evidence_matrix.py -q
env -u SUZKA_CONFIG_PATH just check-all
git diff --check
```

## Executed local verification and review

- Final dedicated U5 renderer/MainLoop/production/scope: **25 passed**.
- Accepted U1–U4/source-import/scope suites: **285 passed**.
- MainLoop/R13 production/FastAPI/legacy builder/v9/R13 evidence/scope subset:
  **183 passed**. Explicit retained-v8 and eventless fixtures remain covered.
- Final `env -u SUZKA_CONFIG_PATH just check-all`: **1895 backend tests passed**,
  Ruff **PASS**, Mypy **118 source files PASS**, Frontend **6 tests / build PASS**.
- The full 4192-source handler, model-time prior-view visibility, post-success
  non-chat/failure visibility, pre-durability read-view rejection, published-save
  fail-stop/restart, forged R09 frame, structural-ID privacy/natural-ID text,
  whole-row omission, source tamper, payload no-repair, and legacy rendering all
  have executed regression coverage. No source event witness was stripped.
- The investigator lacked a command tool and did not establish the timeout cause;
  the primary ran the serial stack-trace reproduction and the final two scenarios
  (**2 passed**, 139.36 seconds total). Test waits remain finite; no production
  performance benchmark or latency guarantee is inferred from those results.

Independent read-only architecture and certified-render correctness reviews
closed their findings. The primary integrated concrete diffs; the focused
verifier's only remaining failure was the new scope test's cleanup method name,
corrected to the existing `shutdown()` API before the final executed suites.
No assertion was disabled or validation failure hidden. Local Python is 3.13.12.

Head-associated CI and actual checkout provenance will be recorded in Draft
PR #282 after publication. Local review/checks do not self-declare Human U5 PASS;
stop for U5 exact-head focused review, with U6 still unapproved.
