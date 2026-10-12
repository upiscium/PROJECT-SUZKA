# G1-P: Externally terminal Context projection

Status: isolated implementation candidate. This is NOT the G1-C event-time capture participant, a G2 source witness or a NarrativeSelf producer.

Tracking: https://github.com/upiscium/PROJECT-SUZKA/issues/288#issuecomment-6107935748

## Semantics

Public GET /api/contexts and GET /api/contexts/{id} use a single private immutable Context snapshot. They never dereference the mutable worker ContextRegistry. Public reads linearize when a READY snapshot is admitted under one short lock. The response is entirely materialized from that generation, and an already-admitted GET may finish transmitting after a newer event begins.

A pending normal CHAT or CONTEXT_UPDATE leaves the last EXTERNALLY TERMINAL Context as the public view. Internally committing AgentState does not update it. Before the event terminal Journal append, public Context is marked PENDING. New GET admissions then return generic 503. After the durable terminal and receipt-retention checkpoint, the entire new view is atomically published before the event future returns. Any uncertain failure in preparation, internal commit, finalization or terminal/publication leaves public GET DEGRADED (503). A verified ordinary handler failure restores the old snapshot and allows the previous ready view.

Context POST transition/relate results capture both the frame and current pointer inside the single serialized event. Returning the result waits until event terminal; later worker mutations cannot change its is_current value.

Public responses have terminal-sequence, Context-revision and visibility headers. These are an API read-generation label, NOT authority to say any episode has a genuine event-time Context capture.

Startup requires reconciled existing R07 Journal plus AgentState. When B has never been installed, it is NOT REQUIRED for this public-only projection and is NEVER backfilled. Older legacy snapshots without any Context root require an empty restored registry. Future installed-but-dirty B baselines cannot be treated as missing; introduction of B needs its own separate Human implementation gate.

## Exact reader scope

The two Context GET routes are rewritten. Context POST transition/relate replies are assembled inside active events. Serialized CHAT/debug handling and their worker-private ContextPromptView and R14 Attention decisions are unchanged. AgentState capture/restore remains unchanged. Existing Value/Belief/R13/Attention read models are currently published at internal commit and are NOT silently shifted by this G1-P change. Debug chat response is sent only after event terminal.

No public generic Context source proof or arbitrary historical lookup is added. No R08 or R14 policy changes, R07 record format changes, memory/DB access changes, B quota/ticket/participant, R15 root changes or private transcript persistence.

## Focused evidence

Tests cover: uncommitted Context GET suppression, pre-model completion GET, internal-commit versus external-finalization GET, PENDING 503, terminal swap, handler recovery, failed terminal and degraded public status, terminal-success-before-publication fault and restart, legacy absence of B, admin mutation response generation and private read-model state transitions.

Focused commands:
    nix develop --command uv run pytest -q tests/test_g1p_context_publication.py tests/test_context_api.py
    nix develop --command uv run ruff check suzka/api/context_read_model.py suzka/api/server.py suzka/api/routes/contexts.py
    nix develop --command uv run mypy suzka/api/context_read_model.py suzka/api/server.py suzka/api/routes/contexts.py

Required before Ready or Merge: all backend tests, exact-head CI, independent correctness and security review, CP7 complete public/private Context-derived reader inventory, and separate Human integration approval. Do not claim G2 physically pure source reading or P1 genuine autobiographical reinterpretation.

Design is bounded to G1-P. If the reader inventory reveals out-of-scope R14 or AgentState mutation needed, stop and request explicit Human rescoping.

## STARTED checkpoint race fence

Following independent Reviewer feedback, STARTED checkpoint admission atomically
moves the public read gate to STARTED_CHECKPOINT_PENDING. The lock is then
released BEFORE the durable Journal STARTED append, so new Context GETs
return generic 503 promptly even if disk I/O is stalled. A verified append
restores the old READY generation; a failed append leaves DEGRADED. A GET
admitted earlier may finish from its immutable old snapshot. No read gate
lock spans durable I/O, handler/model generation or coordinator callbacks.
This is an API consistency safeguard, not a G2 source verifier or a B lock.


## CP7: Context-derived reader inventory and adversarial coverage (2026-10-12)

**Applicability and trust boundary.** This inventory covers the actual G1-P
standalone source on `rebuild/develop`, not the separate R15 U1–U4 work. Its
public **Context** correctness contract is *externally terminal R09 projection*;
it does **not** make every other domain/read API externally terminal in the
same transaction. All of the following have been inspected against the
actual code, not inferred merely from OpenAPI routes.

| Reader / exact source | Context/R14 relationship | External visibility and publication rule |
| --- | --- | --- |
| `GET /api/contexts` and `GET /api/contexts/{id}` — `suzka/api/routes/contexts.py::_published_context,list_contexts,get_context` | Direct R09 Context identity, participant refs, frame status and current pointer | Only `CommittedContextReadModel.snapshot()`, never mutable worker Context; previously admitted whole snapshot or generic 503 during STARTED/PENDING/DEGRADED; unauthorized 401. |
| `POST /api/contexts/{id}/suspend,resume,close,relations` — `suzka/api/routes/contexts.py::_transition,relate_context` | Mutates R09 under real serialized event; response includes `is_current` | Response frame and current pointer are captured **inside the same handler**; `execute()` returns it after durable terminal, not from a later mutable registry. |
| `POST /api/chat` — `suzka/api/routes/chat.py::chat`, `runtime_execution.py::execute` | `main_loop._run_chat` creates/selects R09; reads ContextPromptView, R08 and R14 during the event | Only the completed response crosses HTTP. No GET reader of the uncommitted Context is introduced. |
| `POST /api/chat/debug` — `suzka/api/routes/debug.py::debug_chat` | Exposes private generated **prompt**, Memory excerpts and diagnostics for that DEBUG event; prompt can contain Context derived from **that** handler | Admin only; queued behind earlier AgentRuntime event, delivered after this DEBUG event terminates. A speculative current turn can enter its own prompt, **not** another request's pre-terminal response. Debug is not a source attestor. |
| `GET /api/values*, /api/values/config-seeds` — `suzka/api/routes/values.py` / `main_loop.value_system` | Existing R11 Value state can include declarative `context_ids`, **not** live R09 current Context | Detached R11 read projection is published at **internal** commit (pre-existing), not by G1-P. A concurrent normal CHAT does not change it in the tested matrix; G1-P does not promise cross-domain external-terminal atomicity here. |
| `GET /api/memory/search, episodes/{id}, semantic/{id}` — `suzka/api/routes/memory.py`, `schemas/memory.py` | Source-owned DB1/DB2 content and IDs, **no explicit current R09 Context field or R14 Attention projection in these API schemas** | Independently authorized admin source reads; not protected by G1-P's Context gate. Source-owner external finality/physical no-write/retention is a **G2** problem. User-provided text could itself mention a Context identifier; the CP7 negative test checks newly created **metadata**, not an impossible ban on arbitrary text. |
| `GET /api/state/emotion` — `suzka/api/routes/debug.py::emotion_state` | Live global R10 affect, **not** R09 actor/context attribution | Admin read can reflect in-flight R10 state; does not assert current Context terminality or external cause. Existing R10 contract is not changed. |
| `GET /api/adapters`, `GET /health` — `suzka/api/routes/adapters.py`, `suzka/api/server.py::health` | Adapter inventory / runtime health, no R09 or R14 data | Independent API read, tested as negative-control routes. |
| R14 private `main_loop.attention_view()`, `main_loop.r13_view()`, `_committed_read_views` — `suzka/runtime/main_loop.py` | R14 selection can derive from live Context/Working Memory; private R11/Belief/R13/R14 bundle | Existing bundle publishes at **internal** commit. At the CP2/CP3 test barriers `attention_view()` is genuinely newer than its pre-event value **while the public Context GET remains old or 503**. This is *deliberately disclosed as a distinct private generation*, NOT an externally terminal source proof; no direct HTTP route returns `attention_view`. Other trusted internal consumers must not mislabel it as G2 or leak it out of an active event. |
| Worker-local R09 `ContextRegistry`, `ContextPromptView`, R08 `WorkingMemory`, R14 `_refresh_attention_prompt` — `suzka/runtime/main_loop.py::_run_chat,_refresh_attention_prompt`, `suzka/runtime/working_memory.py` | **Intentionally** observes provisional per-event Context for prompt generation and selection | Runs inside the serialized handler, not a new public GET. Prompt text is available to the in-flight model and later to authenticated DEBUG response **after** terminal, not a separate pre-terminal client. No policy/focus change authorized here. |
| `AgentStateStore.capture/restore_into`, R07 coordinator callbacks, `context_committed_reads` — `suzka/runtime/agent_state.py`, `suzka/api/server.py` | Authoritative snapshot/candidate plus external-terminal publication | Internal capture may exist before external finality; `CommittedContextReadModel` is the only Context GET issuer. Durable errors go DEGRADED/503, verified handler rollback restores prior terminal view. No independent historical G1-C/B Context proof is emitted. |
| Other registered sleep/adapter POST endpoints — `suzka/api/routes/sleep.py,adapters.py` | No direct live R09/Attention read in these route handlers | Mutating/operational endpoints, not direct Context GET readers; unrelated endpoint outputs are not retroactively upgraded into Context source authority. |

**Executable CP7/CP7a matrix.** `tests/test_g1p_cp7_readers.py`
uses actual `TestClient`, the configured real runtime, independent HTTP
requests and deterministic callback/barrier events (no fake trusted G2 ports).
The suite asserts the **exact registered HTTP GET path set**, so new GET routes
force an inventory review. It probes every listed public/admin GET method,
including R11 config-seeds/detail/revisions and **successful** DB1 episode and
DB2 semantic detail reads using deliberately synthetic **pre-existing test
rows**. The DB1/DB2 fixture writers are legacy/test-only and do NOT establish
G2 source authority; no semantic lifecycle or no-write proof is inferred.
During each CP1–CP4 probe, Context GETs with **missing and forged** admin tokens
must both return 401, even when authorized callers see generic 503. A generic
503 is an availability/timing signal to an already-authorized caller, not a
promise that a pending transaction's existence is itself secret.

- **CP1** frozen model generation: R09 has created a provisional Context, but
  list/detail give prior terminal state/404. Every registered public/admin
  HTTP GET is probed, including positive old Memory details; missing and forged
  admin tokens are both rejected.
  Private committed R14 read view remains the previous generation here.
- **CP2** frozen R07 external finalization after internal commit: Context GET
  remains prior terminal/404 and other API readers do not disclose a newly
  introduced Context session ID; **private** R14 read view has already changed.
- **CP3** frozen event-terminal completion: new Context GETs return bounded
  generic 503 with no source ID or generation header; unrelated allowed
  HTTP readers are separately checked, and private R14 can remain ahead.
- **CP1–CP3/CP6** a second authenticated DEBUG_CHAT really enters the runtime
  queue (instrumented `AgentRuntime.submit`) but **cannot finish its HTTP
  response before the earlier event terminal**; only after release may its
  prompt/Context observations become visible. Admin POST generation is
  additionally tested in `tests/test_context_api.py`.
- **CP4/CP5** failing model handler: the old Context view and R09 registry
  are restored; cross-domain negative reads still do not expose a provisional
  session. Failure **after** durable terminal but **before** Context-view swap:
  all new Context GETs are 503, not stale READY; startup independently
  revalidates and reconstructs the real terminal Context.

**What these tests do NOT establish.** They do not imply that R10, R11,
source-owned Memory, the private R14 bundle or a future arbitrary internal
plugin all share the G1-P external-terminal publication barrier. The measured
R14 private early publication is **not** a public Context leak in today's
reachable HTTP routes, but it prohibits treating private `attention_view()`
as a G2 source proof. They do not establish G1-C historical Context capture,
physically read-only Chroma, unforgeable nomination, positive EvidenceView,
NarrativeSelf meaning/admission, or v10/WAL behavior. Any future consumer
exporting the private R14 view, or a discovered Context-derived HTTP field,
requires a new scope/test gate; **do not silently refactor R14 or AgentState
as G1-P**.

**Validation gate:** use the new exact standalone HEAD for scoped CP7 tests,
the existing G1-P/context tests, full backend checks and independent
Correctness/Security review. A green suite never replaces separate Human
Ready/Merge approval.
