# R15 U6b — source-qualified private use in ordinary Chat (MVI)

Parent: #336, #334, #283, #323. Base: merged PR #333 on rebuild/develop.
The user explicitly approved implementation of #336, not merge, Source deployment,
new authority owners, stronger SourceEpoch attestation, or whole R15 acceptance.

## One narrow positive

An active, serialized CHAT/API_CHAT with a short explicit question about its
*previous interpretation* can read the same-context, already WAL-committed U3
S1=A / S2=B. Before the model sees any projection, the private reader checks:

- The **single bound** R15StatePort, existing AgentStateStore and runtime event.
- Process-local owner U3 root + NarrativeContinuation **exactly equal** the
  committed AgentState v10 roots (never caller-provided serialized proof).
- Full currently available terminal R07, active R12, DB1 and R09 anonymous
  Context for the original observed task E and S1, and E2/S2 if revised.
- Exact scoped option universe, sequential event provenance, first-person
  attempted choice and bounded later *user-reported* correction.
- No missing, mismatched, uncommitted, future, changed or uncertain Source;
  second WAL/owner snapshot equals the first one.

The private value includes status (QUALIFIED/HOLD), revision, earlier A/B,
current A/B/UNKNOWN, whether a later user report exists and, since IP-8a, a
closed status class (PRESENT/CONTESTED/CONTRADICTORY/UNKNOWN). An UNKNOWN class
hides both options and shows only "Current interpretation: undetermined.";
CONTESTED/CONTRADICTORY add one fixed English disagreement line. The read
re-derives both S1 and S2 episodes with the v2 table and compares them whole,
so a root admitted under the earlier table is HOLD. It has no user/model
quotes, source identifiers, personal identity, raw history, skill score,
hidden thought or Goal/Tool authorization.

## Safe bounded Chat consumer

Only an explicit, finite recall phrase **and** a recognizable cue to the
bounded referent-resolution exercise (such as 暫定解釈 / 比較実験) in the
user's current request allow the qualifying read to condition Chat. A generic
"前の案" about an unrelated plan is insufficient. This lexical gate is intentionally
conservative, not a general proof of user-topic intent. The projection
prefix contains only fixed strings + finite A/B (or "undetermined"), and labels all interpretations
**provisional**. It explicitly forbids a verified-intent/success/skill/trust
claim. New participant or changed Context forces HOLD. Otherwise the existing
prompt/generator is unchanged. Debug Chat is never given this private prefix.

A successful generation re-qualifies the same source and committed WAL
**before** any response/Experience/DB1 participant is created. If a formerly
qualified view becomes HOLD/different after generation, abort this Chat event
without publishing its source-conditioned response. This is a conservative
fail-closed outcome, not a claim of interruption-free ordinary response.

No new NARRATIVE_REVIEW event, R15 root mutation, public endpoint, R16
Decision/Action authority, WAL schema or Experience writer has been added.
Cold start/v9 or no U3 returns HOLD.

## Startup binding and exact change owners

- NEW suzka/r15/private_projection.py: bounded private read.
- suzka/r15/state_port.py: startup-only private binding and serialized read.
- suzka/runtime/main_loop.py: existing Chat prompt use and postgeneration check.
- suzka/api/server.py: **minimal necessary one-time wiring** passing the
  pre-existing app.state.agent_state_store and AgentRuntime immediately after
  the existing qualified Narrative source gate binds. This additional startup
  file was not listed in #336's initial three-file candidates. It adds
  no Source/StateWAL authority and needs explicit scope scrutiny before merge.
- tests/test_r15_u6b_runtime_use.py: synthetic provider over real isolated
  R07/R09/R12/DB1/V10 and restart/fault scenarios.
- This contract document.

## Limits and remaining acceptance gates

This is the finite referent-resolution **MVI**, not general semantic
introspection, accurate reconstruction of actual human intent, personality,
trust or self-competence. The prompt conditions a model, not its guaranteed
semantic honesty. Tests use a deterministic synthetic provider, not a real
model. Existing ordinary R12 Source reads may perform recovery; no advanced
G2 historical no-repair attestation is claimed (Post-R26 #320).

Runtime serialization prevents concurrent *subject* events in this process.
The pre/post model reads reject tested source changes, but cannot guarantee
external multi-owner atomicity between final check and HTTP publication.
Review and test source replacement, WAL race, user-visible leakage and
rollback behavior within this precise MVI scope. Do not broaden into a
general SourceEpoch system without separate authorization.

Exact-HEAD CI, independent full source Correctness/Security/Architecture
review and **separate Human Merge permission** are required. R15 U2/U4
positive updates and final U7 remain subsequent independently scoped work.
Full 128 MiB AgentState cap and untouched 16 MiB reserve stay binding;
this change adds no persisted field.

## R15 -> R16 interpretation read (IP-8a)

`R15StatePort.read_interpretation` (adapter:
`suzka/runtime/r15_interpretation_reader.py`) returns only closed values:
`EPISODE(stage, status class)`, `NOT_ELIGIBLE` or `UNAVAILABLE`. It runs only
inside the R16 plan-preview review event (`PLAN_PREVIEW_REVIEW` /
`RUNTIME_PLAN_PREVIEW_REVIEW`) and must be called before that event opens its
R07 transaction: the source read refuses while any transaction is open.

Order: an admitted episode for X is re-validated exactly like the private
read; otherwise X's own answer must match the prior-choice pattern exactly once
(else `NOT_ELIGIBLE` without reading anything else); then the nearest earlier
qualified CHAT turn of the same Context, found by a descending scan of the
retained Journal, decides `UNAVAILABLE` (setup-shaped) or `NOT_ELIGIBLE`.
Known fail-closed effects: no predecessor in the retained window (including a
Context's first turn, whose start sequence is not recorded) is `UNAVAILABLE`,
and so is any earlier turn of another Context whose Experience cannot be read
during the scan. Both only affect X whose own answer is R15-shaped.
