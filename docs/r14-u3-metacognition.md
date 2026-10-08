# R14 U3 — pure event-scoped Metacognition

U1 and U2 remain accepted. This unit adds an ephemeral observation producer and
deterministic assessment function, not production integration. U4–U6 remain
outside this change. Human U3 review is pending; local checks are not Human PASS.

## Responsibility contract

- **Owns:** validation of supplied current typed observation inputs, bounded
  event/focus provenance, and deterministic current-event assessment heuristics.
- **May:** detach and validate inputs; observe resource metadata and exactly
  linked ordinary-active Belief records; return immutable observations and the
  existing U1 `MetacognitiveAssessment`.
- **Must not:** mutate sources or Attention, rerank/refresh/restore Attention,
  generate model inference, recommend Decision/Action, learn verified outcomes,
  change SelfModel capability/bias/traits, retain assessment history/calibration,
  or acquire runtime, persistence, route, scheduler, or prompt authority.
- **Depends on:** accepted U1 contracts/policy/bounds, U2 Attention continuity,
  neutral Working Memory/Emotion contracts, typed R13 current records, and pure
  R12 Belief records and structured conflict checks.
- **Used by:** later explicitly authorized event owners. No MainLoop,
  PromptBuilder, AgentState v9, WAL, or API wiring is added here.

## API and source authority

`observe_metacognition(event, attention, *, working_memory_items=None,
working_memory_revision=None, working_memory_view=None, emotion_state=None,
focus_records=(), belief_records=None, current_context_id=None)` produces a
sealed `MetacognitionObservation`. Its public constructor rejects direct metric
or generic-witness construction. `assess_metacognition(observation)` validates a
detached copy and returns the existing immutable assessment shape. Neither API
accepts caller-specified scores, weights, confidence, model self-reports, or
arbitrary `MetacognitiveEvidenceWitness` labels as evidence authority.

The focus witness binds the supplied Attention root revision, state digest, and
focused IDs to the assessment event. A newer assessment event is allowed;
future roots, regression, and inconsistent retained event identities are not.
Supplied focused R13 records must match the current Attention primary source
kind/ID/revision/digest/event and active/adopted lifecycle. The observer does not
refresh stale joins. Beliefs must be joined through exact typed BELIEF targets
or evidence references; Motivation `related_refs` are also explicit links.
Working Memory episodic/semantic IDs, text similarity, and source score are not
Belief links. Context ID is only an opaque trusted scope gate, not knowledge.

Working Memory membership/revision must be supplied together. A view requires
complete coherent membership, decision/selection coverage, and source revision
and byte checks. Bounded UTF-8 preflight precedes content encoding. Missing view
means unknown load, not an empty measured view. Neutral Emotion is a derived
assessment-event observation: it has no invented R10 source event or revision.
Legacy finite `optimal_loss`, including negative values, remains valid but is
not used as confidence or pressure.

Resource witnesses marked `SUPPORTING` support **observed resource facts only**
and have no confidence ceiling. Linked ordinary-active Belief witnesses observe
**recorded confidence**, not world truth or evidential polarity. Value confidence
is normative and is not accepted as factual confidence. Experience ACTIVE status
does not prove committed authority, so no Experience shortcut is added. Raw
user/model text and R12 evidence categories do not establish support polarity.

Checksums bind declared values; they do not authenticate the producer. Supplied
snapshots must be coherently captured by the trusted future event owner. U3
checks exact typed relationships, bounds, checksums, and retained chronology,
but cannot prove global source freshness/completeness or committed capture from
an in-process value alone. Belief records have no stored whole-record digest;
U3 recomputes that digest and validates nested proposition/revision digests.

## Fixed deterministic V1 policy

All units use scale **1000000**, exact binary64 rational conversion, and integer
flooring. There are no per-event configuration or operator priority inputs.

| Measurement | Rule |
| --- | --- |
| Load | `max(membership_count / item_capacity, projected_bytes / projection_max_bytes)` with a complete coherent view only |
| Saturation | `min(focus_count / focus_capacity, 1)`; actual current arousal >= 0.75 uses capacity 8, otherwise nominal policy capacity 16 |
| Emotion influence | `max(arousal, abs(valence))` from observed current neutral Emotion only |
| Supplied-link coverage | Distinct focus targets covered by supplied ordinary-active linked Beliefs / full actual focus count |
| Quality | `scale - floor((400*load_units + 350*saturation_units + 250*emotion_units) / 1000)`; all three inputs must be observed |
| Confidence | `floor(min_recorded_ceiling_units * coverage_units * quality_units / scale**2) / scale`, also capped by the raw supporting typed ceiling |

Missing Emotion does not become observed arousal/influence zero. The nominal
capacity is a policy bound; quality still stays unknown without observed Emotion.
Measured zero resource load/influence is distinct from `None`. No focus means an
undefined coverage denominator. Coverage also stays unknown for an empty Belief
observation, any missing declared link, or a supplied linked Belief that is
inactive, proposed, expired, uncertain, or outside the current scope. Known
partial coverage is only a supplied verified-link fraction, not proof of all
relevant knowledge. Confidence requires observed coverage, quality, and a
supporting typed recorded-confidence ceiling; missing any means `None`.

The resource-pressure weights **(400, 350, 250)** are fixed heuristics, not
empirical calibration or stable capability/bias measurements. They reduce
current quality/confidence; they do not change Belief truth or generate actions.

All absent/all-UNKNOWN evidence produces `UNKNOWN` with **every numeric field
unset**, even if the retained focus count alone could be divided by a policy
capacity. Other observed non-conflicting inputs produce `UNCERTAIN`. U3 **never
emits `SUFFICIENT`**: supplied-link coverage, even 100%, cannot establish the
completeness of relevant evidence. The accepted U1 enum remains unchanged.

`CONTRADICTORY` requires two current ordinary-active Beliefs sharing at least
one exactly linked focus target and an existing structured conflict candidate:
same structured subject/predicate, different object, compatible current scope.
Different unstructured text, missingness, high load, and incompatible scopes
cannot create contradiction. This observes a conflict, not which Belief is true.
Known conflict coverage is reported as sufficiency zero. Numeric confidence
zero still requires an actual supporting ceiling and observed coverage/quality;
without them confidence remains unknown. Only the existing closed reason codes
are used. Provenance/evidence boundaries remain explicit.

## Bounds, privacy, failure, and recovery

- Focus-record inputs: at most **16**. Belief inputs and combined evidence:
  at most **32**. Resource witnesses count against that same combined bound.
  Overflow rejects explicitly; no witnesses or conflicts are silently dropped.
- Reasons: at most **16**, unique and sorted. Observation canonical ceiling:
  **26750 bytes**; existing assessment ceiling: **26926 bytes**. These are
  conservative independently derived ephemeral envelopes, not persisted quotas.
- No source record, prompt, rendered content, proposition, or free-form rationale
  is retained in the observation/assessment. Output contains bounded opaque
  references, digests, events, closed reasons, and measurements. Input text may
  be validated/hashed for provenance, not interpreted as scoring authority.
- Malformed, over-bound, future, stale-join, or tampered inputs reject without
  mutation or checksum repair of published values. Retained Belief revision
  events must be chronological; retained current histories must bind the latest
  revision to its subject admission and admission digest. Exact permitted shared
  source events remain valid. Historyless legacy provenance remains explicitly
  limited, without fabricated events. Repeating exact inputs produces identical
  canonical assessment bytes.
- No state is published or durably stored. U2 restore retains focus continuity
  but leaves competition/prompt witnesses absent. U3 does not recreate them or
  treat retained focus alone as fresh source evidence. A caller can later supply
  fresh typed observations explicitly; this does not modify the restored owner.
- No format migration or capacity changes: full Attention universe **4192**,
  Attention JSON **4193496 bytes**, v9 including full future reserve
  **130544419 bytes**, and remaining margin **3673309 bytes** remain unchanged.
  U3 does not spend future reserve or change the v8/v9 codec.

## Verification commands

Run from the R14 worktree, without a process-local config override:

```sh
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_metacognition_evidence.py tests/test_metacognition_assessment.py tests/test_metacognition_assessment_boundaries.py tests/test_metacognition_contracts.py tests/test_r14_u3_scope.py -q
env -u SUZKA_CONFIG_PATH uv run pytest tests/test_attention_contracts.py tests/test_attention_bounds.py tests/test_attention_adapters.py tests/test_attention_policy.py tests/test_r14_source_contract_imports.py tests/test_r14_u1_scope.py tests/test_attention_system.py tests/test_attention_system_boundaries.py -q
env -u SUZKA_CONFIG_PATH just check-all
git diff --check
```

Tests cover source binding/chronology, scalar missingness versus measured zero,
fixed-point math and confidence ceilings, conflicts versus unstructured text,
privacy, source purity, bounds/overflow, full-size Attention roots, fresh-process
dependency neutrality, and restore without reranking or feedback. Compatibility
and full backend/frontend checks remain required before delivery.

## Local verification evidence

Final U3 source state was checked on Python 3.13.12:

- Focused U3, accepted assessment contracts, and U3 scope: **56 passed**.
- Accepted Attention/source-import/U1-scope/U2 suites: **127 passed**; the
  accepted 12 assessment-contract tests are also included in the focused run.
- Working Memory/Context/Emotion/R13/v8/WAL compatibility: **370 passed**.
- Ruff: **PASS**. Mypy: **115 source files, PASS**.
- `env -u SUZKA_CONFIG_PATH just check-all`: **1768 backend tests passed**,
  **6 frontend tests passed**, and frontend production build **PASS**.
- Capacity derivation: unchanged. `git diff --check`: **PASS**.

Independent read-only architecture/correctness reviews were used to address
source bounds, missingness, event identity/chronology, and admission binding.
Their source reviews are separate from the executed verification above. A first
full-gate attempt exceeded the shell's 120-second limit; reruns with a 600-second
limit completed successfully. No test failure was hidden or check disabled.

These are local results, not new-head CI or Human U3 acceptance. Human authorized
U3 publication on 2026-10-07. Head-associated CI and exact checkout provenance
will be recorded in Draft PR #282; Human U3 exact-head focused review remains
pending. U4–U6, Ready, and Merge remain outside this authorization.
