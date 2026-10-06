# R14 U1 contracts, source map, fixed policy and capacity

This is implementation evidence for #281 / Draft PR #282, not a readiness or
focused-review verdict. D1–D13=A and A1–A24 remain authoritative. Only U1 is
implemented: immutable shapes, pure source adapters and competition functions,
resource limits, and a future-v9 capacity **projection**. There is no
AttentionSystem, Metacognition computation, MainLoop/PromptBuilder integration,
AgentState v9 field, WAL/recovery change, route, scheduler or R15–R19 producer.

## Target and source boundary

The mechanically derived simultaneous universe is **4192**:

| Current target kind | Source-owned hard maximum |
| --- | ---: |
| WorkingMemory member | 4096 |
| ACTIVE Motivation | 32 |
| ADOPTED Goal | 32 |
| ACTIVE Commitment | 32 |

No tracked-cache limit narrows this universe. WorkingMemory membership, not
only its already-selected content, supplies the WM identities. Unresolved,
archived, unavailable and projection-omitted members remain in the complete
source projection with explicit unavailability; they do not become valid
prompt content. Full supplied snapshot/view coverage is checked. Capturing
those separate source reads coherently remains the later ordered producer's
responsibility; a pure adapter cannot prove atomic capture by its caller.

| Input | Exact witness and signals | Deliberate exclusions |
| --- | --- | --- |
| WM member + optional current R08 view | Canonical item/source identity, current WM revision, Attention-derived metadata digest; activation/salience; exact flat R09 compatibility when available; resolved row byte/digest evidence | R08 score is validated, not used as an Attention priority. Missing content does not erase known activation/salience. |
| ACTIVE Motivation | Record ID/revision/record digest/latest revision event; strength, persistence, satiation, uncertainty | No model rationale, candidate evidence or inferred Goal meaning. |
| ADOPTED Goal | ID/revision/record digest/latest revision event; deadline urgency at the explicit caller event time | No priority/utility/Goal-winner override; proposals and other lifecycles reject. |
| ACTIVE Commitment | ID/revision/record digest/latest revision event; deadline urgency at explicit event time | No responsibility-kind hard override, acceptance/release proof producer or automatic Action. |
| Flat R08/R09 Context evidence | Exact typed known relation and existing fixed score | LEGACY_UNKNOWN / UNKNOWN_CONTEXT remain absent signals; unsupported composite protocol evidence is not interpreted. |
| Current R10 EmotionState | Sealed event-scoped **derived observation** of arousal | No invented upstream Emotion ID/revision/event. Global arousal changes resource capacity only, never candidate salience/novelty. |

Value, Belief, Experience and standalone Appraisal signal adapters are not
introduced in U1. Their closed witness kinds may be represented, but no
current adapter fabricates their relations or committed status. Source text,
similarity, embeddings, model certainty and arbitrary caller/operator scores
are not signal authority. Unsupported novelty remains `None`.

Source-local revisions and Attention revisions are independent namespaces.
Shared source event identity/sequence/time, when available, must be coherent
and not future. Hashes prove exact internal binding, not producer authentication.

## Deterministic policy

Policy is version 1; units are integers in `[0, 1000000]`. Source binary64
fractions are converted by exact `Fraction.from_float` and integer flooring.
No configurable weights, clock, random tie-break or model ranking is used.

| Base dimension | Fixed weight |
| --- | ---: |
| activation | 2 |
| salience | 2 |
| strength | 2 |
| persistence | 1 |
| urgency | 2 |
| context compatibility | 1 |

The sum is 10. Each unavailable weighted dimension contributes a **policy
fallback of 500000 units**, not an observed zero. The original `None`, missing
dimensions and fallback reason remain in evidence/digests. A measured zero is
different. A neutral fallback lets heterogeneous upstream schemas compete
without pretending that each domain measured all six dimensions.

From the weighted base, measured satiation subtracts at most 200000 and measured
uncertainty at most 100000. Continuity then applies:

- incumbent focus bonus: 100000;
- focused streak bonus: `min(streak, 4) * 12500`;
- nonfocused unattended streak bonus: `min(streak, 4) * 12500`;
- entering a different focus while prior focus exists: subtract 50000;
- habituation penalty: at most 150000, proportional to retained units;
- inhibition penalty: at most 500000, proportional to retained units.

The result is clamped to `[0, 1000000]`; minimum eligible score is 150000.
Numeric score descending and stable typed candidate ID ascending give the
total order. Source kind never bypasses that competition. These are Attention
resource scores, not source truth, Goal priority, utility or Action selection.

Pure continuity arithmetic increases focused habituation by 50000 (capped),
recovers unfocused habituation by 25000, and recovers inhibition by 25000.
Focused/unattended counters are mutually exclusive **current streaks**, not
lifetime totals; overflow rejects rather than clamps. Each candidate's streak
is bounded by the event sequence independently; counts across candidates are
not summed against one global event count. No helper mutates continuity.

Deadline urgency has a fixed 86400-second horizon. No deadline means `None`;
due/overdue means 1; a deadline at least one horizon ahead means measured 0.
Intermediate urgency uses explicit UTC timedelta microseconds and fixed-point
integer flooring, then the common binary64 signal quantization. Source deadline
and record digest are unchanged.

Competition returns a proposal with decisions for every supplied candidate.
It does not advance history, receipts or counters. Re-evaluating the same exact
last event is evaluation only; U2 must enforce operation-input identity using
retained receipts before publishing any transition. Restore never reranks.

## Focus and prompt resource contracts

- Focus capacity: **16**; exact global arousal at/above **0.75** reduces it to
  **8**. Absent Emotion keeps the default 16 with explicit missingness, not an
  invented observed arousal of zero.
- Unfinished refs: **16**, current eligible identities only. Overflow returns a
  bounded fail-closed error carrying the proposed union (at most 32 IDs), never
  silently evicting unfinished evidence.
- Final Attention-selected projection budget: **131072 bytes**.
- Fixed ASCII framing: **241 bytes**, derived from the exact authority intro
  and four headings in `policy.py`; one additional newline per included row.
- Maximum source-rendered byte witness: **100663555 bytes**, derived from R08's
  16 MiB source projection limit, worst-case ASCII JSON escaping and the actual
  closed WM row metadata. This is not the final prompt budget.

Source adapters retain only exact row byte counts and digests, never rendered
content. The R13 row format reuses the reviewed minimal current-state serializers;
the WM format is closed JSON with item/source IDs, source kind and resolved text.
Text length affects explicit byte admission, not importance scoring.

Prompt selection walks the deterministic focus order. A single too-large row
receives `OVER_BUDGET_SINGLE`; insufficient remaining space receives
`OMIT_BYTE_BUDGET`; other rows may still fit. No source is truncated or mutated.
The result is a selection/byte witness, not an actual model prompt. Later U5
must use exactly those witnessed encodings or revalidate bytes/digests before
rendering. Current ordinary prompts remain unchanged in U1.

## Continuity and event-scoped assessment

Attention continuity is reference-first: all current candidate identities,
primary source witnesses, availability, quantized habituation/inhibition,
streak counters, current focus/unfinished refs, and bounded event/revision
evidence. It contains neither source payloads nor source-derived scoring vectors.

Retained global revisions are bounded to **16**, receipts to **256**. Independent
immutable anchors fence each suffix; matching retained revision/receipt events
must agree on identity/time/result-state digest. Exact retained event+input
replay is identifiable; older unverifiable replay fails closed. Missing proofs,
broken digests, future schemas, dangling focus and count/counter conflicts do
not get repaired. Published projection/result checksums are revalidated against
copies rather than recomputed into mutated input objects.

`MetacognitiveAssessment` is a current immutable value, not a computation or
calibration store. Its evidence sufficiency, epistemic boundary, bounded
confidence/load/saturation/Emotion influence/quality, typed focus/evidence
witnesses and closed reasons contain no capability, bias/trait, recommendation,
Decision, Action or outcome-history fields. Missing evidence remains distinct
from contradiction and overload; confidence requires a typed supporting bound.
Its conservative event-scoped JSON ceiling is **26926 bytes**, with at most
**32 evidence witnesses** and **16 reason codes**. **No assessment history is
persisted or counted as a v9 field.** U3 owns actual assessment derivation.

## Exact future-v9 capacity gate

The continuity derivation uses actual canonical serializers and per-target-kind
legal row envelopes, not process-local guards or a selected sample. It composes
all **4192** candidate slots, full revision/receipt suffixes and anchors, maximum
legal IDs/enums/counters/UTC timestamps, and complete key/separator/root bytes.
An executable legal full-bound fixture matches that envelope and round-trips.

| Continuity component | Maximum canonical bytes |
| --- | ---: |
| candidates | 3995505 |
| focused IDs | 1073 |
| unfinished IDs | 1073 |
| last event | 224 |
| revision history | 44961 |
| receipts | 149249 |
| revision anchor | 538 |
| receipt anchor | 514 |
| revision | 10 |
| schema version / policy version | 1 / 1 |
| state digest / authority digest | 66 / 66 |
| Root braces, keys and separators | 215 |
| **Complete Attention JSON value** | **4193496** |

| Future v9 projection | Bytes |
| --- | ---: |
| Unchanged v8 base maximum | 109573688 |
| Attention JSON value | 4193496 |
| New `attention_state` root field overhead | 19 |
| **v9 before reserve** | **113767203** |
| Full future-state reserve | 16777216 |
| **v9 including reserve** | **130544419** |
| Hard cap | 134217728 |
| **Remaining margin beyond reserve** | **3673309** |

The admissible R14 addition was 7866824 bytes, including the new root key;
the JSON value ceiling is therefore **7866805 bytes**. V8→v9 version digits both
occupy one byte. The existing projector is called only in a test with explicit
`base_schema_version=8`, `schema_version=9`, and the derived `attention_state`
value maximum. **No AgentState v9 production schema/migration exists in U1.**

## Dependency and verification caveats

Existing immutable WM contracts/ID primitives and ContextRelation were moved
to neutral modules; existing EmotionState and unchanged validation helpers were
likewise moved without importing appraisal/provider stacks. Legacy paths
re-export the same classes, fields, IDs and validation; selection, admission,
Emotion/appraisal computation and persistence behavior are unchanged. Defining
Python `__module__` paths change; canonical persisted formats do not. Legacy
module attributes remain available, but no new pickle persistence contract is
introduced or claimed.

U1 tests cover pure imports, exact source/lifecycle/event relationships,
unknown/zero/contradiction distinctions, policy math/ties/continuity, count and
byte boundaries, proof/checksum tampering, parser resource limits, full schema
capacity and compatibility re-exports. Final focused/full results and exact
head-associated CI are reported in Draft PR #282, not inferred from this file.
Stop after U1 publication for exact-head focused review; U2–U6 remain unauthorized.
