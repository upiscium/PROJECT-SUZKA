# R15 MVI U6a — private committed-cognition projection (test-only)

Related [Issue #334](https://github.com/upiscium/PROJECT-SUZKA/issues/334).
Based on the actually merged [PR #333](https://github.com/upiscium/PROJECT-SUZKA/pull/333)
at \`rebuild/develop@36e038793f8d157f4280ffb3695fcc9a9a33b985\`.
This experiment is **not a production U6 implementation** or acceptance of all
R15. It changes NO \`suzka/\` module, database owner, API, AgentState codec,
StateWAL, Context registry or currently installed service.

## Why it exists

U1–U5 can create and persist a narrowly scoped first-person interpretation
of the observed answer to a closed A/B comparison. That is not sufficient to
feed it back into a later model turn or R16 Decision. A genuine U6 needs a
small, read-only **committed and currently source-qualified** projection,
which returns HOLD when evidence has become unavailable.

**Owns:** experimental internal read-only projection of an already committed
NarrativeSelf interpretation (original A, later reported B, or HOLD).

**May:** read the actual V10 snapshot from \`AgentStateStore\`, compare it with
the exact process-local U3 root from the existing private \`R15StatePort\`,
reopen **all** referenced R07/R12/DB1 records via the current qualified source
gate and R09 anonymous Context, check model S1/S2 against actual E/E2
statements, recheck committed bytes, and return only a tiny typed value.

**Must not:** mutate any U1–U5 owner; produce an R07 NARRATIVE_REVIEW; install
an interpretation from an operator/model/user assertion; claim E2 establishes
objective human intent, person identity, capability or reciprocity; invoke
R16 Goal/Decision, R17/18 Action or publish a public Chat projection. No
unbounded private text, source IDs, prompt, model rationale or user identity
appear in the returned view. There is no new production persistence schema.

**Depends on:** existing P1b-A ordinary cooperative read contracts and P1b-B1
strict V10 WAL recovery. R07/R12 owner reads may run normal backend recovery:
this prototype **does not** certify nonrepairing historical Source reads or
global SourceEpoch (deferred #320).

**Used by:** a later, separately approved **private** U6 owner projection /
safe-expression or R16 context reader. The current code is NOT linked into
\`MainLoop\`, \`/api/chat\`, any prompt builder or Decision/Goal authority.

## Test and truth boundary

The synthetic trial uses the existing deterministic test model through a
fresh isolated SUZKA \`TestClient\` with **real R07/R12/DB1/StateWAL**:
before a source-qualified S1 there is no private current hypothesis;
after E→S1, only A may be privately read; after independently committed
E2→S2, B may be read along with the original A; a separate no-E2 case
retains A. Each case is then shut down and restarted into a new MainLoop:
same private typed view is obtained without any new Narrative review event.

Missing either original E or later E2, a newly ambiguous R09 participant,
an uncommitted process-local-vs-WAL root mismatch or an unavailable WAL yields
\`HOLD\` (all values withheld), **not** a stale B marked as qualified.
Repeated reads do not revise the subject or invent an event.

All source text is **synthetic and ephemeral**. Model quality is not
evaluated; this is a deterministic client / source-consumption experiment,
not a second independent real-model study. Earlier actual Gemma4 work is
separately documented in Issues #321/#329.

## Non-deferrable for a real U6 consumer

- Source-bound positive U3 read must remain under one subject/runtime owner
  generation; a caller-supplied checkpoint alone is not qualified state.
- Bind the private read to the established serialized runtime boundary,
  with clearly owned view lifetime and bounded disclosure. The prototype
  relies on sequential test execution: **it does not prove a concurrent
  no-mixed-generation read**.
- A later prompt may describe a **provisional user-reported correction**,
  never an external person's objectively verified intent or its own verified
  capability. Same-turn style/stance affects expression only per A17.
- No raw R15 read model is public; public display or R16 Decision use requires
  a separate exact Human source/consumer gate and negative tests.
- R15 Relationship and SelfModel are still UNKNOWN-only in the production
  owner. Their eventual positive update needs independent admissible
  Experience/task and participant provenance; neither is fabricated here.

## Reproduce

\`\`\`bash
python -m pytest -q tests/test_r15_u6_private_preflight.py
ruff check experiments/r15_p1/private_consumer_preflight.py tests/test_r15_u6_private_preflight.py
mypy --follow-imports=silent --ignore-missing-imports experiments/r15_p1/private_consumer_preflight.py
\`\`\`

**Status:** TEST-ONLY preflight; no Ready/Merge or actual U6 code authority.
