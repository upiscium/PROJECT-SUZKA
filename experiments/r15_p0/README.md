# R15 P0 — objective reclassification pilot

TEST ONLY. This is an evaluator-side benchmark, not an R15 semantic producer, NarrativeSelf admission, SelfModel mutation, or evidence that Suzka has learned.

Design: https://github.com/upiscium/PROJECT-SUZKA/issues/288#issuecomment-6104775710

## Scope and truth boundary

The evaluator preselects a hidden A/B target in a synthetic scenario. It observes which alternative the assistant's *visible* answer uses, if any. In a separate later chat it reveals that benchmark target, then asks about the original observable response and the later target.

This tests externally observable answer reclassification. It does NOT prove any original internal Belief, original subjective understanding, real-human intent, or later R15 narrative revision. Asking for clarification when the initial reference is ambiguous is legitimate. A forced-provisional-choice lane is a mechanism probe, not advice to discourage clarification.

The existing Suzka public POST /api/chat returns episode_id and response, but neither a typed chosen referent nor verified R07/R12/R09 joined proof. Its client_session_id changes R09 context selection but does not isolate process-wide SessionState and working-memory effects. Each trial MUST therefore use a fresh isolated app/process, separate ephemeral DB1, Journal/WAL and AgentState directories. The runner requires --isolated-instance but cannot verify this assertion.

## Files and deterministic tests

- pilot.py: synthetic case table, manifest precommit, chat HTTP capture, strict run-record validation and manually annotated scoring.
- tests/test_r15_p0_pilot.py: 4 core classes plus adversarial and failure tests. All fixtures are synthetic, not real Suzka semantic output.

From the repository root:

    python3 -m unittest discover -s tests -p test_r15_p0_pilot.py -q
    python3 -m compileall -q experiments/r15_p0
    python3 -m experiments.r15_p0.pilot cases

The pilot case table contains 8 normal cases (4 natural, 4 forced; equal hidden A/B targets) and 4 controls (2 explicit, 2 no-reveal). These are exploratory counts, not a statistically powered evaluation.

## One isolated live trial

First create a manifest BEFORE any interaction. Keep it outside all prompts, because it contains the hidden target:

    python3 -m experiments.r15_p0.pilot seal p0-00 /tmp/p0-00.manifest.json

In a FRESH instance of Suzka listening on 127.0.0.1:8000, run:

    python3 -m experiments.r15_p0.pilot run \
      /tmp/p0-00.manifest.json /tmp/p0-00.capture.json \
      --base-url http://127.0.0.1:8000 --isolated-instance

The CLI restricts HTTP to loopback with an explicit port and refuses redirects. It sends setup, ambiguous question E, later reveal E2 (omitted in no-reveal controls), and post-reveal probe. The target is not explicitly designated until the E2 message.

Capture files store synthetic prompts and visible responses only, with restrictive permissions. These are experimental data, NOT an approved second archive of real user text. Do not run against production conversations.

A capture.started.json sidecar is written before the first HTTP call. If the run fails after any mutation, do NOT reuse or retry that case in the same instance: use a fresh test instance and a new case/run path. The manifest checksum and file creation order guard accidental changes only; they are NOT hostile-operator cryptographic custody.

## Independent scoring

First export only the materials observed BEFORE the reveal, hiding the case identifier and target from the first-pass observer:

    python3 -m experiments.r15_p0.pilot blind       /tmp/p0-00.manifest.json /tmp/p0-00.capture.json       /tmp/p0-00.blind.json

The blinded packet contains setup, question E, and the first visible answer only. The later reveal and response are withheld. Explicit-reference controls cannot be blinded and are therefore refused by this command. The case manifest itself must never be sent to the first-pass observer.

An observer who has NOT seen the target must annotate the first visible answer as A, B, CLARIFY, OTHER or UNKNOWN. Both letters may occur in the text without an actual selection: NEVER use simple substring matching as proof of an A/B selection.

A later observer classifies the post-reveal answer's three components. If earlier E availability in the model prompt is not independently observed, mark retrieval UNKNOWN.

Example annotation for a hypothetical p0-00 case (not actual model output):

    {
      "case_id": "p0-00",
      "choice_at_E": "CLARIFY",
      "probe_prior": "CLARIFY",
      "probe_target": "A",
      "probe_relation": "NO_CHOICE",
      "retrieval": "UNKNOWN",
      "annotator": "blinded-observer-1"
    }

Then:

    python3 -m experiments.r15_p0.pilot score \
      /tmp/p0-00.manifest.json /tmp/p0-00.capture.json \
      /tmp/p0-00.annotation.json

Score results have authority_scope P0_BENCHMARK_ONLY_NOT_R15. They are NOT native R15 roots. Report all trials, including failed, clarification, no-retrieval, source-invalid and UNKNOWN. Distinguish wrong past-choice, wrong revealed-target, false confident mismatch and evidence-unavailable cases; never select only successful responses.

## Acceptance limitations

- P0-HARNESS PASS: fixture and negative tests pass. This validates the evaluator only.
- P0-LIVE PILOT: 8 real provider interactions and 4 controls on separately initialized test instances; score visible answers and probes with blind annotation. A case with no definite A/B selection cannot evidence correction of a mistaken selection. Absence of genuine mismatches yields INCONCLUSIVE coverage, not manufactured PASS.
- P1 NarrativeSelf: a genuinely admitted first-person interpretation BEFORE the contrary experience, qualified independent E2 evidence and an R15-owned semantic revision of the same prior episode. P0 factual relabeling or model text saying "I learned" cannot satisfy P1.

No R07-R15 production code, U1 semantic contract, source authentication, runtime persistence, model, DB1 source authority or private-data retention policy is modified by this pilot. G1-G4 and #288 remain HOLD.
