# R15 U6b — serialized private read/use feasibility (TEST ONLY)

Parent: Issue #336, using the completed U6a read prototype in Draft PR #335.
MVI target: one later ordinary Chat turn may use the subject's earlier,
committed, source-requalified S1/S2 without creating a new Subject review.

## Scope

**Owns:** a test-only hook around the existing SuzkaMainLoop._build_prompt
call during a real serialized API_CHAT worker. It invokes the private
U6a source-requalified WAL read before ordinary model generation and
records only the bounded structured result in test memory.

**May:** check that AgentRuntime.current_event() is the actual Chat event,
then read U3 / StateWAL / R07 / R12 / DB1 / R09 through the preexisting
test-only preflight reader. A committed Narrative A/B candidate is
QUALIFIED; WAL unavailable or inconsistent produces HOLD.

**Must not:** modify any production prompt builder, invoke new
NARRATIVE_REVIEW, commit another R15 Revision, publish source IDs, user
text, a human identity, trust, trait, competence, model self-endorsement,
or external Action. No public output behavior is claimed from the test.
No new SourceEpoch/global historical attestation is attempted.

**Depends on:** the exact earlier source-bound U3 owner, v10 StateWAL,
and the private U6a read fixture; operates in a fresh synthetic TestClient.

**Used by:** the separate Human U6b implementation gate in #336,
which must decide how a current QUALIFIED B (or HOLD) can safely affect
the next private response/structured expression without creating a
new authority or leaking raw user/model text.

## Verified boundaries

- Under a real existing serialized Chat event, the U6 private reader can
  obtain the original and current source-scoped candidate and compare it
  with the committed WAL state. No new R07 Narrative review occurs.
- With E2, later current B and earlier A remain provisional. With no E2,
  the later Chat still privately reads A. Neither is verified human intent.
- If the authoritative WAL view is missing, the same Chat path receives
  HOLD and cannot use an invented current referent.
- This test does not modify any model prompt, so it is NOT yet proof that
  Suzka's public response behavior is influenced by prior cognition.
- The scoped runtime event and cooperative source reads are the intended
  MVI concurrency/lifetime boundary. Additional adversarial concurrent
  WAL mutation, mid-request source replacement and same-turn output privacy
  tests remain production acceptance tasks, not accomplished by this demo.

## Production requirements

Issue #336 requires a separate Human code approval for the exact
private_projection module, existing R15StatePort read-only delegation and
one MainLoop use point. A valid view must come from the single serialized
runtime's real source/StateWAL, never from caller-provided JSON. If the
view is HOLD, omit any source-dependent expression. Positive bounded
expression may say it *tentatively interpreted* B after a reported user
correction, not that the real speaker's actual intention was B.

Keep all R07/R09/R12/StateWAL storage implementations unchanged unless
new concrete tests reveal a necessary additional owner contract. Do not
infer Relationship or SelfModel positive state, R16 Decision completeness,
or whole R15 stage PASS from this experiment.

## Verification

Run tests/test_r15_u6_serialized_preflight.py against fresh disposable
SUZKA Source plus the existing U6a tests, Ruff and scoped MyPy.
