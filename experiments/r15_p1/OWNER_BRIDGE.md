# R15 P1b — owner-backed source bridge and U1 state replay (TEST ONLY)

Parent: https://github.com/upiscium/PROJECT-SUZKA/issues/326
MVI: https://github.com/upiscium/PROJECT-SUZKA/issues/323

This is NOT an approved NarrativeSelf positive admission mechanism.
It is an isolated engineering proof that existing R07/R12/DB1 owner interfaces
and R15 U1 data formats can encode and recover an A-to-B reinterpretation.

This deliberately lives in the P1 experimental tree, not suzka/r15/.
It does not change or access real installed Source, service, original PR #285,
AgentRuntime state, persistent AgentState v10 or StateWAL.

## Responsibilities

- Owns: only evaluator-side verification of captured synthetic P1 stages against
  genuine per-case R07 terminal events, R12 Experience and DB1 episode data;
  reconstruction of finite source-scoped option meanings and R15 U1 JSON roots.
- May: on disposable Source, call EventJournal.inspect(),
  ExperienceStore.load_committed_current() and
  DualMemorySystem.get_committed_episodic(); derive bounded A/B/UNKNOWN meaning
  keys from an original setup episode; validate and replay simulated revisions.
- Must not: issue real SourceSetVerified, create/admit a production R15 claim,
  trust arbitrary model/operator-reported source values, claim verified human
  intent, infer another person's identity or verified competence, or mutate
  a real AgentRuntime/Source/Chroma installation.
- Depends on: the #322 P1a sealed case, the real saved pre-E2 S1 checkpoint,
  an operator-provided disposable Suzka app, and existing owners' exact data.
- Used by: P1b MVI design reviews only, not deployed runtime consumers.

## What each task proves and what remains

| Task | Demonstrated test-only behavior | Production gap |
|---|---|---|
| Meaning | Restores A/B from a source-scoped canonical U1 statement ID; requires original bounded option mapping | A subject-owned qualified proposal and source mapping |
| Revision | U1 revision 1 stores S1 and revision 2 stores S2; S2 links to S1 and retains counterevidence | An actual serialized NarrativeSelfSystem state transition governed by D9 |
| Continuity | Both U1 roots roundtrip; fresh parser can project the current referent after restart | Actual AgentState v10 / StateWAL and runtime consumer integration |

A later USER E2 statement saying B is evidence of what the user said, not
independent proof of objective human intention. The test adapter uses
genuine upstream SUZKA Owner interfaces, but its positive S1/S2 state
objects and RevisionProofs are NOT subject-authorized transitions.

## Source validation and limitations

The adapter verifies R07 Journal chain inspection and matching lifecycle
ACCEPTED / TRANSACTION_COMPLETED / COMPLETED for the exact event ID and
processing sequence; active and committed R12 Experience with matching
source event/sequence/episode/context; and exact original DB1 committed
user input / visible answer. It also validates sealed P0/P1 chronology and
the original pre-E2 checkpoint.

R07 Journal.inspect is owner-provided, but ExperienceStore.load_committed_current
may perform normal recovery/compaction. Creating DualMemorySystem may
perform startup scrub. These are NOT certified physically read-only paths.
Use only disposable/copied Source, NEVER a real user production Source.

R12 and DB1 store the context ID, but the full event-time R09 participant
association and independently verified real-world identity are NOT proven.
Current R15 U1 SourceWitness is a DECLARATIVE tuple; the synthetic test root
digest does not confer SourceEpoch/custody authority. The prototype maps
two existing Chat event IDs into simulated RevisionProof records for
representation/replay: those Chat events were not independently committed
NarrativeSelf mutation events. The real owner needs separate positive subject
admission under a new AgentRuntime event and revalidated source.

## Recommended narrow next production contract

1. Do not change existing U1 or v10 merely because the representation is
   unclear: for a finite original source-owned A/B/UNKNOWN option set, the
   current statement IDs/episode links preserve distinguishable meaning.
   If the true source cannot restore the mapping, require a separately
   reviewed bounded U1 schema and complete 128MiB/16MiB reserve proof.
2. Build a narrow source-enrolled producer and ONE serialized U3 admission.
   Validate actual R07/R12 event and event-time R09 Context; preserve S1
   when E2 arrives and refuse forged/missing/uncommitted/duplicated sources.
   No direct operator/model/Web assertion can set self/relationship/competence.
3. Only then integrate U5/U6 using normal AgentState v10/StateWAL lifecycle.
   No independent authoritative side archive or complex generalized verifier.
4. Full G2 SourceEpoch/historical tamper attestation remains Post-R26 #320.

## Reproducibility and negative tests

tests/test_r15_p1_owner_bridge.py uses disposable FastAPI TestClient with
a fixed deterministic synthetic generation provider, but authentic Journal,
Experience and DB1 owner operations. It checks S1 saved before E2 delivery,
S2 revision and S1 provenance retention, restart read of current B,
no-E2 control without a second revision, source-scoped decoding, JSON
tamper/nonoverwrite, removed DB1 E refusal before E2 and removed R12 E2
refusal before simulated S2 admission.

The P1a real Gemma4 trial and original observations are held separately on
Adam under ~/Documents/Research/PROJECT-SUZKA-Results/r15-p1-20261017/.
The test-only bridge has also reconstructed a 1,935-byte S1 root and a
4,353-byte S2 root from a disposable copy of that actual synthetic Source.
These results do NOT prove actual R15 subject-state mutation or legitimate
capacity to judge the intentions of another person.

Focused test commands from this branch:

    python -m pytest -q tests/test_r15_p1_owner_bridge.py
    python -m pytest -q tests/test_r15_p1_owner_bridge.py tests/test_r15_p1_protocol.py tests/test_r15_p0_pilot.py tests/test_r15_p0_evidence_gate.py tests/test_r15_u3_narrative_self_system.py

Independent review and exact-head CI are separate requirements. This
experiment alone never authorizes Ready/Merge or production writes.
