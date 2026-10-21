# R15 U4b — source-qualified attempted-task SelfModel admission (MVI)

Parent: Issue #338 and #334; merged baseline `rebuild/develop@9084333e9c0c6290b68f68e712a7aad523170153` (PR #342). The Human approved **implementation**, not Merge, installed Source or deployment.

## Owns and use

The sole process-local `SelfModelSystem` may now admit **one** first-person *observed attempt* for the closed task `task:referent-resolution`, provided the same runtime already holds an actual v10/WAL-committed U3 NarrativeSelf interpretation. The stored `CapabilityHypothesis` has `attempted=PRESENT`; the associated estimate/confidence quantify only the **observable occurrence of a completed first-person attempt**, **never task accuracy, human intent or verified competence**. `VerifiedCompetence` remains exactly `UNKNOWN` until independently verified R17/R18 outcomes. No person identity, role, personality trait or generic skill setter is implemented.

The new `SELF_ATTEMPT_REVIEW / runtime.self_attempt_review` event uses the one already configured serialized `AgentRuntime`. Its handler is an internal private `R15StatePort._review_source_qualified_self_attempt()`, **not** an HTTP or model-exposed endpoint. No source reference, task name, candidate root, typed JSON or EventRef is accepted from the caller. One startup-only `QualifiedSelfAttemptGate` is bound to the existing port/NarrativeSelf source reader, current `AgentStateStore` and Runtime.

## Qualification and ordinary transaction

The Gate requires the **actual current internal Actor worker event** and exact bound owner identities. It loads the committed v10 state, matches process-local U3 NarrativeSelf root+continuation and current U4 root to the committed snapshot, and reopens the finite original R07 terminal CHAT / R12 active Experience / DB1 visible response with anonymous consistent R09 Context. It reuses the existing private U6 source-qualified semantic reconstruction, supporting S1-only A and S1=A→S2=B after a later user report. User-reported E2 never constitutes ground truth.

A source-qualified original E is admitted once with a genuine `RevisionProof` tied to the current internal review event and its original Experience witness. The existing `AgentStateStore.capture` and `StateRecoveryCoordinator.commit_internal_candidate` publish one new v10/WAL root on the normal Actor event path. A repeated review requalifies evidence and returns **no-change** rather than duplicating a capability or increasing success confidence; exact root survives restart. A generic operator/model assertion alone cannot reach this path.

A restored **legal U1** capability of another task type or with an inconsistent U4 proof is **not** treated as an admitted attempt by `SelfTaskView`; it returns UNKNOWN unless the persisted root matches this narrow producer's canonical task/proof shape. This is structural verification, not a current Source attestation: when an original E is retired **after a committed U4 admission**, the historical attempt remains in the root while any subsequent attempted review refuses without requalifying the retired E. No verified ability or fresh evidence claim is made by the read-only view. Future consumers that need *currently usable* Source evidence must requalify it at their own event/use boundary.

When the original E or E2 is missing/retired, the Actor/source binding is absent, the existing U3 root differs from committed v10, Context becomes participant-ambiguous, or the event type/source is wrong, the review is **refused** before U4 admission. A handler failure after an in-process revision uses the existing terminal R07 failure callback and full v10 restore. An uncertain post-WAL failure remains governed by the existing fail-stop/recovery mechanism; this change does not create a new recovery owner or suggest success is known.

## Non-goals and limits

- No ability score, objectively verified task outcome, durable limitation/trait/identity, relationship trust, new R16 Decision/Goal/Tool authority, general task taxonomy or autobiographical free text.
- No changes to `AgentStateSnapshotV10`, StateWAL writer/retention/schema, R07/R09/R12/DB1, public API, production configuration or installed Source. Existing full 128 MiB AgentState capacity and 16 MiB reserve are not reallocated.
- Existing cooperative Source readers may perform benign maintenance; full cross-owner historical SourceEpoch/nonrepairing attestation is Post-R26 #320.
- Strong unchanged-A/B cross-model WAL-generation coherence was explicitly deferred to Post-R26 #343 and is not falsely claimed here.
- A `SelfTaskView` is a current *committed-root projection*, not a fresh historical Source verification when read. It never upgrades `VerifiedCompetence` from UNKNOWN.
- The event is internally invocable by already trusted runtime code and was exercised using the real isolated Actor/WAL pipeline. No automatic user-triggered skill inference is installed.

## Verification and approval boundary

The focused acceptance suite uses real isolated terminal R07/R12/DB1/U3/U4/StateWAL code with a deterministic synthetic provider, proving one genuine revision, no-E2 attempt, restart and duplicate idempotence, missing evidence refusal, unauthorized event refusal and handler failure rollback. Source text/identity/verified competence are not copied into a public response. Additional full source Ruff/MyPy and integration regressions are required before Merge.

**Status:** implementation candidate. Independent exact-source review, exact-head CI and separate Human Merge approval are mandatory. If the current trusted-source producer cannot be used, U4 stays UNKNOWN; do not infer positive capability from syntax or test fixtures.
