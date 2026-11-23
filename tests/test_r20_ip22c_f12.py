"""IP-22c, F12-r: a TRUE rollback on a realistic S0-profile root (D1 sec. 6,
risk 13).

The feasibility spike came first and PASSED: on a root that holds the whole
chain (Decision, forward Intent, forecast, Policy, challenge, CLAIM, executor
effect, Outcome, Feedback, Feedback Wake, cycle Decision, Outbox message), with
StateWAL generations rotated, the merged true-rollback recipe is NOT refused:
startup completes on a NEW SubjectHistory incarnation.  So F12 is reported with
this end-to-end test as well as the merged unit tests.

Scope limit (stated, not hidden): the root is the S0 CONFIG (profile S0: R18 +
R19 on, plan-hold receipts on, driver activated, self-goal not activated) with
the scripted provider and the R15 admission NOT performed (interpretation
NOT_REVIEWED); the R15 S1 / S2 part and the SIGKILL tier belong to IP-22d.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import pytest

from experiments.r20.factory import create_subject
from experiments.r20.provider import ScriptedProvider
from experiments.r20.tier1 import T1Runtime
from r20_chain import (
    FAST_R19,
    adopted,
    chat,
    goal_id,
    incarnation,
    operations,
    run_full_chain,
    settled,
    true_rollback,
    until,
    wait_interest,
    X_TEXT,
)
from suzka.r16.history_records import HistoryOperationKind as K
from suzka.r16.successor_policy import SuccessorRow
from suzka.r18.contracts import ActionIntentV1, ExecutionClaimV1
from suzka.r18.target import TargetStore

_ACTIVE_BRANCH_KINDS = (
    K.R18_INTENT_ISSUE,
    K.R18_EXECUTION_CLAIM,
    K.R18_OUTCOME_COMMIT,
    K.R19_WAKE_CLAIM,
    K.R19_OUTBOX_EMIT,
)


def _rows(state: Any, inc: str) -> list[tuple[str, str]]:
    return [
        (row.operation_kind.value, row.operation_digest)
        for row in state.subject_history_store.operations(inc)
    ]


def _status(state: Any, intent: ActionIntentV1) -> Any:
    store = TargetStore.open(Path(state.settings.r18.target_path))
    try:
        return store.status(intent.obligation_group_key)
    finally:
        store.close()


def test_f12_r_a_true_rollback_on_a_realistic_s0_root_leaves_no_actionable_old_branch(
    r20_base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = create_subject("S0", base=r20_base, r19_overrides=FAST_R19)
    runtime = T1Runtime(subject, ScriptedProvider(), executor=True, scheduler=True)
    runtime.start()
    try:
        ids = run_full_chain(runtime, subject)
        intent = ids["intent"]
        state = runtime.state
        # the chain is complete: every actionable row of the old branch exists
        for kind in _ACTIVE_BRANCH_KINDS:
            assert operations(state, kind), kind
        until(lambda: operations(state, K.R17_FEEDBACK_COMMIT))
        time.sleep(1.0)  # let the last event settle before the process stops
        old_incarnation = incarnation(state)
        old_rows = _rows(state, old_incarnation)
        old_effect = _status(state, intent)
        assert old_effect.forward is not None and old_effect.artifact is not None
        settings = runtime.subject.resolved_settings()
    finally:
        runtime.stop()

    corrupted = true_rollback(settings, subject.root)
    assert corrupted  # a real transition record was corrupted

    writes: list[int] = []
    real_write = TargetStore.forward_write

    def counting(store: Any, *args: Any, **kwargs: Any) -> Any:
        writes.append(1)
        return real_write(store, *args, **kwargs)

    monkeypatch.setattr(TargetStore, "forward_write", counting)

    runtime.start()  # NOT refused: fail-closed did not trigger on a realistic root
    try:
        state = runtime.state
        new_incarnation = incarnation(state)
        # the incarnation changed only because of the adopted true rollback
        assert new_incarnation != old_incarnation
        # the old branch is still readable as audit, byte for byte
        assert _rows(state, old_incarnation) == old_rows
        # the scheduler and the executor run on the new incarnation: give them
        # several ticks, then nothing of the old branch may be actionable
        time.sleep(6.0)
        for kind in _ACTIVE_BRANCH_KINDS:
            assert operations(state, kind) == [], kind  # nothing re-issued / re-claimed
        assert writes == []  # no second target effect
        # the real target effect survives, once, reconciled (not re-executed)
        survivor = _status(state, intent)
        assert survivor.artifact == old_effect.artifact
        assert survivor.forward is not None
        assert survivor.forward.approval_ref == old_effect.forward.approval_ref
        # the same request again: the next Decision on the new incarnation is
        # row 2c (Plan A, no WRITE) because the effect is from a prior incarnation
        client = runtime.client
        assert client is not None
        assert client.post("/api/r16/driver/activate").status_code in (200, 409)
        again = chat(client, X_TEXT)
        wait_interest(state, again["experience_id"])
        review = chat(client, f"REVIEW_EXPERIENCE:{again['experience_id']}")
        assert settled(client, review["command_ticket"]["ticket_id"])["status"] == "adopted"
        commit = chat(client, f"COMMIT_REVIEW_EXPERIENCE:1:{again['experience_id']}")
        assert settled(client, commit["command_ticket"]["ticket_id"])["status"] == "accepted"
        until(lambda: len(operations(state, K.R16_DECISION_ISSUE)) >= 2)
        time.sleep(2.0)
        preview = state._plan_preview_owner._submit_preview(goal_id(state)).result(timeout=30).value
        assert preview.row is SuccessorRow.ROW_2C
        decisions = operations(state, K.R16_DECISION_ISSUE)
        assert any(b"PRIOR_INCARNATION_EFFECT_PRESENT" in row.pending_bytes for row in decisions)
        for kind in _ACTIVE_BRANCH_KINDS:
            assert operations(state, kind) == [], kind
        assert adopted(state, ExecutionClaimV1, strict=True) == []
        assert writes == []
        assert _rows(state, old_incarnation) == old_rows
    finally:
        runtime.stop()
