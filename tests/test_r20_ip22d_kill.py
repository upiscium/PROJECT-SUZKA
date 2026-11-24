"""IP-22d: the SIGKILL tier K1..K6 (IP-22 D1 r3 sec. 4-3; F06-c extended).

T2 results are SMOKE and are never counted as covered: the provider is
scripted (deterministic) but a poll decides when the process group is killed, so
the crash window is not controlled.  What each T2 test asserts is the set of
invariants after the restart; whether the kill landed in the named window is
RECORDED (``window_confirmed``: the predicate still holds on the frozen store).
The exact windows are the merged T1 tests (crash matrix, IP-17c claim, IP-19,
IP-20d).  K1 has no T2 form: ``confirm`` writes the challenge and the claim in
one request.  Its T1 form here is the in-process restart between the challenge
and the claim, through the REAL approval route.

After any restart no R16 command to X can be sent (``SOURCE_NOT_OBSERVED``):
every kill point comes after the U2c acceptance and the tests send none.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os

import pytest

from experiments.r20 import hooks
from experiments.r20.approval import ApprovalOperator, recompute
from experiments.r20.client import AsgiTransport, HttpClient
from experiments.r20.factory import create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.kill import KILL_TABLE, T2_POINTS, run_kill_point
from experiments.r20.manifest import Manifest
from experiments.r20.provider import ScriptedProvider
from experiments.r20.s0 import FINALIZED, KillPoint
from experiments.r20.tier1 import T1Runtime
from r20_chain import adopted, drive_to_allowed
from suzka.r18.contracts import ExecutionClaimV1


def test_the_kill_table_names_k1_to_k6_and_k1_is_t1_only() -> None:
    assert tuple(KILL_TABLE) == hooks.KILL_POINTS == ("K1", "K2", "K3", "K4", "K5", "K6")
    assert {point for point, row in KILL_TABLE.items() if row["tier"] == "T1"} == hooks.T1_ONLY_KILL_POINTS == {"K1"}
    assert T2_POINTS == ("K2", "K3", "K4", "K5", "K6")
    assert KILL_TABLE["K1"]["second_approval"] is True
    assert all(not KILL_TABLE[point]["second_approval"] for point in T2_POINTS)
    assert all(row["state"] and row["expect"] for row in KILL_TABLE.values())


def test_k1_has_no_t2_form(tmp_path: Path) -> None:
    with pytest.raises(GuardRefusal) as refused:
        run_kill_point(tmp_path, "K1")
    assert refused.value.code == "NO_T2_FORM"
    with pytest.raises(ValueError):
        KillPoint.holds("K1", {})


def _counts(**finalized: int) -> Mapping[tuple[str, str], int]:
    return {(kind, FINALIZED): n for kind, n in finalized.items() if n}


def test_each_kill_predicate_holds_in_exactly_its_own_window_of_the_chain() -> None:
    """Pure: along the chain the stage rows appear in order; the predicates then
    hold one window at a time, so a poll cannot name the wrong point."""

    chain = [
        ("R18_EXECUTION_CLAIM",), ("R18_OUTCOME_COMMIT",), ("R17_FEEDBACK_COMMIT",),
        ("R19_WAKE_CLAIM",), ("R16_CYCLE_DECISION",), ("R19_OUTBOX_EMIT",),
    ]
    progress: dict[str, int] = {}
    seen: dict[str, list[str]] = {}
    for stage in [(), *chain]:
        for kind in stage:
            progress[kind] = 1
        seen[",".join(progress) or "start"] = [p for p in T2_POINTS if KillPoint.holds(p, _counts(**progress))]
    windows = list(seen.values())
    assert windows == [[], ["K2"], ["K3"], ["K4"], ["K5"], ["K6"], []]


@pytest.mark.parametrize("point", T2_POINTS)
def test_f06_c_a_real_sigkill_at_each_t2_point_keeps_the_invariants_after_the_restart(
    r20_base: Path, point: str
) -> None:
    result, session = run_kill_point(r20_base, point)
    try:
        # the kill really happened inside the named window (as far as the store shows)
        assert result.window_hit, f"{point}: the chain went past the window on every attempt"
        assert result.window_confirmed, f"{point}: the frozen store no longer shows the named state"
        assert result.pids[0] != result.pids[1] != 0
        failed = [name for name, held in result.checks.items() if not held]
        assert not failed, (point, failed, result.detail)
        if point != "K2":
            assert result.finished and result.effects == 1
            assert result.detail["ablation_pair"] == ["5", "6"]
        manifest = Manifest(session.subject.evidence_dir, session.subject.run_id, subject_root=session.subject.root)
        manifest.append("kill_points", result.manifest_entry())
        manifest.result(f"F06-c.{point}", "SMOKE_T2_NOT_COUNTED")
        manifest.write()
    finally:
        session.stop()
    # the killed child is gone with its whole process group
    with pytest.raises(ProcessLookupError):
        os.killpg(result.pids[0], 0)


def test_k1_t1_a_failure_between_the_challenge_write_and_the_claim_voids_it_and_needs_a_new_approval(
    r20_base: Path,
) -> None:
    """K1 (T1 only): ``confirm`` writes the challenge and the claim in one request, so
    the window cannot be caught by a poll (and a restart between ``review`` and
    ``confirm`` loses only an in-memory challenge: nothing was written).  The window is
    reached by the existing fault-injection seam (``owner._validate_commit``, as
    ``test_a_crash_before_the_claim_wal_is_tombstoned_and_a_new_boot_needs_a_new_attempt``
    does) behind the REAL HTTP route and the scripted operator: the challenge IS
    written, the claim fails, the process goes away.  The next boot voids the challenge
    (BOOT_CHANGED), which counts as attempt 1: the new review + confirm is attempt 2,
    a NEW challenge, and exactly one claim exists.  (A second Human approval in
    ``HUMAN`` mode.)"""

    from suzka.runtime.agent_runtime import AgentEventType

    subject = create_subject("S0", base=r20_base)
    runtime = T1Runtime(subject, ScriptedProvider(), executor=False, scheduler=False)
    runtime.start()
    try:
        client, state = runtime.client, runtime.state
        assert client is not None
        ids = drive_to_allowed(client, state)
        intent_id = ids["intent"].intent_id
        operator = ApprovalOperator(HttpClient(AsgiTransport(client), subject.port), subject.port)
        shown = operator.review(intent_id)
        assert shown.status == 200
        first = recompute(shown.json())
        assert first.ok

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R18_EXECUTION_CLAIM:
                raise RuntimeError("simulated pre-WAL failure")

        state.approval_owner._validate_commit = crash
        held = operator.confirm(intent_id, first.nonce, first.code)
        assert held.status != 200 and held.json()["status"] == "hold"  # the claim failed
        assert adopted(state, ExecutionClaimV1) == []
        # the challenge WAS written: it is a Journal event of its own, before the failed claim
        challenges = [
            r for r in state.event_journal.inspect().records
            if r.event_type is AgentEventType.R18_CHALLENGE_ISSUE and r.lifecycle.value == "completed"
        ]
        assert len(challenges) == 1
    finally:
        runtime.stop()
    runtime.restart()
    try:
        client, state = runtime.client, runtime.state
        assert client is not None
        operator = ApprovalOperator(HttpClient(AsgiTransport(client), subject.port), subject.port)
        late = operator.confirm(intent_id, first.nonce, first.code)
        assert late.status == 409 and late.json()["status"] == "hold"  # BOOT_CHANGED: nothing claimed
        assert adopted(state, ExecutionClaimV1) == []
        done = operator.approve(intent_id)
        assert done.status == 200 and done.json()["status"] == "claimed"
        assert done.json()["attempt"] == 2  # the voided challenge counted as attempt 1
        assert len(adopted(state, ExecutionClaimV1, strict=True)) == 1
    finally:
        runtime.stop()
