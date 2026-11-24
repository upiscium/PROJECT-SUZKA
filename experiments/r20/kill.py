"""The SIGKILL process tier K1..K6 (IP-22 D1 r3 sec. 4-3), T2 smoke.

"Provider deterministic, kill window non-deterministic": a poll decides WHEN to
kill the child's process group, so the exact crash window is not controlled.
What is asserted is the set of INVARIANTS after the restart; whether the kill
landed inside the named window is RECORDED (``window_confirmed``: the predicate
still holds on the frozen store after the child died), not assumed.  Only T1
results count as covered; the exact windows are the merged T1 tests, and K1 has
no T2 form at all (``confirm`` writes the challenge and the claim in one request).

Two meanings of "incarnation" are kept apart: a PROCESS LIFE ends at every
restart (and is what the driver's observation map is bound to), the
SubjectHistory INCARNATION id changes only after an adopted true rollback.  After
a restart NO command to X can be sent (``SOURCE_NOT_OBSERVED``), so every kill
point comes after the U2c acceptance and the run continues on the driver's and
R19's per-start scans only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any

from experiments.r20.ablation import DefectClass, ablate_feedback, replay_defect
from experiments.r20.guards import GuardRefusal
from experiments.r20.polling import poll_until
from experiments.r20.s0 import FINALIZED, KINDS, Evidence, KillPoint, S0Session
from suzka.r16.successor_policy import FeedbackState, SuccessorPolicyInputV2, SuccessorRow

#: The expected-result table of D1 sec. 4-3 (all after U2c acceptance).
#: ``tier``: the only tier in which the point can be hit exactly.
KILL_TABLE: dict[str, dict[str, Any]] = {
    "K1": {
        "state": "CHALLENGE written, no claim",
        "tier": "T1",
        "expect": (
            "the open challenge is voided by BOOT_CHANGED and counts as attempt 1; the new review + confirm "
            "is attempt 2 (one of MAX_CHALLENGES_PER_INTENT = 3); a NEW challenge is needed"
        ),
        "second_approval": True,
    },
    "K2": {
        "state": "EXECUTION_CLAIM row visible, no Outcome row",
        "tier": "T2",
        "expect": (
            "one target effect or none, never two; the startup sweep reads the target back "
            "(Outcome VERIFIED, or absent)"
        ),
        "second_approval": False,
    },
    "K3": {
        "state": "Outcome visible, no Feedback",
        "tier": "T2",
        "expect": "the Feedback scan completes exactly one Assessment and one Feedback",
        "second_approval": False,
    },
    "K4": {
        "state": "Feedback visible, no Wake claim",
        "tier": "T2",
        "expect": "R19's startup scan materializes the FEEDBACK_ADMITTED Wake once",
        "second_approval": False,
    },
    "K5": {
        "state": "Wake claim visible, no cycle Decision",
        "tier": "T2",
        "expect": "the claim lease / attempt rules give one Decision and one resolution",
        "second_approval": False,
    },
    "K6": {
        "state": "cycle Decision visible, no Outbox message",
        "tier": "T2",
        "expect": "one Outbox message per closed cycle (dedupe)",
        "second_approval": False,
    },
}
T2_POINTS = tuple(point for point, row in KILL_TABLE.items() if row["tier"] == "T2")
# Counters that name ONE logical result each: never more than one, kill or no kill.
SINGLE = (
    "R18_EXECUTION_CLAIM", "R18_OUTCOME_COMMIT", "R17_ASSESSMENT_COMMIT", "R17_FEEDBACK_COMMIT",
    "R16_CYCLE_DECISION", "R19_OUTBOX_EMIT",
)
POLL_INTERVAL_S = 0.005
KILL_WAIT_S = 90.0
CONVERGE_S = 150.0


@dataclass
class KillResult:
    point: str
    tier: str = "T2"
    window_hit: bool = False
    window_confirmed: bool = False  # the predicate still holds on the frozen store
    at_kill: dict[str, int] = field(default_factory=dict)
    frozen: dict[str, int] = field(default_factory=dict)
    after: dict[str, int] = field(default_factory=dict)
    effects: int = 0
    outcome_states: list[str] = field(default_factory=list)
    finished: bool = False
    serving_after_restart_s: float | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    pids: tuple[int, int] = (0, 0)

    @property
    def ok(self) -> bool:
        return self.window_hit and self.window_confirmed and all(self.checks.values())

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "tier": self.tier,
            "window_hit": self.window_hit,
            "window_confirmed": self.window_confirmed,
            "pids": list(self.pids),
            "at_kill": self.at_kill,
            "after": self.after,
            "effects": self.effects,
            "checks": self.checks,
            "label": "SMOKE_T2_NOT_COUNTED",
        }


def logical_counts(session: S0Session) -> dict[str, int]:
    """What a no-kill run and a killed run must agree on (D1 sec. 7 item 10):
    one logical Decision / Feedback / Wake / Outbox / effect each."""

    counters = session.counters()
    evidence = session.evidence
    return {
        "R16_DECISION_ISSUE": counters["R16_DECISION_ISSUE"],
        "R16_CYCLE_DECISION": counters["R16_CYCLE_DECISION"],
        "R17_FEEDBACK_COMMIT": counters["R17_FEEDBACK_COMMIT"],
        "R19_OUTBOX_EMIT": counters["R19_OUTBOX_EMIT"],
        "wake_feedback_admitted": evidence.wake_causes().count("FEEDBACK_ADMITTED"),
        "verified_outcomes": len(evidence.outcome_states()),
        "target_effects": evidence.target_effects(),
    }


def replay_all(evidence: Evidence) -> list[str]:
    """Every sealed input replays to its stored row; the names of any that do not."""

    bad: list[str] = []
    for row in evidence.decisions():
        if replay_defect(row.sealed_input, SuccessorRow(row.row)) is not None:
            bad.append(row.decision_id)
    return bad


def run_kill_point(base: Path, point: str, *, attempts: int = 3) -> tuple[KillResult, S0Session]:
    """One T2 S0 subject, killed at ``point``, restarted, run on.  A missed
    window (the chain went past it before the poll saw it) is retried on a NEW
    subject, at most ``attempts`` times; the caller gets the last result."""

    if point not in T2_POINTS:
        raise GuardRefusal("NO_T2_FORM", f"{point} is {KILL_TABLE.get(point, {}).get('tier', 'unknown')} only")
    result = KillResult(point)
    for _ in range(attempts):
        session = S0Session(base)
        try:
            result = _once(session, point)
        except BaseException:
            session.stop()
            raise
        # a window that the frozen store no longer shows is a miss too (the poll raced the server)
        if result.window_hit and result.window_confirmed:
            return result, session
        session.stop()
    return result, session


def _once(session: S0Session, point: str) -> KillResult:
    session.start()
    ids = session.drive_to_open_approval()
    intent_id = ids["approval"]["intent_id"]

    def approve() -> None:
        session.approve(intent_id)

    return kill_phase(session, point, approve)


def kill_phase(
    session: S0Session,
    point: str,
    approve: Callable[[], object],
    *,
    kill_wait_s: float = KILL_WAIT_S,
    converge_s: float = CONVERGE_S,
    chat_after_restart: bool = True,
) -> KillResult:
    """From an open approval on: ``approve`` runs in a worker (the scripted
    operator, or the Human's wait), the group is killed when ``point`` first holds, a
    new child serves the same root, and the invariants are judged.  The poll races
    the server (the window is not controlled)."""

    result = KillResult(point)
    first = session.child
    assert first is not None
    evidence = session.evidence

    def approve_quietly() -> None:
        try:
            approve()
        except (OSError, ValueError, GuardRefusal):
            pass  # the child may be killed under the request: that is the point

    worker = threading.Thread(target=approve_quietly, name="r20-approve", daemon=True)
    worker.start()
    hit = poll_until(
        lambda: _snapshot_if(point, evidence),
        timeout_s=kill_wait_s,
        interval_s=POLL_INTERVAL_S,
    )
    session.kill()
    worker.join(timeout=10)
    result.window_hit = bool(hit)
    result.at_kill = _flat(hit) if hit else {}
    result.pids = (first.pid, 0)
    # the child is dead: its stores are quiescent, so this reads the state it left
    frozen = evidence.counts()
    result.frozen = _flat(frozen)
    result.window_confirmed = KillPoint.holds(point, frozen)
    result.detail["effects_frozen"] = evidence.target_effects()
    if not result.window_hit:
        return result

    session.start()
    second = session.child
    assert second is not None and second.pid != first.pid
    result.pids = (first.pid, second.pid)
    result.serving_after_restart_s = session.serving_after_s
    # the restarted subject serves a new turn; it needs no R16 command
    # (a COUNTED run asks /health instead: a chat turn here could land between the Feedback and the Wake review)
    if chat_after_restart:
        turn = session.request("POST", "/api/chat", {"message": "after the restart", "attachments": []})
        result.checks["subject_answers_after_restart"] = turn.status == 200
    else:
        result.checks["subject_answers_after_restart"] = session.request("GET", "/health", timeout_s=30).status == 200

    session.wait_for(lambda: session.finished(), converge_s, 0.5)
    time.sleep(3)  # a duplicate or a stray further row would show now
    _judge(session, result, frozen)
    return result


def _snapshot_if(point: str, evidence: Evidence) -> dict[tuple[str, str], int] | None:
    counts = evidence.counts()
    return counts if KillPoint.holds(point, counts) else None


def _flat(counts: Mapping[tuple[str, str], int] | None) -> dict[str, int]:
    return {f"{kind}:{phase}": n for (kind, phase), n in sorted((counts or {}).items())}


def _judge(session: S0Session, result: KillResult, frozen: Mapping[tuple[str, str], int]) -> None:
    evidence = session.evidence
    counters = session.counters()
    result.after = {kind: counters[kind] for kind in KINDS}
    result.effects = evidence.target_effects()
    result.outcome_states = evidence.outcome_states()
    result.finished = session.finished()
    checks = result.checks
    # no double effect
    checks["at_most_one_target_effect"] = result.effects <= 1 and evidence.target_artifacts() <= 1
    # one logical result per key (a claim, an outcome, an assessment, a Feedback, a cycle Decision, a message)
    checks["one_logical_result_each"] = all(counters[kind] <= 1 for kind in SINGLE)
    checks["at_most_one_feedback_wake"] = evidence.wake_causes().count("FEEDBACK_ADMITTED") <= 1
    checks["at_most_one_cycle_resolution"] = evidence.table_count("logical_cycle_resolutions") <= 1
    checks["decisions_at_most_row3_row4_row5"] = counters["R16_DECISION_ISSUE"] <= 2
    # no loss, no change: every FINALIZED row count seen at the freeze is still there
    checks["nothing_lost"] = all(counters[kind] >= frozen.get((kind, FINALIZED), 0) for kind in KINDS)
    # replay of every sealed input equals its stored row (a difference is an ENGINEERING_DEFECT)
    bad = replay_all(evidence)
    checks["replay_of_every_sealed_input_equal"] = not bad
    if bad:
        result.detail["replay_defect"] = {"class": DefectClass.ENGINEERING_DEFECT.value, "decisions": bad}
    point = result.point
    if point == "K2":
        # one effect or none; with the effect present the Outcome is VERIFIED
        checks["k2_effect_or_none_and_verified"] = result.effects in (0, 1) and (
            result.effects == 0 or "ACTIVE_VERIFIED_AUTHORIZED" in result.outcome_states
        )
    else:
        checks["run_completed"] = result.finished
        checks["one_effect"] = result.effects == 1
    if point == "K3":
        checks["k3_one_assessment_one_feedback"] = counters["R17_ASSESSMENT_COMMIT"] == 1 and counters["R17_FEEDBACK_COMMIT"] == 1
    if point == "K4":
        checks["k4_the_wake_is_materialized_once"] = evidence.wake_causes().count("FEEDBACK_ADMITTED") == 1
    if point in ("K5", "K6"):
        checks["k56_one_decision_one_resolution_one_message"] = (
            counters["R16_CYCLE_DECISION"] == 1
            and evidence.table_count("logical_cycle_resolutions") == 1
            and counters["R19_OUTBOX_EMIT"] == 1
        )
    if result.finished:
        cycle = [row for row in evidence.decisions() if row.operation_kind == "R16_CYCLE_DECISION"]
        if len(cycle) == 1:
            sealed = SuccessorPolicyInputV2.from_bytes(cycle[0].sealed_input)
            result.detail["cycle_decision_row"] = cycle[0].row
            result.detail["cycle_feedback_state"] = sealed.feedback.value
            if sealed.feedback is FeedbackState.ABSENT:
                # nothing to ablate: the cycle was woken without a Feedback
                result.detail["ablation_pair"] = "NOT_APPLICABLE_FEEDBACK_ABSENT"
            else:
                ablation = ablate_feedback(cycle[0].sealed_input)
                result.detail["ablation_pair"] = [row.value for row in ablation.pair]
                checks["row5_ablation_pair_5_6"] = ablation.s0_pair_defect() is None and ablation.is_s0_ablation


__all__ = [
    "KILL_TABLE", "T2_POINTS", "KillResult", "kill_phase", "logical_counts", "replay_all", "run_kill_point",
]
