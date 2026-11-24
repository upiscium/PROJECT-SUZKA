"""The real-run drivers (IP-23a): G1 / G2 gates, the S0 attempt with its planned
SIGKILL, the separate-subject negative variant, S1, and the F1 pair driver.

They run UNCHANGED with the scripted provider (the dry runs of the tests: every
refusal and every assertion is exercised end to end through the existing T2
machinery) or with a real model (only through a Human launch).  A real run needs
a ``HumanLaunch`` attestation, the HUMAN approval actor and a free GPU; a
SCRIPTED approver with a real model is refused here as well as in the child.

Results are closed dictionaries of identifiers, digests, row names, counts,
timings and labels; no model text.  The drivers never state an R20 result: S0 is
``{reached, BLOCKED}``, F1 is ``{ARCHITECTURAL_PATH_OBSERVED, PARTIAL_VALID_EVIDENCE,
NOT_YET_TESTABLE}``, and the Stage judgement is the Human's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol
import uuid

from experiments.r20 import artifacts, measure
from experiments.r20.ablation import DefectClass
from experiments.r20.caps import Session
from experiments.r20.dry_run import (
    STEP_NAMES,
    Step,
    _sealed,
    _sealed_from_receipt,
    check_reference,
    populate_reference_manifest,
)
from experiments.r20.f1_table import evaluate_f1_status
from experiments.r20.guards import GuardRefusal
from experiments.r20.kill import KillResult, kill_phase
from experiments.r20.launch import HumanLaunch
from experiments.r20.model_config import ModelOverride, package_versions
from experiments.r20.polling import poll_until
from experiments.r20.prereg import EXPECTED_STRUCTURAL_COUNTS, S1_FIXTURE, S1_SALIENCE_MIN
from experiments.r20.provider import ApprovalActor, ProviderMode, check_modes, r15_protocol_script
from experiments.r20.s0 import S0Session, TurnCapExceeded
from suzka.r15.request_templates import S2_TEMPLATES
from suzka.r16.successor_policy import Interpretation, InterpretationStage, SuccessorPolicyInputV2, evaluate

REFUSE = {"mode": "refuse"}  # DC-5 supervisor mode stays refuse (the Human's decision)
GATE_PASSED = "GATE_PASSED"
NOT_YET_TESTABLE = "NOT_YET_TESTABLE"
STAGE_B_RELEASE = "stage_b_release.json"
ADOPT_MAX_S = 45.0
ADOPT_MAX_S_REAL = 240.0
IP_R15B_TEMPLATE = "S2_UNCERTAINTY_FOCUSED_V1"
S1_R19: dict[str, int] = {"feedback_wake_delay_s": 0, "goal_wake_delay_s": 0, "commitment_wake_delay_s": 604_800,
                          "tick_s": 1, "claim_lease_s": 15}
REAL_S1_R19: dict[str, int] = {**S1_R19, "tick_s": 2, "claim_lease_s": 120}
# A refusal that ends the whole run instead of being one member's infrastructure failure.
STOPPING_REFUSALS = frozenset({
    "CAP_MAX_SUBJECTS", "CAP_STAGE_SUBJECTS", "CAP_SESSION_HOURS", "CAP_TOTAL_HOURS", "STOP_FILE",
    "GPU_FREE_BELOW_MINIMUM", "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH", "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER",
    "HUMAN_LAUNCH_REQUIRED", "SECOND_REAL_PROCESS", "TURN_CAP_EXCEEDED",
})
CLASS_OF_INTERPRETATION = {
    Interpretation.SETTLED: "PRESENT",
    Interpretation.UNCERTAIN: "CONTESTED",  # CONTESTED and UNKNOWN share the row (4b)
    Interpretation.CONTRADICTORY: "CONTRADICTORY",
}


# ---------------------------------------------------------------- approvers


class Approver(Protocol):
    actor: ApprovalActor

    def waiter(self, session: S0Session, approval: dict[str, Any]) -> Callable[[], Any]: ...


def ssh_line(port: int, host: str) -> str:
    """The exact tunnel the Human opens: the browser then sees ``localhost:<port>``,
    which the lab's approval origin requires."""

    return f"ssh -L {port}:127.0.0.1:{port} {host}"


class ScriptedApprover:
    """The harness's stand-in operator.  Fixture-only: refused with a real model."""

    actor = ApprovalActor.SCRIPTED

    def __init__(self) -> None:
        self.last: Any = None

    def waiter(self, session: S0Session, approval: dict[str, Any]) -> Callable[[], Any]:
        def run() -> Any:
            self.last = session.approve(str(approval["intent_id"]))
            return self.last

        return run


class HumanApprover:
    """The Human, personally, through an ssh tunnel to the lab port.  It SENDS
    NOTHING: it prints the URL and the exact tunnel line, then waits until the
    claim row appears (the Human confirmed in the browser) or the window ends."""

    actor = ApprovalActor.HUMAN

    def __init__(self, announce: Callable[[str], None], ssh_host: str, *, poll_s: float = 2.0) -> None:
        self._announce = announce
        self.ssh_host = ssh_host
        self.poll_s = poll_s

    def instructions(self, port: int, approval: dict[str, Any]) -> str:
        from experiments.r20.approval import ApprovalOperator

        return (
            "APPROVAL NEEDED (S0 step 4), by you personally.\n"
            f"  1. on your workstation:  {ssh_line(port, self.ssh_host)}\n"
            f"  2. open:                 {ApprovalOperator.url_for(port)}\n"
            f"  3. read the challenge, type the displayed confirmation code, confirm.\n"
            f"  intent: {approval.get('intent_id')}   window ends (UTC): {approval.get('window_end_utc')}\n"
            "  This is the lab subject's approval only; production's approval page is not involved."
        )

    def waiter(self, session: S0Session, approval: dict[str, Any]) -> Callable[[], Any]:
        def run() -> Any:
            self._announce(self.instructions(session.subject.port, approval))
            end = datetime.strptime(str(approval["window_end_utc"]), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            while datetime.now(timezone.utc) <= end:
                if session.evidence.finalized("R18_EXECUTION_CLAIM") >= 1:
                    return {"claimed": True}
                time.sleep(self.poll_s)
            return {"claimed": session.evidence.finalized("R18_EXECUTION_CLAIM") >= 1}

        return run


# ------------------------------------------------------------------ context


@dataclass
class RunContext:
    base: Path
    approver: Approver
    model: ModelOverride | None = None
    launch: HumanLaunch | None = None
    budget: Session | None = None
    turn_cap_s: float = 120.0
    prereg_sha256: str | None = None
    gpu_check: Callable[[], object] | None = None
    stop_check: Callable[[], None] = lambda: None
    code_check: Callable[[], None] | None = None  # head / clean tree / digests, recomputed per subject
    announce: Callable[[str], None] = print
    sessions: list[S0Session] = field(default_factory=list)

    @property
    def real(self) -> bool:
        return self.model is not None and self.model.is_real

    def check_actor(self) -> None:
        """HD-22-3 / HD-23-11: a scripted approver never goes with a real model;
        a real model needs this process's Human launch."""

        launched = self.launch is not None and self.launch.valid_here()
        check_modes(ProviderMode.REAL if self.real else ProviderMode.SCRIPTED, self.approver.actor,
                    human_launch=self.real and launched)

    def new_session(self, profile: str, *, stage: str, **options: Any) -> S0Session:
        self.stop_check()
        if self.code_check is not None:
            self.code_check()
        self.check_actor()
        if stage == "F1":
            guard_f1(self)  # the single choke point: no F1 subject is created for a real model before this passes
        if self.real and self.gpu_check is not None:
            self.gpu_check()  # >= 14 GiB free, device-wide, from the CUDA runtime
        run_id = uuid.uuid4().hex[:8]
        if self.budget is not None:
            self.budget.begin_subject(run_id, stage=stage)  # a stopped attempt counts
        options.setdefault("dc", dict(REFUSE))
        session = S0Session(
            self.base, profile=profile, run_id=run_id, model=self.model if self.real else None,
            launch=self.launch if self.real else None, turn_cap_s=self.turn_cap_s, **options,
        )
        self.sessions.append(session)
        return session

    def stamp(self, session: S0Session, facts: dict[str, Any]) -> None:
        """Fields every real or scripted manifest carries."""

        session.manifest.stamp(self.prereg_sha256, fixture_only=not self.real)


# ------------------------------------------------------------- measurements


def measurements(session: S0Session) -> dict[str, Any]:
    """Load time, GPU numbers, per-turn seconds (numbers only)."""

    gpu = measure.summarize(measure.read_log(session.subject.evidence_dir / "gpu.jsonl")) if session.real else {}
    return {
        "serving_after_s": None if session.serving_after_s is None else round(session.serving_after_s, 2),
        "turn_seconds": list(session.turn_seconds),
        "max_turn_seconds": max(session.turn_seconds, default=0.0),
        "gpu": gpu,
        "packages": package_versions() if session.real else {},
    }


def finish_session(session: S0Session, extra: dict[str, Any]) -> dict[str, Any]:
    session.manifest.set("real_run", {**measurements(session), **extra})
    session.manifest.write()
    session.stop()
    return {"run_id": session.subject.run_id, **extra}


# ----------------------------------------------------------------- evidence


def first_row4(session: S0Session, timeout_s: float = 60.0) -> dict[str, Any]:
    """The first row-4 / 4b Decision (HD-22-10 (a)) or, when the policy held, the
    HOLD receipts.  ``qualifies`` is the K = 3 selector: row 4, SETTLED, S2."""

    evidence = session.evidence

    def look() -> dict[str, Any] | None:
        decisions = [d for d in evidence.decisions() if d.row in {"4", "4b"}]
        receipts = evidence.records("plan_hold_receipts")
        if not decisions and not receipts:
            return None
        found: dict[str, Any] = {"row": None, "interpretation": None, "stage": None, "qualifies": False,
                                 "receipts": [{"hold_id": r.get("hold_id"), "policy_row": r.get("policy_row")}
                                              for r in receipts]}
        if decisions:
            sealed = _sealed(decisions[0].sealed_input)
            found.update(
                row=decisions[0].row, decision_id=decisions[0].decision_id,
                interpretation=sealed.interpretation.value, stage=sealed.interpretation_stage.value,
                qualifies=decisions[0].row == "4" and sealed.interpretation is Interpretation.SETTLED
                and sealed.interpretation_stage is InterpretationStage.S2,
            )
        return found

    return poll_until(look, timeout_s=timeout_s, interval_s=0.5) or {
        "row": None, "interpretation": None, "stage": None, "qualifies": False, "receipts": [],
        "note": "no row-4 / 4b Decision and no HOLD receipt appeared",
    }


def adopt_and_commit(session: S0Session) -> tuple[Any, Any]:
    """U2a INTEREST -> Goal adoption -> U2c acceptance.  A policy HOLD (a
    CONTRADICTORY S2 holds at the adoption) is not an error of the driver: it is the
    evidence the attempt is reported with (its receipt), so it is returned, not raised."""

    try:
        adopted: Any = session.adopt_goal(max_s=ADOPT_MAX_S_REAL if session.real else ADOPT_MAX_S)
    except GuardRefusal as refusal:
        if refusal.code != "GOAL_NOT_ADOPTED":
            raise
        return {"held": refusal.code}, None
    return adopted, session.accept_commitment()


def structural_counts(session: S0Session) -> dict[str, int]:
    """The counts of S0 step 9, read against ``EXPECTED_STRUCTURAL_COUNTS``."""

    evidence = session.evidence
    counters = session.counters()
    return {
        "row4_decisions": sum(1 for d in evidence.decisions() if d.row == "4"),
        "intents": counters["R18_INTENT_ISSUE"],
        "execution_claims": counters["R18_EXECUTION_CLAIM"],
        "target_effects": evidence.target_effects(),
        "feedback": counters["R17_FEEDBACK_COMMIT"],
        "feedback_admitted_wakes": evidence.wake_causes().count("FEEDBACK_ADMITTED"),
        "outbox_messages": evidence.table_count("outbox_messages"),
    }


def compare_structural(counts: dict[str, int]) -> dict[str, Any]:
    differing = {name: {"expected": want, "observed": counts.get(name)}
                 for name, want in EXPECTED_STRUCTURAL_COUNTS.items() if counts.get(name) != want}
    return {"counts": counts, "expected": dict(EXPECTED_STRUCTURAL_COUNTS), "equal": not differing, "differing": differing}


# --------------------------------------------------------------------- G1


def run_g1(ctx: RunContext) -> dict[str, Any]:
    """DRV-1 D with the model: a throwaway S0-profile subject; setup + E (= X); the
    U2a review must run and a Goal must be adopted on X.  X neutral for the model
    => ``NOT_YET_TESTABLE`` and nothing further is spent.  Also measures the load
    time, the GPU memory and the turn times."""

    session = ctx.new_session("S0", stage="G1")
    outcome: dict[str, Any] = {"gate": "G1", "label": NOT_YET_TESTABLE, "x_not_skipped_as_neutral": False}
    try:
        session.start()
        session.activate()
        from experiments.r15_p0.pilot import Case

        prompts = Case("r20-g1", "B", "forced", 0, "normal").prompts()
        for key in ("setup", "E"):
            data = session.chat(prompts[key])
            session.experience_ids.append(str(data["experience_id"]))
        try:
            session.adopt_goal()
            adopted = True
        except GuardRefusal as refusal:
            adopted = False
            outcome["not_adopted"] = refusal.code
        events = session.evidence.journal_events()
        reviews = [e for e in events if e["event_type"] == "motivation_source_review"]
        outcome["u2a_review_events"] = len(reviews)
        passed = adopted and len(reviews) >= 1
        outcome["x_not_skipped_as_neutral"] = passed
        outcome["label"] = GATE_PASSED if passed else NOT_YET_TESTABLE
        outcome["reading"] = ("the U2a review ran and a Goal was adopted on X, so X was not skipped as "
                              "neutral_no_eligible_appraisal; otherwise NOT_YET_TESTABLE")
        ctx.stamp(session, outcome)
    except TurnCapExceeded as stop:
        outcome["stop_reason"] = stop.code
    except BaseException:
        session.stop()
        raise
    return {**finish_session(session, outcome), **measurements(session)}


# ---------------------------------------------------------------------- S0


def run_s0_attempt(ctx: RunContext, attempt: int, *, kill_point: str = "K4", **options: Any) -> dict[str, Any]:
    """One K-attempt: a fresh S0 subject, the full chain, the Human's approval, the
    planned single SIGKILL, restart, and the structural counts of step 9.  An
    attempt that is not row 4 / SETTLED / S2 stops early and is reported with its
    receipt; it is not S0 and it counts toward K."""

    session = ctx.new_session("S0", stage="S0", **options)
    record: dict[str, Any] = {"attempt": attempt, "qualifies_as_s0": False, "label": "NOT_S0", "stop_reason": None}
    steps: dict[str, Step] = {str(n): Step() for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)}
    try:
        session.start()
        chain: dict[str, Any] = {"activation": session.activate()}
        chain["r15"] = session.run_r15_protocol()
        chain["adopt"], chain["commit"] = adopt_and_commit(session)
        selector = first_row4(session)
        record["first_row4"] = selector
        if not selector["qualifies"]:
            record["not_s0_because"] = (
                f"first row-4 Decision: row {selector['row']}, {selector['interpretation']}/{selector['stage']}; "
                f"receipts {selector['receipts']}"
            )
            return finish_session(session, {**record, "chain": {"commit": chain["commit"]}})
        chain["approval"] = session.open_approval()
        waiter = ctx.approver.waiter(session, chain["approval"])
        window_s = float(session.subject.resolved_settings().r18.approval_window_seconds)
        human = ctx.approver.actor is ApprovalActor.HUMAN
        killed: KillResult = kill_phase(
            session, kill_point, waiter,
            kill_wait_s=(window_s + 120.0) if human else 90.0,
            converge_s=900.0 if ctx.real else 150.0,
            chat_after_restart=False,
        )
        record["kill"] = killed.manifest_entry()
        if isinstance(ctx.approver, ScriptedApprover) and ctx.approver.last is not None:
            last = ctx.approver.last
            chain["confirm"] = {"status": last.status, "body": dict(last.json()) if last.status < 500 else {}}
        facts: dict[str, Any] = {}
        check_reference(session, steps, chain, facts, finish_wait_s=60.0, approval_actor=ctx.approver.actor)
        step9 = Step()
        step9.require("kill_window_hit", killed.window_hit and killed.window_confirmed)
        for name, held in killed.checks.items():
            step9.require(name, held)
        counted = compare_structural(structural_counts(session))
        step9.require("counts_equal_the_preregistered_structural_counts", counted["equal"], counted)
        steps["10"] = step9
        record["step9_structural"] = counted
        record["steps"] = {key: ("DONE" if not step.problems and step.evidence else "NOT_DONE") for key, step in steps.items()}
        record["problems"] = {key: step.problems for key, step in steps.items() if step.problems}
        record["qualifies_as_s0"] = not record["problems"]
        record["label"] = "S0_REACHED_CANDIDATE" if record["qualifies_as_s0"] else "NOT_S0"
        populate_reference_manifest(
            session, facts, approval_actor=ctx.approver.actor,
            provider=session.provider_identity() | ({"packages": package_versions()} if ctx.real else {}),
        )
        ctx.stamp(session, facts)
        for number, step in steps.items():
            session.manifest.step(int(number), STEP_NAMES[int(number)],
                                  "DONE" if step.evidence and not step.problems else "NOT_DONE",
                                  {**step.evidence, "failed": list(step.problems)} if step.problems else step.evidence)
        session.manifest.append("kill_points", record["kill"])
        record["journal"] = facts.get("journal")
        record["ablation"] = facts.get("ablation")
    except TurnCapExceeded as stop:
        record["stop_reason"] = stop.code
    except BaseException:
        session.stop()
        raise
    return finish_session(session, record)


def run_negative_variant(ctx: RunContext, *, wait_s: float | None = None) -> dict[str, Any]:
    """MISSING_APPROVAL on a separate subject: the chain to an open approval, then
    NOTHING is approved.  Expected: zero claims, target bytes unchanged, the subject
    keeps running, and (after the window, >= 600 s) row 2c.  ``wait_s`` shortens the
    wait for a dry run; ``row_2c_confirmed`` is then ``None`` (T1 F05-w covers the close)."""

    session = ctx.new_session("S0", stage="S0")
    record: dict[str, Any] = {"variant": "MISSING_APPROVAL", "stop_reason": None}
    try:
        session.start()
        session.activate()
        session.run_r15_protocol()
        adopt_and_commit(session)
        selector = first_row4(session)
        record["first_row4"] = selector
        if not selector["qualifies"]:
            record["label"] = "NOT_RUN_NOT_SETTLED"
            return finish_session(session, record)
        approval = session.open_approval()
        digest_before = session.evidence.target_digest()
        window_s = float(session.subject.resolved_settings().r18.approval_window_seconds)
        full = wait_s is None
        waited = (window_s + 60.0) if full else wait_s
        if ctx.real:
            ctx.announce("NEGATIVE VARIANT: do NOT approve. The intent closes at its window end "
                         f"({approval.get('window_end_utc')}); this takes about {window_s / 60:.0f} minutes.")
        assert waited is not None
        closed = bool(poll_until(lambda: any(d.row == "2c" for d in session.evidence.decisions()),
                                 timeout_s=waited, interval_s=1.0)) if full else False
        if not full:
            time.sleep(waited)
        record["zero_claims"] = session.evidence.finalized("R18_EXECUTION_CLAIM") == 0
        record["target_unchanged"] = session.evidence.target_digest() == digest_before
        record["subject_running"] = session.request("GET", "/health", timeout_s=30).status == 200
        record["row_2c_confirmed"] = closed if full else None
        record["window_waited_s"] = round(waited, 1)
        ok = record["zero_claims"] and record["target_unchanged"] and record["subject_running"] and (closed or not full)
        record["label"] = "VARIANT_HELD" if ok else "VARIANT_DID_NOT_HOLD"
        ctx.stamp(session, record)
    except TurnCapExceeded as stop:
        record["stop_reason"] = stop.code
    except BaseException:
        session.stop()
        raise
    return finish_session(session, record)


def run_s0_block(
    ctx: RunContext, *, kill_point: str = "K4", max_attempts: int = 3, negative_wait_s: float | None = None,
    **options: Any,
) -> dict[str, Any]:
    """The K = 3 rule: up to 3 attempts in the pre-registered order; the first
    attempt that qualifies is S0; none qualifies => BLOCKED.  No extra attempts."""

    attempts: list[dict[str, Any]] = []
    for number in range(1, max_attempts + 1):
        outcome = run_s0_attempt(ctx, number, kill_point=kill_point, **options)
        attempts.append(outcome)
        if outcome.get("stop_reason") in {"TURN_CAP_EXCEEDED"}:
            break  # a stop reason ends the block; the stopped attempt counted
        if outcome["qualifies_as_s0"]:
            break
    reached = any(item["qualifies_as_s0"] for item in attempts)
    block: dict[str, Any] = {"attempts": attempts, "s0": "reached" if reached else "BLOCKED"}
    if reached and not attempts[-1].get("stop_reason"):
        block["negative_variant"] = run_negative_variant(ctx, wait_s=negative_wait_s)
    return block


# ----------------------------------------------------------------------- S1


def read_salience(session: S0Session, experience_id: str) -> float | None:
    """The ``subjective_salience`` of one Experience record, by a plain read of its
    JSON files (the number only; nothing else is kept)."""

    settings = session.subject.resolved_settings()
    records = settings.memory.persist_directory.parent / "experience" / "records" / experience_id
    best: float | None = None
    for path in sorted(records.glob("*.json")) if records.is_dir() else []:
        try:
            found = _find_key(json.loads(path.read_text(encoding="utf-8")), "subjective_salience")
        except (OSError, ValueError):
            continue
        if found is not None:
            best = found
    return best


def _find_key(value: Any, key: str) -> float | None:
    if isinstance(value, dict):
        if key in value:
            item = value[key]
            try:
                return float.fromhex(item) if isinstance(item, str) else float(item)
            except (TypeError, ValueError):
                return None
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def _s1_options(ctx: RunContext) -> dict[str, Any]:
    return {"r19": dict(REAL_S1_R19 if ctx.real else S1_R19), "script": ()}


def run_g2(ctx: RunContext) -> dict[str, Any]:
    """The S1 salience gate on a throwaway S1-profile subject: the pre-registered
    fixture turn must reach salience >= 0.5, else S1 is ``NOT_YET_TESTABLE``."""

    session = ctx.new_session("S1", stage="G2", **_s1_options(ctx))
    outcome: dict[str, Any] = {"gate": "G2", "label": NOT_YET_TESTABLE, "salience": None, "threshold": S1_SALIENCE_MIN}
    try:
        session.start()
        session.activate()
        data = session.chat(S1_FIXTURE)
        salience = read_salience(session, str(data["experience_id"]))
        outcome["salience"] = salience
        outcome["label"] = GATE_PASSED if salience is not None and salience >= S1_SALIENCE_MIN else NOT_YET_TESTABLE
        ctx.stamp(session, outcome)
    except TurnCapExceeded as stop:
        outcome["stop_reason"] = stop.code
    except BaseException:
        session.stop()
        raise
    return finish_session(session, outcome)


def run_s1_attempt(ctx: RunContext, attempt: int) -> dict[str, Any]:
    """One S1 run: the fixture turn grounds a SELF-origin Goal (self-goal path
    active, R18 off); a GOAL_ADMITTED Wake without a user turn re-evaluates; the
    cycle closes with the D-R19 resolution; a user request on the occupied X gets
    the reasoned HOLD ``SELF_GOAL_OCCUPIES_TARGET``.  The observed result is reported
    against the pre-registered expectation; nothing is judged here."""

    session = ctx.new_session("S1", stage="S1", **_s1_options(ctx))
    record: dict[str, Any] = {"attempt": attempt, "stop_reason": None}
    try:
        session.start()
        record["activation"] = session.activate()
        data = session.chat(S1_FIXTURE)
        session.experience_ids.extend(["", str(data["experience_id"])])  # X is the second id by convention
        evidence = session.evidence
        record["self_goal_decision"] = bool(poll_until(
            lambda: any(d.row == "3" for d in evidence.decisions()), timeout_s=session.step_timeout_s, interval_s=0.5))
        record["goal_wake"] = bool(poll_until(
            lambda: "GOAL_ADMITTED" in evidence.wake_causes(), timeout_s=session.step_timeout_s, interval_s=0.5))
        record["cycle_closed"] = bool(poll_until(
            lambda: evidence.table_count("logical_cycle_resolutions") >= 1, timeout_s=session.step_timeout_s, interval_s=0.5))
        events = evidence.journal_events()
        decisions = evidence.decisions()
        adoption = [d.sequence for d in decisions if d.row == "3"]
        wake_review = [e["sequence"] for e in events if e["event_type"] == "r16_wake_review"]
        chat_between = [e["sequence"] for e in events if e["event_type"] == "chat" and adoption and wake_review
                        and adoption[0] < e["sequence"] < wake_review[0]]
        record["no_user_turn_before_the_wake_review"] = bool(adoption and wake_review) and not chat_between
        cycle = [d for d in decisions if d.operation_kind == "R16_CYCLE_DECISION"]
        record["resolution_kinds"] = [str(r.get("result_kind")) for r in evidence.records("logical_cycle_resolutions")]
        # the same fingerprint gives EXISTING_DECISION: the Wake closes on the row-3 Decision already issued
        relevant = cycle[0] if cycle else next((d for d in decisions if d.row == "3"), None)
        if relevant is not None:
            output = evaluate(SuccessorPolicyInputV2.from_bytes(relevant.sealed_input)).output.value
            record["wake_outcome"] = {"decision_row": relevant.row, "output": output, "new_cycle_decision": bool(cycle)}
            record["matches_expectation"] = bool(record["cycle_closed"]) and output in {"PLAN_A", "HOLD"}
        else:
            record["matches_expectation"] = False
        record["goal_adoption_rows"] = [d.row for d in decisions if d.row in {"3"}]
        # a user request on the occupied X
        ticket = session._command(f"COMMIT_REVIEW_EXPERIENCE:1:{session.x}")  # noqa: SLF001
        status = poll_until(lambda: (lambda s: None if s == "pending" else s)(session.ticket_status(ticket)),
                            timeout_s=session.ticket_timeout_s)
        reason = session.request("GET", f"/api/chat/commands/{ticket}").json().get("reason")
        record["user_request_on_x"] = {"status": status, "reason": reason}
        record["self_goal_occupies_target"] = str(reason).upper() == "SELF_GOAL_OCCUPIES_TARGET"
        record["expected"] = "Plan A continues, or a HOLD with a reason; cycle closed by the D-R19 resolution"
        ctx.stamp(session, record)
    except TurnCapExceeded as stop:
        record["stop_reason"] = stop.code
    except BaseException:
        session.stop()
        raise
    return finish_session(session, record)


def run_s1_block(ctx: RunContext, *, max_attempts: int = 2) -> dict[str, Any]:
    gate = run_g2(ctx)
    block: dict[str, Any] = {"g2": gate, "attempts": []}
    if gate["label"] != GATE_PASSED:
        block["s1"] = NOT_YET_TESTABLE
        return block
    for number in range(1, max_attempts + 1):
        attempt = run_s1_attempt(ctx, number)
        block["attempts"].append(attempt)
        if attempt.get("stop_reason") or (attempt.get("cycle_closed") and attempt.get("self_goal_decision")):
            break
    block["s1"] = "OBSERVED"
    return block


# ----------------------------------------------------------------------- F1


def s2_status_script(status: str) -> tuple[str, ...]:
    """Scripted typed S2 answer for an R15 status class (the dry run only)."""

    script = list(r15_protocol_script())
    answer = json.loads(script[4])
    recipe = {
        "PRESENT": ("B", "MAY_HAVE_MISREAD"),
        "CONTESTED": ("B", "LIKELY_ALIGNED"),
        "UNKNOWN": ("UNKNOWN", "UNKNOWN"),
        "CONTRADICTORY": ("A", "LIKELY_ALIGNED"),
    }
    if status not in recipe:
        raise GuardRefusal("F1_STATUS_UNKNOWN", status)
    answer["assumed_referent"], answer["assessment"] = recipe[status]
    script[4] = json.dumps(answer, separators=(",", ":"), sort_keys=True)
    return tuple(script)


def non_s_part(sealed: SuccessorPolicyInputV2) -> dict[str, Any]:
    """The branch-independent part of a sealed input: states and ref KINDS, no
    event ids, hashes or times (branch-specific by construction)."""

    return {
        "required_source": sealed.required_source.value, "predecessor": sealed.predecessor.value,
        "lineage": sealed.lineage.value, "commitment": sealed.commitment.value, "ledger": sealed.ledger.value,
        "feedback": sealed.feedback.value, "version": sealed.version.value,
        "source_ref_kinds": sorted(ref.kind for ref in sealed.source_refs),
        "has_commitment_deadline": sealed.commitment_deadline_utc is not None,
    }


def run_f1_member(ctx: RunContext, pair: int, arm: str, status: str | None) -> dict[str, Any]:
    """One F1 branch: a fresh F1-profile subject (r18 off).  ``status`` is the
    scripted S2 class (dry run); a real model supplies its own."""

    options: dict[str, Any] = {"script": s2_status_script(status)} if status is not None and not ctx.real else {}
    session = ctx.new_session("F1", stage="F1", **options)
    member: dict[str, Any] = {"pair": pair, "arm": arm, "stop_reason": None, "admitted": False,
                              "defect": None, "sealed_row": None}
    try:
        session.start()
        session.activate()
        reviews = session.run_r15_protocol()
        member["admitted"] = reviews["S1"]["revision"] == 1 and reviews["S2"]["revision"] == 2
        adopted, committed = adopt_and_commit(session)
        member["goal_held"] = committed is None  # a HOLD at adoption (CONTRADICTORY): the receipt carries it
        member["u2c"] = committed
        selector = first_row4(session, timeout_s=session.ticket_timeout_s)
        evidence = session.evidence
        member["action_intents"] = evidence.table_count("action_intents")
        member["receipts"] = selector["receipts"]
        decisions = [d for d in evidence.decisions() if d.row in {"4", "4b"}]
        if decisions:
            sealed = _sealed(decisions[0].sealed_input)
            klass = CLASS_OF_INTERPRETATION.get(sealed.interpretation)
            member.update(sealed_row=decisions[0].row, status_class=klass)
            if klass is not None:
                check = evaluate_f1_status(decisions[0].sealed_input, klass)
                member["expected_row"] = check.expected_row.value
                member["defect"] = (check.defect.value if check.defect is not None
                                    else (DefectClass.GATE_CONFORMANCE_DEFECT.value
                                          if check.evaluated_row.value != decisions[0].row else None))
        elif selector["receipts"]:
            receipt = evidence.records("plan_hold_receipts")[0]
            sealed = _sealed_from_receipt(receipt)
            member.update(sealed_row=receipt.get("policy_row"), status_class="CONTRADICTORY", hold_sealed=True)
            member["expected_row"] = "2b"
            member["defect"] = None if receipt.get("policy_row") == "2b" else DefectClass.GATE_CONFORMANCE_DEFECT.value
        # The branch-independent comparison is made on the FIRST policy input of each branch (the Goal
        # adoption stage), the one stage both branches reach: a HOLD at adoption has no later input.
        issued = sorted((d for d in evidence.decisions() if d.operation_kind == "R16_DECISION_ISSUE"), key=lambda d: d.sequence)
        if issued:
            member["non_s"] = non_s_part(_sealed(issued[0].sealed_input))
        elif selector["receipts"]:
            member["non_s"] = non_s_part(_sealed_from_receipt(evidence.records("plan_hold_receipts")[0]))
        member["measure"] = measurements(session)
        ctx.stamp(session, member)
    except TurnCapExceeded as stop:
        member["stop_reason"] = stop.code
    except GuardRefusal as refusal:
        if refusal.code in STOPPING_REFUSALS:
            session.stop()
            raise
        member["infra"] = refusal.code
    except Exception as error:  # an infrastructure failure of the member, reported
        member["infra"] = type(error).__name__
    except BaseException:
        session.stop()
        raise
    finish_session(session, {k: v for k, v in member.items() if k not in {"measure", "non_s"}})
    return member


def classify_pair(first: dict[str, Any], second: dict[str, Any]) -> str:
    """INFRA_FAILED > NOT_ADMITTED > CONFOUNDED > VALID."""

    members = (first, second)
    if any(m.get("infra") or m.get("stop_reason") for m in members):
        return "INFRA_FAILED"
    if any(
        not m["admitted"] or m.get("action_intents") != 0 or m["sealed_row"] is None
        or (m.get("status_class") == "CONTRADICTORY" and not m.get("hold_sealed"))
        or (m.get("status_class") != "CONTRADICTORY" and m.get("u2c") != "accepted")
        for m in members
    ):
        return "NOT_ADMITTED"
    if any(m.get("defect") for m in members) or first.get("non_s") != second.get("non_s"):
        return "CONFOUNDED"
    return "VALID"


def run_f1_pair(ctx: RunContext, pair: dict[str, Any], statuses: dict[str, str | None]) -> dict[str, Any]:
    """One pair: two branches, the pre-registered branch order, same seed / fixture;
    ``statuses`` maps an arm to its scripted S2 class (dry run only)."""

    guard_f1(ctx)
    kind = str(pair["kind"])
    arms = {"TREATMENT": ("NEUTRAL", "UNCERTAINTY"), "CONTROL_NEUTRAL": ("NEUTRAL", "NEUTRAL"),
            "CONTROL_UNCERTAINTY": ("UNCERTAINTY", "UNCERTAINTY")}[kind]
    order = list(arms) if pair["first_branch"] == arms[0] else list(reversed(arms))
    members = [run_f1_member(ctx, int(pair["pair"]), arm, statuses.get(arm)) for arm in order]
    by_arm = {m["arm"]: m for m in members} if kind == "TREATMENT" else {}
    klass = classify_pair(members[0], members[1])
    result: dict[str, Any] = {"pair": pair["pair"], "kind": kind, "seed": pair["seed"], "class": klass,
                              "members": [{k: v for k, v in m.items() if k not in {"non_s", "measure"}} for m in members]}
    if kind == "TREATMENT" and klass == "VALID":
        result["discordant"] = by_arm["NEUTRAL"]["sealed_row"] != by_arm["UNCERTAINTY"]["sealed_row"]
    if kind != "TREATMENT" and klass == "VALID":
        result["discordant"] = members[0]["sealed_row"] != members[1]["sealed_row"]
    return result


def summarize_f1(pairs: list[dict[str, Any]], *, planned: int = 12) -> dict[str, Any]:
    treatment = [p for p in pairs if p["kind"] == "TREATMENT" and p["class"] == "VALID"]
    controls = [p for p in pairs if p["kind"] != "TREATMENT" and p["class"] == "VALID"]
    discordant = [p for p in treatment if p.get("discordant")]
    e_r15 = (len(discordant) / len(treatment)) if treatment else None
    all_valid = len(pairs) == planned and all(p["class"] == "VALID" for p in pairs)
    if not discordant:
        ceiling = NOT_YET_TESTABLE
    elif all_valid:
        ceiling = "ARCHITECTURAL_PATH_OBSERVED"
    else:
        ceiling = "PARTIAL_VALID_EVIDENCE"
    return {
        "pairs": len(pairs), "planned": planned,
        "classes": {name: sum(1 for p in pairs if p["class"] == name)
                    for name in ("INFRA_FAILED", "NOT_ADMITTED", "CONFOUNDED", "VALID")},
        "valid_treatment_pairs": len(treatment), "discordant_valid_treatment_pairs": len(discordant),
        "e_r15": e_r15, "control_pairs_valid": len(controls),
        "control_discordance": sum(1 for p in controls if p.get("discordant")),
        "claim_ceiling": ceiling, "i_to_s": "DESCRIPTIVE_ONLY / CAUSALITY_UNESTABLISHED",
    }


def stage_b_released(base: Path, prereg_sha256: str | None) -> bool:
    """The release file has no nonce and no MAC: a same-uid writer can forge it, so this gate holds
    against CODE PATHS only (a driver, a script that does not write the file).  The other stage-B
    gates are the registered IP-R15b template and the Human's own launch."""

    path = Path(base) / STAGE_B_RELEASE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        info = path.stat()
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict) and data.get("prereg_sha256") == prereg_sha256 and prereg_sha256 is not None
        and data.get("released_by") == "human" and info.st_uid == os.getuid() and not (info.st_mode & 0o077)
    )


def guard_f1(ctx: RunContext) -> None:
    """The F1 pair driver runs with a REAL model only after the Human wrote the
    stage-B release file (bound to the pre-registration hash), IP-R15b's template is
    registered, and the arm template is wired.  The scripted dry run needs none."""

    if not ctx.real:
        return
    if not stage_b_released(ctx.base, ctx.prereg_sha256):
        raise GuardRefusal("STAGE_B_NOT_RELEASED", f"{STAGE_B_RELEASE} written by the Human is required")
    if IP_R15B_TEMPLATE not in S2_TEMPLATES:
        raise GuardRefusal("IP_R15B_NOT_MERGED", f"{IP_R15B_TEMPLATE} is not registered")
    raise GuardRefusal("F1_ARM_NOT_WIRED", "the UNCERTAINTY arm's S2 template is not wired into the protocol driver")


def run_f1(ctx: RunContext, plan: dict[str, Any], statuses_for: Callable[[dict[str, Any]], dict[str, str | None]],
           *, only: list[int] | None = None) -> dict[str, Any]:
    guard_f1(ctx)
    results: list[dict[str, Any]] = []
    consecutive_infra = 0
    by_number = {int(p["pair"]): p for p in plan["pairs"]}
    for number in plan["trial_order"]:
        if only is not None and number not in only:
            continue
        pair = by_number[number]
        outcome = run_f1_pair(ctx, pair, statuses_for(pair))
        results.append(outcome)
        consecutive_infra = consecutive_infra + 1 if outcome["class"] == "INFRA_FAILED" else 0
        if consecutive_infra >= 3:
            return {"pairs": results, "summary": summarize_f1(results, planned=len(plan["pairs"])),
                    "stop_reason": "THREE_CONSECUTIVE_INFRA_FAILURES"}
    # ``planned`` is always the pre-registered pair count: a subset (a dry run) can only reach the PARTIAL ceiling
    return {"pairs": results, "summary": summarize_f1(results, planned=len(plan["pairs"])), "stop_reason": None}


__all__ = [
    "GATE_PASSED", "IP_R15B_TEMPLATE", "NOT_YET_TESTABLE", "STAGE_B_RELEASE", "Approver", "HumanApprover",
    "RunContext", "ScriptedApprover", "artifacts", "ssh_line", "classify_pair", "compare_structural", "first_row4",
    "guard_f1", "read_salience", "run_f1", "run_f1_pair", "run_g1", "run_g2", "run_negative_variant", "run_s0_attempt",
    "run_s0_block", "run_s1_attempt", "run_s1_block", "s2_status_script", "stage_b_released", "structural_counts",
    "summarize_f1",
]
