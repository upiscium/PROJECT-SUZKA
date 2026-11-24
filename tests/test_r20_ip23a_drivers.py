"""IP-23a: the drivers (G1 / G2 / S0 with its planned SIGKILL / negative variant /
S1 / F1 pairs) run end to end through the existing T2 machinery with the SCRIPTED
provider, plus every refusal that keeps a real model out of a non-Human path.
Nothing here loads a model; real-model mode is only constructed, never started."""

from __future__ import annotations

import json
import stat
import time
from pathlib import Path
from typing import Any

import pytest
from r20_ip23_helpers import human_launch, real_model, refusal_code

from experiments.r20 import drivers, prereg
from experiments.r20.caps import Caps, Ledger, Session
from experiments.r20.drivers import (
    HumanApprover,
    RunContext,
    ScriptedApprover,
    classify_pair,
    compare_structural,
    guard_f1,
    s2_status_script,
    stage_b_released,
    summarize_f1,
)
from experiments.r20.guards import GuardRefusal
from experiments.r20.s0 import S0Session, TurnCapExceeded


def scripted(base: Path, **changes: Any) -> RunContext:
    return RunContext(base=base, approver=ScriptedApprover(), prereg_sha256="c" * 64, **changes)


def real_context(base: Path, **changes: Any) -> RunContext:
    values: dict[str, Any] = {
        "base": base, "approver": HumanApprover(lambda _text: None, "host"), "model": real_model(),
        "launch": human_launch(), "prereg_sha256": "c" * 64,
    }
    values.update(changes)
    return RunContext(**values)


def manifest_of(outcome: dict[str, Any], base: Path) -> dict[str, Any]:
    return json.loads((base / outcome["run_id"] / "evidence" / "manifest.json").read_text())


# ---------------------------------------------------------- real-mode refusals


def test_a_scripted_approval_with_a_real_model_is_refused_before_a_subject_exists(r20_base: Path) -> None:
    ctx = real_context(r20_base, approver=ScriptedApprover())
    assert refusal_code(ctx.new_session, "S0", stage="G1") == "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER"
    assert refusal_code(drivers.run_g1, ctx) == "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER"
    assert refusal_code(drivers.run_s0_attempt, ctx, 1) == "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER"
    assert [p for p in r20_base.iterdir()] == [] and not ctx.sessions


def test_a_real_context_without_this_processes_human_launch_is_refused(r20_base: Path) -> None:
    assert refusal_code(real_context(r20_base, launch=None).new_session, "S0", stage="G1") == "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH"
    foreign = human_launch()
    object.__setattr__(foreign, "pid", foreign.pid + 1)
    assert refusal_code(real_context(r20_base, launch=foreign).new_session, "S0", stage="G1") == "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH"
    assert refusal_code(S0Session, r20_base, model=real_model(), launch=None) == "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH"
    assert [p for p in r20_base.iterdir()] == []


def test_the_gpu_and_the_stop_check_and_the_budget_run_before_every_real_subject(r20_base: Path) -> None:
    def low() -> None:
        raise GuardRefusal("GPU_FREE_BELOW_MINIMUM", "free 3 GiB")

    assert refusal_code(real_context(r20_base, gpu_check=low).new_session, "S0", stage="G1") == "GPU_FREE_BELOW_MINIMUM"

    def stop() -> None:
        raise GuardRefusal("STOP_FILE", "stop requested")

    assert refusal_code(real_context(r20_base, stop_check=stop).new_session, "S0", stage="G1") == "STOP_FILE"
    budget = Session(Ledger(r20_base), Caps(max_subjects=1), "L", "G1")
    ctx = real_context(r20_base, budget=budget, gpu_check=lambda: None)
    ctx.new_session("S0", stage="G1")  # constructs the subject; no child, no model
    assert refusal_code(ctx.new_session, "S0", stage="G1") == "CAP_MAX_SUBJECTS"  # a stopped attempt counts
    assert Ledger(r20_base).subjects_used() == 1


def test_a_real_session_keeps_raw_transcripts_private_and_never_starts_here(r20_base: Path) -> None:
    ctx = real_context(r20_base, gpu_check=lambda: None)
    session = ctx.new_session("S0", stage="G1")
    private = session.capture_dir
    assert private == session.subject.run_dir / "private" and private != session.subject.evidence_dir
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert "14 days" in (private / "RETENTION.txt").read_text()
    assert session.real and session.step_timeout_s == 300.0 and session.provider_identity()["mode"] == "real"
    assert session.child is None and not session.subject.provider_config.exists()
    # a scripted session keeps the (fixture-only) capture where it always was
    plain = scripted(r20_base).new_session("S0", stage="G1")
    assert plain.capture_dir == plain.subject.evidence_dir and not plain.real


def test_the_turn_cap_stops_a_real_session(r20_base: Path) -> None:
    ctx = real_context(r20_base, gpu_check=lambda: None, turn_cap_s=0.2)
    session = ctx.new_session("S0", stage="G1")

    class Slow:
        def request(self, method: str, path: str, body: Any = None, headers: Any = None) -> Any:
            time.sleep(0.4)
            raise AssertionError("never reached: the answer came too late")

    class Late:
        def request(self, method: str, path: str, body: Any = None, headers: Any = None) -> Any:
            time.sleep(0.3)
            return type("R", (), {"status": 200, "text": "", "json": lambda self: {}})()

    class TimedOut:
        def request(self, method: str, path: str, body: Any = None, headers: Any = None) -> Any:
            raise TimeoutError

    session.client = Late()  # type: ignore[assignment]
    with pytest.raises(TurnCapExceeded) as late:
        session.chat("x")
    assert late.value.code == "TURN_CAP_EXCEEDED"
    session.client = TimedOut()  # type: ignore[assignment]
    with pytest.raises(TurnCapExceeded):
        session.chat("x")
    assert Slow  # (documents the third shape: a hung request is cut by the transport timeout)


def test_the_human_approver_prints_the_exact_tunnel_and_url_and_sends_nothing(r20_base: Path) -> None:
    lines: list[str] = []
    approver = HumanApprover(lines.append, "lab-host", poll_s=0.01)
    approval = {"intent_id": "intent-1", "window_end_utc": "2020-01-01T00:00:00Z"}
    text = approver.instructions(43210, approval)
    assert drivers.ssh_line(43210, "lab-host") == "ssh -L 43210:127.0.0.1:43210 lab-host" and "ssh -L 43210:127.0.0.1:43210 lab-host" in text
    assert "http://localhost:43210/api/r18/approve" in text and "intent-1" in text and "production" in text

    class Evidence:
        def __init__(self, claimed: int) -> None:
            self.claimed = claimed

        def finalized(self, kind: str) -> int:
            return self.claimed

    class Session:
        subject = type("S", (), {"port": 43210})()
        client = None  # the approver has no client: it cannot send anything

        def __init__(self, claimed: int) -> None:
            self.evidence = Evidence(claimed)

    assert approver.waiter(Session(0), approval)() == {"claimed": False}  # the window is over, nothing approved
    assert lines and "APPROVAL NEEDED" in lines[0]
    future = {"intent_id": "i", "window_end_utc": "2999-01-01T00:00:00Z"}
    assert approver.waiter(Session(1), future)() == {"claimed": True}
    assert approver.actor.value == "HUMAN" and ScriptedApprover.actor.value == "SCRIPTED"


# ------------------------------------------------------- pure classification


def member(**changes: Any) -> dict[str, Any]:
    base = {"arm": "NEUTRAL", "admitted": True, "action_intents": 0, "sealed_row": "4", "u2c": "accepted", "defect": None,
            "status_class": "PRESENT", "non_s": {"a": 1}}
    base.update(changes)
    return base


def test_the_pair_class_priority_is_infra_then_not_admitted_then_confounded() -> None:
    ok = member()
    assert classify_pair(ok, member()) == "VALID"
    assert classify_pair(ok, member(infra="OSError", admitted=False, defect="X")) == "INFRA_FAILED"
    assert classify_pair(ok, member(stop_reason="TURN_CAP_EXCEEDED")) == "INFRA_FAILED"
    assert classify_pair(ok, member(admitted=False, defect="GATE_CONFORMANCE_DEFECT")) == "NOT_ADMITTED"
    assert classify_pair(ok, member(action_intents=1)) == "NOT_ADMITTED"  # F1 has r18 off: zero ActionIntent rows
    assert classify_pair(ok, member(u2c="held")) == "NOT_ADMITTED"
    assert classify_pair(ok, member(status_class="CONTRADICTORY", sealed_row="2b")) == "NOT_ADMITTED"  # HOLD_UNSEALED
    sealed = member(status_class="CONTRADICTORY", sealed_row="2b", hold_sealed=True, u2c=None)
    assert classify_pair(ok, sealed) == "VALID"
    assert classify_pair(ok, member(defect="GATE_CONFORMANCE_DEFECT")) == "CONFOUNDED"
    assert classify_pair(ok, member(non_s={"a": 2})) == "CONFOUNDED"


def test_the_f1_claim_ceilings() -> None:
    def pair(number: int, kind: str = "TREATMENT", klass: str = "VALID", discordant: bool = True) -> dict[str, Any]:
        return {"pair": number, "kind": kind, "class": klass, "discordant": discordant}

    full = [pair(n) for n in range(1, 9)] + [pair(9, "CONTROL_NEUTRAL", discordant=False), pair(10, "CONTROL_NEUTRAL", discordant=False),
                                            pair(11, "CONTROL_UNCERTAINTY", discordant=False), pair(12, "CONTROL_UNCERTAINTY", discordant=False)]
    done = summarize_f1(full)
    assert done["claim_ceiling"] == "ARCHITECTURAL_PATH_OBSERVED" and done["e_r15"] == 1.0 and done["control_discordance"] == 0
    partial = [*full[:-1], pair(12, "CONTROL_UNCERTAINTY", klass="CONFOUNDED")]
    assert summarize_f1(partial)["claim_ceiling"] == "PARTIAL_VALID_EVIDENCE"
    assert summarize_f1(full[:2])["claim_ceiling"] == "PARTIAL_VALID_EVIDENCE"  # a subset can never reach the top ceiling
    none = [pair(n, discordant=False) for n in range(1, 9)] + full[8:]
    assert summarize_f1(none)["claim_ceiling"] == "NOT_YET_TESTABLE" and summarize_f1(none)["e_r15"] == 0.0
    assert summarize_f1([])["claim_ceiling"] == "NOT_YET_TESTABLE" and summarize_f1([])["e_r15"] is None
    assert summarize_f1(full)["i_to_s"] == "DESCRIPTIVE_ONLY / CAUSALITY_UNESTABLISHED"
    assert "PASS" not in json.dumps(summarize_f1(full))


def test_the_structural_counts_are_compared_exactly() -> None:
    expected = dict(prereg.EXPECTED_STRUCTURAL_COUNTS)
    assert compare_structural(dict(expected))["equal"] is True
    wrong = {**expected, "intents": 2, "outbox_messages": 0}
    result = compare_structural(wrong)
    assert result["equal"] is False and set(result["differing"]) == {"intents", "outbox_messages"}
    assert compare_structural({})["equal"] is False


def test_the_scripted_status_answers_cover_the_four_classes() -> None:
    for status in ("PRESENT", "CONTESTED", "UNKNOWN", "CONTRADICTORY"):
        assert len(s2_status_script(status)) == 5
    assert refusal_code(s2_status_script, "NOPE") == "F1_STATUS_UNKNOWN"


# -------------------------------------------------------------- F1 real guard


def test_f1_refuses_a_real_model_until_the_human_released_stage_b(r20_base: Path) -> None:
    scripted_ctx = scripted(r20_base)
    guard_f1(scripted_ctx)  # the scripted dry run needs nothing
    ctx = real_context(r20_base)
    assert refusal_code(guard_f1, ctx) == "STAGE_B_NOT_RELEASED"
    assert refusal_code(drivers.run_f1_pair, ctx, {"kind": "TREATMENT", "pair": 1, "first_branch": "NEUTRAL", "seed": 1}, {}) == "STAGE_B_NOT_RELEASED"
    plan = prereg.derive_f1_plan(1, "t")
    assert refusal_code(drivers.run_f1, ctx, plan, lambda _p: {}) == "STAGE_B_NOT_RELEASED"
    (r20_base / drivers.STAGE_B_RELEASE).write_text(json.dumps({"prereg_sha256": "c" * 64, "released_by": "human"}))
    import os

    os.chmod(r20_base / drivers.STAGE_B_RELEASE, 0o600)
    assert stage_b_released(r20_base, "c" * 64)
    assert refusal_code(guard_f1, ctx) == "IP_R15B_NOT_MERGED"  # the arm template is not registered at this head
    assert [p.name for p in r20_base.iterdir()] == [drivers.STAGE_B_RELEASE]  # no subject was created


def test_f1_stays_refused_while_the_arm_template_is_not_wired(r20_base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (r20_base / drivers.STAGE_B_RELEASE).write_text(json.dumps({"prereg_sha256": "c" * 64, "released_by": "human"}))
    import os

    os.chmod(r20_base / drivers.STAGE_B_RELEASE, 0o600)
    monkeypatch.setattr(drivers, "S2_TEMPLATES", {drivers.IP_R15B_TEMPLATE: object()})
    assert refusal_code(guard_f1, real_context(r20_base)) == "F1_ARM_NOT_WIRED"


# ----------------------------------------------------------- scripted dry runs


def test_g1_scripted_dry_run_reports_the_gate_and_the_measurements(r20_base: Path) -> None:
    ctx = scripted(r20_base, budget=Session(Ledger(r20_base), Caps(), "L", "G1"))
    outcome = drivers.run_g1(ctx)
    assert outcome["label"] == "GATE_PASSED" and outcome["x_not_skipped_as_neutral"] is True and outcome["u2a_review_events"] >= 1
    assert outcome["serving_after_s"] > 0 and len(outcome["turn_seconds"]) >= 3 and outcome["gpu"] == {}  # no GPU log for a fixture
    manifest = manifest_of(outcome, r20_base)
    assert manifest["fixture_only"] is True and manifest["preregistration_sha256"] == "c" * 64
    assert manifest["real_run"]["label"] == "GATE_PASSED"
    assert Ledger(r20_base).subjects_used("G1") == 1


def test_s0_block_scripted_reaches_s0_with_the_kill_the_structural_counts_and_the_negative_variant(r20_base: Path) -> None:
    ctx = scripted(r20_base, budget=Session(Ledger(r20_base), Caps(), "L", "S0"))
    block = drivers.run_s0_block(ctx, negative_wait_s=2.0)
    assert block["s0"] == "reached" and len(block["attempts"]) == 1  # the first attempt qualifies: no further attempt
    attempt = block["attempts"][0]
    assert attempt["qualifies_as_s0"] and attempt["label"] == "S0_REACHED_CANDIDATE" and not attempt["problems"]
    selector = attempt["first_row4"]
    assert (selector["row"], selector["interpretation"], selector["stage"], selector["qualifies"]) == ("4", "SETTLED", "S2", True)
    assert attempt["kill"]["point"] == "K4" and attempt["kill"]["window_hit"] and attempt["kill"]["window_confirmed"]
    assert len(attempt["kill"]["pids"]) == 2 and attempt["kill"]["pids"][0] != attempt["kill"]["pids"][1]  # one SIGKILL, one restart
    counted = attempt["step9_structural"]
    assert counted["equal"] and counted["counts"] == prereg.EXPECTED_STRUCTURAL_COUNTS
    assert attempt["ablation"]["pair"] == ["5", "6"] and set(attempt["steps"].values()) == {"DONE"}
    negative = block["negative_variant"]
    assert negative["label"] == "VARIANT_HELD" and negative["zero_claims"] and negative["target_unchanged"] and negative["subject_running"]
    assert negative["row_2c_confirmed"] is None  # the dry run did not wait the >= 600 s window: T1 F05-w covers the close
    manifest = manifest_of(attempt, r20_base)
    assert manifest["approval_actor"] == "SCRIPTED" and manifest["fixture_only"] is True and manifest["admission_path"] == "HARNESS_BRIDGE"
    assert manifest["preregistration_sha256"] == "c" * 64 and len(manifest["kill_points"]) == 1
    assert {entry["step"] for entry in manifest["steps"]} == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    text = json.dumps(manifest)
    for raw in ("synthetic visible answer", "比較用の案"):  # the closed key set holds no text
        assert raw not in text
    assert Ledger(r20_base).subjects_used("S0") == 2  # the attempt and the negative variant


def test_the_k3_rule_blocks_s0_after_three_non_settled_attempts_and_reports_each(r20_base: Path) -> None:
    ctx = scripted(r20_base, budget=Session(Ledger(r20_base), Caps(), "L", "S0"))
    block = drivers.run_s0_block(ctx, script=s2_status_script("CONTESTED"))
    assert block["s0"] == "BLOCKED" and "negative_variant" not in block  # no further subject is spent
    assert [a["attempt"] for a in block["attempts"]] == [1, 2, 3]  # at most three, no extra attempt
    for attempt in block["attempts"]:
        selector = attempt["first_row4"]
        assert selector["row"] == "4b" and selector["interpretation"] == "UNCERTAIN" and not attempt["qualifies_as_s0"]
        assert "row 4b" in attempt["not_s0_because"]  # reported with why
    assert Ledger(r20_base).subjects_used("S0") == 3


def test_s1_gate_g2_blocks_s1_when_the_salience_is_below_the_threshold(r20_base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drivers, "S1_SALIENCE_MIN", 0.95)
    block = drivers.run_s1_block(scripted(r20_base))
    assert block["s1"] == "NOT_YET_TESTABLE" and block["attempts"] == [] and block["g2"]["salience"] == pytest.approx(0.8)


def test_s1_scripted_dry_run_reports_the_observed_result_against_the_expectation(r20_base: Path) -> None:
    ctx = scripted(r20_base, budget=Session(Ledger(r20_base), Caps(), "L", "S1"))
    block = drivers.run_s1_block(ctx)
    assert block["g2"]["label"] == "GATE_PASSED" and block["g2"]["salience"] >= 0.5 and block["s1"] == "OBSERVED"
    attempt = block["attempts"][0]
    assert attempt["activation"] == {"driver": 200, "self_goal": 200}
    assert attempt["self_goal_decision"] and attempt["goal_wake"] and attempt["cycle_closed"]
    assert attempt["no_user_turn_before_the_wake_review"] is True
    assert attempt["resolution_kinds"] == ["EXISTING_DECISION"]  # the row-3 Decision exists: the same fingerprint, no ABANDON path
    assert attempt["wake_outcome"]["output"] == "PLAN_A" and attempt["matches_expectation"] is True
    assert attempt["user_request_on_x"]["reason"] == "self_goal_occupies_target" and attempt["self_goal_occupies_target"] is True
    assert Ledger(r20_base).subjects_used("G2") == 1 and Ledger(r20_base).subjects_used("S1") == 1


def test_f1_pair_driver_scripted_dry_run_with_a_sealed_contradictory_member_and_a_control(r20_base: Path) -> None:
    ctx = scripted(r20_base, budget=Session(Ledger(r20_base), Caps(), "L", "F1"))
    plan = prereg.derive_f1_plan(7, "dry")

    def statuses(pair: dict[str, Any]) -> dict[str, str | None]:
        return {"NEUTRAL": "PRESENT", "UNCERTAINTY": "CONTRADICTORY" if pair["pair"] == 1 else "PRESENT"}

    run = drivers.run_f1(ctx, plan, statuses, only=[1, 12])
    by_pair = {p["pair"]: p for p in run["pairs"]}
    treatment, control = by_pair[1], by_pair[12]
    assert treatment["kind"] == "TREATMENT" and treatment["class"] == "VALID", treatment
    rows = {m["arm"]: m["sealed_row"] for m in treatment["members"]}
    assert rows == {"NEUTRAL": "4", "UNCERTAINTY": "2b"} and treatment["discordant"] is True
    assert all(m["action_intents"] == 0 for m in treatment["members"])  # r18 is off in F1
    assert any(m.get("hold_sealed") and m["receipts"] for m in treatment["members"])  # sealed by a store receipt, not a log copy
    assert control["kind"] == "CONTROL_UNCERTAINTY" and control["class"] == "VALID" and control["discordant"] is False
    order = [m["arm"] for m in treatment["members"]]
    first = next(p for p in plan["pairs"] if p["pair"] == 1)["first_branch"]
    assert order[0] == first  # the pre-registered branch order
    summary = run["summary"]
    assert summary["claim_ceiling"] == "PARTIAL_VALID_EVIDENCE" and summary["planned"] == 12 and summary["pairs"] == 2
    assert Ledger(r20_base).subjects_used("F1") == 4
