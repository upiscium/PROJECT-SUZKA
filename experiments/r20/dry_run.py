"""The deterministic S0 dry run (IP-22d; IP-22 D1 r3 sec. 7): ``python -m experiments.r20.dry_run``.

It drives S0 steps 1-9 on NEW genesis subjects on the PROCESS tier with the
SCRIPTED provider and the scripted approval operator, and writes a manifest in
which every checklist item is labelled ``DONE`` / ``NOT_DONE`` / ``PENDING``
with the evidence ids it rests on.  It claims no R20 result and no real-model
evidence: ``fixture_only`` is true, the R15 S1 / S2 trigger is the harness
(``HARNESS_BRIDGE``; the admission itself is the real gate and owner), and the
approval is ``SCRIPTED``.

Subjects (each is a NEW genesis subject; none is reused):

* ``A`` the reference S0 run: items 1-9, 11, 12 (no kill);
* ``B`` the same chain with ONE SIGKILL (default K4) and a restart: item 10, its
  logical counts compared with ``A``;
* ``C`` a separate-subject negative variant: SIGKILL at K2 (item 6, D2-6);
* ``D`` an F1-profile subject whose S2 answer is CONTRADICTORY: item 15, the
  2b plan-hold receipt (written once; IP-14b-3 is merged);
* ``E`` the same without the receipt switch (the eve shape): writes none.

Items 13 (S1) and 14 (K = 3 pre-registration) are NOT run here: item 13 needs the
S1 profile with the self-goal activation (a separate subject type, IP-23's S1),
item 14 is the order / template the Human pre-registers before the first real run.
They are reported as such, never as passed.  T2 runs on the real wall clock:
the provider is deterministic, the kill window is not; nothing here is a T1
"covered" claim.  The F12-r tail of item 10 (no stale-branch action after a true
rollback) is the F12 row of the F-matrix (T1), not re-run here.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from experiments.r20 import factory
from experiments.r20.ablation import (
    ablate_feedback,
    interpretation_variants,
    replay_defect,
)
from experiments.r20.guards import GuardRefusal
from experiments.r20.kill import logical_counts, replay_all, run_kill_point
from experiments.r20.manifest import HOLD_SEALED, HOLD_UNSEALED
from experiments.r20.provider import NON_NEUTRAL_LOSS, ApprovalActor, ProviderMode
from experiments.r20.s0 import FINALIZED, S0Session
from suzka.r16.successor_policy import (
    FeedbackState,
    Interpretation,
    InterpretationStage,
    SuccessorPolicyInputV2,
    SuccessorRow,
)

STAMP = "%Y-%m-%dT%H:%M:%SZ"
STEP_NAMES = {
    1: "S0 step 1: R15 S1 then S2 on X / E2",
    2: "S0 step 2 order: R15 S2 terminal before the U2c acceptance",
    3: "S0 step 3: the first row-4 Decision is SETTLED / S2",
    4: "DRV-1 D: X is not skipped as neutral",
    5: "S0 step 4: Intent, forecast, Policy, approval, CLAIM, effect",
    6: "S0 step 5: independent verification (and a separate-subject negative variant)",
    7: "S0 step 6: R17 MATCH_SELF Feedback",
    8: "S0 step 7: FEEDBACK_ADMITTED Wake without a user turn, Outbox, ACK",
    9: "S0 step 8: row 5 and the ablation",
    10: "S0 step 9: one SIGKILL and a restart",
    11: "timing (#373 A4-4)",
    12: "Journal budget",
    13: "S1 (self-goal subject)",
    14: "K = 3 pre-registration and report",
    15: "2b plan-hold receipt (IP-14b-3)",
}


class Step:
    """One checklist item being filled; ``fail`` marks an attempted item that
    did not hold (it is then NOT_DONE and the command exits non-zero)."""

    def __init__(self) -> None:
        self.evidence: dict[str, Any] = {}
        self.problems: list[str] = []

    def require(self, name: str, condition: bool, detail: Any = None) -> None:
        self.evidence[name] = detail if detail is not None else condition
        if not condition:
            self.problems.append(name)


# ------------------------------------------------------------------ helpers


def _sealed(row_bytes: bytes) -> SuccessorPolicyInputV2:
    return SuccessorPolicyInputV2.from_bytes(row_bytes)


def _utc(text: str) -> datetime:
    return datetime.strptime(text, STAMP).replace(tzinfo=timezone.utc)


def _try(step: Step, name: str, function: Callable[[], Any]) -> Any:
    try:
        return function()
    except Exception as error:  # an attempted item that failed is reported, never swallowed
        step.problems.append(f"{name}: {type(error).__name__}: {str(error)[:200]}")
        return None


# ------------------------------------------------------------------ run A


def reference_run(base: Path) -> tuple[S0Session, dict[str, Any], dict[str, Step]]:
    """Subject A: items 1-9, 11, 12.  Returns the session (still running),
    the raw facts and one ``Step`` per item."""

    steps: dict[str, Step] = {str(n): Step() for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)}
    session = S0Session(base)
    try:
        return _reference_body(session, steps)
    except BaseException:
        session.stop()
        raise


def _reference_body(session: S0Session, steps: dict[str, Step]) -> tuple[S0Session, dict[str, Any], dict[str, Step]]:
    session.start()
    facts: dict[str, Any] = {}
    chain = _try(steps["1"], "chain", session.drive_to_claim)
    if chain is None:
        return session, facts, steps
    facts["chain"] = chain
    finished = bool(session.wait_for(session.finished, 150, 0.5))
    facts["finished"] = finished
    evidence = session.evidence
    decisions = evidence.decisions()
    facts["decisions"] = decisions
    row3 = next((d for d in decisions if d.row == "3"), None)
    row4 = next((d for d in decisions if d.row == "4"), None)
    cycle = next((d for d in decisions if d.operation_kind == "R16_CYCLE_DECISION"), None)

    # ---- 1: R15 S1 / S2, provider and revision ids, the structural link, the limitation analysis
    one = steps["1"]
    reviews = chain["r15"]
    one.require("s1_revision_1", reviews["S1"]["revision"] == 1 and reviews["S1"]["changed"] is True, reviews["S1"])
    one.require("s2_revision_2", reviews["S2"]["revision"] == 2 and reviews["S2"]["changed"] is True, reviews["S2"])
    one.require("admission_path_label", reviews["S1"]["path"] == "HARNESS_BRIDGE" == reviews["S2"]["path"], "HARNESS_BRIDGE")
    if row4 is not None:
        sealed4 = _sealed(row4.sealed_input)
        experience_refs = [ref.ref_id for ref in sealed4.source_refs if ref.kind == "experience"]
        one.require(
            "common_input_x_event",
            session.x in experience_refs and len(experience_refs) == 1,
            {"x_experience_id": session.x, "row4_source_experience_refs": experience_refs},
        )
        variants = interpretation_variants(row4.sealed_input)
        facts["limitation_variants"] = [
            {"interpretation": v.interpretation.value, "stage": v.stage.value, "row": v.row.value, "output": v.output.value}
            for v in variants
        ]
        one.evidence["limitation_analysis_variants"] = facts["limitation_variants"]
    else:
        one.problems.append("no row-4 Decision")
    one.evidence["provider"] = factory.read_identity_record(session.subject.subject_json)["provider"]
    one.evidence["provider_script_sha256"] = _script_digest(session)

    # ---- 2: order (the journal sequence of the second narrative review precedes the goal adoption)
    two = steps["2"]
    events = evidence.journal_events()
    reviews_seq = [e["sequence"] for e in events if e["event_type"] == "narrative_review"]
    adoption_seq = [e["sequence"] for e in events if e["event_type"] == "goal_adoption_review"]
    acceptance_seq = [e["sequence"] for e in events if e["event_type"] == "commitment_acceptance_review"]
    two.require("two_narrative_reviews", len(reviews_seq) == 2, reviews_seq)
    two.require(
        "s2_terminal_before_goal_adoption_and_acceptance",
        bool(reviews_seq and adoption_seq and acceptance_seq)
        and max(reviews_seq) < min(adoption_seq) < min(acceptance_seq),
        {"narrative_review": reviews_seq, "goal_adoption": adoption_seq, "commitment_acceptance": acceptance_seq},
    )
    if row4 is not None:
        stage = _sealed(row4.sealed_input).interpretation_stage
        two.require("row4_stage_is_S2", stage is InterpretationStage.S2, stage.value)
    else:
        two.problems.append("no row-4 Decision")

    # ---- 3: the first row-4 Decision
    three = steps["3"]
    three.require("row3_precedes_row4", row3 is not None and row4 is not None and row3.sequence < row4.sequence)
    if row3 is not None and row4 is not None:
        three.require("row4_predecessor_is_the_row3_decision", row4.predecessor_key == row3.decision_id,
                      {"row3": row3.decision_id, "row4": row4.decision_id})
        sealed4 = _sealed(row4.sealed_input)
        three.require(
            "settled_s2",
            sealed4.interpretation is Interpretation.SETTLED and sealed4.interpretation_stage is InterpretationStage.S2,
            {"interpretation": sealed4.interpretation.value, "stage": sealed4.interpretation_stage.value},
        )
        three.evidence["reading"] = "the first row-4 Decision (HD-22-10 (a)); the row-3 Decision comes first by design"

    # ---- 4: DRV-1 D
    four = steps["4"]
    interest_reviews = [e for e in events if e["event_type"] == "motivation_source_review"]
    four.require("the_u2a_interest_review_ran", len(interest_reviews) >= 1, [e["sequence"] for e in interest_reviews])
    if row3 is not None:
        refs3 = _sealed(row3.sealed_input).source_refs
        four.require("the_goal_decision_names_x_and_a_goal",
                     any(r.kind == "experience" and r.ref_id == session.x for r in refs3)
                     and any(r.kind == "goal" for r in refs3), [(r.kind, r.ref_id[:24]) for r in refs3])
    else:
        four.problems.append("no row-3 Decision")
    four.evidence["provider_loss"] = NON_NEUTRAL_LOSS
    four.evidence["note"] = "the Goal was adopted on X by REVIEW_EXPERIENCE, which needs the INTEREST Motivation of X (the U2a review event ran): X was not skipped as neutral"

    # ---- 5: Intent, forecast, Policy, approval, claim, effect
    five = steps["5"]
    counters = session.counters()
    five.require("one_intent", counters["R18_INTENT_ISSUE"] == 1, counters["R18_INTENT_ISSUE"])
    expectation = evidence.sequences("R17_EXPECTATION_COMMIT")
    claim = evidence.sequences("R18_EXECUTION_CLAIM")
    five.require("forecast_committed_before_the_claim", bool(expectation and claim) and expectation[0] < claim[0],
                 {"forecast_sequence": expectation, "claim_sequence": claim})
    verdicts = evidence.records("policy_verdicts")
    five.require("policy_allow", len(verdicts) == 1 and verdicts[0].get("verdict") == "ALLOW",
                 [v.get("verdict") for v in verdicts])
    five.require("approval_over_http_by_the_scripted_operator", chain["confirm"]["status"] == 200
                 and chain["confirm"]["body"].get("status") == "claimed", chain["confirm"]["body"].get("status"))
    five.evidence["approval_actor"] = ApprovalActor.SCRIPTED.value
    five.require("one_execution_claim", counters["R18_EXECUTION_CLAIM"] == 1, counters["R18_EXECUTION_CLAIM"])
    five.require("one_target_effect", evidence.target_effects() == 1, evidence.target_effects())

    # ---- 6: independent verification (the negative variant is subject C, added later)
    six = steps["6"]
    six.require("active_verified_authorized", evidence.outcome_states() == ["ACTIVE_VERIFIED_AUTHORIZED"], evidence.outcome_states())

    # ---- 7: R17 MATCH_SELF
    seven = steps["7"]
    feedback = evidence.records("decision_feedback")
    seven.require("one_feedback", len(feedback) == 1, len(feedback))
    if cycle is not None:
        seven.require("feedback_is_match_self", _sealed(cycle.sealed_input).feedback is FeedbackState.MATCH_SELF,
                      _sealed(cycle.sealed_input).feedback.value)
    else:
        seven.problems.append("no wake-origin Decision to read the Feedback from")
    seven.evidence["note"] = (
        "D-R17: in V1 an authorized, verified effect implies MATCH_SELF, so MATCH_SELF is not independent "
        "evidence that the forecast was right"
    )

    # ---- 8: Wake without a user turn, Outbox, ACK
    eight = steps["8"]
    causes = evidence.wake_causes()
    eight.require("causes", "OUTCOME_VERIFIED_NO_FEEDBACK" not in causes and causes.count("FEEDBACK_ADMITTED") == 1, causes)
    feedback_seq = evidence.sequences("R17_FEEDBACK_COMMIT")
    wake_review = [e["sequence"] for e in events if e["event_type"] == "r16_wake_review"]
    eight.require("control_chat_events_are_visible_in_the_journal",
                  sum(1 for e in events if e["event_type"] == "chat") >= 5, sum(1 for e in events if e["event_type"] == "chat"))
    chat_between = [
        e["sequence"] for e in events
        if e["event_type"] == "chat" and feedback_seq and wake_review and feedback_seq[0] < e["sequence"] < wake_review[0]
    ]
    eight.require("no_chat_turn_between_feedback_and_the_decision", bool(feedback_seq and wake_review) and not chat_between,
                  {"feedback_sequence": feedback_seq, "wake_review_sequence": wake_review})
    claimed_cause = _claimed_wake_cause(evidence)
    eight.require("the_claimed_wake_is_feedback_admitted", claimed_cause == "FEEDBACK_ADMITTED", claimed_cause)
    eight.require("one_outbox_message", evidence.table_count("outbox_messages") == 1, evidence.table_count("outbox_messages"))
    ack = _try(eight, "ack", lambda: _ack_through_the_real_route(session))
    if ack is not None:
        eight.evidence["ack"] = ack
        eight.require("ack_row", ack["ack_status"] in (200, 201) and evidence.table_count("outbox_acks") == 1,
                      {"status": ack["ack_status"], "rows": evidence.table_count("outbox_acks")})
    eight.evidence["wake_causes_in_storage_order"] = causes

    # ---- 9: row 5 and the ablation
    nine = steps["9"]
    if cycle is not None:
        nine.require("wake_origin_decision_is_row_5", cycle.row == "5", cycle.row)
        result = ablate_feedback(cycle.sealed_input)
        pair = [row.value for row in result.pair]
        nine.require("ablation_pair_5_6", result.is_s0_ablation and result.s0_pair_defect() is None and pair == ["5", "6"], pair)
        nine.require("replay_of_the_sealed_input_equal", replay_defect(cycle.sealed_input, SuccessorRow.ROW_5) is None)
        nine.require("outputs", result.factual_write_eligible is False and result.ablated_write_eligible is False,
                     {"factual": result.factual_output.value, "ablated": result.ablated_output.value})
        facts["ablation"] = {"pair": pair, "factual_output": result.factual_output.value,
                             "ablated_output": result.ablated_output.value, "label": result.label.value}
        resolutions = evidence.table_count("logical_cycle_resolutions", f"decision_id = '{cycle.decision_id}'")
        nine.require("resolution_row_exported_with_the_decision", resolutions == 1, resolutions)
        nine.evidence["cause_label"] = "first eligible unwoken cause (IP-21 D1 12-1)"
    else:
        nine.problems.append("no wake-origin Decision")

    # ---- 11: timing, computed from the real settings and the observed times
    eleven = steps["11"]
    settings = session.subject.resolved_settings()
    delay, tick, lease = settings.r19.feedback_wake_delay_s, settings.r19.tick_s, settings.r19.claim_lease_s
    eleven.require("settings", delay + tick <= lease, {"feedback_wake_delay_s": delay, "tick_s": tick, "claim_lease_s": lease})
    approval = chain["approval"]
    window = settings.r18.approval_window_seconds
    eleven.require(
        "approval_window_end_within_the_u2c_deadline",
        _utc(approval["window_end_utc"]) <= _utc(approval["commitment_deadline_utc"]),
        {"approval_window_end_utc": approval["window_end_utc"], "commitment_deadline_utc": approval["commitment_deadline_utc"],
         "approval_window_seconds": window},
    )
    if cycle is None or row4 is None:
        eleven.problems.append("no wake-origin Decision to time")
    else:
        deadline = _utc(approval["commitment_deadline_utc"])
        eleven.require("the_cycle_decision_precedes_the_u2c_deadline", _utc(_sealed(cycle.sealed_input).now_utc) < deadline,
                       _sealed(cycle.sealed_input).now_utc)
    eleven.require("approval_window_at_least_600_s", window >= 600, window)
    eleven.evidence["wake_expiry_s"] = 604_800
    eleven.evidence["note"] = "WakeExpiry is the 7-day Wake TTL (a constant of the contract), compared only by the "\
        "cycle Decision time being inside the U2c deadline"

    # ---- 12: the Journal budget
    twelve = steps["12"]
    counts = evidence.journal_counts()
    limit = int(settings.event_journal.max_total_bytes or 0)
    soft = int(settings.event_journal.soft_band_bytes or 0)
    soft_start = limit - soft
    facts["journal"] = {**counts, "max_total_bytes": limit, "soft_band_start_bytes": soft_start,
                        "share_of_soft_band_start": round(counts["live_bytes"] / soft_start, 6)}
    twelve.evidence.update(facts["journal"])
    twelve.require("run_stays_below_the_soft_band", counts["live_bytes"] < soft_start,
                   {"live_bytes": counts["live_bytes"], "soft_band_start_bytes": soft_start})
    twelve.evidence["finding_if_crossed"] = "a run that crosses the soft band is a finding (HP-8)"
    facts["identity"] = asdict(factory.verify_first_start(session.subject, provider={"mode": "scripted", "fixture_only": True}))
    facts["reference_counts"] = logical_counts(session)
    facts["replay_defects"] = replay_all(evidence)
    return session, facts, steps


def _script_digest(session: S0Session) -> str:
    from experiments.r20.provider import DEFAULT_ANSWER, script_digest

    config = session.provider_config
    return script_digest(config["script"], config.get("default", DEFAULT_ANSWER), NON_NEUTRAL_LOSS)


def _claimed_wake_cause(evidence: Any) -> str | None:
    from experiments.r20.polling import short_read

    rows = short_read(
        evidence.history,
        "SELECT w.record_bytes FROM wake_claims c JOIN wakeups w ON w.record_id = c.wakeup_id ORDER BY c.rowid LIMIT 1",
    )
    return None if not rows else str(json.loads(bytes(rows[0][0])).get("cause"))


def _ack_through_the_real_route(session: S0Session) -> dict[str, Any]:
    """GET the Inbox of the current Context, then POST the ACK, both over HTTP to
    the harness child (destination-bound)."""

    contexts = session.request("GET", "/api/contexts").json()
    context_id = str(contexts["current_context_id"])
    inbox = session.request("GET", f"/api/r19/inbox/messages?context_id={context_id}").json()
    messages = inbox["messages"]
    if len(messages) != 1:
        raise GuardRefusal("INBOX_NOT_ONE_MESSAGE", str(len(messages)))
    message = messages[0]
    reply = session.request(
        "POST",
        f"/api/r19/inbox/messages/{message['message_id']}/ack",
        {"context_id": context_id, "status": "ACKNOWLEDGED"},
    )
    return {
        "message_id": message["message_id"],
        "template": message["template"],
        "state_before": message["state"],
        "ack_status": reply.status,
    }


# --------------------------------------------------------------- the others


def kill_run(base: Path, point: str) -> tuple[Step, dict[str, Any]]:
    """One separate subject killed at ``point``.  The comparison with the no-kill
    reference is made by the caller (the reference runs concurrently)."""

    step = Step()
    entry: dict[str, Any] = {"point": point}
    try:
        result, session = run_kill_point(base, point)
    except Exception as error:  # an attempted item that failed is reported
        step.problems.append(f"{point}: {type(error).__name__}: {str(error)[:200]}")
        return step, entry
    try:
        step.require("kill_window_hit", result.window_hit, {"point": point})
        step.evidence["window_confirmed_on_the_frozen_store"] = result.window_confirmed
        step.evidence["pids"] = list(result.pids)
        for name, held in result.checks.items():
            step.require(name, held)
        entry = {**result.manifest_entry(), "logical_counts": logical_counts(session)}
        return step, entry
    except Exception as error:
        step.problems.append(f"{point}: {type(error).__name__}: {str(error)[:200]}")
        return step, entry
    finally:
        session.stop()


def receipt_runs(base: Path) -> tuple[Step, dict[str, Any]]:
    """Item 15: exactly one 2b receipt for a CONTRADICTORY F1-profile subject;
    none with the switch off (the eve shape)."""

    step = Step()
    out: dict[str, Any] = {}
    for name, profile in (("with_receipts", "F1"), ("receipts_off", "NEG-r15")):
        session = S0Session(base, profile=profile, contradictory=True)
        try:
            session.start()
            session.activate()
            session.run_r15_protocol()
            session.adopt_goal()  # the Goal adoption's plan preview is the (only) HOLD under test
            session.wait_for(lambda: session.evidence.counts().get(("R16_PLAN_HOLD", FINALIZED), 0) >= 1, 20, 0.2)
            time.sleep(3)  # a second receipt for the same evidence would show now
            evidence = session.evidence
            receipts = evidence.records("plan_hold_receipts")
            out[name] = {"receipts": len(receipts), "decisions": session.counters()["R16_DECISION_ISSUE"]}
            previews = [e for e in evidence.journal_events() if e["event_type"] == "plan_preview_review"]
            step.require(f"{name}_the_plan_preview_ran", len(previews) >= 1, len(previews))
            if profile == "F1":
                step.require("exactly_one_receipt", len(receipts) == 1, len(receipts))
                if receipts:
                    receipt = receipts[0]
                    sealed = _sealed_from_receipt(receipt)
                    step.require("sealed_row_2b", receipt.get("policy_row") == "2b", receipt.get("policy_row"))
                    step.require("sealed_input_is_contradictory_s2",
                                 sealed.interpretation is Interpretation.CONTRADICTORY
                                 and sealed.interpretation_stage is InterpretationStage.S2)
                    step.require("receipt_input_replays_to_row_2b", replay_defect(
                        bytes.fromhex(str(receipt["policy_input_bytes"])) if isinstance(receipt.get("policy_input_bytes"), str)
                        else sealed.canonical_bytes(), SuccessorRow.ROW_2B) is None)
                    step.evidence["receipt_id"] = receipt.get("hold_id")
                    out["receipt_id"] = receipt.get("hold_id")
                step.require("a_hold_issues_no_decision", session.counters()["R16_DECISION_ISSUE"] == 0)
                # OBSERVATION (not an acceptance item): a DIFFERENT command is different evidence
                session.accept_commitment()
                session.wait_for(lambda: len(evidence.records("plan_hold_receipts")) >= 2, 20, 0.2)
                time.sleep(2)
                both = evidence.records("plan_hold_receipts")
                keys = [item.get("review_intent_key") for item in both]
                step.evidence["observation_after_commit_review_experience"] = {
                    "receipts": len(both),
                    "distinct_review_keys": len(set(keys)) == len(keys),
                    "commitment_state_of_each": [item.get("policy_input", {}).get("commitment", {}).get("state") for item in both],
                    "reading": "COMMIT_REVIEW_EXPERIENCE is different evidence (the Commitment exists): its own HOLD, its own "
                               "receipt; not a replay of the same evidence",
                }
            else:
                step.require("none_with_the_switch_off", len(receipts) == 0 and session.counters()["R16_PLAN_HOLD"] == 0, len(receipts))
                step.require("the_hold_still_issued_no_decision", session.counters()["R16_DECISION_ISSUE"] == 0)
        except Exception as error:
            step.problems.append(f"{name}: {type(error).__name__}: {str(error)[:200]}")
        finally:
            session.stop()
    step.evidence["replay_of_the_same_evidence"] = (
        "merged T1 test tests/test_r16_ip14b3_plan_hold_writer.py::test_the_2b_hold_is_sealed_exactly_once "
        "(the SAME evidence again: no second receipt) and is NOT re-run in this subject; the chain cannot re-send "
        "the same evidence over HTTP "
        "(a second REVIEW_EXPERIENCE after the adoption is a hold and issues no preview)"
    )
    return step, out


def _sealed_from_receipt(receipt: dict[str, Any]) -> SuccessorPolicyInputV2:
    """The receipt carries its sealed input as closed JSON (``policy_input``)."""

    from suzka.r16.cycle_contracts import PlanHoldReceiptV1  # noqa: F401  (the closed record type)

    raw = receipt.get("policy_input_bytes")
    if isinstance(raw, str):
        return SuccessorPolicyInputV2.from_bytes(bytes.fromhex(raw))
    # the stored JSON object: re-encode canonically and parse with the closed schema
    encoded = json.dumps(receipt["policy_input"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return SuccessorPolicyInputV2.from_bytes(encoded)


# ------------------------------------------------------------------- driver


NOT_RUN_NOTES = {
    "13": "NOT RUN in IP-22d: the S1 profile (self-goal activation) is a separate subject type of IP-23; nothing was driven",
    "14": "PENDING: the order of up to three runs and the fixed template are pre-registered by the Human before the "
          "first real run (IP-23); there is no real run to report",
}


def apply_labels(manifest: Any, steps: dict[str, Step]) -> dict[int, list[str]]:
    """Label every checklist item.  ``DONE`` needs evidence and no problem; an
    attempted item that did not hold, or was never reached, is ``NOT_DONE`` and is
    returned in the problem map (the command then exits non-zero); items 13 and 14
    are not run here by design (``NOT_DONE`` / ``PENDING``, ``attempted: false``)."""

    problems: dict[int, list[str]] = {}
    for number in range(1, 16):
        key = str(number)
        if key in ("13", "14"):
            label = "NOT_DONE" if key == "13" else "PENDING"
            manifest.step(number, STEP_NAMES[number], label, {"attempted": False}, NOT_RUN_NOTES[key])
            continue
        step = steps.get(key)
        if step is None or (not step.evidence and not step.problems):
            manifest.step(number, STEP_NAMES[number], "NOT_DONE", {"attempted": False}, "not reached")
            problems[number] = ["not reached"]
            continue
        note = None
        if key == "10":
            note = "the F12-r tail (no stale-branch action after a true rollback) is the F12 row of the F-matrix (T1)"
        if step.problems:
            problems[number] = list(step.problems)
            manifest.step(number, STEP_NAMES[number], "NOT_DONE",
                          {**step.evidence, "attempted": True, "failed": list(step.problems)}, note)
        else:
            manifest.step(number, STEP_NAMES[number], "DONE", {**step.evidence, "attempted": True}, note)
    return problems


def run_dry_run(base: Path, *, kill_point: str = "K4") -> tuple[dict[str, Any], int, Path]:
    """Run everything; returns the manifest dict, the exit code (0 = no
    attempted item failed; 1 = an attempted item failed) and the manifest path.

    The subjects (A reference, B killed at ``kill_point``, C killed at K2,
    D / E receipts, the last two one after the other) are independent genesis
    subjects on their own ports and roots; four groups run CONCURRENTLY to keep the command short.  Nothing is shared."""

    from concurrent.futures import ThreadPoolExecutor

    if kill_point == "K1":
        raise GuardRefusal("K1_IS_T1_ONLY", "K1 has no T2 form")
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="r20-dry") as pool:
        future_a = pool.submit(reference_run, base)
        future_b = pool.submit(kill_run, base, kill_point)
        future_c = pool.submit(kill_run, base, "K2")
        future_d = pool.submit(receipt_runs, base)
        session, facts, steps = future_a.result()
        kill_step, kill_entry = future_b.result()
        neg_step, neg_entry = future_c.result()
        receipt_step, receipt_facts = future_d.result()
    session.stop()
    manifest = session.manifest
    reference = facts.get("reference_counts")
    if reference is not None and kill_entry.get("logical_counts") is not None:
        kill_step.require(
            "logical_counts_equal_the_no_kill_reference",
            kill_entry["logical_counts"] == reference,
            {"killed": kill_entry["logical_counts"], "reference": reference},
        )
    else:
        kill_step.problems.append("no counts to compare with the no-kill reference")
    manifest.append("kill_points", kill_entry)
    manifest.append("restarts", {"point": kill_point, "pids": kill_entry.get("pids")})
    manifest.append("kill_points", neg_entry)

    steps["10"] = kill_step
    if "active_verified_authorized" in steps["6"].evidence:  # only on top of the reference's own check
        steps["6"].evidence["negative_variant_subject"] = {
            "variant": "SIGKILL at K2, restart: one effect or none, never two (a separate subject, D2-6)",
            "checks": neg_step.evidence, "problems": neg_step.problems,
        }
    if neg_step.problems:
        steps["6"].problems.append("negative variant failed: " + "; ".join(neg_step.problems))
    steps["15"] = receipt_step
    problems = apply_labels(manifest, steps)

    # the remaining manifest fields of the reference subject
    manifest.set("tier", "T2")
    manifest.set("profile", "S0")
    manifest.set("approval_actor", ApprovalActor.SCRIPTED.value)
    manifest.set("provider", {"mode": ProviderMode.SCRIPTED.value, "fixture_only": True,
                              "script_sha256": _script_digest(session)})
    manifest.set("identity", facts.get("identity"))
    if facts.get("identity"):
        manifest.set("code_head", facts["identity"]["git_head"])
        manifest.set("config_sha256", facts["identity"]["config_sha256"])
    manifest.set("destination", {"port": session.subject.port, "bound_client": True})
    manifest.set("journal", facts.get("journal", {}))
    for row in facts.get("decisions", []):
        sealed = _sealed(row.sealed_input)
        manifest.append("policy_inputs", {
            "decision_id": row.decision_id, "kind": row.operation_kind, "sequence": row.sequence, "row": row.row,
            "sealed_input_digest": row.sealed_digest, "interpretation": sealed.interpretation.value,
            "stage": sealed.interpretation_stage.value, "feedback": sealed.feedback.value,
        })
    if facts.get("ablation") is not None:
        manifest.append("ablation", facts["ablation"])
    for entry in session.evidence.records("logical_cycle_resolutions"):
        manifest.append("logical_cycle_resolutions", {
            "logical_cycle_id": entry.get("logical_cycle_id"), "result_kind": entry.get("result_kind"),
            "decision_id": entry.get("decision_id"),
        })
    receipt_id = receipt_facts.get("receipt_id")
    manifest.append("plan_hold_receipts", {"row": "2b", "outcome": HOLD_SEALED if receipt_id else HOLD_UNSEALED,
                                           "receipt_id": receipt_id, "subject": "D (F1, CONTRADICTORY)"})
    for timeline_event in session.evidence.journal_events():
        manifest.append("timeline", timeline_event)
    for number, label_ in sorted((entry["step"], entry["label"]) for entry in manifest.to_dict()["steps"]):
        manifest.result(f"step_{number:02d}", label_)
    manifest.result("limits", "provider deterministic; kill window non-deterministic; T2 is smoke and is not counted as covered")
    path = manifest.write()
    return manifest.to_dict(), 1 if problems else 0, path


def format_report(data: dict[str, Any]) -> str:
    lines = [f"S0 dry run  run_id={data['run_id']}  tier={data['tier']}  fixture_only={data['fixture_only']}"]
    for entry in data["steps"]:
        mark = entry["label"]
        extra = ""
        if entry["evidence"].get("failed"):
            extra = "  FAILED: " + "; ".join(entry["evidence"]["failed"])[:300]
        lines.append(f"  {entry['step']:>2} {mark:<8} {entry['name']}{extra}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.r20.dry_run")
    parser.add_argument("--base", help="an absolute throwaway base directory (default: a new one under /tmp)")
    parser.add_argument("--kill-point", default="K4", choices=["K2", "K3", "K4", "K5", "K6"])
    parser.add_argument("--strict", action="store_true", help="exit 3 unless every item is DONE")
    args = parser.parse_args(argv)
    base = Path(args.base) if args.base else Path(tempfile.mkdtemp(prefix="r20-dry-", dir="/tmp"))
    data, code, path = run_dry_run(base, kill_point=args.kill_point)
    print(format_report(data))
    print(f"manifest: {path}")
    if code:
        return code
    if args.strict and any(entry["label"] != "DONE" for entry in data["steps"]):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
