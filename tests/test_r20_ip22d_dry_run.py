"""IP-22d: the deterministic S0 dry run (IP-22 D1 r3 sec. 7) and its manifest.

The full command drives five NEW genesis subjects on the process tier with the
scripted provider (a reference run, a run killed at K4, a run killed at K2, an
F1-profile receipt run and its receipts-off twin).  The assertions are on the
manifest: every checklist item is labelled DONE / NOT_DONE / PENDING with the
evidence ids it rests on, and an item without evidence or with a failed check is
never DONE.  Nothing here is a real model, a production path or a "covered"
claim: T2 runs on the real clock and the kill window is not deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.r20 import dry_run
from experiments.r20.dry_run import STEP_NAMES, Step, apply_labels, run_dry_run
from experiments.r20.guards import GuardRefusal
from experiments.r20.manifest import STEP_LABELS, Manifest


def test_a_step_is_done_only_with_evidence_and_no_problem(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path, "unit")
    good = Step()
    good.require("a", True, {"id": 1})
    bad = Step()
    bad.require("b", False, 0)
    empty = Step()
    steps = {"1": good, "2": bad, "3": empty}
    problems = apply_labels(manifest, steps)
    labels = {entry["step"]: entry["label"] for entry in manifest.to_dict()["steps"]}
    assert labels[1] == "DONE"
    assert labels[2] == "NOT_DONE" and 2 in problems  # attempted and failed: reported, command exits non-zero
    assert labels[3] == "NOT_DONE" and 3 in problems  # no evidence is not a pass
    assert labels[4] == "NOT_DONE" and labels[13] == "NOT_DONE" and labels[14] == "PENDING"
    assert set(labels.values()) <= set(STEP_LABELS)
    assert set(labels) == set(range(1, 16))
    by_step = {entry["step"]: entry for entry in manifest.to_dict()["steps"]}
    assert by_step[13]["evidence"] == {"attempted": False} and "IP-23" in by_step[13]["note"]
    assert by_step[2]["evidence"]["failed"] == ["b"]


def test_the_manifest_refuses_an_unknown_step_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Manifest(tmp_path, "unit").step(1, "x", "PASS")


def test_every_dry_run_item_of_the_d1_checklist_has_a_name() -> None:
    assert sorted(STEP_NAMES) == list(range(1, 16))


def test_the_dry_run_refuses_k1_which_is_t1_only(r20_base: Path) -> None:
    with pytest.raises(GuardRefusal) as refused:
        run_dry_run(r20_base, kill_point="K1")
    assert refused.value.code == "K1_IS_T1_ONLY"


def _steps(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {entry["step"]: entry for entry in data["steps"]}


def test_the_dry_run_command_labels_every_step_with_evidence(r20_base: Path) -> None:
    data, code, path = run_dry_run(r20_base)
    assert json.loads(path.read_text(encoding="utf-8")) == data  # what is returned is what was written
    steps = _steps(data)
    failed = {number: entry["evidence"].get("failed") for number, entry in steps.items() if entry["evidence"].get("failed")}
    assert code == 0 and not failed, failed
    # items 1-12 and 15 are DONE; 13 (S1) is NOT_DONE and 14 (K = 3) PENDING, by design and never "passed"
    done = {number for number, entry in steps.items() if entry["label"] == "DONE"}
    assert done == set(range(1, 13)) | {15}
    assert steps[13]["label"] == "NOT_DONE" and steps[13]["evidence"] == {"attempted": False}
    assert steps[14]["label"] == "PENDING" and steps[14]["evidence"] == {"attempted": False}
    # every DONE item rests on evidence it names
    for number in done:
        evidence = steps[number]["evidence"]
        assert evidence["attempted"] is True
        assert set(evidence) - {"attempted"}, number

    # the manifest is outside every subject and says what it is
    assert data["fixture_only"] is True and data["tier"] == "T2" and data["profile"] == "S0"
    assert data["admission_path"] == "HARNESS_BRIDGE" and data["approval_actor"] == "SCRIPTED"
    assert data["provider"]["mode"] == "scripted"
    assert "subject" not in path.parts[-3:] and path.parent.name == "evidence"
    assert (path.parents[1] / "subject").is_dir()

    # item 1 / 3: R15 S1 then S2, then the FIRST row-4 Decision is SETTLED / S2
    assert steps[1]["evidence"]["s1_revision_1"]["revision"] == 1 and steps[1]["evidence"]["s2_revision_2"]["revision"] == 2
    assert [v["row"] for v in steps[1]["evidence"]["limitation_analysis_variants"]] == ["4", "4", "4b", "2b"]
    assert steps[3]["evidence"]["settled_s2"] == {"interpretation": "SETTLED", "stage": "S2"}
    rows = [item["row"] for item in data["policy_inputs"]]
    assert rows == ["3", "4", "5"]
    # item 2: the second narrative review precedes the goal adoption and the U2c acceptance
    order = steps[2]["evidence"]["s2_terminal_before_goal_adoption_and_acceptance"]
    assert max(order["narrative_review"]) < min(order["goal_adoption"]) < min(order["commitment_acceptance"])
    # item 5 / 6: one claim, one effect, forecast before the claim, Policy ALLOW, verified
    assert steps[5]["evidence"]["one_target_effect"] == 1 and steps[5]["evidence"]["policy_allow"] == ["ALLOW"]
    assert steps[6]["evidence"]["active_verified_authorized"] == ["ACTIVE_VERIFIED_AUTHORIZED"]
    # item 8 / 9: the Feedback Wake, one Outbox message, the ACK row, the ablation pair
    assert steps[8]["evidence"]["the_claimed_wake_is_feedback_admitted"] == "FEEDBACK_ADMITTED"
    assert steps[8]["evidence"]["ack"]["ack_status"] == 201
    assert data["ablation"][0]["pair"] == ["5", "6"]
    assert len(data["logical_cycle_resolutions"]) == 1
    # item 10 / 6: one SIGKILL (K4) with its counts equal to the reference, and the K2 variant on a separate subject
    killed = steps[10]["evidence"]["logical_counts_equal_the_no_kill_reference"]
    assert killed["killed"] == killed["reference"]
    assert [entry["point"] for entry in data["kill_points"]] == ["K4", "K2"]
    assert all(entry["label"] == "SMOKE_T2_NOT_COUNTED" for entry in data["kill_points"])
    assert steps[6]["evidence"]["negative_variant_subject"]["problems"] == []
    # item 12: the Journal budget is measured and compared with the soft band
    assert data["journal"]["live_bytes"] > 0 and data["journal"]["live_bytes"] < data["journal"]["soft_band_start_bytes"]
    # item 15: exactly one 2b receipt for the CONTRADICTORY F1 subject, none with the switch off
    receipt = steps[15]["evidence"]
    assert receipt["exactly_one_receipt"] == 1 and receipt["sealed_row_2b"] == "2b" and receipt["none_with_the_switch_off"] == 0
    assert data["plan_hold_receipts"][0]["outcome"] == "SEALED" and data["plan_hold_receipts"][0]["receipt_id"]
    # the labels are in the results map too, and nothing claims coverage
    assert data["results"]["step_13"] == "NOT_DONE" and data["results"]["step_14"] == "PENDING"
    assert "not counted" in data["results"]["limits"]
    assert dry_run.main.__module__ == "experiments.r20.dry_run"
