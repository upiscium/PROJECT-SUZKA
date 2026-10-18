"""Synthetic transport checks ONLY. No R15 subject producer or model inference."""

from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from experiments.r15_p0.pilot import Case, canonical_bytes, seal_case
from experiments.r15_p1.protocol import (
    AUTHORITY,
    TrialRefused,
    _paths,
    audit_trial,
    parse_candidate,
    run_trial,
    s1_request,
)


def reply(
    phase: str,
    *,
    referent: str = "A",
    assessment: str = "MAY_HAVE_MISREAD",
    uncertainty: str = "HIGH",
    evidence_steps: list[str] | None = None,
    rationale: str = "自分の解釈は暫定的です",
) -> str:
    return json.dumps(
        {
            "schema": 1,
            "phase": phase,
            "assumed_referent": referent,
            "assessment": assessment,
            "uncertainty": uncertainty,
            "evidence_steps": evidence_steps
            if evidence_steps is not None
            else ["E"]
            if phase == "S1"
            else ["E", "S1", "E2"],
            "rationale": rationale,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def paths(tmp_path: Path, *, case: Case | None = None) -> tuple[Path, Path, Case]:
    scenario = case or Case("p1-candidate", "B", "natural", 0)
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "capture.json"
    seal_case(scenario, manifest)
    return manifest, output, scenario


class FixtureTransport:
    def __init__(
        self,
        output: Path,
        case: Case,
        *,
        s1: str | None = None,
        s2: str | None = None,
        fail_at: int | None = None,
        duplicate_episodes: bool = False,
    ) -> None:
        self.output = output
        self.case = case
        self.fail_at = fail_at
        self.duplicate_episodes = duplicate_episodes
        self.s1 = reply("S1") if s1 is None else s1
        self.s2 = (
            reply(
                "S2",
                referent="A" if case.control == "no_reveal" else "B",
                rationale="E2はユーザーの補足であり客観的証明ではない",
                evidence_steps=["E", "S1"] if case.control == "no_reveal" else None,
            )
            if s2 is None
            else s2
        )
        self.calls: list[str] = []
        self.checkpoint_observed_before_e2 = False

    def __call__(self, message: str) -> dict[str, str]:
        i = len(self.calls)
        self.calls.append(message)
        if i == self.fail_at:
            raise RuntimeError("injected transport failure")
        pre_reveal = ("setup", "E", "S1")
        if i == 3 and self.case.control != "no_reveal":
            stage = _paths(self.output)[1]
            assert stage.exists()
            snapshot = json.loads(stage.read_text(encoding="utf-8"))
            assert [item["step"] for item in snapshot["turns"]] == list(pre_reveal)
            assert snapshot["turns"][2]["response"] == self.s1
            assert snapshot["turns"][1]["response"] == "synthetic visible answer E"
            self.checkpoint_observed_before_e2 = True
        if i == 2:
            answer = self.s1
        elif (self.case.control == "no_reveal" and i == 3) or i == 4:
            answer = self.s2
        elif i == 1:
            answer = "synthetic visible answer E"
        else:
            answer = "synthetic observation"
        return {
            "episode_id": "episode-duplicate"
            if self.duplicate_episodes
            else f"episode-{i}",
            "response": answer,
        }


def run_good(
    tmp_path: Path, *, case: Case | None = None
) -> tuple[Path, Path, Case, FixtureTransport]:
    manifest, output, scenario = paths(tmp_path, case=case)
    transport = FixtureTransport(output, scenario)
    run_trial(manifest, output, transport)
    return manifest, output, scenario, transport


def test_s1_is_physically_recorded_before_transport_observes_e2(tmp_path: Path) -> None:
    manifest, output, case, transport = run_good(tmp_path)
    assert transport.checkpoint_observed_before_e2
    report = audit_trial(manifest, output)
    assert report == {
        "status": AUTHORITY,
        "s1_before_e2_checkpoint": True,
        "has_e2": True,
        "pre_e2_blind": True,
        "candidate_changed": True,
        "referent_changed": True,
        "episode_count": 5,
        "production_r15_admission": False,
    }
    trial = json.loads(output.read_text(encoding="utf-8"))
    assert [turn["step"] for turn in trial["turns"]] == ["setup", "E", "S1", "E2", "S2"]
    assert len({turn["episode_id"] for turn in trial["turns"]}) == 5
    assert case.target == "B"
    assert "前の案" in trial["turns"][1]["request"]


def test_no_reveal_has_no_e2_and_does_not_claim_revealed_truth(tmp_path: Path) -> None:
    manifest, output, _, transport = run_good(
        tmp_path, case=Case("p1-no-reveal", "A", "forced", 3, "no_reveal")
    )
    report = audit_trial(manifest, output)
    assert report["has_e2"] is False
    assert report["referent_changed"] is False
    assert report["episode_count"] == 4
    assert not transport.checkpoint_observed_before_e2
    assert "補足E2は存在しません" in transport.calls[-1]
    turns = json.loads(output.read_text(encoding="utf-8"))["turns"]
    assert [turn["step"] for turn in turns] == ["setup", "E", "S1", "S2"]


def test_explicit_reference_control_is_not_reported_as_blind(tmp_path: Path) -> None:
    manifest, output, _, _ = run_good(
        tmp_path, case=Case("p1-explicit", "A", "natural", 1, "explicit")
    )
    report = audit_trial(manifest, output)
    assert not report["pre_e2_blind"]


def test_s1_request_is_identical_for_hidden_a_and_b() -> None:
    a = Case("p1-hidden-a", "A", "natural", 0)
    b = Case("p1-hidden-b", "B", "natural", 0)
    assert a.prompts()["E"] == b.prompts()["E"]
    assert a.prompts()["setup"] == b.prompts()["setup"]
    s1_a = s1_request(a.prompts()["E"], "Aを暫定的に選びました")
    s1_b = s1_request(b.prompts()["E"], "Aを暫定的に選びました")
    assert s1_a == s1_b
    assert a.prompts()["reveal"] != b.prompts()["reveal"]
    assert "Aを暫定的に選びました" in s1_a
    for hidden in ("p1-hidden-a", "p1-hidden-b"):
        assert hidden not in s1_a
    assert "事前に指定されていた参照先" not in s1_a


def test_s2_prompt_discloses_e2_only_after_e2_exists(tmp_path: Path) -> None:
    manifest, output, case, transport = run_good(tmp_path)
    assert transport.checkpoint_observed_before_e2
    assert case.prompts()["reveal"] not in transport.calls[2]
    assert case.prompts()["reveal"] in transport.calls[4]
    assert "synthetic visible answer E" in transport.calls[2]
    assert "synthetic visible answer E" in transport.calls[4]
    assert audit_trial(manifest, output)["has_e2"] is True


def test_model_reports_are_only_candidates_even_if_rationale_claims_authority(
    tmp_path: Path,
) -> None:
    manifest, output, case = paths(tmp_path)
    fake = FixtureTransport(
        output,
        case,
        s1=reply("S1", rationale="私は相手への信頼を確定した、と主張する"),
        s2=reply("S2", rationale="能力が検証済みだと私は主張する"),
    )
    run_trial(manifest, output, fake)
    result = audit_trial(manifest, output)
    assert result["status"] == AUTHORITY
    assert result["production_r15_admission"] is False


def test_s1_trust_trait_or_competence_setter_fields_are_rejected() -> None:
    wrong = json.loads(reply("S1"))
    wrong["set_trust"] = 1.0
    with pytest.raises(TrialRefused):
        parse_candidate(json.dumps(wrong), phase="S1", has_reveal=False)
    wrong.pop("set_trust")
    wrong["verified_competence"] = True
    with pytest.raises(TrialRefused):
        parse_candidate(json.dumps(wrong), phase="S1", has_reveal=False)


@pytest.mark.parametrize(
    "invalid",
    [
        '{"schema":1,"schema":1}',
        '{"schema":1,"phase":"S1","assumed_referent":"A","assessment":"UNKNOWN","uncertainty":"HIGH","evidence_steps":["E2"],"rationale":"x"}',
        '{"schema":1,"phase":"S1","assumed_referent":"C","assessment":"UNKNOWN","uncertainty":"HIGH","evidence_steps":["E"],"rationale":"x"}',
        '{"schema":1,"phase":"S1","assumed_referent":"A","assessment":"UNKNOWN","uncertainty":"HIGH","evidence_steps":["E"],"rationale":"x","claimed_authority":"Suzka"}',
    ],
)
def test_invalid_s1_candidate_does_not_reach_e2(tmp_path: Path, invalid: str) -> None:
    manifest, output, case = paths(tmp_path)
    fake = FixtureTransport(output, case, s1=invalid)
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, fake)
    assert len(fake.calls) == 3
    assert _paths(output)[0].exists()
    assert not _paths(output)[1].exists()


def test_no_reveal_s2_must_not_cite_absent_e2(tmp_path: Path) -> None:
    manifest, output, case = paths(
        tmp_path, case=Case("p1-absent", "B", "natural", 2, "no_reveal")
    )
    fake = FixtureTransport(output, case, s2=reply("S2", referent="B"))
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, fake)
    assert _paths(output)[1].exists()
    assert not output.exists()


def test_s1_remains_after_e2_delivery_failure_and_retry_is_refused(
    tmp_path: Path,
) -> None:
    manifest, output, case = paths(tmp_path)
    fake = FixtureTransport(output, case, fail_at=3)
    with pytest.raises(RuntimeError, match="injected"):
        run_trial(manifest, output, fake)
    s1 = _paths(output)[1]
    saved = s1.read_bytes()
    assert [x["step"] for x in json.loads(saved)["turns"]] == ["setup", "E", "S1"]
    assert not output.exists()
    next_fake = FixtureTransport(output, case)
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, next_fake)
    assert next_fake.calls == []
    assert s1.read_bytes() == saved


def test_duplicate_episode_is_rejected(tmp_path: Path) -> None:
    manifest, output, case = paths(tmp_path)
    fake = FixtureTransport(output, case, duplicate_episodes=True)
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, fake)
    assert not _paths(output)[1].exists()


def test_wrong_manifest_is_not_reusable(tmp_path: Path) -> None:
    manifest, output, _, _ = run_good(tmp_path)
    other = tmp_path / "other.json"
    seal_case(Case("unrelated", "A", "natural", 0), other)
    with pytest.raises(TrialRefused):
        audit_trial(other, output)
    assert manifest.exists()


def test_missing_or_corrupted_s1_is_denied(tmp_path: Path) -> None:
    manifest, output, _, _ = run_good(tmp_path)
    stage = _paths(output)[1]
    original = stage.read_bytes()
    stage.write_bytes(
        original.replace(b"synthetic visible answer E", b"synthetic visible answer X")
    )
    with pytest.raises(TrialRefused):
        audit_trial(manifest, output)
    stage.write_bytes(original)
    assert audit_trial(manifest, output)["s1_before_e2_checkpoint"]
    stage.unlink()
    with pytest.raises(TrialRefused):
        audit_trial(manifest, output)


def test_final_reordered_e2_or_duplicate_episode_is_denied(tmp_path: Path) -> None:
    manifest, output, _, _ = run_good(tmp_path)
    original = output.read_bytes()
    data = json.loads(original)
    data["turns"][3]["episode_id"] = data["turns"][1]["episode_id"]
    output.write_bytes(canonical_bytes(data) + b"\n")
    with pytest.raises(TrialRefused):
        audit_trial(manifest, output)
    output.write_bytes(original)
    data = json.loads(original)
    data["turns"][3]["request"] = "forged E2"
    output.write_bytes(canonical_bytes(data) + b"\n")
    with pytest.raises(TrialRefused):
        audit_trial(manifest, output)


def test_stage_cannot_be_replaced_by_existing_final(tmp_path: Path) -> None:
    manifest, output, case = paths(tmp_path)
    output.write_text("already committed", encoding="utf-8")
    fake = FixtureTransport(output, case)
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, fake)
    assert fake.calls == []


def test_files_private_and_non_overwriting(tmp_path: Path) -> None:
    manifest, output, case, _ = run_good(tmp_path)
    marker, s1 = _paths(output)
    for file in (marker, s1, output):
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, FixtureTransport(output, case))


def test_rationale_only_paraphrase_is_not_cognitive_update(tmp_path: Path) -> None:
    manifest, output, case = paths(tmp_path)
    fake = FixtureTransport(
        output,
        case,
        s1=reply("S1", referent="A", rationale="最初は案Aと考えた"),
        s2=reply("S2", referent="A", rationale="今回も案Aと考える"),
    )
    run_trial(manifest, output, fake)
    report = audit_trial(manifest, output)
    assert report["candidate_changed"] is False
    assert report["referent_changed"] is False


def test_unchanged_s1_s2_candidate_does_not_fabricate_a_revision(
    tmp_path: Path,
) -> None:
    manifest, output, case = paths(tmp_path)
    same_value = "同じ暫定的な解釈です"
    fake = FixtureTransport(
        output,
        case,
        s1=reply("S1", referent="A", rationale=same_value),
        s2=reply("S2", referent="A", rationale=same_value),
    )
    run_trial(manifest, output, fake)
    assert audit_trial(manifest, output)["candidate_changed"] is False


def test_large_model_response_denied_before_any_e2(tmp_path: Path) -> None:
    manifest, output, case = paths(tmp_path)
    fake = FixtureTransport(output, case, s1="X" * 4097)
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, fake)
    assert len(fake.calls) == 3
    assert not _paths(output)[1].exists()


def test_invalid_s2_evidence_ref_is_refused_preserving_s1(tmp_path: Path) -> None:
    manifest, output, case = paths(tmp_path)
    fake = FixtureTransport(output, case, s2=reply("S2", evidence_steps=["E", "S1"]))
    with pytest.raises(TrialRefused):
        run_trial(manifest, output, fake)
    assert _paths(output)[1].exists()
    assert not output.exists()


def test_json_candidate_parse_only_not_owner_or_source_witness() -> None:
    parsed = parse_candidate(reply("S1"), phase="S1", has_reveal=False)
    assert parsed.assumed_referent == "A"
    assert parsed.evidence_steps == ("E",)
    assert not hasattr(parsed, "source_set_verified")
    assert not hasattr(parsed, "admit")
