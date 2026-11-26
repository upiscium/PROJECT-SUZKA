"""``--conversation-trace`` of the smoke (observation only): the allow-listed config override, the results
labels and the prompt-assembly detector against a really rendered trace.  No model, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.r20 import factory, smoke
from experiments.r20.guards import GuardRefusal
from experiments.r20.model_config import apply_config_overrides, validate_config_overrides
from experiments.r20.recording_provider import assembly_numbers
from suzka.runtime.conversation_trace import RECENT_CONVERSATION_HEADING


def results_of(base: Path) -> dict:
    (path,) = sorted(base.glob("results-smoke-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_overrides_are_an_allow_list_of_one_key() -> None:
    assert validate_config_overrides(None) == {}
    assert validate_config_overrides({"conversation_trace.enabled": True}) == {"conversation_trace.enabled": True}
    for bad in ({"conversation_trace.max_items_per_context": 3}, {"api.port": 1}, {"conversation_trace": {"enabled": True}}):
        with pytest.raises(GuardRefusal) as caught:
            validate_config_overrides(bad)
        assert caught.value.code == "CONFIG_OVERRIDE_NOT_ALLOWED"
    with pytest.raises(GuardRefusal) as caught:
        validate_config_overrides({"conversation_trace.enabled": 1})
    assert caught.value.code == "CONFIG_OVERRIDE_TYPE"


def test_apply_changes_only_the_enabled_key() -> None:
    raw: dict = {"conversation_trace": {"enabled": False, "max_items_per_context": 20}}
    apply_config_overrides(raw, {"conversation_trace.enabled": True})
    assert raw == {"conversation_trace": {"enabled": True, "max_items_per_context": 20}}


def test_build_config_default_is_unchanged_and_override_only_flips_enabled() -> None:
    kwargs = dict(deterministic=True, r18=False, r19=False, run_id="abc")
    profile = factory.PROFILES["S0"]
    root = Path("/nonexistent-root")
    plain = factory.build_config(profile, root, 4000, factory.Tier.T1, **kwargs)
    same = factory.build_config(profile, root, 4000, factory.Tier.T1, config_overrides=None, **kwargs)
    on = factory.build_config(profile, root, 4000, factory.Tier.T1, config_overrides={"conversation_trace.enabled": True}, **kwargs)
    assert plain == same and plain["conversation_trace"]["enabled"] is False
    assert on["conversation_trace"]["enabled"] is True
    on["conversation_trace"]["enabled"] = False
    assert on == plain  # nothing else differs


def test_a_disallowed_override_creates_nothing(r20_base: Path) -> None:
    with pytest.raises(GuardRefusal):
        factory.create_subject("S0", base=r20_base, config_overrides={"api.port": 1})
    assert list(r20_base.iterdir()) == []


def test_the_detector_matches_a_really_rendered_trace_block() -> None:
    from datetime import datetime, timedelta, timezone

    from suzka.runtime import conversation_trace as ct

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    parameters = ct.TraceParameters.from_settings(__import__("suzka.config.schema", fromlist=["x"]).ConversationTraceSettings())

    class _Item:
        turn_kind = ct.TraceTurnKind.CHAT
        event_sequence = 1
        episode_id = "e1"
        requested_at = now - timedelta(minutes=5)

    class _Turn:
        item = _Item()
        retention = 0.9
        user_text = "最初の質問"
        response_text = "最初の答えです"

    rendered = ct.render_trace([_Turn()], now, parameters).text
    assert rendered.startswith(RECENT_CONVERSATION_HEADING)
    row = {"method": "generate_chat", "system_text": "sys", "user_text": f"{rendered}\n\nUser: 次の質問"}
    found = assembly_numbers(row, previous_request="最初の質問", previous_response="最初の答えです")
    assert found["trace_block_present"] is True and found["trace_turns"] == 1
    assert found["previous_request_in_input"] is True and found["previous_reply_in_input"] is True
    empty = assembly_numbers({"method": "generate_chat", "system_text": "s", "user_text": "User: 次の質問"})
    assert empty["trace_block_present"] is False and empty["trace_turns"] == 0


def test_the_detector_sees_a_turn_that_arrives_as_json_escaped_working_memory_evidence() -> None:
    line = json.dumps({"source_kind": "episodic", "text": "User: 最初の質問\nAssistant: 最初の答えです"}, separators=(",", ":"))
    row = {"method": "generate", "prompt": f"WorkingMemory evidence:\n{line}\nUser: 次"}
    found = assembly_numbers(row, previous_request="最初の質問", previous_response="最初の答えです")
    assert found["working_memory_episodic_items"] == 1 and found["trace_block_present"] is False
    assert found["previous_request_in_input"] is True and found["previous_reply_in_input"] is True


def test_scripted_flag_enables_the_config_and_reports_labels_only(r20_base: Path) -> None:
    assert smoke.main(["--base", str(r20_base), "--scripted", "--turns", "3", "--conversation-trace",
                       "--record-model-calls"]) == 0
    top = results_of(r20_base)
    assert top["conversation_trace"] == {"requested": True, "config_applied": True}
    run = top["runs"][0]
    assert run["ended"] == "completed" and run["conversation_trace"] == {"requested": True, "config_applied": True}
    (config,) = r20_base.glob("*/subject/config.yaml")
    assert yaml.safe_load(config.read_text(encoding="utf-8"))["conversation_trace"]["enabled"] is True
    assemblies = [turn["prompt_assembly"] for turn in run["turns"]]
    assert assemblies[0]["trace_block_present"] is False and assemblies[0]["trace_turns"] == 0
    # The trace skips turns Working Memory already shows, and the scripted subject's Working Memory shows
    # every earlier turn (JSON-escaped): the block stays absent here, the earlier turns are still seen.
    assert [item["working_memory_episodic_items"] for item in assemblies] == [0, 1, 2]
    assert assemblies[1]["previous_request_in_input"] is True and assemblies[2]["previous_request_in_input"] is True
    assert "synthetic fixture answer" not in json.dumps(top)


def test_flag_off_keeps_the_config_disabled_and_results_comparable(r20_base: Path) -> None:
    assert smoke.main(["--base", str(r20_base), "--scripted", "--turns", "2", "--record-model-calls"]) == 0
    off = results_of(r20_base)
    assert off["conversation_trace"] == {"requested": False, "config_applied": False}
    (config,) = r20_base.glob("*/subject/config.yaml")
    assert yaml.safe_load(config.read_text(encoding="utf-8"))["conversation_trace"]["enabled"] is False
    assert [turn["prompt_assembly"]["trace_block_present"] for turn in off["runs"][0]["turns"]] == [False, False]


def test_repeats_with_the_flag_share_aggregate_keys_with_the_default(r20_base: Path) -> None:
    def one(name: str, *extra: str) -> dict:
        base = r20_base / name
        base.mkdir(mode=0o700)
        assert smoke.main(["--base", str(base), "--scripted", "--turns", "2", "--repeats", "2",
                           "--record-model-calls", *extra]) == 0
        return results_of(base)

    on, off = one("on", "--conversation-trace"), one("off")
    assert on["completed_runs"] == off["completed_runs"] == 2
    assert on["conversation_trace"]["config_applied"] is True
    assert on["aggregate"]["per_label"].keys() == off["aggregate"]["per_label"].keys()
    for label in on["aggregate"]["per_label"]:
        assert on["aggregate"]["per_label"][label]["prompt_assembly"].keys() == off["aggregate"]["per_label"][label]["prompt_assembly"].keys()
    assert on["aggregate"]["per_label"]["E"]["prompt_assembly"]["working_memory_episodic_items"] == {"mean": 1.0, "of": 2}
    assert off["aggregate"]["per_label"]["E"]["prompt_assembly"]["trace_block_present"] == {"true": 0, "of": 2}
