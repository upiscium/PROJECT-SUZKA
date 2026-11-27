"""Lab allocator test of the smoke (observation only): the ``--alloc-conf`` allow-list, the child-only environment,
the ``--empty-cache`` hook (fake torch, a failing ``empty_cache``), numbers-only results, flag-off unchanged.
No model, no GPU."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import types
from typing import Any

import pytest
from test_r20_smoke_restore_matrix import _matrix_args, matrix_results

from experiments.r20 import alloc_probe, serve, smoke, supervisor
from experiments.r20.factory import Tier, create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.supervisor import ChildProcess, child_environment

GIB = 1024**3
CONFS = list(alloc_probe.ALLOWED_ALLOC_CONF)


class FakeCuda:
    def __init__(self, *, fail_empty: bool = False, fail_reserved: bool = False) -> None:
        self.reserved = 12 * GIB
        self.free = 1 * GIB
        self.empty_calls = 0
        self.fail_empty = fail_empty
        self.fail_reserved = fail_reserved

    def memory_reserved(self) -> int:
        if self.fail_reserved:
            raise RuntimeError("no reading")
        return self.reserved

    def mem_get_info(self) -> tuple[int, int]:
        return self.free, 24 * GIB

    def empty_cache(self) -> None:
        self.empty_calls += 1
        if self.fail_empty:
            raise RuntimeError("boom")
        self.free += self.reserved - 8 * GIB
        self.reserved = 8 * GIB


class Inner:
    def generate(self, prompt: str) -> str:
        return "plain:" + prompt

    def generate_chat(self, system_text: str, user_text: str) -> str:
        return "chat:" + user_text


class Raising:
    def generate(self, prompt: str) -> str:
        raise ValueError("model failed")


def parse(base: Path, *extra: str) -> Any:
    return smoke.parser().parse_args(["--base", str(base), "--scripted", *extra])


# ------------------------------------------------------------------------------------------ flag validation


@pytest.mark.parametrize("value", CONFS)
def test_the_three_allowed_values_pass(r20_base: Path, value: str) -> None:
    smoke._check_arguments(parse(r20_base, "--alloc-conf", value))  # noqa: SLF001
    assert alloc_probe.check_alloc_conf(value) == value


@pytest.mark.parametrize("value", [
    "", "expandable_segments:False", "expandable_segments:True ", "expandable_segments:true", "max_split_size_mb:128",
    "garbage_collection_threshold:0.8", "garbage_collection_threshold:0.6,expandable_segments:True",
    "expandable_segments:True;garbage_collection_threshold:0.6", "expandable_segments:True,max_split_size_mb:1",
    "backend:cudaMallocAsync",
])
def test_anything_else_is_refused_before_anything_exists(r20_base: Path, value: str) -> None:
    with pytest.raises(GuardRefusal) as caught:
        smoke._check_arguments(parse(r20_base, "--alloc-conf", value))  # noqa: SLF001
    assert caught.value.code == "ALLOC_CONF_NOT_ALLOWED" and list(r20_base.iterdir()) == []
    assert smoke.main(["--base", str(r20_base), "--scripted", "--alloc-conf", value]) == smoke.EXIT_REFUSED


def test_the_empty_cache_mode_is_a_closed_list(r20_base: Path) -> None:
    for mode in ("none", "after-load", "per-turn"):
        smoke._check_arguments(parse(r20_base, "--empty-cache", mode))  # noqa: SLF001
    with pytest.raises(SystemExit):
        parse(r20_base, "--empty-cache", "always")
    args = parse(r20_base)
    args.empty_cache = "always"
    with pytest.raises(GuardRefusal) as caught:
        smoke._check_arguments(args)  # noqa: SLF001
    assert caught.value.code == "EMPTY_CACHE_MODE_INVALID"


def test_the_defaults_mean_the_test_was_not_asked_for(r20_base: Path) -> None:
    args = parse(r20_base)
    assert (args.alloc_conf, args.empty_cache) == (None, "none")
    assert smoke._alloc_request(args) is None and smoke._allocator_header(args) is None  # noqa: SLF001
    assert smoke._alloc_request(parse(r20_base, "--empty-cache", "per-turn")) == (None, "per-turn")  # noqa: SLF001
    assert smoke._alloc_request(parse(r20_base, "--alloc-conf", CONFS[0])) == (CONFS[0], "none")  # noqa: SLF001


def test_the_matrix_takes_the_flags_as_global_options(r20_base: Path) -> None:
    smoke._check_arguments(_matrix_args(r20_base, "--alloc-conf", CONFS[2], "--empty-cache", "after-load"))  # noqa: SLF001


def test_the_serve_parser_takes_the_flag_and_defaults_to_absent() -> None:
    assert serve.parse(["root", "--run-id", "r", "--empty-cache", "per-turn"]).empty_cache == "per-turn"
    assert serve.parse(["root", "--run-id", "r"]).empty_cache is None
    with pytest.raises(SystemExit):
        serve.parse(["root", "--run-id", "r", "--empty-cache", "sometimes"])


# ------------------------------------------------------------------------------------------ the child environment


def test_the_child_environment_gets_the_variable_only_when_requested(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    parent = {"PATH": "/bin", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:1", "SUZKA_SECRET": "x"}
    plain = child_environment(subject, parent)
    assert alloc_probe.ALLOC_CONF_ENV not in plain  # the parent's own value is never inherited
    assert child_environment(subject, parent, alloc_conf=None) == plain
    for value in CONFS:
        env = child_environment(subject, parent, alloc_conf=value)
        assert env[alloc_probe.ALLOC_CONF_ENV] == value
        assert {key: item for key, item in env.items() if key != alloc_probe.ALLOC_CONF_ENV} == plain
    with pytest.raises(GuardRefusal) as caught:
        child_environment(subject, parent, alloc_conf="max_split_size_mb:1")
    assert caught.value.code == "ALLOC_CONF_NOT_ALLOWED"


class _Spawned:
    pid = 4242

    def poll(self) -> None:
        return None


def _spawn_capture(monkeypatch: pytest.MonkeyPatch, subject: Any, **keywords: Any) -> tuple[list[str], dict[str, str]]:
    seen: dict[str, Any] = {}

    def popen(command: list[str], **options: Any) -> _Spawned:
        seen["command"], seen["env"] = command, options["env"]
        return _Spawned()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor, "process_start_ticks", lambda pid: None)
    child = ChildProcess.spawn(subject, **keywords)
    ChildProcess._live.remove(child)  # noqa: SLF001
    return seen["command"], seen["env"]


def test_spawn_command_and_environment_are_unchanged_without_the_flags(
    r20_base: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    command, env = _spawn_capture(monkeypatch, subject)
    assert "--empty-cache" not in command and alloc_probe.ALLOC_CONF_ENV not in env
    command_on, env_on = _spawn_capture(monkeypatch, subject, alloc_conf=CONFS[0], empty_cache="after-load")
    assert command_on[: len(command)] == command and command_on[len(command):] == ["--empty-cache", "after-load"]
    assert env_on.pop(alloc_probe.ALLOC_CONF_ENV) == CONFS[0] and env_on == env
    with pytest.raises(GuardRefusal):
        _spawn_capture(monkeypatch, subject, alloc_conf="bogus")
    with pytest.raises(GuardRefusal):
        _spawn_capture(monkeypatch, subject, empty_cache="bogus")


# ------------------------------------------------------------------------------------------ the hook


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _loaded(tmp_path: Path, mode: str, cuda: FakeCuda) -> alloc_probe.EmptyCacheProvider:
    load = alloc_probe.wrap_loader(lambda *a, **k: Inner(), mode, tmp_path, lambda: cuda)
    provider = load("settings")
    assert isinstance(provider, alloc_probe.EmptyCacheProvider)
    return provider


def test_after_load_empties_once_and_per_turn_after_every_call(tmp_path: Path) -> None:
    cuda = FakeCuda()
    provider = _loaded(tmp_path, "after-load", cuda)
    assert cuda.empty_calls == 1
    assert provider.generate_chat("s", "u") == "chat:u" and provider.generate("p") == "plain:p"
    assert cuda.empty_calls == 1  # after-load: not per call
    rows = _rows(tmp_path / alloc_probe.PROBE_NAME)
    assert [row["event"] for row in rows] == ["after_load", "per_turn", "per_turn"]
    assert (rows[0]["reserved_before_bytes"], rows[0]["reserved_after_bytes"]) == (12 * GIB, 8 * GIB)
    assert [row["empty_cache_called"] for row in rows] == [True, False, False]

    cuda = FakeCuda()
    tmp = tmp_path / "per"
    tmp.mkdir()
    provider = _loaded(tmp, "per-turn", cuda)
    provider.generate_chat("s", "u")
    provider.generate("p")
    assert cuda.empty_calls == 3
    summary = alloc_probe.summarize(alloc_probe.read_rows(tmp))
    assert summary["empty_cache_calls"] == 3 and summary["empty_cache_errors"] == 0
    assert summary["reserved_after_load_bytes"] == 8 * GIB and summary["probe_max_reserved_bytes"] == 12 * GIB


def test_mode_none_only_observes(tmp_path: Path) -> None:
    cuda = FakeCuda()
    provider = _loaded(tmp_path, "none", cuda)
    provider.generate("p")
    assert cuda.empty_calls == 0
    summary = alloc_probe.summarize(alloc_probe.read_rows(tmp_path))
    assert summary["empty_cache_calls"] == 0 and summary["reserved_after_load_bytes"] == 12 * GIB


def test_a_failing_empty_cache_never_crashes_a_turn_and_is_recorded(tmp_path: Path) -> None:
    cuda = FakeCuda(fail_empty=True)
    provider = _loaded(tmp_path, "per-turn", cuda)  # the load survives a failing call
    assert provider.generate_chat("s", "u") == "chat:u"  # the reply is delivered unchanged
    summary = alloc_probe.summarize(alloc_probe.read_rows(tmp_path))
    assert summary["empty_cache_calls"] == 0 and summary["empty_cache_errors"] == 2
    row = _rows(tmp_path / alloc_probe.PROBE_NAME)[-1]
    assert row["error"] == "RuntimeError" and row["empty_cache_called"] is False


def test_failing_readings_and_a_failing_device_are_gaps_not_crashes(tmp_path: Path) -> None:
    provider = _loaded(tmp_path, "per-turn", FakeCuda(fail_reserved=True))
    assert provider.generate("p") == "plain:p"
    assert alloc_probe.read_rows(tmp_path)[-1]["reserved_before_bytes"] is None

    def no_cuda() -> Any:
        raise ImportError("no torch")

    other = tmp_path / "x"
    other.mkdir()
    broken = alloc_probe.wrap_loader(lambda *a, **k: Inner(), "per-turn", other, no_cuda)("s")
    assert broken.generate("p") == "plain:p"
    assert alloc_probe.read_rows(other)[-1]["error"] == "ImportError"


def test_a_model_error_propagates_and_the_cache_is_still_emptied(tmp_path: Path) -> None:
    cuda = FakeCuda()
    provider = alloc_probe.wrap_loader(lambda *a, **k: Raising(), "per-turn", tmp_path, lambda: cuda)("s")
    with pytest.raises(ValueError):
        provider.generate("p")
    assert cuda.empty_calls == 2


def test_generate_chat_is_probed_like_the_inner_provider(tmp_path: Path) -> None:
    provider = alloc_probe.wrap_loader(lambda *a, **k: Raising(), "none", tmp_path, FakeCuda)("s")
    assert getattr(provider, "generate_chat", None) is None
    assert callable(getattr(_loaded(tmp_path / ".", "none", FakeCuda()), "generate_chat", None))


def test_the_probe_file_is_0600_and_holds_numbers_booleans_and_labels_only(tmp_path: Path) -> None:
    provider = _loaded(tmp_path, "per-turn", FakeCuda())
    provider.generate_chat("SECRET system", "SECRET user")
    path = tmp_path / alloc_probe.PROBE_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "SECRET" not in path.read_text(encoding="utf-8")
    for row in _rows(path):
        assert all(value is None or isinstance(value, (int, bool, str)) for value in row.values())
        assert set(row) == {
            "seq", "event", "mode", "empty_cache_called", "error", "reserved_before_bytes", "reserved_after_bytes",
            "free_before_bytes", "free_after_bytes",
        }


def test_the_free_reading_is_device_wide_mem_get_info(tmp_path: Path) -> None:
    cuda = FakeCuda()
    _loaded(tmp_path, "after-load", cuda)
    row = alloc_probe.read_rows(tmp_path)[0]
    assert row["free_before_bytes"] == 1 * GIB and row["free_after_bytes"] == 5 * GIB
    assert alloc_probe.summarize([row])["probe_min_free_bytes"] == 1 * GIB


# ------------------------------------------------------------------------------------------ the results


def test_flag_off_results_have_no_allocator_keys(r20_base: Path) -> None:
    smoke._smoke(  # noqa: SLF001
        parse(r20_base, "--turns", "2"), attest=lambda *a, **k: pytest.fail("no attestation"), gpu_runner=None,
    )
    (path,) = sorted(r20_base.glob("results-smoke-*.json"))
    top = json.loads(path.read_text(encoding="utf-8"))
    assert "allocator" not in top and "allocator" not in top["runs"][0]
    assert set(top["requested"]) == {"repeats", "turns", "record_model_calls"}


def test_a_single_run_records_the_requested_labels_and_numbers_only(r20_base: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = smoke._smoke(  # noqa: SLF001
        parse(r20_base, "--turns", "2", "--alloc-conf", CONFS[0], "--empty-cache", "after-load"),
        attest=lambda *a, **k: pytest.fail("no attestation"), gpu_runner=None,
    )
    assert code == 0
    (path,) = sorted(r20_base.glob("results-smoke-*.json"))
    top = json.loads(path.read_text(encoding="utf-8"))
    assert top["allocator"] == {"alloc_conf_requested": CONFS[0], "empty_cache_mode": "after-load"}
    allocator = top["runs"][0]["allocator"]
    assert allocator["alloc_conf_requested"] == CONFS[0] and allocator["empty_cache_mode"] == "after-load"
    # a scripted fixture has no CUDA hook: the numbers are present and honestly empty
    assert allocator["empty_cache_calls"] == 0 and allocator["reserved_after_load_bytes"] is None
    assert {"empty_cache_errors", "max_reserved_bytes", "min_free_bytes", "calls"} <= set(allocator)
    assert "allocator:" in capsys.readouterr().out


def test_allocator_record_combines_the_sampler_and_the_hook(tmp_path: Path) -> None:
    provider = _loaded(tmp_path, "per-turn", FakeCuda())
    provider.generate("p")
    args = parse(tmp_path, "--alloc-conf", CONFS[1], "--empty-cache", "per-turn")
    gpu_summary = {"min_free": 3 * GIB // 4, "peak_reserved": 15 * GIB, "peak_allocated": 7 * GIB}
    record = smoke._allocator_record(args, tmp_path, gpu_summary)  # noqa: SLF001
    assert record is not None
    assert record["max_reserved_bytes"] == 15 * GIB  # the sampler's peak beats the hook's readings
    assert record["min_free_bytes"] == 3 * GIB // 4
    assert record["reserved_after_load_bytes"] == 8 * GIB and record["empty_cache_calls"] == 2
    assert [call["event"] for call in record["calls"]] == ["after_load", "per_turn"]
    for key, value in record.items():
        assert key in {"alloc_conf_requested", "empty_cache_mode", "calls"} or value is None or isinstance(value, int)
    assert smoke._allocator_record(parse(tmp_path), tmp_path, gpu_summary) is None  # noqa: SLF001


def test_the_matrix_records_the_flags_once_in_the_header_and_two_table_columns(
    r20_base: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = smoke._smoke(  # noqa: SLF001
        _matrix_args(r20_base, "--alloc-conf", CONFS[2], "--empty-cache", "per-turn", "--matrix-conditions", "X0,X7"),
        attest=lambda *a, **k: pytest.fail("no attestation"), gpu_runner=None,
    )
    assert code == 0
    top = matrix_results(r20_base)
    assert top["allocator"] == {"alloc_conf_requested": CONFS[2], "empty_cache_mode": "per-turn"}
    for item in top["conditions"]:
        assert item["allocator"]["alloc_conf_requested"] == CONFS[2] and item["allocator"]["empty_cache_calls"] == 0
        assert {"reserved_after_load_bytes", "max_reserved_bytes", "min_free_bytes"} <= set(item["allocator"])
    header = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("cond"))
    assert "reserved after load GiB" in header and "max reserved GiB" in header


def test_the_flag_off_matrix_has_no_allocator_keys_or_columns(r20_base: Path, capsys: pytest.CaptureFixture[str]) -> None:
    smoke._smoke(  # noqa: SLF001
        _matrix_args(r20_base, "--matrix-conditions", "X0"), attest=lambda *a, **k: pytest.fail("no attestation"), gpu_runner=None,
    )
    top = matrix_results(r20_base)
    assert "allocator" not in top and "allocator" not in top["conditions"][0] and "allocator" not in top["conditions"][0]["runs"][0]
    header = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("cond"))
    assert "reserved" not in header


def test_the_attestation_summary_names_the_allocator_test_only_when_asked(r20_base: Path) -> None:
    lines = smoke._summary_lines(parse(r20_base, "--alloc-conf", CONFS[0]), types.SimpleNamespace(  # noqa: SLF001
        model_id="m", revision="r", hf_home="h", load_in_4bit=True, temperature=None,
    ))
    assert any("PYTORCH_CUDA_ALLOC_CONF=" + CONFS[0] in line for line in lines)
    plain = smoke._summary_lines(parse(r20_base), types.SimpleNamespace(  # noqa: SLF001
        model_id="m", revision="r", hf_home="h", load_in_4bit=True, temperature=None,
    ))
    assert not any("allocator" in line for line in plain)
