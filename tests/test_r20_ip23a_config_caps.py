"""IP-23a: the model-configuration override, the GPU
probe (no NVML), ports, artifacts.  No model, no GPU, no process."""

from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest
import yaml
from r20_ip23_helpers import MODEL_ID, REVISION, make_cache, real_model, refusal_code

from experiments.r20 import artifacts, factory, gpu, guards
from experiments.r20.factory import Tier, create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.model_config import ModelOverride, cached_revision, check_pinned_revision

# ------------------------------------------------------------ model override


def test_the_default_stays_dummy_and_an_override_changes_only_the_model_sections(r20_base: Path) -> None:
    default = create_subject("S0", base=r20_base, tier=Tier.T2, run_id="d1")
    assert yaml.safe_load(default.config_path.read_text())["model"]["provider"] == "dummy"
    real = create_subject("S0", base=r20_base, tier=Tier.T2, run_id="r1", model=real_model(max_new_tokens=300))
    raw = yaml.safe_load(real.config_path.read_text())
    assert raw["model"] == {
        "primary_id": MODEL_ID, "fallback_id": MODEL_ID, "provider": "transformers", "device": "auto", "dtype": "auto",
        "load_in_4bit": True,
    }
    assert raw["generation"]["max_new_tokens"] == 300
    baseline = yaml.safe_load(default.config_path.read_text())
    assert raw["emotion"] == baseline["emotion"] and raw["r19"] == baseline["r19"]
    # the real subject still passes every isolation check of the factory
    factory.verify_prepared(real)


def test_the_model_id_is_explicit_and_never_defaulted_or_read_from_a_host() -> None:
    assert refusal_code(ModelOverride, provider="transformers") == "MODEL_ID_REQUIRED"
    for bad in ("", "no-slash", "../etc/passwd", "a/b/c", "org/name with space"):
        assert refusal_code(ModelOverride, provider="transformers", model_id=bad) == "MODEL_ID_REQUIRED"
    assert refusal_code(ModelOverride, provider="transformers", model_id=MODEL_ID, revision="main") == "MODEL_REVISION_INVALID"
    assert refusal_code(ModelOverride, provider="transformers", model_id=MODEL_ID, hf_home=Path("rel")) == "HF_HOME_NOT_ABSOLUTE"
    assert refusal_code(ModelOverride, provider="transformers", model_id=MODEL_ID, hf_home=Path("~/x")) == "HF_HOME_NOT_ABSOLUTE"
    assert refusal_code(ModelOverride, provider="other") == "MODEL_PROVIDER_UNKNOWN"
    assert refusal_code(ModelOverride, provider="transformers", model_id=MODEL_ID, max_new_tokens=0) == "MODEL_MAX_NEW_TOKENS"
    assert ModelOverride().provider == "dummy" and not ModelOverride().is_real
    source = (Path(factory.__file__).parent / "model_config.py").read_text()
    # the id, the cache and the digest come from arguments: no environment variable, no host file is read
    assert "environ" not in source and "getenv" not in source and "melchior" not in source


def test_the_cache_must_hold_exactly_the_pinned_revision(tmp_path: Path) -> None:
    hf = make_cache(tmp_path / "hf")
    assert cached_revision(hf, MODEL_ID) == REVISION
    assert check_pinned_revision(real_model(hf)) == REVISION
    other = "f" * 40
    assert refusal_code(check_pinned_revision, real_model(hf, revision=other)) == "MODEL_REVISION_MISMATCH"
    assert refusal_code(check_pinned_revision, real_model(tmp_path / "nowhere")) == "MODEL_NOT_IN_CACHE"
    assert refusal_code(check_pinned_revision, real_model(make_cache(tmp_path / "nosnap", snapshot=False))) == "MODEL_SNAPSHOT_MISSING"
    assert refusal_code(check_pinned_revision, ModelOverride(provider="transformers", model_id=MODEL_ID)) == "MODEL_PIN_REQUIRED"
    assert refusal_code(check_pinned_revision, ModelOverride()) == "MODEL_NOT_REAL"
    (hf / "hub" / "models--example-org--example-12b-it" / "refs" / "main").write_text("not-a-digest")
    assert refusal_code(cached_revision, hf, MODEL_ID) == "MODEL_REVISION_UNREADABLE"


def test_the_record_carries_identifiers_only() -> None:
    record = real_model().record()
    assert record["mode"] == "real" and record["model_id"] == MODEL_ID and record["revision_pinned"] == REVISION
    assert record["fixture_only"] is False and ModelOverride().record()["fixture_only"] is True


# --------------------------------------------------------------- the GPU probe


def test_the_gpu_probe_uses_the_cuda_runtime_never_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "mem_get_info" in gpu.PROBE_CODE and "nvidia-smi" not in gpu.PROBE_CODE and "nvml" not in gpu.PROBE_CODE.lower()
    source = Path(gpu.__file__).read_text()
    assert source.count("nvidia-smi") == 1  # the docstring says it is never used; it is never a command
    assert '["nvidia-smi"' not in source and "'nvidia-smi'" not in source


def test_the_gpu_free_minimum_and_clear_refusals() -> None:
    def ok(free_gib: float) -> str:
        return json.dumps({"free": int(free_gib * gpu.GIB), "total": 24 * gpu.GIB, "reserved": 0})

    assert gpu.check_free(runner=lambda: ok(14.0)).free_bytes == 14 * gpu.GIB
    assert refusal_code(gpu.check_free, runner=lambda: ok(13.9)) == "GPU_FREE_BELOW_MINIMUM"
    assert refusal_code(gpu.check_free, runner=lambda: json.dumps({"error": "CUDA_UNAVAILABLE"})) == "CUDA_UNAVAILABLE"
    assert refusal_code(gpu.check_free, runner=lambda: json.dumps({"error": "CUDA_PROBE_FAILED"})) == "CUDA_PROBE_FAILED"
    assert refusal_code(gpu.check_free, runner=lambda: "not json") == "CUDA_PROBE_FAILED"
    assert refusal_code(gpu.check_free, runner=lambda: "[]") == "CUDA_PROBE_FAILED"
    assert refusal_code(gpu.check_free, runner=lambda: json.dumps({"free": "x"})) == "CUDA_PROBE_FAILED"

    def boom() -> str:
        raise OSError("no python")

    assert refusal_code(gpu.check_free, runner=boom) == "CUDA_PROBE_FAILED"


def test_the_probe_in_a_subprocess_says_cuda_is_unavailable_without_torch_cuda() -> None:
    """The real subprocess runner on this machine: either a reading or a clear refusal, never a crash."""

    try:
        reading = gpu.read_gpu()
    except GuardRefusal as refusal:
        assert refusal.code in {"CUDA_UNAVAILABLE", "CUDA_PROBE_FAILED"}
    else:  # a developer machine with a GPU
        assert reading.total_bytes > 0


# ---------------------------------------------------------- ports and the base


def test_a_real_subject_uses_a_high_loopback_port_never_a_forbidden_one(r20_base: Path) -> None:
    for index in range(12):
        subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2, run_id=f"p{index}", model=real_model())
        settings = subject.settings()
        assert settings.api.host == "127.0.0.1"
        assert 41000 <= subject.port <= 60999 and subject.port not in guards.FORBIDDEN_PORTS
        assert settings.r18.approval_origins == [f"http://localhost:{subject.port}"]
    assert guards.FORBIDDEN_PORTS == {8000, 8080, 8101}


def test_production_locations_are_refused_as_a_base(tmp_path: Path) -> None:
    for place in ("/etc/r20", "/opt/project-suzka/r20", "/var/lib/suzka/r20"):
        assert refusal_code(create_subject, "NEG-plain", base=Path(place), tier=Tier.T2) in {"PRODUCTION_PATH"}


# ------------------------------------------------------------------ artifacts


def test_raw_transcripts_live_in_a_private_directory_with_a_retention_note(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    private = artifacts.private_dir(run_dir)
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    note = private / "RETENTION.txt"
    assert stat.S_IMODE(note.stat().st_mode) == 0o600
    assert "14 days" in note.read_text()
    transcript = artifacts.write_private(private / "p1.capture.json", b"raw")
    assert stat.S_IMODE(transcript.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        artifacts.write_private(transcript, b"again")
