"""IP-23a: the model-configuration override, the caps and the ledger, the GPU
probe (no NVML), ports, artifacts.  No model, no GPU, no process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time

import pytest
import yaml
from r20_ip23_helpers import MODEL_ID, REVISION, make_cache, real_model, refusal_code

from experiments.r20 import artifacts, factory, gpu, guards
from experiments.r20.caps import (
    HARD_MAX_SUBJECTS,
    HARD_SESSION_SECONDS,
    HARD_TOTAL_SECONDS,
    HARD_TURN_SECONDS,
    STAGE_LIMITS,
    Caps,
    Ledger,
    Session,
)
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


# -------------------------------------------------------------------- caps


def test_the_caps_are_the_humans_and_a_value_can_only_lower_them() -> None:
    caps = Caps()
    assert (caps.max_subjects, caps.total_seconds, caps.session_seconds, caps.turn_seconds, caps.min_free_bytes) == (
        42, 10 * 3600, 2.5 * 3600, 120.0, 14 * 1024**3,
    )
    assert sum(STAGE_LIMITS.values()) == HARD_MAX_SUBJECTS == 42
    assert Caps(max_subjects=5, turn_seconds=60.0).max_subjects == 5
    for bad in ({"max_subjects": 43}, {"total_seconds": HARD_TOTAL_SECONDS + 1}, {"session_seconds": HARD_SESSION_SECONDS + 1},
                {"turn_seconds": HARD_TURN_SECONDS + 1}, {"min_free_bytes": 1}, {"max_subjects": 0}):
        assert refusal_code(Caps, **bad) == "CAP_ABOVE_LIMIT"


def test_the_ledger_counts_subjects_and_hours_across_launches(tmp_path: Path) -> None:
    clock = [0.0]
    ledger = Ledger(tmp_path)
    caps = Caps(max_subjects=3)
    first = Session(ledger, caps, "L1", "S0", clock=lambda: clock[0])
    first.begin_subject("a")
    first.begin_subject("b")
    clock[0] = 100.0
    first.end("completed")
    second = Session(ledger, caps, "L2", "S0", clock=lambda: clock[0])
    second.begin_subject("c")
    assert ledger.subjects_used() == 3 and ledger.subjects_used("S0") == 3
    assert refusal_code(second.begin_subject, "d") == "CAP_MAX_SUBJECTS"
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600


def test_per_stage_limits_the_session_cap_and_the_total_cap(tmp_path: Path) -> None:
    clock = [0.0]
    ledger = Ledger(tmp_path)
    session = Session(ledger, Caps(), "L1", "G1", clock=lambda: clock[0])
    session.begin_subject("g1a")
    assert refusal_code(session.begin_subject, "g1b") == "CAP_STAGE_SUBJECTS"  # G1 is one subject
    clock[0] = HARD_SESSION_SECONDS + 1
    assert refusal_code(session.check) == "CAP_SESSION_HOURS"
    assert refusal_code(session.begin_subject, "x", stage="S0") == "CAP_SESSION_HOURS"
    # hours of earlier launches count toward the total
    ledger.append(kind="end", launch="OLD1", elapsed_s=HARD_SESSION_SECONDS)
    ledger.append(kind="end", launch="OLD2", elapsed_s=HARD_SESSION_SECONDS)
    ledger.append(kind="end", launch="OLD3", elapsed_s=HARD_SESSION_SECONDS)
    ledger.append(kind="end", launch="OLD4", elapsed_s=HARD_SESSION_SECONDS)
    fresh_clock = [0.0]
    late = Session(ledger, Caps(), "L9", "S0", clock=lambda: fresh_clock[0])
    assert refusal_code(late.check) == "CAP_TOTAL_HOURS"


def test_an_unknown_stage_is_refused(tmp_path: Path) -> None:
    assert refusal_code(Session, Ledger(tmp_path), Caps(), "L", "NOPE") == "STAGE_UNKNOWN"


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
    assert "14 days" in note.read_text() and "erase" in note.read_text()
    transcript = artifacts.write_private(private / "p1.capture.json", b"raw")
    assert stat.S_IMODE(transcript.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        artifacts.write_private(transcript, b"again")
    assert artifacts.private_problems(tmp_path) == []
    os.chmod(transcript, 0o644)
    assert artifacts.private_problems(tmp_path)


def _real_base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir(mode=0o700)
    artifacts.mark_base(base, "d" * 64)
    return base


def _run_dir(base: Path, name: str, *, age_days: float = 0) -> Path:
    run = base / name
    (run / "subject").mkdir(parents=True)
    (run / "subject" / guards.MARKER_NAME).write_text("{}")
    (run / "evidence").mkdir()
    (run / "evidence" / "manifest.json").write_text("{}")
    artifacts.private_dir(run)
    (run / "serve.log").write_text("log")
    (run / "home").mkdir()
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(run / "evidence", (old, old))
    return run


def test_erase_needs_confirmation_a_real_run_base_and_keeps_the_evidence(tmp_path: Path) -> None:
    base = _real_base(tmp_path)
    run = _run_dir(base, "r1")
    assert refusal_code(artifacts.erase, base, confirmed=False) == "ERASE_NOT_CONFIRMED"
    plain = tmp_path / "plain"
    plain.mkdir()
    assert refusal_code(artifacts.erase, plain, confirmed=True) == "ERASE_NOT_A_REAL_RUN_BASE"
    erased = artifacts.erase(base, confirmed=True)
    assert [item.run_id for item in erased] == ["r1"]
    assert not (run / "subject").exists() and not (run / "private").exists() and not (run / "serve.log").exists()
    assert (run / "evidence" / "manifest.json").exists()  # the closed-key manifest is not raw


def test_erase_by_age_and_by_name(tmp_path: Path) -> None:
    base = _real_base(tmp_path)
    old = _run_dir(base, "old", age_days=20)
    new = _run_dir(base, "new", age_days=1)
    erased = artifacts.erase(base, older_than_days=14, confirmed=True)
    assert [item.run_id for item in erased] == ["old"] and (new / "subject").exists() and not (old / "subject").exists()
    assert artifacts.erase(base, run_ids=["absent"], confirmed=True) == []
    assert [item.run_id for item in artifacts.erase(base, run_ids=["new"], confirmed=True)] == ["new"]


def test_erase_refuses_a_symlinked_part(tmp_path: Path) -> None:
    base = _real_base(tmp_path)
    run = _run_dir(base, "r2")
    keep = tmp_path / "elsewhere"
    keep.mkdir()
    (keep / "x").write_text("x")
    import shutil

    shutil.rmtree(run / "home")
    (run / "home").symlink_to(keep)
    assert refusal_code(artifacts.erase, base, confirmed=True) == "ERASE_SYMLINK"
    assert (keep / "x").exists()
