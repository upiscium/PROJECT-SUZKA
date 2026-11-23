"""IP-22a: the evidence manifest, the provider modes of ``serve`` and the
attachment points left for 22b / 22c / 22d."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.r20 import hooks, serve
from experiments.r20.factory import create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.manifest import ADMISSION_PATH, HOLD_UNSEALED, KEYS, Manifest
from experiments.r20.provider import ProviderMode, ScriptedProvider


def test_the_manifest_lives_outside_the_subject_and_has_a_closed_key_set(r20_base: Path) -> None:
    subject = create_subject("S0", base=r20_base)
    manifest = Manifest(subject.evidence_dir, subject.run_id, subject_root=subject.root)
    assert subject.root not in manifest.path.parents
    with pytest.raises(GuardRefusal) as inside:
        Manifest(subject.root / "evidence", subject.run_id, subject_root=subject.root)
    assert inside.value.code == "EVIDENCE_INSIDE_SUBJECT"
    with pytest.raises(KeyError):
        manifest.set("anything_else", 1)
    with pytest.raises(KeyError):
        manifest.append("run_id", 1)
    manifest.set("profile", "S0")
    manifest.set("approval_actor", "SCRIPTED")
    manifest.set("provider", ScriptedProvider().identity)
    manifest.append("timeline", {"step": 1, "event_type": "CHAT"})
    manifest.result("identity_unique", "PASS")
    path = manifest.write()
    data = json.loads(path.read_text())
    assert set(data) == set(KEYS)
    assert data["admission_path"] == ADMISSION_PATH == "HARNESS_BRIDGE"
    assert data["fixture_only"] is True and data["provider"]["fixture_only"] is True
    assert not list(subject.root.rglob("manifest.json"))
    assert not path.with_name("manifest.json.tmp").exists()


def test_a_hold_is_unsealed_unless_a_store_receipt_id_is_named(r20_base: Path) -> None:
    subject = create_subject("F1", base=r20_base)
    manifest = Manifest(subject.evidence_dir, subject.run_id)
    assert manifest.record_hold("2b") == {"row": "2b", "outcome": HOLD_UNSEALED, "receipt_id": None}
    sealed = manifest.record_hold("2b", receipt_id="receipt-from-subject-history")
    assert sealed["outcome"] == "SEALED"
    assert [entry["outcome"] for entry in manifest.to_dict()["plan_hold_receipts"]] == [HOLD_UNSEALED, "SEALED"]


def test_journal_bytes_are_recorded_for_the_budget_measurement(tmp_path: Path) -> None:
    (tmp_path / "j").write_bytes(b"x" * 10)
    manifest = Manifest(tmp_path, "r")
    assert manifest.record_journal_bytes(tmp_path / "j", tmp_path / "missing") == 10
    assert manifest.to_dict()["journal"] == {"live_bytes": 10}


def test_provider_modes_scripted_dummy_and_real(r20_base: Path, tmp_path: Path) -> None:
    config = tmp_path / "p.json"
    config.write_text(json.dumps({"script": ["a", "b"], "default": "d", "loss": 5.0}))
    scripted = serve.load_provider(ProviderMode.SCRIPTED, config)
    assert [scripted.generate("x"), scripted.generate("x"), scripted.generate("x")] == ["a", "b", "d"]
    assert scripted.calculate_loss("c", "t") == 5.0
    assert serve.load_provider(ProviderMode.SCRIPTED, None).generate("x")
    # dummy and real keep the app's own ``load_model_provider`` (``provider: dummy``)
    assert serve.load_provider(ProviderMode.DUMMY, None) is None
    assert serve.load_provider(ProviderMode.REAL, None) is None
    config.write_text("[1]")
    with pytest.raises(GuardRefusal):
        serve.load_provider(ProviderMode.SCRIPTED, config)


def test_the_hooks_for_22b_22c_22d_exist_and_are_empty() -> None:
    assert hooks.KILL_POINTS == ("K1", "K2", "K3", "K4", "K5", "K6")
    assert hooks.T1_ONLY_KILL_POINTS == {"K1"}  # confirm writes challenge and claim in one request
    assert hooks.ABLATION == {} and hooks.F_CASES == {} and hooks.DRY_RUN_STEPS == {}
    table: dict = {}
    hooks.register(table, "F01-m", lambda: None)
    with pytest.raises(ValueError):
        hooks.register(table, "F01-m", lambda: None)
