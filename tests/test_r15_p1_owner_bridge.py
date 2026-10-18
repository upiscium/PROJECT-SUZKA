"""R15 P1b synthetic integration: actual R07/R12/DB1, no R15 owner authority."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p0.pilot import Case, seal_case
from experiments.r15_p1.owner_bridge import (
    SourceBridgeRefused,
    TrialOwnerRead,
    create_s1_root,
    create_s2_root,
    decode_key,
    load_test_root,
    option_key,
    recover_option,
    projected_current_referent,
    save_test_root,
)
from experiments.r15_p1.protocol import audit_trial, run_trial
from suzka.api.server import create_app
from suzka.config import load_settings
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem, ExperienceStore
from suzka.models.dummy_provider import DummyProvider

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config.yaml"


def answer(phase: str, selected: str, *, no_reveal: bool) -> str:
    return json.dumps(
        {
            "schema": 1,
            "phase": phase,
            "assumed_referent": selected,
            "assessment": (
                "LIKELY_ALIGNED"
                if phase == "S1"
                else "MAY_HAVE_MISREAD"
                if not no_reveal
                else "UNKNOWN"
            ),
            "uncertainty": "LOW" if phase == "S1" else "MEDIUM",
            "evidence_steps": (
                ["E"]
                if phase == "S1"
                else ["E", "S1"]
                if no_reveal
                else ["E", "S1", "E2"]
            ),
            "rationale": "synthetic fixture only, not subject adoption",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


class SequencedProvider(DummyProvider):
    def __init__(self, *, no_reveal: bool = False) -> None:
        self.calls = 0
        self.no_reveal = no_reveal

    def generate(self, prompt: str) -> str:
        i = self.calls
        self.calls += 1
        if i == 2:
            return answer("S1", "A", no_reveal=self.no_reveal)
        if i == (3 if self.no_reveal else 4):
            return answer(
                "S2", "A" if self.no_reveal else "B", no_reveal=self.no_reveal
            )
        if i == 1:
            return "元の回答では案Aを暫定的に選びました"
        if i == 3:
            return "補足では案Bと言われました"
        return "テスト用の可視回答"


def settings_in(root: Path):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    settings = load_settings(CONFIG)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": root / "chroma",
                    "db1_collection": "p1_hippocampus",
                    "db2_collection": "p1_cortex",
                }
            ),
            "sleep": settings.sleep.model_copy(
                update={"dream_dataset_path": root / "dreams" / "dataset.jsonl"}
            ),
            "qlora": settings.qlora.model_copy(
                update={"output_dir": root / "adapters", "dry_run": True}
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": root / "adapter_registry.json",
                    "eval_result_dir": root / "eval_results",
                    "eval_sets": [],
                }
            ),
            "api": settings.api.model_copy(
                update={"admin_token_env": "SUZKA_P1B_TEST_TOKEN"}
            ),
            "agent_state": settings.agent_state.model_copy(
                update={"path": root / "agent_state.json"}
            ),
            "event_journal": settings.event_journal.model_copy(
                update={"path": root / "event_journal.jsonl"}
            ),
            "state_wal": settings.state_wal.model_copy(
                update={"directory": root / "private" / "state_wal"}
            ),
        }
    )


def scenario(tmp_path: Path, *, no_reveal: bool = False):
    source = tmp_path / "disposable_source"
    settings = settings_in(source)
    manifest = tmp_path / "manifest.json"
    capture = tmp_path / "capture.json"
    seal_case(
        Case(
            "p1b-fake-no-e2" if no_reveal else "p1b-fake-with-e2",
            "B",
            "forced",
            0,
            "no_reveal" if no_reveal else "normal",
        ),
        manifest,
    )
    provider = SequencedProvider(no_reveal=no_reveal)
    os.environ["SUZKA_P1B_TEST_TOKEN"] = "not-real"
    app = create_app(settings)
    app.state.model_provider = provider
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    return app, provider, source, manifest, capture


@pytest.mark.parametrize("no_reveal", [False, True])
def test_real_disposable_owners_stage_before_e2_then_recover_after_restart(
    tmp_path: Path, no_reveal: bool
) -> None:
    app, provider, source, manifest, capture = scenario(tmp_path, no_reveal=no_reveal)
    stage_file = capture.with_name(capture.name + ".s1.json")
    checkpoint_s1 = tmp_path / "r15_test_s1.json"
    checkpoint_s2 = tmp_path / "r15_test_s2.json"
    original_s1 = None

    with TestClient(app) as client:
        owners = TrialOwnerRead(
            client.app.state.event_journal,
            ExperienceStore.from_memory_root(source / "chroma"),
            client.app.state.memory_system,
        )
        call_index = 0

        def transport(message: str) -> dict[str, str]:
            nonlocal call_index, original_s1
            if call_index == 3:
                # Executed AFTER the P1 runner has saved S1 and BEFORE E2
                # can be observed, even for the no-E2 control.
                assert stage_file.is_file()
                original_s1 = create_s1_root(manifest, stage_file, owners)
                assert original_s1.revision == 1
                save_test_root(checkpoint_s1, original_s1)
                assert load_test_root(checkpoint_s1) == original_s1
            call_index += 1
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            result = response.json()
            return {
                "episode_id": result["episode_id"],
                "response": result["response"],
            }

        run_trial(manifest, capture, transport)
        assert original_s1 is not None
        audit = audit_trial(manifest, capture)
        assert audit["s1_before_e2_checkpoint"] is True
        assert audit["production_r15_admission"] is False
        latest = create_s2_root(manifest, capture, original_s1, owners)
        save_test_root(checkpoint_s2, latest)
        assert load_test_root(checkpoint_s2) == latest
        assert provider.calls == (4 if no_reveal else 5)
        setup_id = json.loads(capture.read_text())["turns"][0]["episode_id"]
        assert recover_option(original_s1, setup_episode_id=setup_id, stage="S1") == "A"
        if no_reveal:
            assert latest == original_s1
            assert audit["referent_changed"] is False
        else:
            assert latest.revision == 2
            assert len(latest.episodes) == 2
            assert recover_option(latest, setup_episode_id=setup_id, stage="S2") == "B"
            assert len(latest.revision_history) == 2
            assert latest.revision_history[1].event.event_sequence > (
                latest.revision_history[0].event.event_sequence
            )
            assert audit["referent_changed"] is True

    # A fresh parser instance after the app has shut down must read exactly
    # the same snapshot/revisions, without resurrecting a fabricated truth.
    restored = load_test_root(checkpoint_s2)
    assert restored == latest
    projected = projected_current_referent(restored, setup_episode_id=setup_id)
    assert projected["current_referent"] == ("A" if no_reveal else "B")
    assert projected["selected_stage"] == ("S1" if no_reveal else "S2")
    assert projected["production_decision_authority"] is False
    assert (source / "experience" / "records").is_dir()


def test_option_key_needs_exact_original_source_scope() -> None:
    original = option_key("original-setup", "A")
    assert decode_key("original-setup", original) == "A"
    with pytest.raises(SourceBridgeRefused):
        decode_key("another-setup", original)
    with pytest.raises(SourceBridgeRefused):
        option_key("original-setup", "unbounded-observer-prose")


def test_snapshot_injection_is_detected_by_existing_u1_parser(tmp_path: Path) -> None:
    # The actual Stage 1/2 fixture above verifies positive persistence;
    # this test rejects arbitrary bytes rather than silently repairing them.
    from suzka.r15.contracts import NarrativeSelfState

    root = NarrativeSelfState(1, 0, (), (), None, ())
    file = tmp_path / "state.json"
    save_test_root(file, root)
    assert load_test_root(file) == root
    changed = file.read_bytes().replace(b'"revision":0', b'"revision":1')
    assert changed != file.read_bytes()
    file.write_bytes(changed)
    with pytest.raises(ValueError):
        load_test_root(file)


def test_no_arbitrary_overwrite_of_synthetic_snapshot(tmp_path: Path) -> None:
    from suzka.r15.contracts import NarrativeSelfState

    state = NarrativeSelfState(1, 0, (), (), None, ())
    file = tmp_path / "state.json"
    save_test_root(file, state)
    with pytest.raises(FileExistsError):
        save_test_root(file, state)


def test_missing_e_record_denies_s1_before_e2(tmp_path: Path) -> None:
    app, provider, source, manifest, capture = scenario(tmp_path)
    stage_path = capture.with_name(capture.name + ".s1.json")
    with TestClient(app) as client:
        owners = TrialOwnerRead(
            client.app.state.event_journal,
            ExperienceStore.from_memory_root(source / "chroma"),
            client.app.state.memory_system,
        )
        observed: list[str] = []

        def transport(message: str) -> dict[str, str]:
            if len(observed) == 3:
                saved_stage = json.loads(stage_path.read_text(encoding="utf-8"))
                e_episode = saved_stage["turns"][1]["episode_id"]
                # Deliberate mutation of a disposable synthetic DB1 test copy.
                client.app.state.memory_system.db1.delete(ids=[e_episode])
                with pytest.raises(SourceBridgeRefused):
                    create_s1_root(manifest, stage_path, owners)
                raise SourceBridgeRefused("test stopped before E2 on missing E")
            observed.append(message)
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            data = response.json()
            return {"episode_id": data["episode_id"], "response": data["response"]}

        with pytest.raises(SourceBridgeRefused):
            run_trial(manifest, capture, transport)
        assert provider.calls == 3
        assert stage_path.exists()
        assert not capture.exists()


def test_removed_e2_experience_refuses_s2_and_retains_s1(tmp_path: Path) -> None:
    app, provider, source, manifest, capture = scenario(tmp_path)
    with TestClient(app) as client:
        owners = TrialOwnerRead(
            client.app.state.event_journal,
            ExperienceStore.from_memory_root(source / "chroma"),
            client.app.state.memory_system,
        )

        def transport(message: str) -> dict[str, str]:
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            data = response.json()
            return {"episode_id": data["episode_id"], "response": data["response"]}

        run_trial(manifest, capture, transport)
        checkpoint = capture.with_name(capture.name + ".s1.json")
        prior = create_s1_root(manifest, checkpoint, owners)
        stage_digest = prior.canonical_bytes()
        e2_episode = json.loads(capture.read_text())["turns"][3]["episode_id"]
        records_root = owners.experience.records_root
        current_e2 = []
        for item in records_root.iterdir():
            rec = owners.experience.load_committed_current(item.name).record
            if rec.source_episode_id == e2_episode:
                current_e2.append(item / "0.json")
        assert len(current_e2) == 1
        # Remove/restore ONLY the disposable test Source to simulate disappearance.
        victim = current_e2[0]
        original = victim.read_bytes()
        victim.unlink()
        try:
            with pytest.raises(SourceBridgeRefused):
                create_s2_root(manifest, capture, prior, owners)
            assert prior.canonical_bytes() == stage_digest
        finally:
            victim.write_bytes(original)
            victim.chmod(0o600)
        assert create_s2_root(manifest, capture, prior, owners).revision == 2
        assert provider.calls == 5
