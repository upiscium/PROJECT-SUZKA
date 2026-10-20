"""R15 U6b: real isolated Chat uses qualified WAL-backed cognition or HOLD.

Only the provider's *answers* are synthetic; R07/R12/DB1, v10 StateWAL,
AgentRuntime, Context and NarrativeSelf admissions are actual test owners.
No public source projection API, Decision authority or verified human intent.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.protocol import run_trial
from suzka.api.server import create_app
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.memory.experience_participant import experience_id_for_event
from suzka.r15.private_projection import HOLD
from suzka.runtime.agent_state import AgentStateSnapshotV10

from test_r15_p1_owner_bridge import SequencedProvider, scenario
from test_r15_p1b_positive import _admit, _source_ref


RECALL = "前の案について、現在の暫定解釈を説明して。"
UNRELATED = "こんにちは。今日の調子は？"
HINT = "Internal bounded first-person provisional interpretation"


def _stages(tmp_path: Path, *, no_reveal: bool):
    app, provider, _source, manifest, capture = scenario(tmp_path, no_reveal=no_reveal)
    settings = app.state.settings
    refs = []
    with TestClient(app) as client:
        runtime = client.app.state.agent_runtime
        owner = client.app.state.r15_state_port.narrative
        assert client.app.state.r15_state_port._private_read_store is (
            client.app.state.agent_state_store
        )
        assert client.app.state.r15_state_port._private_read_runtime is runtime

        def transport(message: str) -> dict[str, str]:
            if len(refs) == 3:
                first = _admit(runtime, owner, tuple(refs))
                assert first.changed and first.snapshot.revision == 1
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            result = response.json()
            refs.append(_source_ref(client, result["episode_id"]))
            return {"episode_id": result["episode_id"], "response": result["response"]}

        run_trial(manifest, capture, transport)
        if not no_reveal:
            latest = _admit(runtime, owner, tuple(refs))
            assert latest.changed and latest.snapshot.revision == 2
        assert owner.snapshot().revision == (1 if no_reveal else 2)
        assert isinstance(
            client.app.state.agent_state_store.load(), AgentStateSnapshotV10
        )
        yield client, provider, tuple(refs)

    restored = create_app(settings)
    new_provider = SequencedProvider(no_reveal=no_reveal)
    restored.state.model_provider = new_provider
    restored.state.memory_system = DualMemorySystem(settings)
    restored.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(restored) as restarted:
        assert restarted.app.state.r15_state_port.narrative.snapshot().revision == (
            1 if no_reveal else 2
        )
        yield restarted, new_provider, tuple(refs)


def _capture_answer(provider: SequencedProvider, monkeypatch: pytest.MonkeyPatch):
    prompts = []

    def response(prompt: str) -> str:
        prompts.append(prompt)
        # A minimal provider proves that the *actual later response* depends
        # on the source-qualified current prompt value rather than stale A.
        if "- My current tentative option: B" in prompt:
            return "前の案について、現在の暫定解釈は案Bです（未検証）。"
        if "- My current tentative option: A" in prompt:
            return "前の案について、現在の暫定解釈は案Aです（未検証）。"
        return "過去の解釈の根拠は現在確認できません。"

    monkeypatch.setattr(provider, "generate", response)
    return prompts


@pytest.mark.parametrize("no_reveal", [False, True])
def test_next_real_chat_and_restart_use_current_provisional_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_reveal: bool
) -> None:
    expected = "A" if no_reveal else "B"
    for client, provider, _ in _stages(tmp_path, no_reveal=no_reveal):
        prompts = _capture_answer(provider, monkeypatch)
        port = client.app.state.r15_state_port
        before = port.narrative.snapshot()
        wal = client.app.state.agent_state_store.load()
        response = client.post("/api/chat", json={"message": RECALL, "attachments": []})
        assert response.status_code == 200
        assert f"案{expected}" in response.json()["response"]
        assert len(prompts) == 1
        assert HINT in prompts[0]
        assert "- My earlier tentative option: A" in prompts[0]
        assert f"- My current tentative option: {expected}" in prompts[0]
        assert "verified human intent" in prompts[0]
        assert "source_event_id" not in prompts[0]
        assert "Source IDs" in prompts[0]  # only the generic denial, never an ID
        assert port.narrative.snapshot() == before
        assert client.app.state.agent_state_store.load().narrative_self_state == (
            wal.narrative_self_state
        )


@pytest.mark.parametrize(
    "message",
    (
        UNRELATED,
        "前の案で採用したストレージ構成を説明して。",
        "以前の参照先URLを教えて。",
        "以前の解釈で使ったDocker設定について教えて。",
    ),
)
def test_nonrecall_chat_never_injects_private_interpretation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, provider, _ = next(stages)
    prompts = _capture_answer(provider, monkeypatch)
    response = client.post("/api/chat", json={"message": message, "attachments": []})
    assert response.status_code == 200
    assert len(prompts) == 1 and HINT not in prompts[0]
    stages.close()


@pytest.mark.parametrize("missing_index", [1, 3])
def test_missing_original_or_later_experience_forces_hold_no_prompt_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_index: int
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, provider, refs = next(stages)
    prompts = _capture_answer(provider, monkeypatch)
    store = client.app.state.experience_store
    ref = refs[missing_index]
    path = store.record_path(
        experience_id_for_event(ref.event_id, ref.event_sequence), 0
    )
    saved = path.read_bytes()
    path.unlink()
    before = client.app.state.r15_state_port.narrative.snapshot()
    try:
        response = client.post("/api/chat", json={"message": RECALL, "attachments": []})
        assert response.status_code == 200
        assert HINT not in prompts[-1]
        assert "確認できません" in response.json()["response"]
        assert client.app.state.r15_state_port.narrative.snapshot() == before
    finally:
        path.write_bytes(saved)
        path.chmod(0o600)
    stages.close()


def test_private_read_wal_unavailable_holds_without_extra_narrative_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, provider, _ = next(stages)
    prompts = _capture_answer(provider, monkeypatch)
    port = client.app.state.r15_state_port
    original_read = port._read_private_narrative_in_chat
    original_store_load = client.app.state.agent_state_store.load

    def unavailable_only_during_private_read(*args):
        with monkeypatch.context() as patch:
            patch.setattr(client.app.state.agent_state_store, "load", lambda: None)
            return original_read(*args)

    with monkeypatch.context() as patch:
        patch.setattr(
            port,
            "_read_private_narrative_in_chat",
            unavailable_only_during_private_read,
        )
        response = client.post("/api/chat", json={"message": RECALL, "attachments": []})
    assert response.status_code == 200
    assert len(prompts) == 1 and HINT not in prompts[0]
    assert client.app.state.agent_state_store.load == original_store_load
    assert port.narrative.snapshot().revision == 2
    stages.close()


def test_current_wal_mismatch_yields_hold_not_uncommitted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, provider, _ = next(stages)
    prompts = _capture_answer(provider, monkeypatch)
    store = client.app.state.agent_state_store
    saved = store.load()
    assert isinstance(saved, AgentStateSnapshotV10)
    port = client.app.state.r15_state_port
    original_read = port._read_private_narrative_in_chat
    from suzka.r15.state_port import R15TechnicalSnapshot

    def mismatched_only_during_private_read(*args):
        with monkeypatch.context() as patch:
            patch.setattr(port, "_capture_for_agent_state", R15TechnicalSnapshot.empty)
            return original_read(*args)

    with monkeypatch.context() as patch:
        patch.setattr(
            port, "_read_private_narrative_in_chat", mismatched_only_during_private_read
        )
        response = client.post("/api/chat", json={"message": RECALL, "attachments": []})
    assert response.status_code == 200
    assert HINT not in prompts[-1]
    assert store.load().narrative_self_state == saved.narrative_self_state
    stages.close()


def test_postgeneration_race_blocks_source_conditioned_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, provider, _ = next(stages)
    prompts = _capture_answer(provider, monkeypatch)
    port = client.app.state.r15_state_port
    original_read = port._read_private_narrative_in_chat
    reads = 0

    def race(*args):
        nonlocal reads
        reads += 1
        return original_read(*args) if reads == 1 else HOLD

    with monkeypatch.context() as patch:
        patch.setattr(port, "_read_private_narrative_in_chat", race)
        # The second private check takes place after generation, but before
        # the Chat transaction creates durable external-response participants.
        with pytest.raises(
            RuntimeError, match="private interpretation became unavailable"
        ):
            client.post("/api/chat", json={"message": RECALL, "attachments": []})
    assert len(prompts) == 1 and HINT in prompts[0]
    assert reads == 2
    assert port.narrative.snapshot().revision == 2
    stages.close()


def test_outside_chat_cannot_request_private_projection(tmp_path: Path) -> None:
    stages = _stages(tmp_path, no_reveal=True)
    client, _, _ = next(stages)
    port = client.app.state.r15_state_port
    assert (
        port._read_private_narrative_in_chat(
            object(), client.app.state.main_loop.context_registry.current_context_id
        )
        is HOLD
    )
    assert port.relationship.snapshot().revision == 0
    assert port.self_model.snapshot().revision == 0
    stages.close()


def test_cold_no_u3_does_not_invent_an_interpretation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        prompts = _capture_answer(provider, monkeypatch)
        response = client.post("/api/chat", json={"message": RECALL, "attachments": []})
        assert response.status_code == 200
        assert "確認できません" in response.json()["response"]
        assert len(prompts) == 1 and HINT not in prompts[0]
        assert client.app.state.r15_state_port.narrative.snapshot().revision == 0


def test_runtime_private_store_cannot_be_rebound_after_startup(tmp_path: Path) -> None:
    app, _, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        port = client.app.state.r15_state_port
        original = port._private_read_store
        assert original is client.app.state.agent_state_store
        with pytest.raises((TypeError, RuntimeError)):
            port._bind_private_read_sources(object(), client.app.state.agent_runtime)
        assert port._private_read_store is original
