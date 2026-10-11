"""G1-P: externally terminal public Context, *not* G1-C event-time capture.

All failures are evaluated against the API's LAST externally terminal read
model. These tests do not create B witnesses, R15 roots or source authority.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from suzka.api.context_read_model import (
    CommittedContextReadModel,
    ContextPublicationStatus,
    ContextPublicationUnavailable,
    PublishedContextView,
)
from suzka.runtime import ContextRegistry, ContextType
from test_fastapi_backend import ThinkingProvider, _client, admin_headers


def _empty_state() -> PublishedContextView:
    return PublishedContextView(ContextRegistry().state, 0, "0" * 64)


def _send_chat(client, session: str) -> object:
    return client.post(
        "/api/chat",
        json={
            "message": "G1-P synthetic committed-context test",
            "attachments": [],
            "client_session_id": session,
        },
    )


def _public(client):
    return client.get("/api/contexts", headers=admin_headers())


def test_private_model_admitted_get_linearizes_before_pending() -> None:
    first = _empty_state()
    m = CommittedContextReadModel(first, ready=True)
    admitted = m.snapshot()
    registry = ContextRegistry()
    registry.create("synthetic", ContextType.CONVERSATION, "fixture")
    following = PublishedContextView(registry.state, 1, "1" * 64)

    m.begin_terminal(event_id="event-1", sequence=1, expected_previous=admitted)
    assert m.status is ContextPublicationStatus.TERMINAL_PUBLICATION_PENDING
    assert admitted.state.frames == ()  # an already-admitted GET may finish
    with pytest.raises(ContextPublicationUnavailable):
        m.snapshot()  # a NEW GET must get 503

    m.publish_terminal(event_id="event-1", sequence=1, candidate=following)
    assert m.snapshot() == following
    assert m.snapshot().state.frames[0].context_id == "synthetic"


def test_private_model_rejects_stale_and_duplicate_terminal() -> None:
    model = CommittedContextReadModel(_empty_state(), ready=True)
    original = model.snapshot()
    with pytest.raises(ValueError, match="sequence"):
        model.begin_terminal(
            event_id="old", sequence=0, expected_previous=original
        )
    assert model.status is ContextPublicationStatus.READY
    with pytest.raises(ContextPublicationUnavailable):
        model.begin_terminal(
            event_id="stale", sequence=1, expected_previous=_empty_state()
        )
    assert model.status is ContextPublicationStatus.DEGRADED
    with pytest.raises(ContextPublicationUnavailable):
        model.snapshot()

    fresh = CommittedContextReadModel(_empty_state(), ready=True)
    initial = fresh.snapshot()
    fresh.begin_terminal(event_id="event", sequence=1, expected_previous=initial)
    with pytest.raises(ContextPublicationUnavailable):
        fresh.publish_terminal(
            event_id="wrong",
            sequence=1,
            candidate=PublishedContextView(initial.state, 1, "1" * 64),
        )
    assert fresh.status is ContextPublicationStatus.DEGRADED


def test_private_model_cannot_publish_uncommitted_view_by_restore() -> None:
    m = CommittedContextReadModel(_empty_state(), ready=True)
    original = m.snapshot()
    saved = m.begin_handler_restore()
    assert saved is original
    with pytest.raises(ContextPublicationUnavailable):
        m.snapshot()
    m.complete_handler_restore(saved)
    assert m.snapshot() is original
    m.degrade()
    assert m.begin_handler_restore() is None
    m.complete_handler_restore(original)  # old handle cannot defeat degraded
    with pytest.raises(ContextPublicationUnavailable):
        m.snapshot()


def test_context_get_never_reads_uncommitted_registry_directly(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        old = _public(client)
        assert old.status_code == 200
        assert old.json()["contexts"] == []
        assert old.headers["x-context-terminal-sequence"] == "0"
        live = client.app.state.main_loop.context_registry
        live.create("speculative-only", ContextType.CONVERSATION, "fixture")
        assert "speculative-only" in (x.context_id for x in live.state.frames)

        listed = _public(client)
        assert listed.status_code == 200
        assert listed.json() == old.json()
        assert listed.headers["x-context-terminal-sequence"] == "0"
        detail = client.get(
            "/api/contexts/speculative-only", headers=admin_headers()
        )
        assert detail.status_code == 404
        assert detail.json() == {"detail": "Context not found"}


class _BarrierProvider(ThinkingProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.started = Event()
        self.release = Event()
        self.fail = fail

    def generate(self, prompt: str) -> str:
        self.started.set()
        if not self.release.wait(20):
            raise RuntimeError("fixture model barrier timeout")
        if self.fail:
            raise RuntimeError("fixture model failure")
        return super().generate(prompt)


@pytest.mark.parametrize("fail", [False, True])
def test_inflight_model_get_sees_only_prior_terminal(
    tmp_path: Path, fail: bool
) -> None:
    provider = _BarrierProvider(fail=fail)
    with _client(tmp_path, provider=provider) as client:
        initial = _public(client)
        assert initial.status_code == 200
        assert initial.json()["contexts"] == []

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_send_chat, client, "g1p-barrier")
            try:
                assert provider.started.wait(15), "chat did not reach model barrier"
                # R09 already mutated during the active worker, but never published.
                live = client.app.state.main_loop.context_registry.state
                assert live.frames
                assert _public(client).json() == initial.json()
                assert client.get(
                    f"/api/contexts/{live.frames[0].context_id}",
                    headers=admin_headers(),
                ).status_code == 404
            finally:
                provider.release.set()
            if fail:
                with pytest.raises(RuntimeError, match="fixture model failure"):
                    future.result(timeout=40)
            else:
                assert future.result(timeout=40).status_code == 200
        if fail:
            assert _public(client).json() == initial.json()
        else:
            after = _public(client)
            assert after.status_code == 200
            assert len(after.json()["contexts"]) == 1
            assert after.headers["x-context-terminal-sequence"] == "1"


def test_after_internal_commit_before_external_finalization_is_not_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        coordinator = client.app.state.transaction_coordinator
        original = coordinator.finalize_event
        entered, release = Event(), Event()

        def stalled_finalize(event, evidence):
            entered.set()
            if not release.wait(20):
                raise RuntimeError("finalize barrier timeout")
            return original(event, evidence)

        monkeypatch.setattr(coordinator, "finalize_event", stalled_finalize)
        prior = _public(client).json()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_send_chat, client, "g1p-finalize")
            try:
                assert entered.wait(15)
                assert _public(client).status_code == 200
                assert _public(client).json() == prior
            finally:
                release.set()
            assert future.result(timeout=40).status_code == 200
        after = _public(client)
        assert len(after.json()["contexts"]) == 1


def test_terminal_pending_denies_new_get_until_atomic_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        recovery = client.app.state.state_recovery
        original = recovery.complete_committed_event
        entered, release = Event(), Event()

        def stalled_complete(event, evidence):
            entered.set()
            if not release.wait(20):
                raise RuntimeError("terminal barrier timeout")
            return original(event, evidence)

        monkeypatch.setattr(recovery, "complete_committed_event", stalled_complete)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_send_chat, client, "g1p-pending")
            try:
                assert entered.wait(15)
                pub = client.app.state.context_committed_reads
                assert pub.status is ContextPublicationStatus.TERMINAL_PUBLICATION_PENDING
                listed = _public(client)
                assert listed.status_code == 503
                assert listed.json() == {"detail": "Committed Context unavailable"}
                live = client.app.state.main_loop.context_registry.state
                assert live.frames
                detail = client.get(
                    f"/api/contexts/{live.frames[0].context_id}",
                    headers=admin_headers(),
                )
                assert detail.status_code == 503
                assert "g1p-pending" not in detail.text
            finally:
                release.set()
            assert future.result(timeout=40).status_code == 200

        ready = _public(client)
        assert ready.status_code == 200
        assert len(ready.json()["contexts"]) == 1
        assert ready.headers["x-context-terminal-sequence"] == "1"


def test_uncertain_terminal_failure_keeps_get_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        recovery = client.app.state.state_recovery

        def fail_terminal(event, evidence):
            raise RuntimeError("test-only terminal persistence failure")

        monkeypatch.setattr(recovery, "complete_committed_event", fail_terminal)
        outcome = _send_chat(client, "g1p-fail-terminal")
        assert outcome.status_code == 500
        assert client.app.state.context_committed_reads.status is (
            ContextPublicationStatus.DEGRADED
        )
        assert _public(client).status_code == 503
        assert client.get(
            "/api/contexts/conversation.default", headers=admin_headers()
        ).status_code == 503


def test_terminal_record_before_public_view_swap_crash_recovers_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The durable event may be COMPLETED even when publication fails. A fresh
    # in-process GET must not assert the stale READY generation in that gap.
    with _client(tmp_path) as client:
        public = client.app.state.context_committed_reads

        def fail_after_terminal(*_args, **_kwargs):
            raise RuntimeError("synthetic post-terminal view swap failure")

        monkeypatch.setattr(public, "publish_terminal", fail_after_terminal)
        outcome = _send_chat(client, "g1p-post-terminal-crash")
        assert outcome.status_code == 500
        assert public.status is ContextPublicationStatus.DEGRADED
        assert _public(client).status_code == 503

    # Startup is a distinct, authorized recovery phase. It may republish only
    # after its existing Journal/AgentState checks classify the saved root.
    with _client(tmp_path) as recovered:
        observed = _public(recovered)
        assert observed.status_code == 200
        assert len(observed.json()["contexts"]) == 1
        assert observed.json()["contexts"][0]["source_session_id"] == (
            "g1p-post-terminal-crash"
        )


def test_failure_restoration_already_degraded_cannot_reenable_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path) as client:
        coordinator = client.app.state.transaction_coordinator

        def fail_preparation(*_args, **_kwargs):
            raise RuntimeError("synthetic preparation failure")

        monkeypatch.setattr(coordinator, "prepare_result", fail_preparation)
        response = _send_chat(client, "g1p-fail-preparation")
        assert response.status_code == 500
        assert client.app.state.context_committed_reads.status is (
            ContextPublicationStatus.DEGRADED
        )
        assert _public(client).status_code == 503


def test_started_journal_checkpoint_failure_disables_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The STARTED append runs BEFORE handler/Context publication checkpoints.
    # Even though the old read model is structurally READY, an indeterminate
    # runtime state must prevent any new public GET.
    with _client(tmp_path) as client:
        journal = client.app.state.event_journal

        def fail_started(_event):
            raise RuntimeError("synthetic started journal failure")

        monkeypatch.setattr(journal, "append_started", fail_started)
        outcome = _send_chat(client, "g1p-started-fail")
        assert outcome.status_code in (500, 503)
        assert _public(client).status_code == 503


def test_frozen_context_view_recovers_after_clean_restart(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert _send_chat(client, "g1p-clean-restart").status_code == 200
        published = _public(client).json()
        assert published["contexts"]
        live = client.app.state.main_loop.context_registry.state
        assert published["current_context_id"] == live.current_context_id
        assert client.app.state.context_committed_reads.snapshot().state == live

    with _client(tmp_path) as restarted:
        restored = _public(restarted)
        assert restored.status_code == 200
        assert restored.json() == published
        assert restarted.app.state.context_committed_reads.status is ContextPublicationStatus.READY


def test_api_context_mutation_reply_and_public_generation_match(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert _send_chat(client, "g1p-admin-mutate").status_code == 200
        before = _public(client)
        cid = before.json()["current_context_id"]
        assert cid is not None
        result = client.post(
            f"/api/contexts/{cid}/suspend", headers=admin_headers()
        )
        assert result.status_code == 200
        assert result.json()["status"] == "suspended"
        assert result.json()["is_current"] is False

        after = _public(client)
        assert after.headers["x-context-terminal-sequence"] == "2"
        assert after.json()["current_context_id"] is None
        detail = client.get(f"/api/contexts/{cid}", headers=admin_headers())
        assert detail.status_code == 200
        assert detail.headers["x-context-terminal-sequence"] == (
            after.headers["x-context-terminal-sequence"]
        )
        assert detail.json()["status"] == "suspended"
        assert detail.json()["is_current"] is False
