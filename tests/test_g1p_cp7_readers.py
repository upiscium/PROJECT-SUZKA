"""G1-P CP7 cross-reader checks on exact standalone R14 base.

CP7 here is an *API exposure boundary*, not a claim that every private R14
read model publishes at R07 external terminal, nor that Memory is G2-pure.
The R14/Working Memory model is allowed to compute within a serialized event.
These tests assert its provisional Context never bypasses the Context GET
terminal gate or a response-emitting downstream consumer.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from suzka.api.context_read_model import ContextPublicationStatus
from suzka.runtime import AgentEventType
from test_fastapi_backend import ThinkingProvider, _client, admin_headers
from test_g1p_context_publication import _BarrierProvider, _public, _send_chat


def _get(client, path: str, *, query: str | None = None):
    args = {"params": {"query": query}} if query is not None else {}
    return client.get(path, headers=admin_headers(), **args)


def _seed_cross_reader_fixtures(client) -> tuple[str, str]:
    """Pre-existing source rows for positive GETs, not G2 source authority.

    The R09-compatible legacy fixture writers are TEST-ONLY and do not create
    R07+B+R12 joined attestation or any R15 subjective admission.
    """
    memory = client.app.state.memory_system
    episode_id = memory.save_episodic(
        "CP7 historical episode input", "CP7 historical episode output"
    )
    semantic_id = memory.save_legacy_semantic("CP7 historical semantic fixture")
    assert _get(client, f"/api/memory/episodes/{episode_id}").status_code == 200
    assert _get(client, f"/api/memory/semantic/{semantic_id}").status_code == 200
    return episode_id, semantic_id


def _external_reader_probe(
    client,
    *,
    candidate_id: str,
    session: str,
    episode_id: str,
    semantic_id: str,
    expected_context_status: int,
    prior_context: dict[str, object],
    prior_values: dict[str, object],
) -> None:
    """Exercise real HTTP consumers while the serialized event is frozen.

    The prior /api/values projection is unchanged by a CHAT event. R10 emotion
    and source-owned Memory are independently mutable and are NOT asserted to
    be the same terminal generation as G1-P Context.
    """
    listed = _public(client)
    detail = _get(client, f"/api/contexts/{candidate_id}")
    assert listed.status_code == expected_context_status
    assert detail.status_code == (404 if expected_context_status == 200 else 503)
    if expected_context_status == 200:
        assert listed.json() == prior_context
        assert listed.headers["x-context-visibility"] == "ready"
        assert detail.json() == {"detail": "Context not found"}
    else:
        assert listed.json() == {"detail": "Committed Context unavailable"}
        assert detail.json() == {"detail": "Committed Context unavailable"}
    for response in (listed, detail):
        assert session not in response.text
        assert candidate_id not in response.text
        if expected_context_status == 503:
            assert "x-context-terminal-sequence" not in response.headers

    # The admin-only Context router MUST deny a missing or forged token even
    # when the trusted caller is in a 503 publication window.
    assert client.get("/api/contexts").status_code == 401
    assert client.get(f"/api/contexts/{candidate_id}").status_code == 401
    forged_headers = {"X-SUZKA-Admin-Token": "cp7-invalid-credential"}
    assert client.get("/api/contexts", headers=forged_headers).status_code == 401
    assert (
        client.get(f"/api/contexts/{candidate_id}", headers=forged_headers).status_code
        == 401
    )

    # Positive existing DB1/DB2 source IDs are fixture-only, not proof of G2.
    # A CHAT cannot mutate the R11 Value root under this event; if the configured
    # R11 Value list is empty, missing detail/revisions must return generic 404.
    values = prior_values["values"]
    assert isinstance(values, list)
    value_id = values[0]["value_id"] if values else "cp7-missing-value"
    expected_value_detail = 200 if values else 404

    # All registered HTTP GET paths are inventoried by the router surface test.
    readers = {
        "value": _get(client, "/api/values"),
        "value-config-seeds": _get(client, "/api/values/config-seeds"),
        "value-detail": _get(client, f"/api/values/{value_id}"),
        "value-revisions": _get(client, f"/api/values/{value_id}/revisions"),
        "emotion": _get(client, "/api/state/emotion"),
        # This is the actual in-flight CHAT message, not an unrelated probe.
        # DB1 retrieval may differ by lifecycle; we inspect only Context-derived
        # metadata without claiming R12/G2 terminal source attestation.
        "memory": _get(
            client,
            "/api/memory/search",
            query="G1-P synthetic committed-context test",
        ),
        "memory-episode": _get(client, f"/api/memory/episodes/{episode_id}"),
        "memory-semantic": _get(client, f"/api/memory/semantic/{semantic_id}"),
        "adapters": _get(client, "/api/adapters"),
        "health": client.get("/health"),
    }
    for name, response in readers.items():
        wanted_status = (
            expected_value_detail
            if name in {"value-detail", "value-revisions"}
            else 200
        )
        assert response.status_code == wanted_status, (name, response.text[:250])
        assert candidate_id not in response.text, name
        assert session not in response.text, name
        assert "x-context-terminal-sequence" not in response.headers
    assert readers["value"].json() == prior_values


@pytest.mark.parametrize("phase", ["model", "external_finalize", "terminal_pending"])
def test_cp7_context_derived_reader_matrix_does_not_expose_pending_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """CP1–CP3: all inventoried external reads during exact phase barriers."""
    started, release = Event(), Event()
    provider = _BarrierProvider() if phase == "model" else ThinkingProvider()
    session = f"cp7-reader-{phase}"
    with _client(tmp_path, provider=provider) as client:
        episode_id, semantic_id = _seed_cross_reader_fixtures(client)
        prior_context = _public(client).json()
        prior_values = _get(client, "/api/values").json()
        prior_attention = client.app.state.main_loop.attention_view()
        expected_event_sequence = (
            client.app.state.context_committed_reads.snapshot().terminal_sequence + 1
        )

        if phase == "external_finalize":
            owner = client.app.state.transaction_coordinator
            original = owner.finalize_event

            def paused_finalize(event, evidence):
                started.set()
                if not release.wait(20):
                    raise AssertionError("CP7 finalize barrier timed out")
                return original(event, evidence)

            monkeypatch.setattr(owner, "finalize_event", paused_finalize)
        elif phase == "terminal_pending":
            owner = client.app.state.state_recovery
            original = owner.complete_committed_event

            def paused_terminal(event, evidence):
                started.set()
                if not release.wait(20):
                    raise AssertionError("CP7 terminal barrier timed out")
                return original(event, evidence)

            monkeypatch.setattr(owner, "complete_committed_event", paused_terminal)
        else:
            started = provider.started
            release = provider.release

        with ThreadPoolExecutor(max_workers=1) as pool:
            incoming = pool.submit(_send_chat, client, session)
            try:
                assert started.wait(15), f"CP7 did not reach {phase}"
                live = client.app.state.main_loop.context_registry.state
                assert len(live.frames) == 1
                assert live.frames[0].source_session_id == session
                candidate_id = live.frames[0].context_id
                publication = client.app.state.context_committed_reads
                expected = 503 if phase == "terminal_pending" else 200
                assert publication.status is (
                    ContextPublicationStatus.TERMINAL_PUBLICATION_PENDING
                    if phase == "terminal_pending"
                    else ContextPublicationStatus.READY
                )
                _external_reader_probe(
                    client,
                    candidate_id=candidate_id,
                    session=session,
                    episode_id=episode_id,
                    semantic_id=semantic_id,
                    expected_context_status=expected,
                    prior_context=prior_context,
                    prior_values=prior_values,
                )
                current_attention = client.app.state.main_loop.attention_view()
                if phase == "model":
                    # R14 may have a private provisional owner state, but its
                    # published read view is still the pre-internal-commit one.
                    assert current_attention == prior_attention
                else:
                    # Existing R14 read projections publish at INTERNAL commit,
                    # before the Context externally-terminal publication gate.
                    # This private availability is NOT an R07/G2 source proof.
                    assert current_attention != prior_attention
                    assert current_attention.event is not None
                    assert (
                        current_attention.event.event_sequence
                        == expected_event_sequence
                    )
                assert current_attention is not None
            finally:
                release.set()
            assert incoming.result(timeout=45).status_code == 200

        committed = _public(client)
        assert committed.status_code == 200
        assert committed.headers["x-context-terminal-sequence"] == "1"
        assert committed.json()["contexts"][0]["source_session_id"] == session


@pytest.mark.parametrize("phase", ["model", "external_finalize", "terminal_pending"])
def test_cp7_second_debug_event_cannot_export_prompt_before_first_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """CP1–CP3/CP6: debug prompt escapes only after its ordered event resolves."""
    started, release, admitted_debug = Event(), Event(), Event()
    provider = _BarrierProvider() if phase == "model" else ThinkingProvider()
    with _client(tmp_path, provider=provider) as client:
        session = f"cp7-queued-{phase}"
        prior = _public(client).json()
        terminal_published, debug_handler_entered = Event(), Event()
        publication = client.app.state.context_committed_reads
        first_sequence = publication.snapshot().terminal_sequence + 1
        original_publish_terminal = publication.publish_terminal

        def observe_terminal_publication(**kwargs):
            result = original_publish_terminal(**kwargs)
            if kwargs.get("sequence") == first_sequence:
                terminal_published.set()  # AFTER actual whole Context view swap
            return result

        monkeypatch.setattr(
            publication, "publish_terminal", observe_terminal_publication
        )
        if phase == "model":
            started, release = provider.started, provider.release
        else:
            if phase == "external_finalize":
                owner = client.app.state.transaction_coordinator
                checkpoint = "finalize_event"
            else:
                owner = client.app.state.state_recovery
                checkpoint = "complete_committed_event"
            original = getattr(owner, checkpoint)
            first = True

            def paused_checkpoint(event, evidence):
                nonlocal first
                if first:
                    first = False
                    started.set()
                    if not release.wait(20):
                        raise AssertionError("CP7 debug barrier timed out")
                return original(event, evidence)

            monkeypatch.setattr(owner, checkpoint, paused_checkpoint)

        runtime = client.app.state.agent_runtime
        original_submit = runtime.submit

        def observed_submit(event_type, source, handler):
            if event_type is AgentEventType.DEBUG_CHAT:

                def assert_prior_terminal_before_debug_execution():
                    # Deterministic at the actual DEBUG handler boundary. A
                    # concurrent event regression cannot hide behind a fast
                    # one-shot check of a Future before it ran.
                    assert terminal_published.is_set(), (
                        "DEBUG executed before previous event terminal publication"
                    )
                    debug_handler_entered.set()
                    return handler()

                submitted = original_submit(
                    event_type, source, assert_prior_terminal_before_debug_execution
                )
                admitted_debug.set()
                return submitted
            return original_submit(event_type, source, handler)

        monkeypatch.setattr(runtime, "submit", observed_submit)

        def debug_request():
            return client.post(
                "/api/chat/debug",
                headers=admin_headers(),
                json={
                    "message": "CP7 queued debug request",
                    "attachments": [],
                    "client_session_id": f"{session}-later",
                    "debug": True,
                },
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(_send_chat, client, session)
            try:
                assert started.wait(15), "First event never reached phase barrier"
                live = client.app.state.main_loop.context_registry.state
                assert live.frames[0].source_session_id == session
                second_future = pool.submit(debug_request)
                assert admitted_debug.wait(15), "Debug was not submitted"
                # Inspect the *actual* runtime queue under its condition lock.
                # The first CHAT is still active and DEBUG is enqueued behind
                # it; the guarded DEBUG handler also checks publication when
                # it eventually executes after this barrier is released.
                with runtime._condition:
                    assert runtime._active is not None
                    assert runtime._active[1].event_type is AgentEventType.CHAT
                    queued_debug = [
                        item
                        for item in runtime._pending
                        if item.event.event_type is AgentEventType.DEBUG_CHAT
                    ]
                    assert len(queued_debug) == 1
                    assert not queued_debug[0].future.done()
                assert not second_future.done(), (
                    "Debug response escaped before terminal"
                )
                if phase == "terminal_pending":
                    assert _public(client).status_code == 503
                else:
                    assert _public(client).json() == prior
            finally:
                release.set()
            assert first_future.result(timeout=45).status_code == 200
            debug_response = second_future.result(timeout=45)
            assert debug_response.status_code == 200
            assert terminal_published.is_set()
            assert debug_handler_entered.is_set()
            # This DEBUG result is allowed to see the prior, now-committed
            # episode; it MUST NOT return before the first terminal checkpoint.
            assert "prompt" in debug_response.json()
        assert _public(client).headers["x-context-terminal-sequence"] == "2"


def test_cp7_precommit_abort_does_not_expose_context_via_other_api_reads(
    tmp_path: Path,
) -> None:
    """CP4/CP5: failed model handler restores R09 and its public projection."""
    provider = _BarrierProvider(fail=True)
    with _client(tmp_path, provider=provider) as client:
        episode_id, semantic_id = _seed_cross_reader_fixtures(client)
        prior_context = _public(client).json()
        prior_values = _get(client, "/api/values").json()
        with ThreadPoolExecutor(max_workers=1) as pool:
            event_future = pool.submit(_send_chat, client, "cp7-precommit-fail")
            try:
                assert provider.started.wait(15)
                candidate_id = client.app.state.main_loop.context_registry.state.frames[
                    0
                ].context_id
                _external_reader_probe(
                    client,
                    candidate_id=candidate_id,
                    session="cp7-precommit-fail",
                    episode_id=episode_id,
                    semantic_id=semantic_id,
                    expected_context_status=200,
                    prior_context=prior_context,
                    prior_values=prior_values,
                )
            finally:
                provider.release.set()
            with pytest.raises(RuntimeError, match="fixture model failure"):
                event_future.result(timeout=45)
        assert _public(client).json() == prior_context
        assert client.app.state.main_loop.context_registry.state.frames == ()


def test_cp7_post_terminal_publication_failure_never_reenables_context_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CP4: no provisional Context survives the degraded/restart boundary."""
    session = "cp7-crash-before-view-swap"
    with _client(tmp_path) as client:
        episode_id, semantic_id = _seed_cross_reader_fixtures(client)
        public = client.app.state.context_committed_reads
        prior_values = _get(client, "/api/values").json()

        def fail_publication(**_kwargs):
            raise RuntimeError("test-only CP7 post-terminal publication failure")

        monkeypatch.setattr(public, "publish_terminal", fail_publication)
        result = _send_chat(client, session)
        assert result.status_code == 500
        candidate_id = client.app.state.main_loop.context_registry.state.frames[
            0
        ].context_id
        assert public.status is ContextPublicationStatus.DEGRADED
        _external_reader_probe(
            client,
            candidate_id=candidate_id,
            session=session,
            episode_id=episode_id,
            semantic_id=semantic_id,
            expected_context_status=503,
            prior_context={},
            prior_values=prior_values,
        )
    # A fresh startup is a reconciliation event, NOT the previously degraded
    # process silently setting READY from a stale handle.
    with _client(tmp_path) as restarted:
        restored = _public(restarted)
        assert restored.status_code == 200
        assert restored.json()["contexts"][0]["source_session_id"] == session


def test_cp7_registered_http_get_routes_match_source_audited_inventory(
    tmp_path: Path,
) -> None:
    """A future GET route forces an explicit CP7 inventory review, not silence."""
    with _client(tmp_path) as client:
        exposed_gets = {
            route.path
            for route in client.app.routes
            if "GET" in (getattr(route, "methods", None) or ())
            and (route.path.startswith("/api/") or route.path == "/health")
        }
        assert exposed_gets == {
            "/api/contexts",
            "/api/contexts/{context_id}",
            "/api/values/config-seeds",
            "/api/values",
            "/api/values/{value_id}",
            "/api/values/{value_id}/revisions",
            "/api/memory/search",
            "/api/memory/episodes/{episode_id}",
            "/api/memory/semantic/{memory_id}",
            "/api/state/emotion",
            "/api/adapters",
            "/health",
        }
