"""F15 DC-4 end to end on the REAL application, and the S0-c merge gate.

A real app (supervised, ``degraded_continuation.mode = "continue"``) is started on
state a previous run wrote; an open transaction needing ``memory.experience`` with
a damaged pending artifact is armed offline (the application is stopped, its
Journal lease released), so the quarantine is the real DC-4 path:
reconcile -> ``QUARANTINED`` -> ``CycleRestart`` -> the supervisor rebuilds -> the
empty store.

S0-c (D1 sec. 1, 5.2 item 8): with the Experience store emptied by a real
quarantine, R15 (an AgentState with a narrative revision and stale turn
references) and R16 (a stale motivation source) must produce per-operation
unavailability and NO repeated failure: the Runtime must never end FAILED and the
supervisor must never reach ``RESTART_LIMIT`` (which would stop the subject).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

import suzka.api.server as server
from dc4_kit import Rig
from dc_kit import QuarantineAccessSpy, assert_trees_equal, tree_snapshot
from suzka.config import Settings
from suzka.models.dummy_provider import DummyProvider
from suzka.runtime import AgentRuntimeStatus
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.degraded_continuation import DcState
from suzka.runtime.event_journal import AbortOutcome, EventLifecycle
from test_r15_p1_owner_bridge import settings_in

SERVING = DcState.SERVING


class _Provider(DummyProvider):
    loss_value = 10000.0

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return "synthetic fixture"

    def calculate_loss(self, context_text: str, target_text: str) -> float:
        self.calls += 1
        return self.loss_value


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    control = SimpleNamespace(make=lambda: _Provider())
    monkeypatch.setattr(server, "load_model_provider", lambda settings: control.make())
    return control


def _settings(tmp_path: Path, mode: str) -> Settings:
    settings = settings_in(tmp_path / "dc4")
    return settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            ),
            "degraded_continuation": settings.degraded_continuation.model_copy(
                update={"mode": mode}
            ),
        }
    )


def _fast(app: Any) -> Any:
    supervisor = app.state.dc_supervisor
    supervisor._restart_delay = lambda n: 0.05
    supervisor._reeval_delay = lambda n: 0.25
    return supervisor


def _chat(client: TestClient, message: str) -> dict[str, Any]:
    response = client.post("/api/chat", json={"message": message, "attachments": []})
    assert response.status_code == 200, response.text
    data: dict[str, Any] = response.json()
    return data


def arm_open_transaction(
    settings: Settings, *, semantic: bool = False
) -> tuple[Path, dict[str, bytes], str]:
    """Offline (the app is stopped): an open PRE_INTERNAL transaction needing
    ``memory.experience`` (or ``memory.semantic``) whose pending artifact is
    damaged.  Returns the store root, its byte snapshot before the quarantine,
    and the transaction id."""

    rig = Rig(settings.memory.persist_directory.parent, settings=settings, attach=True)
    try:
        sequence = rig.journal.inspect().processing_high_water + 1
        prepared = rig.prepare(
            "dc4-armed",
            sequence,
            experience=not semantic,
            semantic=semantic,
            commit_internal=False,
        )
        store = rig.semantic_store if semantic else rig.experience_store
        store.pending_path(prepared.transaction_id).write_text(json.dumps({"conflict": 1}))
        return store.root, tree_snapshot(store.root), prepared.transaction_id
    finally:
        rig.journal.close()


def arm_open_experience_transaction(settings: Settings) -> tuple[Path, dict[str, bytes], str]:
    return arm_open_transaction(settings)


def _terminal(client: TestClient, transaction_id: str) -> str | None:
    inspection = client.app.state.event_journal.inspect()
    for group, name in (
        (inspection.completed_transactions, "completed"),
        (inspection.reconciled_transactions, "reconciled"),
        (inspection.aborted_transactions, "aborted"),
    ):
        if any(t.transaction_id == transaction_id for t in group):
            return name
    return None


def test_real_app_quarantines_the_experience_store_and_keeps_serving(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    # run 1: an ordinary life, so the stores hold real history
    settings = _settings(tmp_path, "continue")
    with TestClient(server.create_app(settings)) as client:
        assert client.app.state.dc_supervisor.wait_serving(120)
        first = _chat(client, "hello one")
        _chat(client, "hello two")
    root, before, transaction_id = arm_open_experience_transaction(settings)

    # run 2: the real DC-4 path
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(120)
        assert supervisor.cycles_launched == 2  # detect + quarantine, then serve
        assert client.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        health = client.get("/health").json()
        assert health["dc_reason"] == "domain_quarantined"
        assert health["dc_participants"] == "memory.experience"
        assert _terminal(client, transaction_id) == "aborted"
        (aborted,) = [
            t
            for t in client.app.state.event_journal.inspect().aborted_transactions
            if t.transaction_id == transaction_id
        ]
        assert dict(aborted.abort_outcomes)["memory.experience"] is AbortOutcome.QUARANTINED
        epoch = client.get("/health/dc").json()["dc_epoch_id"]
        quarantined = root.with_name(root.name + ".quarantine-" + epoch)
        assert_trees_equal(before, tree_snapshot(quarantined))
        # the subject answers, and the new store holds only NEW records
        reply = _chat(client, "hello after the quarantine")
        assert reply["episode_id"] != first["episode_id"]
        assert not any(
            "dc4-armed" in p.name for p in root.rglob("*") if p.is_file() and p.parent != root
        )

    # run 3: a restart does not quarantine again and still shows the degraded domain
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(120)
        assert supervisor.cycles_launched == 1
        assert client.get("/health").json()["dc_reason"] == "domain_quarantined"
        with QuarantineAccessSpy() as refused:
            _chat(client, "hello again")
        assert refused == []


def test_mode_refuse_never_quarantines_and_never_creates_a_file(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _settings(tmp_path, "refuse")
    with TestClient(server.create_app(settings)) as client:
        _chat(client, "hello")
    root, before, transaction_id = arm_open_experience_transaction(settings)
    names_before = sorted(p.name for p in root.parent.iterdir())
    app = server.create_app(settings)
    with TestClient(app) as client:
        # legacy: the Runtime is not started, the start is DEGRADED-UP
        assert client.get("/health").json()["status"] == "degraded"
        assert _terminal(client, transaction_id) is None
    assert_trees_equal(before, tree_snapshot(root))
    assert sorted(p.name for p in root.parent.iterdir()) == names_before
    assert not (root.parent / "quarantine").exists()


# =========================================================== S0-c: R15 / R16


def _r15_phase_one(tmp_path: Path) -> tuple[Settings, tuple[Any, ...]]:
    """The R15 trial scenario: real chats, a narrative review admitted (revision
    1).  Returns the settings and the qualified turn references."""

    from experiments.r15_p1.protocol import run_trial
    from test_r15_p1_owner_bridge import scenario
    from test_r15_p1b_positive import _admit, _source_ref

    app, _provider, _source, manifest, capture = scenario(tmp_path)
    settings = app.state.settings
    with TestClient(app) as client:
        runtime = client.app.state.agent_runtime
        owner = client.app.state.r15_state_port.narrative
        refs: list[Any] = []

        def transport(message: str) -> dict[str, str]:
            if len(refs) == 3:
                first = _admit(runtime, owner, tuple(refs))
                assert first.changed and first.snapshot.revision == 1
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            data = response.json()
            refs.append(_source_ref(client, data["episode_id"]))
            return {"episode_id": data["episode_id"], "response": data["response"]}

        run_trial(manifest, capture, transport)
        assert owner.snapshot().revision == 1
    return settings, tuple(refs)


def test_s0c_r15_with_an_emptied_experience_store_does_not_loop_or_stop_the_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_r15_p1_owner_bridge import SequencedProvider
    from test_r15_p1b_positive import _admit

    settings, refs = _r15_phase_one(tmp_path)
    assert len(refs) >= 3
    continue_settings = settings.model_copy(
        update={
            "degraded_continuation": settings.degraded_continuation.model_copy(
                update={"mode": "continue"}
            )
        }
    )
    root, before, transaction_id = arm_open_experience_transaction(continue_settings)
    monkeypatch.setattr(
        server, "load_model_provider", lambda _settings: SequencedProvider()
    )

    app = server.create_app(continue_settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(180)
        assert supervisor.cycles_launched == 2
        runtime = client.app.state.agent_runtime
        assert runtime.status is AgentRuntimeStatus.ACCEPTING
        assert client.get("/health").json()["dc_reason"] == "domain_quarantined"
        # the old Experience records are GONE from the live store
        assert not (root / "records").exists() or not any((root / "records").iterdir())
        owner = client.app.state.r15_state_port.narrative
        assert owner.snapshot().revision == 1  # the AgentState still carries R15

        # (1) R15 re-review of the stale turns, repeated: no loop, no stop
        outcomes: list[str] = []
        causes: list[str] = []
        for _ in range(6):
            try:
                result = _admit(runtime, owner, refs)
                outcomes.append("changed" if result.changed else "unchanged")
            except Exception as error:  # a per-operation failure of THAT event
                outcomes.append(type(error).__name__)
                cause = error.__cause__ or error.__context__
                causes.append(f"{type(cause).__name__}:{getattr(cause, 'reason', None)}")
            assert runtime.status is AgentRuntimeStatus.ACCEPTING, outcomes
            assert supervisor.phase is SERVING, outcomes
        print("S0-C R15 outcomes:", outcomes, sorted(set(causes)))
        # the R15 mapping: a source the emptied store no longer holds is the
        # per-operation unavailability the D1 names, never an unhandled error
        assert set(causes) == {"QualifiedProposalUnavailable:JournalSourceReason.SOURCE_MISSING"}
        assert owner.snapshot().revision == 1  # nothing was admitted on missing evidence
        assert supervisor.cycles_launched == 2  # never restarted again
        # (2) the subject still answers, and a new turn creates a NEW Experience
        data = _chat(client, "a fresh turn after the quarantine")
        assert data["response"]
        assert runtime.status is AgentRuntimeStatus.ACCEPTING
        # (3) the failed reviews are per-event FAILED records; none is a fail-stop
        failed = [
            r
            for r in client.app.state.event_journal.inspect().records
            if r.lifecycle is EventLifecycle.FAILED
            and r.event_type is AgentEventType.NARRATIVE_REVIEW
        ]
        assert len(failed) == len(outcomes)
        assert (client.app.state.experience_store.root / "records").exists()  # new turn

    # a restart on the emptied store: SERVING at once, no new quarantine
    app = server.create_app(continue_settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(180)
        assert supervisor.cycles_launched == 1
        assert client.app.state.r15_state_port.narrative.snapshot().revision == 1


def test_s0c_r16_with_an_emptied_experience_store_does_not_loop_or_stop_the_subject(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    from suzka.r16.source_owner import _MotivationSourceRef

    settings = _settings(tmp_path, "continue")
    with TestClient(server.create_app(settings)) as client:
        assert client.app.state.dc_supervisor.wait_serving(120)
        state = client.app.state
        outcome = state.agent_runtime.submit(
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            lambda: state.main_loop.chat("synthetic source only", None),
        ).result(timeout=30)
        source = _MotivationSourceRef(
            outcome.event.event_id,
            outcome.event.processing_sequence,
            outcome.value.episode_id,
        )
    arm_open_experience_transaction(settings)

    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(180)
        assert supervisor.cycles_launched == 2
        state = client.app.state
        owner = state._motivation_source_owner
        results: list[str] = []
        for _ in range(8):  # a retry storm would show here
            try:
                value = owner._submit(source).result(timeout=30).value
                results.append(str(value))
            except Exception as error:
                results.append(type(error).__name__)
            assert state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING, results
            assert supervisor.phase is SERVING, results
        print("S0-C R16 results:", results)
        assert supervisor.cycles_launched == 2
        # the evidence is gone, so it is NEVER applied
        assert "interest_applied" not in results
        assert not state.main_loop.export_motivation_state().records
        _chat(client, "the subject still answers")


def test_s0c_r16_goal_adopted_on_an_experience_survives_that_experience_being_emptied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The R16 case the D1 names: AgentState holds an adopted Goal whose source
    Experience is gone.  Reviewing / re-proposing it must end as a per-operation
    result, repeatedly, with the Runtime ACCEPTING and no restart."""

    from test_r16_u2a_source_owner import _Provider as R16Provider
    from test_r16_u2a_source_owner import _app as r16_app
    from test_r16_u2b_goal_owner import chat, preexisting_interest, x_id
    from suzka.r16.goal_owner import _GoalSourceSelection

    base = settings_in(tmp_path / "s0c-goal")
    base = base.model_copy(
        update={
            "emotion": base.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            )
        }
    )
    with TestClient(r16_app(base)) as client:
        state = client.app.state
        x = chat(state, "source experience; fixture")
        preexisting_interest(state, x)
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        owner = state._goal_source_owner
        proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(timeout=30)
        goal_id = proposed.value.subject_id
        adopted = owner._submit_review(goal_id).result(timeout=30)
        assert adopted.value.status.value == "adopted"
        goals_before = state.main_loop.export_goal_state()
    continue_settings = base.model_copy(
        update={
            "degraded_continuation": base.degraded_continuation.model_copy(
                update={"mode": "continue"}
            )
        }
    )
    root, before, _transaction_id = arm_open_experience_transaction(continue_settings)
    monkeypatch.setattr(server, "load_model_provider", lambda _s: R16Provider())

    app = server.create_app(continue_settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(180)
        assert supervisor.cycles_launched == 2
        state = client.app.state
        assert not (root / "records").exists() or not any((root / "records").iterdir())
        assert state.main_loop.export_goal_state() == goals_before  # AgentState intact
        owner = state._goal_source_owner
        results: list[str] = []
        for _ in range(6):
            for call in (
                lambda: owner._submit_review(goal_id),
                lambda: owner._submit_proposal(_GoalSourceSelection(p, x)),
            ):
                try:
                    value = call().result(timeout=30).value
                    results.append(f"{value.status.value}:{value.reason}")
                except Exception as error:
                    results.append(type(error).__name__)
            assert state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING, results
            assert supervisor.phase is SERVING, results
        print("S0-C R16 goal results:", sorted(set(results)))
        assert supervisor.cycles_launched == 2  # never restarted again
        assert state.main_loop.export_goal_state() == goals_before
        _chat(client, "the subject still answers after the R16 holds")


# ====================================== S0-c: Semantic (why it is OFF by default)


def _semantic_orphan(settings: Settings) -> None:
    """A real Semantic revision finalized into the store AND its DB2 projection,
    so a quarantine of the store leaves an R12 projection row behind (the shape
    after any real Semantic consolidation)."""

    from dc4_kit import _semantic_participant, binding_for
    from suzka.runtime.event_journal import TransactionKind
    from suzka.runtime.transaction_coordinator import TransactionCoordinator

    rig = Rig(settings.memory.persist_directory.parent, settings=settings, attach=True)
    try:
        sequence = rig.journal.inspect().processing_high_water + 1
        event = rig.event("sem-old", sequence, sleep=True)
        transaction_id = TransactionCoordinator.derive_transaction_id(
            event, TransactionKind.EVENT_MUTATION
        )
        participant = _semantic_participant(
            rig.memory, rig.semantic_store, event, transaction_id, "old"
        )
        stub = SimpleNamespace(transaction_id=transaction_id, event=event)
        binding = binding_for(stub, "memory.semantic", participant.operation_digest)  # type: ignore[arg-type]
        participant.prepare(binding)
        participant.finalize(binding)
        assert rig.memory.db2.get()["ids"]
    finally:
        rig.journal.close()


def _wait_for(predicate: Any, timeout: float = 60.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def test_s0c_semantic_is_off_by_default_so_a_semantic_fault_is_left_as_it_was(
    tmp_path: Path, provider: SimpleNamespace
) -> None:
    settings = _settings(tmp_path, "continue")
    with TestClient(server.create_app(settings)) as client:
        assert client.app.state.dc_supervisor.wait_serving(120)
        _chat(client, "hello")
    _semantic_orphan(settings)
    root, before, _transaction_id = arm_open_transaction(settings, semantic=True)
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        _wait_for(lambda: supervisor.phase is DcState.RECONCILING)
        health = client.get("/health").json()
        assert health["dc_state"] == "RECONCILING"
        assert health["dc_reason"] == "external_participant_reconciliation_required"
    # the legacy outcome: nothing moved, nothing created, nothing journaled
    assert_trees_equal(before, tree_snapshot(root))
    assert not any(".quarantine-" in p.name for p in root.parent.iterdir())
    assert not (root.parent / "quarantine").exists()


def test_s0c_a_semantic_quarantine_would_fail_every_chat_which_is_why_it_is_off(
    tmp_path: Path, provider: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documents the S0-c finding (D1 F-7).  With Semantic enabled the quarantine
    itself works and the app serves, but any DB2 projection row of the old store
    makes EVERY chat raise ``SemanticMemoryReadError``.  When the read path is
    changed by a design decision this test must be revisited, not deleted."""

    from suzka.memory.dual_memory_system import SemanticMemoryReadError
    from suzka.runtime import domain_quarantine

    monkeypatch.setattr(
        domain_quarantine,
        "DEFAULT_QUARANTINE_PARTICIPANTS",
        frozenset({"memory.experience", "memory.semantic"}),
    )
    settings = _settings(tmp_path, "continue")
    with TestClient(server.create_app(settings)) as client:
        assert client.app.state.dc_supervisor.wait_serving(120)
        _chat(client, "hello")
    _semantic_orphan(settings)
    arm_open_transaction(settings, semantic=True)
    app = server.create_app(settings)
    supervisor = _fast(app)
    with TestClient(app) as client:
        assert supervisor.wait_serving(120)
        assert client.get("/health").json()["dc_participants"] == "memory.semantic"
        for _ in range(3):
            with pytest.raises(SemanticMemoryReadError):
                client.post(
                    "/api/chat", json={"message": "tell me what you know", "attachments": []}
                )
            assert client.app.state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
