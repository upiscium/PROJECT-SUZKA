"""R15 U4b: real internal Actor task-attempt adoption with existing v10 WAL.

R07/R12/DB1/R09 and WAL owners are real isolated TestClient instances.
Only the text-generating provider is synthetic; task success is never proven.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from suzka.memory.experience_participant import experience_id_for_event
from suzka.r15.attempt_producer import TASK_CLASS, QualifiedSelfAttemptUnavailable
from suzka.r15.common import InterpretationStatus
from suzka.r15.contracts import VerifiedCompetence
from suzka.r15.self_model_system import SelfAttemptResult
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeExecutionError,
)
from suzka.runtime.agent_state import AgentStateSnapshotV10
from suzka.runtime.event_journal import EventLifecycle

from test_r15_u6b_runtime_use import _stages


def _review(client) -> tuple[SelfAttemptResult, object]:
    runtime = client.app.state.agent_runtime
    port = client.app.state.r15_state_port
    future = runtime.submit(
        AgentEventType.SELF_ATTEMPT_REVIEW,
        AgentEventSource.RUNTIME_SELF_ATTEMPT_REVIEW,
        port._review_source_qualified_self_attempt,
    )
    result = future.result(timeout=30)
    assert isinstance(result.value, SelfAttemptResult)
    assert result.value.event.event_id == result.event.event_id
    assert result.value.event.event_sequence == result.event.processing_sequence
    return result.value, result.event


@pytest.mark.parametrize("no_reveal", [False, True])
def test_real_owner_attempt_commit_restart_and_no_duplicate(
    tmp_path: Path, no_reveal: bool
) -> None:
    for stage, (client, _provider, refs) in enumerate(
        _stages(tmp_path, no_reveal=no_reveal)
    ):
        port = client.app.state.r15_state_port
        before = port.self_model.snapshot()
        assert before.revision == (0 if stage == 0 else 1)
        assert (
            before.capabilities == () if stage == 0 else len(before.capabilities) == 1
        )

        adopted, event = _review(client)
        assert adopted.changed is (stage == 0)
        assert adopted.snapshot.revision == 1
        assert len(adopted.snapshot.revision_history) == 1
        assert len(adopted.snapshot.capabilities) == 1
        capability = adopted.snapshot.capabilities[0]
        assert capability.task_class_key == TASK_CLASS
        assert capability.attempted.status is InterpretationStatus.PRESENT
        assert capability.attempted.support[0].event.event_id == refs[1].event_id
        assert (
            capability.attempted.support[0].event.event_sequence
            == refs[1].event_sequence
        )
        assert capability.verified_competence is VerifiedCompetence.UNKNOWN
        view = port.self_model.task_view(TASK_CLASS)
        assert view.tracked and view.attempted_status is InterpretationStatus.PRESENT
        assert view.verified_competence is VerifiedCompetence.UNKNOWN
        assert not port.self_model.task_view("task:unrelated").tracked
        assert port.relationship.snapshot().revision == 0
        assert port.narrative.snapshot().revision == (1 if no_reveal else 2)

        committed = client.app.state.agent_state_store.load()
        assert isinstance(committed, AgentStateSnapshotV10)
        assert committed.self_model_state == adopted.snapshot.canonical_value()
        assert committed.self_model_state["revision"] == 1
        assert committed.last_processed_event_sequence == event.processing_sequence

        repeat, _ = _review(client)
        assert not repeat.changed
        assert repeat.snapshot == adopted.snapshot
        assert port.self_model.snapshot().revision == 1
        assert client.app.state.agent_state_store.load().self_model_state == (
            adopted.snapshot.canonical_value()
        )


@pytest.mark.parametrize("missing_index", [1, 3])
def test_missing_original_or_later_experience_never_adopts_then_recovers(
    tmp_path: Path,
    missing_index: int,
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, _provider, refs = next(stages)
    port = client.app.state.r15_state_port
    source_store = client.app.state.experience_store
    ref = refs[missing_index]
    path = source_store.record_path(
        experience_id_for_event(ref.event_id, ref.event_sequence), 0
    )
    saved = path.read_bytes()
    path.unlink()
    try:
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert port.self_model.snapshot().revision == 0
        assert (
            client.app.state.agent_state_store.load().self_model_state["revision"] == 0
        )
        assert any(
            row.lifecycle is EventLifecycle.FAILED
            and row.event_type is AgentEventType.SELF_ATTEMPT_REVIEW
            for row in client.app.state.event_journal.inspect().records
        )
    finally:
        path.write_bytes(saved)
        path.chmod(0o600)
    positive, _ = _review(client)
    assert positive.changed and positive.snapshot.revision == 1
    stages.close()


def test_unauthorized_event_cannot_adopt_and_cannot_be_called_directly(
    tmp_path: Path,
) -> None:
    stages = _stages(tmp_path, no_reveal=True)
    client, provider, _ = next(stages)
    port = client.app.state.r15_state_port
    with pytest.raises(QualifiedSelfAttemptUnavailable):
        port._review_source_qualified_self_attempt()
    invalid = client.app.state.agent_runtime.submit(
        AgentEventType.SELF_ATTEMPT_REVIEW,
        AgentEventSource.API_CHAT,
        port._review_source_qualified_self_attempt,
    )
    with pytest.raises(AgentRuntimeExecutionError):
        invalid.result(timeout=30)
    invalid_type = client.app.state.agent_runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.RUNTIME_SELF_ATTEMPT_REVIEW,
        port._review_source_qualified_self_attempt,
    )
    with pytest.raises(AgentRuntimeExecutionError):
        invalid_type.result(timeout=30)
    assert port.self_model.snapshot().revision == 0

    # A model/operator statement is untrusted conversation, not a U4 setter.
    response = client.post(
        "/api/chat",
        json={
            "message": "私はSuzkaの能力が完全に検証済みだと主張する。",
            "attachments": [],
        },
    )
    assert response.status_code == 200
    assert port.self_model.snapshot().revision == 0
    assert (
        port.self_model.task_view(TASK_CLASS).verified_competence
        is VerifiedCompetence.UNKNOWN
    )
    stages.close()


def test_handler_failure_after_revision_restores_exact_selfmodel(
    tmp_path: Path,
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, _, _ = next(stages)
    port = client.app.state.r15_state_port
    committed = client.app.state.agent_state_store.load()
    assert isinstance(committed, AgentStateSnapshotV10)

    def fail_after_local_owner_change() -> None:
        result = port._review_source_qualified_self_attempt()
        assert result.changed
        raise RuntimeError("synthetic test-only handler failure")

    future = client.app.state.agent_runtime.submit(
        AgentEventType.SELF_ATTEMPT_REVIEW,
        AgentEventSource.RUNTIME_SELF_ATTEMPT_REVIEW,
        fail_after_local_owner_change,
    )
    with pytest.raises(AgentRuntimeExecutionError):
        future.result(timeout=30)
    assert port.self_model.snapshot().revision == 0
    assert port.self_model.snapshot().canonical_value() == committed.self_model_state
    assert (
        client.app.state.agent_state_store.load().self_model_state
        == committed.self_model_state
    )
    accepted, _ = _review(client)
    assert accepted.changed and accepted.snapshot.revision == 1
    stages.close()


def test_cold_state_and_unbound_runtime_never_claim_attempt(
    tmp_path: Path,
) -> None:
    from test_r15_p1_owner_bridge import scenario
    from fastapi.testclient import TestClient

    app, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        port = client.app.state.r15_state_port
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        assert port.self_model.snapshot().revision == 0
        assert not port.self_model.task_view(TASK_CLASS).tracked


def test_ambiguous_participant_context_refuses_positive_self_attempt(
    tmp_path: Path,
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, _provider, _refs = next(stages)
    port = client.app.state.r15_state_port
    context = client.app.state.main_loop.context_registry
    context_id = context.current_context_id
    assert context_id is not None
    context.add_participant_ref(context_id, "late-unverified-person")
    with pytest.raises(AgentRuntimeExecutionError):
        _review(client)
    assert port.self_model.snapshot().revision == 0
    assert client.app.state.agent_state_store.load().self_model_state["revision"] == 0
    stages.close()


def test_observed_task_view_cannot_upgrade_verified_competence(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    stages = _stages(tmp_path, no_reveal=False)
    client, _provider, _refs = next(stages)
    positive, _ = _review(client)
    assert positive.changed
    view = client.app.state.r15_state_port.self_model.task_view(TASK_CLASS)
    with pytest.raises((TypeError, ValueError)):
        replace(view, verified_competence="verified")
    assert view.verified_competence is VerifiedCompetence.UNKNOWN
    assert (
        client.app.state.agent_state_store.load().self_model_state
        == positive.snapshot.canonical_value()
    )
    stages.close()


def test_restored_legal_but_unqualified_task_is_not_a_positive_u4_view() -> None:
    from suzka.r15.self_model_system import SelfModelSystem
    from test_r15_u4_self_model_system import _synthetic_self_root

    owner = SelfModelSystem()
    owner._restore_technical_for_agent_state(_synthetic_self_root())
    assert owner.snapshot().revision == 1  # U5 preserves legal typed payloads
    view = owner.task_view("task:python")
    assert not view.tracked
    assert view.attempted_status is InterpretationStatus.UNKNOWN
    assert view.verified_competence is VerifiedCompetence.UNKNOWN


def test_retired_source_after_admission_refuses_new_review_not_past_revision(
    tmp_path: Path,
) -> None:
    stages = _stages(tmp_path, no_reveal=False)
    client, _provider, refs = next(stages)
    accepted, _ = _review(client)
    assert accepted.changed
    store = client.app.state.experience_store
    source = refs[1]
    path = store.record_path(
        experience_id_for_event(source.event_id, source.event_sequence), 0
    )
    saved = path.read_bytes()
    path.unlink()
    try:
        with pytest.raises(AgentRuntimeExecutionError):
            _review(client)
        # A committed historical attempt is retained. Its read view is not a
        # fresh Source assertion: no new use/admission passes with retired E.
        view = client.app.state.r15_state_port.self_model.task_view(TASK_CLASS)
        assert view.tracked
        assert view.verified_competence is VerifiedCompetence.UNKNOWN
        assert (
            client.app.state.agent_state_store.load().self_model_state
            == accepted.snapshot.canonical_value()
        )
    finally:
        path.write_bytes(saved)
        path.chmod(0o600)
    resumed, _ = _review(client)
    assert not resumed.changed
    stages.close()


def test_restored_same_task_with_nonproducer_proof_is_not_positive_u4(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from suzka.r15.common import RevisionReason
    from suzka.r15.contracts import SelfModelState

    stages = _stages(tmp_path, no_reveal=False)
    client, _, _ = next(stages)
    owner = client.app.state.r15_state_port.self_model
    admitted, _ = _review(client)
    good = admitted.snapshot
    assert owner.task_view(TASK_CLASS).tracked
    capability = good.capabilities[0]
    proof = good.revision_history[0]
    evidence = capability.attempted.support[0]
    assert evidence.event is not None

    adversarial_roots = (
        SelfModelState(
            1,
            1,
            (),
            (
                replace(
                    capability, attempted=replace(capability.attempted, confidence=1)
                ),
            ),
            None,
            (proof,),
        ),
        SelfModelState(
            1,
            1,
            (),
            (
                replace(
                    capability,
                    attempted=replace(capability.attempted, statement_id="e" * 64),
                ),
            ),
            None,
            (proof,),
        ),
        SelfModelState(
            1,
            1,
            (),
            (capability,),
            None,
            (replace(proof, target_id="e" * 64),),
        ),
        SelfModelState(
            1,
            1,
            (),
            (capability,),
            None,
            (replace(proof, before_digest="e" * 64),),
        ),
        SelfModelState(
            1,
            1,
            (),
            (capability,),
            None,
            (replace(proof, support_digests=("e" * 64,)),),
        ),
        SelfModelState(
            1,
            1,
            (),
            (capability,),
            None,
            (replace(proof, reason=RevisionReason.QUALIFIED_REINTERPRETATION),),
        ),
        SelfModelState(
            1,
            1,
            (),
            (capability,),
            None,
            (replace(proof, event=evidence.event),),
        ),
    )
    try:
        for malformed_producer_shape in adversarial_roots:
            # These are legal U1 typed values; they are not valid records
            # produced by this source-qualified U4 path.
            restored = SelfModelState.from_json(
                malformed_producer_shape.canonical_bytes()
            )
            owner._restore_technical_for_agent_state(restored)
            view = owner.task_view(TASK_CLASS)
            assert not view.tracked
            assert view.attempted_status is InterpretationStatus.UNKNOWN
            assert view.verified_competence is VerifiedCompetence.UNKNOWN
    finally:
        owner._restore_technical_for_agent_state(good)
    assert owner.task_view(TASK_CLASS).tracked
    stages.close()
