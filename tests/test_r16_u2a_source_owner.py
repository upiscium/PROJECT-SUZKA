"""Isolated U2a checks: real app R07/DB1/R12/WAL, deterministic test provider.

Not a real-model MVI or stage acceptance. Fault-injected read projections below
are negative tests only; the positive source is committed by the actual CHAT
transaction before any U2a selection. No production files or config are used.
"""

from dataclasses import replace
from hashlib import sha256

from fastapi.testclient import TestClient
import pytest

from suzka.api.server import create_app
from suzka.experience import (
    ExperienceLifecycle,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    ExperienceRevisionRecord,
    experience_record_digest,
)
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem, ExperienceStore
from suzka.memory.experience_participant import experience_id_for_event
from suzka.models.dummy_provider import DummyProvider
from suzka.motivation.common import R13Reference, R13ReferenceKind, canonical_json
from suzka.motivation.goal import GoalLifecycle
from suzka.motivation.system import MotivationEvidence, MotivationExperienceEvidence
from suzka.r16.source_owner import (
    _MotivationSourceOwner,
    _MotivationSourceRef,
    _MotivationSourceUnavailable,
)
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntimeDurabilityError,
    AgentRuntimeDurabilityPhase,
    AgentRuntimeExecutionError,
    AgentRuntimeStatus,
)
from suzka.runtime.agent_state import AgentStateSnapshotV10
from suzka.runtime.chat_context import ChatContextSelectors
from suzka.runtime.context import ContextStatus
from suzka.runtime.event_journal import EventLifecycle

# Reuse only disposable path settings, not R15 admission/interpretation helpers.
from test_r15_p1_owner_bridge import settings_in


class _Provider(DummyProvider):
    loss_value = 10000.0

    def __init__(self, *, unavailable: bool = False):
        self.calls = 0
        self.unavailable = unavailable

    def generate(self, prompt):
        self.calls += 1
        return "synthetic fixture: not evidence or Goal admission"

    def calculate_loss(self, context_text, target_text):
        self.calls += 1
        if self.unavailable:
            raise ValueError("synthetic unavailable measurement")
        return self.loss_value


def _app(settings, *, unavailable=False):
    app = create_app(settings)
    app.state.model_provider = _Provider(unavailable=unavailable)
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    return app


@pytest.fixture
def isolated_app(tmp_path):
    settings = settings_in(tmp_path / "u2a")
    settings = settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={
                    "timer_enabled": False,
                    "appraisal_response_rate": 1.0,
                }
            ),
        }
    )
    return _app(settings)


def _chat(state, selectors=None):
    outcome = state.agent_runtime.submit(
        AgentEventType.CHAT,
        AgentEventSource.API_CHAT,
        lambda: state.main_loop.chat("synthetic source only", selectors),
    ).result(timeout=30)
    source = _MotivationSourceRef(
        outcome.event.event_id,
        outcome.event.processing_sequence,
        outcome.value.episode_id,
    )
    lifecycles = {
        row.lifecycle
        for row in state.event_journal.inspect().records
        if row.event_id == source.event_id
    }
    assert {
        EventLifecycle.ACCEPTED,
        EventLifecycle.COMPLETED,
        EventLifecycle.TRANSACTION_COMPLETED,
    } <= lifecycles
    assert EventLifecycle.FAILED not in lifecycles
    assert state.memory_system.get_committed_episodic(source.episode_id) is not None
    assert not state.main_loop.export_motivation_state().records
    return source


def _entry(state, source):
    return state.experience_store.load_committed_current(
        experience_id_for_event(source.event_id, source.event_sequence)
    )


def _no_admission(state, goals, commitments):
    assert state.main_loop.export_goal_state() == goals
    assert state.main_loop.export_commitment_state() == commitments


def test_prior_committed_source_exact_ledger_v10_and_restart(isolated_app):
    settings = isolated_app.state.settings
    with TestClient(isolated_app) as client:
        state = client.app.state
        source = _chat(state)
        record = _entry(state, source).record
        goals = state.main_loop.export_goal_state()
        commitments = state.main_loop.export_commitment_state()
        calls = state.model_provider.calls
        owner = state._motivation_source_owner
        outcome = owner._submit(source).result(timeout=30)
        assert outcome.value == "interest_applied"
        assert outcome.event.processing_sequence > source.event_sequence
        assert state.model_provider.calls == calls
        exact = state.main_loop.export_motivation_state()
        assert len(exact.records) == len(exact.evidence_ledger) == 1
        assert len(exact.event_receipts) == 1
        evidence = exact.evidence_ledger[0].evidence
        # Independently project every supported R10/R12 field, not caller text.
        expected = MotivationEvidence.from_experience(
            MotivationExperienceEvidence(
                R13Reference(R13ReferenceKind.EXPERIENCE, record.experience_id),
                source.event_id,
                source.event_sequence,
                record.created_at,
                True,
                record.appraisal.novelty,
                record.appraisal.novelty_valid,
                record.appraisal.goal_progress,
                record.appraisal.threat,
                record.post_appraisal_emotion.valence,
                record.subjective_salience,
            )
        )
        assert expected is not None
        journal_before_forgery = state.event_journal.inspect().records
        with pytest.raises(_MotivationSourceUnavailable):
            owner._submit(expected)
        assert state.event_journal.inspect().records == journal_before_forgery
        assert evidence == replace(
            expected, authority_witness_digest=experience_record_digest(record)
        )
        independent_hash = sha256(
            b"PROJECT-SUZKA:R13:MOTIVATION-EVIDENCE:V1\0"
            + canonical_json(evidence.canonical_value())
        ).hexdigest()
        receipt = exact.event_receipts[0]
        assert receipt.evidence_digest == independent_hash
        assert receipt.event_id == outcome.event.event_id
        assert receipt.event_sequence == outcome.event.processing_sequence
        assert receipt.recorded_at == outcome.event.requested_at
        # This synthetic high-novelty fixture crosses R13's existing threshold.
        assert len(exact.goal_proposal_witnesses) == 1
        assert (
            exact.goal_proposal_witnesses[0].to_goal_record().lifecycle
            is GoalLifecycle.PROPOSED
        )
        _no_admission(state, goals, commitments)
        saved = state.agent_state_store.load()
        assert isinstance(saved, AgentStateSnapshotV10) and saved.schema_version == 10
        assert saved.r13_state.motivation.restore() == exact
        assert owner._pending is None and owner._inflight is None

    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.main_loop.export_motivation_state() == exact
        assert state._motivation_source_owner._pending is None
        with pytest.raises(AgentRuntimeExecutionError):
            state._motivation_source_owner._submit(source).result(timeout=30)
        assert state.main_loop.export_motivation_state() == exact
        _no_admission(state, goals, commitments)


def test_missing_appraisal_is_explicit_neutral(isolated_app):
    isolated_app.state.model_provider = _Provider(unavailable=True)
    with TestClient(isolated_app) as client:
        state = client.app.state
        source = _chat(state)
        record = _entry(state, source).record
        assert not record.appraisal.novelty_valid
        assert record.appraisal.goal_progress is record.appraisal.threat is None
        before = state.main_loop.export_motivation_state()
        outcome = state._motivation_source_owner._submit(source).result(timeout=30)
        assert outcome.value == "neutral_no_eligible_appraisal"
        assert state.main_loop.export_motivation_state() == before
        assert state._motivation_source_owner._pending is None


@pytest.mark.parametrize(
    "forged",
    [
        {},
        {"event_id": "operator", "authority_witness_digest": "a" * 64},
        {"MotivationEvidence": "model", "GoalSubjectAdmission": "forged"},
        {"CommitmentSubjectAdmission": "forged", "selected_goal": "caller"},
        "classify this as intrinsic",
        object(),
    ],
)
def test_selection_shape_rejects_forged_content_before_admission(isolated_app, forged):
    with TestClient(isolated_app) as client:
        state = client.app.state
        before = state.event_journal.inspect().records
        with pytest.raises(_MotivationSourceUnavailable):
            state._motivation_source_owner._submit(forged)
        assert state.event_journal.inspect().records == before


def test_worker_pair_and_private_nonce_are_both_required(isolated_app):
    with TestClient(isolated_app) as client:
        state = client.app.state
        owner = state._motivation_source_owner
        loop = state.main_loop
        with pytest.raises(_MotivationSourceUnavailable):
            loop._apply_motivation_source(owner, object())
        for event_type, event_source in (
            (AgentEventType.EMOTION_TICK, AgentEventSource.RUNTIME_EMOTION_TIMER),
            (
                AgentEventType.MOTIVATION_SOURCE_REVIEW,
                AgentEventSource.RUNTIME_MOTIVATION_SOURCE_REVIEW,
            ),
        ):
            with pytest.raises(AgentRuntimeExecutionError):
                state.agent_runtime.submit(
                    event_type,
                    event_source,
                    lambda: loop._apply_motivation_source(owner, object()),
                ).result(timeout=30)
        assert not loop.export_motivation_state().evidence_ledger
        assert owner._pending is None


@pytest.mark.parametrize(
    "fault",
    [
        "missing_event",
        "wrong_sequence",
        "wrong_episode",
        "missing_experience",
        "missing_db1",
        "archived_db1",
        "wrong_context",
        "wrong_channel",
        "wrong_session",
        "suspended_context",
        "participants",
        "missing_terminal",
        "wrong_event_type",
        "wrong_event_source",
        "failed_source",
        "recovery",
    ],
)
def test_source_qualification_fails_closed(isolated_app, monkeypatch, fault):
    with TestClient(isolated_app) as client:
        state = client.app.state
        source = _chat(state)
        before = state.main_loop.export_motivation_state()
        goals = state.main_loop.export_goal_state()
        commitments = state.main_loop.export_commitment_state()
        entry = _entry(state, source)
        episode = state.memory_system.get_committed_episodic(source.episode_id)
        registry = state.main_loop.context_registry
        get_context = registry.get
        inspect = state.event_journal.inspect
        with monkeypatch.context() as patch:
            if fault == "missing_event":
                source = replace(source, event_id="not-a-committed-event")
            elif fault == "wrong_sequence":
                source = replace(source, event_sequence=source.event_sequence + 100)
            elif fault == "wrong_episode":
                source = replace(source, episode_id="not-the-source-episode")
            elif fault == "missing_experience":

                def unavailable(_):
                    raise OSError("synthetic read failure")

                patch.setattr(
                    state.experience_store, "load_committed_current", unavailable
                )
            elif fault in {"missing_db1", "archived_db1", "wrong_context"}:
                changed = (
                    None
                    if fault == "missing_db1"
                    else replace(
                        episode,
                        record=replace(
                            episode.record,
                            **(
                                {"archived": True}
                                if fault == "archived_db1"
                                else {"context_id": "different-context"}
                            ),
                        ),
                    )
                )
                patch.setattr(
                    state.memory_system, "get_committed_episodic", lambda _: changed
                )
            elif fault in {
                "wrong_channel",
                "wrong_session",
                "suspended_context",
                "participants",
            }:
                changes = {
                    "wrong_channel": {"source_channel": "different-channel"},
                    "wrong_session": {"source_session_id": "different-session"},
                    "suspended_context": {"status": ContextStatus.SUSPENDED},
                    "participants": {"participant_refs": ("ambiguous-token",)},
                }[fault]
                patch.setattr(
                    registry, "get", lambda key: replace(get_context(key), **changes)
                )
            else:

                def damaged_inspection():
                    actual = inspect()
                    rows = actual.records
                    if fault == "missing_terminal":
                        rows = tuple(
                            row
                            for row in rows
                            if not (
                                row.event_id == source.event_id
                                and row.lifecycle
                                is EventLifecycle.TRANSACTION_COMPLETED
                            )
                        )
                    elif fault in {"wrong_event_type", "wrong_event_source"}:
                        changes = (
                            {"event_type": AgentEventType.DEBUG_CHAT}
                            if fault == "wrong_event_type"
                            else {"source": AgentEventSource.API_CHAT_DEBUG}
                        )
                        rows = tuple(
                            row.model_copy(update=changes)
                            if row.event_id == source.event_id
                            else row
                            for row in rows
                        )
                    elif fault == "failed_source":
                        terminal = next(
                            row
                            for row in rows
                            if (
                                row.event_id == source.event_id
                                and row.lifecycle is EventLifecycle.COMPLETED
                            )
                        )
                        rows = (
                            *rows,
                            terminal.model_copy(
                                update={"lifecycle": EventLifecycle.FAILED}
                            ),
                        )
                    return replace(
                        actual,
                        records=rows,
                        external_reconciliation_required=(fault == "recovery"),
                    )

                patch.setattr(state.event_journal, "inspect", damaged_inspection)
            with pytest.raises(AgentRuntimeExecutionError):
                state._motivation_source_owner._submit(source).result(timeout=30)
        assert entry.record.lifecycle is ExperienceLifecycle.ACTIVE
        assert state.main_loop.export_motivation_state() == before
        _no_admission(state, goals, commitments)
        assert state._motivation_source_owner._pending is None


def _revise(state, source, event, *, supersede=False):
    entry = _entry(state, source)
    old = entry.record
    operation, reason = (
        (ExperienceRevisionOperation.SUPERSEDE, ExperienceRevisionReason.SUPERSESSION)
        if supersede
        else (
            ExperienceRevisionOperation.REASSESS,
            ExperienceRevisionReason.REASSESSMENT,
        )
    )
    revision = ExperienceRevisionRecord(
        old.experience_id,
        old.revision + 1,
        operation,
        reason,
        event.requested_at,
        previous_revision_digest=old.revision_history[-1].record_digest,
        event_id=event.event_id,
        event_sequence=event.processing_sequence,
        evidence_refs=(old.experience_id,),
    )
    updated = replace(
        old,
        revision=old.revision + 1,
        lifecycle=ExperienceLifecycle.SUPERSEDED
        if supersede
        else ExperienceLifecycle.ACTIVE,
        superseded_by_id="replacement-experience" if supersede else None,
        revision_history=(*old.revision_history, revision),
    )
    state.experience_store.publish_revision(
        updated,
        "b" * 64,
        entry.source_episode_operation_digest,
        expected_revision=old.revision,
        expected_digest=experience_record_digest(old),
    )
    return updated


@pytest.mark.parametrize("fault", ["capture", "revision_race", "superseded_race"])
def test_pre_wal_fault_restores_exact_prior_state(isolated_app, monkeypatch, fault):
    with TestClient(isolated_app) as client:
        state = client.app.state
        source = _chat(state)
        before = state.main_loop.export_motivation_state()
        durable_before = state.agent_state_store.load()
        owner = state._motivation_source_owner
        validate = owner._validate_commit
        wal_before = state.state_wal.inspect().active_manifest
        goals = state.main_loop.export_goal_state()
        commitments = state.main_loop.export_commitment_state()
        with monkeypatch.context() as patch:
            if fault == "capture":

                def fail_capture(*args, **kwargs):
                    raise ValueError("synthetic pre-WAL capture failure")

                patch.setattr(state.agent_state_store, "capture", fail_capture)
            else:

                def change_before_validation(event):
                    assert owner._pending is not None
                    assert (
                        len(state.main_loop.export_motivation_state().evidence_ledger)
                        == 1
                    )
                    changed = _revise(
                        state, source, event, supersede=fault == "superseded_race"
                    )
                    assert _entry(state, source).record == changed
                    validate(event)

                patch.setattr(owner, "_validate_commit", change_before_validation)
            with pytest.raises(AgentRuntimeDurabilityError) as failure:
                owner._submit(source).result(timeout=30)
        assert failure.value.phase is AgentRuntimeDurabilityPhase.INTERNAL_COMMIT
        if fault != "capture":
            assert failure.value.failure_type == "_MotivationSourceUnavailable"
        assert state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert state.main_loop.export_motivation_state() == before
        assert state.agent_state_store.load() == durable_before
        assert state.state_wal.inspect().active_manifest == wal_before
        assert owner._pending is None and owner._inflight is None
        _no_admission(state, goals, commitments)


def test_wal_commit_without_terminal_recovers_ledger_not_ephemeral_owner(
    isolated_app, monkeypatch
):
    settings = isolated_app.state.settings
    with TestClient(isolated_app) as client:
        state = client.app.state
        source = _chat(state)
        complete = state.state_recovery.complete_committed_event

        def fail_terminal(event, evidence):
            if event.event_type is AgentEventType.MOTIVATION_SOURCE_REVIEW:
                raise ValueError("synthetic loss after WAL publication")
            return complete(event, evidence)

        with monkeypatch.context() as patch:
            patch.setattr(
                state.state_recovery, "complete_committed_event", fail_terminal
            )
            with pytest.raises(AgentRuntimeDurabilityError) as failure:
                state._motivation_source_owner._submit(source).result(timeout=30)
        assert failure.value.phase is AgentRuntimeDurabilityPhase.TERMINAL_COMPLETION
        exact = state.main_loop.export_motivation_state()
        assert len(exact.evidence_ledger) == 1
        assert state.agent_state_store.load().r13_state.motivation.restore() == exact
        assert state._motivation_source_owner._pending is None
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.main_loop.export_motivation_state() == exact
        assert state._motivation_source_owner._pending is None
        assert state.agent_runtime.status is AgentRuntimeStatus.ACCEPTING
        with pytest.raises(AgentRuntimeExecutionError):
            state._motivation_source_owner._submit(source).result(timeout=30)
        assert state.main_loop.export_motivation_state() == exact


def test_no_http_route_schema_field_or_public_u1_export(isolated_app):
    import suzka.r16 as r16

    assert not hasattr(r16, "_MotivationSourceOwner")
    assert not any("motivation" in route.path for route in isolated_app.routes)
    assert not any(
        "source_owner" in name for name in AgentStateSnapshotV10.model_fields
    )


def test_mismatched_store_and_duplicate_owner_refused(isolated_app, tmp_path):
    with TestClient(isolated_app) as client:
        state = client.app.state
        for experience in (
            ExperienceStore(tmp_path / "unrelated-experience"),
            state.experience_store,
        ):
            with pytest.raises((_MotivationSourceUnavailable, RuntimeError)):
                _MotivationSourceOwner(
                    state.main_loop,
                    state.agent_runtime,
                    state.event_journal,
                    experience,
                    state.memory_system,
                    state.main_loop.context_registry,
                )


def test_same_token_in_different_contexts_never_authenticates_actor(isolated_app):
    with TestClient(isolated_app) as client:
        state = client.app.state
        first = _chat(
            state,
            ChatContextSelectors(
                client_session_id="session-a",
                interlocutor_key="same-token",
            ),
        )
        second = _chat(
            state,
            ChatContextSelectors(
                client_session_id="session-b",
                interlocutor_key="same-token",
            ),
        )
        assert (
            _entry(state, first).record.context_id
            != _entry(state, second).record.context_id
        )
        for source in (first, second):
            with pytest.raises(AgentRuntimeExecutionError):
                state._motivation_source_owner._submit(source).result(timeout=30)
        assert not state.main_loop.export_motivation_state().records


def test_internal_r16_event_without_qualified_owner_fails_before_wal(isolated_app):
    """Legacy-compatible no-owner mode must not authorize an R16 mutation."""
    with TestClient(isolated_app) as client:
        state = client.app.state
        assert isinstance(state._motivation_source_owner, _MotivationSourceOwner)
        before_motivation = state.main_loop.export_motivation_state()
        before_state = state.agent_state_store.load()
        before_manifest = state.state_wal.inspect().active_manifest
        goals = state.main_loop.export_goal_state()
        commitments = state.main_loop.export_commitment_state()
        state._motivation_source_owner = None

        with pytest.raises(AgentRuntimeDurabilityError) as error:
            state.agent_runtime.submit(
                AgentEventType.MOTIVATION_SOURCE_REVIEW,
                AgentEventSource.RUNTIME_MOTIVATION_SOURCE_REVIEW,
                lambda: "unqualified handler never authorizes Source",
            ).result(timeout=30)

        assert error.value.phase is AgentRuntimeDurabilityPhase.INTERNAL_COMMIT
        assert error.value.failure_type == "StateRecoveryError"
        assert state.agent_runtime.status is AgentRuntimeStatus.FAILED
        assert state.main_loop.export_motivation_state() == before_motivation
        assert state.agent_state_store.load() == before_state
        assert state.state_wal.inspect().active_manifest == before_manifest
        _no_admission(state, goals, commitments)
