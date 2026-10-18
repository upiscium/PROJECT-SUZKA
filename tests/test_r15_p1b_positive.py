"""P1b-A narrow positive U3 owner test with actual isolated R07/R12/DB1.

No live production Source, no direct R15 state setter, no U5/StateWAL claim.
The semantic source is synthetic deterministic-model input; all actual
Chat/Journal/Experience/DB1 operations are genuine isolated app operations.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1.protocol import audit_trial, run_trial
from suzka.memory.experience_participant import experience_id_for_event
from suzka.r15.narrative_self_system import NarrativeSelfSystem
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
    restore_scoped_option,
)
from suzka.runtime.agent_runtime import (
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeExecutionError,
)
from suzka.runtime.event_journal import EventLifecycle

# Reuse only the existing synthetic fixture as test data, never as a
# production R15 source/admission implementation.
from test_r15_p1_owner_bridge import scenario


def _source_ref(client: TestClient, episode_id: str) -> QualifiedTurnRef:
    journal = client.app.state.event_journal.inspect()
    store = client.app.state.experience_store
    matches: list[QualifiedTurnRef] = []
    for record in journal.records:
        if record.lifecycle is not EventLifecycle.COMPLETED:
            continue
        if record.event_type is not AgentEventType.CHAT:
            continue
        entry = store.load_committed_current(
            experience_id_for_event(record.event_id, record.processing_sequence)
        )
        if entry.record.source_episode_id == episode_id:
            matches.append(
                QualifiedTurnRef(
                    record.event_id, record.processing_sequence, episode_id
                )
            )
    assert len(matches) == 1
    return matches[0]


def _admit(
    runtime: AgentRuntime,
    owner: NarrativeSelfSystem,
    references: tuple[QualifiedTurnRef, ...],
):
    future = runtime.submit(
        AgentEventType.NARRATIVE_REVIEW,
        AgentEventSource.RUNTIME_NARRATIVE_REVIEW,
        lambda: owner._review_committed_turns(references),
    )
    return future.result(timeout=30).value


def _completed_fixture(
    tmp_path: Path, *, no_reveal: bool = False, admit_s2: bool = True
):
    app, provider, source, manifest, capture = scenario(tmp_path, no_reveal=no_reveal)
    # Owner is created after actual app starts, when the bound event authority
    # and process-local Context/Journal objects exist.
    with TestClient(app) as client:
        runtime: AgentRuntime = client.app.state.agent_runtime
        gate = QualifiedNarrativeSourceGate(
            runtime,
            client.app.state.event_journal,
            client.app.state.experience_store,
            client.app.state.memory_system,
            client.app.state.main_loop.context_registry,
        )
        owner = NarrativeSelfSystem(source_gate=gate)
        turns: list[QualifiedTurnRef] = []
        first_results = []

        def transport(message: str) -> dict[str, str]:
            # The runner has completed E+S1 and committed its checkpoint;
            # this function is invoked BEFORE the E2 HTTP observation.
            if len(turns) == 3:
                result = _admit(runtime, owner, tuple(turns))
                first_results.append(result)
                assert result.changed and result.snapshot.revision == 1
            response = client.post(
                "/api/chat", json={"message": message, "attachments": []}
            )
            assert response.status_code == 200
            obj = response.json()
            turn = _source_ref(client, obj["episode_id"])
            turns.append(turn)
            return {"episode_id": obj["episode_id"], "response": obj["response"]}

        run_trial(manifest, capture, transport)
        assert len(first_results) == 1
        initial = first_results[0]
        assert audit_trial(manifest, capture)["s1_before_e2_checkpoint"] is True
        snapshot = owner.snapshot()
        assert initial.snapshot.revision == 1
        if no_reveal:
            assert initial.snapshot == snapshot
        final = None
        if not no_reveal and admit_s2:
            final = _admit(runtime, owner, tuple(turns))
        yield (
            client,
            runtime,
            owner,
            tuple(turns),
            initial,
            final,
            manifest,
            capture,
            source,
            provider,
        )


@pytest.mark.parametrize("no_reveal", [False, True])
def test_real_serialized_u3_owner_admits_revision_only_after_committed_e2(
    tmp_path: Path,
    no_reveal: bool,
) -> None:
    for (
        client,
        runtime,
        owner,
        refs,
        initial,
        final,
        manifest,
        capture,
        source,
        provider,
    ) in _completed_fixture(tmp_path, no_reveal=no_reveal):
        assert initial.changed is True
        assert initial.stage == "S1"
        assert initial.selected_referent == "A"
        assert initial.snapshot.revision == 1
        assert len(initial.snapshot.revision_history) == 1
        assert owner.selected_view().revision == (1 if no_reveal else 2)
        assert len(owner.snapshot().episodes) == (1 if no_reveal else 2)
        assert owner.snapshot() == (
            owner.snapshot().__class__.from_json(owner.snapshot().canonical_bytes())
        )
        assert source.joinpath("event_journal.jsonl").is_file()
        if no_reveal:
            assert final is None
            assert owner.snapshot().revision == 1
            assert owner.snapshot().episodes[0] == initial.snapshot.episodes[0]
        else:
            assert final is not None and final.changed is True
            assert final.event.event_sequence > initial.event.event_sequence
            assert final.stage == "S2"
            assert final.selected_referent == "B"
            assert final.snapshot.revision == 2
            assert initial.snapshot.episodes[0] in final.snapshot.episodes
            assert (
                len({ep.interpretation.statement_id for ep in final.snapshot.episodes})
                == 2
            )
            for ep in final.snapshot.episodes:
                assert restore_scoped_option(
                    initial.source_scope, ep.interpretation.statement_id
                ) in ("A", "B")
            assert (
                initial.snapshot.revision_history[0]
                == owner.snapshot().revision_history[0]
            )
            latest = _admit(runtime, owner, refs)
            assert latest.changed is False
            assert latest.snapshot == owner.snapshot()


def test_process_local_u3_refuses_external_pseudo_admission(tmp_path: Path) -> None:
    for client, runtime, owner, refs, initial, final, *_others in _completed_fixture(
        tmp_path
    ):
        before = owner.snapshot().canonical_bytes()
        with pytest.raises(QualifiedProposalUnavailable):
            owner._review_committed_turns(refs)
        assert owner.snapshot().canonical_bytes() == before
        assert not any(
            hasattr(owner, name)
            for name in (
                "set_trust",
                "adopt_claim",
                "set_competence",
                "install_root",
                "restore_snapshot",
            )
        )


def test_missing_source_and_forged_refs_refuse_without_mutation(tmp_path: Path) -> None:
    for client, runtime, owner, refs, initial, final, *_others in _completed_fixture(
        tmp_path
    ):
        baseline = owner.snapshot().canonical_bytes()
        forged = (
            QualifiedTurnRef(
                "forged-event", refs[0].event_sequence, refs[0].episode_id
            ),
            *refs[1:],
        )
        with pytest.raises(AgentRuntimeExecutionError):
            _admit(runtime, owner, forged)
        assert owner.snapshot().canonical_bytes() == baseline
        # Any repeated but wrong second source ref is rejected.
        wrong = (*refs[:3], refs[3], refs[3])
        with pytest.raises(AgentRuntimeExecutionError):
            _admit(runtime, owner, wrong)
        assert owner.snapshot().canonical_bytes() == baseline


def test_identity_boundaries_preserve_unknown_identity(tmp_path: Path) -> None:
    for client, runtime, owner, refs, initial, final, *_others in _completed_fixture(
        tmp_path
    ):
        context = client.app.state.main_loop.context_registry
        existing = owner.snapshot().canonical_bytes()
        ctx_id = context.current_context_id
        assert ctx_id is not None
        # A newly assigned participant is not evidence they existed at E.
        context.add_participant_ref(ctx_id, "synthetic-new-participant")
        with pytest.raises(AgentRuntimeExecutionError):
            _admit(runtime, owner, refs)
        assert owner.snapshot().canonical_bytes() == existing


def test_source_gate_needs_a_live_runtime_worker(tmp_path: Path) -> None:
    for client, runtime, owner, refs, *_others in _completed_fixture(tmp_path):
        gate = owner._source_gate
        assert gate is not None
        with pytest.raises(QualifiedProposalUnavailable):
            gate.qualify(refs, stage="S2")


def test_repeated_s1_without_e2_does_not_create_second_revision(
    tmp_path: Path,
) -> None:
    for _, runtime, owner, refs, initial, final, *_ in _completed_fixture(
        tmp_path, no_reveal=True
    ):
        assert final is None and owner.snapshot().revision == 1
        reread = _admit(runtime, owner, refs[:3])
        assert not reread.changed
        assert reread.snapshot.revision == 1
        assert reread.snapshot.episodes == initial.snapshot.episodes
        # A pretend E2 using one duplicate real S2 source cannot pass.
        fake_later = (*refs[:3], refs[3], refs[3])
        with pytest.raises(AgentRuntimeExecutionError):
            _admit(runtime, owner, fake_later)
        assert owner.snapshot().revision == 1


def test_missing_committed_e2_fails_before_owner_revision(
    tmp_path: Path,
) -> None:
    from suzka.memory.experience_participant import experience_id_for_event

    for client, runtime, owner, refs, initial, final, *_ in _completed_fixture(
        tmp_path, admit_s2=False
    ):
        assert final is None
        assert owner.snapshot().revision == 1
        e2 = refs[3]
        experience = client.app.state.experience_store
        experience_id = experience_id_for_event(e2.event_id, e2.event_sequence)
        path = experience.record_path(experience_id, 0)
        original = path.read_bytes()
        # Only the disposable pytest root is mutated for this fault injection.
        path.unlink()
        try:
            with pytest.raises(AgentRuntimeExecutionError):
                _admit(runtime, owner, refs)
            assert owner.snapshot().revision == 1
            assert owner.snapshot().episodes == initial.snapshot.episodes
        finally:
            path.write_bytes(original)
            path.chmod(0o600)
        recovered = _admit(runtime, owner, refs)
        assert recovered.changed and recovered.snapshot.revision == 2


def test_narrative_update_requires_exact_runtime_type_and_source(
    tmp_path: Path,
) -> None:
    for client, runtime, owner, refs, *_ in _completed_fixture(tmp_path):
        previous = owner.snapshot().canonical_bytes()
        future = runtime.submit(
            AgentEventType.NARRATIVE_REVIEW,
            AgentEventSource.API_CHAT,  # NOT the internally qualified source
            lambda: owner._review_committed_turns(refs),
        )
        with pytest.raises(AgentRuntimeExecutionError):
            future.result(timeout=30)
        assert owner.snapshot().canonical_bytes() == previous


@pytest.mark.parametrize(
    "invalid",
    [
        '{"schema":1,"schema":1}',
        '{"schema":1,"phase":"S1","assumed_referent":"B","assessment":"UNKNOWN","uncertainty":"LOW","evidence_steps":["E"],"rationale":"x","set_trust":1}',
        '{"schema":1,"phase":"S1","assumed_referent":"B","assessment":"UNKNOWN","uncertainty":"LOW","evidence_steps":["E2"],"rationale":"x"}',
        '{"schema":1,"phase":"S1","assumed_referent":"B","assessment":"UNKNOWN","uncertainty":"LOW","evidence_steps":["E"],"rationale":"x","verified_competence":true}',
    ],
)
def test_direct_claim_shapes_are_not_authoritative(invalid: str) -> None:
    from suzka.r15.qualified_proposal import _parse_candidate

    with pytest.raises(QualifiedProposalUnavailable):
        _parse_candidate(invalid, "S1")


def test_no_source_gate_is_not_an_alternate_admission_surface(
    tmp_path: Path,
) -> None:
    for _, runtime, owner, refs, *_ in _completed_fixture(tmp_path):
        empty = NarrativeSelfSystem()
        original = empty.snapshot().canonical_bytes()
        future = runtime.submit(
            AgentEventType.NARRATIVE_REVIEW,
            AgentEventSource.RUNTIME_NARRATIVE_REVIEW,
            lambda: empty._review_committed_turns(refs),
        )
        with pytest.raises(AgentRuntimeExecutionError):
            future.result(timeout=30)
        assert empty.snapshot().canonical_bytes() == original


def test_wrong_original_source_scope_cannot_decode_provisional_meaning(
    tmp_path: Path,
) -> None:
    from suzka.r15.qualified_proposal import scoped_option_key

    for _, _, owner, _, initial, final, *_ in _completed_fixture(tmp_path):
        source_scope = initial.source_scope
        assert (
            restore_scoped_option(source_scope, scoped_option_key(source_scope, "A"))
            == "A"
        )
        with pytest.raises(QualifiedProposalUnavailable):
            restore_scoped_option(
                source_scope + "-other",
                owner.snapshot().episodes[0].interpretation.statement_id,
            )


def test_a_new_owner_cannot_pretend_to_restore_unpersisted_u3(
    tmp_path: Path,
) -> None:
    for _, _, owner, _, initial, final, *_ in _completed_fixture(tmp_path):
        assert owner.snapshot().revision == 2
        freshly_instantiated = NarrativeSelfSystem()
        assert freshly_instantiated.snapshot().revision == 0
        # U1 serialization works, but this is NOT U5 AgentState/WAL replay.
        restored_value = owner.snapshot().__class__.from_json(
            owner.snapshot().canonical_bytes()
        )
        assert restored_value == owner.snapshot()
        assert freshly_instantiated.snapshot() != restored_value
