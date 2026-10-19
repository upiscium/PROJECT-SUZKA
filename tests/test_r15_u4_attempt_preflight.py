"""TEST ONLY source-backed U4 attempted-task representation; NOT U4 admission."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.r15_p1.u4_attempt_preflight import (
    TASK_CLASS,
    SelfAttemptUnavailable,
    inspect_bounded_referent_attempt,
)
from suzka.memory.experience_participant import experience_id_for_event
from suzka.r15.common import InterpretationStatus
from suzka.r15.contracts import (
    NarrativeSelfState,
    SelfModelState,
    VerifiedCompetence,
)
from suzka.r15.qualified_proposal import QualifiedTurnRef
from suzka.r15.self_model_system import SelfModelSystem

from test_r15_p1bb_failures import source_ready_for_later_reinterpretation
from test_r15_p1b_positive import _admit


def test_actual_completed_experience_can_encode_attempt_without_verified_skill(
    tmp_path: Path,
) -> None:
    with source_ready_for_later_reinterpretation(tmp_path) as (
        _client,
        runtime,
        narrative_owner,
        refs,
        _settings,
    ):
        accepted = _admit(runtime, narrative_owner, refs)
        assert accepted.changed and accepted.snapshot.revision == 2
        source_gate = narrative_owner._source_gate
        assert source_gate is not None
        detached = inspect_bounded_referent_attempt(
            source_gate, narrative_owner.snapshot(), refs
        )
        assert detached.event_count == 5
        assert detached.original_interpretation == "A"
        assert detached.current_interpretation == "B"
        assert detached.corrected_by_user_statement
        assert not detached.admission_to_u4
        assert detached.root.revision == 0
        assert len(detached.root.capabilities) == 1
        capability = detached.root.capabilities[0]
        assert capability.task_class_key == TASK_CLASS
        assert capability.attempted.status is InterpretationStatus.PRESENT
        assert capability.attempted.support[0].event is not None
        assert capability.attempted.support[0].event.event_id == refs[1].event_id
        assert capability.verified_competence is VerifiedCompetence.UNKNOWN
        assert (
            SelfModelState.from_json(detached.root.canonical_bytes()) == detached.root
        )
        assert SelfModelSystem().snapshot() == SelfModelState(1, 0, (), (), None, ())
        with pytest.raises(ValueError):
            replace(capability, verified_competence="verified")
        encoded = detached.root.canonical_bytes()
        assert b"model_rationale" not in encoded
        assert b"verified_competence" in encoded


def test_missing_or_forged_events_fail_closed_without_real_u4_state_change(
    tmp_path: Path,
) -> None:
    with source_ready_for_later_reinterpretation(tmp_path) as (
        _client,
        runtime,
        narrative_owner,
        refs,
        _settings,
    ):
        source_gate = narrative_owner._source_gate
        assert source_gate is not None
        initial = narrative_owner.snapshot()
        with pytest.raises(SelfAttemptUnavailable):
            inspect_bounded_referent_attempt(source_gate, initial, refs)
        _admit(runtime, narrative_owner, refs)
        baseline = narrative_owner.snapshot()
        for bogus in (
            refs[:3],  # the later user statement does not exist in this slice
            refs[:-1],
            (refs[0], refs[1], refs[1], refs[3], refs[4]),
            (
                QualifiedTurnRef(
                    "forged-E", refs[1].event_sequence, refs[1].episode_id
                ),
                *refs[1:],
            ),
            (refs[0], refs[2], refs[1], refs[3], refs[4]),
        ):
            with pytest.raises(SelfAttemptUnavailable):
                inspect_bounded_referent_attempt(source_gate, baseline, bogus)
        assert narrative_owner.snapshot() == baseline
        assert SelfModelSystem().snapshot().revision == 0


def test_retracted_later_source_refuses_attempt_update(
    tmp_path: Path,
) -> None:
    with source_ready_for_later_reinterpretation(tmp_path) as (
        client,
        runtime,
        narrative_owner,
        refs,
        _settings,
    ):
        _admit(runtime, narrative_owner, refs)
        source_gate = narrative_owner._source_gate
        assert source_gate is not None
        root = narrative_owner.snapshot()
        entry_id = experience_id_for_event(refs[3].event_id, refs[3].event_sequence)
        path = client.app.state.experience_store.record_path(entry_id, 0)
        stored = path.read_bytes()
        path.unlink()
        try:
            with pytest.raises(SelfAttemptUnavailable):
                inspect_bounded_referent_attempt(source_gate, root, refs)
            assert narrative_owner.snapshot() == root
        finally:
            path.write_bytes(stored)
            path.chmod(0o600)
        valid = inspect_bounded_referent_attempt(source_gate, root, refs)
        assert (
            valid.root.capabilities[0].verified_competence is VerifiedCompetence.UNKNOWN
        )


def test_unrelated_empty_or_caller_supplied_narrative_has_no_u4_authority(
    tmp_path: Path,
) -> None:
    with source_ready_for_later_reinterpretation(tmp_path) as (
        client,
        runtime,
        narrative_owner,
        refs,
        _settings,
    ):
        source_gate = narrative_owner._source_gate
        assert source_gate is not None
        _admit(runtime, narrative_owner, refs)
        with pytest.raises(SelfAttemptUnavailable):
            inspect_bounded_referent_attempt(
                source_gate,
                NarrativeSelfState(1, 0, (), (), None, ()),
                refs,
            )
        with pytest.raises(SelfAttemptUnavailable):
            inspect_bounded_referent_attempt(
                source_gate,
                narrative_owner.snapshot(),
                tuple(refs[:3]),
            )
        # The factory produces only detached data. There is intentionally
        # no R15 U4 subject adoption, WAL update or model public API here.
        assert client.app.state.r15_state_port.self_model.snapshot().revision == 0
        assert (
            client.app.state.agent_state_store.load().self_model_state["revision"] == 0
        )
