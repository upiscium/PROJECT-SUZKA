"""IP-5 (#383 / #370 D2 P1-5, #354 scope): closed owner outcomes and reasons.

U2b / U2c / U3-A owners return a value with a status and, for HOLD, one
reason from a closed vocabulary. An unqualified proposal is a HOLD, not a
FAILED event.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.r16.goal_owner import _GoalSourceSelection, _GoalSourceUnavailable
from suzka.r16.outcomes import (
    R16OwnerOutcome,
    R16OwnerReason,
    R16OwnerStatus,
    owner_reason,
)
from suzka.runtime.event_journal import EventLifecycle
from suzka.runtime.journal_source import JournalSourceReason

from test_r16_ip4c_past_retention import _archive, app  # noqa: F401
from test_r16_u2b_goal_owner import chat, preexisting_interest, x_id


def test_outcome_shape_is_closed() -> None:
    assert R16OwnerOutcome.hold(R16OwnerReason.SOURCE_MISSING).subject_id is None
    with pytest.raises(ValueError):
        R16OwnerOutcome(R16OwnerStatus.HOLD)
    with pytest.raises(ValueError):
        R16OwnerOutcome(R16OwnerStatus.ADOPTED, "goal-x", R16OwnerReason.SOURCE_MISSING)
    with pytest.raises(ValueError):
        R16OwnerOutcome(R16OwnerStatus.ADOPTED)
    with pytest.raises(TypeError):
        R16OwnerOutcome("hold", None, R16OwnerReason.SOURCE_MISSING)  # type: ignore[arg-type]


def test_reasons_map_journal_reasons_and_default_to_unavailable() -> None:
    assert owner_reason(
        _GoalSourceUnavailable("x", reason=JournalSourceReason.SOURCE_PAST_RETENTION)
    ) is R16OwnerReason.SOURCE_PAST_RETENTION
    assert owner_reason(
        _GoalSourceUnavailable("x", reason=JournalSourceReason.SOURCE_MISSING)
    ) is R16OwnerReason.SOURCE_MISSING
    assert owner_reason(_GoalSourceUnavailable("x")) is (
        R16OwnerReason.SOURCE_UNAVAILABLE
    )
    assert owner_reason(ValueError("plain")) is R16OwnerReason.SOURCE_UNAVAILABLE


def _proposed(state):  # type: ignore[no-untyped-def]
    x = chat(state, "source experience; fixture")
    preexisting_interest(state, x)
    p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
    owner = state._goal_source_owner
    proposed = owner._submit_proposal(_GoalSourceSelection(p, x)).result(timeout=30)
    assert proposed.value.status is R16OwnerStatus.PROPOSED
    return x, p, proposed.value.subject_id


def test_second_request_for_the_same_target_holds_goal_already_exists(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        x, _p, goal_id = _proposed(state)
        p2 = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        before = state.main_loop.export_goal_state()
        held = state._goal_source_owner._submit_proposal(
            _GoalSourceSelection(p2, x)
        ).result(timeout=30)
        assert held.value == R16OwnerOutcome.hold(R16OwnerReason.GOAL_ALREADY_EXISTS)
        assert state.main_loop.export_goal_state() == before
        # The held proposal event completed; it did not fail.
        rows = [
            row.lifecycle
            for row in state.event_journal.records
            if row.event_id == held.event.event_id
        ]
        assert EventLifecycle.COMPLETED in rows
        assert EventLifecycle.FAILED not in rows


def test_a_subject_goal_on_the_target_holds_self_goal_occupies_target(
    app, monkeypatch  # noqa: F811
) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x, _p, goal_id = _proposed(state)
        goals = state.main_loop._goal_system
        original_get = goals.get

        class _SelfGoal:
            origin_refs = (R13Reference(R13ReferenceKind.EXPERIENCE, x_id(x)),)

        monkeypatch.setattr(
            goals,
            "get",
            lambda key: _SelfGoal() if key == goal_id else original_get(key),
        )
        p2 = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        held = state._goal_source_owner._submit_proposal(
            _GoalSourceSelection(p2, x)
        ).result(timeout=30)
        assert held.value.reason is R16OwnerReason.SELF_GOAL_OCCUPIES_TARGET


def test_reviews_and_previews_carry_their_reasons(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        _x, _p, goal_id = _proposed(state)
        # A Plan preview of a Goal that is not adopted holds GOAL_NOT_ADOPTED.
        preview = state._plan_preview_owner._submit_preview(goal_id).result(
            timeout=30
        )
        assert preview.value.hold is not None
        assert preview.value.owner_reason is R16OwnerReason.GOAL_NOT_ADOPTED
        missing = state._plan_preview_owner._submit_preview("goal-unknown").result(
            timeout=30
        )
        assert missing.value.owner_reason is R16OwnerReason.GOAL_NOT_ADOPTED

        owner = state._goal_source_owner
        adopted = owner._submit_review(goal_id).result(timeout=30)
        assert adopted.value == R16OwnerOutcome(R16OwnerStatus.ADOPTED, goal_id)
        # Reviewing again: it is no longer waiting for review.
        again = owner._submit_review(goal_id).result(timeout=30)
        assert again.value == R16OwnerOutcome.hold(R16OwnerReason.NOT_PENDING_REVIEW)
        positive = state._plan_preview_owner._submit_preview(goal_id).result(
            timeout=30
        )
        assert positive.value.hold is None and positive.value.owner_reason is None


def test_proposal_over_an_archived_target_holds_past_retention(app) -> None:  # noqa: F811
    with TestClient(app) as client:
        state = client.app.state
        x = chat(state, "old experience; fixture")
        preexisting_interest(state, x)
        _archive(state, x)
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        held = state._goal_source_owner._submit_proposal(
            _GoalSourceSelection(p, x)
        ).result(timeout=30)
        assert held.value == R16OwnerOutcome.hold(
            R16OwnerReason.SOURCE_PAST_RETENTION
        )
