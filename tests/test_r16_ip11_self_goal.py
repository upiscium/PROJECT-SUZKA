"""IP-11: the subject's own Goal proposal (#379 D0-D3, D4/D5).

Real server, real runtime, real Journal; no model calls.
"""

from __future__ import annotations


from fastapi.testclient import TestClient
import pytest

from suzka.motivation.common import R13Reference, R13ReferenceKind
from suzka.motivation.goal import GoalLifecycle
from suzka.r16.activation import (
    R16SelfGoalAlreadyActive,
    self_goal_activation,
    self_goal_activation_handler,
)
from suzka.r16.commitment_owner import _CommitmentSourceSelection
from suzka.r16.goal_owner import GoalOrigin, _SelfGoalSelection
from suzka.r16.outcomes import R16OwnerReason, R16OwnerStatus
from suzka.r16.source_owner import INTEREST_APPLIED
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType

from test_r16_u2a_source_owner import _app, settings_in
from test_r16_u2b_goal_owner import chat, x_id


def _settings(root, *, small_journal=False, **self_goal):
    settings = settings_in(root)
    if small_journal:
        settings = settings.model_copy(
            update={
                "event_journal": settings.event_journal.model_copy(
                    update={"max_bytes": 32 * 1024}
                )
            }
        )
    return settings.model_copy(
        update={
            "emotion": settings.emotion.model_copy(
                update={"timer_enabled": False, "appraisal_response_rate": 1.0}
            ),
            "self_goal": settings.self_goal.model_copy(
                update={"min_strength": 0.01, **self_goal}
            ),
        }
    )


@pytest.fixture
def app(tmp_path):
    return _app(_settings(tmp_path / "self"))


def _activate(state) -> int:
    return state.agent_runtime.submit(
        AgentEventType.R16_SELF_GOAL_ACTIVATION,
        AgentEventSource.API_R16_SELF_GOAL_ACTIVATE,
        self_goal_activation_handler(state.event_journal, state.agent_runtime),
    ).result(timeout=30).value


def _interest(state, text="a subjective experience to review; fixture"):
    x = chat(state, text)
    result = state._motivation_source_owner._submit(x).result(timeout=30)
    assert result.value == INTEREST_APPLIED
    return x


def _propose(state, x):
    return state._goal_source_owner._submit_self_proposal(
        _SelfGoalSelection(x)
    ).result(timeout=30).value


def _review(state, goal_id):
    return state._goal_source_owner._submit_review(goal_id).result(timeout=30).value


def test_self_goal_is_proposed_and_adopted_without_any_user_request(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        _activate(state)
        x = _interest(state)
        proposed = _propose(state, x)
        assert proposed.status is R16OwnerStatus.PROPOSED
        goal = state.main_loop._goal_system.get(proposed.subject_id)
        assert goal is not None and goal.lifecycle is GoalLifecycle.PROPOSED
        assert [ref.kind for ref in goal.origin_refs] == [R13ReferenceKind.MOTIVATION]
        kinds = {ref.kind for ref in goal.evidence_refs}
        assert kinds == {
            R13ReferenceKind.EXPERIENCE,
            R13ReferenceKind.MOTIVATION,
            R13ReferenceKind.EVENT,
        }
        assert R13ReferenceKind.USER_REQUEST not in kinds
        adopted = _review(state, proposed.subject_id)
        assert adopted.status is R16OwnerStatus.ADOPTED
        goal = state.main_loop._goal_system.get(proposed.subject_id)
        assert goal.lifecycle is GoalLifecycle.ADOPTED
        assert state._goal_source_owner._goal_origin(goal) is GoalOrigin.SUBJECT_MOTIVATION


def test_self_proposal_without_activation_holds(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x = _interest(state)
        held = _propose(state, x)
        assert (held.status, held.reason) == (
            R16OwnerStatus.HOLD,
            R16OwnerReason.SELF_GOAL_NOT_ACTIVATED,
        )
        assert state.main_loop.export_goal_state().records == ()


def test_interest_from_before_activation_does_not_count(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        x = _interest(state)
        _activate(state)
        held = _propose(state, x)
        assert held.reason is R16OwnerReason.SELF_GOAL_NOT_ACTIVATED


def test_activation_is_unique(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        sequence = _activate(state)
        with pytest.raises(Exception) as raised:
            _activate(state)
        assert isinstance(raised.value, R16SelfGoalAlreadyActive) or isinstance(
            raised.value.__cause__, R16SelfGoalAlreadyActive
        )
        marker = self_goal_activation(state.event_journal.inspect())
        assert marker is not None and marker.processing_sequence == sequence


def test_weak_interest_holds(tmp_path) -> None:
    with TestClient(_app(_settings(tmp_path / "weak", min_strength=1.0))) as client:
        state = client.app.state
        _activate(state)
        x = _interest(state)
        held = _propose(state, x)
        assert held.reason is R16OwnerReason.MOTIVATION_BELOW_THRESHOLD


def test_stale_interest_holds(app, monkeypatch) -> None:
    """D2 P2-3 / D3 P2-4: the owner compares against the frozen event time."""

    from suzka.r16 import goal_owner as module

    with TestClient(app) as client:
        state = client.app.state
        _activate(state)
        x = _interest(state)
        real = module.timedelta
        # Every INTEREST is "older" than a negative window.
        monkeypatch.setattr(module, "timedelta", lambda **_kw: real(seconds=-1))
        held = _propose(state, x)
        assert held.reason is R16OwnerReason.MOTIVATION_STALE


def test_daily_budget_counts_only_earlier_self_goals(tmp_path) -> None:
    """D5 P2-1: the third succeeds (and survives WAL), the fourth holds."""

    with TestClient(_app(_settings(tmp_path / "budget", per_day=3))) as client:
        state = client.app.state
        _activate(state)
        outcomes = [
            _propose(state, _interest(state, f"experience number {index}; fixture"))
            for index in range(4)
        ]
        assert [o.status for o in outcomes[:3]] == [R16OwnerStatus.PROPOSED] * 3
        assert outcomes[3].reason is R16OwnerReason.SELF_GOAL_BUDGET_EXHAUSTED
        # The third proposal was committed, and can be adopted.
        assert _review(state, outcomes[2].subject_id).status is R16OwnerStatus.ADOPTED


def test_existing_goal_and_user_commitment_come_first(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        _activate(state)
        # A user Goal on X first: the self proposal holds.
        x = _interest(state)
        from suzka.r16.goal_owner import _GoalSourceSelection

        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        user = state._goal_source_owner._submit_proposal(
            _GoalSourceSelection(p, x)
        ).result(timeout=30).value
        assert user.status is R16OwnerStatus.PROPOSED
        assert _propose(state, x).reason is R16OwnerReason.GOAL_ALREADY_EXISTS

        # A user Commitment on another X first (D3 P1-1).
        y = _interest(state, "another experience; fixture")
        c = chat(state, f"COMMIT_REVIEW_EXPERIENCE:1:{x_id(y)}")
        committed = state._commitment_source_owner._submit_proposal(
            _CommitmentSourceSelection(c, y)
        ).result(timeout=30).value
        assert committed.status is R16OwnerStatus.PROPOSED
        assert _propose(state, y).reason is R16OwnerReason.TARGET_HAS_COMMITMENT


def test_self_goal_occupies_x_for_user_requests(app) -> None:
    """D2 P1-3 / D3 addendum: no Commitment record on a self-occupied X."""

    from suzka.r16.goal_owner import _GoalSourceSelection

    with TestClient(app) as client:
        state = client.app.state
        _activate(state)
        x = _interest(state)
        assert _propose(state, x).status is R16OwnerStatus.PROPOSED
        c = chat(state, f"COMMIT_REVIEW_EXPERIENCE:1:{x_id(x)}")
        held = state._commitment_source_owner._submit_proposal(
            _CommitmentSourceSelection(c, x)
        ).result(timeout=30).value
        assert (held.status, held.reason) == (
            R16OwnerStatus.HOLD,
            R16OwnerReason.SELF_GOAL_OCCUPIES_TARGET,
        )
        assert state.main_loop.export_commitment_state().records == ()
        p = chat(state, f"REVIEW_EXPERIENCE:{x_id(x)}")
        again = state._goal_source_owner._submit_proposal(
            _GoalSourceSelection(p, x)
        ).result(timeout=30).value
        assert again.reason is R16OwnerReason.SELF_GOAL_OCCUPIES_TARGET


def test_self_proposal_never_admits_a_user_or_llm_source(app) -> None:
    """The selector names X only; a user CHAT with no INTEREST never grounds it."""

    with TestClient(app) as client:
        state = client.app.state
        _activate(state)
        x = chat(state, "a chat with no Motivation evaluation; fixture")
        held = _propose(state, x)
        assert held.status is R16OwnerStatus.HOLD
        assert held.reason is R16OwnerReason.MOTIVATION_UNAVAILABLE
        with pytest.raises(Exception):
            state._goal_source_owner._submit_self_proposal(object())  # type: ignore[arg-type]


def test_self_goal_plan_never_takes_plan_b(tmp_path) -> None:
    """D5 P1-1: an adopted self Goal yields Plan A (row 3), never B."""

    settings = _settings(tmp_path / "plan")
    settings = settings.model_copy(
        update={
            "subject_history": settings.subject_history.model_copy(
                update={"enabled": True, "path": tmp_path / "h" / "subject_history.sqlite3"}
            )
        }
    )
    from suzka.r16.decision_issuer import DecisionIssueStatus
    from suzka.r16.successor_policy import (
        CommitmentState,
        SuccessorPolicyInputV2,
        SuccessorRow,
    )

    with TestClient(_app(settings)) as client:
        state = client.app.state
        _activate(state)
        x = _interest(state)
        goal_id = _propose(state, x).subject_id
        assert _review(state, goal_id).status is R16OwnerStatus.ADOPTED
        issued = state._plan_preview_owner._submit_preview(goal_id).result(timeout=30).value
        assert issued.status is DecisionIssueStatus.ISSUED
        assert issued.row is SuccessorRow.ROW_3
        assert issued.decision.decision.selected.candidate_id == "candidate-a"
        sealed = SuccessorPolicyInputV2.from_bytes(issued.decision.policy_input_bytes)
        assert sealed.commitment is CommitmentState.ABSENT


def test_self_origin_with_a_commitment_is_ineligible_row_1(tmp_path) -> None:
    """D5 P2-5: INELIGIBLE observed directly (state built past the owners)."""

    from types import SimpleNamespace

    from suzka.r16 import decision_issuer as module
    from suzka.r16.goal_owner import GoalOrigin as Origin
    from suzka.r16.successor_policy import CommitmentState

    issuer = object.__new__(module._DecisionIssuer)
    record = SimpleNamespace(deadline=SimpleNamespace(at=None))
    issuer._preview = SimpleNamespace(
        _loop=SimpleNamespace(
            _commitment_system=SimpleNamespace(get=lambda _id: record)
        )
    )
    q = SimpleNamespace(
        commitment_id="commitment-1",
        commitment_revision=1,
        commitment_digest="d" * 64,
        self_origin=True,
        goal_origin=Origin.SUBJECT_MOTIVATION,
    )
    state, ref, deadline = issuer._commitment(q)  # type: ignore[arg-type]
    assert state is CommitmentState.INELIGIBLE and ref is not None and deadline
    q.self_origin = False
    assert issuer._commitment(q)[0] is CommitmentState.ACTIVE  # type: ignore[arg-type]


def test_self_proposal_needs_its_owner_to_commit(app) -> None:
    """P2-4: the commit checkpoint requires the exact Goal owner."""

    with TestClient(app) as client:
        state = client.app.state
        _activate(state)
        x = _interest(state)
        owner = state._goal_source_owner
        client.app.state._goal_source_owner = None
        try:
            with pytest.raises(Exception):
                owner._submit_self_proposal(_SelfGoalSelection(x)).result(timeout=30)
        finally:
            client.app.state._goal_source_owner = owner


def _self_activation_records(client):  # type: ignore[no-untyped-def]
    from test_r16_ip6b_driver import ADMIN
    from test_r16_ip6a_driver_plumbing import _chat as plain_chat

    from suzka.runtime.journal_compaction import JournalCompactionPolicy

    state = client.app.state
    assert client.post("/api/r16/self-goal/activate", headers=ADMIN).status_code == 200
    for number in range(8):
        plain_chat(client, f"fill {number}")
    journal = state.event_journal
    records = journal.records
    state.journal_compaction.policy = JournalCompactionPolicy(0)
    plain_chat(client, "trigger compaction")
    compaction = journal.inspect().latest_compaction
    assert compaction is not None
    full = (*records, *(r for r in journal.records if r not in records))
    boundary = next(
        index
        for index, record in enumerate(full)
        if record.record_hash == compaction.boundary_checkpoint_hash
    )
    return full, boundary, compaction


def test_c11_marker_is_carried_verified_and_kind_checked(tmp_path, monkeypatch) -> None:
    """C11 (D4 §1): the self-Goal marker survives compaction like C9's."""

    from test_r16_ip6b_driver import TOKEN
    from test_r16_ip4c_past_retention import app as _unused  # noqa: F401

    from suzka.runtime.event_journal import (
        EventJournal,
        EventJournalCompaction,
        EventJournalIntegrityError,
        EventLifecycle,
    )

    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", TOKEN)
    settings = _settings(tmp_path / "c11", small_journal=True)
    with TestClient(_app(settings)) as client:
        records, boundary, compaction = _self_activation_records(client)
        marker = compaction.self_goal_activation_record
        assert marker is not None
        assert marker in client.app.state.event_journal.inspect().carried_records
        assert self_goal_activation(client.app.state.event_journal.inspect()) == marker
        EventJournal._verify_declared_compaction(records, boundary, compaction, None)
        missing = compaction.model_copy(update={"self_goal_activation_record": None})
        with pytest.raises(EventJournalIntegrityError):
            EventJournal._verify_declared_compaction(records, boundary, missing, None)
        data = compaction.model_dump(mode="python")
        for bad in (
            marker.model_copy(update={"lifecycle": EventLifecycle.ACCEPTED}),
            marker.model_copy(update={"event_type": AgentEventType.CHAT}),
            marker.model_copy(update={"source": AgentEventSource.API_CHAT}),
        ):
            with pytest.raises(ValueError):
                EventJournalCompaction.model_validate(
                    {**data, "self_goal_activation_record": bad.model_dump(mode="python")}
                )
    # A fresh process still sees the carried marker.
    with TestClient(_app(settings)) as client:
        assert self_goal_activation(client.app.state.event_journal.inspect()) == marker


def test_a_compaction_without_the_marker_keeps_its_bytes(tmp_path, monkeypatch) -> None:
    from test_r16_ip6b_driver import TOKEN
    from test_r16_ip6a_driver_plumbing import _chat as plain_chat

    from suzka.runtime.journal_compaction import JournalCompactionPolicy

    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", TOKEN)
    with TestClient(_app(_settings(tmp_path / "nomarker", small_journal=True))) as client:
        state = client.app.state
        for number in range(8):
            plain_chat(client, f"fill {number}")
        state.journal_compaction.policy = JournalCompactionPolicy(0)
        plain_chat(client, "trigger compaction")
        compaction = state.event_journal.inspect().latest_compaction
        assert compaction is not None
        assert compaction.self_goal_activation_record is None
        assert "self_goal_activation_record" not in compaction.model_dump(mode="json")


def test_target_reference_is_experience(app) -> None:
    with TestClient(app) as client:
        state = client.app.state
        _activate(state)
        x = _interest(state)
        goal = state.main_loop._goal_system.get(_propose(state, x).subject_id)
        assert goal.target == R13Reference(R13ReferenceKind.EXPERIENCE, x_id(x))


# --- driver (fakes from the IP-6b tests) -------------------------------------


def _self_driver(*, self_goal_active=True, limit=3):
    from test_r16_ip6b_driver import _Clock, _FakeOwner, _FakeRuntime

    from suzka.r16.driver import R16Driver

    runtime, source, goal, commitment, preview = (
        _FakeRuntime(),
        _FakeOwner(),
        _FakeOwner(),
        _FakeOwner(),
        _FakeOwner(),
    )
    driver = R16Driver(
        runtime,  # type: ignore[arg-type]
        source,  # type: ignore[arg-type]
        goal,  # type: ignore[arg-type]
        commitment,  # type: ignore[arg-type]
        preview,  # type: ignore[arg-type]
        active=True,
        self_goal_active=self_goal_active,
        self_goal_per_incarnation=limit,
        clock=_Clock(),
    )
    driver.start()
    return driver, source, goal


def _x(n: int) -> str:
    return f"experience-00000000-0000-4000-8000-{n:012x}"


def test_driver_follows_only_an_interest(monkeypatch) -> None:
    from test_r16_ip6b_driver import _resolve, _turn, _until

    driver, source, goal = _self_driver()
    try:
        driver.notify_chat_terminal(_turn(5, _x(1)), "hello")
        _until(lambda: len(source.calls) == 1)
        _resolve(source.futures[0], 6, "other_motivation_applied")
        driver.notify_chat_terminal(_turn(7, _x(2)), "hi")
        _until(lambda: len(source.calls) == 2)
        _resolve(source.futures[1], 8, INTEREST_APPLIED)
        _until(lambda: any(name == "self_proposal" for name, _ in goal.calls))
        assert [name for name, _ in goal.calls] == ["self_proposal"]
        # PROPOSED -> the existing review stage.
        _resolve(goal.futures[0], 9, _owner(R16OwnerStatus.PROPOSED, "goal-x2"))
        _until(lambda: len(goal.calls) == 2)
        assert goal.calls[1] == ("review", "goal-x2")
    finally:
        driver.stop()


def test_driver_needs_the_self_goal_marker_and_caps_per_incarnation() -> None:
    from test_r16_ip6b_driver import _resolve, _turn, _until

    driver, source, goal = _self_driver(self_goal_active=False, limit=1)
    try:
        driver.notify_chat_terminal(_turn(5, _x(1)), "hello")
        _until(lambda: len(source.calls) == 1)
        _resolve(source.futures[0], 6, INTEREST_APPLIED)
        import time

        time.sleep(0.5)
        assert goal.calls == []
        driver.activate_self_goal()
        for index, sequence in ((2, 7), (3, 9)):
            driver.notify_chat_terminal(_turn(sequence, _x(index)), "hi")
            _until(lambda n=index: len(source.calls) == n)
            _resolve(source.futures[index - 1], sequence + 1, INTEREST_APPLIED)
        _until(lambda: len(goal.calls) == 1)
        time.sleep(0.5)
        assert len(goal.calls) == 1  # per_incarnation = 1
    finally:
        driver.stop()


def _owner(status, subject):
    from suzka.r16.outcomes import R16OwnerOutcome

    return R16OwnerOutcome(status, subject)


def test_rescan_selects_eligible_interests_and_proposed_self_goals(app) -> None:
    from datetime import datetime, timezone

    from suzka.r16.driver import rescan_from, self_goal_targets
    from suzka.r16.source_owner import _MotivationSourceRef

    with TestClient(app) as client:
        state = client.app.state
        marker = _activate(state)
        x = _interest(state)
        y = _interest(state, "a second experience; fixture")
        goal_id = _propose(state, y).subject_id

        def ref(experience_id):
            record = state.experience_store.load_committed_current(experience_id).record
            return _MotivationSourceRef(
                record.source_event_id, record.source_event_sequence, record.source_episode_id
            )

        targets = self_goal_targets(
            state.main_loop.export_motivation_state(),
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
            ref,
            marker_sequence=marker,
            min_strength=0.01,
            stale_after_hours=24,
            now=datetime.now(timezone.utc),
        )
        assert [t.experience_id for t in targets] == [x_id(x)]
        assert targets[0].ref == x
        rescan = rescan_from(
            state.main_loop.export_goal_state(),
            state.main_loop.export_commitment_state(),
            targets,
        )
        assert goal_id in rescan.goal_ids


def test_s1_shape_end_to_end_without_any_user_request(tmp_path, monkeypatch) -> None:
    """#379 R20 S1 shape: CHAT X -> U2a INTEREST -> self proposal -> adoption
    -> Plan preview, all chained by the driver; the only user input is X."""

    from test_r16_ip6b_driver import ADMIN, TOKEN, _chat, _until

    monkeypatch.setenv("SUZKA_P1B_TEST_TOKEN", TOKEN)
    settings = _settings(tmp_path / "s1", min_strength=0.15)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert client.post("/api/r16/driver/activate", headers=ADMIN).status_code == 200
        response = client.post("/api/r16/self-goal/activate", headers=ADMIN)
        assert response.status_code == 200
        assert client.post("/api/r16/self-goal/activate", headers=ADMIN).status_code == 409
        x = _chat(client, "something worth looking into later, I think")

        def adopted():
            return [
                goal
                for goal in state.main_loop.export_goal_state().records
                if goal.lifecycle is GoalLifecycle.ADOPTED
                and goal.origin_refs[0].kind is R13ReferenceKind.MOTIVATION
            ]

        goals = _until(adopted)
        assert goals[0].target.reference == x["experience_id"]
        # The preview ran for the adopted self Goal.
        _until(
            lambda: any(
                row.event_type is AgentEventType.PLAN_PREVIEW_REVIEW
                for row in state.event_journal.inspect().records
            )
        )
        assert state.main_loop.export_commitment_state().records == ()
