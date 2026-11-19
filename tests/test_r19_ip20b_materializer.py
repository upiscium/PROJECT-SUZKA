"""IP-20b: the R19 Wake materializer (IP-20 D1 r3 §3, §9-2; HD-20-1 (a), HD-20-5 (b)).

Every Wake below is written by the real ``WakeMaterializerV1`` through the real
R07 runtime and the subject.history record participant, read back through
``adopted_records()``.  The cause chains (Goal, Commitment, Decision, Intent,
forecast, fixture claim / Outcome, Assessment, Feedback) are the ones the IP-16b
and IP-19 tests build; the verdict / challenge / claim / Outcome rows are
fixtures (the real executor is IP-18's), and each stage is followed by one scan
the way the IP-20d Scheduler's nudge would run it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi.testclient import TestClient
import pytest

from suzka.r16.decision_issuer import _IssuerHold
from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import HistoryOperationKind as K, RecordKind as R
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError
from suzka.r16.outcomes import R16OwnerReason
from suzka.r17.contracts import OutcomeState
from suzka.r18.contracts import VerifiedState
from suzka.r19.contracts import (
    COMMITMENT_REF_KIND,
    TerminalReason,
    WakeCause,
    WakeTerminalKind,
    WakeTerminalV1,
    WakeV1,
    may_supersede,
)
from suzka.r19.materializer import (
    MaterializeReason,
    MaterializeStatus,
    SkipReason,
    WakeMaterializerV1,
)
from suzka.runtime.agent_runtime import AgentEventType

import test_r16_ip10_decision_issuer as ip10
import test_r16_ip16b_ledger_reader as ip16b
import test_r17_ip19_feedback_owner as ip19
import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_commitment, _qualified_goal

MATERIALIZE = AgentEventType.R19_WAKE_MATERIALIZE


def _settings(root, *, r19: bool = True, **r19_fields: Any):  # type: ignore[no-untyped-def]
    base = ip16a2._settings(root)
    return base.model_copy(
        update={"r19": base.r19.model_copy(update={"enabled": r19, **r19_fields})}
    )


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip20b"))


# ------------------------------------------------------------------ helpers


def _scan(state):  # type: ignore[no-untyped-def]
    return state.wake_materializer.submit_scan().result(timeout=30).value


def _wakes(state) -> list[WakeV1]:  # type: ignore[no-untyped-def]
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return [item.record for item in found.records if type(item.record) is WakeV1]


def _wake_rows(state):  # type: ignore[no-untyped-def]
    return [
        row
        for row in state.subject_history_store.operations(ip10._incarnation(state))
        if row.operation_kind is K.R19_WAKE_MATERIALIZE
    ]


def _skips(result) -> list[tuple[WakeCause, SkipReason]]:  # type: ignore[no-untyped-def]
    return list(result.skipped)


def _events(state, event_type) -> int:  # type: ignore[no-untyped-def]
    return ip19._events(state, event_type)


# ------------------------------------------------------------ the four causes


def test_goal_admitted_makes_one_wake_and_a_second_scan_nothing(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        assert isinstance(state.wake_materializer, WakeMaterializerV1)
        _x, gid = _qualified_goal(state)
        result = _scan(state)
        assert result.status is MaterializeStatus.COMMITTED
        wake = result.wake
        assert (wake.cause, wake.goal_id, wake.cause_ref.kind) == (
            WakeCause.GOAL_ADMITTED,
            gid,
            "goal-admission",
        )
        assert wake.origin_refs == ()  # a user Goal
        assert wake.frozen_predecessor.absence_horizon_sequence == wake.horizon_sequence
        assert wake.horizon_sequence < wake.cause_sequence + 10_000
        assert _wakes(state) == [wake]
        # The Wake row is finalized and its event is terminal.
        assert [row.phase for row in _wake_rows(state)] == [HistoryPhase.FINALIZED]
        # The window: cause + 3600 s, expiring in 7 days.
        assert wake.not_before_utc < wake.expires_at_utc
        # Nothing more: the cause event already has its Wake.
        again = _scan(state)
        assert again.status is MaterializeStatus.NONE_LEFT
        assert _skips(again) == [(WakeCause.GOAL_ADMITTED, SkipReason.CAUSE_ALREADY_WOKEN)]
        assert len(_wakes(state)) == 1
        ip16a2._no_failed_events(state)


def test_commitment_after_goal_makes_a_second_wake_that_supersedes_the_goal_wake(app) -> None:  # type: ignore[no-untyped-def]
    """Goal-then-Commitment (HD-20-5 (b), A-11): the Commitment Wake carries
    its Commitment ref and deadline, and ``may_supersede`` lets it cancel the
    older GOAL_ADMITTED Wake."""

    with TestClient(app) as client:
        state = client.app.state
        x, _gid = _qualified_goal(state)
        goal_wake = _scan(state).wake
        assert goal_wake.cause is WakeCause.GOAL_ADMITTED
        commitment_id = _qualified_commitment(state, x)
        result = _scan(state)
        assert result.status is MaterializeStatus.COMMITTED
        wake = result.wake
        assert wake.cause is WakeCause.COMMITMENT_ADMITTED
        assert [ref.kind for ref in wake.origin_refs] == [COMMITMENT_REF_KIND]
        assert wake.origin_refs[0].ref_id == commitment_id
        assert wake.cause_ref.ref_id == commitment_id
        assert wake.cause_ref.kind == "commitment-admission"
        # The fingerprint moved with the Commitment: a new Wake, a new cycle.
        assert wake.wakeup_id != goal_wake.wakeup_id
        assert wake.logical_cycle_id != goal_wake.logical_cycle_id
        assert wake.cause_sequence > goal_wake.cause_sequence
        # The Goal Wake froze the empty predecessor; the Commitment Wake reads
        # the head again (the same absence here: no Decision was issued).
        assert may_supersede(goal_wake, wake)
        assert not may_supersede(wake, goal_wake)
        assert {item.wakeup_id for item in _wakes(state)} == {goal_wake.wakeup_id, wake.wakeup_id}
        ip16a2._no_failed_events(state)


def test_goal_and_commitment_before_the_first_scan_make_one_wake(app) -> None:  # type: ignore[no-untyped-def]
    """The same evidence is one Wake (``wakeup_id`` hashes the fingerprint, the
    cause is provenance, #369 A1-3): the later cause is skipped SAME_FINGERPRINT
    and writes nothing."""

    with TestClient(app) as client:
        state = client.app.state
        x, _gid = _qualified_goal(state)
        _qualified_commitment(state, x)
        first = _scan(state)
        assert first.status is MaterializeStatus.COMMITTED
        assert first.wake.cause is WakeCause.GOAL_ADMITTED
        second = _scan(state)
        assert second.status is MaterializeStatus.NONE_LEFT
        assert (WakeCause.COMMITMENT_ADMITTED, SkipReason.SAME_FINGERPRINT) in _skips(second)
        assert len(_wakes(state)) == 1


def test_a_later_cause_wake_skips_an_older_cause_candidate(app) -> None:  # type: ignore[no-untyped-def]
    """Rule (b) of D-R19 §3: a Wake of the same Goal with a later cause exists."""

    with TestClient(app) as client:
        state = client.app.state
        x, _gid = _qualified_goal(state)
        _qualified_commitment(state, x)
        # Only the COMMITMENT candidate is scanned first (the Goal's is hidden).
        materializer = state.wake_materializer
        real = materializer._admission_candidates
        materializer._admission_candidates = lambda: [
            c for c in real() if c.cause is WakeCause.COMMITMENT_ADMITTED
        ]
        first = _scan(state)
        assert first.wake.cause is WakeCause.COMMITMENT_ADMITTED
        materializer._admission_candidates = real
        second = _scan(state)
        assert second.status is MaterializeStatus.NONE_LEFT
        assert (WakeCause.GOAL_ADMITTED, SkipReason.LATER_CAUSE_SCHEDULED) in _skips(second)
        assert len(_wakes(state)) == 1


# ------------------------------------------------------ the two Outcome causes


def _only_outcome_causes(state) -> None:  # type: ignore[no-untyped-def]
    """Hide the Goal / Commitment admissions: these tests are about the two
    Outcome causes alone (an admission candidate skipped as SAME_FINGERPRINT
    stays a candidate, see the module notes of the report)."""

    state.wake_materializer._admission_candidates = lambda: []


def test_feedback_makes_a_wake_with_the_outcome_and_commitment_refs(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        gid, intent, _forecast, _claim = ip19._chain_to_claim(state)
        _only_outcome_causes(state)
        outcome = ip19._commit_outcome(state, intent)
        assessed = ip19._assess(state)
        assert assessed.assessment.outcome_state is OutcomeState.ACTIVE_VERIFIED_AUTHORIZED
        # An Assessment of an authorized Outcome is not "no Feedback" (#369
        # D2-A1), and no Wake exists before a qualified Feedback (#370 A1-4).
        quiet = _scan(state)
        assert (quiet.status, _skips(quiet)) == (MaterializeStatus.NONE_LEFT, [])
        assert _wakes(state) == []
        feedback = ip19._feedback(state).feedback
        result = _scan(state)
        assert result.status is MaterializeStatus.COMMITTED
        wake = result.wake
        assert (wake.cause, wake.goal_id) == (WakeCause.FEEDBACK_ADMITTED, gid)
        assert wake.cause_ref == feedback.policy_ref()
        assert wake.cause_event_id == feedback.admitting_event_id
        assert wake.cause_sequence == feedback.admitting_sequence
        # One r18-outcome ref plus the Goal's commitment ref and deadline (P2-6).
        assert [ref.kind for ref in wake.origin_refs] == ["commitment", "r18-outcome"]
        assert wake.verified_outcome_key == (outcome.verified_outcome_id, 1)
        assert wake.commitment_ref() is not None
        assert wake.commitment_deadline_utc is not None
        assert wake.not_before_utc < wake.expires_at_utc <= wake.commitment_deadline_utc
        assert _wakes(state) == [wake]
        # The next scan: the cause event has its Wake.
        again = _scan(state)
        assert (again.status, _skips(again)) == (
            MaterializeStatus.NONE_LEFT,
            [(WakeCause.FEEDBACK_ADMITTED, SkipReason.CAUSE_ALREADY_WOKEN)],
        )
        ip16a2._no_failed_events(state)


def test_a_feedback_wake_supersedes_a_commitment_wake_of_the_same_goal(app) -> None:  # type: ignore[no-untyped-def]
    """P2-6: the Outcome-cause Wake carries the commitment ref and deadline, so
    ``may_supersede(commitment_wake, feedback_wake)`` holds (D-6, HD-20-5 (b))."""

    with TestClient(app) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = ip19._chain_to_claim(state)
        materializer = state.wake_materializer
        real = materializer._admission_candidates
        materializer._admission_candidates = lambda: [
            c for c in real() if c.cause is WakeCause.COMMITMENT_ADMITTED
        ]
        commitment_wake = _scan(state).wake
        assert commitment_wake.cause is WakeCause.COMMITMENT_ADMITTED
        ip19._commit_outcome(state, intent)
        ip19._assess(state)
        ip19._feedback(state)
        _only_outcome_causes(state)
        feedback_wake = _scan(state).wake
        assert feedback_wake.cause is WakeCause.FEEDBACK_ADMITTED
        assert may_supersede(commitment_wake, feedback_wake)


@pytest.mark.parametrize(
    "outcome_state", (VerifiedState.VERIFIED_ABSENT, VerifiedState.UNAUTHORIZED_EFFECT)
)
def test_a_terminal_not_eligible_verdict_makes_the_no_feedback_wake(app, outcome_state) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        gid, intent, _forecast, _claim = ip19._chain_to_claim(state)
        _only_outcome_causes(state)
        outcome = ip19._commit_outcome(state, intent, outcome_state)
        assessment = ip19._assess(state).assessment
        assert assessment.positive_feedback_eligible is False
        result = _scan(state)
        assert result.status is MaterializeStatus.COMMITTED
        wake = result.wake
        assert (wake.cause, wake.goal_id) == (WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK, gid)
        assert wake.cause_ref.kind == "r17-assessment"
        assert wake.cause_ref.ref_id == assessment.assessment_id
        assert wake.verified_outcome_key == (outcome.verified_outcome_id, 1)
        assert [ref.kind for ref in wake.origin_refs] == ["commitment", "r18-outcome"]
        # One Wake per verified Outcome revision (the exclusivity key).
        assert _scan(state).status is MaterializeStatus.NONE_LEFT
        assert len(_wakes(state)) == 1
        ip16a2._no_failed_events(state)


def test_no_assessment_means_no_outcome_wake(app) -> None:  # type: ignore[no-untyped-def]
    """The materializer reads Assessments, never the absence of a Feedback."""

    with TestClient(app) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = ip19._chain_to_claim(state)
        _only_outcome_causes(state)
        ip19._commit_outcome(state, intent, VerifiedState.VERIFIED_ABSENT)
        result = _scan(state)
        assert (result.status, _skips(result)) == (MaterializeStatus.NONE_LEFT, [])
        assert _wakes(state) == []


def test_a_feedback_whose_outcome_has_a_newer_revision_is_skipped(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, intent, _forecast, _claim = ip19._chain_to_claim(state)
        _only_outcome_causes(state)
        ip19._commit_outcome(state, intent)
        ip19._assess(state)
        ip19._feedback(state)
        ip19._commit_outcome(state, intent, VerifiedState.VERIFIED_ABSENT, revision=2)
        result = _scan(state)
        assert result.status is MaterializeStatus.NONE_LEFT
        assert _skips(result) == [(WakeCause.FEEDBACK_ADMITTED, SkipReason.OUTCOME_SUPERSEDED)]
        assert _wakes(state) == []


# ---------------------------------------------------------- self-origin Goal


def test_a_self_origin_goal_wake_cites_its_motivation_experience_and_event(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import test_r16_ip11_self_goal as ip11

    base = ip11._history_settings(tmp_path / "self")
    settings = base.model_copy(update={"r19": base.r19.model_copy(update={"enabled": True})})
    with TestClient(_app(settings)) as client:
        state = client.app.state
        ip11._activate(state)
        x = ip11._interest(state)
        goal_id = ip11._propose(state, x).subject_id
        ip11._review(state, goal_id)
        result = _scan(state)
        assert result.status is MaterializeStatus.COMMITTED
        wake = result.wake
        assert (wake.cause, wake.goal_id) == (WakeCause.GOAL_ADMITTED, goal_id)
        assert wake.goal_origin.value == "SUBJECT_MOTIVATION"
        assert [ref.kind for ref in wake.origin_refs] == ["event", "experience", "motivation"]
        assert [ref.kind for ref in wake.origin_refs] == sorted(ref.kind for ref in wake.origin_refs)
        assert len(wake.origin_refs) <= 4
        assert _wakes(state) == [wake]


# ----------------------------------------------------- r19.enabled and holds


def test_disabled_holds_before_reading_anything(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", r19=False))) as client:
        state = client.app.state
        materializer = state.wake_materializer
        assert materializer is not None and materializer.enabled is False
        _qualified_goal(state)
        reads: list[str] = []
        monkeypatch.setattr(
            materializer, "_open_scan", lambda *_a: reads.append("scan") or pytest.fail("read")
        )
        monkeypatch.setattr(materializer, "_candidates", lambda *_a: reads.append("cands"))
        result = _scan(state)
        assert (result.status, result.reason) == (MaterializeStatus.HOLD, MaterializeReason.R19_DISABLED)
        assert reads == []
        assert _wake_rows(state) == []
        ip16a2._no_failed_events(state)


class _Failing:
    """The owner's own handle on a collaborator, failing one call (the runtime
    keeps its own handle, so only the materializer sees the failure)."""

    def __init__(self, real: object, name: str, error: BaseException) -> None:
        self._real, self._name, self._error = real, name, error

    def __getattr__(self, name: str) -> Any:
        if name == self._name:

            def fail(*_args: object, **_kwargs: object) -> Any:
                raise self._error

            return fail
        return getattr(self._real, name)


def test_an_unreconciled_journal_or_blocked_history_is_a_transient_hold(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        materializer = state.wake_materializer
        journal, store = materializer._journal, materializer._store
        materializer._journal = _Failing(journal, "inspect", OSError("journal"))
        held = _scan(state)
        assert (held.status, held.reason) == (MaterializeStatus.HOLD, MaterializeReason.HISTORY_UNAVAILABLE)
        materializer._journal = journal
        # A store read failure inside the scan is the same transient hold.
        materializer._store = _Failing(store, "operations", HistoryStoreCapacityError("store"))
        held = _scan(state)
        assert (held.status, held.reason) == (MaterializeStatus.HOLD, MaterializeReason.HISTORY_UNAVAILABLE)
        materializer._store = store
        assert _scan(state).status is MaterializeStatus.COMMITTED
        ip16a2._no_failed_events(state)


# ------------------------------------------------------- skips write nothing


def _assert_skip_writes_nothing(state, result, expected) -> None:  # type: ignore[no-untyped-def]
    assert result.status is MaterializeStatus.NONE_LEFT
    assert _skips(result) == expected
    assert _wakes(state) == [] and _wake_rows(state) == []


def test_a_candidate_that_does_not_qualify_is_skipped_and_retried(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """D1 §3-1 step 5: no fingerprint (UNAVAILABLE source / A-2 TARGET_UNAVAILABLE)
    is NOT_QUALIFIED, logged, never frozen, and retried by the next scan."""

    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        monkeypatch.setattr("suzka.r19.contracts.wake_fingerprint", lambda _value: None)
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.NOT_QUALIFIED)]
        )
        monkeypatch.undo()
        assert _scan(state).status is MaterializeStatus.COMMITTED


def test_an_unqualified_source_or_an_issuer_hold_is_not_qualified(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        issuer = state.decision_issuer
        real = issuer._evaluate

        def hold(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise _IssuerHold(R16OwnerReason.SOURCE_UNAVAILABLE)

        monkeypatch.setattr(issuer, "_evaluate", hold)
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.NOT_QUALIFIED)]
        )
        monkeypatch.setattr(issuer, "_evaluate", real)
        # An unqualified Goal source is the same skip (the issuer's row-0 / row-1 input).
        monkeypatch.setattr(
            state._plan_preview_owner,
            "_qualify",
            lambda *_a, **_k: (_ for _ in ()).throw(
                __import__("suzka.r16.plan_preview", fromlist=["x"])._PlanSourceUnavailable("gone")
            ),
        )
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.NOT_QUALIFIED)]
        )


def test_a_stale_cause_is_skipped(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """(a): the Goal in the cause is no longer the qualified current one."""

    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        issuer = state.decision_issuer
        real = issuer._evaluate

        def other_admission(*args, **kwargs):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            evaluated = real(*args, **kwargs)
            return replace(evaluated, q=replace(evaluated.q, goal_adoption_event_id="x" * 36))

        monkeypatch.setattr(issuer, "_evaluate", other_admission)
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.STALE_CAUSE)]
        )


def test_a_cause_that_cannot_be_timed_or_is_over_age_makes_no_wake(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """HD-20-1 (a): current incarnation, retained Journal record, at most
    ``materialize_max_age_s`` old."""

    import suzka.r19.materializer as module

    with TestClient(_app(_settings(tmp_path / "age", materialize_max_age_s=1))) as client:
        state = client.app.state
        _qualified_goal(state)
        materializer = state.wake_materializer
        # Over age: the cause's Journal record is older than the bound.
        materializer._max_age = timedelta(seconds=-1)
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.OUT_OF_SCOPE)]
        )
        materializer._max_age = timedelta(days=7)
        # Past retention (the record is gone): out of scope.
        real = module.locate_journal_source
        monkeypatch.setattr(
            module,
            "locate_journal_source",
            lambda inspection, _e, _s: (module.JournalSourceStage.MISSING, ()),
        )
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.OUT_OF_SCOPE)]
        )
        monkeypatch.setattr(module, "locate_journal_source", real)
        # An old incarnation: the cause sequence is not above the horizon.
        reader = module.SubjectHistoryReader
        real_classify = reader.classify

        def rolled_back(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            return replace(real_classify(self, *args, **kwargs), horizon=10**9)

        monkeypatch.setattr(reader, "classify", rolled_back)
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.OUT_OF_SCOPE)]
        )
        monkeypatch.undo()
        assert _scan(state).status is MaterializeStatus.COMMITTED


def test_a_goal_wake_whose_due_time_cannot_fit_is_not_schedulable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """(d) ``wake_window`` is None: logged, no row; the harness must choose the delays."""

    settings = _settings(tmp_path / "late", goal_wake_delay_s=604_800)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _qualified_goal(state)
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.WAKE_NOT_SCHEDULABLE)]
        )


def test_a_cause_after_t_is_skipped_and_never_raised(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A rolled-back clock: ``wake_window`` raises ``ValueError``; the scan
    skips, and the event completes without failing."""

    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        monkeypatch.setattr(
            state.wake_materializer, "_scope", lambda scan, candidate: "2999-01-01T00:00:00Z"
        )
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.NOT_SCHEDULABLE)]
        )
        ip16a2._no_failed_events(state)


def test_the_no_deadline_seal_becomes_none(app) -> None:
    """The issuer seals a Commitment without a deadline as ``NO_DEADLINE_UTC``;
    the Wake field and the window take ``None`` for it (D1 §3-1)."""

    from suzka.r16.decision_issuer import NO_DEADLINE_UTC

    with TestClient(app) as client:
        state = client.app.state
        x, _gid = _qualified_goal(state)
        _qualified_commitment(state, x)
        issuer = state.decision_issuer
        real = issuer._evaluate

        def no_deadline(*args, **kwargs):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            evaluated = real(*args, **kwargs)
            return replace(
                evaluated, value=replace(evaluated.value, commitment_deadline_utc=NO_DEADLINE_UTC)
            )

        issuer._evaluate = no_deadline  # type: ignore[method-assign]
        wake = _scan(state).wake
        assert wake.commitment_deadline_utc is None
        assert wake.expires_at_utc < NO_DEADLINE_UTC


# ------------------------------------------------- same fingerprint, for good


def test_the_same_fingerprint_never_makes_a_second_wake_even_after_the_cycle_ended(app) -> None:  # type: ignore[no-untyped-def]
    """R20-F10: the same evidence reuses one ``wakeup_id``; the ``wake:`` key
    stays live whatever happens to the cycle, so a FAILED terminal does not free it."""

    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        wake = _scan(state).wake

        def build(event):  # type: ignore[no-untyped-def]
            record = WakeTerminalV1.for_wake(
                wake,
                WakeTerminalKind.FAILED_TERMINAL,
                TerminalReason.EXPIRED_WAKE,
                event_id=event.event_id,
                event_sequence=event.processing_sequence,
            )
            return ip16b._operation(state, event, K.R19_WAKE_TERMINAL, R.WAKE_TERMINAL, record)

        ip16b._commit_record(state, build)
        # The Wake is FAILED_TERMINAL; the scan finds ``wake:`` live and writes nothing.
        assert state.subject_history_store.live_keys((f"wake:{wake.wakeup_id}",))
        again = _scan(state)
        assert again.status is MaterializeStatus.NONE_LEFT
        assert len(_wakes(state)) == 1


# ---------------------------------------------------- PREPARE can never raise


def _never_prepare(monkeypatch) -> list[object]:  # type: ignore[no-untyped-def]
    from suzka.r16.history_participant import SubjectHistoryRecordParticipant

    entered: list[object] = []
    real = SubjectHistoryRecordParticipant.prepare

    def spy(self, binding):  # type: ignore[no-untyped-def]
        entered.append(binding)
        return real(self, binding)

    monkeypatch.setattr(SubjectHistoryRecordParticipant, "prepare", spy)
    return entered


@pytest.mark.parametrize(
    "key_of",
    (
        lambda wake: f"wake:{wake.wakeup_id}",
        lambda wake: f"wake-cause:{wake.incarnation_id}:{wake.cause_event_id}",
        lambda wake: f"wake-outcome:{wake.incarnation_id}:o:1",
    ),
    ids=("wake", "wake-cause", "wake-outcome"),
)
def test_a_live_key_holds_before_prepare(app, monkeypatch, key_of) -> None:  # type: ignore[no-untyped-def]
    """IP-14b §2-6: a key of the Wake that is live in the raw index (a PREPARED
    or FINALIZED-not-terminal row the reader cannot see) is HOLD IN_FLIGHT, and
    ``participant.prepare`` is never entered with a colliding key."""

    from suzka.r16.history_store import PrepareCollisions

    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        entered = _never_prepare(monkeypatch)
        store = state.subject_history_store
        real_collisions = store.prepare_collisions
        seen: list[tuple[str, ...]] = []

        def collisions(operation):  # type: ignore[no-untyped-def]
            seen.append(operation.reserved_keys)
            key = key_of(operation.record)
            return PrepareCollisions(frozenset({key}), False)

        monkeypatch.setattr(store, "prepare_collisions", collisions)
        held = _scan(state)
        assert (held.status, held.reason) == (MaterializeStatus.HOLD, MaterializeReason.IN_FLIGHT)
        assert entered == [] and _wake_rows(state) == []
        assert seen and any(key.startswith("wake:") for key in seen[0])
        # Collision gone: the same scan now commits.
        monkeypatch.setattr(store, "prepare_collisions", real_collisions)
        assert _scan(state).status is MaterializeStatus.COMMITTED
        assert len(entered) == 1
        ip16a2._no_failed_events(state)


def test_capacity_and_a_switched_off_store_hold_before_prepare(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)
        entered = _never_prepare(monkeypatch)
        store = state.subject_history_store

        def no_room(_length):  # type: ignore[no-untyped-def]
            raise HistoryStoreCapacityError("no headroom")

        monkeypatch.setattr(store, "check_headroom", no_room)
        held = _scan(state)
        assert (held.status, held.reason) == (MaterializeStatus.HOLD, MaterializeReason.HISTORY_CAPACITY)
        monkeypatch.undo()
        entered = _never_prepare(monkeypatch)
        # The store's own family switch (a materializer enabled over a store opened
        # with r19 off would raise at PREPARE): HOLD R19_DISABLED, no prepare.
        monkeypatch.setattr(store, "_r19_enabled", False)
        held = _scan(state)
        assert (held.status, held.reason) == (MaterializeStatus.HOLD, MaterializeReason.R19_DISABLED)
        assert entered == [] and _wake_rows(state) == []
        monkeypatch.undo()
        assert _scan(state).status is MaterializeStatus.COMMITTED
        ip16a2._no_failed_events(state)


def test_an_invalid_wake_is_skipped_never_raised(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A closed contract refusal while building the Wake is a logged skip, so
    the scan never raises into the runtime."""

    import suzka.r19.materializer as module

    with TestClient(app) as client:
        state = client.app.state
        _qualified_goal(state)

        def bad(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("a Wake becomes due before it expires")

        monkeypatch.setattr(module, "WakeV1", bad)
        _assert_skip_writes_nothing(
            state, _scan(state), [(WakeCause.GOAL_ADMITTED, SkipReason.INVALID_WAKE)]
        )
        ip16a2._no_failed_events(state)


# ------------------------------------------------------- _clear and wiring


def test_two_scans_in_a_row_and_a_failed_one_leave_no_pending_state(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The server's ``_clear`` wiring: a stale ``_pending`` would make the next
    event raise "already pending"."""

    with TestClient(app) as client:
        state = client.app.state
        materializer = state.wake_materializer
        x, _gid = _qualified_goal(state)
        assert _scan(state).status is MaterializeStatus.COMMITTED
        assert (materializer._pending, materializer._inflight) == (None, None)
        _qualified_commitment(state, x)
        assert _scan(state).status is MaterializeStatus.COMMITTED
        assert (materializer._pending, materializer._inflight) == (None, None)
        # A scan that fails mid-handler (after it prepared) clears its pending.
        real = materializer._prepare

        def prepare_then_fail(scan, operation):  # type: ignore[no-untyped-def]
            real(scan, operation)
            raise RuntimeError("mid-handler failure")

        ip16a2._second_goal(state)
        monkeypatch.setattr(materializer, "_prepare", prepare_then_fail)
        with pytest.raises(Exception):
            _scan(state)
        monkeypatch.undo()
        assert materializer._pending is None
        assert _scan(state).status is MaterializeStatus.COMMITTED


def test_the_server_wires_validate_commit_and_clear(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        materializer = state.wake_materializer
        _qualified_goal(state)
        validated: list[object] = []
        cleared: list[int] = []
        real_validate, real_clear = materializer._validate_commit, materializer._clear
        monkeypatch.setattr(
            materializer, "_validate_commit", lambda event: validated.append(event) or real_validate(event)
        )
        monkeypatch.setattr(materializer, "_clear", lambda: cleared.append(1) or real_clear())
        assert _scan(state).status is MaterializeStatus.COMMITTED
        scan_events = [e for e in validated if e.event_type is MATERIALIZE]
        assert len(scan_events) == 1 and cleared
        # The pending of a scan is still set when the pre-WAL check runs for it.
        assert materializer._pending is None


def test_validate_commit_refuses_stale_or_changed_state(app) -> None:  # type: ignore[no-untyped-def]
    from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource

    with TestClient(app) as client:
        state = client.app.state
        materializer = state.wake_materializer
        _qualified_goal(state)
        captured: list[object] = []
        real = materializer._prepare

        def spy(scan, operation):  # type: ignore[no-untyped-def]
            result = real(scan, operation)
            captured.append((scan.event, materializer._pending))
            return result

        materializer._prepare = spy  # type: ignore[method-assign]
        assert _scan(state).status is MaterializeStatus.COMMITTED
        event, pending = captured[0]
        # Another event with a pending set is a stale pending; the same event
        # with no PREPARED row (the scan's row is FINALIZED now) is refused.
        other = AgentEvent(
            "11111111-1111-4111-8111-111111111111",
            AgentEventType.CHAT,
            AgentEventSource.API_CHAT,
            event.requested_at,
            event.processing_sequence + 1000,
        )
        materializer._pending = pending
        with pytest.raises(RuntimeError):
            materializer._validate_commit(other)
        with pytest.raises(RuntimeError):
            materializer._validate_commit(event)  # its row is no longer PREPARED
        materializer._clear()
        materializer._validate_commit(event)  # nothing pending: nothing to check


def test_the_owner_refuses_an_unqualified_binding(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        issuer, runtime = state.decision_issuer, state.agent_runtime
        for kwargs in (
            {"enabled": 1},
            {"enabled": True, "max_age_s": 0},
            {"enabled": True, "max_age_s": 604_801},
            {"enabled": True, "timing": object()},
        ):
            with pytest.raises(TypeError):
                WakeMaterializerV1(issuer, runtime, **kwargs)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            WakeMaterializerV1(object(), runtime, enabled=True)  # type: ignore[arg-type]


# ----------------------------------------------------------- the driver hook


def test_the_driver_never_imports_r19_and_calls_a_nonblocking_hook() -> None:
    import ast
    from pathlib import Path

    import suzka.r16.driver as driver_module

    tree = ast.parse(Path(driver_module.__file__).read_text())
    imported = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not [name for name in imported if name.startswith("suzka.r19")]


def _hooked_driver(**kwargs: Any):  # type: ignore[no-untyped-def]
    from suzka.r16.driver import R16Driver

    from test_r17_ip19_driver import _FakeFeedback

    calls: list[int] = []
    feedback = _FakeFeedback()
    driver = R16Driver(
        ip16a2._FakeRuntime(),  # type: ignore[arg-type]
        *[ip16a2._FakeOwner() for _ in range(3)],  # type: ignore[arg-type]
        ip16a2._FakeOwner(),  # type: ignore[arg-type]
        active=True,
        clock=ip16a2._Clock(),
        feedback_owner=feedback,  # type: ignore[arg-type]
        on_cause_terminal=lambda: calls.append(1),
        **kwargs,
    )
    return driver, feedback, calls


def test_the_driver_nudges_after_each_cause_is_terminal() -> None:
    """D1 §3-3: Goal ADOPTED, Commitment ACCEPTED, Assessment and Feedback
    COMMITTED or EXISTING; never after a HOLD or NONE_LEFT."""

    from concurrent.futures import Future

    from suzka.r16.driver import _Done, _Stage
    from suzka.r16.outcomes import R16OwnerOutcome, R16OwnerReason, R16OwnerStatus
    from suzka.r16.driver import _FeedbackTarget
    from suzka.r17.feedback_owner import (
        AssessmentCommitStatus,
        FeedbackCommitStatus,
        R17FeedbackReason,
        _AssessmentResult,
        _FeedbackResult,
    )
    from suzka.runtime.agent_runtime import AgentEventOutcome
    from test_r17_ip19_driver import _assessed, _fed, _none_left_assessment, _none_left_feedback

    driver, feedback, calls = _hooked_driver()

    def done(stage, value):  # type: ignore[no-untyped-def]
        future: Future[Any] = Future()
        future.set_result(AgentEventOutcome(ip16a2._event(), value))
        driver._on_done(_Done(stage, future, None))

    done(_Stage.GOAL_REVIEW, R16OwnerOutcome(R16OwnerStatus.ADOPTED, "goal-1"))
    done(_Stage.COMMITMENT_REVIEW, R16OwnerOutcome(R16OwnerStatus.ACCEPTED, "commitment-1"))
    assert len(calls) == 2
    done(_Stage.GOAL_REVIEW, R16OwnerOutcome.hold(R16OwnerReason.GOAL_NOT_ADOPTED))
    assert len(calls) == 2
    driver._assessment_done(_assessed("a1"))
    assert len(calls) == 3
    driver._assessment_done(_none_left_assessment())
    driver._assessment_done(
        _AssessmentResult(
            AssessmentCommitStatus.HOLD, None, None, R17FeedbackReason.OUTCOME_UNKNOWN
        )
    )
    assert len(calls) == 3
    driver._feedback_done(_FeedbackTarget(None), _fed("a1"))
    driver._feedback_done(_FeedbackTarget("a1"), _fed("a1"))
    assert len(calls) == 5
    driver._feedback_done(_FeedbackTarget(None), _none_left_feedback())
    assert len(calls) == 5
    existing = _fed("a2")
    existing = _FeedbackResult(
        FeedbackCommitStatus.EXISTING, existing.assessment_id, existing.feedback_id, None, existing.feedback
    )
    driver._feedback_done(_FeedbackTarget("a2"), existing)
    assert len(calls) == 6
    existing_assessment = _assessed("a3")
    existing_assessment = _AssessmentResult(
        AssessmentCommitStatus.EXISTING,
        existing_assessment.outcome_id,
        existing_assessment.assessment_id,
        None,
        existing_assessment.assessment,
    )
    driver._assessment_done(existing_assessment)
    assert len(calls) == 7


def test_a_failing_or_absent_hook_never_stops_the_driver() -> None:
    from suzka.r16.driver import R16Driver

    driver, _feedback, _calls = _hooked_driver()
    driver._on_cause_terminal = lambda: (_ for _ in ()).throw(RuntimeError("scheduler"))
    driver._cause_terminal()  # swallowed, logged
    plain = R16Driver(
        ip16a2._FakeRuntime(),  # type: ignore[arg-type]
        *[ip16a2._FakeOwner() for _ in range(3)],  # type: ignore[arg-type]
        ip16a2._FakeOwner(),  # type: ignore[arg-type]
        active=True,
        clock=ip16a2._Clock(),
    )
    assert plain._on_cause_terminal is None
    plain._cause_terminal()
