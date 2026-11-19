"""IP-20a: settings, event kinds, DTO amendments, R15 gate, issuer plumbing,
``AgentRuntime.is_idle`` (IP-20 D1 r3 §9-1).  No writer and no scheduler exist
yet, so none is tested here.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from suzka.config.schema import R19Settings, Settings
from suzka.r16 import decision_issuer as issuer_module
from suzka.r16.decision_issuer import (
    DEADLINE_MARGIN,
    _DecisionIssuer,
    _Evaluation,
    _IssuerHold,
)
from suzka.r16.history_codec import PolicyKind
from suzka.r16.outcomes import R16OwnerReason
from suzka.r16.successor_policy import (
    Interpretation,
    InterpretationStage,
    LedgerState,
    PredecessorState,
    SuccessorRow,
    as_version,
    utc_timestamp,
)
from suzka.r18.facts import TARGET_UNAVAILABLE_LEDGER
from suzka.r19 import contracts
from suzka.r19.contracts import (
    ATTEMPT_MAX_EVENTS,
    ATTEMPT_MAX_MODEL_CALLS,
    COMMITMENT_ADMISSION_REF_KIND,
    COMMITMENT_REF_KIND,
    CYCLE_MAX_EVENTS,
    EVENT_REF_KIND,
    EXPERIENCE_REF_KIND,
    GOAL_ADMISSION_REF_KIND,
    MAX_ATTEMPTS,
    MAX_WAKE_TTL,
    MOTIVATION_REF_KIND,
    WakeTiming,
    WakeV1,
    wake_fingerprint,
)
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.event_journal import (
    SOFT_BAND_ADMITTED_EVENT_TYPES,
    EventJournalAdmissionBand,
)
from suzka.runtime.r15_interpretation_reader import R15InterpretationReader

from test_r15_ip8a_interpretation_v2 import _as_current, _r16_read, _run
from test_r16_ip10_decision_issuer import _settings
from test_r16_ip8b_successor_policy import _active, _verified
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_goal
from test_r19_ip13b_contracts import _wake
from tests.test_event_journal_capacity_bands import _at_band, _typed

V2 = PolicyKind.SUCCESSOR_POLICY_V2
V3 = PolicyKind.SUCCESSOR_POLICY_V3

WAKE_TYPES = (
    AgentEventType.R19_WAKE_MATERIALIZE,
    AgentEventType.R19_WAKE_CLAIM,
    AgentEventType.R16_WAKE_REVIEW,
    AgentEventType.R19_WAKE_TERMINAL,
)
WAKE_SOURCES = (
    AgentEventSource.RUNTIME_R19_WAKE_MATERIALIZE,
    AgentEventSource.RUNTIME_R19_WAKE_CLAIM,
    AgentEventSource.RUNTIME_R16_WAKE_REVIEW,
    AgentEventSource.RUNTIME_R19_WAKE_TERMINAL,
)


# ----------------------------------------------------------------- settings


def test_r19_settings_defaults_are_the_d_r19_numbers_and_r19_stays_off() -> None:
    settings = R19Settings()
    assert settings.enabled is False
    timing = WakeTiming()
    for name in (
        "goal_wake_delay_s",
        "commitment_wake_delay_s",
        "feedback_wake_delay_s",
        "deadline_margin_s",
        "claim_lease_s",
        "tick_s",
    ):
        assert getattr(settings, name) == getattr(timing, name)
    assert (settings.rescan_s, settings.materialize_max_age_s) == (1800, 604_800)
    assert settings.materialize_max_age_s == int(MAX_WAKE_TTL.total_seconds())
    # A configuration with no ``r19`` section (eve's /etc) takes these defaults.
    default_factory = Settings.model_fields["r19"].default_factory
    assert default_factory is not None and default_factory() == settings  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "changes",
    [
        {"claim_lease_s": 0},
        {"tick_s": 0},
        {"deadline_margin_s": 0},  # not_before == expires must be impossible
        {"rescan_s": 0},
        {"materialize_max_age_s": 0},
        {"materialize_max_age_s": 604_801},  # above the Wake TTL
        {"goal_wake_delay_s": -1},
        {"feedback_wake_delay_s": 604_801},
        {"claim_lease_s": True},
        {"tick_s": 1.5},
        {"unknown_number": 3},
    ],
)
def test_r19_settings_are_validated(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        R19Settings(**changes)


def test_r19_settings_are_validated_through_wake_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model builds a ``WakeTiming`` from its numbers, so a rule added there
    reaches the settings with no second copy."""

    seen: list[WakeTiming] = []
    real = WakeTiming.__post_init__

    def spy(self: WakeTiming) -> None:
        seen.append(self)
        real(self)

    monkeypatch.setattr(WakeTiming, "__post_init__", spy)
    R19Settings(claim_lease_s=90, tick_s=15)
    assert [(item.claim_lease_s, item.tick_s) for item in seen] == [(90, 15)]


# -------------------------------------------------------------- event kinds


def test_the_four_wake_event_kinds_and_sources_exist() -> None:
    assert [item.value for item in WAKE_TYPES] == [
        "r19_wake_materialize",
        "r19_wake_claim",
        "r16_wake_review",
        "r19_wake_terminal",
    ]
    assert [item.value for item in WAKE_SOURCES] == [
        "runtime.r19.wake_materialize",
        "runtime.r19.wake_claim",
        "runtime.r16.wake_review",
        "runtime.r19.wake_terminal",
    ]
    # IP-21's two kinds are not defined here.
    assert not {item.name for item in AgentEventType} & {
        "R19_WAKE_COMPLETE",
        "R19_INBOX_ACK",
    }


@pytest.mark.parametrize("event_type", WAKE_TYPES)
def test_each_wake_event_is_admitted_in_the_soft_band_and_refused_in_the_hard_band(
    tmp_path: Path, event_type: AgentEventType
) -> None:
    assert event_type in SOFT_BAND_ADMITTED_EVENT_TYPES
    _path, _clock, soft, _following = _at_band(
        tmp_path / "soft", EventJournalAdmissionBand.SOFT
    )
    assert soft.admission_status().admits(event_type)
    assert soft.append_accepted_if_admission_available(
        _typed(f"soft-{event_type.value}", event_type)
    )
    soft.close()
    _path, _clock, hard, _following = _at_band(
        tmp_path / "hard", EventJournalAdmissionBand.HARD
    )
    assert not hard.admission_status().admits(event_type)
    before = hard.records
    assert not hard.append_accepted_if_admission_available(
        _typed(f"hard-{event_type.value}", event_type)
    )
    assert hard.records == before  # a refusal writes nothing
    hard.close()


def test_the_r18_intent_issue_is_still_refused_in_the_soft_band() -> None:
    assert AgentEventType.R18_ACTION_INTENT_ISSUE not in SOFT_BAND_ADMITTED_EVENT_TYPES


# ----------------------------------------------------------------- is_idle


def test_is_idle_sees_a_running_handler_that_pending_count_cannot() -> None:
    """A-17: a read-only view of "no waiting and no running event"."""

    runtime = AgentRuntime(8, allow_volatile=True)
    assert runtime.is_idle() is False  # not accepting yet
    runtime.start()
    entered, release = Event(), Event()

    def hold() -> str:
        entered.set()
        assert release.wait(timeout=30)
        return "done"

    try:
        assert runtime.is_idle() is True
        first = runtime.submit(AgentEventType.CHAT, AgentEventSource.API_CHAT, hold)
        assert entered.wait(timeout=30)
        assert runtime.pending_count() == 0  # the running event is not waiting...
        assert runtime.is_idle() is False  # ...and is_idle still sees it
        second = runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: "next"
        )
        assert runtime.pending_count() == 1 and runtime.is_idle() is False
        release.set()
        assert first.result(timeout=30).value == "done"
        assert second.result(timeout=30).value == "next"
        deadline = time.monotonic() + 10
        while not runtime.is_idle() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.is_idle() is True
    finally:
        release.set()
        runtime.shutdown()
    assert runtime.is_idle() is False  # stopped


def test_is_idle_changes_nothing() -> None:
    runtime = AgentRuntime(2, allow_volatile=True)
    runtime.start()
    try:
        status, pending = runtime.status, runtime.pending_count()
        for _ in range(3):
            assert runtime.is_idle() is True
        assert (runtime.status, runtime.pending_count()) == (status, pending)
        assert runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, lambda: 1
        ).result(timeout=30)
    finally:
        runtime.shutdown()


# ------------------------------------------------------------ DTO: A-1, A-2


@pytest.mark.parametrize("kind", [V2, V3])
def test_wake_policy_kind_round_trips_and_is_part_of_the_bytes(
    kind: PolicyKind,
) -> None:
    wake = _wake(policy_kind=kind)
    decoded = WakeV1.from_json(wake.canonical_bytes())
    assert decoded == wake and decoded.policy_kind is kind
    assert json.loads(wake.canonical_bytes())["policy_kind"] == kind.value


def test_wake_policy_kind_is_validated_and_does_not_change_the_identity() -> None:
    v2, v3 = _wake(policy_kind=V2), _wake(policy_kind=V3)
    assert v2.wake_digest != v3.wake_digest
    # wakeup_id / logical_cycle_id are built from (incarnation, goal, kind, fingerprint) only.
    assert (v2.wakeup_id, v2.logical_cycle_id) == (v3.wakeup_id, v3.logical_cycle_id)
    with pytest.raises(ValueError):
        _wake(policy_kind=PolicyKind.SUCCESSOR_POLICY_V1)
    with pytest.raises(TypeError):
        _wake(policy_kind="SUCCESSOR_POLICY_V3")
    row = json.loads(v2.canonical_bytes())
    del row["policy_kind"]
    with pytest.raises(ValueError):
        WakeV1.from_json(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )
    row["policy_kind"] = "SUCCESSOR_POLICY_V1"
    with pytest.raises(ValueError):
        WakeV1.from_json(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )


def test_wake_fingerprint_never_pins_target_unavailable() -> None:
    """A-2 / HD-20-3 (a): the sweep gate's 60 s bound makes the state time-varying."""

    blocked = _active(
        version=V3,
        ledger=LedgerState.TARGET_UNAVAILABLE,
        ledger_refs=TARGET_UNAVAILABLE_LEDGER.refs,
    )
    assert wake_fingerprint(blocked) is None
    assert (
        wake_fingerprint(replace(blocked, ledger=LedgerState.NONE, ledger_refs=()))
        is not None
    )
    # Every other qualified ledger state is still frozen as before.
    for value in (_active(), _verified(), _active(version=V3)):
        assert wake_fingerprint(value) == value.evidence_fingerprint()


def test_as_version_matrix() -> None:
    """V2 -> V3 always; V3 -> V2 is None for a V3-only ledger state, and a None
    is "no match", never an error and never a match (forward note 1)."""

    v2 = _verified()
    assert v2.version is V2
    as_v3 = as_version(v2, V3)
    assert as_v3 is not None and as_v3.version is V3
    assert (
        as_v3.evidence_fingerprint() != v2.evidence_fingerprint()
    )  # one domain per version
    assert as_version(as_v3, V2) == v2
    assert as_version(v2, V2) is v2
    for state in (
        LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT,
        LedgerState.TARGET_UNAVAILABLE,
    ):
        v3_only = _active(
            version=V3, ledger=state, ledger_refs=TARGET_UNAVAILABLE_LEDGER.refs
        )
        assert as_version(v3_only, V2) is None
        assert as_version(v3_only, V3) is v3_only
    # A Wake frozen in V2 is compared in V2: the current V3 input is expressed in it.
    current = _active(version=V3)
    frozen = _active(version=V2)
    back = as_version(current, V2)
    assert (
        back is not None
        and back.evidence_fingerprint() == frozen.evidence_fingerprint()
    )


def test_origin_ref_kinds_of_a_self_origin_goal_are_pinned() -> None:
    """§14-9: named next to ``GOAL_ADMISSION_REF_KIND`` and fixed by this test."""

    assert (MOTIVATION_REF_KIND, EXPERIENCE_REF_KIND, EVENT_REF_KIND) == (
        "motivation",
        "experience",
        "event",
    )
    assert (
        GOAL_ADMISSION_REF_KIND,
        COMMITMENT_ADMISSION_REF_KIND,
        COMMITMENT_REF_KIND,
    ) == (
        "goal-admission",
        "commitment-admission",
        "commitment",
    )
    assert (
        len(
            {
                MOTIVATION_REF_KIND,
                EXPERIENCE_REF_KIND,
                EVENT_REF_KIND,
                COMMITMENT_REF_KIND,
            }
        )
        == 4
    )
    assert contracts.MAX_ORIGIN_REFS == 4


def test_budget_arithmetic_against_the_constants() -> None:
    """§4-6: per attempt 1 claim + at most 1 review; per cycle 1 materialize +
    3 attempts x 2 + 1 terminal + 1 complete (IP-21) = 9 <= 12."""

    per_attempt = 1 + 1
    assert per_attempt <= ATTEMPT_MAX_EVENTS == 4
    per_cycle = 1 + MAX_ATTEMPTS * per_attempt + 1 + 1
    assert per_cycle == 9 and per_cycle <= CYCLE_MAX_EVENTS == 12
    assert CYCLE_MAX_EVENTS - per_cycle == 3  # the stated slack
    # The Wake path has no model client, so the real count (0) is below the ceiling.
    assert ATTEMPT_MAX_MODEL_CALLS == 2
    assert MAX_ATTEMPTS == 3 and contracts.ATTEMPT_MAX_SECONDS == 10


# ----------------------------------------------------------------- R15 gate


def test_r15_read_gate_allows_exactly_the_three_review_events(tmp_path: Path) -> None:
    """A-3 (D-R19 §8-4): PLAN_PREVIEW_REVIEW, R16_WAKE_REVIEW and
    R19_WAKE_MATERIALIZE, each with its own source and before the transaction;
    UNAVAILABLE for every other type, every other source and after it opened."""

    allowed = {
        (
            AgentEventType.PLAN_PREVIEW_REVIEW,
            AgentEventSource.RUNTIME_PLAN_PREVIEW_REVIEW,
        ),
        (AgentEventType.R16_WAKE_REVIEW, AgentEventSource.RUNTIME_R16_WAKE_REVIEW),
        (
            AgentEventType.R19_WAKE_MATERIALIZE,
            AgentEventSource.RUNTIME_R19_WAKE_MATERIALIZE,
        ),
    }
    for client, refs in _run(
        tmp_path, ("A", "LIKELY_ALIGNED", "LOW"), ("B", "MAY_HAVE_MISREAD", "LOW")
    ):
        episode = refs[1]
        for event_type, source in sorted(allowed, key=lambda item: item[0].value):
            assert (
                _r16_read(client, episode, event_type=event_type, source=source).outcome
                == "EPISODE"
            )
        # Every other type (any of a few sources), and the new types with a foreign source.
        foreign_sources = (
            AgentEventSource.API_CHAT,
            AgentEventSource.RUNTIME_PLAN_PREVIEW_REVIEW,
            AgentEventSource.RUNTIME_R16_WAKE_REVIEW,
            AgentEventSource.RUNTIME_R19_WAKE_MATERIALIZE,
        )
        for event_type in AgentEventType:
            for source in foreign_sources:
                if (event_type, source) in allowed:
                    continue
                read = _r16_read(client, episode, event_type=event_type, source=source)
                assert read.outcome == "UNAVAILABLE", (event_type, source)
        for event_type, _source in allowed:
            for source in AgentEventSource:
                if (event_type, source) in allowed:
                    continue
                read = _r16_read(client, episode, event_type=event_type, source=source)
                assert read.outcome == "UNAVAILABLE", (event_type, source)
        # After the event opened its R07 transaction the source read refuses.
        journal = client.app.state.event_journal
        real = journal.inspect()
        for event_type, source in sorted(allowed, key=lambda item: item[0].value):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    journal,
                    "inspect",
                    lambda: replace(real, open_transactions=(object(),)),
                )
                assert (
                    _r16_read(
                        client, episode, event_type=event_type, source=source
                    ).outcome
                    == "UNAVAILABLE"
                )
        # A read outside any running event is UNAVAILABLE too.
        reader = R15InterpretationReader(client.app.state.r15_state_port)
        patch, event = _as_current(
            client,
            AgentEventType.R16_WAKE_REVIEW,
            AgentEventSource.RUNTIME_R16_WAKE_REVIEW,
        )
        patch.undo()
        assert (
            reader.read(event, episode.event_id, episode.event_sequence).outcome
            == "UNAVAILABLE"
        )
        break


# ------------------------------------------------------ issuer: _evaluate


def _stub_issuer(
    monkeypatch: pytest.MonkeyPatch, *, spy: dict[str, object]
) -> tuple[_DecisionIssuer, SimpleNamespace]:
    """A ``_DecisionIssuer`` whose ledger inputs are stubbed so only the
    plumbing of ``now`` and ``before_sequence`` is exercised."""

    issuer = _DecisionIssuer.__new__(_DecisionIssuer)
    issuer._target_facts = lambda: issuer_module.EMPTY_FACTS  # type: ignore[method-assign]
    issuer._blocked_target = lambda: False  # type: ignore[method-assign]
    issuer._group_key = lambda q: "a" * 64  # type: ignore[method-assign]
    issuer._store = SimpleNamespace(  # type: ignore[assignment]
        has_record_operations=lambda: True,
        has_live_claim_keys=lambda: True,
        live_key_owner=lambda key: None,
    )
    real_derive, real_feedback = (
        issuer_module.derive_ledger,
        issuer_module.read_feedback,
    )

    def derive(
        records: object, group_key: str, now: datetime, **kwargs: object
    ) -> object:
        spy["now"] = now
        return real_derive(records, group_key, now, **kwargs)  # type: ignore[arg-type]

    def feedback(
        records: object, key: str, state: object, before_sequence: int
    ) -> object:
        spy["before_sequence"] = before_sequence
        return real_feedback(records, key, state, before_sequence)  # type: ignore[arg-type]

    monkeypatch.setattr(issuer_module, "derive_ledger", derive)
    monkeypatch.setattr(issuer_module, "read_feedback", feedback)
    classified = SimpleNamespace(global_block=None, incarnation="i" * 64, records=())
    return issuer, classified


def test_the_ledger_takes_the_callers_instant_and_the_callers_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A-5 / §9-1: T replaces the clock in ``derive_ledger``; ``before_sequence``
    is the one given (the evaluating event's own), never read from anywhere else."""

    spy: dict[str, object] = {}
    issuer, classified = _stub_issuer(monkeypatch, spy=spy)
    issuer._clock = lambda: pytest.fail("the live clock must not be read")  # type: ignore[assignment]
    frozen = datetime(2026, 11, 16, 2, 0, 0, tzinfo=timezone.utc)
    q = SimpleNamespace(commitment_id="commitment-1")
    issuer._ledger(classified, PredecessorState.ABSENT_INITIAL, q, 77, frozen)  # type: ignore[arg-type]
    assert spy == {"now": frozen, "before_sequence": 77}


def test_the_deadline_check_uses_the_callers_instant_and_margin() -> None:
    issuer = _DecisionIssuer.__new__(_DecisionIssuer)
    issuer._clock = lambda: pytest.fail("the live clock must not be read")  # type: ignore[assignment]
    now = datetime(2026, 11, 16, 2, 0, 0, tzinfo=timezone.utc)
    commitments = {
        "c1": SimpleNamespace(deadline=SimpleNamespace(at=now + timedelta(seconds=12)))
    }
    commitments["c0"] = SimpleNamespace(deadline=SimpleNamespace(at=None))
    issuer._preview = SimpleNamespace(  # type: ignore[assignment]
        _loop=SimpleNamespace(_commitment_system=SimpleNamespace(get=commitments.get))
    )
    q = SimpleNamespace(commitment_id="c1")
    issuer._check_deadline(q, now=now)  # type: ignore[arg-type]  # 12 s left, 5 s margin
    # The Wake review's wider margin (5 s + 10 s handler budget, D1 §6 step 4).
    with pytest.raises(_IssuerHold) as hold:
        issuer._check_deadline(q, now=now, margin=timedelta(seconds=15))  # type: ignore[arg-type]
    assert hold.value.reason is R16OwnerReason.DEADLINE_ELAPSED
    with pytest.raises(_IssuerHold):
        issuer._check_deadline(q, now=now + timedelta(seconds=8))  # type: ignore[arg-type]  # 4 s < 5 s
    issuer._check_deadline(
        SimpleNamespace(commitment_id=None), now=now, margin=timedelta(days=1)
    )  # type: ignore[arg-type]
    issuer._check_deadline(
        SimpleNamespace(commitment_id="c0"), now=now, margin=timedelta(days=1)
    )  # type: ignore[arg-type]
    assert DEADLINE_MARGIN == timedelta(seconds=5)


@pytest.mark.parametrize(
    ("event_type", "source"),
    [
        (AgentEventType.R16_WAKE_REVIEW, AgentEventSource.RUNTIME_R16_WAKE_REVIEW),
        (
            AgentEventType.R19_WAKE_MATERIALIZE,
            AgentEventSource.RUNTIME_R19_WAKE_MATERIALIZE,
        ),
    ],
)
def test_evaluate_runs_in_a_wake_event_at_a_frozen_instant_and_writes_nothing(
    tmp_path: Path, event_type: AgentEventType, source: AgentEventSource
) -> None:
    """Real server, real runtime: ``_evaluate`` in an R16_WAKE_REVIEW or
    R19_WAKE_MATERIALIZE event seals T = ``requested_at`` and the event's own
    sequence, reads R15 through the widened gate, sets no ``_pending`` and
    writes no row."""

    with TestClient(_app(_settings(tmp_path / "evaluate"))) as client:
        state = client.app.state
        _x, goal_id = _qualified_goal(state)
        issuer = state.decision_issuer
        runtime = state.agent_runtime
        seen: dict[str, Any] = {}

        def handle() -> str:
            event = runtime.current_event()
            assert event is not None and event.event_type is event_type
            seen["event"] = event
            seen["evaluation"] = issuer._evaluate(
                event,
                goal_id,
                now=event.requested_at,
                before_sequence=event.processing_sequence,  # type: ignore[arg-type]
            )
            seen["pending"] = issuer._pending
            return "evaluated"

        issuer._clock = lambda: pytest.fail("the live clock must not be read")  # type: ignore[assignment]
        assert (
            runtime.submit(event_type, source, handle).result(timeout=30).value
            == "evaluated"
        )
        event: AgentEvent = seen["event"]
        evaluation = seen["evaluation"]
        assert type(evaluation) is _Evaluation
        value = evaluation.value
        assert value.now_utc == utc_timestamp(event.requested_at)  # sealed T
        assert (value.event_id, value.event_sequence) == (
            event.event_id,
            event.processing_sequence,
        )
        assert value.as_of_r07_sequence < event.processing_sequence
        # The R15 gate was passed: an unreviewed Experience is NOT_REVIEWED, not UNAVAILABLE.
        assert (value.interpretation, value.interpretation_stage) == (
            Interpretation.NOT_REVIEWED,
            InterpretationStage.NONE,
        )
        assert evaluation.result.row is SuccessorRow.ROW_3
        assert seen["pending"] is None and issuer._pending is None
        assert state.subject_history_store.has_operations() is False
        assert state.event_journal.inspect().open_transactions == ()

        # The same evaluation in a CHAT event cannot read R15: UNAVAILABLE, a row 1.
        def handle_chat() -> str:
            chat_event = runtime.current_event()
            assert chat_event is not None
            seen["chat"] = issuer._evaluate(
                chat_event,
                goal_id,
                now=chat_event.requested_at,
                before_sequence=chat_event.processing_sequence,  # type: ignore[arg-type]
            )
            return "chat"

        runtime.submit(
            AgentEventType.CHAT, AgentEventSource.API_CHAT, handle_chat
        ).result(timeout=30)
        assert seen["chat"].value.interpretation is Interpretation.UNAVAILABLE


# ------------------------------------------------------------------- §8


def test_with_r19_disabled_nothing_r19_starts_and_the_subject_runs(
    tmp_path: Path,
) -> None:
    """D1 §8: the default (``r19.enabled`` false) starts no R19 thread and
    submits no R19 event; IP-20a adds neither a Scheduler nor an owner, so the
    PLAN_PREVIEW path is the only one that runs."""

    import threading

    with TestClient(_app(_settings(tmp_path / "off"))) as client:
        state = client.app.state
        assert state.settings.r19.enabled is False
        _x, goal_id = _qualified_goal(state)
        assert (
            state._plan_preview_owner._submit_preview(goal_id)
            .result(timeout=30)
            .value.is_authoritative
        )
        assert [
            item.name
            for item in threading.enumerate()
            if item.name.startswith("suzka-r19")
        ] == []
        assert getattr(state, "wake_scheduler", None) is None
        assert not [
            item.event_type
            for item in state.event_journal.records
            if item.event_type in WAKE_TYPES
        ]
