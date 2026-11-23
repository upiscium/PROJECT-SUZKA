"""IP-17a: the R18 Policy owner, the approval window, row 2c and the
pre-migration replay check (#411 D1 r7 §1, §2, §3a, §3b, §8a, §12-1..§12-3).

Challenges and expiries are IP-17b's: where a test needs them they are
FIXTURES committed through the real R07 runtime and the subject.history
record participant (as in the IP-16b ledger tests).
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import product
import os
import sqlite3
import time
from types import SimpleNamespace
from typing import Any, Final

from fastapi.testclient import TestClient
import pytest

from suzka.r16.decision_issuer import DecisionIssueStatus
from suzka.r16.history_codec import PolicyKind, PolicyRow
from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import (
    MAX_CHALLENGES_PER_INTENT,
    HistoryOperationKind,
    RecordKind,
)
from suzka.r16.history_store import HistoryPhase, HistoryStoreCapacityError
from suzka.r16.replay_check import main as replay_main, replay_rows, replay_store
from suzka.r16.successor_policy import (
    CLOSED_FORWARD_LEDGER,
    CommitmentState,
    FeedbackState,
    Interpretation,
    LedgerState,
    PolicyOutputKind,
    PredecessorState,
    RequiredSourceState,
    LineageState,
    SuccessorRow,
    evaluate,
)
from suzka.r17.contracts import ExpectedObservationV1
from suzka.r18 import policy_owner as policy_module
from suzka.r18.admission import ALWAYS_OPEN_UNTIL_IP18, gate_open
from suzka.r18.contracts import (
    ActionIntentV1,
    Direction,
    PolicyDenyReason,
    PolicyVerdict,
    PolicyVerdictV1,
)
from suzka.r18.intent_issuer import (
    IntentIssueStatus,
    R18IntentReason,
)
from suzka.r18.ledger import INTENT_REF_KIND, VERDICT_REF_KIND, derive_ledger
from suzka.r18.policy_owner import (
    MAX_POLICY_SCANS_PER_STARTUP,
    PolicyCommitStatus,
    R18PolicyReason,
    ReviewArtifactPolicyV1,
    _PolicyResult,
)
from suzka.r18.target import TargetStore
from suzka.runtime.agent_runtime import (
    AgentEventOutcome,
    AgentEventType,
    AgentRuntimeDurabilityError,
)
from suzka.runtime.event_journal import (
    SOFT_BAND_ADMITTED_EVENT_TYPES,
    EventJournalAdmissionBand,
    EventJournalAdmissionStatus,
    EventLifecycle,
)

import test_r16_ip10_decision_issuer as ip10
import test_r16_ip16b_ledger_reader as ip16b
import test_r17_ip16a3_expectation_issuer as ip16a3
import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_ip8b_successor_policy import _consistent, _ledger_states
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_commitment

_settings = ip16a2._settings
_row4 = ip16a2._row4
_intent = ip16a2._intent
_no_failed_events = ip16a2._no_failed_events
_forward = ip16b._forward
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
V2 = PolicyKind.SUCCESSOR_POLICY_V2
V3 = PolicyKind.SUCCESSOR_POLICY_V3
_R18_EVENTS = frozenset(
    {
        AgentEventType.R18_ACTION_INTENT_ISSUE,
        AgentEventType.R17_EXPECTATION_COMMIT,
        AgentEventType.R18_POLICY_COMMIT,
    }
)


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "ip17a"))


def _policy(state, intent_id):  # type: ignore[no-untyped-def]
    return state.policy_owner._submit_policy(intent_id).result(timeout=30).value


def _scan(state, after=None):  # type: ignore[no-untyped-def]
    return state.policy_owner._submit_scan(after).result(timeout=30).value


def _verdict_rows(state):  # type: ignore[no-untyped-def]
    return [
        row
        for row in state.subject_history_store.operations(ip10._incarnation(state))
        if row.operation_kind is HistoryOperationKind.R18_POLICY_COMMIT
    ]


def _decision_rows(state):  # type: ignore[no-untyped-def]
    return [
        row
        for row in state.subject_history_store.operations(ip10._incarnation(state))
        if row.operation_kind is HistoryOperationKind.R16_DECISION_ISSUE
        and row.phase is HistoryPhase.FINALIZED
    ]


def _r18_events(state) -> int:  # type: ignore[no-untyped-def]
    """R17 / R18 events admitted so far (one ACCEPTED record each)."""

    return sum(
        1
        for row in state.event_journal.records
        if row.event_type in _R18_EVENTS and row.lifecycle is EventLifecycle.ACCEPTED
    )


def _adopted(state):  # type: ignore[no-untyped-def]
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return found.records


def _after_window(intent: ActionIntentV1) -> datetime:
    return intent.approval_window_end


def _closed_preview(state, gid, intent, monkeypatch, seen=None):  # type: ignore[no-untyped-def]
    """A PLAN_PREVIEW at the approval window's end (the issuer's clock)."""

    with monkeypatch.context() as patch:
        patch.setattr(state.decision_issuer, "_clock", lambda: _after_window(intent))
        return ip16b._preview(state, gid, seen)


def _assert_row_2c(result) -> None:  # type: ignore[no-untyped-def]
    assert (result.status, result.row) == (DecisionIssueStatus.ISSUED, SuccessorRow.ROW_2C)
    committed = result.decision
    assert committed.policy_kind is V3
    assert committed.policy_row is PolicyRow.ROW_2C
    assert committed.decision.selected.candidate_id == "candidate-a"


# ======================================================== row 2c (pure)


# The SuccessorPolicy tables before IP-17a (64e5bbe), over every state tuple
# of the ip8b builder: V2 in full, V3 without the closed forward ledger states (IP-17a's two and IP-18b's four).
_PRE_IP17A_V2: Final = "5098da70263a8f3bfcbb9ec259caaf06d9110a6dc9c1cd006703afa790579c7c"
_PRE_IP17A_V3_UNCLOSED: Final = "7fb412985a95a103129ab94c1b8d1d12a7d56dd0306a4c328cc31bda830ab106"


def _table(version: PolicyKind, *, skip_closed: bool) -> tuple[str, dict[SuccessorRow, int]]:
    hashed = sha256()
    rows: dict[SuccessorRow, int] = {}
    for states in product(
        RequiredSourceState,
        PredecessorState,
        LineageState,
        CommitmentState,
        _ledger_states(version),
        FeedbackState,
        Interpretation,
    ):
        closed = states[4] in CLOSED_FORWARD_LEDGER
        if skip_closed and closed:
            continue
        row = evaluate(_consistent(*states, version=version)).row
        if closed:
            rows[row] = rows.get(row, 0) + 1
        hashed.update(("|".join(s.value for s in states) + "=" + row.value + "\n").encode())
    return hashed.hexdigest(), rows


def test_v2_and_every_unclosed_v3_tuple_evaluate_exactly_as_before() -> None:
    """§12-2 replay safety: V2 is unchanged, and V3 changes only for the two
    closed ledger states (every persisted Decision has ledger NONE)."""

    v2, _ = _table(V2, skip_closed=False)
    assert v2 == _PRE_IP17A_V2
    v3, _ = _table(V3, skip_closed=True)
    assert v3 == _PRE_IP17A_V3_UNCLOSED
    _, closed = _table(V3, skip_closed=False)
    # A closed ledger is never a pending row 2 nor Plan B / NO_OP under V3.
    assert set(closed) <= {
        SuccessorRow.ROW_0,
        SuccessorRow.ROW_1,
        SuccessorRow.ROW_2B,
        SuccessorRow.ROW_2C,
        SuccessorRow.ROW_7,
    }
    assert closed[SuccessorRow.ROW_2C] > 0


@pytest.mark.parametrize("ledger", sorted(CLOSED_FORWARD_LEDGER))
def test_row_2c_is_v3_only_after_rows_2_and_2b(ledger: LedgerState) -> None:
    def value(version, **changes):  # type: ignore[no-untyped-def]
        states = dict(
            required=RequiredSourceState.QUALIFIED,
            predecessor=PredecessorState.QUALIFIED,
            lineage=LineageState.CONSISTENT,
            commitment=CommitmentState.ACTIVE,
            ledger=ledger,
            feedback=FeedbackState.ABSENT,
            interpretation=Interpretation.SETTLED,
        )
        states.update(changes)
        return _consistent(*states.values(), version=version)

    result = evaluate(value(V3))
    assert (result.row, result.output, result.write_eligible) == (
        SuccessorRow.ROW_2C,
        PolicyOutputKind.PLAN_A,
        False,
    )
    assert result.history_row is PolicyRow.ROW_2C
    # V2 + closed -> 2; CONTRADICTORY + closed -> 2b (2c comes after 2b).
    if ledger in _ledger_states(V2):
        assert evaluate(value(V2)).row is SuccessorRow.ROW_2
    assert evaluate(value(V3, interpretation=Interpretation.CONTRADICTORY)).row is SuccessorRow.ROW_2B
    # UNCERTAIN and no Commitment still re-decide to Plan A.
    assert evaluate(value(V3, interpretation=Interpretation.UNCERTAIN)).row is SuccessorRow.ROW_2C
    assert evaluate(value(V3, commitment=CommitmentState.ABSENT)).row is SuccessorRow.ROW_2C
    # Row 0's ABSENT_INITIAL rule exempts the closed states under V3 only.
    assert evaluate(value(V3, predecessor=PredecessorState.ABSENT_INITIAL)).row is SuccessorRow.ROW_2C
    if ledger in _ledger_states(V2):
        assert evaluate(value(V2, predecessor=PredecessorState.ABSENT_INITIAL)).row is SuccessorRow.ROW_0
    pending = value(V3, ledger=LedgerState.INTENT_ISSUED, predecessor=PredecessorState.ABSENT_INITIAL)
    assert evaluate(pending).row is SuccessorRow.ROW_0
    # Earlier rows still win.
    assert evaluate(value(V3, lineage=LineageState.UNAVAILABLE)).row is SuccessorRow.ROW_1
    assert evaluate(value(V3, commitment=CommitmentState.EXPIRED)).row is SuccessorRow.ROW_1
    assert evaluate(value(V3, required=RequiredSourceState.CONFLICT)).row is SuccessorRow.ROW_0


def test_row_2c_is_a_v3_only_decision_row() -> None:
    from test_r16_ip9_subject_history import _committed, _event

    _committed(_event("row-2c", 3), row=PolicyRow.ROW_2C, kind=V3)
    for kind in (PolicyKind.SUCCESSOR_POLICY_V1, V2):
        with pytest.raises(ValueError, match="2c"):
            _committed(_event("row-2c", 3), row=PolicyRow.ROW_2C, kind=kind)


# =================================================== the window (pure)


def test_the_window_end_is_bound_into_the_intent() -> None:
    from test_r18_ip13_contracts import _intent as fixture_intent

    intent = fixture_intent()
    later = replace(intent, approval_window_end_utc="2026-11-16T02:00:00Z")
    assert later.intent_id != intent.intent_id and later.intent_digest != intent.intent_digest
    assert ActionIntentV1.from_json(intent.canonical_bytes()) == intent
    assert b'"approval_window_end_utc":"2026-11-16T01:00:00Z"' in intent.canonical_bytes()
    for bad in ("2026-11-16T01:00:00", "2026-11-16 01:00:00Z", "2026-13-16T01:00:00Z"):
        with pytest.raises(ValueError):
            replace(intent, approval_window_end_utc=bad)
    end = intent.approval_window_end
    assert intent.window_open(end - timedelta(microseconds=1))
    assert not intent.window_open(end)
    with pytest.raises(ValueError):
        intent.window_open(end.replace(tzinfo=None))


def test_the_ledger_closes_an_unclaimed_intent_at_the_window_end() -> None:
    """§3a (HD-13 (a)): derived from the records and the clock, no event."""

    chain = ip16b._fixture_chain()
    group = chain.intent.obligation_group_key
    end = chain.intent.approval_window_end
    before, after = end - timedelta(seconds=1), end
    at = ip16b._at
    base = [chain.intent, chain.forecast]

    def state(records, now):  # type: ignore[no-untyped-def]
        return derive_ledger(at(*records), group, now).state

    assert state(base, before) is LedgerState.INTENT_ISSUED
    assert state(base, after) is LedgerState.APPROVAL_EXPIRED_UNCLAIMED
    assert state([*base, chain.verdict], after) is LedgerState.APPROVAL_EXPIRED_UNCLAIMED
    assert state([*base, chain.verdict, chain.challenges[0]], after) is (
        LedgerState.APPROVAL_EXPIRED_UNCLAIMED
    )
    denied = replace(chain.verdict, verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.SOURCE_NOT_CURRENT)
    assert state([*base, denied], before) is LedgerState.POLICY_DENIED
    assert state([*base, denied], after) is LedgerState.POLICY_DENIED
    # A Claim made inside the window is never closed by the clock.
    claimed = [*base, chain.verdict, chain.challenges[0], chain.claim]
    assert state(claimed, after + timedelta(days=30)) is LedgerState.CLAIMED
    # The refs are the same records the open Intent had (no closure record).
    read = derive_ledger(at(*base, chain.verdict), group, after)
    assert [ref.kind for ref in read.refs] == sorted([INTENT_REF_KIND, VERDICT_REF_KIND])
    assert ip16b._policy(read) is SuccessorRow.ROW_2C
    assert ip16b._policy(read, PredecessorState.ABSENT_INITIAL) is SuccessorRow.ROW_2C
    with pytest.raises(ValueError):
        derive_ledger(at(*base), group, after.replace(tzinfo=None))


# ===================================================== the Policy owner


def test_a_forecasted_intent_gets_one_allow_then_existing(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        assert type(state.policy_owner) is ReviewArtifactPolicyV1
        assert state.r16_driver._policy is state.policy_owner
        gid, _decision, intent, forecast = _forward(state)
        before = len(ip10._history_transactions(state))
        allowed = _policy(state, intent.intent_id)
        assert type(allowed) is _PolicyResult
        assert (allowed.status, allowed.intent_id, allowed.reason) == (
            PolicyCommitStatus.ALLOWED,
            intent.intent_id,
            None,
        )
        verdict = allowed.verdict
        assert type(verdict) is PolicyVerdictV1
        assert (verdict.direction, verdict.verdict, verdict.deny_reason) == (
            Direction.FORWARD,
            PolicyVerdict.ALLOW,
            None,
        )
        assert (verdict.intent_id, verdict.intent_digest, verdict.operation_key) == (
            intent.intent_id,
            intent.intent_digest,
            intent.forward_key,
        )
        assert len(ip10._history_transactions(state)) == before + 1
        assert [row.phase for row in _verdict_rows(state)] == [HistoryPhase.FINALIZED]
        assert [(r.record_kind, r.record) for r in _adopted(state)] == [
            (RecordKind.ACTION_INTENT, intent),
            (RecordKind.ACTION_FORECAST, forecast),
            (RecordKind.POLICY_VERDICT, verdict),
        ]
        assert state.policy_owner._pending is None
        again = _policy(state, intent.intent_id)
        assert (again.status, again.verdict) == (PolicyCommitStatus.EXISTING, verdict)
        assert _scan(state).status is PolicyCommitStatus.NONE_LEFT
        assert len(ip10._history_transactions(state)) == before + 1
        # Inside the window an ALLOW is still pending: row 2, nothing issued.
        held = ip16b._preview(state, gid)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2)
        _no_failed_events(state)


def test_the_intent_carries_a_bounded_window(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§3a / §12-1: ``min(now + window, deadline - margin)``, to the second."""

    with TestClient(app) as client:
        state = client.app.state
        _x, _gid, decision = _row4(state)
        now = datetime(2026, 11, 18, 3, 4, 5, 678_000, tzinfo=timezone.utc)
        monkeypatch.setattr(state.intent_issuer, "_clock", lambda: now)
        issued = _intent(state, decision.lineage_scope)
        assert issued.status is IntentIssueStatus.ISSUED
        expected = (now + timedelta(seconds=3600)).replace(microsecond=0)
        commitment = state.main_loop._commitment_system.get(
            [ref for ref in decision.decision.source_refs if ref.kind == "commitment"][0].ref_id
        )
        if commitment.deadline.at is not None:
            expected = min(expected, (commitment.deadline.at - ip16a2.decision_module.DEADLINE_MARGIN).replace(microsecond=0))
        assert issued.intent.approval_window_end_utc == expected.strftime(UTC_FORMAT)
        assert state.settings.r18.approval_window_seconds == 3600


def test_the_window_setting_is_bounded() -> None:
    from pydantic import ValidationError

    from suzka.config.schema import R18Settings

    assert R18Settings().approval_window_seconds == 3600
    for value in (599, 86_401, "3600", 3600.0):
        with pytest.raises(ValidationError):
            R18Settings(approval_window_seconds=value)  # type: ignore[arg-type]
    assert R18Settings(approval_window_seconds=600).approval_window_seconds == 600


def test_scan_walks_after_the_cursor_and_skips_closed_windows(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        assert _scan(state).status is PolicyCommitStatus.NONE_LEFT
        _gid, _decision, intent, _forecast = _forward(state)
        # After the Intent itself: nothing left.
        assert _scan(state, intent.intent_id).status is PolicyCommitStatus.NONE_LEFT
        # A cursor that is not an adopted Intent ends the chain (no id).
        stray = _scan(state, "intent-" + "0" * 64)
        assert (stray.status, stray.intent_id, stray.reason) == (
            PolicyCommitStatus.HOLD,
            None,
            R18PolicyReason.INTENT_UNKNOWN,
        )
        # A closed window is skipped by the scan; an explicit request is CLOSED.
        with monkeypatch.context() as patch:
            patch.setattr(state.policy_owner, "_clock", lambda: _after_window(intent))
            assert _scan(state).status is PolicyCommitStatus.NONE_LEFT
            closed = _policy(state, intent.intent_id)
            assert (closed.status, closed.intent_id) == (PolicyCommitStatus.CLOSED, intent.intent_id)
        assert _verdict_rows(state) == []
        scanned = _scan(state)
        assert (scanned.status, scanned.intent_id) == (PolicyCommitStatus.ALLOWED, intent.intent_id)
        assert _scan(state).status is PolicyCommitStatus.NONE_LEFT
        _no_failed_events(state)


def test_holds_write_nothing(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        owner = state.policy_owner
        assert _policy(state, "intent-" + "0" * 64).reason is R18PolicyReason.INTENT_UNKNOWN
        inverse = _policy(state, "inverse-intent-" + "0" * 64)
        assert (inverse.status, inverse.reason) == (
            PolicyCommitStatus.HOLD,
            R18PolicyReason.INVERSE_DEFERRED,
        )
        _x, _gid, decision = _row4(state)
        intent = _intent(state, decision.lineage_scope).intent
        # No forecast yet (FORECAST_MISSING, transient).
        held = _policy(state, intent.intent_id)
        assert (held.reason, held.intent_id) == (R18PolicyReason.FORECAST_MISSING, intent.intent_id)
        assert ip16a3._forecast(state, intent.intent_id).forecast is not None
        # The Intent row exists but is not adopted yet.
        real = SubjectHistoryReader._records

        def without(kind):  # type: ignore[no-untyped-def]
            def records(self, *args):  # type: ignore[no-untyped-def]
                return tuple(item for item in real(self, *args) if item.record_kind is not kind)

            return records

        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "_records", without(RecordKind.ACTION_INTENT))
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.NOT_YET_TERMINAL
        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "_records", without(RecordKind.ACTION_FORECAST))
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.FORECAST_MISSING
        with monkeypatch.context() as patch:
            patch.setattr(
                ExpectedObservationV1,
                "check_binding",
                lambda self, **kwargs: (_ for _ in ()).throw(ValueError("other")),
            )
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.BINDING_MISMATCH
        # §2-7: a closed AdmissionGate.
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_gate", SimpleNamespace(is_open=lambda: False))
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.ADMISSION_GATE_CLOSED
        # §3b: the ``verdict:`` key reserved but not adopted.
        store = state.subject_history_store
        real_keys = store.live_keys
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "live_keys",
                lambda keys: frozenset(k for k in keys if k.startswith("verdict:")) | real_keys(keys),
            )
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.IN_FLIGHT

        def full(length):  # type: ignore[no-untyped-def]
            raise HistoryStoreCapacityError("full")

        with monkeypatch.context() as patch:
            patch.setattr(store, "check_headroom", full)
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.HISTORY_CAPACITY
        # The target cannot be opened: HOLD, never created.
        path = state.settings.r18.target_path
        moved = path.with_name("moved.sqlite3")
        path.rename(moved)
        try:
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.TARGET_UNAVAILABLE
            assert not path.exists()
        finally:
            moved.rename(path)
        # A Context transition: the sources are unavailable (HOLD, not DENY).
        with monkeypatch.context() as patch:
            from suzka.r18 import requalify as requalify_module
            from suzka.r18.requalify import RequalifyFailed, RequalifyReason

            def unavailable(*args, **kwargs):  # type: ignore[no-untyped-def]
                raise RequalifyFailed(RequalifyReason.SOURCE_UNAVAILABLE)

            patch.setattr(policy_module, "requalify", unavailable)
            assert _policy(state, intent.intent_id).reason is R18PolicyReason.SOURCE_UNAVAILABLE
            assert requalify_module.requalify is not unavailable
        assert _verdict_rows(state) == []
        assert owner._pending is None
        assert _policy(state, intent.intent_id).status is PolicyCommitStatus.ALLOWED
        _no_failed_events(state)


def test_the_intent_issuer_asks_the_admission_gate(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        assert state.intent_issuer._gate is state.r18_gate  # IP-18a: the SweepGate
        assert state.policy_owner._gate is state.r18_gate
        assert gate_open(ALWAYS_OPEN_UNTIL_IP18)

        def broken() -> bool:
            raise RuntimeError("no gate")

        assert not gate_open(SimpleNamespace(is_open=broken))  # type: ignore[arg-type]
        assert not gate_open(SimpleNamespace(is_open=lambda: 1))  # type: ignore[arg-type]
        _x, _gid, decision = _row4(state)
        monkeypatch.setattr(state.intent_issuer, "_gate", SimpleNamespace(is_open=lambda: False))
        held = _intent(state, decision.lineage_scope)
        assert held.reason is R18IntentReason.ADMISSION_GATE_CLOSED
        assert ip16a2._intent_rows(state) == []


def test_source_drift_is_deny_then_row_2c_then_a_new_request_is_2c(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§2-5 DENY SOURCE_NOT_CURRENT; §8a DENY -> 2c; a second identical
    request on the closed group is 2c, not row 0 (§12-2)."""

    from suzka.r16.review_value import REVIEW_VALUE_ID

    with TestClient(app) as client:
        state = client.app.state
        seen = ip16b._Seen(monkeypatch)
        gid, _decision, intent, _forecast = _forward(state)
        for action in ("freeze", "unfreeze"):
            response = client.post(f"/api/values/{REVIEW_VALUE_ID}/{action}")
            assert response.status_code == 200, response.text
        denied = _policy(state, intent.intent_id)
        assert (denied.status, denied.verdict.verdict, denied.verdict.deny_reason) == (
            PolicyCommitStatus.DENIED,
            PolicyVerdict.DENY,
            PolicyDenyReason.SOURCE_NOT_CURRENT,
        )
        assert _policy(state, intent.intent_id).status is PolicyCommitStatus.EXISTING
        events = _r18_events(state)
        result = ip16b._preview(state, gid, seen)
        _assert_row_2c(result)
        assert seen.last.ledger is LedgerState.POLICY_DENIED
        assert ip16b._preview(state, gid).status is DecisionIssueStatus.EXISTING
        # The Intent issuer refuses the Plan A head.
        assert _intent(state, intent.lineage_scope).reason is R18IntentReason.NOT_ROW_4
        # A new X with the same text: the same (closed) group.
        second = ip16a2._second_goal(state)
        fresh = ip16b._preview(state, second, seen)
        _assert_row_2c(fresh)
        assert seen.last.predecessor is PredecessorState.ABSENT_INITIAL
        assert seen.last.ledger is LedgerState.POLICY_DENIED
        _qualified_commitment(state, state._ip16_second_x)
        _assert_row_2c(ip16b._preview(state, second, seen))
        assert _intent(state, fresh.lineage_scope).reason is R18IntentReason.NOT_ROW_4
        assert _r18_events(state) == events + 2  # the two refused Intent requests
        _no_failed_events(state)


def test_a_target_row_for_the_group_is_target_mismatch(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, _decision, intent, _forecast = _forward(state)
        target = TargetStore.open(state.settings.r18.target_path)
        try:
            target.fence(intent.obligation_group_key, Direction.FORWARD)
        finally:
            target.close()
        denied = _policy(state, intent.intent_id)
        assert (denied.status, denied.verdict.deny_reason) == (
            PolicyCommitStatus.DENIED,
            PolicyDenyReason.TARGET_MISMATCH,
        )
        _no_failed_events(state)


def test_a_superseded_decision_is_denied_even_with_the_clock_set_back(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§2-4 / §3a: once the subject re-decided after closure (row 2c), the
    Intent's Decision is superseded for good."""

    with TestClient(app) as client:
        state = client.app.state
        gid, decision, intent, _forecast = _forward(state)
        _assert_row_2c(_closed_preview(state, gid, intent, monkeypatch))
        # The Policy owner's clock is back inside the window.
        assert intent.window_open(state.policy_owner._clock())
        denied = _policy(state, intent.intent_id)
        assert (denied.status, denied.verdict.deny_reason) == (
            PolicyCommitStatus.DENIED,
            PolicyDenyReason.SOURCE_NOT_CURRENT,
        )
        _no_failed_events(state)


def test_validate_commit_sees_exactly_its_own_prepared_row(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        _gid, _decision, intent, _forecast = _forward(state)
        owner = state.policy_owner
        seen: list[str] = []
        real = owner._validate_commit

        def spy(event):  # type: ignore[no-untyped-def]
            if owner._pending is not None:
                seen.extend(row.phase.value for row in _verdict_rows(state))
            return real(event)

        owner._validate_commit = spy  # type: ignore[method-assign]
        try:
            assert _policy(state, intent.intent_id).status is PolicyCommitStatus.ALLOWED
        finally:
            owner._validate_commit = real  # type: ignore[method-assign]
        assert seen == [HistoryPhase.PREPARED.value]


def test_crash_after_prepare_is_tombstoned_then_recommitted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "crash-prepare")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _gid, _decision, intent, _forecast = _forward(state)

        def crash(event):  # type: ignore[no-untyped-def]
            if event.event_type is AgentEventType.R18_POLICY_COMMIT:
                raise RuntimeError("simulated pre-WAL failure")

        state.policy_owner._validate_commit = crash  # type: ignore[method-assign]
        with pytest.raises(AgentRuntimeDurabilityError):
            _policy(state, intent.intent_id)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _verdict_rows(state)][:1] == [HistoryPhase.ABORTED]
        # The restarted driver's scan may already have committed it.
        result = _policy(state, intent.intent_id)
        assert result.status in (PolicyCommitStatus.ALLOWED, PolicyCommitStatus.EXISTING)
        assert [row.phase for row in _verdict_rows(state)] == [
            HistoryPhase.ABORTED,
            HistoryPhase.FINALIZED,
        ]


def test_disabled_r18_holds_without_reading_and_the_driver_gets_no_owner(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", r18=False))) as client:
        state = client.app.state
        owner = state.policy_owner
        assert type(owner) is ReviewArtifactPolicyV1 and not owner.enabled
        assert state.r16_driver._policy is None

        def no_read(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("read while r18 is disabled")

        with monkeypatch.context() as patch:
            patch.setattr(SubjectHistoryReader, "classify", no_read)
            patch.setattr(state._plan_preview_owner, "_state_snapshot", no_read)
            for held in (_policy(state, "intent-" + "1" * 64), _scan(state)):
                assert (held.status, held.reason) == (
                    PolicyCommitStatus.HOLD,
                    R18PolicyReason.R18_DISABLED,
                )
        assert _verdict_rows(state) == []
        assert state.subject_history_store.has_record_operations() is False
        _no_failed_events(state)


def test_no_owner_without_subject_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "nohistory")
    settings = settings.model_copy(
        update={"subject_history": settings.subject_history.model_copy(update={"enabled": False})}
    )
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.policy_owner is None
        assert state.r16_driver._policy is None


def test_the_soft_band_refuses_policy_and_admits_the_preview() -> None:
    assert AgentEventType.R18_POLICY_COMMIT not in SOFT_BAND_ADMITTED_EVENT_TYPES
    soft = EventJournalAdmissionStatus(
        True, "journal_soft_limit", 0, 1, 0, 1, True, False, False,
        band=EventJournalAdmissionBand.SOFT,
    )
    assert not soft.admits(AgentEventType.R18_POLICY_COMMIT)
    # §8a: re-decision needs only PLAN_PREVIEW, which the soft band admits.
    assert soft.admits(AgentEventType.PLAN_PREVIEW_REVIEW)


# ============================================ §8a: no stop hinders the subject


def test_never_reviewed_closes_at_the_window_end_to_row_2c(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        seen = ip16b._Seen(monkeypatch)
        gid, _decision, intent, _forecast = _forward(state)
        assert _policy(state, intent.intent_id).status is PolicyCommitStatus.ALLOWED
        events = _r18_events(state)
        # One second before the end the obligation is still pending.
        with monkeypatch.context() as patch:
            patch.setattr(
                state.decision_issuer, "_clock", lambda: _after_window(intent) - timedelta(seconds=1)
            )
            assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
        _assert_row_2c(_closed_preview(state, gid, intent, monkeypatch, seen))
        assert seen.last.ledger is LedgerState.APPROVAL_EXPIRED_UNCLAIMED
        assert _r18_events(state) == events  # no event closed it
        _no_failed_events(state)


def test_three_expiries_close_to_row_2c_inside_the_window(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app) as client:
        state = client.app.state
        seen = ip16b._Seen(monkeypatch)
        gid, _decision, intent, forecast = _forward(state)
        verdict = _policy(state, intent.intent_id).verdict
        for attempt in range(1, MAX_CHALLENGES_PER_INTENT + 1):
            challenge = ip16b._challenge(state, intent, verdict, attempt, forecast)
            assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
            ip16b._expire(state, challenge)
        _assert_row_2c(ip16b._preview(state, gid, seen))
        assert seen.last.ledger is LedgerState.APPROVAL_EXPIRED_UNCLAIMED
        _no_failed_events(state)


def test_drift_after_allow_closes_at_the_window_end(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from suzka.r16.review_value import REVIEW_VALUE_ID

    with TestClient(app) as client:
        state = client.app.state
        gid, _decision, intent, _forecast = _forward(state)
        assert _policy(state, intent.intent_id).status is PolicyCommitStatus.ALLOWED
        for action in ("freeze", "unfreeze"):
            assert client.post(f"/api/values/{REVIEW_VALUE_ID}/{action}").status_code == 200
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
        _assert_row_2c(_closed_preview(state, gid, intent, monkeypatch))


def test_r18_off_still_closes_to_row_2c(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """§8 / §8a: with R18 switched off (records present) nothing submits an
    R18 event, but the ledger reader still derives closure."""

    settings = _settings(tmp_path / "switch-off")
    off = settings.model_copy(
        update={"r18": settings.r18.model_copy(update={"enabled": False})}
    )
    with TestClient(_app(settings)) as client:
        state = client.app.state
        gid, _decision, intent, _forecast = _forward(state)
        assert _policy(state, intent.intent_id).status is PolicyCommitStatus.ALLOWED
    with TestClient(_app(off)) as client:
        state = client.app.state
        assert not state.policy_owner.enabled and state.r16_driver._policy is None
        events = _r18_events(state)
        assert ip16b._preview(state, gid).row is SuccessorRow.ROW_2
        _assert_row_2c(_closed_preview(state, gid, intent, monkeypatch))
        assert _r18_events(state) == events
        _no_failed_events(state)


# ================================================================ the driver


class _FakePolicy(ip16a2._FakeOwner):
    def _submit_policy(self, intent_id: str) -> Future[Any]:
        return self._record("policy", intent_id)

    def _submit_scan(self, after: str | None) -> Future[Any]:
        return self._record("scan", after)


def _driver():  # type: ignore[no-untyped-def]
    from suzka.r16.driver import R16Driver

    expectation = ip16a3._FakeExpectation()
    policy = _FakePolicy()
    clock = ip16a2._Clock()
    driver = R16Driver(
        ip16a2._FakeRuntime(),  # type: ignore[arg-type]
        *[ip16a2._FakeOwner() for _ in range(3)],  # type: ignore[arg-type]
        ip16a2._FakeOwner(),  # type: ignore[arg-type]
        active=True,
        clock=clock,
        intent_issuer=ip16a2._FakeOwner(),  # type: ignore[arg-type]
        expectation_issuer=expectation,  # type: ignore[arg-type]
        policy_owner=policy,  # type: ignore[arg-type]
    )
    return driver, expectation, policy, clock


def _verdict(intent_id: str, verdict=PolicyVerdict.ALLOW):  # type: ignore[no-untyped-def]
    return SimpleNamespace(intent_id=intent_id, verdict=verdict)


def _result(status, intent_id="intent-a", reason=None):  # type: ignore[no-untyped-def]
    if status in (PolicyCommitStatus.ALLOWED, PolicyCommitStatus.EXISTING):
        return _PolicyResult(status, intent_id, None, _verdict(intent_id))  # type: ignore[arg-type]
    if status is PolicyCommitStatus.DENIED:
        return _PolicyResult(status, intent_id, None, _verdict(intent_id, PolicyVerdict.DENY))  # type: ignore[arg-type]
    return _PolicyResult(status, intent_id, reason)


def _resolve(future: Future[Any], value: object) -> None:
    future.set_result(AgentEventOutcome(ip16a2._event(), value))


@pytest.mark.parametrize("status", ("forecast_committed", "forecast_existing"))
def test_driver_submits_one_policy_after_any_forecast_result(status) -> None:  # type: ignore[no-untyped-def]
    from suzka.r17.expectation_issuer import ExpectationCommitStatus

    driver, expectation, policy, clock = _driver()
    driver._maybe_forecast("intent-a")
    _resolve(expectation.futures[-1], ip16a3._forecast_result(ExpectationCommitStatus(status)))
    ip16a2._drain(driver)
    assert policy.calls == [("policy", "intent-a")]
    # Transient HOLDs are retried with the bounded delays (3 times).
    for attempt in range(4):
        _resolve(
            policy.futures[-1],
            _result(PolicyCommitStatus.HOLD, reason=R18PolicyReason.FORECAST_MISSING),
        )
        ip16a2._drain(driver)
        clock.now += 10
        ip16a2._drain(driver)
        assert len(policy.calls) == min(attempt + 2, 4)
    # Once per Intent this incarnation.
    driver._maybe_policy("intent-a")
    assert len(policy.calls) == 4


def test_driver_does_not_retry_other_policy_holds_or_after_a_forecast_hold() -> None:
    from suzka.r17.expectation_issuer import ExpectationCommitStatus, R17ExpectationReason

    driver, expectation, policy, clock = _driver()
    driver._maybe_forecast("intent-a")
    _resolve(
        expectation.futures[-1],
        ip16a3._forecast_result(
            ExpectationCommitStatus.HOLD, reason=R17ExpectationReason.BINDING_MISMATCH
        ),
    )
    ip16a2._drain(driver)
    assert policy.calls == []
    driver._maybe_policy("intent-b")
    _resolve(policy.futures[-1], _result(PolicyCommitStatus.HOLD, "intent-b", R18PolicyReason.TARGET_UNAVAILABLE))
    ip16a2._drain(driver)
    clock.now += 10
    ip16a2._drain(driver)
    assert policy.calls == [("policy", "intent-b")]


def test_driver_scan_chains_on_the_cursor_and_is_bounded() -> None:
    driver, _expectation, policy, _clock = _driver()
    driver._maybe_policy_scan(None)
    n = 0
    while policy.futures and not policy.futures[-1].done():
        n += 1
        # Alternate a written verdict and a HOLD on one Intent (skipped).
        value = (
            _result(PolicyCommitStatus.ALLOWED, f"intent-{n}")
            if n % 2
            else _result(PolicyCommitStatus.HOLD, f"intent-{n}", R18PolicyReason.SOURCE_UNAVAILABLE)
        )
        _resolve(policy.futures[-1], value)
        ip16a2._drain(driver)
    assert len(policy.calls) == MAX_POLICY_SCANS_PER_STARTUP == 16
    assert policy.calls == [("scan", None), *(("scan", f"intent-{i}") for i in range(1, 16))]
    for final in (
        _result(PolicyCommitStatus.NONE_LEFT, None),
        _result(PolicyCommitStatus.HOLD, "intent-x", R18PolicyReason.R18_DISABLED),
        _result(PolicyCommitStatus.HOLD, None, R18PolicyReason.HISTORY_UNAVAILABLE),
    ):
        driver, _expectation, policy, _clock = _driver()
        driver._maybe_policy_scan(None)
        _resolve(policy.futures[-1], _result(PolicyCommitStatus.DENIED, "intent-1"))
        ip16a2._drain(driver)
        _resolve(policy.futures[-1], final)
        ip16a2._drain(driver)
        assert policy.calls == [("scan", None), ("scan", "intent-1")]


def test_driver_posts_the_policy_scan_once_and_only_when_enabled() -> None:
    driver, _expectation, _policy_fake, _clock = _driver()
    driver._post_policy_scan()
    driver._post_policy_scan()
    assert driver._queue.qsize() == 1
    disabled, _expectation, policy, _clock = _driver()
    policy.enabled = False
    disabled._post_policy_scan()
    assert disabled._queue.qsize() == 0


def test_crash_between_forecast_and_policy_is_completed_by_the_restart_scan(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "restart-scan")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        state.r16_driver._policy = None
        _gid, _decision, intent, _forecast = _forward(state)
        assert client.post("/api/r16/driver/activate").status_code == 200
        time.sleep(0.5)
        assert _verdict_rows(state) == []
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.r16_driver.active
        ip16a2._until(lambda: len(_verdict_rows(state)) == 1, timeout=30)

        def adopted_verdicts() -> list[Any]:
            return [r.record for r in _adopted(state) if r.record_kind is RecordKind.POLICY_VERDICT]

        # Review P3-1: poll until the verdict's event is terminal (adopted).
        ip16a2._until(lambda: len(adopted_verdicts()) == 1, timeout=30)
        verdicts = adopted_verdicts()
        assert [(v.intent_id, v.verdict) for v in verdicts] == [(intent.intent_id, PolicyVerdict.ALLOW)]
        _no_failed_events(state)


# ======================================================= the replay check


def test_replay_check_reports_only_real_row_changes() -> None:
    value = _consistent(
        RequiredSourceState.QUALIFIED,
        PredecessorState.QUALIFIED,
        LineageState.CONSISTENT,
        CommitmentState.ACTIVE,
        LedgerState.NONE,
        FeedbackState.ABSENT,
        Interpretation.SETTLED,
        version=V3,
    )
    rows = [
        ("decision-a", V3.value, "4", value.canonical_bytes()),
        ("decision-b", V2.value, "4", replace(value, version=V2).canonical_bytes()),
        ("decision-c", PolicyKind.SUCCESSOR_POLICY_V1.value, "3", b'{"row":"3"}'),
        ("decision-d", V3.value, "4", b"not json"),
        ("decision-e", V2.value, "4", value.canonical_bytes()),  # kind mismatch
    ]
    report = replay_rows(rows)
    assert (report.total, report.replayed, report.unchanged, report.not_replayed_v1) == (5, 2, 2, 1)
    assert [item.decision_id for item in report.errors] == ["decision-d", "decision-e"]
    assert report.changes == () and not report.ok
    moved = _consistent(
        RequiredSourceState.QUALIFIED,
        PredecessorState.QUALIFIED,
        LineageState.CONSISTENT,
        CommitmentState.ACTIVE,
        LedgerState.APPROVAL_EXPIRED_UNCLAIMED,
        FeedbackState.ABSENT,
        Interpretation.SETTLED,
        version=V3,
    )
    changed = replay_rows([("decision-f", V3.value, "4", moved.canonical_bytes())])
    assert [(c.decision_id, c.stored_row, c.replayed_row) for c in changed.changes] == [
        ("decision-f", "4", "2c")
    ]
    assert not changed.ok


def test_replay_check_on_a_store_is_zero_and_read_only(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """§12-2: a production-path store (R18 off, V3 heads of rows 3 / 4 / 4b)
    replays with zero row changes, and the check writes nothing."""

    settings = _settings(tmp_path / "replay", r18=False)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        x, gid = ip16a2._qualified_goal(state)
        assert ip10._issue(state, gid).row is SuccessorRow.ROW_3
        _qualified_commitment(state, x)
        assert ip10._issue(state, gid).row is SuccessorRow.ROW_4
        second = ip16a2._second_goal(state)
        assert ip10._issue(state, second).row is SuccessorRow.ROW_3
        _qualified_commitment(state, state._ip16_second_x)
        monkeypatch.setattr(
            state.decision_issuer,
            "_interpretation",
            lambda *args: SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTESTED"),
        )
        assert ip10._issue(state, second).row is SuccessorRow.ROW_4B
    path = settings.subject_history.path
    copy = tmp_path / "copy" / "subject_history.sqlite3"
    copy.parent.mkdir()
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with sqlite3.connect(copy) as target:
            connection.backup(target)
    finally:
        connection.close()
    before = copy.read_bytes()
    report = replay_store(copy)
    assert (report.total, report.replayed, report.unchanged) == (4, 4, 4)
    assert report.by_kind == {V3.value: 4}
    assert report.ok and report.changes == () and report.errors == ()
    assert copy.read_bytes() == before
    assert sorted(os.listdir(copy.parent)) == ["subject_history.sqlite3"]
    assert replay_main([str(copy)]) == 0
    assert '"ok": true' in capsys.readouterr().out


def test_the_policy_path_reads_the_target_only() -> None:
    """§7: no TargetStore / writer in the Policy owner or the requalification."""

    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "suzka" / "r18"
    for name in ("policy_owner.py", "requalify.py", "admission.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert not imported & {"TargetStore", "initialize", "rotate_runtime_instance"}, name
