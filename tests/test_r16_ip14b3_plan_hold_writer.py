"""IP-14b-3: the plan-hold receipt writer (#411 IP-14b D1 r2 FINAL, HD-3 (a)).

The real server and the real R07 runtime: a policy HOLD row of a
PLAN_PREVIEW_REVIEW seals one ``R16_PLAN_HOLD`` receipt when
``subject_history.plan_hold_receipts`` is on, and nothing changes when it is
off.  Every pre-check of the writer mirrors a condition PREPARE / FINALIZE
would refuse by raising (a PREPARE exception fail-stops the runtime), so each
one has a test that violates it and finds a plain HOLD, no PREPARE, no row.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import r19_wake_fixtures as fx
from suzka.r16 import decision_issuer as issuer_module
from suzka.r16.cycle_contracts import PlanHoldReceiptV1
from suzka.r16.decision_issuer import (
    PLAN_HOLD_RECEIPT_SKIPPED,
    DecisionIssueStatus,
    _PendingPlanHold,
)
from suzka.r16.history_adoption import PLAN_HOLD_RECORDS_PRESENT_BUT_DISABLED
from suzka.r16.history_codec import GENESIS, PolicyKind
from suzka.r16.history_participant import SubjectHistoryRecordParticipant
from suzka.r16.history_reader import SubjectHistoryReader
from suzka.r16.history_records import (
    HistoryOperationKind,
    RecordKind,
)
from suzka.r16.history_store import (
    HistoryPhase,
    HistoryStoreCapacityError,
    HistoryStoreUnavailable,
)
from suzka.r16.plan_preview import _PlanSourceUnavailable
from suzka.r16.successor_policy import SuccessorPolicyInputV2, SuccessorRow
from suzka.runtime.agent_runtime import AgentRuntimeDurabilityError
from suzka.runtime.event_journal import EventLifecycle
from test_r16_ip10_decision_issuer import _issue
from test_r16_ip10_decision_issuer import _settings as _ip10_settings
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_goal

K = HistoryOperationKind
CONTRADICTORY = SimpleNamespace(outcome="EPISODE", stage="S2", status="CONTRADICTORY")
SETTLED = SimpleNamespace(outcome="EPISODE", stage="S1", status="PRESENT")


def _settings(root, *, holds: bool = True, r19: bool = False):  # type: ignore[no-untyped-def]
    base = _ip10_settings(root)
    return base.model_copy(
        update={
            "subject_history": base.subject_history.model_copy(
                update={"plan_hold_receipts": holds}
            ),
            "r19": base.r19.model_copy(update={"enabled": r19}),
        }
    )


def _with_holds(settings, holds: bool):  # type: ignore[no-untyped-def]
    return settings.model_copy(
        update={
            "subject_history": settings.subject_history.model_copy(
                update={"plan_hold_receipts": holds}
            )
        }
    )


def _contradictory(state) -> None:  # type: ignore[no-untyped-def]
    state.decision_issuer._interpretation = lambda *args: CONTRADICTORY


def _receipts(state) -> list:  # type: ignore[no-untyped-def]
    return fx.operations(state, K.R16_PLAN_HOLD)


def _transactions(state) -> int:  # type: ignore[no-untyped-def]
    return len(fx.operations(state))


def _adopted_receipts(state) -> list[PlanHoldReceiptV1]:  # type: ignore[no-untyped-def]
    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    assert found.blocked is None
    return [
        item.record
        for item in found.records
        if item.record_kind is RecordKind.PLAN_HOLD and type(item.record) is PlanHoldReceiptV1
    ]


def _no_failed_events(state) -> None:  # type: ignore[no-untyped-def]
    assert not any(
        record.lifecycle is EventLifecycle.FAILED for record in state.event_journal.inspect().records
    )


class _Prepares:
    """Counts ``participant.prepare`` (a prepare that raises fail-stops the runtime)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real = SubjectHistoryRecordParticipant.prepare
        outer = self

        def spy(self_, binding):  # type: ignore[no-untyped-def]
            outer.count += 1
            return real(self_, binding)

        monkeypatch.setattr(SubjectHistoryRecordParticipant, "prepare", spy)


def _shape(held) -> tuple:  # type: ignore[no-untyped-def]
    return (held.status, held.row, held.hold, held.owner_reason, held.decision_id)


@pytest.fixture
def world(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "hold"))) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        yield SimpleNamespace(state=state, gid=gid, monkeypatch=monkeypatch)


# ------------------------------------------------------ flag off: unchanged


def test_flag_off_the_hold_path_is_unchanged_and_writes_nothing(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "off", holds=False))) as client:
        state = client.app.state
        assert state.subject_history_store.plan_hold_receipts_enabled is False
        _x, gid = _qualified_goal(state)
        _contradictory(state)
        prepares = _Prepares(monkeypatch)
        before = len(state.event_journal.inspect().completed_transactions)
        held = _issue(state, gid)
        assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)
        # No R07 transaction, no participant, no SubjectHistory write.
        assert prepares.count == 0
        assert state.subject_history_store.has_operations() is False
        assert len(state.event_journal.inspect().completed_transactions) == before
        assert state.decision_issuer._pending is None
        _no_failed_events(state)


def test_the_switch_on_returns_the_same_hold_value_as_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    shapes = []
    for holds in (False, True):
        with TestClient(_app(_settings(tmp_path / f"cmp-{holds}", holds=holds))) as client:
            state = client.app.state
            _x, gid = _qualified_goal(state)
            _contradictory(state)
            shapes.append(_shape(_issue(state, gid)))
    assert shapes[0] == shapes[1]
    assert shapes[0][:2] == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)


# ------------------------------------------------- flag on: the 2b receipt


def test_the_2b_hold_is_sealed_exactly_once(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    _contradictory(state)
    held = _issue(state, world.gid)
    assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)
    assert held.decision is None and held.decision_id is None
    rows = _receipts(state)
    assert [row.phase for row in rows] == [HistoryPhase.FINALIZED]
    assert state.decision_issuer._pending is None
    (receipt,) = _adopted_receipts(state)
    assert receipt.policy_row is SuccessorRow.ROW_2B
    assert receipt.hold_reason is held.hold.reason  # type: ignore[union-attr]
    assert receipt.predecessor_key == GENESIS and receipt.predecessor_digest is None
    assert receipt.policy_kind is PolicyKind.SUCCESSOR_POLICY_V3
    sealed = SuccessorPolicyInputV2.from_bytes(receipt.policy_input_bytes)
    assert sealed.event_id == receipt.event_id == rows[0].event_id
    # It reserves no Decision slot and writes no lineage columns.
    assert rows[0].lineage_scope is None and rows[0].review_intent_key is None
    assert state.subject_history_store.has_plan_hold_operations()
    assert not state.subject_history_store.has_record_operations()
    assert not state.subject_history_store.has_r19_operations()

    # The same evidence again: nothing new (the plan-hold key is live).
    before = _transactions(state)
    prepares = _Prepares(world.monkeypatch)
    again = _issue(state, world.gid)
    assert _shape(again) == _shape(held)
    assert prepares.count == 0 and _transactions(state) == before
    assert len(_adopted_receipts(state)) == 1
    _no_failed_events(state)


def test_the_receipt_is_independent_of_r19(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with TestClient(_app(_settings(tmp_path / "r19", r19=True))) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        _contradictory(state)
        _issue(state, gid)
        assert len(_receipts(state)) == 1
        assert not state.subject_history_store.has_r19_operations()


def test_a_non_settled_run_after_a_decision_names_its_predecessor(world) -> None:  # type: ignore[no-untyped-def]
    """S0's non-SETTLED run: a Decision head exists, the interpretation turns
    CONTRADICTORY (new evidence): row 2b on a qualified predecessor."""

    state = world.state
    first = _issue(state, world.gid)
    assert first.status is DecisionIssueStatus.ISSUED and first.decision is not None
    assert _receipts(state) == []
    _contradictory(state)
    held = _issue(state, world.gid)
    assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)
    (receipt,) = _adopted_receipts(state)
    assert receipt.predecessor_key == first.decision.decision_id
    assert receipt.predecessor_digest == first.decision.content_digest
    # The Decision chain and the head are untouched by the receipt.
    head = SubjectHistoryReader(state.event_journal, state.subject_history_store).head(
        first.decision.lineage_scope
    )
    assert head.decision == first.decision and head.depth == 1


# ----------------------------------------------- HOLDs that are not recorded


def test_an_unavailable_interpretation_row_1_is_not_recorded(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state

    def broken(*args):  # type: ignore[no-untyped-def]
        raise RuntimeError("R15 unavailable")

    state.decision_issuer._interpretation = broken
    prepares = _Prepares(world.monkeypatch)
    held = _issue(state, world.gid)
    assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_1)
    assert prepares.count == 0 and _transactions(state) == 0


def test_a_conflicting_predecessor_row_0_is_not_recorded(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    assert _issue(state, world.gid).status is DecisionIssueStatus.ISSUED
    rows = _transactions(state)
    world.monkeypatch.setattr(state.decision_issuer, "_head_verified", lambda *args: False)
    held = _issue(state, world.gid)
    assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_0)
    assert _transactions(state) == rows and _receipts(state) == []


def test_an_issuer_refusal_is_not_recorded(world) -> None:  # type: ignore[no-untyped-def]
    """Capacity of a Decision / source unavailable are transient refusals, not
    policy rows: no policy result exists to seal."""

    state = world.state
    world.monkeypatch.setattr(
        state.subject_history_store,
        "check_headroom",
        lambda length: (_ for _ in ()).throw(HistoryStoreCapacityError("full")),
    )
    held = _issue(state, world.gid)  # a Decision row (3) that cannot be admitted
    assert held.status is DecisionIssueStatus.HOLD and held.owner_reason is not None
    assert _transactions(state) == 0


# ------------------------------------------- mirrored pre-checks (no raise)


def _store_not_active(world):  # type: ignore[no-untyped-def]
    """The evaluation reads the store too, so only the writer's own view of the
    phase is switched (a store that is not ACTIVE refuses PREPARE)."""

    world.monkeypatch.setattr(world.state.decision_issuer, "_store_active", lambda: False)


def _incarnation_changed(world):  # type: ignore[no-untyped-def]
    world.monkeypatch.setattr(
        "suzka.r16.history_adoption.current_incarnation", lambda journal, store: "0" * 64
    )


def _sources_moved(world):  # type: ignore[no-untyped-def]
    world.monkeypatch.setattr(world.state.decision_issuer, "_sources_moved", lambda *a: True)


def _sources_vanish(world):  # type: ignore[no-untyped-def]
    calls = {"n": 0}
    preview = world.state.decision_issuer._preview
    real = preview._qualify

    def qualify(goal_id, event):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] > 2:  # the two evaluation reads, then the pre-check read
            raise _PlanSourceUnavailable("moved")
        return real(goal_id, event)

    world.monkeypatch.setattr(preview, "_qualify", qualify)


def _headroom(world):  # type: ignore[no-untyped-def]
    def full(length):  # type: ignore[no-untyped-def]
        raise HistoryStoreCapacityError("full")

    world.monkeypatch.setattr(world.state.subject_history_store, "check_headroom", full)


def _store_unavailable(world):  # type: ignore[no-untyped-def]
    def down(operation):  # type: ignore[no-untyped-def]
        raise HistoryStoreUnavailable("busy")

    world.monkeypatch.setattr(world.state.subject_history_store, "prepare_collisions", down)


def _not_representable(world):  # type: ignore[no-untyped-def]
    def bad(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("not representable")

    world.monkeypatch.setattr(issuer_module, "PlanHoldReceiptV1", bad)


def _key_live_prepared(world):  # type: ignore[no-untyped-def]
    """A PREPARED ``R16_PLAN_HOLD`` of another transaction holds the key.  A raw
    PREPARED row with no R07 transaction behind it makes the reader block the
    incarnation (the preview is then row 1, before the writer runs), so the
    live key is injected as IP-20c does for its own PREPARED mirrors."""

    store = world.state.subject_history_store
    real = store.live_keys
    world.monkeypatch.setattr(
        store,
        "live_keys",
        lambda keys: real(keys) | {key for key in keys if key.startswith("plan-hold:")},
    )


_MIRRORS = {
    "store_not_active": (_store_not_active, "STORE_NOT_ACTIVE"),
    "incarnation_changed": (_incarnation_changed, "INCARNATION_CHANGED"),
    "sources_moved": (_sources_moved, "SOURCES_MOVED"),
    "sources_unreadable": (_sources_vanish, "SOURCES_MOVED"),
    "key_live_prepared": (_key_live_prepared, "KEY_LIVE"),
    "no_headroom": (_headroom, "HEADROOM"),
    "store_unavailable": (_store_unavailable, "STORE_UNAVAILABLE"),
    "not_representable": (_not_representable, "NOT_REPRESENTABLE"),
}


@pytest.mark.parametrize("case", sorted(_MIRRORS))
def test_a_failed_pre_check_is_the_same_hold_with_no_prepare_and_no_row(  # type: ignore[no-untyped-def]
    world, case, caplog
) -> None:
    state = world.state
    _contradictory(state)
    violate, reason = _MIRRORS[case]
    rows = _transactions(state)
    violate(world)
    prepares = _Prepares(world.monkeypatch)
    with caplog.at_level(logging.INFO, logger=issuer_module.LOGGER.name):
        held = _issue(state, world.gid)
    assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)
    assert prepares.count == 0
    world.monkeypatch.undo()  # the violation also breaks the incarnation read
    assert _transactions(state) == rows
    assert any(
        PLAN_HOLD_RECEIPT_SKIPPED in record.getMessage() and f"reason={reason}" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]
    assert state.decision_issuer._pending is None
    _no_failed_events(state)


def test_a_live_plan_hold_key_of_a_finalized_row_is_a_skip(world, caplog) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    _contradictory(state)
    _issue(state, world.gid)
    assert [r.phase for r in _receipts(state)] == [HistoryPhase.FINALIZED]
    prepares = _Prepares(world.monkeypatch)
    with caplog.at_level(logging.INFO, logger=issuer_module.LOGGER.name):
        _issue(state, world.gid)
    assert prepares.count == 0 and len(_receipts(state)) == 1
    assert any("reason=KEY_LIVE" in record.getMessage() for record in caplog.records)


def test_an_unexpected_failure_while_building_is_a_logged_hold_never_an_event_failure(  # type: ignore[no-untyped-def]
    world, caplog
) -> None:
    state = world.state
    _contradictory(state)

    def boom(rows):  # type: ignore[no-untyped-def]
        raise RuntimeError("unexpected")

    world.monkeypatch.setattr(issuer_module, "operations_digest", boom)
    with caplog.at_level(logging.INFO, logger=issuer_module.LOGGER.name):
        held = _issue(state, world.gid)
    assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)
    assert state.decision_issuer._pending is None
    assert any("reason=unexpected" in record.getMessage() for record in caplog.records)
    _no_failed_events(state)


# ----------------------------------------------------- _validate_commit / _clear


def test_validate_commit_detects_changed_history_before_the_wal(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A pending receipt whose own row, or the other rows, changed fail-stops."""

    with TestClient(_app(_settings(tmp_path / "vc"))) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        _contradictory(state)
        issuer = state.decision_issuer
        real = issuer._validate_commit

        def tampered(event):  # type: ignore[no-untyped-def]
            pending = issuer._pending
            assert type(pending) is _PendingPlanHold
            issuer._pending = replace(pending, rows_digest=(999, "0" * 64))
            real(event)

        issuer._validate_commit = tampered
        with pytest.raises(AgentRuntimeDurabilityError):
            state._plan_preview_owner._submit_preview(gid).result(timeout=30)
        issuer._clear()


def test_a_stale_pending_receipt_is_refused_for_another_event_type(world) -> None:  # type: ignore[no-untyped-def]
    state = world.state
    issuer = state.decision_issuer
    real_submit = state._plan_preview_owner
    event = SimpleNamespace()
    issuer._pending = SimpleNamespace(event=event)  # type: ignore[assignment]
    with pytest.raises(_PlanSourceUnavailable, match="stale"):
        real_submit._validate_commit(SimpleNamespace(event_type=object()))  # type: ignore[arg-type]
    issuer._clear()
    assert issuer._pending is None


# ---------------------------------------------------- fail-stop and restart


def test_a_crash_before_the_wal_is_aborted_and_a_new_boot_writes_the_receipt_once(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "crash")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        _contradictory(state)
        issuer = state.decision_issuer

        def crash(event):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated pre-WAL failure")

        issuer._validate_commit = crash
        with pytest.raises(AgentRuntimeDurabilityError):
            state._plan_preview_owner._submit_preview(gid).result(timeout=30)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _receipts(state)] == [HistoryPhase.ABORTED]
        assert _adopted_receipts(state) == []
        # The aborted PREPARE released the key: the same evidence seals now.
        _contradictory(state)
        _issue(state, gid)
        assert [row.phase for row in _receipts(state)] == [HistoryPhase.ABORTED, HistoryPhase.FINALIZED]
        assert len(_adopted_receipts(state)) == 1
        _issue(state, gid)
        assert len(_adopted_receipts(state)) == 1


def test_a_crash_after_the_internal_commit_rolls_forward_and_is_not_written_twice(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "forward")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        _contradictory(state)

        def down(self_, binding):  # type: ignore[no-untyped-def]
            from suzka.runtime.transaction_coordinator import ParticipantUnavailableError

            raise ParticipantUnavailableError("simulated crash before FINALIZE")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(SubjectHistoryRecordParticipant, "finalize", down)
            with pytest.raises(Exception):
                state._plan_preview_owner._submit_preview(gid).result(timeout=30)
        assert [row.phase for row in _receipts(state)] == [HistoryPhase.PREPARED]
    with TestClient(_app(settings)) as client:
        state = client.app.state
        assert state.event_journal.inspect().open_transactions == ()
        assert [row.phase for row in _receipts(state)] == [HistoryPhase.FINALIZED]
        assert len(_adopted_receipts(state)) == 1
        # Restart idempotence: the same evidence adds nothing.
        _contradictory(state)
        before = _transactions(state)
        _issue(state, gid)
        assert _transactions(state) == before and len(_adopted_receipts(state)) == 1
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _contradictory(state)
        before = _transactions(state)
        _issue(state, gid)
        assert _transactions(state) == before and len(_receipts(state)) == 1
        _no_failed_events(state)


def test_a_store_restarted_with_the_switch_off_keeps_the_receipt_and_writes_no_more(  # type: ignore[no-untyped-def]
    tmp_path, caplog
) -> None:
    settings = _settings(tmp_path / "off-after")
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        _contradictory(state)
        _issue(state, gid)
        assert len(_receipts(state)) == 1
    with caplog.at_level(logging.WARNING):
        with TestClient(_app(_with_holds(settings, False))) as client:
            state = client.app.state
            assert any(
                PLAN_HOLD_RECORDS_PRESENT_BUT_DISABLED in record.getMessage()
                for record in caplog.records
            )
            _contradictory(state)
            before = _transactions(state)
            held = _issue(state, gid)
            assert (held.status, held.row) == (DecisionIssueStatus.HOLD, SuccessorRow.ROW_2B)
            assert _transactions(state) == before
            assert len(_adopted_receipts(state)) == 1
            _no_failed_events(state)
