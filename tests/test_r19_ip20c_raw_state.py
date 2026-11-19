"""IP-20c: the pre-checks against REAL raw store rows (IP-20 D1 r3 §4-4 step 3, §6-4).

The rows are real PREPARED / FINALIZED operations of a real v4 store, built from
the IP-14b fixtures; only the adopted view is hand-fed (a PREPARED or
FINALIZED-not-terminal row is by definition not adopted).  Each case shows both
sides: the store would refuse (the key is live, the slot is held), and the
eligibility functions hold before PREPARE would ever be reached.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from suzka.r16.history_store import HistoryPhase, HistoryStoreError
from suzka.r19 import review_owner
from suzka.r19.contracts import TerminalReason
from suzka.r19.ledger import (
    WakeHold,
    attempt_end_key,
    claim_eligibility,
    closure_admissible,
    cycle_key,
    derive_wake_view,
    terminal_eligibility,
)
from suzka.r19.raw_state import raw_claims_of

from test_r16_ip14a_history_v2 import _apply, _prepare
from test_r16_ip14b1_history_v4 import INC, Cycle, _store
from test_r19_ip20a_ledger import _claim_rec, _wake_rec

T_LATE = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)  # past every lease of Cycle


def _entry(cycle: Cycle, *adopted_claims: int):  # type: ignore[no-untyped-def]
    records = [_wake_rec(cycle.wake), *(_claim_rec(cycle.claims[n - 1]) for n in adopted_claims)]
    stamp = T_LATE.strftime("%Y-%m-%dT%H:%M:%SZ")
    found = derive_wake_view(records, stamp, clock_floor=None).get(cycle.wake.wakeup_id)
    assert found is not None
    return found


def _claim_check(entry, raw, cycle_live=False):  # type: ignore[no-untyped-def]
    return claim_eligibility(
        entry, incarnation_id=INC, now=T_LATE, live=None, floor=None,
        raw_claims=raw, cycle_key_live=cycle_live,
    )


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = _store(tmp_path)
    cycle = Cycle()
    _apply(store, cycle.materialize())
    return store, cycle


@pytest.mark.parametrize("phase", ["PREPARED", "FINALIZED"])
def test_a_raw_claim_that_is_not_adopted_is_in_flight_never_a_new_index(world, phase) -> None:  # type: ignore[no-untyped-def]
    """Forward note 4: counting the adopted claims alone would pick index 1
    again, whose ``wake-attempt:`` key is already live (a PREPARE collision)."""

    store, cycle = world
    (_prepare if phase == "PREPARED" else _apply)(store, cycle.claim(1))
    raw = raw_claims_of(store.operations(INC), cycle.wake.wakeup_id)
    assert [claim.attempt_index for claim in raw] == [1]
    assert {row.phase for row in store.operations(INC) if row.operation_kind.value == "R19_WAKE_CLAIM"} == {
        HistoryPhase[phase]
    }
    assert _claim_check(_entry(cycle), raw).hold is WakeHold.IN_FLIGHT
    # Terminals look at the raw count too.
    held = terminal_eligibility(
        _entry(cycle), TerminalReason.EXPIRED_WAKE, incarnation_id=INC, now=T_LATE,
        live=None, floor=None, raw_claims=raw, cycle_key_live=False,
    )
    assert held.hold is WakeHold.IN_FLIGHT


def test_the_attempt_key_collision_is_seen_by_the_store_pre_check(world) -> None:  # type: ignore[no-untyped-def]
    store, cycle = world
    _prepare(store, cycle.claim(1))
    again = cycle.claim(1)  # the same Wake, the same index, another event
    key = f"wake-attempt:{cycle.wake.wakeup_id}:1"
    collisions = store.prepare_collisions(again)
    assert collisions.live_keys == frozenset({key}) and not collisions.clear
    with pytest.raises(HistoryStoreError):
        _prepare(store, again)  # what PREPARE would do: the runtime fail-stop


def test_a_prepared_closure_reserves_the_cycle_for_claims_terminals_and_outcomes(world) -> None:  # type: ignore[no-untyped-def]
    store, cycle = world
    _apply(store, cycle.claim(1))
    _prepare(store, cycle.hold())  # a PREPARED (not adopted) closure of this cycle
    cycle_k = cycle_key(INC, cycle.wake.logical_cycle_id)
    end_k = attempt_end_key(cycle.claims[0].claim_id)
    live = store.live_keys((cycle_k, end_k))
    assert live == frozenset({cycle_k, end_k})
    entry = _entry(cycle, 1)
    raw = raw_claims_of(store.operations(INC), cycle.wake.wakeup_id)
    assert _claim_check(entry, raw, cycle_live=cycle_k in live).hold is WakeHold.CYCLE_RESERVED
    held = terminal_eligibility(
        entry, TerminalReason.EXPIRED_WAKE, incarnation_id=INC, now=T_LATE, live=None,
        floor=None, raw_claims=raw, cycle_key_live=cycle_k in live,
    )
    assert held.hold is WakeHold.CYCLE_RESERVED
    assert closure_admissible(
        entry, cycle.claims[0], incarnation_id=INC, raw_claims=raw, live_keys=live
    ) is WakeHold.CYCLE_RESERVED
    # The trigger side: a second closure or a terminal would collide on the key.
    assert not store.prepare_collisions(cycle.terminal(TerminalReason.EXPIRED_WAKE)).clear


def test_an_outcome_first_blocks_the_resolution_and_the_resolution_first_blocks_the_outcome(world) -> None:  # type: ignore[no-untyped-def]
    """The race on ``attempt-end:<claim_id>`` in both orders, on real rows."""

    store, cycle = world
    _apply(store, cycle.claim(1))
    entry = _entry(cycle, 1)
    claim = cycle.claims[0]
    end_k = attempt_end_key(claim.claim_id)
    # outcome first (PREPARED): the resolution is held back, the key is live
    outcome = cycle.outcome()
    _prepare(store, outcome)
    raw = raw_claims_of(store.operations(INC), cycle.wake.wakeup_id)
    live = store.live_keys((cycle_key(INC, cycle.wake.logical_cycle_id), end_k))
    assert live == frozenset({end_k})
    assert closure_admissible(entry, claim, incarnation_id=INC, raw_claims=raw, live_keys=live) is (
        WakeHold.ATTEMPT_ENDED
    )
    assert end_k in store.prepare_collisions(cycle.hold()).live_keys
    # resolution first: the outcome is held back by the same key
    other_store = _store(store.path.parent.parent / "second")
    other = Cycle("goal-x")
    _apply(other_store, other.materialize())
    _apply(other_store, other.claim(1))
    _prepare(other_store, other.hold())
    assert attempt_end_key(other.claims[0].claim_id) in other_store.prepare_collisions(other.outcome()).live_keys


def test_a_claim_or_wake_of_another_incarnation_is_a_mismatch_hold(world) -> None:  # type: ignore[no-untyped-def]
    store, cycle = world
    _apply(store, cycle.claim(1))
    entry = _entry(cycle, 1)
    raw = raw_claims_of(store.operations(INC), cycle.wake.wakeup_id)
    other = "c" * 64
    assert claim_eligibility(
        entry, incarnation_id=other, now=T_LATE, live=None, floor=None, raw_claims=raw, cycle_key_live=False
    ).hold is WakeHold.INCARNATION_MISMATCH
    assert terminal_eligibility(
        entry, TerminalReason.EXPIRED_WAKE, incarnation_id=other, now=T_LATE, live=None,
        floor=None, raw_claims=raw, cycle_key_live=False,
    ).hold is WakeHold.INCARNATION_MISMATCH
    assert closure_admissible(
        entry, cycle.claims[0], incarnation_id=other, raw_claims=raw, live_keys=frozenset()
    ) is WakeHold.INCARNATION_MISMATCH


@pytest.mark.parametrize("phase", ["PREPARED", "FINALIZED"])
def test_a_held_decision_slot_is_found_on_real_rows_and_collides(world, phase) -> None:  # type: ignore[no-untyped-def]
    """The predecessor slot a PLAN_PREVIEW or Wake Decision holds, PREPARED or
    FINALIZED-not-terminal: ``_slot_holder`` classifies it and the store's own
    pre-check (``prepare_collisions``) reports the collision, so no PREPARE."""

    store, first = world
    _apply(store, first.claim(1))
    (_prepare if phase == "PREPARED" else _apply)(store, first.decision())
    second = Cycle("goal-2")
    _apply(store, second.materialize())
    _apply(store, second.claim(1))
    rival = second.decision()  # the same lineage head slot (GENESIS predecessor)
    collisions = store.prepare_collisions(rival)
    assert collisions.decision_slot_occupied and not collisions.clear
    committed = rival.cycle_decision
    assert committed is not None
    assert review_owner._slot_holder(store.operations(INC), committed) is HistoryPhase[phase]
