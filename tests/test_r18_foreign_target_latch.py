"""HD-F15-11 (a): a FOREIGN R18 target closes the gate (BLOCKED_TARGET) and keeps
it closed until the target is this subject's again, without stopping the
subject and without writing a row into the foreign file.

Foreign, as the merged code defines it: another namespace, an adopted Intent
bound to another ``target_instance_id``, or a ledger row whose operation key
is not its own group's. Every fixture is synthetic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import pytest

from suzka.r18.executor import BLOCKED_AFTER_SECONDS
from suzka.r18.sweep import BLOCKED_MIN_DWELL_SECONDS, GateState
from suzka.r18.target import TargetStore, new_runtime_instance_id

import test_r18_ip16a2_intent_issuer as ip16a2
from test_r16_u2a_source_owner import _app, _chat
from test_r18_ip15_target_verifier import _store
from test_r18_ip18a_executor import _opened, _run, _World
from test_r18_ip18b_facts_sweep import _until


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    made = _World(tmp_path)
    made.gate._monotonic = made.clock.monotonic  # type: ignore[attr-defined]
    try:
        yield made
    finally:
        made.close()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _foreign(world: _World, tmp_path: Path) -> TargetStore:
    """Another subject's target (its own instance id), swapped in for ours."""

    (tmp_path / "other").mkdir()
    other = _store(tmp_path / "other")
    world.executor._close_store()
    world.executor._path = other.path
    return other


def _swap_back(world: _World, genuine: Path) -> None:
    world.executor._close_store()
    world.executor._path = genuine


def _tick(world: _World, seconds: float, steps: int = 1) -> None:
    for _ in range(steps):
        world.clock.advance(seconds)
        world.executor.tick()
        world.executor.run_pending()


def test_a_foreign_identity_closes_the_gate_at_once_and_nothing_is_written(world: _World, tmp_path: Path) -> None:
    genuine = world.path
    other = _foreign(world, tmp_path)
    before = _digest(other.path)
    world.executor.start()
    _run(world)
    assert world.gate.state is GateState.BLOCKED_TARGET and not world.gate.is_open()
    assert world.gate.blocked_target() is True
    # No S1 rotation, no fence, no effect: the file is byte-identical.
    assert _digest(other.path) == before
    assert other.meta().runtime_instance_id != world.executor._id
    assert world.writes == 0 and genuine != other.path


def test_a_foreign_namespace_closes_the_gate(world: _World, tmp_path: Path) -> None:
    root = tmp_path / "ns"
    root.mkdir(mode=0o700)
    foreign = TargetStore.create(root / "target.sqlite3", target_namespace="someone-else")
    foreign.close()
    world.executor._close_store()
    world.executor._path = foreign.path
    before = _digest(foreign.path)
    world.executor.start()
    _run(world)
    assert world.gate.state is GateState.BLOCKED_TARGET
    assert _digest(foreign.path) == before


def test_a_ledger_row_with_a_foreign_key_closes_the_gate(world: _World, tmp_path: Path) -> None:
    root = tmp_path / "rows"
    root.mkdir(mode=0o700)
    path = root / "target.sqlite3"
    TargetStore.create(path, target_namespace="review-artifact").close()
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO target_operations (operation_key, obligation_group_key, direction, kind) "
        "VALUES (?, ?, 'FORWARD', 'FENCED')",
        ("a" * 64, "b" * 64),
    )
    connection.commit()
    connection.close()
    world.records.clear()  # no Intent: only the row itself says it is foreign
    world.executor._close_store()
    world.executor._path = path
    before = _digest(path)
    world.executor.start()
    _run(world)
    assert world.gate.state is GateState.BLOCKED_TARGET
    assert _digest(path) == before


def test_a_foreign_target_keeps_the_gate_closed_on_every_probe(world: _World, tmp_path: Path) -> None:
    other = _foreign(world, tmp_path)
    before = _digest(other.path)
    world.executor.start()
    _run(world)
    _tick(world, 20, steps=12)  # four minutes of probes, always re-evaluated
    assert world.gate.state is GateState.BLOCKED_TARGET and not world.gate.is_open()
    assert _digest(other.path) == before


def test_the_genuine_target_returning_reopens_by_itself_with_the_existing_dwell(
    world: _World, tmp_path: Path
) -> None:
    genuine = world.path
    _foreign(world, tmp_path)
    cleared: list[int] = []
    world.gate.subscribe_unblocked(lambda: cleared.append(1))
    world.executor.start()
    _run(world)
    assert world.gate.state is GateState.BLOCKED_TARGET
    _swap_back(world, genuine)
    _tick(world, 30, steps=6)  # the probe finds it
    assert world.gate.is_open()
    assert world.gate.blocked_target() is True and cleared == []  # hysteresis holds
    _tick(world, BLOCKED_MIN_DWELL_SECONDS, steps=1)
    assert world.gate.blocked_target() is False and cleared == [1]
    assert TargetStore.open(genuine).meta().runtime_instance_id == world.executor._id


def test_a_missing_target_still_waits_60_seconds_before_it_blocks(world: _World, tmp_path: Path) -> None:
    world.executor._close_store()
    world.executor._path = tmp_path / "nowhere" / "target.sqlite3"
    world.executor.start()
    _run(world)
    assert world.gate.state is GateState.SWEEPING  # unchanged: not blocked at once
    _tick(world, 2, steps=int(BLOCKED_AFTER_SECONDS // 2) + 1)
    assert world.gate.state is GateState.BLOCKED_TARGET


def test_a_corrupt_target_still_waits_60_seconds_before_it_blocks(world: _World, tmp_path: Path) -> None:
    root = tmp_path / "bad"
    root.mkdir(mode=0o700)
    path = root / "target.sqlite3"
    path.write_bytes(b"not a database" * 100)
    path.chmod(0o600)
    world.executor._close_store()
    world.executor._path = path
    world.executor.start()
    _run(world)
    assert world.gate.state is GateState.SWEEPING
    assert path.read_bytes() == b"not a database" * 100  # never repaired or deleted


def test_a_target_swapped_after_the_gate_opened_is_not_fenced(world: _World, tmp_path: Path) -> None:
    _opened(world)
    other = _foreign(world, tmp_path)
    before = _digest(other.path)
    other_claim = world.claim
    world.executor.enqueue(other_claim)  # write is refused by binding; the settle must be too
    _run(world)
    assert _digest(other.path) == before
    assert other.status(world.intent.obligation_group_key).forward is None
    assert world.gate.state is GateState.BLOCKED_TARGET


def test_a_matching_target_is_not_foreign(world: _World) -> None:
    _opened(world)
    assert world.gate.state is GateState.OPEN


def test_the_identity_check_needs_an_intent_so_a_fresh_target_is_adoptable(world: _World, tmp_path: Path) -> None:
    world.records.clear()
    _foreign(world, tmp_path)
    world.executor.start()
    _run(world)
    assert world.gate.is_open()  # unchanged: nothing to compare an identity with


def test_with_r18_off_a_foreign_target_changes_nothing(tmp_path: Path) -> None:
    settings = ip16a2._settings(tmp_path / "off", r18=False)
    path = settings.r18.target_path
    other = TargetStore.open(path)
    other.rotate_runtime_instance(new_runtime_instance_id())
    other.close()
    before = _digest(path)
    with TestClient(_app(settings, executor=True)) as client:
        state = client.app.state
        assert state.r18_executor is None and state.r18_gate.state is GateState.DISABLED
        _chat(state)
    assert _digest(path) == before


def test_chat_and_cognition_are_unaffected_by_a_blocked_foreign_target(tmp_path: Path) -> None:
    settings = ip16a2._settings(tmp_path / "chat")
    # A target of another namespace sits where ours should be.
    path = settings.r18.target_path
    path.unlink()
    TargetStore.create(path, target_namespace="someone-else").close()
    before = _digest(path)
    with TestClient(_app(settings, executor=True)) as client:
        state = client.app.state
        _until(lambda: state.r18_gate.state is GateState.BLOCKED_TARGET)
        assert not state.r18_gate.is_open()
        _chat(state)  # the subject keeps talking and thinking
        assert state.r18_gate.state is GateState.BLOCKED_TARGET
    assert _digest(path) == before
