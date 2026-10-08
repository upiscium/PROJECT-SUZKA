"""Runtime Attention persistence-port transaction and ownership tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event, Thread

import pytest

import suzka.attention.system as system_module
from suzka.attention.adapters import project_working_memory
from suzka.attention.bounds import maximum_attention_continuity_fixture
from suzka.attention.contracts import AttentionContinuity, AttentionEvent
from suzka.attention.system import AttentionSystem
from suzka.working_memory_contracts import WorkingMemoryDecisionReason
from suzka.runtime.attention_state_port import (
    AttentionRestoreTransaction,
    AttentionStatePort,
)


_NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def _refreshed_system(index: int, sequence: int) -> tuple[AttentionSystem, AttentionContinuity]:
    from test_attention_adapters import _item, _view

    event = AttentionEvent(
        f"attention-port:{sequence}",
        sequence,
        _NOW + timedelta(seconds=sequence),
    )
    item = _item(
        source_id=f"attention-port-source-{index}",
        activation=1.0,
        salience=1.0,
    )
    view = _view(
        item,
        content=f"attention-port-row-{index}",
        reason=WorkingMemoryDecisionReason.SELECTED,
    )
    current_view = replace(view, revision=sequence)
    projection = project_working_memory(
        item,
        revision=sequence,
        event=event,
        view=current_view,
    )
    system = AttentionSystem()
    return system, system.refresh((projection,), event).snapshot


def test_port_is_exact_owner_bound_and_exports_only_detached_continuity() -> None:
    system, snapshot = _refreshed_system(1, 1)
    port = AttentionStatePort(system)
    exported = port.export_attention_state()

    assert type(exported) is AttentionContinuity
    assert exported is not snapshot
    assert exported.canonical_bytes() == snapshot.canonical_bytes()
    assert {
        name for name in vars(AttentionStatePort) if not name.startswith("_")
    } == {"export_attention_state", "prepare_attention_restore"}
    assert not hasattr(port, "refresh")
    assert not hasattr(port, "selected_view")
    assert not hasattr(port, "_bundle")
    with pytest.raises(TypeError, match="exact AttentionSystem"):
        AttentionStatePort(object())  # type: ignore[arg-type]


def test_publish_then_rollback_restores_exact_prior_bundle_and_ephemeral_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, prior = _refreshed_system(1, 1)
    _, desired = _refreshed_system(2, 2)
    before_view = owner.selected_view()
    assert prior.focused_ids
    assert before_view.competition is not None
    assert before_view.prompt is not None
    prior_bundle = owner._bundle
    port = AttentionStatePort(owner)
    transaction = port.prepare_attention_restore(desired)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("restore must not rerun competition or prompt selection")

    monkeypatch.setattr(system_module, "compete_attention", forbidden)
    monkeypatch.setattr(system_module, "select_attention_prompt", forbidden)
    transaction.publish()

    published_view = owner.selected_view()
    assert owner.snapshot().canonical_bytes() == desired.canonical_bytes()
    assert published_view.competition is None
    assert published_view.prompt is None
    assert tuple(target.candidate_id for target in published_view.focused_targets) == (
        desired.focused_ids
    )

    transaction.close()
    assert owner._bundle is prior_bundle
    assert owner.snapshot().canonical_bytes() == prior.canonical_bytes()
    assert owner.selected_view() == before_view
    transaction.close()
    with pytest.raises(RuntimeError, match="closed"):
        transaction.publish()


def test_complete_commits_the_staged_bundle_and_double_publication_is_rejected() -> None:
    owner, _ = _refreshed_system(1, 1)
    _, desired = _refreshed_system(3, 3)
    port = AttentionStatePort(owner)
    transaction = port.prepare_attention_restore(desired)
    assert {
        name
        for name in vars(AttentionRestoreTransaction)
        if not name.startswith("_")
    } == {"publish", "complete", "close"}

    with pytest.raises(RuntimeError, match="not publishable"):
        transaction.complete()
    transaction.publish()
    with pytest.raises(RuntimeError, match="already published"):
        transaction.publish()
    transaction.complete()
    with pytest.raises(RuntimeError, match="already published"):
        transaction.publish()
    transaction.close()
    transaction.close()

    assert owner.snapshot().canonical_bytes() == desired.canonical_bytes()
    view = owner.selected_view()
    assert view.competition is None
    assert view.prompt is None
    assert tuple(target.candidate_id for target in view.focused_targets) == desired.focused_ids


def test_close_without_publication_is_safe_and_does_not_change_owner() -> None:
    owner, before = _refreshed_system(1, 1)
    _, desired = _refreshed_system(4, 4)
    before_bundle = owner._bundle
    transaction = AttentionStatePort(owner).prepare_attention_restore(desired)

    transaction.close()
    transaction.close()

    assert owner._bundle is before_bundle
    assert owner.snapshot().canonical_bytes() == before.canonical_bytes()


def test_invalid_tampered_snapshot_releases_owner_lock_without_repair() -> None:
    owner, prior = _refreshed_system(1, 1)
    _, tampered = _refreshed_system(5, 5)
    object.__setattr__(tampered.candidates[0].source, "digest", "f" * 64)
    port = AttentionStatePort(owner)

    with pytest.raises(ValueError, match="digest"):
        port.prepare_attention_restore(tampered)

    finished = Event()
    errors: list[BaseException] = []

    def read_owner() -> None:
        try:
            assert owner.snapshot().canonical_bytes() == prior.canonical_bytes()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    reader = Thread(target=read_owner)
    reader.start()
    reader.join(2)
    assert finished.is_set()
    assert not reader.is_alive()
    assert not errors
    assert owner.snapshot().canonical_bytes() == prior.canonical_bytes()


def test_open_transaction_holds_actual_system_lock_until_close() -> None:
    owner, prior = _refreshed_system(1, 1)
    _, desired = _refreshed_system(6, 6)
    transaction = AttentionStatePort(owner).prepare_attention_restore(desired)
    started = Event()
    completed = Event()
    results: list[AttentionContinuity] = []

    def read_owner() -> None:
        started.set()
        results.append(owner.snapshot())
        completed.set()

    reader = Thread(target=read_owner)
    reader.start()
    assert started.wait(2)
    assert not completed.wait(0.05)
    transaction.close()
    reader.join(2)

    assert not reader.is_alive()
    assert completed.is_set()
    assert len(results) == 1
    assert results[0].canonical_bytes() == prior.canonical_bytes()


def test_transaction_is_sealed_thread_bound_and_port_bound() -> None:
    owner, _ = _refreshed_system(1, 1)
    other_owner, _ = _refreshed_system(7, 7)
    _, desired = _refreshed_system(8, 8)
    port = AttentionStatePort(owner)
    other_port = AttentionStatePort(other_owner)
    with pytest.raises(TypeError, match="created by their state port"):
        AttentionRestoreTransaction()  # type: ignore[call-arg]

    transaction = port.prepare_attention_restore(desired)
    assert transaction._port is port
    assert transaction._owner is owner
    with pytest.raises(TypeError):
        transaction.publish(other_port)  # type: ignore[call-arg]

    cross_thread_errors: list[BaseException] = []

    def misuse_from_foreign_thread() -> None:
        try:
            transaction.publish()
        except BaseException as error:
            cross_thread_errors.append(error)
        try:
            transaction.close()
        except BaseException as error:
            cross_thread_errors.append(error)

    worker = Thread(target=misuse_from_foreign_thread)
    worker.start()
    worker.join(2)
    assert not worker.is_alive()
    assert len(cross_thread_errors) == 2
    assert all(isinstance(error, RuntimeError) for error in cross_thread_errors)
    transaction.close()
    assert other_owner.snapshot().canonical_bytes() != desired.canonical_bytes()


def test_full_schema_snapshot_survives_port_restore_without_truncation() -> None:
    owner = AttentionSystem()
    maximum = maximum_attention_continuity_fixture()
    transaction = AttentionStatePort(owner).prepare_attention_restore(maximum)
    transaction.publish()
    transaction.complete()
    transaction.close()

    restored = owner.snapshot()
    assert len(restored.candidates) == 4_192
    assert restored.canonical_bytes() == maximum.canonical_bytes()
    assert owner.selected_view().competition is None
    assert owner.selected_view().prompt is None
