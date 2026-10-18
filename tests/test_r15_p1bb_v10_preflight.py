"""P1b-B TEST ONLY: v10 capacity and source-continuity serialization.

These checks do NOT wire or claim real AgentState v10, StateWAL recovery or
durable subject authority. They use disposable actual source-owner fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.r15_p1.continuity_preflight import (
    FIELD,
    PreflightRefused,
    checkpoint_full_legal_max_bytes,
    continuity_envelope,
    projected_v10_with_continuity_bytes,
    require_preflight_capacity,
    validate_continuity_envelope,
)
from suzka.r15.common import canonical_json
from suzka.r15.contracts import NarrativeSelfState
from suzka.r15.narrative_self_system import NarrativeSelfSystem
from test_r15_p1b_positive import _completed_fixture


def _snapshot(
    tmp_path: Path,
    *,
    no_reveal: bool = False,
) -> tuple[NarrativeSelfState, bytes]:
    for client, _runtime, owner, refs, initial, _final, *_others in _completed_fixture(
        tmp_path, no_reveal=no_reveal
    ):
        root = owner.snapshot()
        assert root.revision == (1 if no_reveal else 2)
        context_id = client.app.state.main_loop.context_registry.current_context_id
        assert context_id
        encoded = continuity_envelope(
            root,
            source_scope=initial.source_scope,
            context_id=context_id,
            first_sources=refs[:3],
            revised_sources=() if no_reveal else refs,
        )
        assert (
            validate_continuity_envelope(encoded, root)["root_revision"]
            == root.revision
        )
    # The actual app has been closed; this is only detached data, not U3 restore.
    return root, encoded


def test_full_legal_v10_budget_reserves_all_16_mib() -> None:
    proof = require_preflight_capacity()
    assert proof["untouched_reserve_bytes"] == 16 * 1024 * 1024
    assert proof["hard_cap_bytes"] == 128 * 1024 * 1024
    assert 0 < proof["private_checkpoint_max_bytes"] < 32 * 1024
    assert proof["headroom_beyond_reserve_bytes"] > 0
    assert proof["v10_with_private_checkpoint_and_full_16MiB_reserve_bytes"] == (
        projected_v10_with_continuity_bytes()
    )
    assert (
        proof["v10_with_private_checkpoint_and_full_16MiB_reserve_bytes"]
        > proof["full_r15_only_with_reserve_bytes"]
    )
    assert FIELD == "narrative_admission_checkpoint"


@pytest.mark.parametrize("no_reveal", [False, True])
def test_actual_disposable_owner_root_and_refs_survive_canonical_readback(
    tmp_path: Path,
    no_reveal: bool,
) -> None:
    root, encoded = _snapshot(tmp_path, no_reveal=no_reveal)
    value = validate_continuity_envelope(encoded, root)
    assert value["stage"] == ("S1" if no_reveal else "S2")
    assert value["root_revision"] == (1 if no_reveal else 2)
    assert value["source_scope"] == value["first_sources"][0]["episode_id"]
    if no_reveal:
        assert value["revised_sources"] == []
    assert len(value["first_sources"]) == 3
    assert len(value["revised_sources"]) == (0 if no_reveal else 5)
    assert len(encoded) <= checkpoint_full_legal_max_bytes()
    detached = NarrativeSelfState.from_json(root.canonical_bytes())
    assert detached == root
    assert validate_continuity_envelope(encoded, detached) == value
    assert NarrativeSelfSystem().snapshot().revision == 0
    # A brand-new owner is NOT rehydrated by the detached preflight artifact.


def test_empty_v9_to_v10_bootstrap_has_no_invented_narrative() -> None:
    root = NarrativeSelfState(1, 0, (), (), None, ())
    encoded = continuity_envelope(root, source_scope=None, context_id=None)
    data = validate_continuity_envelope(encoded, root)
    assert data["stage"] == "EMPTY"
    assert data["first_sources"] == data["revised_sources"] == []
    assert data["source_scope"] is data["context_id"] is None
    with pytest.raises(PreflightRefused):
        continuity_envelope(root, source_scope="invented", context_id=None)


def test_wrong_scope_or_corrupted_root_does_not_bind_to_a_new_owner(
    tmp_path: Path,
) -> None:
    root, encoded = _snapshot(tmp_path)
    data = json.loads(encoded)
    data["source_scope"] = "different-staged-episode"
    with pytest.raises(PreflightRefused):
        validate_continuity_envelope(canonical_json(data), root)
    empty = NarrativeSelfState(1, 0, (), (), None, ())
    with pytest.raises(PreflightRefused):
        validate_continuity_envelope(encoded, empty)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"stage": "S1"}),
        lambda d: d.update({"root_revision": 1}),
        lambda d: d.update({"root_digest": "f" * 64}),
        lambda d: d.update({"last_review_event_id": "forged"}),
        lambda d: d["revised_sources"].reverse(),
        lambda d: d["first_sources"][0].update({"event_sequence": True}),
        lambda d: d["first_sources"][0].update({"episode_id": "../source"}),
        lambda d: d["revised_sources"][2].update({"event_id": "missing-event"}),
        lambda d: d.update({"untrusted_operator_set_trust": 1}),
    ],
)
def test_tampered_checkpoint_refused_before_any_real_state_change(
    tmp_path: Path,
    mutate,
) -> None:
    root, encoded = _snapshot(tmp_path)
    data = json.loads(encoded)
    mutate(data)
    with pytest.raises(PreflightRefused):
        validate_continuity_envelope(canonical_json(data), root)
    assert root.revision == 2
    assert len(root.episodes) == 2


def test_noncanonical_duplicate_json_keys_and_oversize_are_refused(
    tmp_path: Path,
) -> None:
    root, encoded = _snapshot(tmp_path)
    assert encoded.startswith(b"{")
    with pytest.raises(PreflightRefused):
        validate_continuity_envelope(
            encoded.replace(b'"stage":"S2"', b'"stage":"S2","stage":"S1"'),
            root,
        )
    with pytest.raises(PreflightRefused):
        validate_continuity_envelope(b" " + encoded, root)
    with pytest.raises(PreflightRefused):
        validate_continuity_envelope(
            b"x" * (checkpoint_full_legal_max_bytes() + 1), root
        )


def test_a_new_nonempty_checkpoint_cannot_claim_review_order_before_sources(
    tmp_path: Path,
) -> None:
    root, encoded = _snapshot(tmp_path)
    value = json.loads(encoded)
    value["last_review_event_sequence"] = 1
    with pytest.raises(PreflightRefused):
        validate_continuity_envelope(canonical_json(value), root)


def test_preflight_does_not_act_as_a_real_agent_state_codec(tmp_path: Path) -> None:
    root, encoded = _snapshot(tmp_path)
    parsed = validate_continuity_envelope(encoded, root)
    assert "saved_at" not in parsed
    assert "event_journal" not in parsed
    assert "raw_prompt" not in parsed
    assert "source_set_verified" not in parsed
    assert not hasattr(NarrativeSelfSystem(), "restore_snapshot")
