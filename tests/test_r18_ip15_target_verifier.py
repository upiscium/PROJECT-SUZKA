"""IP-15: the fixed target and the independent verifier, on real files.

Every intent, approval and claim here is a FIXTURE: nothing comes from a
real Decision, operator or Runtime.
"""

from __future__ import annotations

from suzka.r18.contracts import ApprovalChannel

from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from suzka.r16.successor_policy import InverseAuthorization
from suzka.r17.contracts import ObservedState
from suzka.r18.contracts import (
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    ExecutorResult,
    InverseIntentV1,
    VerifiedState,
    operation_key,
    payload_digest,
)
from suzka.r18.target import (
    SCHEMA_STATEMENTS,
    ArtifactState,
    LedgerKind,
    TargetBindingError,
    TargetCapacityError,
    TargetCorrupt,
    TargetReader,
    TargetStore,
    TargetUnavailable,
    new_runtime_instance_id,
)
from suzka.r18.verifier import verify_forward, verify_inverse

from test_r18_ip13_contracts import (
    BOOT,
    EVENT_V,
    GROUP,
    NONCE,
    _allow,
    _artifact,
    _intent,
    _inverse,
    _outcome,
)

RUNTIME = "00000000-0000-4000-8000-0000000000cc"


def _NOW() -> datetime:
    return datetime(2026, 11, 16, 0, 6, 0, tzinfo=timezone.utc)


CLAIM_EVENT = "00000000-0000-4000-8000-000000000003"


def _store(tmp_path: Path, **kw: Any) -> TargetStore:
    root = tmp_path / "target"
    root.mkdir(mode=0o700)
    store = TargetStore.create(root / "target.sqlite3", target_namespace="review-artifact", **kw)
    store.rotate_runtime_instance(RUNTIME)
    return store


def _reader(store: TargetStore) -> TargetReader:
    return TargetReader.open(store.path)


def _forward(store: TargetStore, group: str = GROUP, nonce: str = NONCE) -> tuple[ActionIntentV1, ApprovalChallengeV1, ExecutionClaimV1]:
    meta = store.meta()
    intent = _intent(
        target_instance_id=meta.target_instance_id,
        artifact=replace(_artifact(), obligation_group_key=group),
    )
    challenge = ApprovalChallengeV1.for_forward(
        intent, "e" * 64, _allow(intent),
        nonce=nonce, issued_utc="2026-11-16T00:00:00Z", expires_utc="2026-11-16T00:10:00Z", boot_id=BOOT, commitment_deadline_utc=None,
    )
    claim = ExecutionClaimV1.claim(
        challenge, runtime_instance_id=RUNTIME, boot_id=BOOT,
        claim_event_id=CLAIM_EVENT, claim_sequence=22, claimed_utc="2026-11-16T00:05:00Z", approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
    )
    return intent, challenge, claim


def _inverse_for(store: TargetStore, intent: ActionIntentV1) -> tuple[InverseIntentV1, ApprovalChallengeV1, ExecutionClaimV1]:
    forward = _outcome()
    inverse = _inverse(
        target_instance_id=intent.target_instance_id,
        obligation_group_key=intent.obligation_group_key,
        forward_outcome_id=forward.verified_outcome_id,
        forward_outcome_digest=forward.outcome_digest,
        forward_payload_digest=intent.payload_digest,
        target_expected_revision=1,
    )
    challenge = ApprovalChallengeV1.for_inverse(
        inverse, _allow(inverse),
        nonce="7" * 64, issued_utc="2026-11-16T01:00:00Z", expires_utc="2026-11-16T01:10:00Z", boot_id=BOOT,
    )
    claim = ExecutionClaimV1.claim(
        challenge, runtime_instance_id=RUNTIME, boot_id=BOOT,
        claim_event_id=CLAIM_EVENT, claim_sequence=32, claimed_utc="2026-11-16T01:05:00Z", approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
    )
    return inverse, challenge, claim


def _verify(store: TargetStore, intent: ActionIntentV1, challenge: ApprovalChallengeV1, consumed: str | None = NONCE) -> Any:
    reader = _reader(store)
    try:
        return verify_forward(
            reader, intent, challenge,
            consumed_nonce=consumed, outcome_revision=1, verifier_event_id=EVENT_V, verifier_sequence=23,
        )
    finally:
        reader.close()


_GUARDS = {
    "target_artifacts_no_delete": next(s for s in SCHEMA_STATEMENTS if "target_artifacts_no_delete" in s),
    "target_artifacts_revert_only": next(s for s in SCHEMA_STATEMENTS if "target_artifacts_revert_only" in s),
}


def _outside(store: TargetStore, *statements: str, parameters: tuple[object, ...] = ()) -> None:
    """An outside writer that bypasses the ledger and then restores the schema,
    so only the data shows the change (R20-F09)."""
    connection = sqlite3.connect(store.path, isolation_level=None)
    try:
        for name in _GUARDS:
            connection.execute(f"DROP TRIGGER {name}")
        for statement in statements:
            connection.execute(statement, parameters)
        for sql in _GUARDS.values():
            connection.execute(sql)
    finally:
        connection.close()


# ------------------------------------------------------------ open / custody


def test_create_is_explicit_and_open_never_creates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    with pytest.raises(TargetCorrupt, match="already exists"):
        TargetStore.create(store.path, target_namespace="review-artifact")
    store.close()
    missing = store.path.with_name("other.sqlite3")
    with pytest.raises(TargetUnavailable):
        TargetStore.open(missing)
    with pytest.raises(TargetUnavailable):
        TargetReader.open(missing)
    assert not missing.exists()  # a lost target is never replaced by an empty one
    reopened = TargetStore.open(store.path)
    assert reopened.meta().runtime_instance_id == RUNTIME


def test_custody_and_schema_are_checked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.close()
    os.chmod(store.path, 0o644)
    with pytest.raises(TargetCorrupt, match="custody"):
        TargetStore.open(store.path)
    os.chmod(store.path, 0o600)
    connection = sqlite3.connect(store.path)
    connection.execute("CREATE TABLE extra (x INTEGER)")
    connection.commit()
    connection.close()
    with pytest.raises(TargetCorrupt, match="schema"):
        TargetReader.open(store.path)
    with pytest.raises(TargetCorrupt):
        TargetStore.open(store.path, max_page_count=10)


def test_a_dropped_guard_is_corruption(tmp_path: Path) -> None:
    store = _store(tmp_path)
    connection = sqlite3.connect(store.path, isolation_level=None)
    connection.execute("DROP TRIGGER target_operations_no_delete")
    connection.close()
    with pytest.raises(TargetCorrupt, match="schema"):
        TargetReader.open(store.path)


def test_reader_is_read_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    reader = _reader(store)
    with pytest.raises(sqlite3.OperationalError):
        reader._connection.execute("INSERT INTO target_operations (operation_key, obligation_group_key, direction, kind) VALUES (?, ?, 'FORWARD', 'FENCED')", ("1" * 64, "2" * 64))


def test_ledger_and_effects_are_append_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.forward_write(intent, challenge, claim, now=_NOW)
    connection = sqlite3.connect(store.path, isolation_level=None)
    for statement in (
        "UPDATE target_operations SET kind = 'FENCED'",
        "DELETE FROM target_operations",
        "DELETE FROM target_artifacts",
        "UPDATE target_artifacts SET payload = x'00'",
        "UPDATE target_meta SET target_namespace = 'other'",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(statement)


# ------------------------------------------------------------------ forward


def test_forward_write_and_authorized_verification(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    receipt = store.forward_write(intent, challenge, claim, now=_NOW)
    assert (receipt.executor_result, receipt.reported_revision, receipt.reported_payload_digest) == (
        ExecutorResult.APPLIED, 1, intent.payload_digest,
    )
    status = store.status(GROUP)
    assert status.artifact is not None and status.artifact.payload == intent.artifact.payload_bytes()
    assert status.forward is not None and status.forward.approval_ref == claim.nonce
    outcome = _verify(store, intent, challenge)
    assert outcome is not None
    assert (outcome.outcome_state, outcome.observed_payload_digest, outcome.effect_operation_key, outcome.foreign_change) == (
        VerifiedState.ACTIVE_VERIFIED_AUTHORIZED, intent.payload_digest, intent.forward_key, False,
    )
    assert outcome.to_r17_view().observed_state is ObservedState.ACTIVE


def test_double_execution_is_suppressed_across_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    assert store.forward_write(intent, challenge, claim, now=_NOW).executor_result is ExecutorResult.APPLIED
    assert store.forward_write(intent, challenge, claim, now=_NOW).executor_result is ExecutorResult.CAS_REJECTED
    store.close()
    again = TargetStore.open(store.path)
    # A new Decision of the same obligation keeps the forward key: still refused.
    renewed, renewed_challenge, renewed_claim = _forward(again, nonce="8" * 64)
    renewed = replace(renewed, selected_candidate_digest="9" * 64)
    renewed_challenge = ApprovalChallengeV1.for_forward(
        renewed, "e" * 64, _allow(renewed),
        nonce="8" * 64, issued_utc="2026-11-16T00:00:00Z", expires_utc="2026-11-16T00:10:00Z", boot_id=BOOT, commitment_deadline_utc=None,
    )
    renewed_claim = ExecutionClaimV1.claim(
        renewed_challenge, runtime_instance_id=RUNTIME, boot_id=BOOT,
        claim_event_id=CLAIM_EVENT, claim_sequence=40, claimed_utc="2026-11-16T00:05:00Z", approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
    )
    assert again.forward_write(renewed, renewed_challenge, renewed_claim, now=_NOW).executor_result is ExecutorResult.CAS_REJECTED
    assert [row.kind for row in again.ledger()] == [LedgerKind.EFFECT]


def test_the_written_bytes_must_be_the_approved_bytes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    wrong = replace(challenge, payload_digest="0" * 64)
    wrong_claim = ExecutionClaimV1.claim(
        wrong, runtime_instance_id=RUNTIME, boot_id=BOOT,
        claim_event_id=CLAIM_EVENT, claim_sequence=22, claimed_utc="2026-11-16T00:05:00Z", approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
    )
    with pytest.raises(TargetBindingError, match="approve"):
        store.forward_write(intent, wrong, wrong_claim, now=_NOW)
    with pytest.raises(TargetBindingError, match="claim"):
        store.forward_write(intent, challenge, replace(claim, nonce="0" * 64), now=_NOW)
    other_target = replace(intent, target_instance_id="00000000-0000-4000-8000-0000000000ff")
    other_challenge = ApprovalChallengeV1.for_forward(
        other_target, "e" * 64, _allow(other_target),
        nonce=NONCE, issued_utc="2026-11-16T00:00:00Z", expires_utc="2026-11-16T00:10:00Z", boot_id=BOOT, commitment_deadline_utc=None,
    )
    other_claim = ExecutionClaimV1.claim(
        other_challenge, runtime_instance_id=RUNTIME, boot_id=BOOT,
        claim_event_id=CLAIM_EVENT, claim_sequence=22, claimed_utc="2026-11-16T00:05:00Z", approval_channel=ApprovalChannel.LOCAL_LOOPBACK_TYPED_CODE_V1,
    )
    with pytest.raises(TargetBindingError, match="another target"):
        store.forward_write(other_target, other_challenge, other_claim, now=_NOW)
    assert store.ledger() == ()


# ---------------------------------------------------------- fence / late CAS


def test_fence_makes_a_late_cas_fail_and_verifies_absent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    assert _verify(store, intent, challenge) is None  # no effect, no fence: undecidable
    fenced = store.fence(GROUP, Direction.FORWARD)
    assert fenced.kind is LedgerKind.FENCED and fenced.operation_key == intent.forward_key
    assert store.fence(GROUP, Direction.FORWARD) == fenced  # idempotent
    assert store.forward_write(intent, challenge, claim, now=_NOW).executor_result is ExecutorResult.FENCED
    outcome = _verify(store, intent, challenge)
    assert outcome is not None and outcome.outcome_state is VerifiedState.VERIFIED_ABSENT
    assert outcome.observed_state is ObservedState.ABSENT and store.status(GROUP).artifact is None


def test_fencing_an_applied_key_reports_the_effect(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.forward_write(intent, challenge, claim, now=_NOW)
    assert store.fence(GROUP, Direction.FORWARD).kind is LedgerKind.EFFECT


def test_a_late_cas_after_rotation_and_fence_is_definitely_not_executed(tmp_path: Path) -> None:
    """The full sweep order: rotate, then fence. A CAS from the old process
    is CAS_REJECTED (the instance check runs first), never IN_DOUBT, and the
    verifier then sees VERIFIED_ABSENT."""

    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.rotate_runtime_instance(new_runtime_instance_id())
    store.fence(GROUP, Direction.FORWARD)
    assert store.forward_write(intent, challenge, claim, now=_NOW).executor_result is ExecutorResult.CAS_REJECTED
    outcome = _verify(store, intent, challenge)
    assert outcome is not None and outcome.outcome_state is VerifiedState.VERIFIED_ABSENT


def test_rotation_needs_a_new_value(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(TargetBindingError, match="new runtime_instance_id"):
        store.rotate_runtime_instance(RUNTIME)


def test_two_connections_racing_apply_once(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = _store(tmp_path)
    other = TargetStore.open(store.path)
    intent, challenge, claim = _forward(store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda target: target.forward_write(intent, challenge, claim, now=_NOW).executor_result, (store, other)))
    assert sorted(results) == sorted([ExecutorResult.APPLIED, ExecutorResult.CAS_REJECTED])
    assert [row.kind for row in store.ledger()] == [LedgerKind.EFFECT]


def test_rotated_instance_rejects_an_older_process(tmp_path: Path) -> None:
    """#380 D2 P2-1: a CAS from before the sweep's rotation fails definitely."""

    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.rotate_runtime_instance(new_runtime_instance_id())
    receipt = store.forward_write(intent, challenge, claim, now=_NOW)
    assert receipt.executor_result is ExecutorResult.CAS_REJECTED
    assert store.ledger() == () and store.status(GROUP).artifact is None


# ------------------------------------------------- verifier vs receipt / foreign


def test_without_a_consumed_nonce_the_effect_is_unauthorized(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    assert store.forward_write(intent, challenge, claim, now=_NOW).executor_result is ExecutorResult.APPLIED
    for consumed in (None, "0" * 64):
        outcome = _verify(store, intent, challenge, consumed)
        assert outcome is not None
        assert (outcome.outcome_state, outcome.authorization) == (
            VerifiedState.UNAUTHORIZED_EFFECT, InverseAuthorization.UNPROVEN,
        )


def test_authorization_needs_this_challenges_nonce(tmp_path: Path) -> None:
    """Review P2-1: the operation key is stable across Decisions, so a later
    challenge for the same key must not inherit an earlier approval."""

    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.forward_write(intent, challenge, claim, now=_NOW)
    later = replace(challenge, nonce="9" * 64)
    outcome = _verify(store, intent, later, consumed=NONCE)
    assert outcome is not None and outcome.outcome_state is VerifiedState.UNAUTHORIZED_EFFECT
    outcome = _verify(store, intent, later, consumed="9" * 64)
    assert outcome is not None and outcome.outcome_state is VerifiedState.UNAUTHORIZED_EFFECT


def test_receipt_says_applied_but_the_effect_is_gone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    receipt = store.forward_write(intent, challenge, claim, now=_NOW)
    assert receipt.executor_result is ExecutorResult.APPLIED
    _outside(store, "DELETE FROM target_artifacts")
    assert _verify(store, intent, challenge) is None  # never ABSENT from a receipt alone


def test_an_effect_without_a_ledger_row_is_not_ours(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    _outside(
        store,
        "INSERT INTO target_artifacts (obligation_group_key, state, revision, payload) VALUES (?, 'ACTIVE', 1, ?)",
        parameters=(GROUP, intent.artifact.payload_bytes()),
    )
    assert store.forward_write(intent, challenge, claim, now=_NOW).executor_result is ExecutorResult.CAS_REJECTED
    assert _verify(store, intent, challenge) is None
    store.fence(GROUP, Direction.FORWARD)
    assert _verify(store, intent, challenge) is None  # fenced but an artifact exists: mixed


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE target_artifacts SET payload = x'7b7d'",
        "UPDATE target_artifacts SET revision = 2",
    ],
)
def test_an_outside_change_is_foreign_and_unauthorized(tmp_path: Path, statement: str) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.forward_write(intent, challenge, claim, now=_NOW)
    _outside(store, statement)
    outcome = _verify(store, intent, challenge)
    assert outcome is not None
    assert (outcome.outcome_state, outcome.foreign_change) == (VerifiedState.UNAUTHORIZED_EFFECT, True)


# ------------------------------------------------------------------ inverse


def test_inverse_reverts_once_and_verifies(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.forward_write(intent, challenge, claim, now=_NOW)
    inverse, inverse_challenge, inverse_claim = _inverse_for(store, intent)
    receipt = store.inverse_write(inverse, inverse_challenge, inverse_claim)
    assert (receipt.executor_result, receipt.reported_revision, receipt.reported_payload_digest) == (ExecutorResult.APPLIED, 2, None)
    assert store.inverse_write(inverse, inverse_challenge, inverse_claim).executor_result is ExecutorResult.CAS_REJECTED
    status = store.status(GROUP)
    # The bytes are kept so the verifier can recompute their digest.
    assert status.artifact is not None and (status.artifact.state, status.artifact.payload) == (
        ArtifactState.REVERTED, intent.artifact.payload_bytes(),
    )
    assert status.forward is not None and status.forward.kind is LedgerKind.EFFECT  # forward receipt kept
    reader = _reader(store)
    for consumed, authorization in ((inverse_claim.nonce, InverseAuthorization.AUTHORIZED), (None, InverseAuthorization.UNPROVEN)):
        outcome = verify_inverse(
            reader, inverse, inverse_challenge,
            consumed_nonce=consumed, outcome_revision=1, verifier_event_id=EVENT_V, verifier_sequence=33,
        )
        assert outcome is not None
        assert (outcome.outcome_state, outcome.authorization, outcome.observed_state) == (
            VerifiedState.REVERTED, authorization, ObservedState.ABSENT,
        )
    # The forward cannot be written again after a revert.
    assert store.forward_write(intent, challenge, claim, now=_NOW).executor_result is ExecutorResult.CAS_REJECTED


def test_fenced_inverse_verifies_absent_and_keeps_the_forward(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, challenge, claim = _forward(store)
    store.forward_write(intent, challenge, claim, now=_NOW)
    inverse, inverse_challenge, inverse_claim = _inverse_for(store, intent)
    assert store.fence(GROUP, Direction.INVERSE).kind is LedgerKind.FENCED
    assert store.inverse_write(inverse, inverse_challenge, inverse_claim).executor_result is ExecutorResult.FENCED
    reader = _reader(store)
    outcome = verify_inverse(
        reader, inverse, inverse_challenge,
        consumed_nonce=inverse_claim.nonce, outcome_revision=1, verifier_event_id=EVENT_V, verifier_sequence=33,
    )
    assert outcome is not None
    assert (outcome.outcome_state, outcome.observed_state, outcome.effect_operation_key) == (
        VerifiedState.INVERSE_VERIFIED_ABSENT, ObservedState.ACTIVE, intent.forward_key,
    )


def test_inverse_needs_our_forward_effect(tmp_path: Path) -> None:
    store = _store(tmp_path)
    intent, _challenge, _claim = _forward(store)
    _outside(
        store,
        "INSERT INTO target_artifacts (obligation_group_key, state, revision, payload) VALUES (?, 'ACTIVE', 1, ?)",
        parameters=(GROUP, intent.artifact.payload_bytes()),
    )
    inverse, inverse_challenge, inverse_claim = _inverse_for(store, intent)
    assert store.inverse_write(inverse, inverse_challenge, inverse_claim).executor_result is ExecutorResult.CAS_REJECTED


# ----------------------------------------------------------------- capacity


@pytest.mark.parametrize(("max_page_count", "reserve_pages"), [(64, 24), (512, 64)])
def test_forward_stops_at_the_reserve_fences_keep_the_inverse_room_and_every_undo_fits(
    tmp_path: Path, max_page_count: int, reserve_pages: int
) -> None:
    store = _store(tmp_path, max_page_count=max_page_count, reserve_pages=reserve_pages)
    applied: list[ActionIntentV1] = []
    with pytest.raises(TargetCapacityError):
        for index in range(10_000):
            group = payload_digest(index.to_bytes(4, "big"))
            intent, challenge, claim = _forward(store, group=group)
            store.forward_write(intent, challenge, claim, now=_NOW)
            applied.append(intent)
    assert applied
    effects = len(store.ledger())
    # The refused forward left nothing behind (checked before COMMIT).
    assert effects == len(applied)
    # The sweep can fence into the reserve until the inverse floor.
    fences = 0
    with pytest.raises(TargetCapacityError, match="room kept for inverses"):
        for index in range(100_000):
            store.fence(payload_digest(b"fence" + index.to_bytes(4, "big")), Direction.FORWARD)
            fences += 1
    assert fences > 0
    # Every applied effect can still be undone.
    for intent in applied:
        inverse, inverse_challenge, inverse_claim = _inverse_for(store, intent)
        assert store.inverse_write(inverse, inverse_challenge, inverse_claim).executor_result is ExecutorResult.APPLIED
    assert operation_key(GROUP, Direction.FORWARD) not in {row.operation_key for row in store.ledger()}
