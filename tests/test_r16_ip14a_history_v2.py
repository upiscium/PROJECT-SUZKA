"""IP-14a-1: SubjectHistory v2 schema, v1 -> v2 migration, record operations.

Design: #411 comment 6910958369 (IP-14a D1 r3). Every R17 / R18 value here
is a FIXTURE ONLY; nothing comes from a real Decision, operator or target.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from suzka.r16 import history_migration
from suzka.r16.history_codec import PolicyKind, PostcommitReceiptV1
from suzka.r16.history_migration import HistoryMigrationError, migrate_v1_to_v2
from suzka.r16.history_records import (
    MAX_CHALLENGES_PER_INTENT,
    ApprovalExpirationV1,
    ExpirationReason,
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_schema_v1 import SCHEMA_HASH_V1, SCHEMA_STATEMENTS_V1
from suzka.r16.history_store import (
    _SCHEMA_QUERY,
    SCHEMA_STATEMENTS,
    HistoryStoreConflict,
    HistoryStoreCorrupt,
    HistoryStoreMigrationRequired,
    HistoryStoreRecordsDisabled,
    SubjectHistoryStore,
    _expected_schema,
)
from suzka.r17.contracts import (
    DecisionFeedbackV1,
    ExpectedObservationV1,
    OutcomeAssessmentV1,
)
from suzka.r18.contracts import (
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    GateResult,
    GateStatus,
    InverseIntentV1,
    PolicyDenyReason,
    PolicyVerdict,
    PolicyVerdictV1,
    ReviewArtifactV1,
    ReviewGate,
    ReviewSourceRef,
    SourceRole,
    VerifiedState,
)

from test_r16_ip9_subject_history import (
    INCARNATION,
    PAGES,
    FakeProof,
    _binding,
    _committed,
    _event,
    _private_dir,
)
from test_r18_ip13_contracts import GROUP, _outcome

TARGET = "00000000-0000-4000-8000-0000000000aa"
BOOT = "00000000-0000-4000-8000-0000000000bb"
RUNTIME = "00000000-0000-4000-8000-0000000000cc"
H = "a" * 64


def _uuid(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, name))


def _receipt(transaction_id: str, event_id: str, sequence: int) -> PostcommitReceiptV1:
    return PostcommitReceiptV1(
        transaction_id=transaction_id,
        event_id=event_id,
        processing_sequence=sequence,
        prepared_state_hash_before="3" * 64,
        prepared_state_hash_after="4" * 64,
        snapshot_sequence=sequence,
        snapshot_hash="4" * 64,
        wal_generation_id=_uuid("generation"),
        wal_record_id=_uuid(f"record-{transaction_id}"),
        wal_record_hash="5" * 64,
    )


# ------------------------------------------------------------------ stores


def _v2_store(tmp_path: Path, *, r18: bool = True, pages: int = PAGES) -> SubjectHistoryStore:
    store = SubjectHistoryStore.create(
        _private_dir(tmp_path) / "subject_history.sqlite3",
        subject_id="suzka",
        journal_lineage_id="lineage-1",
        max_page_count=pages,
        r18_enabled=r18,
    )
    store.activate(adopted_baseline_id="baseline-1", adopted_baseline_hash=H, genesis_incarnation_id=INCARNATION)
    return store


def _decision(
    store: SubjectHistoryStore, name: str = "decision", sequence: int = 3, predecessor: Any = None
) -> Any:
    """A FINALIZED R16 Decision through the real participant."""

    from suzka.r16.history_codec import GENESIS
    from suzka.r16.history_participant import SubjectHistoryParticipant

    event = _event(name, sequence)
    decision = _committed(
        event,
        intent=f"{sequence:x}".rjust(64, "e"),
        predecessor=(GENESIS, None)
        if predecessor is None
        else (predecessor.decision_id, predecessor.content_digest),
    )
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof())
    binding = _binding(event, decision)
    participant.prepare(binding)
    participant.finalize(binding)
    return decision


def _downgrade_to_v1(path: Path) -> None:
    """Rewrite a v2 store that holds only v1 rows as an exact v1 file (a
    stand-in for a production v1 store, with real IP-9 rows)."""

    source = sqlite3.connect(path)
    tables = ("history_meta", "history_operations", "history_decisions", "history_source_refs", "history_postcommit_receipts")
    rows = {table: source.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() for table in tables}
    source.close()
    os.unlink(path)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    target = sqlite3.connect(path, isolation_level=None)
    target.execute("PRAGMA journal_mode = DELETE")
    target.execute("BEGIN")
    creates = [s for s in SCHEMA_STATEMENTS_V1 if s.startswith("CREATE TABLE")]
    for statement in creates:
        target.execute(statement)
    for table in tables:
        for row in rows[table]:
            values = list(row)
            if table == "history_meta":
                values[1], values[2] = 1, SCHEMA_HASH_V1
            target.execute(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in values)})", values)
    for statement in SCHEMA_STATEMENTS_V1:
        if not statement.startswith("CREATE TABLE"):
            target.execute(statement)
    target.execute("COMMIT")
    target.close()


def _v1_store_with_decisions(tmp_path: Path) -> tuple[Path, list[Any]]:
    store = _v2_store(tmp_path, r18=False)
    decisions: list[Any] = []
    for index in range(3):
        decisions.append(_decision(store, f"decision-{index}", 3 + index, decisions[-1] if decisions else None))
    path = store.path
    store.close()
    _downgrade_to_v1(path)
    return path, decisions


# ------------------------------------------------------------------ schema


def test_a_fresh_store_is_exactly_v2(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    connection = sqlite3.connect(store.path)
    assert connection.execute(_SCHEMA_QUERY).fetchall() == _expected_schema(SCHEMA_STATEMENTS)
    names = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
    assert {
        "history_operation_keys", "action_intents", "action_forecasts", "policy_verdicts",
        "approval_challenges", "approval_expirations", "approval_nonce_consumptions",
        "execution_claims", "verified_outcomes", "outcome_assessments", "decision_feedback",
    } <= names
    assert store.meta().schema_version == 2


def test_a_v1_file_requires_migration_and_a_tampered_file_is_corrupt(tmp_path: Path) -> None:
    path, _ = _v1_store_with_decisions(tmp_path)
    with pytest.raises(HistoryStoreMigrationRequired):
        SubjectHistoryStore.open(path, max_page_count=PAGES)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE extra (x INTEGER)")
    connection.commit()
    connection.close()
    with pytest.raises(HistoryStoreCorrupt):
        SubjectHistoryStore.open(path, max_page_count=PAGES)


def test_v3_decisions_are_admitted_by_the_schema(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    from suzka.r16.history_participant import SubjectHistoryParticipant

    event = _event("v3", 3)
    decision = _committed(event, kind=PolicyKind.SUCCESSOR_POLICY_V3)
    participant = SubjectHistoryParticipant(store, INCARNATION, decision, FakeProof())
    binding = _binding(event, decision)
    participant.prepare(binding)
    participant.finalize(binding)
    assert store.finalized_decision(INCARNATION, binding.transaction_id) is not None


# --------------------------------------------------------------- migration


def test_migration_keeps_every_v1_row_and_the_decisions_replay(tmp_path: Path) -> None:
    path, decisions = _v1_store_with_decisions(tmp_path)
    before = sqlite3.connect(path)
    rows = {t: before.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall() for t in ("history_operations", "history_decisions", "history_source_refs", "history_postcommit_receipts")}
    before.close()
    result = migrate_v1_to_v2(path)
    backup = result.backup
    assert result.compacted and os.stat(backup).st_mode & 0o777 == 0o600
    store = SubjectHistoryStore.open(path, max_page_count=PAGES)
    assert store.meta().schema_version == 2
    after = sqlite3.connect(path)
    assert after.execute(_SCHEMA_QUERY).fetchall() == _expected_schema(SCHEMA_STATEMENTS)
    for table, expected in rows.items():
        assert after.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() == expected, table
    for decision in decisions:
        operations = {row.event_id: row for row in store.operations(INCARNATION)}
        row = operations[decision.issuer_event_id]
        finalized = store.finalized_decision(INCARNATION, row.transaction_id)
        assert finalized is not None and finalized.committed_bytes == decision.canonical_bytes()
    # The backup is the untouched v1 file.
    with pytest.raises(HistoryStoreMigrationRequired):
        SubjectHistoryStore.open(backup, max_page_count=PAGES)


@pytest.mark.parametrize("stage", ["backed-up", "dropped", "rebuilt:history_meta", "created", "verified"])
def test_a_killed_process_leaves_v1_and_a_rerun_migrates(tmp_path: Path, stage: str) -> None:
    """Review P2-1: a real crash (os._exit, no ROLLBACK) leaves the v1 file
    and possibly its journal; the next run migrates without manual cleanup."""

    import subprocess
    import sys

    path, _ = _v1_store_with_decisions(tmp_path)
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from suzka.r16 import history_migration as m\n"
        f"m._STAGE_HOOK = lambda name: os._exit(9) if name == {stage!r} else None\n"
        f"m.migrate_v1_to_v2(Path({str(path)!r}))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], check=False)
    assert completed.returncode == 9
    with pytest.raises(HistoryStoreMigrationRequired):
        SubjectHistoryStore.open(path, max_page_count=PAGES)
    assert history_migration.main([str(path)]) == 0  # no manual cleanup needed
    assert SubjectHistoryStore.open(path, max_page_count=PAGES).meta().schema_version == 2


def test_a_failed_compaction_after_commit_is_not_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review P2-2: after COMMIT the store is v2 even if VACUUM fails."""

    path, _ = _v1_store_with_decisions(tmp_path)
    readers: list[sqlite3.Connection] = []

    def hold_a_read_lock(name: str) -> None:
        if name == "committed":
            reader = sqlite3.connect(path, isolation_level=None, timeout=0.1)
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM history_meta").fetchall()
            readers.append(reader)

    monkeypatch.setattr(history_migration, "_STAGE_HOOK", hold_a_read_lock)
    assert history_migration.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "migrated to SubjectHistory v2" in out and "compaction (VACUUM) failed" in out
    readers[0].close()
    assert SubjectHistoryStore.open(path, max_page_count=PAGES).meta().schema_version == 2


def test_migration_refuses_a_store_too_close_to_its_cap(tmp_path: Path) -> None:
    path, _ = _v1_store_with_decisions(tmp_path)
    pages = sqlite3.connect(path).execute("PRAGMA page_count").fetchone()[0]
    with pytest.raises(HistoryMigrationError, match="max_page_count"):
        migrate_v1_to_v2(path, max_page_count=pages + 10)
    with pytest.raises(HistoryStoreMigrationRequired):
        SubjectHistoryStore.open(path, max_page_count=PAGES)
    assert migrate_v1_to_v2(path, max_page_count=PAGES).compacted


def test_prepared_aborted_and_provisional_stores_migrate(tmp_path: Path) -> None:
    from suzka.r16.history_participant import SubjectHistoryParticipant

    store = _v2_store(tmp_path, r18=False)
    first = _decision(store, "first", 3)
    pending_event = _event("pending", 4)
    pending = _committed(pending_event, intent="9" * 64, predecessor=(first.decision_id, first.content_digest))
    SubjectHistoryParticipant(store, INCARNATION, pending, FakeProof()).prepare(_binding(pending_event, pending))
    aborted_event = _event("aborted", 5)
    aborted = _committed(aborted_event, intent="8" * 64, predecessor=(first.decision_id, first.content_digest))
    binding = _binding(aborted_event, aborted)
    store.insert_abort_tombstone(
        incarnation_id="2" * 64, transaction_id=binding.transaction_id, event_id=binding.event_id,
        processing_sequence=binding.processing_sequence, transaction_kind=binding.transaction_kind.value,
        operation_digest=binding.operation_digest,
    )
    path = store.path
    store.close()
    _downgrade_to_v1(path)
    migrate_v1_to_v2(path)
    phases = sorted(row.phase.value for row in SubjectHistoryStore.open(path, max_page_count=PAGES).operations(INCARNATION))
    assert phases == ["FINALIZED", "PREPARED"]

    provisional = SubjectHistoryStore.create(
        _private_dir(tmp_path / "provisional") / "subject_history.sqlite3",
        subject_id="suzka", journal_lineage_id="lineage-1", max_page_count=PAGES,
    )
    provisional_path = provisional.path
    provisional.close()
    _downgrade_to_v1(provisional_path)
    migrate_v1_to_v2(provisional_path)
    assert SubjectHistoryStore.open(provisional_path, max_page_count=PAGES).meta().phase.value == "PROVISIONAL"


@pytest.mark.parametrize("stage", ["backed-up", "dropped", "rebuilt:history_meta", "rebuilt:history_operations", "rebuilt:history_decisions", "created", "verified"])
def test_a_crash_before_commit_leaves_the_v1_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    path, _ = _v1_store_with_decisions(tmp_path)
    original = sqlite3.connect(path)
    snapshot = original.execute(_SCHEMA_QUERY).fetchall(), original.execute("SELECT * FROM history_decisions").fetchall()
    original.close()

    def crash(name: str) -> None:
        if name == stage:
            raise KeyboardInterrupt(name)

    monkeypatch.setattr(history_migration, "_STAGE_HOOK", crash)
    with pytest.raises(KeyboardInterrupt):
        migrate_v1_to_v2(path)
    monkeypatch.setattr(history_migration, "_STAGE_HOOK", None)
    check = sqlite3.connect(path)
    assert (check.execute(_SCHEMA_QUERY).fetchall(), check.execute("SELECT * FROM history_decisions").fetchall()) == snapshot
    check.close()
    with pytest.raises(HistoryStoreMigrationRequired):
        SubjectHistoryStore.open(path, max_page_count=PAGES)
    # It can still be migrated afterwards (a fresh backup name).
    from datetime import UTC, datetime

    migrate_v1_to_v2(path, now=datetime(2030, 1, 1, tzinfo=UTC))
    assert SubjectHistoryStore.open(path, max_page_count=PAGES).meta().schema_version == 2


def test_a_record_row_must_be_in_its_operations_envelope(tmp_path: Path) -> None:
    """Review P3-1: direct SQL cannot FINALIZE a record that is not the one its
    PREPARED operation carries."""

    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    steps = _steps(chain)
    _apply(store, steps[0])
    _apply(store, steps[1])
    transaction_id = _prepare(store, steps[2])
    other = replace(chain.verdict, policy_sequence=chain.seq(2) + 50)
    connection = sqlite3.connect(store.path, isolation_level=None)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO policy_verdicts (record_id, incarnation_id, transaction_id, record_bytes, record_digest, intent_id, direction, verdict) VALUES (?, ?, ?, ?, ?, ?, 'FORWARD', 'ALLOW')",
            (other.verdict_digest, INCARNATION, transaction_id, other.canonical_bytes(), other.verdict_digest, chain.intent.intent_id),
        )


def test_migration_refuses_what_is_not_exactly_v1(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    store.close()
    with pytest.raises(HistoryMigrationError, match="not exactly schema v1"):
        migrate_v1_to_v2(store.path)
    path, _ = _v1_store_with_decisions(tmp_path / "other")
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER history_decisions_no_delete")
    connection.commit()
    connection.close()
    with pytest.raises(HistoryMigrationError):
        migrate_v1_to_v2(path)


def test_the_migration_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path, _ = _v1_store_with_decisions(tmp_path)
    assert history_migration.main([str(path)]) == 0
    assert "v1 backup" in capsys.readouterr().out
    assert history_migration.main([str(path)]) == 1  # already v2: refused
    assert history_migration.main([]) == 2


# --------------------------------------------------------- record fixtures


class Chain:
    """One forward obligation through every R17 / R18 record kind."""

    def __init__(
        self,
        decision: Any,
        group: str = GROUP,
        offset: int = 0,
        *,
        incarnation: str = INCARNATION,
        base: int = 100,
        transaction_kind: str = "EVENT_MUTATION",
    ) -> None:
        self.decision = decision
        self.incarnation = incarnation
        self.transaction_kind = transaction_kind
        self.sequence = base + offset
        sources = tuple(
            ReviewSourceRef(role, decision.decision_id, 1, decision.content_digest)
            if role is SourceRole.DECISION
            else ReviewSourceRef(role, f"{role.value.lower()}-1", 1, H)
            for role in sorted(SourceRole)
        )
        artifact = ReviewArtifactV1(
            "review-artifact", group, sources, tuple(GateResult(g, GateStatus.PASS) for g in sorted(ReviewGate))
        )
        self.intent = ActionIntentV1(
            lineage_scope="x-1",
            incarnation_id=incarnation,
            admitting_event_id=self.event("intent"),
            admitting_sequence=self.seq(0),
            decision_id=decision.decision_id,
            decision_committed_digest=decision.content_digest,
            policy_kind=PolicyKind.SUCCESSOR_POLICY_V2,
            policy_row=decision.policy_row,
            selected_candidate_id="candidate-b",
            selected_candidate_digest=H,
            goal_origin=__import__("suzka.r18.contracts", fromlist=["GoalOrigin"]).GoalOrigin.USER_REQUEST,
            target_instance_id=TARGET,
            target_expected_revision=0,
            artifact=artifact,
        )
        self.forecast = ExpectedObservationV1(
            lineage_scope="x-1",
            incarnation_id=incarnation,
            admitting_event_id=self.event("forecast"),
            admitting_sequence=self.seq(1),
            as_of_r07_sequence=self.seq(0),
            decision_id=decision.decision_id,
            decision_committed_digest=decision.content_digest,
            policy_kind=PolicyKind.SUCCESSOR_POLICY_V2,
            selected_candidate_id="candidate-b",
            selected_candidate_digest=H,
            action_intent_id=self.intent.intent_id,
            action_intent_digest=self.intent.intent_digest,
            obligation_group_key=group,
            forward_key=self.intent.forward_key,
            target_instance_id=TARGET,
            target_namespace="review-artifact",
            target_expected_revision=0,
            predicted_payload_digest=self.intent.payload_digest,
        )
        self.verdict = PolicyVerdictV1(
            Direction.FORWARD, self.intent.intent_id, self.intent.intent_digest, self.intent.forward_key,
            PolicyVerdict.ALLOW, None, self.event("verdict"), self.seq(2),
        )
        self.challenges = [self._challenge(attempt) for attempt in range(1, MAX_CHALLENGES_PER_INTENT + 2)]
        self.claim = ExecutionClaimV1.claim(
            self.challenges[0], runtime_instance_id=RUNTIME, boot_id=BOOT,
            claim_event_id=self.event("claim"), claim_sequence=self.seq(4), claimed_utc="2026-11-16T00:05:00Z",
        )
        self.outcome = replace(
            _outcome(),
            operation_key=self.intent.forward_key,
            observed_payload_digest=self.intent.payload_digest,
            effect_operation_key=self.intent.forward_key,
            target_instance_id=TARGET,
            verifier_event_id=self.event("outcome"),
            verifier_sequence=self.seq(5),
        )
        self.assessment = OutcomeAssessmentV1.issue(
            self.forecast, self.outcome.to_r17_view(), self.event("assessment"), self.seq(6)
        )
        self.feedback = DecisionFeedbackV1.issue(self.assessment, self.event("feedback"), self.seq(7))
        self._offset = offset

    def seq(self, step: int) -> int:
        return self.sequence + step

    def event(self, name: str) -> str:
        return _uuid(f"{name}-{self.sequence}")

    def _challenge(self, attempt: int) -> ApprovalChallengeV1:
        return ApprovalChallengeV1.for_forward(
            self.intent, self.forecast.forecast_digest, self.verdict,
            nonce=f"{attempt:x}".rjust(64, "d") if self.sequence == 100 else f"{attempt:x}".rjust(64, "e"),
            issued_utc="2026-11-16T00:00:00Z", expires_utc="2026-11-16T00:10:00Z", boot_id=BOOT,
        )

    def operation(self, kind: HistoryOperationKind, record_kind: RecordKind, record: object, step: int, attempt: int | None = None) -> HistoryRecordOperationV1:
        event_id = getattr(record, "admitting_event_id", None) or getattr(record, "policy_event_id", None) or getattr(record, "claim_event_id", None) or getattr(record, "verifier_event_id", None) or self.event(f"{record_kind.value}-{step}")
        return HistoryRecordOperationV1(
            kind, self.incarnation, event_id, self.seq(step), self.transaction_kind, record_kind, record, attempt
        )


def _apply(store: SubjectHistoryStore, operation: HistoryRecordOperationV1) -> str:
    transaction_id = _uuid(f"tx-{operation.event_id}-{operation.processing_sequence}")
    store.insert_prepared_record(transaction_id=transaction_id, operation_digest=operation.operation_digest, operation=operation)
    store.finalize_record(
        transaction_id=transaction_id,
        operation=operation,
        receipt=_receipt(transaction_id, operation.event_id, operation.processing_sequence),
    )
    return transaction_id


def _refused_at_finalize(store: SubjectHistoryStore, operation: HistoryRecordOperationV1) -> None:
    """The FINALIZE gate refuses; abort so the reservation is released (the
    real writer's pre-check keeps this from happening after an R07 commit)."""
    transaction_id = _prepare(store, operation)
    with pytest.raises(HistoryStoreConflict):
        store.finalize_record(
            transaction_id=transaction_id,
            operation=operation,
            receipt=_receipt(transaction_id, operation.event_id, operation.processing_sequence),
        )
    store.mark_aborted(operation.incarnation_id, transaction_id)


def _prepare(store: SubjectHistoryStore, operation: HistoryRecordOperationV1) -> str:
    transaction_id = _uuid(f"tx-{operation.event_id}-{operation.processing_sequence}")
    store.insert_prepared_record(transaction_id=transaction_id, operation_digest=operation.operation_digest, operation=operation)
    return transaction_id


K = HistoryOperationKind
R = RecordKind


def _steps(chain: Chain) -> list[HistoryRecordOperationV1]:
    return [
        chain.operation(K.R18_INTENT_ISSUE, R.ACTION_INTENT, chain.intent, 0),
        chain.operation(K.R17_EXPECTATION_COMMIT, R.ACTION_FORECAST, chain.forecast, 1),
        chain.operation(K.R18_POLICY_COMMIT, R.POLICY_VERDICT, chain.verdict, 2),
        chain.operation(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, chain.challenges[0], 3, attempt=1),
        chain.operation(K.R18_EXECUTION_CLAIM, R.EXECUTION_CLAIM, chain.claim, 4),
        chain.operation(K.R18_OUTCOME_COMMIT, R.VERIFIED_OUTCOME, chain.outcome, 5),
        chain.operation(K.R17_ASSESSMENT_COMMIT, R.OUTCOME_ASSESSMENT, chain.assessment, 6),
        chain.operation(K.R17_FEEDBACK_COMMIT, R.DECISION_FEEDBACK, chain.feedback, 7),
    ]


# ------------------------------------------------------------- record ops


def test_a_full_forward_chain_finalizes_every_record(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    for operation in _steps(chain):
        transaction_id = _apply(store, operation)
        for row in operation.rows:
            stored = store.record_rows(row.table, INCARNATION, transaction_id)
            assert len(stored) == 1
            for name, value in row.columns:
                assert stored[0][name] == value
    assert store.has_record_operations()
    assert store.live_keys((f"operation:{chain.intent.forward_key}", f"claim:{chain.intent.forward_key}")) == {
        f"operation:{chain.intent.forward_key}", f"claim:{chain.intent.forward_key}",
    }


def test_the_envelope_round_trips_and_binds_its_event(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    for operation in _steps(chain):
        assert HistoryRecordOperationV1.from_bytes(operation.canonical_bytes()) == operation
    with pytest.raises(ValueError, match="another event"):
        replace(_steps(chain)[0], processing_sequence=999)
    with pytest.raises(ValueError, match="does not belong"):
        HistoryRecordOperationV1(K.R18_POLICY_COMMIT, INCARNATION, chain.event("x"), 1, "EVENT_MUTATION", R.ACTION_INTENT, chain.intent)
    with pytest.raises(ValueError, match="attempt"):
        chain.operation(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, chain.challenges[0], 3)
    with pytest.raises(ValueError):
        HistoryRecordOperationV1(K.R16_DECISION_ISSUE, INCARNATION, chain.event("x"), 1, "EVENT_MUTATION", R.ACTION_INTENT, chain.intent)


def test_records_are_refused_while_r18_is_disabled(tmp_path: Path) -> None:
    store = _v2_store(tmp_path, r18=False)
    chain = Chain(_decision(store))
    with pytest.raises(HistoryStoreRecordsDisabled):
        _prepare(store, _steps(chain)[0])
    assert not store.has_record_operations()


def test_uniqueness_is_reserved_at_prepare_and_across_incarnations(tmp_path: Path) -> None:
    """§2a: a second operation for the same key is refused at PREPARE, before
    R07 could commit it, even from another incarnation."""

    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    first = _steps(chain)[0]
    _prepare(store, first)
    again = replace(first, event_id=_uuid("again"), processing_sequence=500, record=replace(chain.intent, admitting_event_id=_uuid("again"), admitting_sequence=500))
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, again)
    other = replace(again, incarnation_id="2" * 64, record=replace(again.record, incarnation_id="2" * 64))
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, other)


def test_abort_releases_the_reservation(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    first = _steps(chain)[0]
    transaction_id = _prepare(store, first)
    store.mark_aborted(INCARNATION, transaction_id)
    assert store.live_keys(first.reserved_keys) == frozenset()
    retry = replace(first, event_id=_uuid("retry"), processing_sequence=600, record=replace(chain.intent, admitting_event_id=_uuid("retry"), admitting_sequence=600))
    _apply(store, retry)


def test_expiry_and_claim_exclude_each_other(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    for operation in _steps(chain)[:4]:
        _apply(store, operation)
    expiry = chain.operation(K.R18_APPROVAL_EXPIRE, R.APPROVAL_EXPIRATION, ApprovalExpirationV1.for_challenge(chain.challenges[0], ExpirationReason.BOOT_CHANGED), 8)
    _prepare(store, expiry)
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, _steps(chain)[4])  # the claim of the same challenge


def test_a_new_challenge_needs_every_earlier_one_expired(tmp_path: Path) -> None:
    """Human decision §11-3: up to 3 challenges per intent, each after the
    previous one is recorded as expired."""

    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    for operation in _steps(chain)[:4]:
        _apply(store, operation)
    second = chain.operation(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, chain.challenges[1], 9, attempt=2)
    _refused_at_finalize(store, second)  # the first challenge is still live
    for attempt in range(1, MAX_CHALLENGES_PER_INTENT + 1):
        if attempt > 1:
            _apply(store, chain.operation(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, chain.challenges[attempt - 1], 10 + attempt, attempt=attempt))
        _apply(store, chain.operation(K.R18_APPROVAL_EXPIRE, R.APPROVAL_EXPIRATION, ApprovalExpirationV1.for_challenge(chain.challenges[attempt - 1], ExpirationReason.EXPIRED), 20 + attempt))
    with pytest.raises(ValueError):
        chain.operation(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, chain.challenges[3], 30, attempt=4)


def test_a_later_challenge_can_be_claimed(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    for operation in _steps(chain)[:4]:
        _apply(store, operation)
    _apply(store, chain.operation(K.R18_APPROVAL_EXPIRE, R.APPROVAL_EXPIRATION, ApprovalExpirationV1.for_challenge(chain.challenges[0], ExpirationReason.BOOT_CHANGED), 20))
    _apply(store, chain.operation(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, chain.challenges[1], 21, attempt=2))
    claim = ExecutionClaimV1.claim(
        chain.challenges[1], runtime_instance_id=RUNTIME, boot_id=BOOT,
        claim_event_id=chain.event("claim-2"), claim_sequence=chain.seq(22), claimed_utc="2026-11-16T00:05:00Z",
    )
    _apply(store, chain.operation(K.R18_EXECUTION_CLAIM, R.EXECUTION_CLAIM, claim, 22))
    # The expired first challenge can no longer be claimed.
    stale = chain.operation(K.R18_EXECUTION_CLAIM, R.EXECUTION_CLAIM, chain.claim, 4)
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, stale)


@pytest.mark.parametrize("skip", [1, 2, 3, 4, 5, 6, 7])
def test_each_record_needs_its_previous_stage(tmp_path: Path, skip: int) -> None:
    """§2b: the FINALIZE-time gates refuse a record whose previous stage is
    missing (defence in depth behind the writers' pre-check)."""

    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    steps = _steps(chain)
    for operation in steps[:skip - 1]:
        _apply(store, operation)
    _refused_at_finalize(store, steps[skip])


def test_a_denied_intent_gets_no_challenge(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    deny = replace(chain.verdict, verdict=PolicyVerdict.DENY, deny_reason=PolicyDenyReason.SOURCE_NOT_CURRENT)
    steps = _steps(chain)
    _apply(store, steps[0])
    _apply(store, steps[1])
    _apply(store, chain.operation(K.R18_POLICY_COMMIT, R.POLICY_VERDICT, deny, 2))
    _refused_at_finalize(store, steps[3])


def test_an_inverse_needs_a_verified_forward_effect(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    inverse = InverseIntentV1(
        incarnation_id=INCARNATION,
        admitting_event_id=chain.event("inverse"),
        admitting_sequence=chain.seq(9),
        obligation_group_key=GROUP,
        target_instance_id=TARGET,
        target_namespace="review-artifact",
        target_expected_revision=1,
        forward_outcome_id=chain.outcome.verified_outcome_id,
        forward_outcome_revision=1,
        forward_outcome_digest=chain.outcome.outcome_digest,
        forward_outcome_state=VerifiedState.ACTIVE_VERIFIED_AUTHORIZED,
        forward_payload_digest=chain.intent.payload_digest,
    )
    early = chain.operation(K.R18_INTENT_ISSUE, R.INVERSE_INTENT, inverse, 9)
    _refused_at_finalize(store, early)  # no forward effect yet
    for step in _steps(chain)[:6]:
        _apply(store, step)
    later = replace(inverse, admitting_event_id=chain.event("inverse-2"), admitting_sequence=chain.seq(10))
    _apply(store, chain.operation(K.R18_INTENT_ISSUE, R.INVERSE_INTENT, later, 10))
    # The rest of the inverse chain: verdict (no forecast) -> challenge -> Claim -> REVERTED.
    verdict = PolicyVerdictV1(
        Direction.INVERSE, later.intent_id, later.intent_digest, later.inverse_key,
        PolicyVerdict.ALLOW, None, chain.event("inverse-verdict"), chain.seq(11),
    )
    _apply(store, chain.operation(K.R18_POLICY_COMMIT, R.POLICY_VERDICT, verdict, 11))
    challenge = ApprovalChallengeV1.for_inverse(
        later, verdict, nonce="7" * 64, issued_utc="2026-11-16T01:00:00Z",
        expires_utc="2026-11-16T01:10:00Z", boot_id=BOOT,
    )
    _apply(store, chain.operation(K.R18_CHALLENGE_ISSUE, R.APPROVAL_CHALLENGE, challenge, 12, attempt=1))
    claim = ExecutionClaimV1.claim(
        challenge, runtime_instance_id=RUNTIME, boot_id=BOOT,
        claim_event_id=chain.event("inverse-claim"), claim_sequence=chain.seq(13), claimed_utc="2026-11-16T01:05:00Z",
    )
    _apply(store, chain.operation(K.R18_EXECUTION_CLAIM, R.EXECUTION_CLAIM, claim, 13))
    reverted = replace(
        _outcome(VerifiedState.REVERTED),
        operation_key=later.inverse_key, target_instance_id=TARGET,
        verifier_event_id=chain.event("inverse-outcome"), verifier_sequence=chain.seq(14),
    )
    _apply(store, chain.operation(K.R18_OUTCOME_COMMIT, R.VERIFIED_OUTCOME, reverted, 14))


def test_records_are_immutable_and_finalize_is_complete(tmp_path: Path) -> None:
    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    _apply(store, _steps(chain)[0])
    connection = sqlite3.connect(store.path, isolation_level=None)
    for statement in (
        "UPDATE action_intents SET direction = 'INVERSE'",
        "DELETE FROM action_intents",
        "DELETE FROM history_operation_keys",
        "UPDATE history_operation_keys SET live = 0",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(statement)
    # FINALIZED without the record row is refused by the trigger.
    second = _steps(chain)[1]
    transaction_id = _prepare(store, second)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE history_operations SET phase = 'FINALIZED' WHERE transaction_id = ?", (transaction_id,))


def test_record_capacity_refuses_before_prepare(tmp_path: Path) -> None:
    from suzka.r16.history_store import HistoryStoreCapacityError

    store = _v2_store(tmp_path, pages=80)  # an empty v2 store takes 66 pages
    chain = Chain(_committed(_event("capacity", 3)))
    with pytest.raises(HistoryStoreCapacityError):
        _prepare(store, _steps(chain)[0])
    assert not store.has_record_operations()


# ------------------------------------------------------------ startup check


def test_startup_refuses_records_while_r18_is_disabled(tmp_path: Path) -> None:
    """§11-2: until IP-16, R17 / R18 rows with r18.enabled false stop startup."""

    from suzka.r16.history_adoption import HistoryAdoptionError, open_or_create_store

    store = _v2_store(tmp_path)
    chain = Chain(_decision(store))
    _apply(store, _steps(chain)[0])
    path = store.path
    store.close()

    class Journal:
        def inspect(self):  # noqa: ANN202
            from types import SimpleNamespace

            return SimpleNamespace(journal_lineage_id="lineage-1", latest_compaction=None, baselines=(), active_baseline=None)

    from suzka.r16 import history_adoption

    original = history_adoption.adopted_history_baseline
    history_adoption.adopted_history_baseline = lambda inspection: object()  # type: ignore[assignment]
    try:
        with pytest.raises(HistoryAdoptionError, match="R18_RECORDS_PRESENT_BUT_DISABLED"):
            open_or_create_store(Journal(), path, subject_id="suzka", max_page_count=PAGES)  # type: ignore[arg-type]
        reopened = open_or_create_store(Journal(), path, subject_id="suzka", max_page_count=PAGES, r18_enabled=True)  # type: ignore[arg-type]
        reopened.close()
    finally:
        history_adoption.adopted_history_baseline = original  # type: ignore[assignment]


def test_startup_names_the_migration(tmp_path: Path) -> None:
    from suzka.r16 import history_adoption
    from suzka.r16.history_adoption import HistoryAdoptionError, open_or_create_store

    path, _ = _v1_store_with_decisions(tmp_path)

    class Journal:
        def inspect(self):  # noqa: ANN202
            from types import SimpleNamespace

            return SimpleNamespace(journal_lineage_id="lineage-1")

    original = history_adoption.adopted_history_baseline
    history_adoption.adopted_history_baseline = lambda inspection: object()  # type: ignore[assignment]
    try:
        with pytest.raises(HistoryAdoptionError, match="history_migration"):
            open_or_create_store(Journal(), path, subject_id="suzka", max_page_count=PAGES)  # type: ignore[arg-type]
    finally:
        history_adoption.adopted_history_baseline = original  # type: ignore[assignment]
    assert path.exists()  # never recreated


def test_settings_default_off() -> None:
    from suzka.config.schema import R18Settings, R19Settings

    assert R18Settings().enabled is False and R19Settings().enabled is False


# ------------------------------------------- end to end on the real server


def test_a_real_v1_store_migrates_and_keeps_its_head(tmp_path: Path) -> None:
    """§5 acceptance: a v1 store with an adopted, issued Decision stops
    startup until migrated; afterwards the same head is adopted and the
    issuer returns the existing Decision."""

    from fastapi.testclient import TestClient

    from suzka.r16.decision_issuer import DecisionIssueStatus
    from suzka.r16.history_reader import HeadStatus

    from test_r16_ip10_decision_issuer import _app, _issue, _reader_head, _settings
    from test_r16_u3a_plan_preview import _qualified_goal

    root = tmp_path / "e2e"
    settings = _settings(root)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        _x, gid = _qualified_goal(state)
        issued = _issue(state, gid)
        assert issued.status is DecisionIssueStatus.ISSUED
        committed = issued.decision
        before = _reader_head(state, issued)
    path = settings.subject_history.path
    _downgrade_to_v1(path)
    with pytest.raises(Exception, match="history_migration"):
        with TestClient(_app(settings)):
            pass
    migrate_v1_to_v2(path)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        head = _reader_head(state, issued)
        assert (head.status, head.decision, head.depth) == (HeadStatus.ADOPTED, committed, before.depth)
        again = _issue(state, gid)
        assert again.status is DecisionIssueStatus.EXISTING
        assert again.decision is not None and again.decision.decision_id == committed.decision_id
