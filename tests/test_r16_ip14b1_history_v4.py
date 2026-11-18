"""IP-14b-1: SubjectHistory schema v4 (DDL, store write API, v1 / v2 / v3 -> v4 migration).

Design: IP-14b D1 r2 FINAL (#411; Human decisions HD-1 (a), HD-2 (a),
HD-3 (a), 2026-11-18), with IP-17's row 2c bundled into the same v4. There is
no writer yet (IP-14b-2 / -3, IP-17a, IP-20 / IP-21): the store API and the
SQL triggers are exercised directly. Every value here is a FIXTURE ONLY.

The "trigger matrix" tests port the two throwaway prototypes of D1 r2 §3-3
(20 r1 cases and 29 r2 cases) onto the real v4 DDL, with raw SQL and dummy
bytes, so each SQL rule is shown on its own. The store tests use the real
R16 / R19 values and the store write API.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from suzka.r16 import history_migration
from suzka.r16.contracts import ReasonCode
from suzka.r16.cycle_contracts import (
    MAX_PLAN_HOLD_BYTES,
    CycleClosureV1,
    ExistingDecisionReceiptV1,
    HoldReceiptV1,
    LogicalCycleResolutionV1,
    PlanHoldReceiptV1,
)
from suzka.r16.history_codec import GENESIS, PolicyKind, PolicyRow
from suzka.r16.history_migration import HistoryMigrationError, migrate
from suzka.r16.history_records import (
    PLAN_HOLD_GATED_KINDS,
    R18_GATED_KINDS,
    R19_GATED_KINDS,
    RECORD_OPERATION_KINDS,
    V4_RECORD_TABLES,
    HistoryOperationKind,
    HistoryRecordOperationV1,
    RecordKind,
)
from suzka.r16.history_schema_v1 import SCHEMA_HASH_V1, SCHEMA_STATEMENTS_V1
from suzka.r16.history_schema_v2 import SCHEMA_HASH_V2, SCHEMA_STATEMENTS_V2
from suzka.r16.history_schema_v3 import SCHEMA_HASH_V3, SCHEMA_STATEMENTS_V3
from suzka.r16.history_store import (
    _SCHEMA_QUERY,
    HISTORY_SCHEMA_VERSION,
    SCHEMA_HASH,
    SCHEMA_STATEMENTS,
    HistoryStoreCapacityError,
    HistoryStoreConflict,
    HistoryStoreCorrupt,
    HistoryStoreMigrationRequired,
    HistoryStoreRecordsDisabled,
    SubjectHistoryStore,
    _expected_schema,
)
from suzka.r16.successor_policy import (
    Interpretation,
    InterpretationStage,
    SuccessorRow,
)
from suzka.r19.contracts import (
    AckStatus,
    AttemptOutcome,
    InboxAckV1,
    OutboxMessageV1,
    RefuseReason,
    TerminalReason,
    WakeAttemptOutcomeV1,
    WakeCause,
    WakeClaimV1,
    WakeTerminalKind,
    WakeTerminalV1,
    WakeV1,
)

from test_r16_ip14a_history_v2 import (
    Chain,
    _all_rows,
    _apply,
    _decision,
    _downgrade,
    _prepare,
    _receipt,
    _steps,
    _v2_store,
)
from test_r16_ip8b_successor_policy import _active
from test_r16_ip9_subject_history import INCARNATION, PAGES, _committed, _event, _private_dir
from test_r19_ip13b_contracts import CREATED, EXPIRES, INC, TIMING, _wake

K = HistoryOperationKind
R = RecordKind
H = "a" * 64
OLD = "c" * 64  # a previous incarnation
TRIGGER_TABLES = (
    "wakeups", "wake_claims", "wake_attempt_outcomes", "logical_cycle_resolutions",
    "wake_terminals", "outbox_messages", "outbox_acks", "plan_hold_receipts",
)


def _uuid(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, name))


def _store(tmp_path: Path, *, r18: bool = False, r19: bool = True, holds: bool = True, pages: int = PAGES) -> SubjectHistoryStore:
    store = SubjectHistoryStore.create(
        _private_dir(tmp_path) / "subject_history.sqlite3",
        subject_id="suzka",
        journal_lineage_id="lineage-1",
        max_page_count=pages,
        r18_enabled=r18,
        r19_enabled=r19,
        plan_hold_receipts=holds,
    )
    store.activate(adopted_baseline_id="baseline-1", adopted_baseline_hash=H, genesis_incarnation_id=INCARNATION)
    return store


# ------------------------------------------------------------------ schema


def test_a_fresh_store_is_exactly_v4(tmp_path: Path) -> None:
    store = _store(tmp_path)
    connection = sqlite3.connect(store.path)
    assert connection.execute(_SCHEMA_QUERY).fetchall() == _expected_schema(SCHEMA_STATEMENTS)
    assert store.meta().schema_version == HISTORY_SCHEMA_VERSION == 4
    assert store.meta().schema_hash == SCHEMA_HASH != SCHEMA_HASH_V3
    names = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
    assert set(TRIGGER_TABLES) <= names and set(V4_RECORD_TABLES) == set(TRIGGER_TABLES)
    # D1 r2 §3-1: the 69 v3 statements in order with 10 edits, then 32 new ones.
    assert len(SCHEMA_STATEMENTS) == 101
    changed = [old.split("\n")[0] for old, new in zip(SCHEMA_STATEMENTS_V3, SCHEMA_STATEMENTS, strict=False) if old != new]
    assert [line.split()[2] if line.split()[1] != "UNIQUE" else line.split()[3] for line in changed] == [
        "history_meta", "history_operations", "history_operations_review_intent",
        "history_operations_predecessor", "history_decisions", "history_operations_insert",
        "history_operations_finalize_complete", "history_operations_abort_clean",
        "history_decisions_insert", "history_operation_keys_insert",
    ]
    empty = sqlite3.connect(":memory:")
    for statement in SCHEMA_STATEMENTS:
        empty.execute(statement)
    assert empty.execute("PRAGMA page_count").fetchone()[0] == 99


def test_the_frozen_v3_text_rederives_its_hash() -> None:
    assert sha256("\0".join(SCHEMA_STATEMENTS_V3).encode()).hexdigest() == SCHEMA_HASH_V3
    assert SCHEMA_HASH_V3 == "9bafb69205ad00d60aca03c6b7b591b40192b6a0485a55325840b508d15f3ed3"
    assert len(SCHEMA_STATEMENTS_V3) == 69


def test_the_record_bound_covers_every_family() -> None:
    from suzka.r16 import history_store
    from suzka.r16.cycle_contracts import MAX_RECORD_BYTES as CYCLE
    from suzka.r17.contracts import MAX_RECORD_BYTES as R17
    from suzka.r18.contracts import MAX_RECORD_BYTES as R18
    from suzka.r19.contracts import MAX_RECORD_BYTES as R19

    assert history_store._RECORD_BYTES == max(R17, R18, R19, CYCLE, MAX_PLAN_HOLD_BYTES) == 16_384


def _v4_store_with_decisions(tmp_path: Path, *, records: bool = False) -> tuple[Path, list[Any]]:
    store = _v2_store(tmp_path, r18=True)
    decisions: list[Any] = []
    for index in range(3):
        decisions.append(_decision(store, f"decision-{index}", 3 + index, decisions[-1] if decisions else None))
    if records:
        for operation in _steps(Chain(decisions[-1]))[:8]:
            _apply(store, operation)
    path = store.path
    store.close()
    return path, decisions


def _v3_store(tmp_path: Path, *, records: bool = False) -> tuple[Path, list[Any]]:
    """A stand-in for eve's v3 store (the 52c61ab-made one is tested below)."""

    path, decisions = _v4_store_with_decisions(tmp_path, records=records)
    _downgrade(path, 3, SCHEMA_STATEMENTS_V3, SCHEMA_HASH_V3)
    return path, decisions


@pytest.mark.parametrize(
    ("version", "statements", "schema_hash"),
    [(1, SCHEMA_STATEMENTS_V1, SCHEMA_HASH_V1), (2, SCHEMA_STATEMENTS_V2, SCHEMA_HASH_V2), (3, SCHEMA_STATEMENTS_V3, SCHEMA_HASH_V3)],
)
def test_older_files_require_migration_and_a_tampered_one_is_corrupt(
    tmp_path: Path, version: int, statements: tuple[str, ...], schema_hash: str
) -> None:
    path, _ = _v4_store_with_decisions(tmp_path)
    _downgrade(path, version, statements, schema_hash)
    with pytest.raises(HistoryStoreMigrationRequired, match=f"v{version}"):
        SubjectHistoryStore.open(path, max_page_count=PAGES)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE extra (x INTEGER)")
    connection.commit()
    connection.close()
    with pytest.raises(HistoryStoreCorrupt):
        SubjectHistoryStore.open(path, max_page_count=PAGES)


def test_startup_names_the_migration_for_a_v3_store(tmp_path: Path) -> None:
    from suzka.r16 import history_adoption
    from suzka.r16.history_adoption import HistoryAdoptionError, open_or_create_store

    path, _ = _v3_store(tmp_path)

    class Journal:
        def inspect(self):  # noqa: ANN202
            from types import SimpleNamespace

            return SimpleNamespace(journal_lineage_id="lineage-1")

    original = history_adoption.adopted_history_baseline
    history_adoption.adopted_history_baseline = lambda inspection: object()  # type: ignore[assignment]
    try:
        with pytest.raises(HistoryAdoptionError, match="v3.*history_migration"):
            open_or_create_store(Journal(), path, subject_id="suzka", max_page_count=PAGES)  # type: ignore[arg-type]
    finally:
        history_adoption.adopted_history_baseline = original  # type: ignore[assignment]
    assert path.exists()


# ------------------------------------------------------- raw SQL harness


class Raw:
    """Direct SQL on a real v4 file, as the D1 r2 prototype did."""

    def __init__(self, tmp_path: Path, statements: tuple[str, ...] = SCHEMA_STATEMENTS) -> None:
        if statements is SCHEMA_STATEMENTS:
            store = _store(tmp_path)
            self.path = store.path
            store.close()
            self.c = sqlite3.connect(self.path, isolation_level=None)
        else:
            self.c = sqlite3.connect(":memory:", isolation_level=None)
            for statement in statements:
                self.c.execute(statement)
            self.c.execute(
                "INSERT INTO history_meta VALUES (1, 3, ?, 'subj', ?, 'lin', 'PROVISIONAL', NULL, NULL, NULL)",
                (SCHEMA_HASH_V3, str(uuid4())),
            )
            self.c.execute(
                "UPDATE history_meta SET phase = 'ACTIVE', adopted_baseline_id = 'b', "
                "adopted_baseline_hash = ?, genesis_incarnation_id = ?",
                (H, INC),
            )
        self.c.execute("PRAGMA foreign_keys = ON")
        self.seq = 1000

    def prep(self, kind: str, pending: bytes, keys: tuple[str, ...] = (), lineage: tuple[Any, Any, Any] = (None, None, None), digest: Any = "default", inc: str = INC) -> tuple[str, str]:
        self.seq += 1
        tx, ev = str(uuid4()), str(uuid4())
        self.c.execute(
            "INSERT INTO history_operations VALUES (?, ?, ?, ?, 'K', 'subject.history', ?, ?, 'PREPARED', ?, ?, ?, ?, ?)",
            (inc, tx, ev, self.seq, kind, sha256(tx.encode()).hexdigest(), pending,
             sha256(pending).hexdigest() if digest == "default" else digest, *lineage),
        )
        for key in keys:
            self.c.execute("INSERT INTO history_operation_keys VALUES (?, ?, ?, 1)", (key, inc, tx))
        return tx, ev

    def try_prep(self, *args: Any, **kwargs: Any) -> bool:
        self.c.execute("BEGIN")
        try:
            self.prep(*args, **kwargs)
        except sqlite3.Error:
            self.c.execute("ROLLBACK")
            return False
        self.c.execute("ROLLBACK")
        return True

    def fin(self, tx: str, ev: str, rows: list[tuple[str, dict[str, Any]]], inc: str = INC) -> bool:
        self.c.execute("BEGIN IMMEDIATE")
        try:
            for table, columns in rows:
                names = (*columns, "incarnation_id", "transaction_id")
                self.c.execute(
                    f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})",
                    (*columns.values(), inc, tx),
                )
            sequence = self.c.execute("SELECT processing_sequence FROM history_operations WHERE transaction_id = ?", (tx,)).fetchone()[0]
            self.c.execute(
                "INSERT INTO history_postcommit_receipts VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (inc, tx, ev, sequence, H, "b" * 64, "b" * 64, str(uuid4()), str(uuid4()), H, H),
            )
            self.c.execute("UPDATE history_operations SET phase = 'FINALIZED' WHERE transaction_id = ?", (tx,))
        except sqlite3.Error:
            self.c.execute("ROLLBACK")
            return False
        self.c.execute("COMMIT")
        return True

    def abort(self, tx: str) -> None:
        self.c.execute("UPDATE history_operations SET phase = 'ABORTED' WHERE transaction_id = ?", (tx,))

    def refused(self, sql: str, *params: Any) -> bool:
        try:
            self.c.execute(sql, params)
        except sqlite3.Error:
            return True
        return False

    def healthy(self) -> None:
        assert self.c.execute("PRAGMA foreign_key_check").fetchall() == []
        assert self.c.execute("PRAGMA quick_check").fetchone()[0] == "ok"

    # --- the prototype's building blocks -------------------------------

    def wake(self, name: str, *, lcid: str | None = None, cause: str | None = None, outcome: tuple[str, int] | None = None) -> tuple[str, str]:
        wid, lc = "wakeup-" + _h(name), lcid or _h(f"lc-{name}")
        body = json.dumps({"wake": name}).encode()
        tx, ev = self.prep("R19_WAKE_MATERIALIZE", b'{"e":' + body + b"}", (f"wake:{wid}",))
        ok = self.fin(tx, ev, [("wakeups", dict(
            record_id=wid, record_bytes=body, record_digest=_h(wid), logical_cycle_id=lc,
            cause_event_id=cause or str(uuid4()),
            verified_outcome_id=None if outcome is None else outcome[0],
            outcome_revision=None if outcome is None else outcome[1]))])
        if not ok:
            self.abort(tx)
            raise sqlite3.IntegrityError("wake refused")
        return wid, lc

    def claim(self, wid: str, lc: str, n: int, *, key: str | None = None) -> tuple[str, bool]:
        cid = "wake-claim-" + _h(f"{wid}-{n}-{key}")
        body = json.dumps({"claim": cid}).encode()
        tx, ev = self.prep("R19_WAKE_CLAIM", b'{"e":' + body + b"}", (key or f"wake-attempt:{wid}:{n}",))
        ok = self.fin(tx, ev, [("wake_claims", dict(record_id=cid, record_bytes=body, record_digest=_h(cid), wakeup_id=wid, logical_cycle_id=lc, attempt_index=n))])
        if not ok:
            self.abort(tx)
        return cid, ok

    def resolution(self, lc: str, claim: str, *, kind: str = "R16_CYCLE_RECEIPT", rik: str | None = None, keys: tuple[str, ...] | None = None, receipt_in_pending: bool = True, finalize: bool = True) -> tuple[str, bool]:
        body = json.dumps({"resolution": lc, "claim": claim}).encode()
        receipt = json.dumps({"hold": claim}).encode()
        pending = b'{"e":{"r":' + body + b',"h":' + (receipt if receipt_in_pending else b"{}") + b"}}"
        tx, ev = self.prep(kind, pending, (f"cycle:{INC}:{lc}", f"attempt-end:{claim}") if keys is None else keys)
        if not finalize:
            return tx, True
        ok = self.fin(tx, ev, [("logical_cycle_resolutions", dict(
            record_id="cycle-resolution-" + _h(lc), record_bytes=body, record_digest=_h(f"rd-{lc}"),
            logical_cycle_id=lc, review_intent_key=rik or _h(f"rik-{lc}"), claim_id=claim, result_kind="HOLD",
            result_id="cycle-hold-" + _h(claim), result_digest=_h(f"hd-{lc}"), decision_id=None, receipt_bytes=receipt))])
        if not ok:
            self.abort(tx)
        return tx, ok

    def outcome(self, claim: str, *, kind: str = "R19_WAKE_ATTEMPT_OUTCOME", key: str | None = "unique", in_pending: bool = True) -> bool:
        body = json.dumps({"outcome": claim}).encode()
        pending = b'{"e":' + (body if in_pending else b"{}") + b"}"
        keys = () if key is None else ((f"attempt-end:{claim}:{uuid4()}" if key == "unique" else key),)
        tx, ev = self.prep(kind, pending, keys)
        ok = self.fin(tx, ev, [("wake_attempt_outcomes", dict(record_id=claim, record_bytes=body, record_digest=_h(f"o-{claim}")))])
        if not ok:
            self.abort(tx)
        return ok

    def terminal(self, wid: str, lc: str, reason: str, *, superseding: str | None = None, key_cycle: str | None = None) -> bool:
        body = json.dumps({"terminal": wid, "reason": reason}).encode()
        tx, ev = self.prep("R19_WAKE_TERMINAL", b'{"e":' + body + b"}", (f"cycle:{INC}:{key_cycle or lc}",))
        ok = self.fin(tx, ev, [("wake_terminals", dict(
            record_id="wake-terminal-" + _h(wid), record_bytes=body, record_digest=_h(f"t-{wid}"), wakeup_id=wid,
            terminal="CANCELLED" if reason == "SUPERSEDED" else "FAILED_TERMINAL", reason=reason,
            superseding_wakeup_id=superseding))])
        if not ok:
            self.abort(tx)
        return ok

    def message(self, source: str, digest: str, template: str, context: str = "ctx") -> tuple[str, bool]:
        mid = "outbox-" + _h(f"{source}-{template}-{digest}")
        body = json.dumps({"message": mid}).encode()
        tx, ev = self.prep("R19_OUTBOX_EMIT", b'{"e":' + body + b"}", (f"outbox:{mid}",))
        ok = self.fin(tx, ev, [("outbox_messages", dict(record_id=mid, record_bytes=body, record_digest=_h(mid), source_id=source, result_digest=digest, template=template, context_id=context))])
        if not ok:
            self.abort(tx)
        return mid, ok

    def ack(self, mid: str, context: str) -> bool:
        body = json.dumps({"ack": mid, "context": context}).encode()
        tx, ev = self.prep("R19_INBOX_ACK", b'{"e":' + body + b"}", (f"ack:{mid}:{context}",))
        ok = self.fin(tx, ev, [("outbox_acks", dict(record_id="inbox-ack-" + _h(mid + context), record_bytes=body, record_digest=_h(mid), message_id=mid, context_id=context))])
        if not ok:
            self.abort(tx)
        return ok

    def decision_row(self, tx: str, ev: str, did: str, committed: bytes, *, row: str = "4", kind: str = "SUCCESSOR_POLICY_V3", rik: str, scope: str = "X", predecessor: str = GENESIS) -> tuple[str, dict[str, Any]]:
        sequence = self.c.execute("SELECT processing_sequence FROM history_operations WHERE transaction_id = ?", (tx,)).fetchone()[0]
        return ("history_decisions", dict(
            decision_id=did, lineage_scope=scope, review_intent_key=rik, predecessor_key=predecessor,
            predecessor_digest=None, issuer_event_id=ev, issuer_sequence=sequence, decision_bytes=b"{}",
            decision_digest=H, committed_bytes=committed, committed_digest=_h(did), policy_kind=kind,
            policy_row=row, policy_input_bytes=b"{}", policy_input_digest=H))


def _h(text: str) -> str:
    return sha256(text.encode()).hexdigest()


# ------------------------------------------- trigger matrix (prototype r1)


def test_trigger_matrix_r1_claims_closure_outbox_and_ack(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    wid, lc = raw.wake("w1")  # 1 wake
    c1, ok = raw.claim(wid, lc, 1)
    assert ok  # 2 claim 1
    assert not raw.claim(wid, lc, 3)[1]  # 3 claim 3 out of order
    c2, ok = raw.claim(wid, lc, 2)
    assert ok  # 4 claim 2
    assert not raw.resolution(lc, c1)[1]  # 5 resolution on the stale claim 1
    assert raw.resolution(lc, c2)[1]  # 6 on the latest claim (the refused one's keys were released by its abort)
    assert not raw.try_prep("R19_WAKE_TERMINAL", b"{}", (f"cycle:{INC}:{lc}",))  # 7 terminal PREPARE after closure
    assert not raw.claim(wid, lc, 3)[1]  # 8 claim 3 after closure
    held = _h(f"hd-{lc}")
    assert not raw.message(lc, held, "DECISION_ISSUED")[1]  # 9 wrong template for a HOLD
    mid, ok = raw.message(lc, held, "DECISION_HELD")
    assert ok  # 10 outbox HELD
    assert not raw.ack(mid, "other")  # 11 wrong Context (F11)
    assert raw.ack(mid, "ctx")  # 12 ack
    assert raw.refused("DELETE FROM outbox_acks")  # 13 immutable
    raw.healthy()


def test_trigger_matrix_r1_cycle_decision_slots_and_row_2c(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    wid, lc = raw.wake("w2")
    claim, ok = raw.claim(wid, lc, 1)
    assert ok
    rik = _h("rik-D")
    committed = b'{"committed":1}'
    resolution = json.dumps({"resolution": 2}).encode()
    did = "decision-" + _h("d1")
    tx, ev = raw.prep(
        "R16_CYCLE_DECISION", b'{"e":{"d":' + committed + b',"r":' + resolution + b"}}",
        (f"cycle:{INC}:{lc}", f"attempt-end:{claim}"), lineage=("X", rik, GENESIS),
    )
    # 14 a PLAN_PREVIEW Decision on the same predecessor slot while the cycle Decision is PREPARED
    assert not raw.try_prep("R16_DECISION_ISSUE", b'{"pp":1}', lineage=("X", _h("rik-P"), GENESIS))
    # ... and on the same review-intent slot
    assert not raw.try_prep("R16_DECISION_ISSUE", b'{"pp":2}', lineage=("Y", rik, GENESIS))
    # 15 the cycle Decision, V3 row 2c, with its resolution
    assert raw.fin(tx, ev, [raw.decision_row(tx, ev, did, committed, row="2c", rik=rik), ("logical_cycle_resolutions", dict(
        record_id="cycle-resolution-" + _h("r2"), record_bytes=resolution, record_digest=_h("rd2"), logical_cycle_id=lc,
        review_intent_key=rik, claim_id=claim, result_kind="DECISION_ISSUED", result_id=did, result_digest=_h(did),
        decision_id=did, receipt_bytes=None))])
    tx, ev = raw.prep("R16_DECISION_ISSUE", b'{"c2":1}', lineage=("Y", _h("rik-Y"), GENESIS))
    other = "decision-" + _h("y")
    # 16 row 2c is V3 only
    for kind in ("SUCCESSOR_POLICY_V1", "SUCCESSOR_POLICY_V2"):
        assert not raw.fin(tx, ev, [raw.decision_row(tx, ev, other, b'{"c2":1}', row="2c", kind=kind, rik=_h("rik-Y"), scope="Y")])
    # 17 a V3 PLAN_PREVIEW Decision with row 2c
    assert raw.fin(tx, ev, [raw.decision_row(tx, ev, other, b'{"c2":1}', row="2c", rik=_h("rik-Y"), scope="Y")])
    # 18 a cycle-Decision operation without its lineage columns
    assert not raw.try_prep("R16_CYCLE_DECISION", b'{"z":1}', ())
    # 19 / 20 an R16 operation with NULL lineage columns
    for lineage in ((None, None, None), ("X", None, "GENESIS"), (None, _h("q"), "GENESIS"), ("X", _h("q"), None)):
        assert not raw.try_prep("R16_DECISION_ISSUE", b'{"q":1}', lineage=lineage)
        assert not raw.try_prep("R16_CYCLE_DECISION", b'{"q":1}', lineage=lineage)
    raw.healthy()


def test_the_null_safe_checks_close_a_v3_gap(tmp_path: Path) -> None:
    """D1 r2 §2-1 (e): v3 accepted a CHECK that evaluates to NULL."""

    v3 = Raw(tmp_path / "v3", SCHEMA_STATEMENTS_V3)
    v4 = Raw(tmp_path / "v4")
    for raw, admitted in ((v3, True), (v4, False)):
        assert raw.try_prep("R16_DECISION_ISSUE", b'{"q":1}', lineage=("X", None, GENESIS)) is admitted
        assert raw.try_prep("R17_FEEDBACK_COMMIT", b'{"q":1}', digest=None) is admitted
    # An ACTIVE meta with NULL baseline columns.
    statements = [s for s in SCHEMA_STATEMENTS if s.startswith("CREATE TABLE history_meta")]
    connection = sqlite3.connect(":memory:")
    connection.execute(statements[0])
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO history_meta VALUES (1, 4, ?, 's', ?, 'l', 'ACTIVE', NULL, NULL, NULL)", (H, str(uuid4())))
    v3_meta = sqlite3.connect(":memory:")
    v3_meta.execute(SCHEMA_STATEMENTS_V3[0])
    v3_meta.execute("INSERT INTO history_meta VALUES (1, 3, ?, 's', ?, 'l', 'ACTIVE', NULL, NULL, NULL)", (H, str(uuid4())))


# ------------------------------------------- trigger matrix (prototype r2)


def test_trigger_matrix_r2_attempt_outcomes(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    wid, lc = raw.wake("w1")
    c1, ok = raw.claim(wid, lc, 1)
    assert ok
    assert not raw.outcome(c1, kind="R19_WAKE_CLAIM")  # wrong operation kind
    assert not raw.outcome(c1, in_pending=False)  # bytes not in the pending bytes
    assert not raw.outcome(c1, key=None)  # no live key: FINALIZE incomplete
    assert not raw.outcome("wake-claim-" + _h("nope"))  # unknown claim
    assert raw.outcome(c1)  # the latest claim
    assert not raw.outcome(c1)  # a second outcome for one claim (PK)
    c2, ok = raw.claim(wid, lc, 2)
    assert ok  # an outcome never blocks the next claim in SQL (the gap is IP-20's)
    assert not raw.outcome(c1)  # a stale claim after a newer one
    c3, ok = raw.claim(wid, lc, 3)
    assert ok and raw.outcome(c3)
    assert not raw.resolution(lc, c3, keys=(f"cycle:{INC}:{lc}",))[1]  # a resolution on a claim with an outcome
    assert raw.terminal(wid, lc, "ATTEMPTS_EXHAUSTED")
    wid2, lc2 = raw.wake("w2")
    k1, ok = raw.claim(wid2, lc2, 1)
    assert ok and raw.resolution(lc2, k1)[1]
    assert not raw.outcome(k1)  # the cycle is closed
    assert raw.refused("DELETE FROM wake_attempt_outcomes")
    assert raw.refused("UPDATE wake_attempt_outcomes SET record_digest = record_digest")
    raw.healthy()


def test_trigger_matrix_r2_plan_hold_receipts(tmp_path: Path) -> None:
    raw = Raw(tmp_path)

    def hold(rik: str, *, scope: str = "X", kind: str = "R16_PLAN_HOLD", key: str | None = "default", in_pending: bool = True) -> bool:
        body = json.dumps({"plan_hold": rik, "n": str(uuid4())}).encode()
        pending = b'{"e":' + (body if in_pending else b"{}") + b"}"
        keys = () if key is None else ((f"plan-hold:{INC}:{rik}" if key == "default" else key),)
        tx, ev = raw.prep(kind, pending, keys)
        ok = raw.fin(tx, ev, [("plan_hold_receipts", dict(record_id="plan-hold-" + _h(body.decode()), record_bytes=body, record_digest=_h(rik), lineage_scope=scope, review_intent_key=rik))])
        if not ok:
            raw.abort(tx)
        return ok

    first = _h("p1")
    assert not hold(first, kind="R19_WAKE_TERMINAL")
    assert not hold(first, in_pending=False)
    assert not hold(first, key=None)
    assert not hold("zz" * 32)  # a malformed review_intent_key
    assert not hold(_h("p9"), scope="")
    assert hold(first)
    assert not raw.try_prep("R16_PLAN_HOLD", b'{"x":1}', (f"plan-hold:{INC}:{first}",))  # the key is live
    assert not hold(first, key="plan-hold-alt:" + first)  # the table UNIQUE (incarnation, review_intent_key)
    assert hold(_h("p2"))  # other evidence
    assert not raw.try_prep("R16_PLAN_HOLD", b'{"y":1}', lineage=("X", _h("q"), GENESIS))  # no lineage columns
    assert raw.refused("UPDATE plan_hold_receipts SET lineage_scope = 'Y'")
    raw.healthy()


# --------------------------------------------- further SQL rules (§2-3)


def test_wake_keys_outcome_fk_and_one_wake_per_outcome_revision(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    cause = str(uuid4())
    raw.wake("w1", cause=cause)
    with pytest.raises(sqlite3.IntegrityError):
        raw.wake("w2", cause=cause)  # UNIQUE (incarnation, cause_event_id)
    with pytest.raises(sqlite3.IntegrityError):
        raw.wake("w3", outcome=("outcome-x", 1))  # the cited Outcome must exist
    with pytest.raises(sqlite3.IntegrityError):
        raw.wake("w4", lcid=_h("lc-w1"))  # one Wake per logical cycle
    assert not raw.try_prep("R19_WAKE_MATERIALIZE", b"{}", ("wake:wakeup-" + _h("w1"),))  # the wake: key is live
    raw.healthy()


def test_claims_need_the_wakes_own_cycle_and_stop_at_a_prepared_closure(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    wid, lc = raw.wake("w1")
    assert not raw.claim(wid, _h("other-cycle"), 1)[1]  # a wrong logical_cycle_id
    claim, ok = raw.claim(wid, lc, 1)
    assert ok
    tx, _ = raw.resolution(lc, claim, finalize=False)  # a closure PREPARED, not yet FINALIZED
    assert not raw.claim(wid, lc, 2)[1]
    raw.abort(tx)  # ABORT releases the cycle key
    assert raw.claim(wid, lc, 2)[1]
    assert not raw.claim(wid, lc, 4, key="wake-attempt-x")[1]  # attempt_index is 1..3
    raw.healthy()


def test_one_closure_per_cycle_in_every_order(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    for order in (("resolution", "terminal"), ("terminal", "resolution"), ("resolution", "resolution")):
        wid, lc = raw.wake("-".join(order) + str(uuid4()))
        claims = [raw.claim(wid, lc, n)[0] for n in (1, 2, 3)]
        first, second = order
        if first == "resolution":
            assert raw.resolution(lc, claims[2])[1]
        else:
            assert raw.terminal(wid, lc, "ATTEMPTS_EXHAUSTED")
        # The second closure collides on the cycle: key at PREPARE.
        keys = (f"cycle:{INC}:{lc}",)
        assert not raw.try_prep("R16_CYCLE_RECEIPT" if second == "resolution" else "R19_WAKE_TERMINAL", b"{}", keys)
        assert not raw.try_prep("R16_CYCLE_DECISION", b"{}", keys, lineage=("Z", _h(str(uuid4())), GENESIS))
    raw.healthy()


def test_resolution_rules_kind_receipt_and_same_operation_decision(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    wid, lc = raw.wake("w1")
    claim, _ = raw.claim(wid, lc, 1)
    # A receipt resolution under the Decision kind (needs lineage) is refused.
    assert not raw.try_prep("R16_CYCLE_DECISION", b"{}", (f"cycle:{INC}:{lc}",))
    assert not raw.resolution(lc, claim, receipt_in_pending=False)[1]  # the receipt must be in the pending bytes
    # A Decision of another operation (a PLAN_PREVIEW Decision) cannot be named.
    tx, ev = raw.prep("R16_DECISION_ISSUE", b'{"pp":1}', lineage=("X", _h("rik-pp"), GENESIS))
    pp = "decision-" + _h("pp")
    assert raw.fin(tx, ev, [raw.decision_row(tx, ev, pp, b'{"pp":1}', rik=_h("rik-pp"))])
    body = b'{"r":1}'
    tx, ev = raw.prep("R16_CYCLE_DECISION", b'{"e":' + body + b"}", (f"cycle:{INC}:{lc}",), lineage=("Q", _h("rik-pp2"), GENESIS))
    row = dict(record_id="cycle-resolution-" + _h(lc), record_bytes=body, record_digest=H, logical_cycle_id=lc,
               review_intent_key=_h("rik-pp"), claim_id=claim, result_kind="DECISION_ISSUED", result_id=pp,
               result_digest=H, decision_id=pp, receipt_bytes=None)
    assert not raw.fin(tx, ev, [("logical_cycle_resolutions", row)])
    # Its own Decision with another review_intent_key is refused too.
    own = "decision-" + _h("own")
    committed = b'{"own":1}'
    raw.abort(tx)
    tx, ev = raw.prep("R16_CYCLE_DECISION", b'{"e":{"d":' + committed + b',"r":' + body + b"}}", (f"cycle:{INC}:{lc}",), lineage=("Q", _h("rik-own"), GENESIS))
    assert not raw.fin(tx, ev, [raw.decision_row(tx, ev, own, committed, rik=_h("rik-own"), scope="Q"), ("logical_cycle_resolutions", {**row, "result_id": own, "decision_id": own, "review_intent_key": _h("rik-other")})])
    # The Decision's committed bytes must be embedded in the envelope.
    assert not raw.fin(tx, ev, [raw.decision_row(tx, ev, own, b'{"not-embedded":1}', rik=_h("rik-own"), scope="Q"), ("logical_cycle_resolutions", {**row, "result_id": own, "decision_id": own, "review_intent_key": _h("rik-own")})])
    assert raw.fin(tx, ev, [raw.decision_row(tx, ev, own, committed, rik=_h("rik-own"), scope="Q"), ("logical_cycle_resolutions", {**row, "result_id": own, "decision_id": own, "review_intent_key": _h("rik-own")})])
    raw.healthy()


def test_terminal_rules(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    wid, lc = raw.wake("w1")
    newer, newer_lc = raw.wake("w2")
    assert not raw.terminal(wid, lc, "ATTEMPTS_EXHAUSTED")  # needs exactly 3 claims
    assert not raw.terminal(wid, lc, "SUPERSEDED", superseding=wid)  # not itself
    assert not raw.terminal(wid, lc, "SUPERSEDED", superseding="wakeup-" + _h("unknown"))
    assert not raw.terminal(wid, lc, "EXPIRED_WAKE", key_cycle=newer_lc)  # the key must be this Wake's cycle
    claimed, claimed_lc = raw.wake("w3")
    raw.claim(claimed, claimed_lc, 1)
    assert not raw.terminal(claimed, claimed_lc, "SUPERSEDED", superseding=newer)  # only an unclaimed Wake
    assert raw.terminal(wid, lc, "SUPERSEDED", superseding=newer)
    assert raw.terminal(claimed, claimed_lc, "EXPIRED_WAKE")  # no SQL clock condition (§2-7)
    # CYCLE_FAILED only from a FAILED_TERMINAL with its digest.
    cancelled = "wake-terminal-" + _h(wid)
    assert not raw.message(cancelled, _h(f"t-{wid}"), "CYCLE_FAILED")[1]
    failed = "wake-terminal-" + _h(claimed)
    assert not raw.message(failed, H, "CYCLE_FAILED")[1]
    assert raw.message(failed, _h(f"t-{claimed}"), "CYCLE_FAILED")[1]
    triggers = {row[0] for row in raw.c.execute("SELECT name FROM sqlite_schema WHERE type = 'trigger'")}
    for table in TRIGGER_TABLES:
        assert {f"{table}_insert", f"{table}_no_update", f"{table}_no_delete"} <= triggers
        if raw.c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:
            assert raw.refused(f"UPDATE {table} SET record_digest = record_digest"), table
            assert raw.refused(f"DELETE FROM {table}"), table
    raw.healthy()


def test_abort_and_finalize_completeness_for_every_new_kind(tmp_path: Path) -> None:
    raw = Raw(tmp_path)
    wid, lc = raw.wake("w1")
    # FINALIZE without the row is refused; ABORT with a row is refused.
    tx, _ = raw.prep("R19_WAKE_CLAIM", b"{}", (f"wake-attempt:{wid}:1",))
    assert raw.refused("UPDATE history_operations SET phase = 'FINALIZED' WHERE transaction_id = ?", tx)
    raw.abort(tx)
    assert raw.c.execute("SELECT count(*) FROM history_operation_keys WHERE transaction_id = ? AND live = 1", (tx,)).fetchone()[0] == 0
    body = b'{"c":1}'
    tx, _ = raw.prep("R19_WAKE_CLAIM", b'{"e":' + body + b"}", (f"wake-attempt:{wid}:1",))
    raw.c.execute("BEGIN")
    raw.c.execute("INSERT INTO wake_claims VALUES ('wake-claim-' || ?, ?, ?, ?, ?, ?, ?, 1)", (_h("c"), INC, tx, body, H, wid, lc))
    assert raw.refused("UPDATE history_operations SET phase = 'ABORTED' WHERE transaction_id = ?", tx)
    raw.c.execute("ROLLBACK")
    for kind in sorted(kind.value for kind in RECORD_OPERATION_KINDS if kind.value.startswith(("R19", "R16_CYCLE", "R16_PLAN"))):
        lineage = ("L", _h(kind), GENESIS) if kind == "R16_CYCLE_DECISION" else (None, None, None)
        tx, _ = raw.prep(kind, b"{}", (f"probe:{kind}",), lineage=lineage)
        assert raw.refused("UPDATE history_operations SET phase = 'FINALIZED' WHERE transaction_id = ?", tx), kind
        raw.abort(tx)
    raw.healthy()


# ------------------------------------------------- store API, real values


EVENT_SEQ = 50


class Cycle:
    """One GOAL Wake of incarnation INC through the real R16 / R19 values."""

    def __init__(self, name: str = "goal-1", incarnation: str = INC) -> None:
        self.incarnation = incarnation
        self.wake = _wake(WakeCause.GOAL_ADMITTED, goal_id=name, incarnation_id=incarnation, cause_event_id=_uuid(f"cause-{name}-{incarnation}"))
        self.name = name
        self.seq = EVENT_SEQ
        self.claims: list[WakeClaimV1] = []

    def next(self) -> tuple[str, int]:
        self.seq += 1
        return _uuid(f"{self.name}-{self.incarnation}-{self.seq}"), self.seq

    def op(self, kind: K, record_kind: R, record: object, event: tuple[str, int] | None = None) -> HistoryRecordOperationV1:
        event_id, sequence = event or self.next()
        return HistoryRecordOperationV1(kind, self.incarnation, event_id, sequence, "EVENT_MUTATION", record_kind, record)

    def materialize(self) -> HistoryRecordOperationV1:
        return self.op(K.R19_WAKE_MATERIALIZE, R.WAKE, self.wake)

    def claim(self, attempt: int) -> HistoryRecordOperationV1:
        event_id, sequence = self.next()
        claim = WakeClaimV1.for_wake(self.wake, attempt, claim_event_id=event_id, claim_sequence=sequence, claimed_utc=f"2026-11-16T0{1 + attempt}:00:00Z", timing=TIMING)
        self.claims.append(claim)
        return self.op(K.R19_WAKE_CLAIM, R.WAKE_CLAIM, claim, (event_id, sequence))

    @property
    def review_intent_key(self) -> str:
        from suzka.r16.cycle_contracts import wake_review_intent_key

        return wake_review_intent_key(self.wake.logical_cycle_id, self.wake.frozen_predecessor, PolicyKind.SUCCESSOR_POLICY_V2, self.wake.evidence_fingerprint)

    def hold(self, claim: WakeClaimV1 | None = None) -> HistoryRecordOperationV1:
        claim = claim or self.claims[-1]
        event_id, sequence = self.next()
        receipt = HoldReceiptV1(
            claim.logical_cycle_id, self.review_intent_key, claim.claim_id, claim.claim_digest, claim.claim_event_id,
            event_id, sequence, __import__("suzka.r16.cycle_contracts", fromlist=["HoldReason"]).HoldReason.STALE_SOURCE,
            None, "9" * 64, H, "2026-11-16T05:00:00Z",
        )
        closure = CycleClosureV1(LogicalCycleResolutionV1.of_hold(self.incarnation, receipt), receipt)
        return self.op(K.R16_CYCLE_RECEIPT, R.CYCLE_CLOSURE, closure, (event_id, sequence))

    def existing(self, head: Any) -> HistoryRecordOperationV1:
        claim = self.claims[-1]
        event_id, sequence = self.next()
        receipt = ExistingDecisionReceiptV1(
            claim.logical_cycle_id, self.review_intent_key, claim.claim_id, claim.claim_digest, event_id, sequence,
            "2026-11-16T05:00:00Z", "9" * 64, head.decision_id, head.content_digest, head.policy_row,
        )
        closure = CycleClosureV1(LogicalCycleResolutionV1.of_existing(self.incarnation, receipt), receipt)
        return self.op(K.R16_CYCLE_RECEIPT, R.CYCLE_CLOSURE, closure, (event_id, sequence))

    def decision(self, *, predecessor: tuple[str, str | None] = (GENESIS, None), row: PolicyRow = PolicyRow.ROW_4) -> HistoryRecordOperationV1:
        claim = self.claims[-1]
        event_id, sequence = self.next()
        event = replace(_event("unused", sequence), event_id=event_id)
        decision = _committed(event, incarnation=self.incarnation, intent=self.review_intent_key, predecessor=predecessor, row=row, kind=PolicyKind.SUCCESSOR_POLICY_V3)
        closure = CycleClosureV1(LogicalCycleResolutionV1.of_decision(claim.logical_cycle_id, claim.claim_id, claim.claim_digest, decision), decision)
        return self.op(K.R16_CYCLE_DECISION, R.CYCLE_CLOSURE, closure, (event_id, sequence))

    def outcome(self, outcome: AttemptOutcome = AttemptOutcome.REFUSED) -> HistoryRecordOperationV1:
        reason = RefuseReason.SOURCE_UNAVAILABLE if outcome is AttemptOutcome.REFUSED else None
        return self.op(K.R19_WAKE_ATTEMPT_OUTCOME, R.WAKE_ATTEMPT_OUTCOME, WakeAttemptOutcomeV1(self.claims[-1].claim_id, outcome, reason))

    def terminal(self, reason: TerminalReason, superseding: WakeV1 | None = None) -> HistoryRecordOperationV1:
        event_id, sequence = self.next()
        kind = WakeTerminalKind.CANCELLED if reason is TerminalReason.SUPERSEDED else WakeTerminalKind.FAILED_TERMINAL
        terminal = WakeTerminalV1(self.wake.wakeup_id, self.wake.logical_cycle_id, kind, reason, None if superseding is None else superseding.wakeup_id, event_id, sequence)
        return self.op(K.R19_WAKE_TERMINAL, R.WAKE_TERMINAL, terminal, (event_id, sequence))

    def message(self, closing: HistoryRecordOperationV1) -> HistoryRecordOperationV1:
        record = closing.record
        if isinstance(record, CycleClosureV1):
            decision = record.decision
            message = OutboxMessageV1.for_completion(self.wake, record.resolution, None if decision is None else decision.policy_row, created_utc=CREATED, expires_utc=EXPIRES)
        else:
            assert isinstance(record, WakeTerminalV1)
            message = OutboxMessageV1.for_failed_cycle(self.wake, record, created_utc=CREATED, expires_utc=EXPIRES)
        return self.op(K.R19_OUTBOX_EMIT, R.OUTBOX_MESSAGE, message)

    def ack(self, message: OutboxMessageV1, context: str = "context-1") -> HistoryRecordOperationV1:
        event_id, sequence = self.next()
        ack = InboxAckV1(message.message_id, context, "session-1", AckStatus.ACKNOWLEDGED, event_id, sequence, CREATED)
        return self.op(K.R19_INBOX_ACK, R.INBOX_ACK, ack, (event_id, sequence))


def _plan_hold(name: str = "p", *, incarnation: str = INC, row_changes: dict[str, Any] | None = None) -> PlanHoldReceiptV1:
    event_id = _uuid(f"plan-{name}")
    fields: dict[str, Any] = {
        "event_id": event_id, "event_sequence": 70, "now_utc": "2026-11-16T03:00:00Z",
        "interpretation": Interpretation.CONTRADICTORY, "interpretation_stage": InterpretationStage.S2,
    }
    fields.update(row_changes or {})
    sealed = replace(_active(), **fields)
    return PlanHoldReceiptV1(
        incarnation_id=incarnation, lineage_scope="goal-lineage-1", review_intent_key=_h(f"plan-rik-{name}"),
        predecessor_key=GENESIS, predecessor_digest=None, event_id=event_id, event_sequence=70,
        policy_kind=PolicyKind.SUCCESSOR_POLICY_V2, policy_row=SuccessorRow.ROW_2B,
        hold_reason=ReasonCode.CONFLICTING_EVIDENCE, policy_input_bytes=sealed.canonical_bytes(),
        prepare_checked_at_utc="2026-11-16T03:00:00Z",
    )


def _plan_hold_op(receipt: PlanHoldReceiptV1) -> HistoryRecordOperationV1:
    return HistoryRecordOperationV1(K.R16_PLAN_HOLD, receipt.incarnation_id, receipt.event_id, receipt.event_sequence, "EVENT_MUTATION", R.PLAN_HOLD, receipt)


def _rows_match(store: SubjectHistoryStore, operation: HistoryRecordOperationV1, transaction_id: str) -> None:
    for row in operation.rows:
        stored = store.record_rows(row.table, operation.incarnation_id, transaction_id)
        assert len(stored) == 1 and all(stored[0][name] == value for name, value in row.columns), row.table
    assert store.record_rows_match(operation.incarnation_id, transaction_id, operation)


def test_a_held_cycle_with_attempt_outcomes_outbox_and_ack(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cycle = Cycle()
    operations = [cycle.materialize(), cycle.claim(1), cycle.outcome(), cycle.claim(2), cycle.outcome(AttemptOutcome.DEFERRED_BUDGET), cycle.claim(3)]
    hold = cycle.hold()
    operations.append(hold)
    message = cycle.message(hold)
    operations += [message, cycle.ack(message.record)]  # type: ignore[arg-type]
    for operation in operations:
        assert HistoryRecordOperationV1.from_bytes(operation.canonical_bytes()) == operation
        _rows_match(store, operation, _apply(store, operation))
    assert store.has_r19_operations() and not store.has_record_operations() and not store.has_plan_hold_operations()
    key = f"cycle:{INC}:{cycle.wake.logical_cycle_id}"
    assert store.live_keys((key,)) == {key}
    # The refuse reason survives a restart (it is the only durable copy, D-1).
    path = store.path
    store.close()
    reopened = SubjectHistoryStore.open(path, max_page_count=PAGES)
    rows = reopened.record_rows("wake_attempt_outcomes", INC, _uuid(f"tx-{operations[2].event_id}-{operations[2].processing_sequence}"))
    assert WakeAttemptOutcomeV1.from_json(rows[0]["record_bytes"]).reason is RefuseReason.SOURCE_UNAVAILABLE  # type: ignore[arg-type]


def test_a_cycle_decision_fills_its_lineage_and_competes_for_the_slots(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cycle = Cycle()
    for operation in (cycle.materialize(), cycle.claim(1)):
        _apply(store, operation)
    closure = cycle.decision()
    decision = closure.cycle_decision
    assert decision is not None
    # A PLAN_PREVIEW Decision PREPARED on the same predecessor slot first.
    from suzka.r16.history_codec import history_operation_digest

    event = _event("plan-preview", 90)
    preview = _committed(event, incarnation=INC, intent="7" * 64, kind=PolicyKind.SUCCESSOR_POLICY_V3)
    preview_tx = _uuid("preview")
    store.insert_prepared(transaction_id=preview_tx, event_id=event.event_id, processing_sequence=90, transaction_kind="EVENT_MUTATION", operation_digest=history_operation_digest(preview), decision=preview)
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, closure)
    store.mark_aborted(INC, preview_tx)
    transaction_id = _apply(store, closure)
    row = store.operation(INC, transaction_id)
    assert row is not None and (row.lineage_scope, row.review_intent_key, row.predecessor_key) == (decision.lineage_scope, decision.review_intent_key, decision.predecessor_key)
    finalized = store.finalized_decision(INC, transaction_id)
    assert finalized is not None and finalized.committed_bytes == decision.canonical_bytes()
    _rows_match(store, closure, transaction_id)
    sources = sqlite3.connect(store.path).execute("SELECT count(*) FROM history_source_refs WHERE decision_id = ?", (decision.decision_id,)).fetchone()[0]
    assert sources == len(decision.source_refs)
    # ... and the other order: the FINALIZED cycle Decision holds the slot.
    event = _event("plan-preview-2", 91)
    late = _committed(event, incarnation=INC, intent="6" * 64, kind=PolicyKind.SUCCESSOR_POLICY_V3)
    with pytest.raises(HistoryStoreConflict):
        store.insert_prepared(transaction_id=_uuid("preview-2"), event_id=event.event_id, processing_sequence=91, transaction_kind="EVENT_MUTATION", operation_digest=history_operation_digest(late), decision=late)
    message = cycle.message(closure)
    _apply(store, message)


def test_an_existing_receipt_and_a_failed_cycle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    head = _committed(_event("head", 5), incarnation=INC)
    confirmed = Cycle("goal-confirmed")
    for operation in (confirmed.materialize(), confirmed.claim(1)):
        _apply(store, operation)
    existing = confirmed.existing(head)
    _apply(store, existing)
    _apply(store, confirmed.message(existing))
    failed = Cycle("goal-failed")
    for operation in (failed.materialize(), failed.claim(1), failed.claim(2), failed.claim(3)):
        _apply(store, operation)
    terminal = failed.terminal(TerminalReason.ATTEMPTS_EXHAUSTED)
    _apply(store, terminal)
    cycle_failed = failed.message(terminal)
    _apply(store, cycle_failed)
    with pytest.raises(HistoryStoreConflict):  # the cycle is closed: no claim, at PREPARE
        _prepare(store, failed.claim(3))


def test_an_attempt_outcome_and_a_resolution_collide_at_prepare(tmp_path: Path) -> None:
    """D1 r2 §2-4 (r2 final review): both reserve attempt-end:<claim_id>."""

    store = _store(tmp_path)
    first = Cycle("goal-a")
    for operation in (first.materialize(), first.claim(1)):
        _apply(store, operation)
    _prepare(store, first.outcome())
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, first.hold())
    second = Cycle("goal-b")
    for operation in (second.materialize(), second.claim(1)):
        _apply(store, operation)
    _prepare(store, second.hold())
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, second.outcome())
    with pytest.raises(HistoryStoreConflict):  # and a terminal of the same cycle
        _prepare(store, second.terminal(TerminalReason.EXPIRED_WAKE))


def test_wake_and_ack_keys_are_reserved_at_prepare(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cycle = Cycle()
    _apply(store, cycle.materialize())
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, cycle.materialize())  # wake: and wake-cause:
    outcome_wake = _wake(WakeCause.FEEDBACK_ADMITTED, goal_id="goal-o")
    other_cause = _wake(WakeCause.OUTCOME_VERIFIED_NO_FEEDBACK, goal_id="goal-o", evidence_fingerprint="e" * 64, cause_event_id=_uuid("other-cause"))
    first = HistoryRecordOperationV1(K.R19_WAKE_MATERIALIZE, INC, _uuid("m1"), 60, "EVENT_MUTATION", R.WAKE, outcome_wake)
    second = HistoryRecordOperationV1(K.R19_WAKE_MATERIALIZE, INC, _uuid("m2"), 61, "EVENT_MUTATION", R.WAKE, other_cause)
    assert f"wake-outcome:{INC}:outcome-1:1" in first.reserved_keys and f"wake-outcome:{INC}:outcome-1:1" in second.reserved_keys
    _prepare(store, first)
    with pytest.raises(HistoryStoreConflict):  # one Wake per Outcome revision across both causes
        _prepare(store, second)
    for attempt in (1, 2, 3):
        _apply(store, cycle.claim(attempt))
    hold = cycle.hold()
    _apply(store, hold)
    message = cycle.message(hold)
    _apply(store, message)
    _apply(store, cycle.ack(message.record))  # type: ignore[arg-type]
    with pytest.raises(HistoryStoreConflict):
        _prepare(store, cycle.ack(message.record))  # type: ignore[arg-type]
    wrong = cycle.ack(message.record, context="context-2")  # type: ignore[arg-type]
    other = Cycle("goal-z")
    for operation in (other.materialize(), other.claim(1)):
        _apply(store, operation)
    hold = other.hold()
    _apply(store, hold)
    message = other.message(hold)
    _apply(store, message)
    wrong = other.ack(message.record, context="context-2")  # type: ignore[arg-type]
    transaction_id = _prepare(store, wrong)
    with pytest.raises(HistoryStoreConflict):  # F11: the wrong Context
        store.finalize_record(transaction_id=transaction_id, operation=wrong, receipt=_receipt(transaction_id, wrong.event_id, wrong.processing_sequence))


def test_a_previous_incarnations_reservation_does_not_block(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = Cycle(incarnation=OLD)
    for operation in (old.materialize(), old.claim(1)):
        _apply(store, operation)
    _prepare(store, old.hold())  # left PREPARED by the old incarnation
    current = Cycle()
    for operation in (current.materialize(), current.claim(1), current.hold()):
        _apply(store, operation)


def test_the_envelope_binds_kind_event_incarnation_and_horizon() -> None:
    cycle = Cycle()
    claim_op = cycle.claim(1)
    with pytest.raises(ValueError, match="another event"):
        replace(claim_op, processing_sequence=999)
    hold = cycle.hold()
    with pytest.raises(ValueError, match="R16_CYCLE_DECISION"):
        replace(hold, operation_kind=K.R16_CYCLE_DECISION)
    decision = cycle.decision()
    with pytest.raises(ValueError, match="R16_CYCLE_DECISION"):
        replace(decision, operation_kind=K.R16_CYCLE_RECEIPT)
    with pytest.raises(ValueError, match="another incarnation"):
        replace(cycle.materialize(), incarnation_id=OLD)
    with pytest.raises(ValueError, match="horizon"):
        replace(cycle.materialize(), processing_sequence=cycle.wake.horizon_sequence)
    with pytest.raises(ValueError, match="REFUSED or DEFERRED_BUDGET"):
        cycle.op(K.R19_WAKE_ATTEMPT_OUTCOME, R.WAKE_ATTEMPT_OUTCOME, WakeAttemptOutcomeV1(cycle.claims[0].claim_id, AttemptOutcome.LEASE_EXPIRED, None))
    with pytest.raises(ValueError, match="does not belong"):
        cycle.op(K.R19_WAKE_CLAIM, R.WAKE, cycle.wake)
    with pytest.raises(ValueError, match="does not match"):
        CycleClosureV1(replace(hold.record.resolution, claim_digest="0" * 64), hold.record.result)  # type: ignore[union-attr]
    with pytest.raises(TypeError, match="result kind"):
        CycleClosureV1(hold.record.resolution, decision.record.result)  # type: ignore[union-attr]
    for operation in (hold, decision, cycle.materialize(), cycle.outcome(), cycle.terminal(TerminalReason.EXPIRED_WAKE), _plan_hold_op(_plan_hold())):
        assert HistoryRecordOperationV1.from_bytes(operation.canonical_bytes()) == operation
    assert set(hold.reserved_keys) == {f"cycle:{INC}:{cycle.wake.logical_cycle_id}", f"attempt-end:{cycle.claims[0].claim_id}"}
    assert cycle.terminal(TerminalReason.EXPIRED_WAKE).reserved_keys == (f"cycle:{INC}:{cycle.wake.logical_cycle_id}",)


# ---------------------------------------------------------------- gating


def _one_of_each(store: SubjectHistoryStore) -> dict[K, HistoryRecordOperationV1]:
    """A PREPARE-able operation of every R19 / cycle kind (the parents are
    applied first with every switch on)."""

    cycle = Cycle()
    for operation in (cycle.materialize(), cycle.claim(1)):
        _apply(store, operation)
    hold = cycle.hold()
    _apply(store, hold)
    message = cycle.message(hold)
    _apply(store, message)
    other = Cycle("goal-g")
    _apply(store, other.materialize())
    _apply(store, other.claim(1))
    third = Cycle("goal-h")
    for operation in (third.materialize(), third.claim(1)):
        _apply(store, operation)
    third_hold = third.hold()
    _apply(store, third_hold)
    return {
        K.R19_WAKE_MATERIALIZE: Cycle("goal-new").materialize(),
        K.R19_WAKE_CLAIM: other.claim(2),
        K.R19_WAKE_ATTEMPT_OUTCOME: other.outcome(),
        K.R16_CYCLE_DECISION: other.decision(),
        K.R16_CYCLE_RECEIPT: other.hold(),
        K.R19_WAKE_TERMINAL: other.terminal(TerminalReason.EXPIRED_WAKE),
        K.R19_OUTBOX_EMIT: third.message(third_hold),
        K.R19_INBOX_ACK: cycle.ack(message.record),  # type: ignore[arg-type]
        K.R16_PLAN_HOLD: _plan_hold_op(_plan_hold()),
    }


def test_each_family_follows_only_its_own_switch(tmp_path: Path) -> None:
    store = _store(tmp_path, r18=False, r19=True, holds=True)
    operations = _one_of_each(store)
    path = store.path
    store.close()
    assert set(operations) == set(R19_GATED_KINDS | PLAN_HOLD_GATED_KINDS) and len(R19_GATED_KINDS) == 8
    r18_operation = _steps(Chain(_committed(_event("gate", 3))))[0]
    for r18, r19, holds in ((True, False, False), (False, True, False), (False, False, True), (False, False, False)):
        store = SubjectHistoryStore.open(path, max_page_count=PAGES, r18_enabled=r18, r19_enabled=r19, plan_hold_receipts=holds)
        for kind, operation in operations.items():
            allowed = holds if kind in PLAN_HOLD_GATED_KINDS else r19
            try:
                transaction_id = _prepare(store, operation)
            except HistoryStoreRecordsDisabled:
                assert not allowed, kind
            else:
                assert allowed, kind
                store.mark_aborted(operation.incarnation_id, transaction_id)
        if r18:
            store.mark_aborted(INCARNATION, _prepare(store, r18_operation))
        else:
            with pytest.raises(HistoryStoreRecordsDisabled, match="r18"):
                _prepare(store, r18_operation)
        store.close()


def test_finalize_is_not_gated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cycle = Cycle()
    _apply(store, cycle.materialize())
    claim = cycle.claim(1)
    transaction_id = _prepare(store, claim)
    receipt = _plan_hold_op(_plan_hold())
    hold_tx = _prepare(store, receipt)
    path = store.path
    store.close()
    off = SubjectHistoryStore.open(path, max_page_count=PAGES)  # every switch off
    off.finalize_record(transaction_id=transaction_id, operation=claim, receipt=_receipt(transaction_id, claim.event_id, claim.processing_sequence))
    off.finalize_record(transaction_id=hold_tx, operation=receipt, receipt=_receipt(hold_tx, receipt.event_id, receipt.processing_sequence))
    with pytest.raises(HistoryStoreRecordsDisabled):
        _prepare(off, cycle.claim(2))
    assert off.has_r19_operations() and off.has_plan_hold_operations() and not off.has_record_operations()


def test_the_startup_and_ledger_checks_see_only_r18_data(tmp_path: Path) -> None:
    store = _store(tmp_path, r18=True)
    assert not (store.has_record_operations() or store.has_r19_operations() or store.has_plan_hold_operations())
    _apply(store, _plan_hold_op(_plan_hold()))
    assert store.has_plan_hold_operations() and not store.has_record_operations() and not store.has_r19_operations()
    _apply(store, Cycle().materialize())
    assert store.has_r19_operations() and not store.has_record_operations()
    _apply(store, _steps(Chain(_decision(store)))[0])
    assert store.has_record_operations()
    assert R18_GATED_KINDS.isdisjoint(R19_GATED_KINDS | PLAN_HOLD_GATED_KINDS)


def test_a_new_kind_is_refused_at_capacity(tmp_path: Path) -> None:
    store = _store(tmp_path, pages=112)  # an empty v4 store takes 99 pages
    with pytest.raises(HistoryStoreCapacityError):
        _prepare(store, Cycle().materialize())
    assert not store.has_r19_operations()


def test_the_settings_default_off() -> None:
    from suzka.config.schema import R19Settings, SubjectHistorySettings

    assert SubjectHistorySettings().plan_hold_receipts is False and R19Settings().enabled is False


# ------------------------------------------------------- plan hold receipt


def test_a_plan_hold_receipt_seals_a_replaying_hold_row(tmp_path: Path) -> None:
    receipt = _plan_hold()
    assert PlanHoldReceiptV1.from_json(receipt.canonical_bytes()) == receipt
    assert receipt.hold_id.startswith("plan-hold-") and len(receipt.canonical_bytes()) <= MAX_PLAN_HOLD_BYTES
    for changes, match in (
        ({"policy_row": SuccessorRow.ROW_1}, "replay"),
        ({"hold_reason": ReasonCode.INSUFFICIENT_EVIDENCE}, "replay"),
        ({"policy_kind": PolicyKind.SUCCESSOR_POLICY_V3}, "replay"),
        ({"event_sequence": 71}, "another event"),
        ({"prepare_checked_at_utc": "2026-11-16T03:00:01Z"}, "another event"),
        ({"policy_row": SuccessorRow.ROW_4}, "HOLD row"),
        ({"policy_kind": PolicyKind.SUCCESSOR_POLICY_V1}, "V2 or V3"),
        ({"predecessor_digest": H}, "GENESIS"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(receipt, **changes)
    with pytest.raises(ValueError, match="replay"):  # evidence that evaluates to another row
        _plan_hold("one", row_changes={"interpretation": Interpretation.SETTLED})
    store = _store(tmp_path)
    operation = _plan_hold_op(receipt)
    assert operation.reserved_keys == (f"plan-hold:{INC}:{receipt.review_intent_key}",)
    _rows_match(store, operation, _apply(store, operation))
    again = _plan_hold_op(replace(receipt, lineage_scope="goal-lineage-2"))
    with pytest.raises(HistoryStoreConflict):  # one per (incarnation, review_intent_key)
        _prepare(store, again)


# --------------------------------------------------------------- migration


def test_a_v3_store_migrates_to_v4_and_keeps_every_row(tmp_path: Path) -> None:
    path, decisions = _v3_store(tmp_path, records=True)
    before = _all_rows(path)
    original = path.read_bytes()
    meta_before = sqlite3.connect(path).execute("SELECT * FROM history_meta").fetchall()
    result = migrate(path)
    assert result.from_version == 3 and result.compacted and ".v3-" in result.backup.name
    assert result.backup.read_bytes() == original and os.stat(result.backup).st_mode & 0o777 == 0o600
    after_rows = _all_rows(path)
    assert {t: r for t, r in after_rows.items() if t in before} == before
    assert all(after_rows[t] == [] for t in TRIGGER_TABLES)
    after = sqlite3.connect(path)
    assert after.execute(_SCHEMA_QUERY).fetchall() == _expected_schema(SCHEMA_STATEMENTS)
    assert after.execute("PRAGMA foreign_key_check").fetchall() == []
    meta_after = after.execute("SELECT * FROM history_meta").fetchall()
    assert [(r[0], *r[3:]) for r in meta_after] == [(r[0], *r[3:]) for r in meta_before]
    assert [r[1:3] for r in meta_after] == [(4, SCHEMA_HASH)]
    after.close()
    store = SubjectHistoryStore.open(path, max_page_count=PAGES, r18_enabled=True)
    operations = {row.event_id: row for row in store.operations(INCARNATION)}
    for decision in decisions:
        finalized = store.finalized_decision(INCARNATION, operations[decision.issuer_event_id].transaction_id)
        assert finalized is not None and finalized.committed_bytes == decision.canonical_bytes()
    assert store.has_record_operations()
    with pytest.raises(HistoryMigrationError, match="already schema v4"):
        migrate(path)


def test_an_empty_v3_store_migrates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.path
    store.close()
    _downgrade(path, 3, SCHEMA_STATEMENTS_V3, SCHEMA_HASH_V3)
    assert migrate(path).from_version == 3
    assert SubjectHistoryStore.open(path, max_page_count=PAGES).meta().schema_version == 4


@pytest.mark.parametrize("stage", ["backed-up", "dropped", "rebuilt:history_meta", "rebuilt:history_operations", "rebuilt:history_decisions", "created", "verified"])
def test_a_crash_before_commit_leaves_the_v3_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    path, _ = _v3_store(tmp_path, records=True)
    snapshot = sqlite3.connect(path).execute(_SCHEMA_QUERY).fetchall(), _all_rows(path)

    def crash(name: str) -> None:
        if name == stage:
            raise KeyboardInterrupt(name)

    monkeypatch.setattr(history_migration, "_STAGE_HOOK", crash)
    with pytest.raises(KeyboardInterrupt):
        migrate(path)
    monkeypatch.setattr(history_migration, "_STAGE_HOOK", None)
    assert (sqlite3.connect(path).execute(_SCHEMA_QUERY).fetchall(), _all_rows(path)) == snapshot
    with pytest.raises(HistoryStoreMigrationRequired, match="v3"):
        SubjectHistoryStore.open(path, max_page_count=PAGES)
    migrate(path, now=datetime(2030, 1, 1, tzinfo=UTC))
    assert SubjectHistoryStore.open(path, max_page_count=PAGES).meta().schema_version == 4


@pytest.mark.parametrize("stage", ["backed-up", "dropped", "rebuilt:history_meta", "rebuilt:history_operations", "rebuilt:history_decisions", "created", "verified"])
def test_a_killed_v3_migration_leaves_v3_and_a_rerun_migrates(tmp_path: Path, stage: str) -> None:
    """A real crash (os._exit with a hot journal) at each stage."""

    path, _ = _v3_store(tmp_path, records=True)
    rows = _all_rows(path)
    script = (
        "import os\n"
        "from pathlib import Path\n"
        "from suzka.r16 import history_migration as m\n"
        f"m._STAGE_HOOK = lambda name: os._exit(9) if name == {stage!r} else None\n"
        f"m.migrate(Path({str(path)!r}))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], check=False)
    assert completed.returncode == 9
    with pytest.raises(HistoryStoreMigrationRequired, match="v3"):
        SubjectHistoryStore.open(path, max_page_count=PAGES)
    assert _all_rows(path) == rows
    assert history_migration.main(["--max-page-count", str(PAGES), str(path)]) == 0
    assert SubjectHistoryStore.open(path, max_page_count=PAGES).meta().schema_version == 4
    assert history_migration.main(["--max-page-count", str(PAGES), str(path)]) == 1  # after COMMIT: refused


def test_a_v3_row_that_breaks_a_null_safe_check_rolls_back(tmp_path: Path) -> None:
    path, _ = _v3_store(tmp_path)
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute(
        "INSERT INTO history_operations VALUES (?, ?, ?, 99, 'K', 'subject.history', 'R16_DECISION_ISSUE', ?, 'PREPARED', x'7b7d', ?, 'X', NULL, 'GENESIS')",
        (INC, str(uuid4()), str(uuid4()), H, H),
    )
    connection.close()
    before = path.read_bytes()
    with pytest.raises(HistoryStoreConflict):
        migrate(path)
    assert path.read_bytes() == before


def test_the_page_margin_is_always_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    path, _ = _v3_store(tmp_path)
    pages = sqlite3.connect(path).execute("PRAGMA page_count").fetchone()[0]
    before = path.read_bytes()
    with pytest.raises(HistoryMigrationError, match="max_page_count"):
        migrate(path, max_page_count=2 * pages + 63)
    assert path.read_bytes() == before
    # Without --max-page-count the configured value is used.
    config = tmp_path / "config.yaml"
    import yaml  # type: ignore[import-untyped]

    raw = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    raw["subject_history"]["max_page_count"] = 2 * pages + 63
    config.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("SUZKA_CONFIG_PATH", str(config))
    assert history_migration.main([str(path)]) == 1
    assert "max_page_count" in capsys.readouterr().err and path.read_bytes() == before
    raw["subject_history"]["max_page_count"] = 2 * pages + 64
    config.write_text(yaml.safe_dump(raw))
    assert history_migration.main([str(path)]) == 0


def test_the_cap_is_set_on_the_migration_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the margin check bypassed, the rebuild overflows the cap: SQLITE_FULL rolls back."""

    path, _ = _v3_store(tmp_path, records=True)
    pages = sqlite3.connect(path).execute("PRAGMA page_count").fetchone()[0]
    before = path.read_bytes()
    monkeypatch.setattr(history_migration, "_MIGRATION_PAGES", -10 * pages)
    with pytest.raises(HistoryMigrationError, match="full"):
        migrate(path, max_page_count=pages + 2)
    assert path.read_bytes() == before
    with pytest.raises(HistoryMigrationError, match="larger than max_page_count"):
        migrate(path, max_page_count=pages - 1)


def test_v1_and_v2_stores_migrate_directly_to_v4(tmp_path: Path) -> None:
    for version, statements, schema_hash in ((1, SCHEMA_STATEMENTS_V1, SCHEMA_HASH_V1), (2, SCHEMA_STATEMENTS_V2, SCHEMA_HASH_V2)):
        path, decisions = _v4_store_with_decisions(tmp_path / f"v{version}")
        before = _all_rows(path)
        _downgrade(path, version, statements, schema_hash)
        result = migrate(path)
        assert result.from_version == version
        after = _all_rows(path)
        assert {t: after[t] for t in before} == before
        store = SubjectHistoryStore.open(path, max_page_count=PAGES)
        assert store.meta().schema_version == 4
        operations = {row.event_id: row for row in store.operations(INCARNATION)}
        finalized = store.finalized_decision(INCARNATION, operations[decisions[-1].issuer_event_id].transaction_id)
        assert finalized is not None and finalized.committed_bytes == decisions[-1].canonical_bytes()


def test_the_rebuilt_tables_are_exactly_the_changed_ones() -> None:
    """D1 r2 §3-3: a table whose CREATE text differs must be rebuilt."""

    def creates(statements: tuple[str, ...]) -> dict[str, str]:
        return {s.split()[2]: s for s in statements if s.startswith("CREATE TABLE ")}

    current = creates(SCHEMA_STATEMENTS)
    for source in history_migration._SOURCES:
        old = creates(source.statements)
        changed = {name for name, text in old.items() if current[name] != text}
        assert set(source.rebuilt) == changed, source.version


# --------------------------------------- end to end: a 52c61ab-made v3 store

DEPLOYED = "52c61ab"


def _archive(tmp_path: Path) -> Path:
    root = Path(__file__).parents[1]
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{DEPLOYED}^{{commit}}"], check=False, capture_output=True).returncode != 0:
        pytest.skip(f"{DEPLOYED} is not in this clone")
    target = tmp_path / "deployed"
    target.mkdir()
    archive = subprocess.run(["git", "-C", str(root), "archive", DEPLOYED, "suzka", "tests", "config.yaml"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive, check=True)
    return target


def test_a_v3_store_made_by_the_deployed_code_migrates_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployed (52c61ab) server issues a Decision into a v3 store; the v4
    code refuses it until migrated, then adopts the same head, and the issuer
    returns the existing Decision. The frozen v3 text is that code's text."""

    from fastapi.testclient import TestClient

    from suzka.r16.decision_issuer import DecisionIssueStatus
    from suzka.r16.history_reader import HeadStatus

    from test_r16_ip10_decision_issuer import _app, _issue, _reader_head

    deployed = _archive(tmp_path)
    root = tmp_path / "e2e"
    script = f"""
import json, sys
sys.path[:0] = [{str(deployed)!r}, {str(deployed / "tests")!r}]
from pathlib import Path
from fastapi.testclient import TestClient
from suzka.r16.history_store import SCHEMA_STATEMENTS, HISTORY_SCHEMA_VERSION
from test_r16_ip10_decision_issuer import _app, _issue, _reader_head, _settings
from test_r16_u3a_plan_preview import _qualified_goal
assert HISTORY_SCHEMA_VERSION == 3
settings = _settings(Path({str(root)!r}))
with TestClient(_app(settings)) as client:
    state = client.app.state
    _x, gid = _qualified_goal(state)
    issued = _issue(state, gid)
    head = _reader_head(state, issued)
    print(json.dumps({{"gid": gid, "status": issued.status.value, "decision": issued.decision.decision_id,
                      "scope": issued.decision.lineage_scope, "depth": head.depth,
                      "statements": list(SCHEMA_STATEMENTS)}}))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=deployed, check=True, capture_output=True, text=True)
    made = json.loads(completed.stdout.strip().splitlines()[-1])
    assert made["status"] == DecisionIssueStatus.ISSUED.value and tuple(made["statements"]) == SCHEMA_STATEMENTS_V3
    # The v4 server reads the same root (Journal, state and SubjectHistory).
    settings = _settings_on(root)
    path = settings.subject_history.path
    before = _all_rows(path)
    with pytest.raises(Exception, match="history_migration"):
        with TestClient(_app(settings)):
            pass
    result = migrate(path)
    assert result.from_version == 3
    after = _all_rows(path)
    assert {t: after[t] for t in before} == before
    assert sqlite3.connect(path).execute(_SCHEMA_QUERY).fetchall() == _expected_schema(SCHEMA_STATEMENTS)
    with TestClient(_app(settings)) as client:
        state = client.app.state
        again = _issue(state, made["gid"])
        assert again.status is DecisionIssueStatus.EXISTING
        assert again.decision is not None and again.decision.decision_id == made["decision"]
        head = _reader_head(state, again)
        assert (head.status, head.decision.decision_id, head.depth) == (HeadStatus.ADOPTED, made["decision"], made["depth"])  # type: ignore[union-attr]


def _settings_on(root: Path) -> Any:
    """Point every path of ``settings`` from its own root at ``root``."""

    from test_r16_ip10_decision_issuer import _settings

    fresh = _settings(root.with_name(root.name + "-probe"))
    old_root = str(root.with_name(root.name + "-probe"))

    def move(value: Any) -> Any:
        if isinstance(value, Path) and str(value).startswith(old_root):
            return Path(str(value).replace(old_root, str(root), 1))
        if hasattr(value, "model_copy"):
            return value.model_copy(update={name: move(getattr(value, name)) for name in type(value).model_fields})
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    return move(fresh)
