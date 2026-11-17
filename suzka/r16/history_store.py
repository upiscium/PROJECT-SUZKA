"""R16 U4-1 / IP-14a-1 / IP-16a-1: the SubjectHistoryDB SQLite file, schema v3, trusted opener.

v1 (#376 §A/§B, #376 6738757841 / 6752207008): five tables (``history_meta``,
``history_operations``, ``history_decisions``, ``history_source_refs``,
``history_postcommit_receipts``) and the two R16 partial UNIQUE indexes.

v2 (IP-14a D1 r3, #411 comment 6910958369): the R17 / R18 source-side record
tables, the PREPARE-time key reservations (§2a), the record operation kinds
and ``SUCCESSOR_POLICY_V3``.

v3 (IP-16 D1 r3 §9, #411 comment 6916372789): Intent and forecast uniqueness
and their PREPARE reservations are per incarnation; Claim, nonce and Outcome
stay global. v2 was already deployed on eve (6ed2361), so this is a new
version, not an edit of v2.

A v1 or v2 file is never opened: it raises ``HistoryStoreMigrationRequired``
and is migrated only by the explicit ``suzka.r16.history_migration`` command
(Human decision §11-1). HOLD receipts and R19 cycle resolution are IP-14b.

Authority boundary: this module is storage only.  The single writer authority
for the file is the ``subject.history`` R07 participant; SQL CHECKs and
triggers are defence in depth and never authenticate an issuer.  A FINALIZED
row is *not* an adopted Decision until its R07 transaction and event are
terminal in the retained Journal (read-time check, U4-4).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
from threading import Lock
from typing import Final
from uuid import uuid4

from suzka.r16.history_codec import (
    MAX_PENDING_BYTES,
    MAX_POLICY_INPUT_BYTES,
    SUBJECT_HISTORY_PARTICIPANT_ID,
    CommittedDecisionV1,
    PostcommitReceiptV1,
    canonical_uuid,
    history_operation_digest,
)
from suzka.r16.contracts import MAX_JSON_BYTES, digest, identifier
from suzka.r16.history_records import (
    MAX_CHALLENGES_PER_INTENT,
    OPERATION_TABLES,
    RECORD_OPERATION_KINDS,
    RECORD_TABLES,
    HistoryOperationKind,
    HistoryRecordOperationV1,
)
from suzka.r16.history_schema_v1 import SCHEMA_HASH_V1, SCHEMA_STATEMENTS_V1
from suzka.r16.history_schema_v2 import SCHEMA_HASH_V2, SCHEMA_STATEMENTS_V2
from suzka.r17.contracts import MAX_RECORD_BYTES as _R17_RECORD_BYTES
from suzka.r18.contracts import MAX_RECORD_BYTES as _R18_RECORD_BYTES


HISTORY_SCHEMA_VERSION: Final = 3
_SQLITE_BUSY_TIMEOUT_SECONDS: Final = 5.0
# Conservative bytes a FINALIZE may add for one pending operation: the
# committed envelope, inner U1 bytes, policy input, source rows, receipt and
# their index entries, plus fixed page slack.  A heuristic upper bound, not a
# measured P_max (#376 D2-4 / D1-9: the numeric N stays a Human gate).
_FINALIZE_SLACK_BYTES: Final = 65_536


def _hex(column: str) -> str:
    return f"(length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*')"


def _uuid(column: str) -> str:
    return f"(length({column}) = 36 AND {column} NOT GLOB '*[^0-9a-f-]*')"


def _decision(column: str) -> str:
    return (
        f"(length({column}) = 73 AND substr({column}, 1, 9) = 'decision-'"
        f" AND substr({column}, 10) NOT GLOB '*[^0-9a-f]*')"
    )


_RECORD_BYTES: Final = max(_R17_RECORD_BYTES, _R18_RECORD_BYTES)
_ALL_KINDS: Final = ",".join(f"'{kind.value}'" for kind in HistoryOperationKind)
_RECORD_KINDS: Final = ",".join(
    f"'{kind.value}'" for kind in HistoryOperationKind if kind in RECORD_OPERATION_KINDS
)


def _kind_exists(kind: HistoryOperationKind) -> str:
    """Every table this record kind writes has the operation's row."""
    return " AND ".join(
        f"EXISTS (SELECT 1 FROM {table} t WHERE t.incarnation_id = OLD.incarnation_id"
        " AND t.transaction_id = OLD.transaction_id)"
        for table in OPERATION_TABLES[kind]
    )


def _gate(kind: HistoryOperationKind, bound: str = "NEW.record_bytes") -> str:
    """The row's operation is PREPARED, of this kind, and its pending envelope
    contains the row's canonical bytes (the record is embedded verbatim)."""
    return (
        "NOT EXISTS (SELECT 1 FROM history_operations o"
        " WHERE o.incarnation_id = NEW.incarnation_id AND o.transaction_id = NEW.transaction_id"
        f" AND o.phase = 'PREPARED' AND o.operation_kind = '{kind.value}'"
        f" AND instr(o.pending_bytes, {bound}) > 0)"
    )


def _record_table(name: str, columns: str, constraints: str = "") -> str:
    return f"""CREATE TABLE {name} (
    record_id TEXT PRIMARY KEY NOT NULL CHECK (length(record_id) BETWEEN 1 AND 192),
    incarnation_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    record_bytes BLOB NOT NULL CHECK (length(record_bytes) BETWEEN 2 AND {_RECORD_BYTES}),
    record_digest TEXT NOT NULL CHECK {_hex("record_digest")},
{columns}
    FOREIGN KEY (incarnation_id, transaction_id)
        REFERENCES history_operations (incarnation_id, transaction_id){constraints}
) STRICT"""


def _immutable(table: str) -> tuple[str, str]:
    return (
        f"""CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END""",
        f"""CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END""",
    )


_K = HistoryOperationKind
_DIRECTION = "direction TEXT NOT NULL CHECK (direction IN ('FORWARD', 'INVERSE')),"

_ROLES = (
    "'GOAL','VALUE','CONTEXT','EXPERIENCE','COMMITMENT','FEEDBACK','OUTCOME',"
    "'R15_NARRATIVE_REVISION','MOTIVATION'"
)

# ------------------------------------------------------------ v2 additions
# IP-14a D1 r3 §2 / §2a / §2b / §4. Tables first, then indexes, then
# triggers, so every trigger body names existing tables.
V2_STATEMENTS: Final[tuple[str, ...]] = (
    """CREATE TABLE history_operation_keys (
    reserved_key TEXT NOT NULL CHECK (length(reserved_key) BETWEEN 1 AND 256),
    incarnation_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    live INTEGER NOT NULL CHECK (live IN (0, 1)),
    PRIMARY KEY (incarnation_id, transaction_id, reserved_key),
    FOREIGN KEY (incarnation_id, transaction_id)
        REFERENCES history_operations (incarnation_id, transaction_id)
) STRICT""",
    _record_table(
        "action_intents",
        f"""    {_DIRECTION}
    obligation_group_key TEXT NOT NULL CHECK {_hex("obligation_group_key")},
    operation_key TEXT NOT NULL CHECK {_hex("operation_key")},
    forward_operation_key TEXT NOT NULL CHECK {_hex("forward_operation_key")},
    decision_id TEXT REFERENCES history_decisions (decision_id),
    UNIQUE (incarnation_id, operation_key),
    UNIQUE (incarnation_id, obligation_group_key, direction),
    CHECK ((direction = 'FORWARD') = (decision_id IS NOT NULL)),
    CHECK (direction = 'INVERSE' OR forward_operation_key = operation_key),""",
    ),
    _record_table(
        "action_forecasts",
        f"""    action_intent_id TEXT NOT NULL UNIQUE REFERENCES action_intents (record_id),
    forward_key TEXT NOT NULL CHECK {_hex("forward_key")},
    UNIQUE (incarnation_id, forward_key),""",
    ),
    _record_table(
        "policy_verdicts",
        f"""    intent_id TEXT NOT NULL UNIQUE REFERENCES action_intents (record_id),
    {_DIRECTION}
    verdict TEXT NOT NULL CHECK (verdict IN ('ALLOW', 'DENY')),""",
    ),
    _record_table(
        "approval_challenges",
        f"""    intent_id TEXT NOT NULL REFERENCES action_intents (record_id),
    {_DIRECTION}
    attempt INTEGER NOT NULL CHECK (attempt BETWEEN 1 AND {MAX_CHALLENGES_PER_INTENT}),
    nonce TEXT NOT NULL UNIQUE CHECK {_hex("nonce")},
    operation_key TEXT NOT NULL CHECK {_hex("operation_key")},
    boot_id TEXT NOT NULL CHECK {_uuid("boot_id")},
    expires_utc TEXT NOT NULL CHECK (length(expires_utc) = 20),
    UNIQUE (intent_id, attempt),""",
    ),
    _record_table(
        "approval_expirations",
        f"""    intent_id TEXT NOT NULL REFERENCES action_intents (record_id),
    {_DIRECTION}
    reason TEXT NOT NULL CHECK (reason IN ('EXPIRED', 'BOOT_CHANGED')),""",
        ",\n    FOREIGN KEY (record_id) REFERENCES approval_challenges (record_id)",
    ),
    """CREATE TABLE approval_nonce_consumptions (
    nonce TEXT PRIMARY KEY NOT NULL REFERENCES approval_challenges (nonce),
    incarnation_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    challenge_digest TEXT NOT NULL UNIQUE REFERENCES approval_challenges (record_id),
    claim_id TEXT NOT NULL UNIQUE CHECK (length(claim_id) = 70),
    FOREIGN KEY (incarnation_id, transaction_id)
        REFERENCES history_operations (incarnation_id, transaction_id)
) STRICT""",
    _record_table(
        "execution_claims",
        f"""    {_DIRECTION}
    operation_key TEXT NOT NULL UNIQUE CHECK {_hex("operation_key")},
    nonce TEXT NOT NULL UNIQUE REFERENCES approval_nonce_consumptions (nonce),
    challenge_digest TEXT NOT NULL UNIQUE CHECK {_hex("challenge_digest")},""",
    ),
    f"""CREATE TABLE verified_outcomes (
    verified_outcome_id TEXT NOT NULL CHECK (length(verified_outcome_id) BETWEEN 1 AND 192),
    outcome_revision INTEGER NOT NULL CHECK (outcome_revision > 0),
    incarnation_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    record_bytes BLOB NOT NULL CHECK (length(record_bytes) BETWEEN 2 AND {_RECORD_BYTES}),
    record_digest TEXT NOT NULL CHECK {_hex("record_digest")},
    {_DIRECTION}
    operation_key TEXT NOT NULL CHECK {_hex("operation_key")},
    outcome_state TEXT NOT NULL CHECK (outcome_state IN (
        'ACTIVE_VERIFIED_AUTHORIZED', 'VERIFIED_ABSENT', 'UNAUTHORIZED_EFFECT',
        'REVERTED', 'INVERSE_VERIFIED_ABSENT')),
    PRIMARY KEY (verified_outcome_id, outcome_revision),
    UNIQUE (operation_key, outcome_revision),
    FOREIGN KEY (incarnation_id, transaction_id)
        REFERENCES history_operations (incarnation_id, transaction_id)
) STRICT""",
    _record_table(
        "outcome_assessments",
        """    verified_outcome_id TEXT NOT NULL,
    outcome_revision INTEGER NOT NULL,
    forecast_id TEXT NOT NULL REFERENCES action_forecasts (record_id),
    UNIQUE (verified_outcome_id, outcome_revision),""",
        ",\n    FOREIGN KEY (verified_outcome_id, outcome_revision)\n"
        "        REFERENCES verified_outcomes (verified_outcome_id, outcome_revision)",
    ),
    _record_table(
        "decision_feedback",
        """    assessment_id TEXT NOT NULL UNIQUE REFERENCES outcome_assessments (record_id),""",
    ),
    """CREATE UNIQUE INDEX history_operation_keys_live
    ON history_operation_keys (reserved_key) WHERE live = 1""",
    # Reservations: inserted with their PREPARED record operation; released
    # (live 1 -> 0) only when that operation aborts; never deleted.
    f"""CREATE TRIGGER history_operation_keys_insert BEFORE INSERT ON history_operation_keys
WHEN NEW.live IS NOT 1 OR NOT EXISTS (
    SELECT 1 FROM history_operations o
    WHERE o.incarnation_id = NEW.incarnation_id AND o.transaction_id = NEW.transaction_id
        AND o.phase = 'PREPARED' AND o.operation_kind IN ({_RECORD_KINDS})
)
BEGIN SELECT RAISE(ABORT, 'a key reservation needs its PREPARED operation'); END""",
    """CREATE TRIGGER history_operation_keys_release BEFORE UPDATE ON history_operation_keys
WHEN NOT (
    OLD.live = 1 AND NEW.live = 0
    AND NEW.reserved_key IS OLD.reserved_key
    AND NEW.incarnation_id IS OLD.incarnation_id
    AND NEW.transaction_id IS OLD.transaction_id
    AND EXISTS (
        SELECT 1 FROM history_operations o
        WHERE o.incarnation_id = OLD.incarnation_id AND o.transaction_id = OLD.transaction_id
            AND o.phase = 'ABORTED'
    )
)
BEGIN SELECT RAISE(ABORT, 'a key reservation is released only by its abort'); END""",
    """CREATE TRIGGER history_operation_keys_no_delete BEFORE DELETE ON history_operation_keys
BEGIN SELECT RAISE(ABORT, 'key reservations are immutable'); END""",
    """CREATE TRIGGER history_operations_release_keys
AFTER UPDATE OF phase ON history_operations WHEN NEW.phase = 'ABORTED'
BEGIN
    UPDATE history_operation_keys SET live = 0
    WHERE incarnation_id = NEW.incarnation_id AND transaction_id = NEW.transaction_id
        AND live = 1;
END""",
    # Record inserts: only inside the FINALIZE of their exact PREPARED
    # operation kind, and only after the previous stage exists (§2b).
    # IP-16 §9: Intents repeat across incarnations, so a forecast, verdict or
    # challenge belongs to its Intent's incarnation, and an assessment only
    # to the Outcome of its own Intent's Claim.  Claims, nonces, Outcomes and
    # the inverse gate stay global.
    f"""CREATE TRIGGER action_intents_insert BEFORE INSERT ON action_intents
WHEN {_gate(_K.R18_INTENT_ISSUE)}
    OR (NEW.direction = 'INVERSE' AND NOT (
        EXISTS (SELECT 1 FROM action_intents f
            WHERE f.obligation_group_key = NEW.obligation_group_key
                AND f.direction = 'FORWARD' AND f.operation_key = NEW.forward_operation_key)
        AND EXISTS (SELECT 1 FROM verified_outcomes v
            WHERE v.operation_key = NEW.forward_operation_key
                AND v.outcome_state IN ('ACTIVE_VERIFIED_AUTHORIZED', 'UNAUTHORIZED_EFFECT'))
    ))
BEGIN SELECT RAISE(ABORT, 'action intent is not admissible here'); END""",
    f"""CREATE TRIGGER action_forecasts_insert BEFORE INSERT ON action_forecasts
WHEN {_gate(_K.R17_EXPECTATION_COMMIT)}
    OR NOT EXISTS (SELECT 1 FROM action_intents i
        WHERE i.record_id = NEW.action_intent_id AND i.direction = 'FORWARD'
            AND i.operation_key = NEW.forward_key
            AND i.incarnation_id = NEW.incarnation_id)
BEGIN SELECT RAISE(ABORT, 'forecast is not admissible here'); END""",
    f"""CREATE TRIGGER policy_verdicts_insert BEFORE INSERT ON policy_verdicts
WHEN {_gate(_K.R18_POLICY_COMMIT)}
    OR NOT EXISTS (SELECT 1 FROM action_intents i
        WHERE i.record_id = NEW.intent_id AND i.direction = NEW.direction
            AND i.incarnation_id = NEW.incarnation_id)
    OR (NEW.direction = 'FORWARD' AND NOT EXISTS (
        SELECT 1 FROM action_forecasts f WHERE f.action_intent_id = NEW.intent_id))
BEGIN SELECT RAISE(ABORT, 'policy verdict is not admissible here'); END""",
    f"""CREATE TRIGGER approval_challenges_insert BEFORE INSERT ON approval_challenges
WHEN {_gate(_K.R18_CHALLENGE_ISSUE)}
    OR NOT EXISTS (SELECT 1 FROM policy_verdicts p
        WHERE p.intent_id = NEW.intent_id AND p.verdict = 'ALLOW')
    OR NOT EXISTS (SELECT 1 FROM action_intents i
        WHERE i.record_id = NEW.intent_id AND i.direction = NEW.direction
            AND i.operation_key = NEW.operation_key
            AND i.incarnation_id = NEW.incarnation_id)
    OR (SELECT count(*) FROM approval_challenges c WHERE c.intent_id = NEW.intent_id)
        IS NOT NEW.attempt - 1
    OR (SELECT count(*) FROM approval_challenges c JOIN approval_expirations e
            ON e.record_id = c.record_id WHERE c.intent_id = NEW.intent_id)
        IS NOT NEW.attempt - 1
BEGIN SELECT RAISE(ABORT, 'approval challenge is not admissible here'); END""",
    f"""CREATE TRIGGER approval_expirations_insert BEFORE INSERT ON approval_expirations
WHEN {_gate(_K.R18_APPROVAL_EXPIRE)}
    OR NOT EXISTS (SELECT 1 FROM approval_challenges c
        WHERE c.record_id = NEW.record_id AND c.intent_id = NEW.intent_id
            AND c.direction = NEW.direction)
    OR EXISTS (SELECT 1 FROM approval_nonce_consumptions n
        WHERE n.challenge_digest = NEW.record_id)
BEGIN SELECT RAISE(ABORT, 'approval expiration is not admissible here'); END""",
    f"""CREATE TRIGGER approval_nonce_consumptions_insert
BEFORE INSERT ON approval_nonce_consumptions
WHEN {_gate(_K.R18_EXECUTION_CLAIM, "CAST(NEW.claim_id AS BLOB)")}
    OR NOT EXISTS (SELECT 1 FROM approval_challenges c
        WHERE c.record_id = NEW.challenge_digest AND c.nonce = NEW.nonce)
    OR EXISTS (SELECT 1 FROM approval_expirations e WHERE e.record_id = NEW.challenge_digest)
BEGIN SELECT RAISE(ABORT, 'nonce consumption is not admissible here'); END""",
    f"""CREATE TRIGGER execution_claims_insert BEFORE INSERT ON execution_claims
WHEN {_gate(_K.R18_EXECUTION_CLAIM)}
    OR NOT EXISTS (SELECT 1 FROM approval_nonce_consumptions n
        JOIN approval_challenges c ON c.record_id = n.challenge_digest
        WHERE n.nonce = NEW.nonce AND n.claim_id = NEW.record_id
            AND n.challenge_digest = NEW.challenge_digest
            AND c.operation_key = NEW.operation_key AND c.direction = NEW.direction)
BEGIN SELECT RAISE(ABORT, 'execution claim is not admissible here'); END""",
    f"""CREATE TRIGGER verified_outcomes_insert BEFORE INSERT ON verified_outcomes
WHEN {_gate(_K.R18_OUTCOME_COMMIT)}
    OR NOT EXISTS (SELECT 1 FROM execution_claims c
        WHERE c.operation_key = NEW.operation_key AND c.direction = NEW.direction)
BEGIN SELECT RAISE(ABORT, 'verified outcome is not admissible here'); END""",
    f"""CREATE TRIGGER outcome_assessments_insert BEFORE INSERT ON outcome_assessments
WHEN {_gate(_K.R17_ASSESSMENT_COMMIT)}
    OR NOT EXISTS (SELECT 1 FROM verified_outcomes v
        WHERE v.verified_outcome_id = NEW.verified_outcome_id
            AND v.outcome_revision = NEW.outcome_revision AND v.direction = 'FORWARD')
    OR NOT EXISTS (SELECT 1 FROM action_forecasts f JOIN verified_outcomes v
        ON v.operation_key = f.forward_key
        JOIN execution_claims c ON c.operation_key = v.operation_key AND c.direction = 'FORWARD'
        JOIN approval_challenges ch ON ch.record_id = c.challenge_digest
        WHERE f.record_id = NEW.forecast_id AND ch.intent_id = f.action_intent_id
            AND v.verified_outcome_id = NEW.verified_outcome_id
            AND v.outcome_revision = NEW.outcome_revision)
BEGIN SELECT RAISE(ABORT, 'outcome assessment is not admissible here'); END""",
    f"""CREATE TRIGGER decision_feedback_insert BEFORE INSERT ON decision_feedback
WHEN {_gate(_K.R17_FEEDBACK_COMMIT)}
    OR NOT EXISTS (SELECT 1 FROM outcome_assessments a WHERE a.record_id = NEW.assessment_id)
BEGIN SELECT RAISE(ABORT, 'decision feedback is not admissible here'); END""",
    *(statement for table in RECORD_TABLES for statement in _immutable(table)),
)

# Order matters: it is also the schema hash input.  Every statement is
# verified against sqlite_schema on open, so a tampered schema fails closed.
SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    f"""CREATE TABLE history_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = {HISTORY_SCHEMA_VERSION}),
    schema_hash TEXT NOT NULL CHECK {_hex("schema_hash")},
    subject_id TEXT NOT NULL CHECK (length(subject_id) BETWEEN 1 AND 128),
    store_id TEXT NOT NULL CHECK {_uuid("store_id")},
    journal_lineage_id TEXT NOT NULL
        CHECK (length(journal_lineage_id) BETWEEN 1 AND 128),
    phase TEXT NOT NULL CHECK (phase IN ('PROVISIONAL', 'ACTIVE')),
    adopted_baseline_id TEXT,
    adopted_baseline_hash TEXT,
    genesis_incarnation_id TEXT,
    CHECK (
        (phase = 'PROVISIONAL' AND adopted_baseline_id IS NULL
            AND adopted_baseline_hash IS NULL AND genesis_incarnation_id IS NULL)
        OR (phase = 'ACTIVE'
            AND length(adopted_baseline_id) BETWEEN 1 AND 128
            AND {_hex("adopted_baseline_hash")}
            AND {_hex("genesis_incarnation_id")})
    )
) STRICT""",
    f"""CREATE TABLE history_operations (
    incarnation_id TEXT NOT NULL CHECK {_hex("incarnation_id")},
    transaction_id TEXT NOT NULL CHECK {_uuid("transaction_id")},
    event_id TEXT NOT NULL CHECK {_uuid("event_id")},
    processing_sequence INTEGER NOT NULL CHECK (processing_sequence > 0),
    transaction_kind TEXT NOT NULL CHECK (length(transaction_kind) BETWEEN 1 AND 64),
    participant_id TEXT NOT NULL
        CHECK (participant_id = '{SUBJECT_HISTORY_PARTICIPANT_ID}'),
    operation_kind TEXT NOT NULL CHECK (operation_kind IN ({_ALL_KINDS})),
    operation_digest TEXT NOT NULL CHECK {_hex("operation_digest")},
    phase TEXT NOT NULL CHECK (phase IN ('PREPARED', 'FINALIZED', 'ABORTED')),
    pending_bytes BLOB,
    pending_digest TEXT,
    lineage_scope TEXT,
    review_intent_key TEXT,
    predecessor_key TEXT,
    PRIMARY KEY (incarnation_id, transaction_id),
    UNIQUE (incarnation_id, event_id, processing_sequence),
    CHECK (
        (operation_kind = 'R16_DECISION_ISSUE'
            AND typeof(pending_bytes) = 'blob'
            AND length(pending_bytes) BETWEEN 1 AND {MAX_PENDING_BYTES}
            AND {_hex("pending_digest")}
            AND length(lineage_scope) BETWEEN 1 AND 128
            AND {_hex("review_intent_key")}
            AND (predecessor_key = 'GENESIS' OR {_decision("predecessor_key")}))
        OR (operation_kind = 'R07_BINDING_ONLY_ABORT' AND phase = 'ABORTED'
            AND pending_bytes IS NULL AND pending_digest IS NULL
            AND lineage_scope IS NULL AND review_intent_key IS NULL
            AND predecessor_key IS NULL)
        OR (operation_kind IN ({_RECORD_KINDS})
            AND typeof(pending_bytes) = 'blob'
            AND length(pending_bytes) BETWEEN 1 AND {MAX_PENDING_BYTES}
            AND {_hex("pending_digest")}
            AND lineage_scope IS NULL AND review_intent_key IS NULL
            AND predecessor_key IS NULL)
    )
) STRICT""",
    """CREATE UNIQUE INDEX history_operations_review_intent
    ON history_operations (incarnation_id, review_intent_key)
    WHERE operation_kind = 'R16_DECISION_ISSUE' AND phase IN ('PREPARED', 'FINALIZED')""",
    """CREATE UNIQUE INDEX history_operations_predecessor
    ON history_operations (incarnation_id, lineage_scope, predecessor_key)
    WHERE operation_kind = 'R16_DECISION_ISSUE' AND phase IN ('PREPARED', 'FINALIZED')""",
    f"""CREATE TABLE history_decisions (
    decision_id TEXT PRIMARY KEY NOT NULL CHECK {_decision("decision_id")},
    incarnation_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    lineage_scope TEXT NOT NULL,
    review_intent_key TEXT NOT NULL,
    predecessor_key TEXT NOT NULL,
    predecessor_digest TEXT,
    issuer_event_id TEXT NOT NULL,
    issuer_sequence INTEGER NOT NULL CHECK (issuer_sequence > 0),
    decision_bytes BLOB NOT NULL
        CHECK (length(decision_bytes) BETWEEN 1 AND {MAX_JSON_BYTES}),
    decision_digest TEXT NOT NULL CHECK {_hex("decision_digest")},
    committed_bytes BLOB NOT NULL
        CHECK (length(committed_bytes) BETWEEN 1 AND {MAX_PENDING_BYTES}),
    committed_digest TEXT NOT NULL CHECK {_hex("committed_digest")},
    policy_kind TEXT NOT NULL CHECK (
        policy_kind IN ('SUCCESSOR_POLICY_V1', 'SUCCESSOR_POLICY_V2', 'SUCCESSOR_POLICY_V3')
    ),
    policy_row TEXT NOT NULL CHECK (
        policy_row IN ('3', '4', '5', '6')
        OR (policy_kind IN ('SUCCESSOR_POLICY_V2', 'SUCCESSOR_POLICY_V3')
            AND policy_row IN ('4b', '5b'))
    ),
    policy_input_bytes BLOB NOT NULL
        CHECK (length(policy_input_bytes) BETWEEN 2 AND {MAX_POLICY_INPUT_BYTES}),
    policy_input_digest TEXT NOT NULL CHECK {_hex("policy_input_digest")},
    CHECK ((predecessor_key = 'GENESIS') = (predecessor_digest IS NULL)),
    CHECK (predecessor_digest IS NULL OR {_hex("predecessor_digest")}),
    UNIQUE (incarnation_id, transaction_id),
    UNIQUE (incarnation_id, review_intent_key),
    UNIQUE (incarnation_id, lineage_scope, predecessor_key),
    FOREIGN KEY (incarnation_id, transaction_id)
        REFERENCES history_operations (incarnation_id, transaction_id)
) STRICT""",
    f"""CREATE TABLE history_source_refs (
    decision_id TEXT NOT NULL REFERENCES history_decisions (decision_id),
    source_role TEXT NOT NULL CHECK (source_role IN ({_ROLES})),
    source_kind TEXT NOT NULL CHECK (length(source_kind) BETWEEN 1 AND 128),
    ref_id TEXT NOT NULL CHECK (length(ref_id) BETWEEN 1 AND 128),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    digest TEXT NOT NULL CHECK {_hex("digest")},
    PRIMARY KEY (decision_id, source_role, ref_id)
) STRICT""",
    f"""CREATE TABLE history_postcommit_receipts (
    incarnation_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    processing_sequence INTEGER NOT NULL CHECK (processing_sequence > 0),
    prepared_state_hash_before TEXT NOT NULL
        CHECK {_hex("prepared_state_hash_before")},
    prepared_state_hash_after TEXT NOT NULL
        CHECK {_hex("prepared_state_hash_after")},
    snapshot_sequence INTEGER NOT NULL CHECK (snapshot_sequence >= 0),
    snapshot_hash TEXT NOT NULL CHECK {_hex("snapshot_hash")},
    wal_generation_id TEXT NOT NULL CHECK {_uuid("wal_generation_id")},
    wal_record_id TEXT NOT NULL CHECK {_uuid("wal_record_id")},
    wal_record_hash TEXT NOT NULL CHECK {_hex("wal_record_hash")},
    receipt_digest TEXT NOT NULL CHECK {_hex("receipt_digest")},
    PRIMARY KEY (incarnation_id, transaction_id),
    FOREIGN KEY (incarnation_id, transaction_id)
        REFERENCES history_operations (incarnation_id, transaction_id)
) STRICT""",
    # history_meta: identity is immutable, the only transition is
    # PROVISIONAL -> ACTIVE, and the row can never be deleted.
    """CREATE TRIGGER history_meta_no_delete BEFORE DELETE ON history_meta
BEGIN SELECT RAISE(ABORT, 'history_meta is immutable'); END""",
    """CREATE TRIGGER history_meta_activation_only BEFORE UPDATE ON history_meta
WHEN NOT (
    OLD.phase = 'PROVISIONAL' AND NEW.phase = 'ACTIVE'
    AND NEW.id IS OLD.id AND NEW.schema_version IS OLD.schema_version
    AND NEW.schema_hash IS OLD.schema_hash AND NEW.subject_id IS OLD.subject_id
    AND NEW.store_id IS OLD.store_id
    AND NEW.journal_lineage_id IS OLD.journal_lineage_id
)
BEGIN SELECT RAISE(ABORT, 'history_meta transition is not permitted'); END""",
    # history_operations: insert only while ACTIVE, only as PREPARED (or as a
    # binding-only ABORTED tombstone); later only PREPARED -> FINALIZED/ABORTED
    # with every bound column unchanged; never deleted.
    f"""CREATE TRIGGER history_operations_insert BEFORE INSERT ON history_operations
WHEN (SELECT phase FROM history_meta WHERE id = 1) IS NOT 'ACTIVE'
    OR NOT (
        (NEW.operation_kind = 'R16_DECISION_ISSUE' AND NEW.phase = 'PREPARED')
        OR (NEW.operation_kind = 'R07_BINDING_ONLY_ABORT' AND NEW.phase = 'ABORTED')
        OR (NEW.operation_kind IN ({_RECORD_KINDS}) AND NEW.phase = 'PREPARED')
    )
BEGIN SELECT RAISE(ABORT, 'history operation insert is not permitted'); END""",
    """CREATE TRIGGER history_operations_transition BEFORE UPDATE ON history_operations
WHEN NOT (
    OLD.phase = 'PREPARED' AND NEW.phase IN ('FINALIZED', 'ABORTED')
    AND NEW.incarnation_id IS OLD.incarnation_id
    AND NEW.transaction_id IS OLD.transaction_id
    AND NEW.event_id IS OLD.event_id
    AND NEW.processing_sequence IS OLD.processing_sequence
    AND NEW.transaction_kind IS OLD.transaction_kind
    AND NEW.participant_id IS OLD.participant_id
    AND NEW.operation_kind IS OLD.operation_kind
    AND NEW.operation_digest IS OLD.operation_digest
    AND NEW.pending_bytes IS OLD.pending_bytes
    AND NEW.pending_digest IS OLD.pending_digest
    AND NEW.lineage_scope IS OLD.lineage_scope
    AND NEW.review_intent_key IS OLD.review_intent_key
    AND NEW.predecessor_key IS OLD.predecessor_key
)
BEGIN SELECT RAISE(ABORT, 'history operation transition is not permitted'); END""",
    """CREATE TRIGGER history_operations_finalize_complete
BEFORE UPDATE OF phase ON history_operations
WHEN NEW.phase = 'FINALIZED' AND (
    NOT EXISTS (
        SELECT 1 FROM history_postcommit_receipts r
        WHERE r.incarnation_id = OLD.incarnation_id
            AND r.transaction_id = OLD.transaction_id
    )
    OR (OLD.operation_kind = 'R16_DECISION_ISSUE' AND NOT EXISTS (
        SELECT 1 FROM history_decisions d
        WHERE d.incarnation_id = OLD.incarnation_id
            AND d.transaction_id = OLD.transaction_id
    ))
"""
    + "".join(
        f"    OR (OLD.operation_kind = '{kind.value}' AND NOT ({_kind_exists(kind)}"
        " AND EXISTS (SELECT 1 FROM history_operation_keys k"
        " WHERE k.incarnation_id = OLD.incarnation_id"
        " AND k.transaction_id = OLD.transaction_id AND k.live = 1)))\n"
        for kind in HistoryOperationKind
        if kind in RECORD_OPERATION_KINDS
    )
    + """)
BEGIN SELECT RAISE(ABORT, 'history FINALIZE is incomplete'); END""",
    """CREATE TRIGGER history_operations_abort_clean
BEFORE UPDATE OF phase ON history_operations
WHEN NEW.phase = 'ABORTED' AND (
    EXISTS (
        SELECT 1 FROM history_postcommit_receipts r
        WHERE r.incarnation_id = OLD.incarnation_id
            AND r.transaction_id = OLD.transaction_id
    )
    OR EXISTS (
        SELECT 1 FROM history_decisions d
        WHERE d.incarnation_id = OLD.incarnation_id
            AND d.transaction_id = OLD.transaction_id
    )
"""
    + "".join(
        f"    OR EXISTS (SELECT 1 FROM {table} t WHERE t.incarnation_id = OLD.incarnation_id"
        " AND t.transaction_id = OLD.transaction_id)\n"
        for table in RECORD_TABLES
    )
    + """)
BEGIN SELECT RAISE(ABORT, 'an operation with finalize rows cannot abort'); END""",
    """CREATE TRIGGER history_operations_no_delete BEFORE DELETE ON history_operations
BEGIN SELECT RAISE(ABORT, 'history operations are immutable'); END""",
    # Decision, source and receipt rows are written only inside the FINALIZE
    # of their exact PREPARED operation, then never changed.
    """CREATE TRIGGER history_decisions_insert BEFORE INSERT ON history_decisions
WHEN NOT EXISTS (
    SELECT 1 FROM history_operations o
    WHERE o.incarnation_id = NEW.incarnation_id
        AND o.transaction_id = NEW.transaction_id
        AND o.phase = 'PREPARED'
        AND o.operation_kind = 'R16_DECISION_ISSUE'
        AND o.event_id = NEW.issuer_event_id
        AND o.processing_sequence = NEW.issuer_sequence
        AND o.lineage_scope = NEW.lineage_scope
        AND o.review_intent_key = NEW.review_intent_key
        AND o.predecessor_key = NEW.predecessor_key
        AND o.pending_bytes = NEW.committed_bytes
)
BEGIN SELECT RAISE(ABORT, 'history decision has no matching PREPARED operation'); END""",
    """CREATE TRIGGER history_decisions_no_update BEFORE UPDATE ON history_decisions
BEGIN SELECT RAISE(ABORT, 'history decisions are immutable'); END""",
    """CREATE TRIGGER history_decisions_no_delete BEFORE DELETE ON history_decisions
BEGIN SELECT RAISE(ABORT, 'history decisions are immutable'); END""",
    """CREATE TRIGGER history_source_refs_insert BEFORE INSERT ON history_source_refs
WHEN NOT EXISTS (
    SELECT 1 FROM history_decisions d
    JOIN history_operations o
        ON o.incarnation_id = d.incarnation_id AND o.transaction_id = d.transaction_id
    WHERE d.decision_id = NEW.decision_id AND o.phase = 'PREPARED'
)
BEGIN SELECT RAISE(ABORT, 'history source has no PREPARED decision'); END""",
    """CREATE TRIGGER history_source_refs_no_update BEFORE UPDATE ON history_source_refs
BEGIN SELECT RAISE(ABORT, 'history sources are immutable'); END""",
    """CREATE TRIGGER history_source_refs_no_delete BEFORE DELETE ON history_source_refs
BEGIN SELECT RAISE(ABORT, 'history sources are immutable'); END""",
    """CREATE TRIGGER history_receipts_insert BEFORE INSERT ON history_postcommit_receipts
WHEN NOT EXISTS (
    SELECT 1 FROM history_operations o
    WHERE o.incarnation_id = NEW.incarnation_id
        AND o.transaction_id = NEW.transaction_id
        AND o.phase = 'PREPARED'
        AND o.event_id = NEW.event_id
        AND o.processing_sequence = NEW.processing_sequence
)
BEGIN SELECT RAISE(ABORT, 'history receipt has no PREPARED operation'); END""",
    """CREATE TRIGGER history_receipts_no_update BEFORE UPDATE ON history_postcommit_receipts
BEGIN SELECT RAISE(ABORT, 'history receipts are immutable'); END""",
    """CREATE TRIGGER history_receipts_no_delete BEFORE DELETE ON history_postcommit_receipts
BEGIN SELECT RAISE(ABORT, 'history receipts are immutable'); END""",
    *V2_STATEMENTS,
)

SCHEMA_HASH: Final = sha256(
    "\0".join(SCHEMA_STATEMENTS).encode("utf-8")
).hexdigest()


class HistoryStoreError(Exception):
    """A bounded SubjectHistory storage failure."""


class HistoryStoreUnavailable(HistoryStoreError):
    """The store cannot be used now (I/O, lock, capacity); retry may succeed."""


class HistoryStoreCapacityError(HistoryStoreUnavailable):
    """Admission would endanger the FINALIZE headroom; refuse before PREPARE."""


class HistoryStoreCorrupt(HistoryStoreError):
    """Custody, schema or stored bytes are not the trusted SubjectHistory."""


class HistoryStoreConflict(HistoryStoreError):
    """A uniqueness or transition rule refused the write."""


class HistoryStoreMigrationRequired(HistoryStoreError):
    """The file is a valid v1 or v2 store: run the explicit migration to v3.

    Not corruption and not retryable by waiting; startup stops (§11-1)."""


class HistoryStoreRecordsDisabled(HistoryStoreError):
    """An R17 / R18 record write while ``r18.enabled`` is false (§7)."""


class HistoryPhase(str, Enum):
    PREPARED = "PREPARED"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


class HistoryMetaPhase(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True, slots=True)
class HistoryMeta:
    schema_version: int
    schema_hash: str
    subject_id: str
    store_id: str
    journal_lineage_id: str
    phase: HistoryMetaPhase
    adopted_baseline_id: str | None
    adopted_baseline_hash: str | None
    genesis_incarnation_id: str | None


@dataclass(frozen=True, slots=True)
class HistoryOperationRow:
    incarnation_id: str
    transaction_id: str
    event_id: str
    processing_sequence: int
    transaction_kind: str
    operation_kind: HistoryOperationKind
    operation_digest: str
    phase: HistoryPhase
    pending_bytes: bytes | None
    pending_digest: str | None
    lineage_scope: str | None
    review_intent_key: str | None
    predecessor_key: str | None


@dataclass(frozen=True, slots=True)
class HistoryDecisionRow:
    decision_id: str
    incarnation_id: str
    transaction_id: str
    committed_bytes: bytes
    committed_digest: str
    receipt: PostcommitReceiptV1
    receipt_digest: str


def _error(error: sqlite3.Error, action: str) -> HistoryStoreError:
    code = getattr(error, "sqlite_errorcode", None)
    primary = code & 0xFF if isinstance(code, int) else None
    if isinstance(error, sqlite3.IntegrityError):
        return HistoryStoreConflict(f"SubjectHistory {action} was refused")
    if primary == sqlite3.SQLITE_FULL:
        return HistoryStoreCapacityError(f"SubjectHistory {action} has no capacity")
    if primary in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}:
        return HistoryStoreCorrupt(f"SubjectHistory {action} found a corrupt file")
    return HistoryStoreUnavailable(f"SubjectHistory {action} is unavailable")


def _check_private_directory(directory: Path) -> None:
    try:
        info = os.lstat(directory)
    except OSError as error:
        raise HistoryStoreUnavailable("SubjectHistory directory is unavailable") from error
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
    ):
        raise HistoryStoreCorrupt("SubjectHistory directory custody is invalid")


def _check_private_file(path: Path, *, required: bool) -> os.stat_result | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        if required:
            raise HistoryStoreUnavailable("SubjectHistory file is absent") from None
        return None
    except OSError as error:
        raise HistoryStoreUnavailable("SubjectHistory file is unavailable") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
    ):
        raise HistoryStoreCorrupt("SubjectHistory file custody is invalid")
    return info


def _trusted_path(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise HistoryStoreCorrupt("SubjectHistory path must be absolute")
    # Refuse symlinked ancestors and non-normal spellings ("..", "//").
    if os.path.realpath(path.parent) != str(path.parent) or path.name in {"", ".", ".."}:
        raise HistoryStoreCorrupt("SubjectHistory path is not a trusted real path")
    return path


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + "-journal")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("SubjectHistory integer column is invalid")
    return value


def _blob(value: object) -> bytes:
    if type(value) is not bytes:
        raise TypeError("SubjectHistory blob column is invalid")
    return value


def _finish_interrupted_create(path: Path) -> None:
    """Complete a ``create`` that crashed between ``link`` and ``unlink``.

    The published file then has two links: itself and exactly one private
    ``.<name>.<hex>.creating`` staging name in the same directory with the
    same inode.  Only that exact shape is repaired (the staging name is
    removed); anything else is left for the custody check to refuse.
    """

    try:
        info = os.lstat(path)
    except OSError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 2:
        return
    # Exactly the staging names create() writes: ".<name>.<uuid4 hex>.creating".
    pattern = re.compile(rf"\.{re.escape(path.name)}\.[0-9a-f]{{32}}\.creating")
    try:
        names = [name for name in os.listdir(path.parent) if pattern.fullmatch(name)]
    except OSError:
        return
    matches = []
    for name in names:
        try:
            other = os.lstat(path.parent / name)
        except OSError:
            continue
        if (other.st_dev, other.st_ino) == (info.st_dev, info.st_ino):
            matches.append(name)
    if len(matches) != 1:
        return
    # Assumes the private (owner-only, checked) directory: no other uid can
    # rename a file into the matched name between lstat and unlink.
    try:
        os.unlink(path.parent / matches[0])
        _fsync_directory(path.parent)
    except OSError:
        return


def _row_operation(row: tuple[object, ...]) -> HistoryOperationRow:
    try:
        return HistoryOperationRow(
            incarnation_id=str(row[0]),
            transaction_id=str(row[1]),
            event_id=str(row[2]),
            processing_sequence=_int(row[3]),
            transaction_kind=str(row[4]),
            operation_kind=HistoryOperationKind(row[5]),
            operation_digest=str(row[6]),
            phase=HistoryPhase(row[7]),
            pending_bytes=None if row[8] is None else _blob(row[8]),
            pending_digest=None if row[9] is None else str(row[9]),
            lineage_scope=None if row[10] is None else str(row[10]),
            review_intent_key=None if row[11] is None else str(row[11]),
            predecessor_key=None if row[12] is None else str(row[12]),
        )
    except (TypeError, ValueError) as error:
        raise HistoryStoreCorrupt("SubjectHistory operation row is invalid") from error


_OPERATION_COLUMNS: Final = (
    "incarnation_id, transaction_id, event_id, processing_sequence, "
    "transaction_kind, operation_kind, operation_digest, phase, pending_bytes, "
    "pending_digest, lineage_scope, review_intent_key, predecessor_key"
)


_SCHEMA_QUERY: Final = (
    "SELECT type, name, tbl_name, sql FROM sqlite_schema "
    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
)


def _expected_schema(statements: tuple[str, ...]) -> list[tuple[object, ...]]:
    reference = sqlite3.connect(":memory:")
    try:
        for statement in statements:
            reference.execute(statement)
        return reference.execute(_SCHEMA_QUERY).fetchall()
    finally:
        reference.close()


def _finish(
    connection: sqlite3.Connection,
    incarnation_id: str,
    transaction_id: str,
    receipt: PostcommitReceiptV1,
) -> None:
    """The receipt and the PREPARED -> FINALIZED step every FINALIZE ends with."""

    connection.execute(
        "INSERT INTO history_postcommit_receipts (incarnation_id, "
        "transaction_id, event_id, processing_sequence, "
        "prepared_state_hash_before, prepared_state_hash_after, "
        "snapshot_sequence, snapshot_hash, wal_generation_id, wal_record_id, "
        "wal_record_hash, receipt_digest) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            incarnation_id,
            transaction_id,
            receipt.event_id,
            receipt.processing_sequence,
            receipt.prepared_state_hash_before,
            receipt.prepared_state_hash_after,
            receipt.snapshot_sequence,
            receipt.snapshot_hash,
            receipt.wal_generation_id,
            receipt.wal_record_id,
            receipt.wal_record_hash,
            receipt.receipt_digest,
        ),
    )
    cursor = connection.execute(
        "UPDATE history_operations SET phase = 'FINALIZED' "
        "WHERE incarnation_id = ? AND transaction_id = ? AND phase = 'PREPARED'",
        (incarnation_id, transaction_id),
    )
    if cursor.rowcount != 1:
        raise HistoryStoreConflict("SubjectHistory operation is not PREPARED")


class SubjectHistoryStore:
    """One trusted connection to the SubjectHistoryDB.

    ``create`` is the only way to make a file (an explicit, isolated first
    adoption step); ``open`` never creates or migrates.  Every connection sets
    and reads back its own PRAGMAs (#376 D1-6) because SQLite does not persist
    them across independent connections.
    """

    def __init__(
        self,
        path: Path,
        connection: sqlite3.Connection,
        max_page_count: int,
        *,
        r18_enabled: bool = False,
    ) -> None:
        self.path = path
        self._connection = connection
        self._max_page_count = max_page_count
        self._lock = Lock()
        self._r18_enabled = r18_enabled

    # ------------------------------------------------------------------ open

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        subject_id: str,
        journal_lineage_id: str,
        max_page_count: int,
        r18_enabled: bool = False,
    ) -> SubjectHistoryStore:
        """Create a PROVISIONAL store; refuses an existing path."""

        path = _trusted_path(path)
        identifier(subject_id, "subject_id")
        identifier(journal_lineage_id, "journal_lineage_id")
        _check_private_directory(path.parent)
        if os.path.lexists(path) or os.path.lexists(_sidecar(path)):
            raise HistoryStoreConflict("SubjectHistory file already exists")
        staging = path.with_name(f".{path.name}.{uuid4().hex}.creating")
        try:
            descriptor = os.open(
                staging, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
            )
            os.close(descriptor)
            connection = cls._connect(staging, max_page_count)
            try:
                connection.execute("BEGIN IMMEDIATE")
                for statement in SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO history_meta (id, schema_version, schema_hash, "
                    "subject_id, store_id, journal_lineage_id, phase) "
                    "VALUES (1, ?, ?, ?, ?, ?, 'PROVISIONAL')",
                    (
                        HISTORY_SCHEMA_VERSION,
                        SCHEMA_HASH,
                        subject_id,
                        str(uuid4()),
                        journal_lineage_id,
                    ),
                )
                connection.execute("COMMIT")
            finally:
                connection.close()
            # link() refuses to replace a concurrently created file.
            os.link(staging, path)
        except sqlite3.Error as error:
            raise _error(error, "creation") from error
        except FileExistsError as error:
            raise HistoryStoreConflict("SubjectHistory file already exists") from error
        except OSError as error:
            raise HistoryStoreUnavailable("SubjectHistory creation failed") from error
        finally:
            try:
                os.unlink(staging)
            except FileNotFoundError:
                pass
        _fsync_directory(path.parent)
        return cls.open(path, max_page_count=max_page_count, r18_enabled=r18_enabled)

    @classmethod
    def open(
        cls, path: Path, *, max_page_count: int, r18_enabled: bool = False
    ) -> SubjectHistoryStore:
        """Open an existing store read-write; never creates or migrates."""

        path = _trusted_path(path)
        _check_private_directory(path.parent)
        _finish_interrupted_create(path)
        before = _check_private_file(path, required=True)
        _check_private_file(_sidecar(path), required=False)
        # The store runs in rollback-journal mode only: WAL sidecars are foreign.
        for suffix in ("-wal", "-shm"):
            if os.path.lexists(path.with_name(path.name + suffix)):
                raise HistoryStoreCorrupt("SubjectHistory has an unexpected WAL sidecar")
        try:
            connection = cls._connect(path, max_page_count)
        except sqlite3.Error as error:
            raise _error(error, "open") from error
        try:
            after = _check_private_file(path, required=True)
            assert before is not None and after is not None
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise HistoryStoreCorrupt("SubjectHistory file was replaced")
            cls._verify_schema(connection)
            store = cls(path, connection, max_page_count, r18_enabled=r18_enabled)
            store.meta()
            return store
        except sqlite3.Error as error:
            connection.close()
            raise _error(error, "open") from error
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _connect(path: Path, max_page_count: int) -> sqlite3.Connection:
        if type(max_page_count) is not int or not 16 <= max_page_count <= 2**31 - 2:
            raise HistoryStoreCorrupt("SubjectHistory max_page_count is invalid")
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=rw",
            uri=True,
            timeout=_SQLITE_BUSY_TIMEOUT_SECONDS,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            # Connection.setconfig exists from Python 3.12; on 3.11 the
            # trusted_schema PRAGMA below is still enforced and read back.
            if sys.version_info >= (3, 12):
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
            expected = (
                ("journal_mode = DELETE", "journal_mode", "delete"),
                ("synchronous = EXTRA", "synchronous", 3),
                ("foreign_keys = ON", "foreign_keys", 1),
                ("trusted_schema = OFF", "trusted_schema", 0),
                ("cell_size_check = ON", "cell_size_check", 1),
                (f"max_page_count = {max_page_count}", "max_page_count", max_page_count),
            )
            page_count = connection.execute("PRAGMA page_count").fetchone()
            if page_count is not None and page_count[0] > max_page_count:
                # A grown file under a smaller cap is a capacity/configuration
                # condition, not corruption (review P2-2).
                raise HistoryStoreCapacityError(
                    "SubjectHistory is larger than its configured max_page_count"
                )
            for assignment, name, value in expected:
                connection.execute(f"PRAGMA {assignment}").fetchall()
                actual = connection.execute(f"PRAGMA {name}").fetchone()
                if actual is None or actual[0] != value:
                    raise HistoryStoreCorrupt(
                        f"SubjectHistory PRAGMA {name} could not be enforced"
                    )
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        actual = connection.execute(_SCHEMA_QUERY).fetchall()
        if actual != _expected_schema(SCHEMA_STATEMENTS):
            for version, statements in ((1, SCHEMA_STATEMENTS_V1), (2, SCHEMA_STATEMENTS_V2)):
                if actual == _expected_schema(statements):
                    raise HistoryStoreMigrationRequired(
                        f"SubjectHistory is schema v{version}: run suzka.r16.history_migration"
                    )
            raise HistoryStoreCorrupt("SubjectHistory schema is not schema v3")
        check = connection.execute("PRAGMA quick_check").fetchall()
        if check != [("ok",)]:
            raise HistoryStoreCorrupt("SubjectHistory integrity check failed")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _write(self, action: str) -> Iterator[sqlite3.Connection]:
        """One short ``BEGIN IMMEDIATE`` transaction (#376 D1-5)."""

        with self._lock:
            connection = self._connection
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as error:
                raise _error(error, action) from error
            try:
                yield connection
                connection.execute("COMMIT")
            except sqlite3.Error as error:
                self._rollback()
                raise _error(error, action) from error
            except BaseException:
                self._rollback()
                raise

    def _rollback(self) -> None:
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _read(self, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
        with self._lock:
            try:
                return self._connection.execute(sql, parameters).fetchall()
            except sqlite3.Error as error:
                raise _error(error, "read") from error

    # ------------------------------------------------------------------ meta

    def meta(self) -> HistoryMeta:
        rows = self._read(
            "SELECT schema_version, schema_hash, subject_id, store_id, "
            "journal_lineage_id, phase, adopted_baseline_id, adopted_baseline_hash, "
            "genesis_incarnation_id FROM history_meta WHERE id = 1"
        )
        if len(rows) != 1:
            raise HistoryStoreCorrupt("SubjectHistory meta is absent")
        row = rows[0]
        try:
            meta = HistoryMeta(
                schema_version=_int(row[0]),
                schema_hash=str(row[1]),
                subject_id=str(row[2]),
                store_id=str(row[3]),
                journal_lineage_id=str(row[4]),
                phase=HistoryMetaPhase(row[5]),
                adopted_baseline_id=None if row[6] is None else str(row[6]),
                adopted_baseline_hash=None if row[7] is None else str(row[7]),
                genesis_incarnation_id=None if row[8] is None else str(row[8]),
            )
        except (TypeError, ValueError) as error:
            raise HistoryStoreCorrupt("SubjectHistory meta is invalid") from error
        if (meta.schema_version, meta.schema_hash) in ((1, SCHEMA_HASH_V1), (2, SCHEMA_HASH_V2)):
            raise HistoryStoreMigrationRequired(
                f"SubjectHistory meta is schema v{meta.schema_version}"
            )
        if meta.schema_version != HISTORY_SCHEMA_VERSION or meta.schema_hash != SCHEMA_HASH:
            raise HistoryStoreCorrupt("SubjectHistory meta schema is not v3")
        return meta

    def activate(
        self,
        *,
        adopted_baseline_id: str,
        adopted_baseline_hash: str,
        genesis_incarnation_id: str,
    ) -> HistoryMeta:
        """PROVISIONAL -> ACTIVE once; an identical repeat is idempotent.

        The caller (U4-3) must first prove the exact clean R07 adopted
        baseline that registers ``subject.history``.
        """

        identifier(adopted_baseline_id, "adopted_baseline_id")
        digest(adopted_baseline_hash, "adopted_baseline_hash")
        digest(genesis_incarnation_id, "genesis_incarnation_id")
        current = self.meta()
        if current.phase is HistoryMetaPhase.ACTIVE:
            if (
                current.adopted_baseline_id,
                current.adopted_baseline_hash,
                current.genesis_incarnation_id,
            ) != (adopted_baseline_id, adopted_baseline_hash, genesis_incarnation_id):
                raise HistoryStoreConflict("SubjectHistory is active for another baseline")
            return current
        with self._write("activation") as connection:
            connection.execute(
                "UPDATE history_meta SET phase = 'ACTIVE', adopted_baseline_id = ?, "
                "adopted_baseline_hash = ?, genesis_incarnation_id = ? WHERE id = 1",
                (adopted_baseline_id, adopted_baseline_hash, genesis_incarnation_id),
            )
        return self.meta()

    def _require_active(self) -> None:
        if self.meta().phase is not HistoryMetaPhase.ACTIVE:
            raise HistoryStoreUnavailable("SubjectHistory is not active")

    # -------------------------------------------------------------- capacity

    def free_bytes(self) -> int:
        rows = self._read("PRAGMA page_count")
        page_count = _int(rows[0][0])
        freelist = _int(self._read("PRAGMA freelist_count")[0][0])
        page_size = _int(self._read("PRAGMA page_size")[0][0])
        return max(self._max_page_count - page_count + freelist, 0) * page_size

    @staticmethod
    def required_headroom(pending_length: int) -> int:
        """Bytes PREPARE + FINALIZE of one operation may need (heuristic bound)."""

        return 4 * pending_length + _FINALIZE_SLACK_BYTES

    def check_headroom(self, pending_length: int) -> None:
        """Refuse admission if the FINALIZE could not be completed (#376 D2-4)."""

        if self.free_bytes() < self.required_headroom(pending_length):
            raise HistoryStoreCapacityError("SubjectHistory FINALIZE headroom is unavailable")

    # ------------------------------------------------------------ operations

    def operation(self, incarnation_id: str, transaction_id: str) -> HistoryOperationRow | None:
        rows = self._read(
            f"SELECT {_OPERATION_COLUMNS} FROM history_operations "
            "WHERE incarnation_id = ? AND transaction_id = ?",
            (incarnation_id, transaction_id),
        )
        return _row_operation(rows[0]) if rows else None

    def has_operations(self) -> bool:
        """Whether any operation row exists in any incarnation."""

        return bool(self._read("SELECT 1 FROM history_operations LIMIT 1"))

    def operations(self, incarnation_id: str) -> tuple[HistoryOperationRow, ...]:
        rows = self._read(
            f"SELECT {_OPERATION_COLUMNS} FROM history_operations "
            "WHERE incarnation_id = ? ORDER BY processing_sequence, transaction_id",
            (incarnation_id,),
        )
        return tuple(_row_operation(row) for row in rows)

    def insert_prepared(
        self,
        *,
        transaction_id: str,
        event_id: str,
        processing_sequence: int,
        transaction_kind: str,
        operation_digest: str,
        decision: CommittedDecisionV1,
    ) -> None:
        pending = decision.canonical_bytes()
        if history_operation_digest(decision) != operation_digest:
            raise HistoryStoreConflict("SubjectHistory operation digest conflicts")
        self._require_active()
        self.check_headroom(len(pending))
        with self._write("prepare") as connection:
            connection.execute(
                f"INSERT INTO history_operations ({_OPERATION_COLUMNS}, participant_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'PREPARED', ?, ?, ?, ?, ?, ?)",
                (
                    decision.incarnation_id,
                    canonical_uuid(transaction_id, "transaction_id"),
                    canonical_uuid(event_id, "event_id"),
                    processing_sequence,
                    transaction_kind,
                    HistoryOperationKind.R16_DECISION_ISSUE.value,
                    operation_digest,
                    pending,
                    sha256(pending).hexdigest(),
                    decision.lineage_scope,
                    decision.review_intent_key,
                    decision.predecessor_key,
                    SUBJECT_HISTORY_PARTICIPANT_ID,
                ),
            )

    def insert_abort_tombstone(
        self,
        *,
        incarnation_id: str,
        transaction_id: str,
        event_id: str,
        processing_sequence: int,
        transaction_kind: str,
        operation_digest: str,
    ) -> None:
        """Binding-only ABORTED row, only after PRE_INTERNAL (#376 D1-1)."""

        self._require_active()
        with self._write("abort") as connection:
            connection.execute(
                f"INSERT INTO history_operations ({_OPERATION_COLUMNS}, participant_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'ABORTED', NULL, NULL, NULL, NULL, NULL, ?)",
                (
                    digest(incarnation_id, "incarnation_id"),
                    canonical_uuid(transaction_id, "transaction_id"),
                    canonical_uuid(event_id, "event_id"),
                    processing_sequence,
                    transaction_kind,
                    HistoryOperationKind.R07_BINDING_ONLY_ABORT.value,
                    digest(operation_digest, "operation_digest"),
                    SUBJECT_HISTORY_PARTICIPANT_ID,
                ),
            )

    def mark_aborted(self, incarnation_id: str, transaction_id: str) -> None:
        with self._write("abort") as connection:
            cursor = connection.execute(
                "UPDATE history_operations SET phase = 'ABORTED' "
                "WHERE incarnation_id = ? AND transaction_id = ? AND phase = 'PREPARED'",
                (incarnation_id, transaction_id),
            )
            if cursor.rowcount != 1:
                raise HistoryStoreConflict("SubjectHistory operation is not PREPARED")

    def finalize(
        self,
        *,
        transaction_id: str,
        decision: CommittedDecisionV1,
        receipt: PostcommitReceiptV1,
    ) -> None:
        """One ``BEGIN IMMEDIATE`` FINALIZE: Decision, sources, receipt, phase."""

        committed = decision.canonical_bytes()
        decision_bytes = decision.decision_bytes
        with self._write("finalize") as connection:
            connection.execute(
                "INSERT INTO history_decisions (decision_id, incarnation_id, "
                "transaction_id, lineage_scope, review_intent_key, predecessor_key, "
                "predecessor_digest, issuer_event_id, issuer_sequence, decision_bytes, "
                "decision_digest, committed_bytes, committed_digest, policy_kind, "
                "policy_row, policy_input_bytes, policy_input_digest) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.incarnation_id,
                    transaction_id,
                    decision.lineage_scope,
                    decision.review_intent_key,
                    decision.predecessor_key,
                    decision.predecessor_digest,
                    decision.issuer_event_id,
                    decision.issuer_sequence,
                    decision_bytes,
                    sha256(decision_bytes).hexdigest(),
                    committed,
                    decision.content_digest,
                    decision.policy_kind.value,
                    decision.policy_row.value,
                    decision.policy_input_bytes,
                    decision.policy_input_digest,
                ),
            )
            connection.executemany(
                "INSERT INTO history_source_refs (decision_id, source_role, "
                "source_kind, ref_id, revision, digest) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        decision.decision_id,
                        item.role.value,
                        item.kind,
                        item.ref_id,
                        item.revision,
                        item.digest,
                    )
                    for item in decision.source_refs
                ],
            )
            _finish(connection, decision.incarnation_id, transaction_id, receipt)

    # -------------------------------------------------- v2 record operations

    def has_record_operations(self) -> bool:
        """Whether any R17 / R18 operation row exists (§7 startup check)."""

        kinds = tuple(kind.value for kind in RECORD_OPERATION_KINDS)
        marks = ", ".join("?" for _ in kinds)
        return bool(
            self._read(f"SELECT 1 FROM history_operations WHERE operation_kind IN ({marks}) LIMIT 1", kinds)
        )

    def live_keys(self, keys: tuple[str, ...]) -> frozenset[str]:
        """The given keys that are currently reserved (writers' pre-check, §2a)."""

        if not keys:
            return frozenset()
        marks = ", ".join("?" for _ in keys)
        rows = self._read(
            f"SELECT reserved_key FROM history_operation_keys WHERE live = 1 AND reserved_key IN ({marks})",
            keys,
        )
        return frozenset(str(row[0]) for row in rows)

    def has_live_claim_keys(self) -> bool:
        """Whether any ``claim:`` key is reserved, in any incarnation (IP-16
        D1 r3 §5 / §10-1: the ledger reader's "no R18 data" check)."""

        return bool(
            self._read(
                "SELECT 1 FROM history_operation_keys "
                # A range on the key ("claim:" <= key < "claim;", ';' follows
                # ':') so the live-key index can serve it.
                "WHERE live = 1 AND reserved_key >= 'claim:' AND reserved_key < 'claim;' "
                "LIMIT 1"
            )
        )

    def live_key_owner(self, key: str) -> tuple[str, str] | None:
        """The (incarnation, transaction) holding one live key, if any (§5)."""

        rows = self._read(
            "SELECT incarnation_id, transaction_id FROM history_operation_keys "
            "WHERE live = 1 AND reserved_key = ?",
            (key,),
        )
        if not rows:
            return None
        return str(rows[0][0]), str(rows[0][1])

    def _require_records_enabled(self) -> None:
        if not self._r18_enabled:
            raise HistoryStoreRecordsDisabled("R17 / R18 records are disabled (r18.enabled is false)")

    def insert_prepared_record(
        self,
        *,
        transaction_id: str,
        operation_digest: str,
        operation: HistoryRecordOperationV1,
    ) -> None:
        """PREPARE one record operation and reserve its keys (§2a), atomically.

        The envelope carries the incarnation, event, sequence and transaction
        kind; the caller (IP-14a-2 participant) checks them against R07.
        """

        self._require_records_enabled()
        if type(operation) is not HistoryRecordOperationV1:
            raise TypeError("operation must be a HistoryRecordOperationV1")
        pending = operation.canonical_bytes()
        if operation.operation_digest != operation_digest:
            raise HistoryStoreConflict("SubjectHistory operation digest conflicts")
        self._require_active()
        self.check_headroom(len(pending))
        with self._write("prepare") as connection:
            connection.execute(
                f"INSERT INTO history_operations ({_OPERATION_COLUMNS}, participant_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'PREPARED', ?, ?, NULL, NULL, NULL, ?)",
                (
                    operation.incarnation_id,
                    canonical_uuid(transaction_id, "transaction_id"),
                    operation.event_id,
                    operation.processing_sequence,
                    operation.transaction_kind,
                    operation.operation_kind.value,
                    operation_digest,
                    pending,
                    sha256(pending).hexdigest(),
                    SUBJECT_HISTORY_PARTICIPANT_ID,
                ),
            )
            connection.executemany(
                "INSERT INTO history_operation_keys (reserved_key, incarnation_id, "
                "transaction_id, live) VALUES (?, ?, ?, 1)",
                [(key, operation.incarnation_id, transaction_id) for key in operation.reserved_keys],
            )

    def finalize_record(
        self,
        *,
        transaction_id: str,
        operation: HistoryRecordOperationV1,
        receipt: PostcommitReceiptV1,
    ) -> None:
        """One ``BEGIN IMMEDIATE`` FINALIZE: the record rows, receipt and phase."""

        if type(operation) is not HistoryRecordOperationV1:
            raise TypeError("operation must be a HistoryRecordOperationV1")
        with self._write("finalize") as connection:
            stored = connection.execute(
                "SELECT pending_bytes, operation_kind FROM history_operations "
                "WHERE incarnation_id = ? AND transaction_id = ? AND phase = 'PREPARED'",
                (operation.incarnation_id, transaction_id),
            ).fetchone()
            if stored is None or (_blob(stored[0]), str(stored[1])) != (
                operation.canonical_bytes(),
                operation.operation_kind.value,
            ):
                raise HistoryStoreConflict("SubjectHistory operation is not this PREPARED record")
            for row in operation.rows:
                names = (*(name for name, _ in row.columns), "incarnation_id", "transaction_id")
                values = (*(value for _, value in row.columns), operation.incarnation_id, transaction_id)
                connection.execute(
                    f"INSERT INTO {row.table} ({', '.join(names)}) "
                    f"VALUES ({', '.join('?' for _ in names)})",
                    values,
                )
            _finish(connection, operation.incarnation_id, transaction_id, receipt)

    def finalized_receipt(
        self, incarnation_id: str, transaction_id: str
    ) -> tuple[PostcommitReceiptV1, str] | None:
        """The postcommit receipt of one operation and its stored digest."""

        rows = self._read(
            "SELECT transaction_id, event_id, processing_sequence, "
            "prepared_state_hash_before, prepared_state_hash_after, snapshot_sequence, "
            "snapshot_hash, wal_generation_id, wal_record_id, wal_record_hash, receipt_digest "
            "FROM history_postcommit_receipts WHERE incarnation_id = ? AND transaction_id = ?",
            (incarnation_id, transaction_id),
        )
        if not rows:
            return None
        row = rows[0]
        try:
            receipt = PostcommitReceiptV1(
                transaction_id=str(row[0]),
                event_id=str(row[1]),
                processing_sequence=_int(row[2]),
                prepared_state_hash_before=str(row[3]),
                prepared_state_hash_after=str(row[4]),
                snapshot_sequence=_int(row[5]),
                snapshot_hash=str(row[6]),
                wal_generation_id=str(row[7]),
                wal_record_id=str(row[8]),
                wal_record_hash=str(row[9]),
            )
        except (TypeError, ValueError) as error:
            raise HistoryStoreCorrupt("SubjectHistory receipt row is invalid") from error
        return receipt, str(row[10])

    def record_rows_match(
        self, incarnation_id: str, transaction_id: str, operation: HistoryRecordOperationV1
    ) -> bool:
        """Whether the stored rows of one FINALIZED record operation are exactly
        the rows its envelope derives, and no record table holds any other row
        of that operation (IP-14a D1 §6). Two queries per table of the kind
        plus one count over the other tables."""

        expected: dict[str, list[dict[str, object]]] = {}
        for row in operation.rows:
            expected.setdefault(row.table, []).append(dict(row.columns))
        for table, wanted in expected.items():
            stored = self.record_rows(table, incarnation_id, transaction_id)
            if len(stored) != len(wanted):
                return False
            for found, columns in zip(stored, wanted, strict=True):
                if any(found.get(name) != value for name, value in columns.items()):
                    return False
        others = [table for table in RECORD_TABLES if table not in expected]
        if others:
            total = " + ".join(
                f"(SELECT count(*) FROM {table} WHERE incarnation_id = ? AND transaction_id = ?)"
                for table in others
            )
            rows = self._read(f"SELECT {total}", tuple(v for _ in others for v in (incarnation_id, transaction_id)))
            if _int(rows[0][0]) != 0:
                return False
        return True

    def record_rows(
        self, table: str, incarnation_id: str, transaction_id: str
    ) -> tuple[dict[str, object], ...]:
        """Raw rows of one record table for one operation (IP-14a-2 reader)."""

        if table not in RECORD_TABLES:
            raise ValueError("not a record table")
        connection_rows = self._read_dicts(
            f"SELECT * FROM {table} WHERE incarnation_id = ? AND transaction_id = ?",
            (incarnation_id, transaction_id),
        )
        return connection_rows

    def _read_dicts(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> tuple[dict[str, object], ...]:
        with self._lock:
            try:
                cursor = self._connection.execute(sql, parameters)
                names = [column[0] for column in cursor.description]
                return tuple(dict(zip(names, row, strict=True)) for row in cursor.fetchall())
            except sqlite3.Error as error:
                raise _error(error, "read") from error

    def finalized_decision(
        self, incarnation_id: str, transaction_id: str
    ) -> HistoryDecisionRow | None:
        rows = self._read(
            "SELECT d.decision_id, d.committed_bytes, d.committed_digest, "
            "r.transaction_id, r.event_id, r.processing_sequence, "
            "r.prepared_state_hash_before, r.prepared_state_hash_after, "
            "r.snapshot_sequence, r.snapshot_hash, r.wal_generation_id, "
            "r.wal_record_id, r.wal_record_hash, r.receipt_digest "
            "FROM history_decisions d JOIN history_postcommit_receipts r "
            "ON r.incarnation_id = d.incarnation_id "
            "AND r.transaction_id = d.transaction_id "
            "WHERE d.incarnation_id = ? AND d.transaction_id = ?",
            (incarnation_id, transaction_id),
        )
        if not rows:
            return None
        row = rows[0]
        try:
            receipt = PostcommitReceiptV1(
                transaction_id=str(row[3]),
                event_id=str(row[4]),
                processing_sequence=_int(row[5]),
                prepared_state_hash_before=str(row[6]),
                prepared_state_hash_after=str(row[7]),
                snapshot_sequence=_int(row[8]),
                snapshot_hash=str(row[9]),
                wal_generation_id=str(row[10]),
                wal_record_id=str(row[11]),
                wal_record_hash=str(row[12]),
            )
            return HistoryDecisionRow(
                decision_id=str(row[0]),
                incarnation_id=incarnation_id,
                transaction_id=transaction_id,
                committed_bytes=_blob(row[1]),
                committed_digest=str(row[2]),
                receipt=receipt,
                receipt_digest=str(row[13]),
            )
        except (TypeError, ValueError) as error:
            raise HistoryStoreCorrupt("SubjectHistory decision row is invalid") from error


__all__ = [
    "HISTORY_SCHEMA_VERSION",
    "SCHEMA_HASH",
    "SCHEMA_STATEMENTS",
    "HistoryDecisionRow",
    "HistoryMeta",
    "HistoryMetaPhase",
    "HistoryOperationKind",
    "HistoryOperationRow",
    "HistoryPhase",
    "HistoryStoreCapacityError",
    "HistoryStoreConflict",
    "HistoryStoreCorrupt",
    "HistoryStoreError",
    "HistoryStoreMigrationRequired",
    "HistoryStoreRecordsDisabled",
    "HistoryStoreUnavailable",
    "V2_STATEMENTS",
    "SubjectHistoryStore",
]
