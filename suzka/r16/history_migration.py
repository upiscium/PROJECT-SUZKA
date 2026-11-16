"""IP-14a-1: the one-way SubjectHistory v1 -> v2 migration (D1 r3 §5).

Human decision §11-1 (2026-11-16): v2 code never migrates by itself. An
operator backs up ``.suzka/`` and runs::

    python -m suzka.r16.history_migration /abs/path/subject_history.sqlite3

The steps follow the recipe verified on SQLite 3.50 with DEFENSIVE mode
(D1 review): verify the exact v1 schema, write a 0600 backup with the online
backup API, then in ONE rollback-journal transaction drop every trigger and
user index, rebuild ``history_meta`` / ``history_operations`` /
``history_decisions`` as ``X -> X_v1, CREATE X (exact v2 text), copy, DROP
X_v1`` with ``legacy_alter_table = ON`` (child foreign keys keep naming
``X``), create the v2 objects, and verify the exact v2 schema,
``quick_check``, ``foreign_key_check`` and every v1 row before COMMIT. A crash
anywhere before COMMIT leaves the v1 file; after COMMIT it is v2. There is no
way back except restoring a backup.

The rebuild is parameterised (``_rebuild``) so IP-14b's v2 -> v3 can reuse it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
import sys
from typing import Final

from suzka.r16.history_schema_v1 import SCHEMA_HASH_V1, SCHEMA_STATEMENTS_V1
from suzka.r16.history_store import (
    _SCHEMA_QUERY,
    HISTORY_SCHEMA_VERSION,
    SCHEMA_HASH,
    SCHEMA_STATEMENTS,
    HistoryStoreConflict,
    HistoryStoreCorrupt,
    HistoryStoreError,
    HistoryStoreUnavailable,
    _check_private_directory,
    _check_private_file,
    _expected_schema,
    _fsync_directory,
    _sidecar,
    _trusted_path,
)

_V1_TABLES: Final = (
    "history_meta",
    "history_operations",
    "history_decisions",
    "history_source_refs",
    "history_postcommit_receipts",
)
# Tables whose CHECK constraints change in v2 and must be rebuilt.
_REBUILT: Final = ("history_meta", "history_operations", "history_decisions")
_META_COLUMNS: Final = (
    "id, schema_version, schema_hash, subject_id, store_id, journal_lineage_id, "
    "phase, adopted_baseline_id, adopted_baseline_hash, genesis_incarnation_id"
)

# Tests inject a failure at a named stage to prove crash safety.
_STAGE_HOOK: Callable[[str], None] | None = None


class HistoryMigrationError(HistoryStoreError):
    """The migration refused or rolled back; the file is unchanged."""


def _stage(name: str) -> None:
    if _STAGE_HOOK is not None:
        _STAGE_HOOK(name)


def _create_statement(table: str) -> str:
    prefix = f"CREATE TABLE {table} ("
    matches = [statement for statement in SCHEMA_STATEMENTS if statement.startswith(prefix)]
    if len(matches) != 1:
        raise HistoryMigrationError(f"v2 has no single CREATE for {table}")
    return matches[0]


def _object_name(statement: str) -> str:
    words = statement.split()
    # CREATE TABLE x / CREATE UNIQUE INDEX x / CREATE TRIGGER x / CREATE INDEX x
    return words[3] if words[1] == "UNIQUE" else words[2]


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True, isolation_level=None, timeout=5.0)
    try:
        if sys.version_info >= (3, 12):
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
        for pragma in ("journal_mode = DELETE", "synchronous = EXTRA", "trusted_schema = OFF"):
            connection.execute(f"PRAGMA {pragma}").fetchall()
        if connection.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
            raise HistoryMigrationError("SubjectHistory journal mode could not be enforced")
        return connection
    except BaseException:
        connection.close()
        raise


def _snapshot(connection: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    return {
        table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in _V1_TABLES
    }


def _backup(connection: sqlite3.Connection, path: Path, stamp: str) -> Path:
    target = path.with_name(f"{path.name}.v1-{stamp}.bak")
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    except FileExistsError as error:
        raise HistoryMigrationError("the migration backup already exists") from error
    os.close(descriptor)
    copy = sqlite3.connect(target)
    try:
        connection.backup(copy)
    finally:
        copy.close()
    descriptor = os.open(target, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return target


def _rebuild(
    connection: sqlite3.Connection,
    tables: tuple[str, ...],
    new_statements: tuple[str, ...],
    copy: Callable[[sqlite3.Connection, str], None],
) -> None:
    """Inside an open transaction: drop all triggers and user indexes, rebuild
    ``tables`` with their exact new CREATE text, then create every object of
    ``new_statements`` that does not exist yet (tables, indexes, triggers)."""

    for kind in ("trigger", "index"):
        for (name,) in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = ? AND sql IS NOT NULL", (kind,)
        ).fetchall():
            connection.execute(f"DROP {kind.upper()} {name}")
    _stage("dropped")
    for table in tables:
        connection.execute(f"ALTER TABLE {table} RENAME TO {table}_v1")
        connection.execute(_create_statement(table))
        copy(connection, table)
        connection.execute(f"DROP TABLE {table}_v1")
        _stage(f"rebuilt:{table}")
    existing = {
        str(row[0]) for row in connection.execute("SELECT name FROM sqlite_schema WHERE sql IS NOT NULL")
    }
    for statement in new_statements:
        if _object_name(statement) not in existing:
            connection.execute(statement)
    _stage("created")


def _copy_v1(connection: sqlite3.Connection, table: str) -> None:
    if table == "history_meta":
        connection.execute(
            f"INSERT INTO history_meta ({_META_COLUMNS}) "
            f"SELECT id, ?, ?, subject_id, store_id, journal_lineage_id, phase, "
            "adopted_baseline_id, adopted_baseline_hash, genesis_incarnation_id FROM history_meta_v1",
            (HISTORY_SCHEMA_VERSION, SCHEMA_HASH),
        )
    else:
        connection.execute(f"INSERT INTO {table} SELECT * FROM {table}_v1")


def _verify_v2(connection: sqlite3.Connection, before: dict[str, list[tuple[object, ...]]]) -> None:
    if connection.execute(_SCHEMA_QUERY).fetchall() != _expected_schema(SCHEMA_STATEMENTS):
        raise HistoryMigrationError("the migrated schema is not exactly v2")
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise HistoryMigrationError("the migrated store fails quick_check")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise HistoryMigrationError("the migrated store fails foreign_key_check")
    after = _snapshot(connection)
    for table in _V1_TABLES:
        if table == "history_meta":
            # Only schema_version (column 1) and schema_hash (column 2) change.
            old = [(row[0], *row[3:]) for row in before[table]]
            new = [(row[0], *row[3:]) for row in after[table]]
            if old != new or [row[1:3] for row in after[table]] != [(HISTORY_SCHEMA_VERSION, SCHEMA_HASH)]:
                raise HistoryMigrationError("history_meta did not migrate exactly")
        elif after[table] != before[table]:
            raise HistoryMigrationError(f"{table} rows changed during the migration")


def migrate_v1_to_v2(path: Path, *, now: datetime | None = None) -> Path:
    """Migrate one v1 store in place; return the backup path.

    Refuses anything that is not exactly a v1 store (a v2 store, a corrupt
    file, a hot journal). Raises ``HistoryMigrationError`` and leaves the
    file unchanged when any check fails.
    """

    path = _trusted_path(path)
    _check_private_directory(path.parent)
    _check_private_file(path, required=True)
    if os.path.lexists(_sidecar(path)):
        raise HistoryMigrationError("SubjectHistory has a hot journal; open it with v1 code first")
    for suffix in ("-wal", "-shm"):
        if os.path.lexists(path.with_name(path.name + suffix)):
            raise HistoryStoreCorrupt("SubjectHistory has an unexpected WAL sidecar")
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    try:
        connection = _connect(path)
    except sqlite3.Error as error:
        raise HistoryStoreUnavailable("SubjectHistory cannot be opened for migration") from error
    try:
        if connection.execute(_SCHEMA_QUERY).fetchall() != _expected_schema(SCHEMA_STATEMENTS_V1):
            raise HistoryMigrationError("SubjectHistory is not exactly schema v1")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise HistoryStoreCorrupt("SubjectHistory v1 fails quick_check")
        meta = connection.execute("SELECT schema_version, schema_hash FROM history_meta").fetchall()
        if meta != [(1, SCHEMA_HASH_V1)]:
            raise HistoryMigrationError("SubjectHistory meta is not v1")
        backup = _backup(connection, path, stamp)
        _stage("backed-up")
        before = _snapshot(connection)
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA legacy_alter_table = ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            _rebuild(connection, _REBUILT, SCHEMA_STATEMENTS, _copy_v1)
            _verify_v2(connection, before)
            _stage("verified")
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.execute("PRAGMA legacy_alter_table = OFF")
            connection.execute("PRAGMA foreign_keys = ON")
        _stage("committed")
        # The rebuild left freed pages; compact so the file fits the same cap.
        connection.execute("VACUUM")
        if connection.execute(_SCHEMA_QUERY).fetchall() != _expected_schema(SCHEMA_STATEMENTS):
            raise HistoryStoreCorrupt("SubjectHistory is not v2 after VACUUM")
        return backup
    except sqlite3.IntegrityError as error:
        raise HistoryStoreConflict("SubjectHistory migration broke a constraint") from error
    except sqlite3.Error as error:
        raise HistoryMigrationError(f"SubjectHistory migration failed: {error}") from error
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m suzka.r16.history_migration /abs/path/subject_history.sqlite3", file=sys.stderr)
        return 2
    try:
        backup = migrate_v1_to_v2(Path(args[0]))
    except HistoryStoreError as error:
        print(f"migration refused: {error}", file=sys.stderr)
        return 1
    print(f"migrated to SubjectHistory v2; v1 backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HistoryMigrationError", "main", "migrate_v1_to_v2"]
