"""IP-14a-1 / IP-16a-1: the one-way SubjectHistory v1 or v2 -> v3 migration (D1 r3 §5).

Human decision §11-1 (2026-11-16): the store never migrates by itself. An
operator backs up ``.suzka/`` and runs::

    python -m suzka.r16.history_migration /abs/path/subject_history.sqlite3

The command detects the source version. v1 (``rebuild/develop`` before
IP-14a-1) goes straight to v3 by rebuilding ``history_meta`` /
``history_operations`` / ``history_decisions``. v2 (IP-14a-1 .. 6ed2361,
deployed on eve) rebuilds ``history_meta`` and the two tables whose UNIQUE
scope IP-16 §9 changed (``action_intents``, ``action_forecasts``). A v2 store
that holds any R17 / R18 record operation or key reservation is refused: its
PREPARE reservations use the v2 (global) key format, and no such store exists
outside tests (R18 was never enabled on v2, and startup refuses records while
it is off). Recreate a lab store instead.

The steps follow the recipe verified on SQLite 3.50 with DEFENSIVE mode
(D1 review): under the write lock, verify the exact source schema, copy the
file to a 0600 backup and check the copy, then in the same rollback-journal
transaction drop every trigger and user index, rebuild each changed table as
``X -> X_old, CREATE X (exact v3 text), copy, DROP X_old`` with
``legacy_alter_table = ON`` (child foreign keys keep naming ``X``), create the
missing v3 objects, and verify the exact v3 schema, ``quick_check``,
``foreign_key_check`` and every source row before COMMIT. A crash anywhere
before COMMIT leaves the source file (a crashed attempt's journal is rolled
back by SQLite and the command can simply be run again); after COMMIT it is
v3, and only the VACUUM is best effort. There is no way back except
restoring a backup.

The rebuild is parameterised (``_rebuild``) so IP-14b's next version can
reuse it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
import sys
from typing import Final

from suzka.r16.history_records import RECORD_OPERATION_KINDS
from suzka.r16.history_schema_v1 import SCHEMA_HASH_V1, SCHEMA_STATEMENTS_V1
from suzka.r16.history_schema_v2 import SCHEMA_HASH_V2, SCHEMA_STATEMENTS_V2
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

_META_COLUMNS: Final = (
    "id, schema_version, schema_hash, subject_id, store_id, journal_lineage_id, "
    "phase, adopted_baseline_id, adopted_baseline_hash, genesis_incarnation_id"
)

# v2 adds about 40 pages to an empty v1 store (66 vs 21); keep a margin.
_MIGRATION_PAGES: Final = 64

# Tests inject a failure at a named stage to prove crash safety.
_STAGE_HOOK: Callable[[str], None] | None = None


def _tables(statements: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        statement.split()[2] for statement in statements if statement.startswith("CREATE TABLE ")
    )


@dataclass(frozen=True, slots=True)
class _Source:
    """One version the command migrates from."""

    version: int
    statements: tuple[str, ...]
    schema_hash: str
    # Tables whose CREATE text changes in v3 and must be rebuilt, in an
    # order where a rebuilt parent comes before its rebuilt child.
    rebuilt: tuple[str, ...]

    @property
    def tables(self) -> tuple[str, ...]:
        return _tables(self.statements)


_SOURCES: Final = (
    _Source(1, SCHEMA_STATEMENTS_V1, SCHEMA_HASH_V1, ("history_meta", "history_operations", "history_decisions")),
    _Source(2, SCHEMA_STATEMENTS_V2, SCHEMA_HASH_V2, ("history_meta", "action_intents", "action_forecasts")),
)


class HistoryMigrationError(HistoryStoreError):
    """The migration refused or rolled back before COMMIT; the file is unchanged."""


def _stage(name: str) -> None:
    if _STAGE_HOOK is not None:
        _STAGE_HOOK(name)


def _create_statement(table: str) -> str:
    prefix = f"CREATE TABLE {table} ("
    matches = [statement for statement in SCHEMA_STATEMENTS if statement.startswith(prefix)]
    if len(matches) != 1:
        raise HistoryMigrationError(f"v{HISTORY_SCHEMA_VERSION} has no single CREATE for {table}")
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


def _snapshot(connection: sqlite3.Connection, tables: tuple[str, ...]) -> dict[str, list[tuple[object, ...]]]:
    return {
        table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        for table in tables
    }


def _detect(connection: sqlite3.Connection) -> _Source:
    actual = connection.execute(_SCHEMA_QUERY).fetchall()
    for source in _SOURCES:
        if actual == _expected_schema(source.statements):
            return source
    if actual == _expected_schema(SCHEMA_STATEMENTS):
        raise HistoryMigrationError(f"SubjectHistory is already schema v{HISTORY_SCHEMA_VERSION}")
    raise HistoryMigrationError("SubjectHistory is not exactly schema v1 or v2")


def _holds_records(connection: sqlite3.Connection) -> bool:
    kinds = tuple(kind.value for kind in RECORD_OPERATION_KINDS)
    marks = ", ".join("?" for _ in kinds)
    return bool(
        connection.execute("SELECT 1 FROM history_operation_keys LIMIT 1").fetchall()
        or connection.execute(
            f"SELECT 1 FROM history_operations WHERE operation_kind IN ({marks}) LIMIT 1", kinds
        ).fetchall()
    )


def _backup(path: Path, stamp: str, source: _Source) -> Path:
    """Copy the committed file under the caller's write lock (BEGIN IMMEDIATE:
    no other writer, no pending write of ours, any hot journal already rolled
    back), then prove the copy is exactly the source version."""

    # A crashed earlier attempt may have left a backup with the same stamp;
    # never overwrite one, pick the next free name.
    for attempt in range(100):
        suffix = "" if attempt == 0 else f".{attempt}"
        target = path.with_name(f"{path.name}.v{source.version}-{stamp}{suffix}.bak")
        try:
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            break
        except FileExistsError:
            continue
    else:
        raise HistoryMigrationError("no free name for the migration backup")
    try:
        with open(path, "rb") as source_file:
            while chunk := source_file.read(1 << 20):
                os.write(descriptor, chunk)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    check = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True)
    try:
        if check.execute(_SCHEMA_QUERY).fetchall() != _expected_schema(source.statements) or (
            check.execute("PRAGMA quick_check").fetchall() != [("ok",)]
        ):
            raise HistoryMigrationError(f"the migration backup is not an exact v{source.version} copy")
    finally:
        check.close()
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
        connection.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
        connection.execute(_create_statement(table))
        copy(connection, table)
        connection.execute(f"DROP TABLE {table}_old")
        _stage(f"rebuilt:{table}")
    existing = {
        str(row[0]) for row in connection.execute("SELECT name FROM sqlite_schema WHERE sql IS NOT NULL")
    }
    for statement in new_statements:
        if _object_name(statement) not in existing:
            connection.execute(statement)
    _stage("created")


def _copy(connection: sqlite3.Connection, table: str) -> None:
    if table == "history_meta":
        connection.execute(
            f"INSERT INTO history_meta ({_META_COLUMNS}) "
            f"SELECT id, ?, ?, subject_id, store_id, journal_lineage_id, phase, "
            "adopted_baseline_id, adopted_baseline_hash, genesis_incarnation_id FROM history_meta_old",
            (HISTORY_SCHEMA_VERSION, SCHEMA_HASH),
        )
    else:
        # Rebuilt tables keep their columns and column order in every version.
        connection.execute(f"INSERT INTO {table} SELECT * FROM {table}_old")


def _verify(
    connection: sqlite3.Connection, source: _Source, before: dict[str, list[tuple[object, ...]]]
) -> None:
    current = f"v{HISTORY_SCHEMA_VERSION}"
    if connection.execute(_SCHEMA_QUERY).fetchall() != _expected_schema(SCHEMA_STATEMENTS):
        raise HistoryMigrationError(f"the migrated schema is not exactly {current}")
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise HistoryMigrationError("the migrated store fails quick_check")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise HistoryMigrationError("the migrated store fails foreign_key_check")
    after = _snapshot(connection, source.tables)
    for table in source.tables:
        if table == "history_meta":
            # Only schema_version (column 1) and schema_hash (column 2) change.
            old = [(row[0], *row[3:]) for row in before[table]]
            new = [(row[0], *row[3:]) for row in after[table]]
            if old != new or [row[1:3] for row in after[table]] != [(HISTORY_SCHEMA_VERSION, SCHEMA_HASH)]:
                raise HistoryMigrationError("history_meta did not migrate exactly")
        elif after[table] != before[table]:
            raise HistoryMigrationError(f"{table} rows changed during the migration")


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """The outcome of a committed migration. ``compacted`` is false when the
    best-effort VACUUM after COMMIT failed: the store is migrated either way."""

    from_version: int
    backup: Path
    compacted: bool
    compaction_error: str | None


def migrate(path: Path, *, now: datetime | None = None, max_page_count: int | None = None) -> MigrationResult:
    """Migrate one v1 or v2 store in place to the current version.

    Refuses anything that is not exactly a v1 or v2 store (a current store,
    a corrupt file). A journal left by a crashed earlier attempt is handled
    by SQLite on the first read (a hot journal rolls back; a cold one is
    ignored), so a crashed migration can simply be run again. The exact
    source checks, the backup and the row snapshot all run under the write
    lock. Any failure before COMMIT raises ``HistoryMigrationError`` and
    leaves the source file; after COMMIT the store is current and only the
    VACUUM is best effort.

    ``max_page_count``, when given, refuses a store too close to its cap to
    take the new objects before anything changes.
    """

    path = _trusted_path(path)
    _check_private_directory(path.parent)
    _check_private_file(path, required=True)
    _check_private_file(_sidecar(path), required=False)
    for suffix in ("-wal", "-shm"):
        if os.path.lexists(path.with_name(path.name + suffix)):
            raise HistoryStoreCorrupt("SubjectHistory has an unexpected WAL sidecar")
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    try:
        connection = _connect(path)
    except sqlite3.Error as error:
        raise HistoryStoreUnavailable("SubjectHistory cannot be opened for migration") from error
    committed = False
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA legacy_alter_table = ON")
        # The write lock first: nothing can change between the checks, the
        # backup, the snapshot and the rebuild. The first read also rolls
        # back a hot journal from a crashed earlier attempt.
        connection.execute("BEGIN IMMEDIATE")
        try:
            source = _detect(connection)
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise HistoryStoreCorrupt(f"SubjectHistory v{source.version} fails quick_check")
            meta = connection.execute("SELECT schema_version, schema_hash, phase FROM history_meta").fetchall()
            # PROVISIONAL (enabled, not yet adopted) and ACTIVE both migrate.
            if len(meta) != 1 or meta[0][:2] != (source.version, source.schema_hash):
                raise HistoryMigrationError(f"SubjectHistory meta is not v{source.version}")
            if source.version == 2 and _holds_records(connection):
                raise HistoryMigrationError(
                    "SubjectHistory v2 holds R17 / R18 records; only a store without them migrates"
                )
            if max_page_count is not None:
                pages = int(connection.execute("PRAGMA page_count").fetchone()[0])
                if pages + _MIGRATION_PAGES > max_page_count:
                    raise HistoryMigrationError(
                        "SubjectHistory is too close to max_page_count to take the new objects"
                    )
            backup = _backup(path, stamp, source)
            _stage("backed-up")
            before = _snapshot(connection, source.tables)
            _rebuild(connection, source.rebuilt, SCHEMA_STATEMENTS, _copy)
            _verify(connection, source, before)
            _stage("verified")
            connection.execute("COMMIT")
            committed = True
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
        # Best effort: the rebuild left freed pages. A failure here is not a
        # failed migration (the file is already current).
        try:
            connection.execute("VACUUM")
        except sqlite3.Error as error:
            return MigrationResult(source.version, backup, False, str(error))
        return MigrationResult(source.version, backup, True, None)
    except sqlite3.IntegrityError as error:
        raise HistoryStoreConflict("SubjectHistory migration broke a constraint") from error
    except sqlite3.Error as error:
        if committed:
            raise
        raise HistoryMigrationError(f"SubjectHistory migration failed: {error}") from error
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    max_page_count: int | None = None
    if len(args) == 3 and args[0] == "--max-page-count" and args[1].isdigit():
        max_page_count = int(args[1])
        args = args[2:]
    if len(args) != 1:
        print(
            "usage: python -m suzka.r16.history_migration [--max-page-count N] "
            "/abs/path/subject_history.sqlite3",
            file=sys.stderr,
        )
        return 2
    try:
        result = migrate(Path(args[0]), max_page_count=max_page_count)
    except HistoryStoreError as error:
        print(f"migration refused, the file is unchanged: {error}", file=sys.stderr)
        return 1
    current = f"v{HISTORY_SCHEMA_VERSION}"
    print(f"migrated SubjectHistory v{result.from_version} to {current}; backup: {result.backup}")
    if not result.compacted:
        print(f"note: the store is {current}, but compaction (VACUUM) failed: {result.compaction_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HistoryMigrationError", "MigrationResult", "main", "migrate"]
