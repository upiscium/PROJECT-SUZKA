"""Disposable Chroma 1.5.9 catalog fixtures; no real Source or service access."""

import sqlite3
import stat
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
from chromadb.config import Settings

from suzka import chroma_catalog_preflight as p


def _identity(path: Path) -> p.Identity:
    s = path.stat()
    return p.Identity(
        str(path), s.st_dev, s.st_ino, s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)
    )


@pytest.fixture
def catalog(tmp_path: Path) -> p.Policy:
    assert chromadb.__version__ == "1.5.9"
    source = tmp_path / "synthetic-source"
    source.mkdir()
    root = source / "日本語 ? # %"
    root.mkdir()
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=str(root), settings=settings)
    one = client.create_collection("synthetic-db1")
    two = client.create_collection("synthetic-db2")
    admin = chromadb.AdminClient(
        Settings(
            is_persistent=True,
            persist_directory=str(root),
            anonymized_telemetry=False,
        )
    )
    admin.create_database("synthetic-alternate")
    admin.create_tenant("synthetic-other-tenant")
    admin.create_database("default_database", tenant="synthetic-other-tenant")
    alt = chromadb.PersistentClient(
        path=str(root),
        settings=settings,
        database="synthetic-alternate",
    )
    alt.create_collection("synthetic-db1")
    alt.create_collection("synthetic-db2")
    other = chromadb.PersistentClient(
        path=str(root),
        settings=settings,
        tenant="synthetic-other-tenant",
    )
    other.create_collection("synthetic-db1")
    other.create_collection("synthetic-db2")
    db = root / "chroma.sqlite3"
    with sqlite3.connect(db) as connection:
        database_id = connection.execute(
            "SELECT id FROM databases WHERE name = 'default_database' AND tenant_id = 'default_tenant'"
        ).fetchone()[0]
    return p.Policy(
        _identity(source),
        _identity(root),
        _identity(db),
        "1.5.9",
        "default_tenant",
        "default_database",
        database_id,
        p.Collection("synthetic-db1", str(one.id)),
        p.Collection("synthetic-db2", str(two.id)),
        p.Limits(32_000_000, 100_000, 2048, 50, 100, 200_000, 5.0, 0.15),
    )


def _refused(policy: p.Policy) -> p.Refusal:
    result = p.observe(policy)
    assert result.observation is p.Observation.REFUSED
    assert result.startup == "DISABLED_PENDING_I2"
    assert result.refusal is not None
    return result.refusal


def test_real_catalog_and_nonauthority(catalog: p.Policy) -> None:
    result = p.observe(catalog)
    assert result == p.Result(p.Observation.VERIFIED_UUIDS)
    assert not any(
        "token" in field or "witness" in field
        for field in p.Result.__dataclass_fields__
    )
    assert "%3F" in p._uri(catalog.catalog.path)
    assert "%23" in p._uri(catalog.catalog.path)
    assert "%25" in p._uri(catalog.catalog.path)


def test_same_names_in_many_unrelated_databases_do_not_spend_row_budget(
    catalog: p.Policy,
) -> None:
    settings = Settings(anonymized_telemetry=False)
    admin = chromadb.AdminClient(
        Settings(
            is_persistent=True,
            persist_directory=catalog.chroma_root.path,
            anonymized_telemetry=False,
        )
    )
    for number in range(6):
        name = f"synthetic-unrelated-{number}"
        admin.create_database(name)
        client = chromadb.PersistentClient(
            path=catalog.chroma_root.path,
            settings=settings,
            database=name,
        )
        client.create_collection(catalog.db1.name)
    bounded = replace(catalog, limits=replace(catalog.limits, max_entries=2))
    assert p.observe(bounded) == p.Result(p.Observation.VERIFIED_UUIDS)
    assert (
        _refused(replace(bounded, database_name="synthetic-alternate"))
        is p.Refusal.CATALOG_MISMATCH
    )
    assert (
        _refused(replace(bounded, tenant_id="synthetic-other-tenant"))
        is p.Refusal.CATALOG_MISMATCH
    )


def test_case_insensitive_sqlite_collation_never_substitutes_name(
    tmp_path: Path,
) -> None:
    """The actual stored name must match, regardless of SQLite collation."""
    source = tmp_path / "case-sensitive-source"
    source.mkdir()
    root = source / "chroma"
    root.mkdir()
    db = root / "chroma.sqlite3"
    db_id = str(uuid4())
    db1_id, db2_id = str(uuid4()), str(uuid4())
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE tenants(id TEXT);"
            "CREATE TABLE databases(id TEXT,name TEXT,tenant_id TEXT);"
            "CREATE TABLE collections(id TEXT,name TEXT COLLATE NOCASE,database_id TEXT);"
        )
        conn.execute("INSERT INTO tenants VALUES (?)", ("default_tenant",))
        conn.execute(
            "INSERT INTO databases VALUES (?,?,?)",
            (db_id, "default_database", "default_tenant"),
        )
        conn.execute("INSERT INTO collections VALUES (?,?,?)", (db1_id, "DB1", db_id))
        conn.execute("INSERT INTO collections VALUES (?,?,?)", (db2_id, "db2", db_id))
    policy = p.Policy(
        _identity(source),
        _identity(root),
        _identity(db),
        "1.5.9",
        "default_tenant",
        "default_database",
        db_id,
        p.Collection("db1", db1_id),
        p.Collection("db2", db2_id),
        p.Limits(4_000_000, 1000, 2048, 50, 10, 100_000, 2.0, 0.15),
    )
    assert _refused(policy) is p.Refusal.CATALOG_MISMATCH
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE collections SET name = ? WHERE id = ?", ("db1", db1_id))
    assert p.observe(policy) == p.Result(p.Observation.VERIFIED_UUIDS)


def test_other_uid_writable_root_or_catalog_is_refused(catalog: p.Policy) -> None:
    """Even matching asserted modes cannot bless another-UID sidecar writes."""
    source = Path(catalog.source.path)
    root = Path(catalog.chroma_root.path)
    db = Path(catalog.catalog.path)
    for path, updated in (
        (source, "source"),
        (root, "chroma_root"),
        (db, "catalog"),
    ):
        before = stat.S_IMODE(path.stat().st_mode)
        try:
            path.chmod(before | 0o022)
            policy = replace(catalog, **{updated: _identity(path)})
            assert _refused(policy) is p.Refusal.PATH_UNSAFE
        finally:
            path.chmod(before)


def test_wrong_namespace_and_recreated_uuid(catalog: p.Policy) -> None:
    assert _refused(replace(catalog, tenant_id="other")) is p.Refusal.CATALOG_MISMATCH
    assert (
        _refused(replace(catalog, database_uuid=str(uuid4())))
        is p.Refusal.CATALOG_MISMATCH
    )
    assert (
        _refused(replace(catalog, database_name="synthetic-alternate"))
        is p.Refusal.CATALOG_MISMATCH
    )
    assert (
        _refused(replace(catalog, tenant_id="synthetic-other-tenant"))
        is p.Refusal.CATALOG_MISMATCH
    )
    assert (
        _refused(replace(catalog, db1=replace(catalog.db1, uuid=str(uuid4()))))
        is p.Refusal.CATALOG_MISMATCH
    )
    client = chromadb.PersistentClient(
        path=catalog.chroma_root.path, settings=Settings(anonymized_telemetry=False)
    )
    client.delete_collection(catalog.db1.name)
    recreated = client.create_collection(catalog.db1.name)
    assert str(recreated.id) != catalog.db1.uuid
    assert _refused(catalog) is p.Refusal.CATALOG_MISMATCH


def test_missing_db2(catalog: p.Policy) -> None:
    with sqlite3.connect(catalog.catalog.path) as connection:
        connection.execute("DELETE FROM collections WHERE id = ?", (catalog.db2.uuid,))
    assert _refused(catalog) is p.Refusal.CATALOG_MISMATCH


def test_absent_readonly_no_file_created(catalog: p.Policy) -> None:
    db = Path(catalog.catalog.path)
    db.unlink()
    assert _refused(catalog) in (p.Refusal.ABSENT, p.Refusal.IDENTITY_MISMATCH)
    assert not db.exists()


def test_root_swap_symlink_and_corruption(catalog: p.Policy) -> None:
    root = Path(catalog.chroma_root.path)
    old = root.with_name("previous-fixture")
    root.rename(old)
    root.mkdir()
    assert _refused(catalog) is p.Refusal.IDENTITY_MISMATCH
    root.rmdir()
    root.symlink_to(old, target_is_directory=True)
    _refused(catalog)


def test_corrupt_and_symlinked_main(catalog: p.Policy) -> None:
    db = Path(catalog.catalog.path)
    old = db.with_name("previous-synthetic-db")
    db.rename(old)
    db.symlink_to(old)
    _refused(catalog)
    db.unlink()
    db.write_bytes(b"not a sqlite database")
    _refused(replace(catalog, catalog=_identity(db)))


def test_wrong_metadata_and_path_and_version(catalog: p.Policy) -> None:
    _refused(
        replace(catalog, catalog=replace(catalog.catalog, uid=catalog.catalog.uid + 1))
    )
    assert _refused(replace(catalog, schema_tag="1.5.8")) is p.Refusal.INVALID_POLICY
    assert (
        _refused(
            replace(
                catalog, catalog=replace(catalog.catalog, path="file:///tmp/x?mode=ro")
            )
        )
        is p.Refusal.PATH_UNSAFE
    )
    assert (
        _refused(replace(catalog, source=replace(catalog.source, path="/tmp/../x")))
        is p.Refusal.PATH_UNSAFE
    )
    assert (
        _refused(replace(catalog, limits=replace(catalog.limits, max_vm_steps=1)))
        is p.Refusal.RESOURCE_EXCEEDED
    )


def test_wal_committed_visible_and_payload_growth(catalog: p.Policy) -> None:
    with sqlite3.connect(catalog.catalog.path) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("CREATE TABLE synthetic_payload (value TEXT)")
        writer.execute("INSERT INTO synthetic_payload VALUES (?)", ("x" * 1000,))
        writer.commit()  # WAL still open; catalog snapshot sees committed frames.
        assert p.observe(catalog).observation is p.Observation.VERIFIED_UUIDS
        assert (
            _refused(
                replace(
                    catalog,
                    limits=replace(
                        catalog.limits,
                        max_file_bytes=Path(catalog.catalog.path).stat().st_size + 1,
                    ),
                )
            )
            is p.Refusal.RESOURCE_EXCEEDED
        )


def test_lock_bounded_and_file_limit(catalog: p.Policy) -> None:
    with sqlite3.connect(catalog.catalog.path, timeout=0.1) as writer:
        writer.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        assert (
            _refused(
                replace(
                    catalog, limits=replace(catalog.limits, busy_timeout_seconds=0.02)
                )
            )
            is p.Refusal.UNSTABLE
        )
        assert time.monotonic() - started < 2
        writer.rollback()
    assert (
        _refused(replace(catalog, limits=replace(catalog.limits, max_file_bytes=1)))
        is p.Refusal.RESOURCE_EXCEEDED
    )
    assert (
        _refused(replace(catalog, limits=replace(catalog.limits, max_pages=1)))
        is p.Refusal.RESOURCE_EXCEEDED
    )


def test_schema_mismatch(catalog: p.Policy) -> None:
    with sqlite3.connect(catalog.catalog.path) as connection:
        connection.execute("ALTER TABLE collections RENAME TO obsolete_collections")
    assert _refused(catalog) is p.Refusal.UNSUPPORTED_SCHEMA


def test_invalid_limits(catalog: p.Policy) -> None:
    for limits in (
        replace(catalog.limits, deadline_seconds=float("inf")),
        replace(catalog.limits, busy_timeout_seconds=0),
        replace(catalog.limits, max_entries=0),
    ):
        assert _refused(replace(catalog, limits=limits)) is p.Refusal.INVALID_POLICY
    assert _refused(replace(catalog, tenant_id="x" * 1025)) is p.Refusal.INVALID_POLICY
    assert _refused(replace(catalog, tenant_id="\ud800")) is p.Refusal.INVALID_POLICY
    assert (
        _refused(replace(catalog, database_name="界" * 342)) is p.Refusal.INVALID_POLICY
    )
    assert (
        _refused(replace(catalog, db1=replace(catalog.db1, name="x" * 1025)))
        is p.Refusal.INVALID_POLICY
    )
    assert (
        _refused(replace(catalog, chroma_root=catalog.source)) is p.Refusal.PATH_UNSAFE
    )
