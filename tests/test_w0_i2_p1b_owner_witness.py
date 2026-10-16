"""Disposable data-only tests; no owner or Source import, file, or registration."""

import hashlib
import json
import sys
from dataclasses import replace

import pytest

from suzka.owner_witness_codec import (
    MAX_WITNESS_BYTES,
    CatalogIdentity,
    CollectionIdentity,
    DirectoryIdentity,
    OwnerPresence,
    OwnerWitness,
    WitnessDenied,
    WitnessReason,
    canonicalize_witness_bytes,
    digest_witness_bytes,
    parse_witness_exact_bytes,
)


def witness() -> OwnerWitness:
    directory = DirectoryIdentity
    return OwnerWitness(
        schema_version=1,
        installation_id="00000000-0000-4000-8000-000000000001",
        first_install_attempt="00000000-0000-4000-8000-000000000002",
        source_root=directory("/fixture/source", 1, 10, 100, 100, 0o700),
        source_ancestors=(
            directory("/", 1, 2, 0, 0, 0o755),
            directory("/fixture", 1, 9, 100, 100, 0o755),
        ),
        source_mount=directory("/fixture", 1, 9, 100, 100, 0o755),
        chroma_root=directory("/fixture/source/chroma", 1, 11, 100, 100, 0o700),
        chroma_catalog=CatalogIdentity(
            "/fixture/source/chroma/chroma.sqlite3", 1, 12, 100, 100, 0o600
        ),
        tenant_name="default_tenant",
        database_name="default_database",
        database_id="00000000-0000-4000-8000-000000000003",
        db1=CollectionIdentity("db1", "00000000-0000-4000-8000-000000000004"),
        db2=CollectionIdentity("db2", "00000000-0000-4000-8000-000000000005"),
        journal_lineage_id="00000000-0000-4000-8000-000000000006",
        experience_root=directory("/fixture/source/experience", 1, 13, 100, 100, 0o700),
        semantic_root=directory("/fixture/source/semantic", 1, 14, 100, 100, 0o700),
        adapter_registry_empty_at_genesis=True,
        required_owners=tuple(
            OwnerPresence(owner_class, owner_class.lower())
            for owner_class in (
                "WAL",
                "Journal",
                "AgentState",
                "Experience",
                "Semantic",
                "AdapterRegistry",
            )
        ),
    )


def encoded(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def altered(**changes: object) -> bytes:
    raw = json.loads(canonicalize_witness_bytes(witness()))
    raw.update(changes)
    return encoded(raw)


def test_roundtrip_and_determinism() -> None:
    original = witness()
    data = canonicalize_witness_bytes(original)
    assert parse_witness_exact_bytes(data) == replace(
        original,
        required_owners=tuple(
            sorted(original.required_owners, key=lambda x: x.owner_class)
        ),
    )
    # Input dataclass ordering and JSON mapping insertion order do not change bytes.
    assert (
        canonicalize_witness_bytes(
            replace(original, required_owners=original.required_owners[::-1])
        )
        == data
    )
    assert encoded(dict(reversed(list(json.loads(data).items())))) == data
    assert digest_witness_bytes(data) == hashlib.sha256(data).hexdigest()
    assert len(digest_witness_bytes(data)) == 64


def test_noncanonical_bytes_and_digest_mismatch() -> None:
    data = canonicalize_witness_bytes(witness())
    for other in (data + b"\n", json.dumps(json.loads(data), indent=2).encode()):
        with pytest.raises(WitnessDenied):
            digest_witness_bytes(other)
    changed = data.replace(b'"name":"db1"', b'"name":"db3"', 1)
    assert digest_witness_bytes(changed) != digest_witness_bytes(data)
    with pytest.raises(WitnessDenied) as exc:
        parse_witness_exact_bytes(data + b"\n")
    assert exc.value.reason is WitnessReason.NONCANONICAL


def test_bad_json_and_schema() -> None:
    data = canonicalize_witness_bytes(witness())
    for bad in (
        data.replace(b'"schema_version":1', b'"schema_version":2'),
        data.replace(b'"schema_version":1', b'"schema_version":true'),
        data.replace(b'"schema_version":1', b'"schema_version":-1'),
        data.replace(b'"schema_version":1', b'"schema_version":NaN'),
        data.replace(b'"schema_version":1', b'"schema_version":Infinity'),
        data.replace(b'"db1":', b'"db1":null,"db1":'),
        data.replace(b'"name":"db1"', b'"name":"db1","name":"db1"'),
        altered(unrecognized=True),
        altered(current_wal_generation=1),
        altered(agent_state_content="private"),
        altered(journal_content="private"),
        altered(adapter_registry_json={"adapters": []}),
        altered(chroma_sql_sha256="a" * 64),
        data.replace(b'"db1":', b'"missing_db1":'),
        b"\xff",
        b"{}",
        b"[]",
        b"",
        b"{" + b" " * MAX_WITNESS_BYTES + b"}",
        b"[" * 1000 + b"0" + b"]" * 1000,
    ):
        with pytest.raises(WitnessDenied):
            parse_witness_exact_bytes(bad)


def test_path_numeric_and_identity_refusals() -> None:
    base = json.loads(canonicalize_witness_bytes(witness()))
    changes = (
        ("source_root", "path", "relative"),
        ("source_root", "path", "/fixture/../source"),
        ("source_root", "path", "/fixture//source"),
        ("source_root", "path", "/fixture/source/"),
        ("source_root", "dev", -1),
        ("source_root", "ino", 0),
        ("source_root", "uid", True),
        ("source_root", "gid", 2**63),
        ("source_root", "mode", 0o100700),
        ("source_mount", "ino", 22),
        ("chroma_root", "path", base["source_root"]["path"]),
        ("semantic_root", "path", base["experience_root"]["path"]),
        ("chroma_catalog", "path", "/elsewhere/db"),
        ("db2", "name", base["db1"]["name"]),
        ("db2", "collection_id", base["db1"]["collection_id"]),
        ("db1", "collection_id", "AAAAAAAA-0000-4000-8000-000000000004"),
    )
    for group, field, value in changes:
        raw = json.loads(json.dumps(base))
        raw[group][field] = value
        try:
            parse_witness_exact_bytes(encoded(raw))
        except WitnessDenied:
            pass
        else:
            pytest.fail(f"accepted {group}.{field}")
    for group, value in (
        ("source_ancestors", []),
        ("source_ancestors", [base["source_root"]]),
        ("database_id", base["installation_id"]),
        ("required_owners", base["required_owners"][:-1]),
        ("required_owners", [base["required_owners"][0]] * 6),
        ("adapter_registry_empty_at_genesis", False),
        ("adapter_registry_empty_at_genesis", 1),
    ):
        raw = json.loads(json.dumps(base))
        raw[group] = value
        with pytest.raises(WitnessDenied):
            parse_witness_exact_bytes(encoded(raw))
    with pytest.raises(WitnessDenied):
        canonicalize_witness_bytes(
            replace(
                witness(),
                source_root=replace(witness().source_root, path="/fixture/\ud800"),
            )
        )


@pytest.mark.parametrize(
    "catalog_path",
    [
        "/fixture/source/chroma/subdir/arbitrary.db",
        "/fixture/source/chroma/other.sqlite3",
    ],
)
def test_catalog_is_exact_direct_child(catalog_path: str) -> None:
    base = witness()
    with pytest.raises(WitnessDenied) as exc:
        canonicalize_witness_bytes(
            replace(
                base, chroma_catalog=replace(base.chroma_catalog, path=catalog_path)
            )
        )
    assert exc.value.reason is WitnessReason.INVALID
    raw = json.loads(canonicalize_witness_bytes(base))
    raw["chroma_catalog"]["path"] = catalog_path
    with pytest.raises(WitnessDenied) as exc:
        parse_witness_exact_bytes(encoded(raw))
    assert exc.value.reason is WitnessReason.INVALID


def test_unicode_data_and_surrogates() -> None:
    base = witness()
    updated = replace(
        base,
        source_root=replace(base.source_root, path="/fixture/猫"),
        chroma_root=replace(base.chroma_root, path="/fixture/猫/chroma?#%"),
        chroma_catalog=replace(
            base.chroma_catalog, path="/fixture/猫/chroma?#%/chroma.sqlite3"
        ),
        experience_root=replace(base.experience_root, path="/fixture/猫/experience"),
        semantic_root=replace(base.semantic_root, path="/fixture/猫/semantic"),
    )
    data = canonicalize_witness_bytes(updated)
    assert "猫".encode() in data
    assert parse_witness_exact_bytes(data) == replace(
        updated,
        required_owners=tuple(
            sorted(updated.required_owners, key=lambda owner: owner.owner_class)
        ),
    )
    with pytest.raises(WitnessDenied):
        parse_witness_exact_bytes(data.replace("猫".encode(), b"\\ud800", 1))


def test_static_metadata_present_and_single_component_source() -> None:
    base = witness()
    data = json.loads(canonicalize_witness_bytes(base))
    for name in (
        "source_root",
        "source_mount",
        "chroma_root",
        "chroma_catalog",
        "experience_root",
        "semantic_root",
    ):
        assert set(data[name]) == {"path", "dev", "ino", "uid", "gid", "mode"}
    assert data["journal_lineage_id"] == base.journal_lineage_id
    assert data["adapter_registry_empty_at_genesis"] is True
    assert len(data["required_owners"]) == 6
    shallow = replace(
        base,
        source_root=DirectoryIdentity("/source", 1, 10, 100, 100, 0o700),
        source_ancestors=(base.source_ancestors[0],),
        source_mount=DirectoryIdentity("/source", 1, 10, 100, 100, 0o700),
        chroma_root=replace(base.chroma_root, path="/source/chroma"),
        chroma_catalog=replace(
            base.chroma_catalog, path="/source/chroma/chroma.sqlite3"
        ),
        experience_root=replace(base.experience_root, path="/source/experience"),
        semantic_root=replace(base.semantic_root, path="/source/semantic"),
    )
    assert parse_witness_exact_bytes(canonicalize_witness_bytes(shallow)) == replace(
        shallow,
        required_owners=tuple(
            sorted(shallow.required_owners, key=lambda x: x.owner_class)
        ),
    )


@pytest.mark.parametrize("components", [32, 33, 34])
def test_source_ancestor_array_supports_documented_path_depth(
    components: int,
) -> None:
    """Ancestor count must not silently cap otherwise-valid Source paths at 32."""
    base = witness()
    names = ["a"] * components
    path = "/" + "/".join(names)
    ancestral_paths = ["/"] + [
        "/" + "/".join(names[:index]) for index in range(1, components)
    ]
    ancestors = tuple(
        DirectoryIdentity(name, 1, 100 + index, 100, 100, 0o700)
        for index, name in enumerate(ancestral_paths)
    )
    updated = replace(
        base,
        source_root=replace(base.source_root, path=path),
        source_ancestors=ancestors,
        source_mount=ancestors[-1],
        chroma_root=replace(base.chroma_root, path=path + "/chroma"),
        chroma_catalog=replace(
            base.chroma_catalog, path=path + "/chroma/chroma.sqlite3"
        ),
        experience_root=replace(base.experience_root, path=path + "/experience"),
        semantic_root=replace(base.semantic_root, path=path + "/semantic"),
    )
    canonical = canonicalize_witness_bytes(updated)
    assert len(canonical) <= MAX_WITNESS_BYTES
    decoded = parse_witness_exact_bytes(canonical)
    assert decoded == replace(
        updated,
        required_owners=tuple(
            sorted(updated.required_owners, key=lambda owner: owner.owner_class)
        ),
    )


def test_data_only_no_owner_import_or_io(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    before = set(sys.modules)
    original_open = builtins.open

    def prohibited(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected filesystem IO")

    monkeypatch.setattr(builtins, "open", prohibited)
    try:
        data = canonicalize_witness_bytes(witness())
        parse_witness_exact_bytes(data)
        digest_witness_bytes(data)
    finally:
        monkeypatch.setattr(builtins, "open", original_open)
    assert not any(
        name.startswith(("suzka.source", "suzka.learning", "chromadb"))
        for name in set(sys.modules) - before
    )
