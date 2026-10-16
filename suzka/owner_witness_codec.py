"""Pure, nonauthorizing, schema-1 static OwnerWitness data codec.

All identities are caller assertions. Neither a digest nor a successful parse
authenticates custody, observes an owner, or grants permission to start one.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, NoReturn
from uuid import UUID

MAX_WITNESS_BYTES = 32768
MAX_JSON_DEPTH = 12
MAX_STRING_BYTES = 4096
MAX_PATH_COMPONENTS = 256
# One identity is required for each Source path ancestor (including /).
# The exact published witness still has a separate 32-KiB size bound.
MAX_ARRAY_ITEMS = MAX_PATH_COMPONENTS
MAX_ID = 2**63 - 1

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_OWNER_CLASSES = frozenset(
    {"AgentState", "Journal", "WAL", "AdapterRegistry", "Experience", "Semantic"}
)


class WitnessReason(str, Enum):
    INVALID = "INVALID"
    RESOURCE_EXCEEDED = "RESOURCE_EXCEEDED"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    NONCANONICAL = "NONCANONICAL"


class WitnessDenied(ValueError):
    """Coarse refusal; never includes private inputs in the exception message."""

    def __init__(self, reason: WitnessReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _deny(reason: WitnessReason = WitnessReason.INVALID) -> NoReturn:
    raise WitnessDenied(reason)


def _text(value: object, *, limit: int = MAX_STRING_BYTES) -> str:
    if type(value) is not str:
        _deny()
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError:
        _deny()
    if not 0 < size <= limit or "\x00" in value:
        _deny()
    return value


def _path(value: object, *, allow_root: bool = False) -> str:
    path = _text(value)
    if path == "/" and allow_root:
        return path
    parts = path.split("/")
    if (
        not path.startswith("/")
        or path == "/"
        or len(parts) - 1 > MAX_PATH_COMPONENTS
        or any(part in {"", ".", ".."} for part in parts[1:])
    ):
        _deny()
    return path


def _uuid(value: object) -> str:
    text = _text(value, limit=36)
    try:
        if str(UUID(text)) != text:
            _deny()
    except ValueError:
        _deny()
    return text


def _number(value: object, *, minimum: int = 0, maximum: int = MAX_ID) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        _deny()


def _name(value: object) -> str:
    name = _text(value, limit=128)
    if _NAME.fullmatch(name) is None or name in {".", ".."}:
        _deny()
    return name


def _child(path: str, parent: str) -> bool:
    return (
        path.startswith(parent + "/")
        if parent != "/"
        else path.startswith("/") and path != "/"
    )


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    path: str
    dev: int
    ino: int
    uid: int
    gid: int
    mode: int

    def __post_init__(self) -> None:
        _path(self.path, allow_root=True)
        _number(self.dev)
        _number(self.ino, minimum=1)
        _number(self.uid)
        _number(self.gid)
        _number(self.mode, maximum=0o7777)


@dataclass(frozen=True, slots=True)
class CatalogIdentity:
    path: str
    dev: int
    ino: int
    uid: int
    gid: int
    mode: int

    def __post_init__(self) -> None:
        _path(self.path)
        _number(self.dev)
        _number(self.ino, minimum=1)
        _number(self.uid)
        _number(self.gid)
        _number(self.mode, maximum=0o7777)


@dataclass(frozen=True, slots=True)
class OwnerPresence:
    owner_class: str
    name: str

    def __post_init__(self) -> None:
        if _text(self.owner_class, limit=32) not in _OWNER_CLASSES:
            _deny()
        _name(self.name)


@dataclass(frozen=True, slots=True)
class CollectionIdentity:
    name: str
    collection_id: str

    def __post_init__(self) -> None:
        _name(self.name)
        _uuid(self.collection_id)


@dataclass(frozen=True, slots=True)
class OwnerWitness:
    schema_version: int
    installation_id: str
    first_install_attempt: str
    source_root: DirectoryIdentity
    # Every lexical ancestor from / (inclusive) to the Source parent, in order.
    source_ancestors: tuple[DirectoryIdentity, ...]
    # Explicit mount boundary: either a registered ancestor or the Source root.
    source_mount: DirectoryIdentity
    chroma_root: DirectoryIdentity
    chroma_catalog: CatalogIdentity
    tenant_name: str
    database_name: str
    database_id: str
    db1: CollectionIdentity
    db2: CollectionIdentity
    journal_lineage_id: str
    experience_root: DirectoryIdentity
    semantic_root: DirectoryIdentity
    adapter_registry_empty_at_genesis: bool
    required_owners: tuple[OwnerPresence, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _deny(WitnessReason.UNSUPPORTED_VERSION)
        ids = (
            self.installation_id,
            self.first_install_attempt,
            self.database_id,
            self.db1.collection_id if type(self.db1) is CollectionIdentity else None,
            self.db2.collection_id if type(self.db2) is CollectionIdentity else None,
            self.journal_lineage_id,
        )
        if any(type(item) is not str for item in ids):
            _deny()
        for item in ids:
            _uuid(item)
        if len(set(ids)) != len(ids):
            _deny()
        for directory in (
            self.source_root,
            self.source_mount,
            self.chroma_root,
            self.experience_root,
            self.semantic_root,
        ):
            if type(directory) is not DirectoryIdentity:
                _deny()
            directory.__post_init__()
        if type(self.chroma_catalog) is not CatalogIdentity:
            _deny()
        self.chroma_catalog.__post_init__()
        for collection in (self.db1, self.db2):
            if type(collection) is not CollectionIdentity:
                _deny()
            collection.__post_init__()
        if self.db1.name == self.db2.name:
            _deny()
        _name(self.tenant_name)
        _name(self.database_name)
        if (
            type(self.adapter_registry_empty_at_genesis) is not bool
            or not self.adapter_registry_empty_at_genesis
        ):
            _deny()
        if (
            type(self.source_ancestors) is not tuple
            or not 1 <= len(self.source_ancestors) <= MAX_ARRAY_ITEMS
        ):
            _deny()
        _path(self.source_root.path)
        for root in (self.chroma_root, self.experience_root, self.semantic_root):
            _path(root.path)
        parts = self.source_root.path.split("/")[1:]
        expected = ("/",) + tuple(
            "/" + "/".join(parts[:n]) for n in range(1, len(parts))
        )
        if len(expected) != len(self.source_ancestors):
            _deny()
        for ancestor, path in zip(self.source_ancestors, expected, strict=True):
            if type(ancestor) is not DirectoryIdentity or ancestor.path != path:
                _deny()
            ancestor.__post_init__()
        if self.source_mount not in (*self.source_ancestors, self.source_root):
            _deny()
        roots = (
            self.source_root,
            self.chroma_root,
            self.experience_root,
            self.semantic_root,
        )
        paths = [item.path for item in roots]
        if (
            len(set(paths)) != len(paths)
            or any(not _child(item.path, self.source_root.path) for item in roots[1:])
            or any(
                _child(a, b) or _child(b, a)
                for i, a in enumerate(paths[1:], 1)
                for b in paths[i + 1 :]
            )
        ):
            _deny()
        if self.chroma_catalog.path != self.chroma_root.path + "/chroma.sqlite3":
            _deny()
        if type(self.required_owners) is not tuple or len(self.required_owners) != len(
            _OWNER_CLASSES
        ):
            _deny()
        for owner in self.required_owners:
            if type(owner) is not OwnerPresence:
                _deny()
            owner.__post_init__()
        if {item.owner_class for item in self.required_owners} != _OWNER_CLASSES or len(
            {item.name for item in self.required_owners}
        ) != len(self.required_owners):
            _deny()


def _shape(cls: type, raw: object) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != {field.name for field in fields(cls)}:
        _deny()
    return raw


def _identity(cls: type, raw: object) -> DirectoryIdentity | CatalogIdentity:
    data = _shape(cls, raw)
    return cls(**data)


def _decode(raw: object) -> OwnerWitness:
    data = _shape(OwnerWitness, raw).copy()
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        _deny(WitnessReason.UNSUPPORTED_VERSION)
    for key in (
        "source_root",
        "source_mount",
        "chroma_root",
        "experience_root",
        "semantic_root",
    ):
        data[key] = _identity(DirectoryIdentity, data[key])
    data["chroma_catalog"] = _identity(CatalogIdentity, data["chroma_catalog"])
    ancestors = data["source_ancestors"]
    owners = data["required_owners"]
    if (
        type(ancestors) is not list
        or type(owners) is not list
        or len(ancestors) > MAX_ARRAY_ITEMS
        or len(owners) > MAX_ARRAY_ITEMS
    ):
        _deny()
    data["source_ancestors"] = tuple(
        _identity(DirectoryIdentity, item) for item in ancestors
    )
    data["required_owners"] = tuple(
        OwnerPresence(**_shape(OwnerPresence, item)) for item in owners
    )
    for key in ("db1", "db2"):
        data[key] = CollectionIdentity(**_shape(CollectionIdentity, data[key]))
    return OwnerWitness(**data)


def _json_value(value: object, depth: int = 1) -> object:
    if depth > MAX_JSON_DEPTH:
        _deny(WitnessReason.RESOURCE_EXCEEDED)
    if isinstance(
        value,
        (
            OwnerWitness,
            DirectoryIdentity,
            CatalogIdentity,
            OwnerPresence,
            CollectionIdentity,
        ),
    ):
        return {
            field.name: _json_value(
                tuple(
                    sorted(value.required_owners, key=lambda owner: owner.owner_class)
                )
                if type(value) is OwnerWitness and field.name == "required_owners"
                else getattr(value, field.name),
                depth + 1,
            )
            for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_ARRAY_ITEMS:
            _deny(WitnessReason.RESOURCE_EXCEEDED)
        return [_json_value(item, depth + 1) for item in value]
    if type(value) is str:
        _text(value)
    return value


def canonicalize_witness_bytes(witness: OwnerWitness) -> bytes:
    """Serialize validated, immutable asserted data; never publish it."""
    if type(witness) is not OwnerWitness:
        _deny()
    witness.__post_init__()
    try:
        result = json.dumps(
            _json_value(witness),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        _deny()
    if len(result) > MAX_WITNESS_BYTES:
        _deny(WitnessReason.RESOURCE_EXCEEDED)
    return result


def parse_witness_exact_bytes(data: bytes) -> OwnerWitness:
    """Reject noncanonical encodings as well as malformed or unsupported data."""
    if type(data) is not bytes:
        _deny()
    if not 0 < len(data) <= MAX_WITNESS_BYTES:
        _deny(WitnessReason.RESOURCE_EXCEEDED)

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _deny()
            result[key] = value
        return result

    def no_constant(_: str) -> NoReturn:
        _deny()

    try:
        raw = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )

        def check_depth(value: object, depth: int = 1) -> None:
            if depth > MAX_JSON_DEPTH:
                _deny(WitnessReason.RESOURCE_EXCEEDED)
            if type(value) is dict:
                for key, item in value.items():
                    _text(key)
                    check_depth(item, depth + 1)
            elif type(value) is list:
                if len(value) > MAX_ARRAY_ITEMS:
                    _deny(WitnessReason.RESOURCE_EXCEEDED)
                for item in value:
                    check_depth(item, depth + 1)
            elif type(value) is str:
                _text(value)

        check_depth(raw)
        witness = _decode(raw)
        if canonicalize_witness_bytes(witness) != data:
            _deny(WitnessReason.NONCANONICAL)
        return witness
    except WitnessDenied:
        raise
    except (
        UnicodeError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        OverflowError,
    ):
        _deny()


def digest_witness_bytes(data: bytes) -> str:
    """SHA256 of exact canonical bytes, NOT an authentication or authorization."""
    parse_witness_exact_bytes(data)
    return hashlib.sha256(data).hexdigest()
