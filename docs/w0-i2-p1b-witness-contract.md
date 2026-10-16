# W0-I2-P1b-W — static OwnerWitness codec (schema 1)

**Owns:** immutable, caller-asserted genesis identity *data*, exact canonical
UTF-8 JSON encoding/decoding and SHA256 over those exact bytes. **May:**
validate and return typed data, bytes or digest. **Must not:** read, publish or
replace a protected witness; inspect/create Source owners, Chroma or P0; issue
startup permission. **Depends on:** future separately authenticated operator
custody, Source-keyed lifetime fence and verified owner observations for any
meaningful installed use. **Used by:** a future P2 operator supervisor as a
serialization component only, never as an authorization decision.

`OwnerWitness` is a frozen schema-1 dataclass. It requires canonical UUIDs for
installation, first-install attempt, Chroma database, both distinct named DB1
and DB2 collection incarnations and the **original Journal lineage**. The
registered tenant and database names and IDs must be separately verified in
the same actual Chroma namespace by a future composing authority. SourceRoot
has absolute lexical path/device/inode/UID/GID/mode; its explicit ordered
ancestors (including `/`) carry the same metadata, and `source_mount` names one exact
registered ancestor or the Source root. This is an asserted mount-boundary
policy, **not** a mount verification or an assurance against a path swap.
Chroma root, SQLite catalog, and previously materialized Experience/Semantic
directories carry separate physical path/device/inode/UID/GID/mode assertions.
The catalog's stable file identity is an asserted expected incarnation, **not**
its evolving physical SQL bytes; any future observer must independently
establish the relevant identity policy under its lease and WAL-aware checks.
Required present-owner class/name pairs cover AgentState, Journal, WAL,
AdapterRegistry, Experience and Semantic exactly once. Presence is an
assertion, not an observation. `adapter_registry_empty_at_genesis` must be
literal `true`: this records only that a valid initialized empty registry
existed then. Its later well-formed, possibly populated JSON and ordinarily
atomic-replaced inode are deliberately **not** frozen. No new marker file is
required for empty Experience/Semantic directories.

No changing AgentState/Journal bytes, current WAL generation, registry JSON,
SQLite SQL bytes/hash or learning payloads occur in schema 1; unknown dynamic
fields are refused. Mutable-owner relationships and Journal/WAL lineage must
be observed and composed separately before any future owner start. A matching
digest is only a caller-asserted hash, not custody, completeness or authority.

`canonicalize_witness_bytes` encodes sorted-key compact UTF-8 JSON, with
required owners sorted by class; `parse_witness_exact_bytes` refuses nonexact
bytes, duplicate keys at any nesting, unknown/missing fields, unsupported
versions, invalid UTF-8/surrogates, nonfinite numbers and bad types. Limits:
32,768 file bytes, 12 JSON levels, 4,096 UTF-8 bytes/string, 256 array items
(to hold one Source ancestor per path component), 256 path components, integer 0..2^63-1 (inodes positive), modes 0..07777.
Names are 1..128 ASCII bytes `[A-Za-z0-9][A-Za-z0-9._-]*`; UUIDs must be
canonical lowercase hyphenated; every registered UUID is unique. Absolute
paths are lexical (no empty, `.` or `..` components, no trailing slash/NUL),
not resolved through filesystems. The canonical 32-KiB whole-witness bound
still applies independently to particularly deep or long paths; Source paths
of 33+ components with correspondingly complete ancestry are not rejected
merely because the former unrelated 32-item array cap was reached. Only the
two typed list fields exist and required_owners has exactly six entries.
Root directories must be distinct, nested
under Source and mutually nonoverlapping; the exact native catalog pathname
must equal `chroma_root.path + "/chroma.sqlite3"` for consistency with P1b-C.
Nested paths and alternate filenames are refused; native parent path characters
`?`, `#`, `%` and Unicode remain valid and are not interpreted as file URIs.
This is lexical consistency of caller assertions, not proof of actual custody
or catalog identity. A single-component Source root has
exactly the required `/` ancestor.
The API returns only data or a coarse typed `WitnessDenied`, without private
input in its error message. It performs no filesystem IO.

**Migration/compatibility:** explicit schema 1 only; schema changes require
new separately reviewed codec and protected migration. Existing records and
startup paths are untouched. In future P2, only a witness whose *authenticated
protected* exact bytes match P0 `owner_baseline_sha256` can contribute one
non-authorizing identity input. Neither matching hash nor P0's claimed digest
alone authenticates installation origin. A missing witness, missing P0 record
with old witness, mismatch, orphan stage, uncertain publication or interrupted
genesis means HOLD and manual operator classification, **not** retry genesis
or infer empty state. P2 must separately implement no-replace publication,
fsync/readback, P0 CAS, fresh one-shot genesis, crash-boundary recovery and
verification of mutable owners under the continuously held Source fence.

**Privacy:** the witness contains paths and identity metadata and must be
protected by the later operator custody; this module never logs or stores it.
No private Source data is used in tests. **Verification:** run scoped pytest
`tests/test_w0_i2_p1b_owner_witness.py`, Ruff check/format and mypy on this
module only. No real registration, protected witness publication, live service,
boot approval, historical completeness or R1c/G2 claim results from them.
