# W0-I2-P1a — disconnected Source OWNER preflight

**Owns:** a local, rootless, read-only *partial observation* of caller-named
existing Source filesystem artifacts. **May:** inspect bounded file bytes,
metadata and a bounded Journal directory listing through nofollow descriptors.
**Must not:** initialize, repair, authorize Owner startup, infer absent identity,
inspect Chroma UUIDs or provide a boot capability. **Depends on:** separately
authenticated installed policy and protected operator witness (NOT implemented
here), plus already provisioned initialized state. **Used by:** a future
Human-gated supervisor only as one nonauthorizing input, never by server.py,
dev_asgi.py or an Owner constructor. The module is disconnected from boot.

`observe(Policy)` returns a frozen `Result`: either
`P1A_PARTIAL_UNVERIFIED_CHROMA` or `REFUSED` with a coarse typed refusal.
`startup` is always `DISABLED_IN_P1A`. Caller assertions are UNAUTHENTICATED;
matching an assertion does not establish custody. No file paths, payloads or
JSON exceptions are returned. No positive result claims complete historical
payload integrity, WAL/Journal hash proofs, SourceEpoch/G2, or W0 admission.

The caller explicitly provides six stable directory identities (Source root,
Journal, WAL, WAL generations, previously registered Experience and Semantic),
required snapshot, active Journal, WAL manifest, boot anchor, referenced
generation, AdapterRegistry, and a tuple of known rotated Journal segments.
Directories bind device/inode/UID/GID/mode. Mutable files bind device,
UID/GID/mode, regular single-link type and nonzero size; stable inode is
optional ONLY when a protected witness guarantees it. Valid normal atomic
file replacement does not require frozen inodes. Every path is absolute,
canonical, component-bounded and under Source. WAL manifest and boot anchor
have exact schema-1 keys/types; the required active generation pathname is
bound to the WAL manifest, while an independently checked BootAnchor may
refer to an older generation after a legitimate recovery/rebaseline. This is
syntax and presence, not WAL integrity or historical completeness proof. Strict JSON rejects
duplicate keys and nonfinite constants. Registry requires `{ "adapters": [] }`
or a populated list with exact current entry fields, types and statuses:
missing is not empty. A later separately Human-gated genesis must provision
valid empty registry and Experience/Semantic roots; P1a refuses their absence.

All six limits are mandatory finite positive integers (not bool): per-file
bytes, total metadata visits, path bytes, path components, Journal entries and
rotated segments. No `pathlib.open` or owner inspection methods are used.
Journal `.rotation*`, malformed suffixes, missing known segments, noncontiguous
retained sequence and missing active segment refuse without calling the
repairing `_journal_artifacts_unlocked`. Journal and generation JSONL receive
bounded syntax/schema-version checks; snapshot JSON requires an object root.
These shallow checks do NOT prove records, hashes, complete history or
cross-artifact coherence. A
concurrent service-UID writer can race this non-locking snapshot; stable root
rechecks and descriptor stat comparisons narrow, not eliminate, that window.
Same-UID adversarial tampering is outside the approved W0 threat scope.

**Failure/recovery:** absence, identity mismatch, malformed content, size or
metadata exhaustion returns refusal without mutation; operator classification
is required for interrupted rotation or lost key state. No automatic genesis
or recovery. Read-side effects are OS read access/atime/cache only. SQLite
read-only clients can sometimes create auxiliary files, but P1a never opens
SQLite or Chroma; their WAL/DB1/DB2 UUID inspection is deferred to P1b.
No systemd, privileged policy authentication or service lifetime claim exists.

**Migration/compatibility:** no existing boot path or import changes. No old
Source state is migrated. **Privacy:** results contain no private paths or
contents; callers must protect policy inputs independently.
**Verification:** scoped `nix develop --command python3 -m pytest -p
no:cacheprovider -q tests/test_w0_i2_preowner_preflight.py`, scoped `ruff
check`, `ruff format --check`, and optional mypy. All fixtures use private
pytest tmp_path; neither a live Source nor SQLite is inspected.
