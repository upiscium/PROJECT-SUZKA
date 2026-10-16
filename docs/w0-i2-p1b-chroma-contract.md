# W0-I2 P1b-C: disconnected Chroma catalog observation

**Owns:** a stdlib-only, nonauthorizing observation of the two asserted collection
UUIDs in one Chroma 1.5.9 SQLite catalog snapshot. `observe(Policy)` returns a
frozen `Result(VERIFIED_UUIDS | REFUSED, startup="DISABLED_PENDING_I2")` with a
coarse refusal reason. This is partial catalog evidence, not an OwnerWitness.

**May:** accept *explicit caller assertions* for native absolute source/root/main
database paths; root and main-file device, inode, UID, GID and permissions; schema
tag `1.5.9`; tenant ID, database name and canonical UUID; two distinct names and
canonical UUIDs; and finite positive file/page/path/entries/VM-step/deadline/busy
limits. Tenant ID, database name and collection names have a 1024-byte UTF-8
ceiling. The Chroma root must be strictly inside the asserted source; the main
file must be its direct `chroma.sqlite3` child. Check paths without following
symlinks, check identities before and after, open SQLite with an escaped file URI
with only `?mode=ro`, enable `query_only`, and read the table shape, tenant,
database and both collections in one `BEGIN` snapshot. First select exactly one
tenant and database by asserted namespace and database UUID; then select each
collection by **database ID and name in SQL**, before applying the result-row
budget. Same-name collections in unrelated databases do not consume that budget.
File growth or WAL checkpoint with unchanged incarnation is not an identity
failure; bound aggregate main/WAL/SHM file sizes before and
after and refuse nonregular sidecars. Tests use new isolated rootless Chroma 1.5.9
fixtures.

**Must not:** use `chromadb` in the runtime helper, instantiate a Chroma client,
create a missing main database, use `immutable=1` or `mode=rwc`, mutate logical
SQLite contents, issue a boot token, register a source, claim startup/boot
permission, publish a protected witness, or infer G2/R1c historical completeness.
The schema tag is caller-asserted and the queried 1.5.9 columns are checked; it
is **not** independent package-version attestation or a full schema digest.

**Depends on:** an eventual independently authenticated source-keyed P2
quiescence fence and owner custody protocol. Caller assertions are never trusted
as authorization. SQLite's pathname open is not kernel-atomically bound to the
checked nofollow descriptors. The observed Source root, Chroma directory and
catalog file must not be writable by group/other, even if matching the caller's
asserted modes. This excludes different-UID writers that could otherwise
plant or replace WAL/SHM sidecars during the read; protection of all ancestors,
P2-managed custody and the cooperative Source fence remain separate obligations. Cooperative operator mistakes are in scope;
malicious arbitrary same-UID code/path races and transitive ELF/Python attestation
are deferred.

**Used by:** future I2 admission may consume this partial observation only alongside
its independently required evidence; neither this module nor any success status
enables startup. There is no compatibility change to existing startup behavior.

**Failure / recovery:** absent, swapped, locked, corrupt or unsupported catalogs,
namespace/UUID mismatches, case-insensitive SQLite collation name aliases,
other-UID-writable custody, invalid path/limits and exhausted budgets fail closed
with privacy-safe reasons. Retry only after the independent owner reconciles or
reissues fresh assertions; do not repair catalog state here. Monotonic deadline,
busy timeout and VM-step budget limit normal SQL execution, not a hard preemptive
wall-clock cap on a blocking OS open. SQLite `mode=ro` can create or modify
incidental WAL/SHM sidecars: this is **logical read-only**, not physical zero-write.
WAL commits visible to the read transaction are supported; an uncommitted writer
is not evidence. No source contents, credentials, record payloads or real deployed
paths are included in results.

**Verification:** run scoped pytest on `tests/test_w0_i2_p1b_chroma_catalog.py`
in a disposable Chroma 1.5.9 environment and scoped Ruff check/format and Mypy
on only the new Python files. These tests never address a production source.
