# ADR — G2 source-preserving verification reads

- **Date:** 2026-10-12 (JST)
- **Status:** **ACCEPTED, DESIGN POLICY ONLY** by explicit Human decision
- **Authority:** [Issue #288 — Human amendment](https://github.com/upiscium/PROJECT-SUZKA/issues/288#issuecomment-6119507141)
- **Related:** [A1–A27](https://github.com/upiscium/PROJECT-SUZKA/issues/283#issuecomment-6058471359), [EvidenceView v0.1/v0.2](https://github.com/upiscium/PROJECT-SUZKA/issues/288#issuecomment-6107774289), [G1-C0 v1.1](https://github.com/upiscium/PROJECT-SUZKA/issues/288#issuecomment-6116002437)
- **Implementation status:** No production G2 verifier/issuer exists; positive source proof unavailable

## Context

An earlier G2 design required **no physical filesystem write during verification**, including atime, SQLite WAL/SHM, search-index, and cache mutations. That is stronger than the evidence-integrity objective.

In isolated synthetic ChromaDB 1.5.9 experiments, cold process reopening and fetching one record modified SQLite and vector-index files; in one subsequent synthetic experiment three separate cold reopens each modified Chroma files but returned the same document, metadata, embedding and record count. This is **not** production source-attestation evidence. Chroma's sqlite database must not be classified wholesale as a cache merely because some writes may be bookkeeping.

The Human explicitly removed absolute physical no-write from the G2 acceptance criteria. Source integrity, coherent state and separation of verification from repair remain mandatory.

## Decision

### 1. Authoritative evidence is not mutated by its verification

A G2 verifier **MUST NOT** create, delete, silently repair, reclassify, replace or backfill the original authoritative evidence it is verifying. This includes the committed DB1 episodic document and metadata and its exact original operation digest; R07 Journal transaction and participant finality; the distinct forward-only event-time B/R09 Context preimage; R12 lifecycle/current status and required revision history; deletion, retraction, provenance and retained source identity.

Derived vector indexes, caches, ephemeral workspace, atime, SQLite physical encoding, WAL/SHM and internal housekeeping **MAY** change **only where those effects are classified as not modifying the canonical source record, source eligibility or attested owner incarnation/epoch**. No file name is inherently safe: authoritative records may live in chroma.sqlite3. A write changing the real source under test is disallowed even if an API getter still returns a plausible value.

### 2. Joined observations are from a coherent source epoch

Every R07 + B + DB1 + R12 positive proof requires a coherent, authenticated source-owner snapshot or an equivalent owner-proven read protocol. It **MUST NOT** combine data from different owner epochs or from an unfinished, retracted, deleted, pruned, superseded or ABA-rolled-back source. Concurrent authorized writers require documented coordination or repeatable version-checked snapshots; unprovable consistency fails closed.

The eventual F/F2 subject-admission gate must independently revalidate original source epochs/versions and current eligibility inside its approved atomic commit boundary. A prior successful source lookup, checksum, asserted identifier or model narrative cannot be reused as authority.

### 3. Source verification and recovery are separate operations

G2 is an **authority-nonmutating verifier**, not a source recovery, migration, compaction or schema-repair participant. R07 interrupted-rotation cleanup, R12 load-time temp/compaction recovery, Chroma legacy scrub/upgrade and other changes affecting source authority may happen only in a separately authorized owner-maintenance/reconciliation phase with a classified outcome and a new eligible proof snapshot. The verifier must not secretly perform such repairs and then assert the altered bytes were always the original evidence.

Ordinary backend index/cache writes are not automatically "recovery"; they are acceptable only after source-effect classification and a sound snapshot proof.

## Adversarial and dependency acceptance gates

1. Identify actual source-owned canonical DB1 rows, metadata, event-operation digests, current lifecycle, and relevant SQLite tables. Classify derived vector/index/cache artifacts based on real semantics, not filenames.
2. In **ephemeral synthetic fixtures only**, compare before/after logical IDs, complete documents, metadata, embeddings, row counts and canonical values; compare external source owners R07/B/R12 when separate authorized test fixtures exist. Trace filesystem/SQLite/WAL/index/atime writes diagnostically, not as a blanket rejection criterion.
3. Exercise cold reopen, preopened client, valid/missing/duplicate/malformed IDs, concurrent valid owner writes, crash/restart, partial/failed reconciliation, source retraction, rollback ABA, retention gaps and read/commit races. Validate source IDs inside a trusted Memory/G2 owner even if Chroma accepts an empty ID.
4. Verify privacy: incidental caches and logs must not generate unauthorized cleartext source copies, hidden provenance, or public source enumeration. Same-UID hostile direct file writers remain outside the chosen threat model; model/operator/Web/API forged source assertions remain in scope and fail closed.
5. If source immutability, custody or snapshot coherence cannot be proven, issue **no** positive G2 source certificate. Proposed result labels SOURCE_INTEGRITY_UNPROVEN, SNAPSHOT_COHERENCE_UNPROVEN and RECOVERY_REQUIRED are **design-only names**, not shipped APIs. The old READ_EFFECTS_UNPROVEN rejection must not fire solely because harmless physical writes occurred.
6. A new Memory-owned immutable original-operation artifact is a **fallback** only if source-preserving coherent verification cannot be achieved using the existing Chroma source owner. It needs its own later Human privacy, retention, space and source-owner scope decisions; it is not mandated merely by observing SQLite/index writes.

## Rejected alternative and precedence

**Rejected as overconstraint:** treating *any* physical file metadata/content change during a database getter as an automatic G2 failure. Such changes can be independent of source authority, while proving zero file writes would not by itself establish immutable source truth or absence of multi-owner TOCTOU.

This ADR **supersedes only** the absolute physical no-write condition in earlier Issue #288 evidence, including prior EvidenceView/G1/G2 read designs and Chroma tests. Historical observations and original review claims remain traceable but are not the current acceptance rule on this one dimension.

**Unchanged:** frozen A1–A27, D9 MUST NOT, G1-C separate bounded forward-only capture and no R07 schema migration, same-UID hostile direct-file-writer OUT only, accepted EvidenceView Path V (private verification) versus Path N (actual R08-selected cognitive evidence access), G1-P terminal committed Context GET, G3 separate subjective meaning/admission, capacity/retention/source owners and non-bypass of R14.

**Gate:** This is an accepted policy, not an authorization to implement a production G2 reader/issuer. G1-C1, G2-R source owner code, G2-E, G3, G4/U5–U7 and PR Ready/Merge still require separately reviewed exact-scope Human decisions.
