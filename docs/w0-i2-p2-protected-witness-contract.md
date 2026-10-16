# W0-I2-P2-W — operator-custodied one-shot OwnerWitness publication

**Status:** Rootless **mechanical** storage helper and failure tests, never an
installed operator-supervisor or an authorization to create/continue an actual
SUZKA Source. No production import or real data mutation.

**Owns:** strict one-shot no-replace publication of immutable canonical
OwnerWitness schema-1 bytes to an **explicitly preprovisioned** private
operator-controlled directory outside Source, under a separately preexisting,
never-replaced stable lock inode. The caller supplies asserted custody identity
(device/inode/UID/GID/mode), record UID/GID, installation UUID, finite path,
directory-entry and witness-size budgets. These assertions alone do NOT
authenticate their issuer, host privileges or installation.

**May:** validate the pure P1b-W OwnerWitness codec before publishing; open
private custody paths with nofollow directory descriptors; take a nonblocking
exclusive POSIX flock for one operation; refuse existing witness or abandoned
staging; create ONLY a new hidden stage **inside the supplied operator custody
directory**, write+fsync canonical bytes, link stage to the permanent
installation-specific name using a POSIX **no-replace hard link**, fsync the
parent, unlink the stage, fsync the parent again, read back and compare exact
published canonical bytes SHA-256, and return a nonauthorizing
COMMITTED_WITNESS acknowledgement. The witness filename is derived solely
from the declared canonical installation UUID. A separate observe() reads
existing bytes noncreating, rejecting missing/malformed/wrong-digest or
unsupported source identity. No automatic fallback to an empty witness.

**Must not:** issue any START or genesis permission; initialize actual owners,
SQLite/Chroma/Journal/AgentState/WAL or protected operator registration; modify
a live Source; import P0 and mutate its registration states; overwrite or repair
an existing witness; clean abandoned stages after uncertain outcomes; infer
that a correct digest proves independently authenticated installed custody,
historical Source completeness or absence of old worker processes. Merely
matching P0 owner_baseline_sha256 *in the future* is not sufficient without
independent operator-held source/config/witness-custody proof.

**Depends on:** P1b-W pure canonical codec, separately authenticated operator
custody and the future P2 source-keyed lifetime supervisor. Real first genesis
must explicitly prove no old P0 record, no existing witness and no prior Source
owner artifacts, then commit P0 FIRST_INSTALL_PREPARED -> GENESIS_IN_PROGRESS
**before ANY owner effects**; provision canonical valid-empty AdapterRegistry,
Experience/Semantic roots and Chroma/Journal/WAL/AgentState; quiesce/verify the
old worker and stable owner lineage; publish the witness once under the Source
lifetime fence; only then commit P0 GENESIS_IN_PROGRESS ->
CONTINUATION_READY with independently verified exact digest. This helper
does NOT implement these orchestration transitions, old worker drainage, PID1
generation proof or real root ownership. Failed and ambiguous transitions
require future explicit operator classification, not automatic retry.

**Used by:** the future dedicated operator-controlled P2 supervisor only.
P0 remains a short-lock mechanical CAS, P1a/P1b-C remain nonauthorizing
preflight. P3 production worker entry remains disconnected.

**Crash and recovery contract:** absent protected witness is NOT genesis
authority. Any already existing target is ALREADY_PUBLISHED; any pre-existing
stage is ABANDONED_STAGE, refusing even read, because interrupted prior
publication may have made the final target visible but not durably synchronized.
Errors after stage creation, including fsync failure before/after link, return
UNCERTAIN_COMMIT and intentionally leave stage and possible final target
untouched for manual operator classification. No helper-driven deletion,
repair, P0 reset or re-genesis. POSIX hard-link no-replace behavior and parent
fsync are tested under disposable Linux roots but are not physical power-loss
certification. Custody file opens do not confer adversarial same-UID security;
cooperative operational incidents are the accepted threat scope.

**Privacy and resource limits:** witness stores private paths/identity
metadata. Error reasons contain neither source paths nor witness bytes. Policy
requires private directory 0700, lock 0600, record 0600, canonical installation
UUID and exact-owner metadata; path components and bytes, witness bytes and
directory entries are bounded. No network access, credential reading, or
environment-selected default root.

**Migration/compatibility:** only three new files on a disconnected P2-W
development branch. Existing app/Source, systemd and host remain unchanged.
Protected P2 integration and activation are separate Human gates.

**Verification:** rootless pytest tests/test_w0_i2_p2_protected_witness.py,
Ruff check/format and scoped Mypy. Test matrix: read/publish exact bytes,
same-install repeat no-replace, absent vs corrupt/wrong digest, wrong record,
symlinks, abandoned stage, injected pre-link and post-link fsync failure,
independent process lock contention and custody replacement. Real protected
operator custody and systemd/cgroup crash containment require separately
authorized host tests and must not be claimed by this PR.

    
**R1 review correction: directory capacity and caller umask.** A first
publish reserves **two free directory entries** under the existing custody
flock before creating any stage: one transient stage and one final no-replace
hard-link. A policy with only 1 or 2 total entry slots rejects cleanly if the
required custody lock already occupies one slot. A successful record is
observable under the same bounded policy. Stage creation uses os.fchmod on its
newly created descriptor to restore the exact expected 0600 record mode,
independent of the caller process umask.

**Distinct late fsync window:** a crash or failure during the second parent
directory fsync occurs after the stage unlink. The final witness can be
visible as a single-link file while **no stage entry remains**. This still
returns UNCERTAIN_COMMIT, and observation of existing bytes does NOT permit
automatic retry, P0 READY, a new genesis, or continuation. Separately
authorized manual operator classification is required; rootless injection
does not certify physical power-loss durability.
