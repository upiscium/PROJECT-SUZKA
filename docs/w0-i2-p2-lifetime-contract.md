# W0-I2-P2-L — Source physical-lifetime claim mechanics

**Status:** Rootless local implementation under Issue #309. Disconnected from production launchers. This helper is NOT a working I2 admission service.

**Owns:** One local nonblocking POSIX flock on a stable PRE-EXISTING lock file keyed by SourceRoot physical (st_dev, st_ino), not installation ID nor caller-selected config. Different managed installations naming the same physical Source MUST use the same operator-controlled custody directory; lock_name(dev,inode) deterministically derives the shared filename.

**May:** accept explicit UNA UTHENTICATED source, operator custody and lock metadata (absolute native paths, dev/inode/UID/GID/mode and bounded component limits); walk native paths with nofollow directory descriptors; acquire an exclusive nonblocking lifetime flock on an already provisioned regular 0600 single-link file; recheck its name and FD identity while held; close only after a future supervisor has independently PROVED the old managed writer generation drained. Tests preprovision lock files in rootless private temporary directories. Helper itself never creates a lock or Source directory.

**Must not:** create a missing Source or lock, substitute P0 short registry lock for the lifetime fence, change Source metadata, perform P0 CAS, claim PID/systemd cgroup drained, trust caller-provided operator assertions as installed custody, import owners, return BootPermit or ALLOW_START, inspect historical WAL/Journal, or launch a service. A lock is mechanical exclusion, not startup admission.

**Depends on:** a later separately authorized operator-controlled supervisor supplying a SINGLE protected host-global lock directory, independently authenticated installed SourceRoot identity from P0+OwnerWitness, release/config identity, and a preprovisioned exact lock with verified metadata. P2 must prove actual generation-specific old-worker drain using cgroup/manager evidence BEFORE startup and before releasing any live claim. The fact that a flock was released on process exit/crash does not prove an old child or fork stopped. Nonretryable RECOVERY_REQUIRED remains separately required.

**Used by:** future trusted operator supervisor, never API worker, Source owner, server.main, dev_asgi or standalone application. LiveClaim provides held, recheck, and explicit close only. There is NO automatic context-manager release: unexpected cancellation must not silently unlock a still-live writer before independently proven generation drain. Closing the owner descriptor intentionally does NOT issue LOCK_UN; fork-inherited open-file-description copies retain the lock until the last descriptor closes. This is mechanical fail-closed behavior, not cgroup containment proof or a transferable startup token. The future supervisor must still prove writer death before close.

**Failure/recovery:** missing lock/source, symlinked components, wrong inode/uid/gid/mode, unexpected file replacement, concurrent contender or unreadable custody refuse with typed path-free errors. No auto-provision, fallback, repointing, cleanup or retry-as-genesis. Operator changes/mount repointing which ignore cooperative fences remain outside the kernel-atomic guarantee. Source changed inode must REFUSE against original independently protected witness BEFORE selecting a new key; dev+ino alone is not a historical payload certificate.

**Privacy:** only caller-supplied local paths and metadata; errors exclude private path bytes, contents and credentials. No external API or network.

**Migration/compatibility:** new disconnected helper and tests only; no existing service/Source/startup path changes, no migration or privileged deployment. This is NOT P2-complete; real process-generation and systemd cgroup proof remains necessary.

**Verification:** rootless focused pytest tests/test_w0_i2_p2_lifetime_lock.py plus Ruff/format and Mypy. Synthetic cases cover two installations with one Source, cross-process contention, missing/swapped/symlinked root and lock, preprovisioned exact key, short P0-like operations while lifetime lock remains held, and cleanup/reacquisition. Host-systemd/UID and real Source fault tests require separate Human authority.

    
**R1 code-review corrections:** recheck catches missing/repointed Source,
custody and lock paths as path-free, typed reasons; it does not leak
filesystem error paths. The Source physical key accepts device number zero
for compatibility with P0/P1b static identity ranges, though actual target
filesystem support is a separate installed-host verification. Inodes stay
positive. Caller code must not use with-acquire automatic cleanup; it must
perform independently verified old-worker generation drain before invoking
explicit close, and an unexpected service termination still requires the
later P2-H systemd/PID1 containment proof.
