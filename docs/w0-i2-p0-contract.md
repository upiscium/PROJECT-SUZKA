# W0-I2-P0 — nonauthorizing local registration mechanics

**Status:** Human-scoped implementation of Issue #310, beneath I2 design
Issue #306 and root-supervisor proof Issue #309. This is **not** an
installation guide, a verified bootstrap, or authorization to create,
replace, recover, or read a real Source Owner.

The module is deliberately disconnected from the application. It must not
be imported by server.py, dev_asgi.py, I1a, Journal, WAL, Chroma or an
installed service. It does not read environment settings, install under
/etc or /var, grant root privileges, contact a service, or return a startup
capability. Existing deployed services must not rely on it.

## Trust boundary and outcome

The standard-library-only module suzka/installation_registration.py offers
**local record mechanics**. RegistryCustody must be supplied explicitly
and names a PRE-EXISTING private directory and a distinct, PRE-EXISTING,
never-replaced lock file, with expected device/inode/UID/GID/mode. The
caller also supplies the record name/owner/group/mode, maximum JSON byte
size, path bytes and maximum path components. There are no environment-
selected custody paths, numerical capacity defaults or implicit root UID.

Component-by-component O_NOFOLLOW opens and FD checks verify that actual
objects match the **asserted** facts. **This component cannot authenticate
who asserted them.** Someone who controls both the directory and the input
policy can manufacture plausible local registration. flock excludes only
cooperating processes on the same lock inode; it does not authenticate an
installation or prevent a writer that ignores that lock.

Every successful mutation returns only **COMMITTED_RECORD**, a
nonauthorizing local write acknowledgement. No ALLOW_START, BootPermit,
BootDecision, startup token, verified root source digest, or service claim
is issued. The read() method returns untrusted record data, NOT authority.

## Strict record and state-machine contract

The bounded, versioned JSON record stores claimed installation identity,
one-shot genesis intent ID, configuration and installed-code SHA-256 digests,
source-root pathname and device/inode assertions, ASGI target, revision,
owner-baseline digest assertion and current/previous claim identifiers.
They are all **caller assertions**, not verified installed measurements.

Unknown or duplicate keys, invalid UTF-8, invalid types/UUID/digests,
unsupported schema, malformed JSON, oversized files, wrong custody or
symlinks are rejected. A clean absence does NOT authorize new genesis.

| Current record state | Explicit next state | Mechanical condition |
| --- | --- | --- |
| No record | FIRST_INSTALL_PREPARED | Explicit first initialization, revision 1 |
| FIRST_INSTALL_PREPARED | GENESIS_IN_PROGRESS | Exact previous revision and identity |
| GENESIS_IN_PROGRESS | CONTINUATION_READY | Caller supplies claimed owner-baseline digest |
| Any nonterminal state | RECOVERY_REQUIRED | Explicit terminal classification |
| CONTINUATION_READY | START_CLAIMED | New claim and exact revision |
| START_CLAIMED | CONTINUATION_READY | Caller claims clean stop, consumes active claim |
| RECOVERY_REQUIRED | No transition | Future manual recovery separately gated |

Transitions compare the *entire* expected record with the current record
under the lock and require revision increment by exactly one. The
first-install attempt and installation/config/root/code/ASGI assertions
remain fixed; an immediately reused previous claim, illegal transition,
stale revision or second initialization fails.

The component does NOT verify that genesis finished, a stop was clean, an
owner baseline is authentic or the service child exited. Consequently
GENESIS_IN_PROGRESS -> CONTINUATION_READY stores an explicit assertion,
never a boot permit. An interrupted genesis cannot automatically return
to first installation. RECOVERY_REQUIRED is terminal under P0.

## Physical file and failure contract

Each operation opens its absolute directory component-by-component with
O_DIRECTORY and O_NOFOLLOW; verifies the final path, owner/mode/device/inode,
and opens a separately preprovisioned lock with O_NOFOLLOW. The lock is
checked both before and after a nonblocking exclusive flock. The
registration file is read through O_NOFOLLOW and O_NONBLOCK, with size,
regular-file type, single-hardlink, owner/mode/device and postread identity
checks. Reads require the full, stable file length—not merely a valid JSON
prefix. The caller also specifies a strict maximum directory-entry budget.

Before any read or update, the registry directory is scanned under the
preprovisioned lock for earlier interrupted staging names. An abandoned
staging file results in ABANDONED_STAGE, not automatic cleanup or another
publication. The library never deletes a staging entry it cannot identify
as its own newly created inode. Only a separately authorized operator
recovery/classification can decide whether to retain, inspect or remove
those artifacts. This prevents repeated crashed starts accumulating more
temporary files. A malicious party controlling the directory can still
block this mechanism: the component is not a root authentication boundary.

Publishing uses a new same-directory temporary file, complete write, file
fsync, close, protected-path/lock revalidation, atomic os.replace,
parent-directory fsync and record readback. The stable lock inode is never
renamed or replaced by this module.

An error before replace is a typed denial. Any failure after replace,
before confirmed directory fsync or readback, returns UNCERTAIN_COMMIT.
The caller must never interpret that as a committed authorization. This
also applies to a failure in the final lock/directory identity check after
publication. On restart, an ambiguous state needs protected operator
classification, not an automatic retry of GENESIS_IN_PROGRESS. An abrupt subprocess crash test
is not equivalent to a physical power-failure durability guarantee.

This helper holds its lock for ONE operation only. It does not provide a
lifetime admission lease and cannot prevent an existing service child
from surviving its supervising process.

## Testing and limitations

tests/test_w0_i2_registration.py uses only private, unprivileged tmp_path
roots. It covers strict record validation, size and custody failures,
symlinks/FIFOs/lock replacement, invalid transitions, missing registration,
postreplace uncertainty, competing processes and injected filesystem faults.
No actual service, Source Owner, Journal, AgentState, WAL, Chroma,
Experience, Semantic, AdapterRegistry, systemd or installed /etc-/var-
managed registration is exercised.

Passing the tests **does not** establish root-owned operator authority,
real startup permission, process-lifetime exclusivity, physical power-loss
durability, integrity of historical Source payload or SourceEpoch/G2 proof.

## Prerequisites for a future root-controlled service

Separate explicit Human approvals would be needed to:

1. Pin and verify the actual installed executable, package/module import
   ancestry, ASGI target, service principal and exact original I1a bytes.
   Reject editable source, arbitrary PYTHONPATH, PATH, dev_asgi and
   environment/default-config substitution.
2. Authenticate registration policy and instance identity in a protected
   operator-owned store outside service-owned Source roots. Hold an
   exclusive authority claim for the ENTIRE service child lifetime and
   prove its PID/credential binding, stop, restart and systemd cgroup
   containment even if the child ignores a disconnected socket.
3. Consume a one-shot GENESIS_IN_PROGRESS durably BEFORE any Source Owner
   constructor or systemd StateDirectory write. An interrupted genesis
   must require explicit recovery classification, not automatic bootstrap.
4. Perform noncreating, WAL-aware continuation checks for required
   AgentState, Journal, WAL, Chroma DB1/DB2 UUIDs, Experience, Semantic,
   AdapterRegistry and registered model artifacts. Distinguish valid
   empty owner state from previously materialized then lost state.
   Root/collection identity is not full historical payload integrity.
5. Before any positive owner startup, implement I1b-S0c immutable
   admitted settings and denial of unregistered injected owners. Before
   adapter evaluation effects, implement E2 required input revalidation
   including deterministic_score.

All of these remain separate Human gates. P0 does not install, migrate,
activate or grant a root-controlled first-install/continuation authority;
it also does not implement R1c SourceEpoch or G2 evidence.
