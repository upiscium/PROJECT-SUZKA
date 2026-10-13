# W0-I0 — FHS source-storage examples (not an installation guide)

**Status:** Human-approved **W0-I0 implementation scope only**, tracked in
[Issue #300](https://github.com/upiscium/PROJECT-SUZKA/issues/300).
This stage adds examples, documentation, and isolated tests. It does **not**
change runtime path semantics, create an installation, register an existing
source, migrate data, install a systemd unit, or authorize G2 evidence issuance.

In particular, **do not install or activate the example service unit**.
W0-I1 (trusted configuration/source-owner admission) and W0-I2 (external
installation/continuation registration and pre-service validation) are not
implemented. Migration and activation each require further Human approval.
The active `deploy/systemd/suzka-api.service`, existing
`deploy/env/backend.env.example`, repository `config.yaml`, and runtime
configuration loader remain unchanged.

## Intended paths for a Linux service

| Purpose | Proposed location | Persistence / authority |
| --- | --- | --- |
| Configuration | `/etc/project-suzka/config.yaml` | Operator-managed, read-only to service |
| Credentials | `/etc/project-suzka/backend.env` | Separately restricted; never publish values |
| Private state root | `/var/lib/project-suzka` | Durable, owner-controlled source evidence |
| Journal | `/var/lib/project-suzka/journal/event_journal.jsonl` | R07; keep its adjacent owner lease marker |
| AgentState | `/var/lib/project-suzka/agent_state.json` | Authoritative persisted state |
| StateWAL | `/var/lib/project-suzka/state_wal` | Durable WAL and recovery material |
| Chroma DB1/DB2 | `/var/lib/project-suzka/chroma` | Persistent memory and pending episodic staging |
| Experience | `/var/lib/project-suzka/experience` | R12 history and derived evidence |
| Semantic | `/var/lib/project-suzka/semantic` | Independent semantic records and receipts |
| Dreams | `/var/lib/project-suzka/dreams/dream_dataset.jsonl` | Private derived training data |
| Learned adapters | `/var/lib/project-suzka/adapters` | Retained training output |
| Adapter registry | `/var/lib/project-suzka/adapter_registry.json` | Committed registry |
| Evaluation results | `/var/lib/project-suzka/eval_results` | Durable results |
| Evaluation *inputs* | `/usr/share/project-suzka/eval_sets/default.json` | Proposed packaged, immutable source; **not installed** |
| Runtime-only material | `/run/project-suzka` | Ephemeral sockets/PID; **not** historical proof |
| Replaceable cache | `/var/cache/project-suzka` | Optional; model/download cache configuration not implemented |
| Program and virtualenv | `/opt/project-suzka/venv` | Preinstalled immutable/non-editable package, **not installed** |

The nine `Path`-valued configuration fields are supplied as explicit absolute
paths in `deploy/config/config.fhs.yaml.example`. `ExperienceStore` and
`SemanticStore` currently derive their roots as siblings of the configured
Chroma root, so `.../chroma` maps them to `.../experience` and
`.../semantic`. Episodic staging remains beneath the Chroma memory root.

These paths do not relocate unconfigured third-party caches or subprocess
outputs. The installed evaluation-input path is a **future provenance decision**;
the example deliberately does not provide or install its contents. Current
`load_eval_sets()` ignores missing files, so the example must not be used to
perform a trusted evaluation until a later gate enforces availability, trusted
provenance, and nonempty required inputs.

## Current behavior versus W0 service admission

The current `SUZKA_CONFIG_PATH` variable selects a YAML file. The current
`load_settings()` neither authenticates that file's owner and ancestry nor
resolves relative data paths against a fixed state root. If the variable is
absent, it falls back to a package-relative default configuration. Current
owners can create directories or initialize state during startup.

W0-I0 **does not alter these behaviors**. A later service-mode admission
must refuse missing/untrusted config, relative or ambiguous authority paths,
unregistered injected owners, or unexplained missing persisted state **before**
any constructor performs recovery or creates a new source root. Validation
must bind the exact configuration bytes and trusted root identity to the
running source owner; an arbitrary environment override or a later CWD check
is not such proof. The final policy for ancestor symlinks, `..` components,
hardlinks, and external mounts is an independent review requirement.

The example service uses an absolute interpreter path rather than the existing
CWD-sensitive `uv run`/editable checkout path. This documents an intended
packaged deployment; it does not create or verify the installation at
`/opt/project-suzka/venv/bin/python3`. The current application still lacks
the W0-I1 and W0-I2 startup safeguards, so even a syntactically valid unit is
**not activation-ready**.

## Development and test profiles

A future rootless developer profile may explicitly select
`$XDG_STATE_HOME/project-suzka` (when `XDG_STATE_HOME` is absolute), or
`~/.local/state/project-suzka`. Tests must use isolated temporary roots and
must not write to actual `/etc`, `/var`, or `/opt` locations. W0-I0 does
not implicitly interpret these variables, does not add a profile switch, and
does not replace the existing repository defaults. Select all nine paths
explicitly in a test-specific config; never infer durable source identity
from the current working directory.

The tests for these examples read the repository artifacts and create only
disposable journals under `tmp_path`. They do not open Chroma, load models,
restart a service, or admit real historical source evidence.

Python and pytest may generate ordinary bytecode/test-runner caches if the
parent runner permits them. For a cache-free checkout invocation, use
`PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider ...`; the cross-CWD
child processes explicitly disable bytecode creation. Such caches are not
source-owner evidence or part of the persistent-data migration.

## File ownership and service manager hazards

The intended private state directory is service-owned (for example,
`suzka:suzka`, mode `0700`). Config can be operator-owned and read-only to
the service, while secret credentials need separately restricted admission.

**Do not add `StateDirectory=project-suzka` to an active service before
source migration and pre-service checks are approved.** systemd may create
the specified state directory and recursively change ownership of its
contents when the top-level ownership does not match the configured
`User=`/`Group=`, and can adjust directory modes. Such changes would touch
physical source artifacts before application-level `ExecStartPre` checks.
The example unit intentionally omits `StateDirectory`. Later deployment
must use an independently reviewed, non-mutating pre-service check *before*
any manager operation that could change the stored evidence; simply placing
a check inside the same state-managing unit is insufficient.

`RuntimeDirectory=project-suzka` in the example concerns ephemeral
execution-time state only. The Journal lease and its durable/source-owning
artifacts are **not** relocated into `/run`.

## First install, continuation, and offline migration (future gates only)

A missing directory is **not evidence of a first installation**.
A later operator-controlled declaration, protected independently of the
manager-created state root, must explicitly identify first installation
versus continuation. The proposed first-install registration is one-shot;
a crash between initial source creation and durable continuation
registration must stop for operator classification, not repeat genesis.
This registration is a *deployment admission authority*, not R1c SourceEpoch
or a G2 source certificate.

For a continued or migrated installation, the independent pre-service
procedure must verify the expected state root, owner/mode, path ancestry,
registered source identities, and required Journal/WAL/Chroma/Experience
presence **before** allowing source constructors or systemd ownership
changes. A missing/mismatching source must fail closed; do not silently
initialize a new empty source or fall back to a legacy `.suzka` under an
arbitrary CWD.

Any later migration requires separate explicit Human approval and at least:
quiescing every writer (including DB1/SQLite, AgentState, WAL, Journal,
Semantic and Experience), an operator-controlled confidential backup with
readback, an identity-preserving stage and cutover, source-owner consistency
verification, classified startup maintenance and invalidation of old
in-memory evidence epochs. Never copy live Chroma/WAL files, replay
records to manufacture provenance, or release the old source before a
separately authorized rollback/retention decision. A rollback is new
classified maintenance, not automatic reuse of a previously accepted
cross-owner epoch.

Using absolute paths removes CWD-relative aliasing. It does **not** by
itself guarantee root/lock inode custody, exclude an unregistered writer,
resolve interrupted owner recovery, or prove an atomic R07 + DB1 + R12 + B
view. R1a/R1b production readers, W0-I1/I2, W0 migration/activation,
R1c SourceEpoch and G2 evidence issuance remain separately gated.

See [#299](https://github.com/upiscium/PROJECT-SUZKA/issues/299)
for the approved architectural direction, and
[#300](https://github.com/upiscium/PROJECT-SUZKA/issues/300)
for the five-new-file I0 implementation scope.
