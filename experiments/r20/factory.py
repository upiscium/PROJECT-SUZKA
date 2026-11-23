"""The genesis-subject factory (D1 sec. 3).

Every subject gets a NEW root (``exist_ok=False``) with its own Journal,
SubjectHistory, target and ``obligation.key``; nothing is shared between two
subjects.  SubjectHistory adoption is automatic at the first start of a fresh
Journal (D1 fact 3), so the factory prepares the preconditions
(``subject_history.enabled`` from the first start) and VERIFIES the result
(``verify_first_start``); it performs no separate "adopt" step.

Profiles S0 / F1 / S1 / NEG-plain / NEG-r15 (sec. 3-3).  S0 and F1 set
``subject_history.plan_hold_receipts: true`` (IP-14b HD-3 (a)); the receipt
writer (IP-14b-3) is a separate unit, so until it merges an observed HOLD is
``HOLD_UNSEALED`` and never a manifest copy.

The admin token layer is gone (``api.admin_token_env`` is ignored), so no token
is created or sent; the destination-bound client (``client.py``) is what keeps
an unauthenticated, irreversible activation POST away from a non-harness server.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import os
import re
from pathlib import Path
import socket
import subprocess
from typing import Any
import uuid

import yaml  # type: ignore[import-untyped]

from experiments.r20 import guards
from experiments.r20.guards import GuardRefusal
from experiments.r20.polling import short_read

from suzka.config.schema import Settings
from suzka.config.settings import load_settings
from suzka.r18.obligation import SECRET_FILE
from suzka.r18.target_init import initialize as initialize_target

SUBJECT_JSON = "subject.json"
PORT_RANGE = (41000, 60999)
_AF_UNIX_LIMIT = 100


class Tier(str, Enum):
    T1 = "T1"  # in-process, absolute paths, fake clock possible
    T2 = "T2"  # child process, relative paths with cwd = root, real clock


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    r15: bool  # S1 / S2 admitted by the harness
    driver: bool  # R16 driver activation
    self_goal: bool  # R16 self-goal activation (S1 only, HD-22-4 (a))
    r18: bool
    r19: bool
    plan_hold_receipts: bool


PROFILES: dict[str, Profile] = {
    "S0": Profile("S0", r15=True, driver=True, self_goal=False, r18=True, r19=True, plan_hold_receipts=True),
    "F1": Profile("F1", r15=True, driver=True, self_goal=False, r18=False, r19=False, plan_hold_receipts=True),
    "S1": Profile("S1", r15=False, driver=True, self_goal=True, r18=False, r19=True, plan_hold_receipts=False),
    "NEG-plain": Profile("NEG-plain", r15=False, driver=True, self_goal=False, r18=False, r19=False, plan_hold_receipts=False),
    "NEG-r15": Profile("NEG-r15", r15=True, driver=True, self_goal=False, r18=False, r19=False, plan_hold_receipts=False),
}

# (section, key) of every Path in the repository config the factory relocates;
# each value stays under ``.suzka/`` in the subject root.
_PATH_KEYS = (
    ("memory", "persist_directory"),
    ("sleep", "dream_dataset_path"),
    ("qlora", "output_dir"),
    ("adapter_registry", "path"),
    ("adapter_registry", "eval_result_dir"),
    ("agent_state", "path"),
    ("event_journal", "path"),
    ("state_wal", "directory"),
    ("chat_log", "path"),
    ("subject_history", "path"),
    ("r18", "target_path"),
)


@dataclass(frozen=True, slots=True)
class GenesisSubject:
    """What the factory created.  Nothing here is a secret."""

    run_id: str
    profile: Profile
    tier: Tier
    base: Path
    run_dir: Path
    root: Path
    evidence_dir: Path
    control_dir: Path
    port: int
    target_path: Path | None  # absolute

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def control_socket(self) -> Path:
        return self.control_dir / "ctl.sock"

    @property
    def subject_json(self) -> Path:
        return self.evidence_dir / SUBJECT_JSON

    @property
    def provider_config(self) -> Path:
        return self.run_dir / "provider.json"

    def settings(self) -> Settings:
        """The parsed settings.  T2 relative paths resolve against the cwd,
        so the caller (``serve``) has already changed into the root."""

        return load_settings(self.config_path)

    def resolved_settings(self) -> Settings:
        """Settings with every relative path made absolute under the root
        (for in-process use of a T2 root by a poller).  T1 is unchanged."""

        raw = _raw_config(self.config_path)
        return Settings.model_validate(_absolutize(raw, self.root))


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    """The ids of sec. 3-1 (identifiers, digests; no secret, no private text)."""

    run_id: str
    profile: str
    tier: str
    port: int
    journal_lineage_id: str
    genesis_incarnation_id: str
    history_store_id: str
    history_subject_id: str
    target_instance_id: str | None
    obligation_key_sha256: str | None
    config_sha256: str
    git_head: str
    provider: dict[str, object] = field(default_factory=dict)

    def id_tuple(self) -> tuple[str | None, ...]:
        return (
            self.journal_lineage_id,
            self.genesis_incarnation_id,
            self.history_store_id,
            self.target_instance_id,
            self.obligation_key_sha256,
        )


# ------------------------------------------------------------------ helpers


def _raw_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise GuardRefusal("CONFIG_INVALID", str(path))
    return loaded


def _absolutize(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    out = json.loads(json.dumps(raw))
    for section, key in _PATH_KEYS:
        value = out.get(section, {}).get(key)
        if isinstance(value, str) and not os.path.isabs(value):
            out[section][key] = str(root / value)
    return out


def free_port() -> int:
    """A free loopback port in a high range, never a forbidden one."""

    low, high = PORT_RANGE
    for _ in range(200):
        candidate = low + uuid.uuid4().int % (high - low)
        if candidate in guards.FORBIDDEN_PORTS:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
        return candidate
    raise GuardRefusal("NO_FREE_PORT")


def git_head() -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(guards.repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "unknown"


def _relocate(raw: dict[str, Any], root: Path, tier: Tier) -> None:
    for section, key in _PATH_KEYS:
        value = raw.setdefault(section, {}).get(key)
        if value is None:
            continue
        text = str(value)
        if text.startswith("~") or os.path.isabs(text):
            # The repository config only names relative paths; never keep one.
            raise GuardRefusal("CONFIG_PATH_NOT_RELATIVE", f"{section}.{key}")
        raw[section][key] = str(root / text) if tier is Tier.T1 else text
    registry = raw.setdefault("adapter_registry", {})
    registry["eval_sets"] = []


def build_config(
    profile: Profile,
    root: Path,
    port: int,
    tier: Tier,
    *,
    deterministic: bool,
    r18: bool,
    r19: bool,
    run_id: str,
    r19_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The raw config dict of one subject (the repository config, overridden)."""

    raw = _raw_config(guards.repo_root() / "config.yaml")
    _relocate(raw, root, tier)
    raw.setdefault("api", {}).update({"host": "127.0.0.1", "port": port})
    # SubjectHistory's subject id is the project name: one per subject.
    raw.setdefault("project", {})["name"] = f"PROJECT-SUZKA-R20-{run_id}"
    raw["subject_history"]["enabled"] = True
    raw["subject_history"]["plan_hold_receipts"] = profile.plan_hold_receipts
    if deterministic:
        raw.setdefault("emotion", {}).update(
            {"timer_enabled": False, "appraisal_response_rate": 1.0}
        )
    raw["r18"]["enabled"] = r18
    raw["r18"]["approval_origins"] = [f"http://localhost:{port}"]
    raw["r18"]["approval_trusted_networks"] = []
    r19_section: dict[str, Any] = {"enabled": r19}
    r19_section.update(r19_overrides or {})
    raw["r19"] = r19_section
    return raw


# ------------------------------------------------------------------- create


def create_subject(
    profile: Profile | str,
    *,
    base: Path | str | None = None,
    run_id: str | None = None,
    tier: Tier = Tier.T1,
    deterministic: bool = True,
    r18: bool | None = None,
    r19: bool | None = None,
    r19_overrides: dict[str, int] | None = None,
    environ: dict[str, str] | None = None,
) -> GenesisSubject:
    """Create one new genesis subject.  Refuses (``GuardRefusal``) a root that
    exists or is not strictly under the explicit base, production locations,
    inherited config / ``PYTHONPATH``, forbidden ports and non-loopback hosts.
    """

    chosen = PROFILES[profile] if isinstance(profile, str) else profile
    env = dict(os.environ) if environ is None else environ
    guards.check_inherited_environment(env)
    resolved_base = guards.resolve_base(base, env)
    rid = run_id or uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9]{1,24}", rid):
        raise GuardRefusal("RUN_ID_INVALID", rid)
    run_dir = guards.check_root_location(resolved_base / rid, resolved_base)
    root = run_dir / "subject"
    use_r18 = chosen.r18 if r18 is None else r18
    use_r19 = chosen.r19 if r19 is None else r19
    port = guards.check_port(free_port())
    if tier is Tier.T2 and len(str(run_dir / "control" / "ctl.sock")) > _AF_UNIX_LIMIT:
        raise GuardRefusal("CONTROL_PATH_TOO_LONG", str(run_dir))

    run_dir.mkdir(mode=0o700, exist_ok=False)
    root.mkdir(mode=0o700, exist_ok=False)
    evidence = run_dir / "evidence"
    evidence.mkdir(mode=0o700, exist_ok=False)
    control = run_dir / "control"
    control.mkdir(mode=0o700, exist_ok=False)
    guards.write_marker(root, rid)

    raw = build_config(
        chosen, root, port, tier,
        deterministic=deterministic, r18=use_r18, r19=use_r19, run_id=rid, r19_overrides=r19_overrides,
    )
    (root / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    os.chmod(root / "config.yaml", 0o600)

    target: Path | None = None
    if use_r18:
        # Created before the first start, with its own obligation.key; the
        # initializer refuses an existing target or secret.
        target = root / str(raw["r18"]["target_path"])
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(target.parent, 0o700)
        initialize_target(target)

    subject = GenesisSubject(
        run_id=rid, profile=chosen, tier=tier, base=resolved_base, run_dir=run_dir,
        root=root, evidence_dir=evidence, control_dir=control, port=port, target_path=target,
    )
    verify_prepared(subject)
    return subject


def verify_prepared(subject: GenesisSubject) -> Settings:
    """The preconditions of a first start, re-checkable at any time."""

    relative = subject.tier is Tier.T2
    # The run directory, not the root: the root holds the subject's OWN journal
    # once it has started; an existing journal in any ancestor is a refusal.
    guards.check_production_prefix(subject.root)
    guards.check_no_foreign_journal(subject.run_dir)
    guards.check_marker(subject.root, subject.run_id)
    guards.check_no_symlinks(subject.root)
    settings = subject.resolved_settings()
    # Paths are checked as written: T1 absolute, T2 relative (cwd = root).
    written = subject.settings()
    guards.check_settings_paths(written, subject.root, relative=relative)
    guards.check_settings_network(written)
    if written.api.port != subject.port:
        raise GuardRefusal("PORT_MISMATCH", str(written.api.port))
    if not written.subject_history.enabled:
        raise GuardRefusal("HISTORY_NOT_ENABLED")
    if written.subject_history.plan_hold_receipts != subject.profile.plan_hold_receipts:
        raise GuardRefusal("PLAN_HOLD_RECEIPTS_MISMATCH")
    if written.r18.approval_origins != [f"http://localhost:{subject.port}"]:
        raise GuardRefusal("APPROVAL_ORIGIN_MISMATCH")
    if written.r18.enabled:
        target = settings.r18.target_path
        if not target.exists() or not (target.parent / SECRET_FILE).exists():
            raise GuardRefusal("TARGET_MISSING")
    return settings


# -------------------------------------------------------- first-start check


def read_history_meta(path: Path) -> dict[str, str | None]:
    rows = short_read(
        path,
        "SELECT phase, subject_id, store_id, journal_lineage_id, adopted_baseline_id, "
        "genesis_incarnation_id FROM history_meta WHERE id = 1",
    )
    if len(rows) != 1:
        raise GuardRefusal("HISTORY_META_ABSENT")
    keys = ("phase", "subject_id", "store_id", "journal_lineage_id", "adopted_baseline_id", "genesis_incarnation_id")
    return {key: None if value is None else str(value) for key, value in zip(keys, rows[0], strict=True)}


def read_target_instance(path: Path) -> str:
    rows = short_read(path, "SELECT target_instance_id FROM target_meta")
    if len(rows) != 1:
        raise GuardRefusal("TARGET_META_ABSENT")
    return str(rows[0][0])


def verify_first_start(
    subject: GenesisSubject, *, provider: dict[str, object] | None = None
) -> IdentityRecord:
    """After the first start: the baseline exists, the store is ACTIVE and the
    genesis incarnation id is recorded (D1 sec. 3-2 step 5; fact 3)."""

    settings = subject.resolved_settings()
    meta = read_history_meta(settings.subject_history.path)
    if meta["phase"] != "ACTIVE" or not meta["adopted_baseline_id"] or not meta["genesis_incarnation_id"]:
        raise GuardRefusal("ADOPTION_NOT_ACTIVE", str(meta["phase"]))
    assert meta["journal_lineage_id"] and meta["store_id"] and meta["subject_id"]
    target_id: str | None = None
    key_digest: str | None = None
    if settings.r18.enabled:
        target_id = read_target_instance(settings.r18.target_path)
        key_digest = hashlib.sha256((settings.r18.target_path.parent / SECRET_FILE).read_bytes()).hexdigest()
    return IdentityRecord(
        run_id=subject.run_id,
        profile=subject.profile.name,
        tier=subject.tier.value,
        port=subject.port,
        journal_lineage_id=meta["journal_lineage_id"],
        genesis_incarnation_id=meta["genesis_incarnation_id"],
        history_store_id=meta["store_id"],
        history_subject_id=meta["subject_id"],
        target_instance_id=target_id,
        obligation_key_sha256=key_digest,
        config_sha256=hashlib.sha256(subject.config_path.read_bytes()).hexdigest(),
        git_head=git_head(),
        provider=dict(provider or {}),
    )


def write_identity_record(subject: GenesisSubject, record: IdentityRecord) -> Path:
    """``evidence/subject.json``, written once (exclusive create)."""

    if subject.evidence_dir == subject.root or subject.root in subject.evidence_dir.parents:
        raise GuardRefusal("EVIDENCE_INSIDE_SUBJECT")
    with open(subject.subject_json, "x", encoding="utf-8") as handle:
        json.dump(asdict(record), handle, sort_keys=True, indent=1)
    return subject.subject_json


def read_identity_record(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GuardRefusal("SUBJECT_RECORD_INVALID")
    return data


# --------------------------------------------------------------- uniqueness


def assert_pairwise_distinct(records: list[IdentityRecord]) -> None:
    """Every id of sec. 3-1 is distinct across subjects (a missing one, e.g.
    no target on an r18-off subject, is skipped)."""

    for index in range(5):
        values = [item.id_tuple()[index] for item in records if item.id_tuple()[index] is not None]
        if len(values) != len(set(values)):
            raise GuardRefusal("IDENTITY_COLLISION", ("lineage", "incarnation", "store", "target", "obligation")[index])
    subjects = [item.history_subject_id for item in records]
    if len(subjects) != len(set(subjects)):
        raise GuardRefusal("IDENTITY_COLLISION", "subject")
    ports = [item.port for item in records]
    if len(ports) != len(set(ports)):
        raise GuardRefusal("IDENTITY_COLLISION", "port")


def assert_disjoint_roots(subjects: list[GenesisSubject]) -> None:
    """No shared path prefix and no shared inode between two roots."""

    roots = [Path(os.path.realpath(item.root)) for item in subjects]
    for i, first in enumerate(roots):
        for second in roots[i + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise GuardRefusal("ROOTS_NOT_DISJOINT", str(first))
    seen: dict[tuple[int, int], Path] = {}
    for root in roots:
        for current, _directories, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(current) / name
                stat = path.lstat()
                key = (stat.st_dev, stat.st_ino)
                if key in seen and not _same_root(seen[key], path, roots):
                    raise GuardRefusal("SHARED_INODE", name)
                seen[key] = path


def _same_root(first: Path, second: Path, roots: list[Path]) -> bool:
    def owner(path: Path) -> Path | None:
        return next((root for root in roots if root in path.parents), None)

    return owner(first) == owner(second)


__all__ = [
    "PROFILES",
    "GenesisSubject",
    "IdentityRecord",
    "Profile",
    "Tier",
    "assert_disjoint_roots",
    "assert_pairwise_distinct",
    "build_config",
    "create_subject",
    "free_port",
    "git_head",
    "read_history_meta",
    "read_identity_record",
    "read_target_instance",
    "verify_first_start",
    "verify_prepared",
    "write_identity_record",
]
