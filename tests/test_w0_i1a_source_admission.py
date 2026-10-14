"""W0-I1a read-only source admission tests; all fixture files under tmp_path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest
import yaml

from suzka.config.source_admission import (
    AdmissionReason,
    AdmittedSourceSelection,
    ExpectedInput,
    SourceAdmissionDenied,
    TrustedAdmissionPolicy,
    TrustedDirectory,
    admit_source_configuration,
)
import suzka.config.source_admission as admission


SOURCE_NAMES = (
    ("memory", "persist_directory"),
    ("sleep", "dream_dataset_path"),
    ("qlora", "output_dir"),
    ("adapter_registry", "path"),
    ("adapter_registry", "eval_result_dir"),
    ("agent_state", "path"),
    ("event_journal", "path"),
    ("state_wal", "directory"),
)
REPO = Path(__file__).resolve().parents[1]


def _trusted_dir(path: Path) -> TrustedDirectory:
    s = path.stat()
    return TrustedDirectory(
        path, s.st_dev, s.st_ino, s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)
    )


def _hash(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _file_identity(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_dev, st.st_ino


@pytest.fixture
def source_case(tmp_path: Path) -> tuple[TrustedAdmissionPolicy, dict]:
    config_root = tmp_path / "private_config"
    eval_root = tmp_path / "private_eval"
    state_root = tmp_path / "private_state"
    for directory in (config_root, eval_root, state_root):
        directory.mkdir(mode=0o700)

    config = yaml.safe_load(
        (REPO / "deploy/config/config.fhs.yaml.example").read_text(encoding="utf8")
    )
    for section, name in SOURCE_NAMES:
        old = config[section][name]
        assert str(old).startswith("/var/lib/project-suzka/")
        config[section][name] = str(old).replace(
            "/var/lib/project-suzka", str(state_root), 1
        )
    eval_file = eval_root / "required.json"
    eval_file.write_text(
        json.dumps({"cases": [{"prompt": "synthetic", "expected": "accepted"}]}),
        encoding="utf8",
    )
    eval_file.chmod(0o600)
    config["adapter_registry"]["eval_sets"] = [str(eval_file)]
    config_path = config_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    config_path.chmod(0o600)
    cdev, cino = _file_identity(config_path)
    edev, eino = _file_identity(eval_file)
    policy = TrustedAdmissionPolicy(
        config_root=_trusted_dir(config_root),
        state_root=_trusted_dir(state_root),
        eval_root=_trusted_dir(eval_root),
        config_path=config_path,
        config_device=cdev,
        config_inode=cino,
        config_digest=_hash(config_path.read_bytes()),
        config_uid=os.getuid(),
        config_gid=os.getgid(),
        config_mode=0o600,
        eval_uid=os.getuid(),
        eval_gid=os.getgid(),
        eval_mode=0o600,
        expected_source_paths=tuple(
            (f"{section}.{name}", Path(config[section][name]))
            for section, name in SOURCE_NAMES
        ),
        expected_eval_inputs=(
            ExpectedInput(eval_file, edev, eino, _hash(eval_file.read_bytes())),
        ),
        trusted_ancestor_uids=(0, os.getuid()),
        allowed_sticky_ancestors=(Path("/tmp"),),
        require_single_link=True,
        forbid_mount_crossing=True,
        # TEST-FIXTURE LIMITS ONLY, NOT DEPLOYMENT POLICY.
        max_config_bytes=16384,
        max_yaml_nodes=4096,
        max_yaml_depth=50,
        max_path_components=40,
        max_eval_input_bytes=4096,
        max_eval_cases=16,
        max_metadata_visits=512,
    )
    return policy, config


def _rebase_config(
    policy: TrustedAdmissionPolicy, config: dict
) -> TrustedAdmissionPolicy:
    policy.config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    s = policy.config_path.stat()
    return replace(
        policy,
        config_device=s.st_dev,
        config_inode=s.st_ino,
        config_digest=_hash(policy.config_path.read_bytes()),
    )


def _rebase_eval(
    policy: TrustedAdmissionPolicy, contents: str
) -> TrustedAdmissionPolicy:
    path = policy.expected_eval_inputs[0].path
    path.write_text(contents, encoding="utf8")
    s = path.stat()
    return replace(
        policy,
        expected_eval_inputs=(
            ExpectedInput(path, s.st_dev, s.st_ino, _hash(path.read_bytes())),
        ),
    )


def _denied(policy: TrustedAdmissionPolicy | None, reason: AdmissionReason) -> None:
    with pytest.raises(SourceAdmissionDenied) as caught:
        admit_source_configuration(policy)
    assert caught.value.reason is reason
    assert "synthetic" not in str(caught.value)


def test_absolute_source_paths_are_cwd_independent_and_pending_i2(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _config = source_case
    results: list[AdmittedSourceSelection] = []
    for name in ("elsewhere-a", "elsewhere-b"):
        cwd = tmp_path / name
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        results.append(admit_source_configuration(policy))
        assert not (cwd / ".suzka").exists()
    assert results[0] == results[1]
    result = results[0]
    assert result.source_start_status == "SOURCE_START_BLOCKED_PENDING_I2"
    assert len(result.source_paths) == 8
    assert result.evaluation_inputs[0].cases == (("synthetic", "accepted"),)
    assert dict(result.derived_paths)["experience"] == str(
        policy.state_root.path / "experience"
    )
    assert dict(result.derived_paths)["semantic"] == str(
        policy.state_root.path / "semantic"
    )
    assert dict(result.derived_paths)["episodic_pending"] == str(
        policy.state_root.path / "chroma" / ".r07-episodic-pending"
    )
    assert not (policy.state_root.path / "journal").exists()
    with pytest.raises(FrozenInstanceError):
        result.source_start_status = "CONTINUATION"  # type: ignore[misc]


def test_original_mutable_settings_do_not_alias_snapshot(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, raw = source_case
    admitted = admit_source_configuration(policy)
    before = admitted.source_paths
    raw["memory"]["persist_directory"] = "/tmp/other"
    raw["adapter_registry"]["eval_sets"][0] = "/tmp/changed"
    assert admitted.source_paths == before
    assert admitted.evaluation_inputs[0].cases == (("synthetic", "accepted"),)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        admitted.evaluation_inputs[0].cases += (("bad", "bad"),)  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    [
        "max_config_bytes",
        "max_yaml_nodes",
        "max_yaml_depth",
        "max_path_components",
        "max_eval_input_bytes",
        "max_eval_cases",
        "max_metadata_visits",
    ],
)
def test_policy_never_accepts_missing_limits(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    field: str,
) -> None:
    policy, _ = source_case
    _denied(replace(policy, **{field: 0}), AdmissionReason.POLICY_INVALID)
    _denied(replace(policy, **{field: True}), AdmissionReason.POLICY_INVALID)


def test_missing_policy_and_rule_values(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    _denied(None, AdmissionReason.POLICY_INVALID)
    _denied(replace(policy, require_single_link=None), AdmissionReason.POLICY_INVALID)  # type: ignore[arg-type]
    _denied(replace(policy, forbid_mount_crossing=None), AdmissionReason.POLICY_INVALID)  # type: ignore[arg-type]


def test_missing_config_file_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    policy.config_path.unlink()
    _denied(policy, AdmissionReason.CUSTODY_UNAVAILABLE)


def test_removed_eval_file_denied_even_after_prior_admission(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    assert admit_source_configuration(policy).evaluation_inputs
    policy.expected_eval_inputs[0].path.unlink()
    _denied(policy, AdmissionReason.CUSTODY_UNAVAILABLE)


def test_config_leaf_symlink_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    original = policy.config_path.with_suffix(".actual")
    policy.config_path.rename(original)
    policy.config_path.symlink_to(original)
    _denied(policy, AdmissionReason.CUSTODY_UNAVAILABLE)


def test_eval_leaf_symlink_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    path = policy.expected_eval_inputs[0].path
    other = path.with_suffix(".actual")
    path.rename(other)
    path.symlink_to(other)
    _denied(policy, AdmissionReason.CUSTODY_UNAVAILABLE)


def test_config_mode_and_owner_mismatch_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    policy.config_path.chmod(0o666)
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)
    policy.config_path.chmod(0o600)
    _denied(
        replace(policy, config_uid=os.getuid() + 1), AdmissionReason.IDENTITY_MISMATCH
    )


def test_hardlinked_config_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    os.link(policy.config_path, policy.config_path.with_suffix(".extra"))
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)


def test_replaced_root_inode_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    root = policy.state_root.path
    root.rename(root.with_name("former_root"))
    root.mkdir(mode=0o700)
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)


def test_config_rename_between_fd_read_and_name_check_fails(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = source_case
    original = admission._read_bounded
    swapped = False

    def read_then_swap(fd: int, limit: int) -> bytes:
        nonlocal swapped
        contents = original(fd, limit)
        if not swapped:
            swapped = True
            replacement = policy.config_path.with_suffix(".replacement")
            replacement.write_bytes(contents)
            replacement.chmod(0o600)
            os.replace(replacement, policy.config_path)
        return contents

    monkeypatch.setattr(admission, "_read_bounded", read_then_swap)
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)
    assert swapped


def test_symlinked_config_ancestor_fails(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    nested = policy.config_root.path / "actual"
    nested.mkdir(mode=0o700)
    moved = nested / "config.yaml"
    policy.config_path.rename(moved)
    alias = policy.config_root.path / "alias"
    alias.symlink_to(nested, target_is_directory=True)
    changed = replace(policy, config_path=alias / moved.name)
    _denied(changed, AdmissionReason.CUSTODY_UNAVAILABLE)


def test_world_writable_nested_config_parent_fails(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    nested = policy.config_root.path / "nested"
    nested.mkdir(mode=0o700)
    moved = nested / "config.yaml"
    policy.config_path.rename(moved)
    nested.chmod(0o777)
    _denied(replace(policy, config_path=moved), AdmissionReason.IDENTITY_MISMATCH)


def test_missing_state_root_does_not_bootstrap(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    state = policy.state_root.path
    state.rmdir()
    _denied(policy, AdmissionReason.CUSTODY_UNAVAILABLE)
    assert not state.exists()


@pytest.mark.parametrize(
    "field, bound",
    [
        ("max_config_bytes", 8),
        ("max_yaml_nodes", 2),
        ("max_yaml_depth", 1),
        ("max_metadata_visits", 2),
    ],
)
def test_config_and_metadata_resource_excesses_are_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    field: str,
    bound: int,
) -> None:
    policy, _ = source_case
    _denied(replace(policy, **{field: bound}), AdmissionReason.RESOURCE_EXCEEDED)


def test_yaml_alias_or_duplicate_key_is_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    old = policy.config_path.read_text()
    for extra in ("\nfoo: &anchor 3\nbar: *anchor\n", "\nproject: {}\n"):
        changed = policy.config_path
        changed.write_text(old + extra)
        s = changed.stat()
        admitted_policy = replace(
            policy,
            config_device=s.st_dev,
            config_inode=s.st_ino,
            config_digest=_hash(changed.read_bytes()),
        )
        _denied(admitted_policy, AdmissionReason.CONFIG_INVALID)


@pytest.mark.parametrize(
    "replacement",
    [
        ".suzka/relative.jsonl",
        "/var/lib/../external/out.jsonl",
        "/tmp/foreign-source.jsonl",
        "/var//lib//project-suzka/file.jsonl",
    ],
)
def test_unadmitted_or_ambiguous_journal_path_is_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    replacement: str,
) -> None:
    policy, config = source_case
    config["event_journal"]["path"] = replacement
    rebased = _rebase_config(policy, config)
    with pytest.raises(SourceAdmissionDenied) as denied:
        admit_source_configuration(rebased)
    assert denied.value.reason in (
        AdmissionReason.PATH_UNSUPPORTED,
        AdmissionReason.SOURCE_MISMATCH,
    )


def test_source_path_collision_is_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, config = source_case
    config["event_journal"]["path"] = config["agent_state"]["path"]
    changed = _rebase_config(policy, config)
    _denied(changed, AdmissionReason.SOURCE_MISMATCH)
    pairs = list(policy.expected_source_paths)
    pairs[-1] = (pairs[-1][0], pairs[0][1])
    _denied(
        replace(policy, expected_source_paths=tuple(pairs)),
        AdmissionReason.POLICY_INVALID,
    )


def test_unadmitted_eval_source_path_is_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, config = source_case
    config["adapter_registry"]["eval_sets"] = ["/tmp/unregistered.json"]
    _denied(_rebase_config(policy, config), AdmissionReason.EVAL_UNAVAILABLE)


@pytest.mark.parametrize(
    "document",
    [
        '{"cases":[]}',
        '{"cases":[{"prompt":"","expected":"X"}]}',
        '{"cases":[{"prompt":"X","expected":4}]}',
        '{"cases":{}}',
        '{"cases": [1,2]}',
        '{"cases":[{"prompt":"ok","expected":"x"}],"cases":[]}',
        "not json",
    ],
)
def test_empty_or_malformed_evaluation_input_is_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    document: str,
) -> None:
    policy, _ = source_case
    changed = _rebase_eval(policy, document)
    _denied(changed, AdmissionReason.EVAL_UNAVAILABLE)


def test_eval_source_limit_and_content_identity_are_checked(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    _denied(replace(policy, max_eval_input_bytes=3), AdmissionReason.RESOURCE_EXCEEDED)
    _denied(replace(policy, max_eval_cases=0), AdmissionReason.POLICY_INVALID)
    _denied(
        replace(policy, max_eval_cases=1, expected_eval_inputs=()),
        AdmissionReason.POLICY_INVALID,
    )
    p = policy.expected_eval_inputs[0].path
    p.write_text(json.dumps({"cases": [{"prompt": "modified", "expected": "x"}]}))
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)


def test_eval_leaf_replacement_same_bytes_has_new_inode(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    target = policy.expected_eval_inputs[0].path
    replacement = target.with_name("replacement.json")
    replacement.write_bytes(target.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, target)
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)


def test_group_writable_path_ancestor_denied_without_chmod(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    tmp_path: Path,
) -> None:
    policy, _ = source_case
    original = stat.S_IMODE(tmp_path.stat().st_mode)
    tmp_path.chmod(0o770)
    try:
        _denied(policy, AdmissionReason.IDENTITY_MISMATCH)
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o770
    finally:
        tmp_path.chmod(original)


def test_untrusted_ancestor_uid_policy_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    disallowed = (os.getuid() + 9999,)
    _denied(
        replace(policy, trusted_ancestor_uids=disallowed),
        AdmissionReason.IDENTITY_MISMATCH,
    )
    _denied(replace(policy, trusted_ancestor_uids=()), AdmissionReason.POLICY_INVALID)
    _denied(
        replace(policy, allowed_sticky_ancestors=None), AdmissionReason.POLICY_INVALID
    )  # type: ignore[arg-type]


def test_symlinked_trusted_root_and_ancestor_denied(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    path = policy.config_root.path
    moved = path.with_name("renamed_config_root")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=True)
    _denied(policy, AdmissionReason.CUSTODY_UNAVAILABLE)


def test_no_env_fallback_or_default_configuration_selection(
    source_case: tuple[TrustedAdmissionPolicy, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = source_case
    monkeypatch.setenv("SUZKA_CONFIG_PATH", "/tmp/untrusted.yaml")
    snapshot = admit_source_configuration(policy)
    assert snapshot.config.path == str(policy.config_path)
    assert snapshot.source_start_status.endswith("PENDING_I2")


def test_eval_case_count_rejected_before_any_output(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    changed = _rebase_eval(
        policy,
        json.dumps(
            {"cases": [{"prompt": f"case-{i}", "expected": "X"} for i in range(4)]}
        ),
    )
    _denied(replace(changed, max_eval_cases=2), AdmissionReason.EVAL_UNAVAILABLE)
    assert not (policy.state_root.path / "eval_results").exists()
    assert not (policy.state_root.path / "journal").exists()


def test_read_only_failure_keeps_empty_state_root_empty(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, config = source_case
    root = policy.state_root.path
    assert tuple(root.iterdir()) == ()
    config["event_journal"]["path"] = "/tmp/forbidden-journal.jsonl"
    _denied(_rebase_config(policy, config), AdmissionReason.SOURCE_MISMATCH)
    assert tuple(root.iterdir()) == ()
    assert stat.S_IMODE(root.stat().st_mode) == policy.state_root.mode


def test_eval_case_budget_is_aggregate_across_files(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, config = source_case
    existing = policy.expected_eval_inputs[0]
    second = policy.eval_root.path / "second.json"
    second.write_text(
        json.dumps(
            {
                "cases": [
                    {"prompt": "s2", "expected": "a"},
                    {"prompt": "s3", "expected": "b"},
                ]
            }
        )
    )
    second.chmod(0o600)
    st = second.stat()
    additional = ExpectedInput(second, st.st_dev, st.st_ino, _hash(second.read_bytes()))
    config["adapter_registry"]["eval_sets"] = [str(existing.path), str(second)]
    policy = _rebase_config(policy, config)
    policy = replace(
        policy,
        expected_eval_inputs=(existing, additional),
        max_eval_cases=2,
    )
    _denied(policy, AdmissionReason.RESOURCE_EXCEEDED)
    assert not (policy.state_root.path / "eval_results").exists()


def test_fifo_config_input_denies_without_blocking(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    assert admission._FILE_FLAGS & os.O_NONBLOCK
    policy.config_path.unlink()
    os.mkfifo(policy.config_path, mode=0o600)
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)


def test_fifo_evaluation_input_denies_without_blocking(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    policy, _ = source_case
    assert admission._FILE_FLAGS & os.O_NONBLOCK
    candidate = policy.expected_eval_inputs[0].path
    candidate.unlink()
    os.mkfifo(candidate, mode=0o600)
    _denied(policy, AdmissionReason.IDENTITY_MISMATCH)


def test_verified_original_config_bytes_survive_file_disappearance(
    source_case: tuple[TrustedAdmissionPolicy, dict],
) -> None:
    """I1b can consume exactly the admitted bytes without reopening a path."""
    from suzka.config.schema import Settings

    policy, original_dict = source_case
    admitted = admit_source_configuration(policy)
    verified = admitted._validated_config_bytes
    assert type(verified) is bytes
    assert hashlib.sha256(verified).hexdigest() == admitted.config.digest
    assert "_validated_config_bytes=" not in repr(admitted)

    original_model = Settings.model_validate(original_dict)
    # All effective service settings are recoverable, not just nine paths:
    # model, generation, emotion, adapter thresholds, etc.
    reconstructed = Settings.model_validate(yaml.safe_load(verified))
    assert reconstructed == original_model

    # An external rename/removal after I1a cannot change the immutable bytes
    # already observed and verified by the file-descriptor admission.
    policy.config_path.unlink()
    assert not policy.config_path.exists()
    assert Settings.model_validate(
        yaml.safe_load(admitted._validated_config_bytes)
    ) == (original_model)
    with pytest.raises(FrozenInstanceError):
        admitted._validated_config_bytes = b"changed"  # type: ignore[misc]
