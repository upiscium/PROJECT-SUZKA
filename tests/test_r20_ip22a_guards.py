"""IP-22a: the isolation guards, provider modes and the architecture limits
(D1 sec. 3-4, 3-5, sec. 11).  Every refusal has a test."""

from __future__ import annotations

import uuid
import ast
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import yaml

from experiments.r20 import factory, guards, provider, serve
from experiments.r20.factory import PROFILES, Tier, create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.provider import ApprovalActor, ProviderMode, check_modes

ROOT = Path(__file__).resolve().parents[1]


def refusal_code(call, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
    with pytest.raises(GuardRefusal) as caught:
        call(*args, **kwargs)
    return caught.value.code


# ------------------------------------------------------------- the 6 refusals


def test_refuses_an_existing_root(r20_base: Path) -> None:
    create_subject("NEG-plain", base=r20_base, run_id="aa11")
    assert refusal_code(create_subject, "NEG-plain", base=r20_base, run_id="aa11") == "ROOT_EXISTS"


def test_refuses_a_root_outside_the_base(r20_base: Path, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere" / "run"
    assert refusal_code(guards.check_root_location, other, r20_base) == "ROOT_OUTSIDE_BASE"
    # ``..`` cannot climb out of the base either.
    assert refusal_code(guards.check_root_location, r20_base / ".." / f"outside-{uuid.uuid4().hex}", r20_base) == "ROOT_OUTSIDE_BASE"


@pytest.mark.parametrize("path", ["/opt/project-suzka/x", "/var/lib/suzka/run", "/etc/suzka/sub"])
def test_refuses_production_locations(path: str) -> None:
    assert refusal_code(guards.check_not_production, Path(path)) == "PRODUCTION_PATH"


@pytest.mark.parametrize("path", ["/var/lib/suzka/lab", "/etc/r20-base", "/opt/project-suzka/lab"])
def test_refuses_a_base_inside_production_before_creating_anything(path: str) -> None:
    assert refusal_code(guards.resolve_base, Path(path)) == "PRODUCTION_PATH"
    assert not os.path.lexists(path)
    assert refusal_code(guards.resolve_base, Path("relative/base")) == "BASE_NOT_ABSOLUTE"


def test_refuses_a_location_holding_an_existing_journal(tmp_path: Path) -> None:
    existing = tmp_path / "someone" / ".suzka"
    existing.mkdir(parents=True)
    (existing / "event_journal.jsonl").write_text("")
    assert refusal_code(guards.check_not_production, tmp_path / "someone" / "new") == "EXISTING_JOURNAL"
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "event_journal.jsonl").write_text("")
    assert refusal_code(guards.check_not_production, plain / "sub") == "EXISTING_JOURNAL"


def test_refuses_the_repository_state_directory() -> None:
    assert refusal_code(guards.check_not_production, ROOT / ".suzka" / "lab") == "REPOSITORY_STATE_PATH"


def test_refuses_forbidden_ports_and_non_loopback_hosts() -> None:
    for port in (8000, 8080, 8101):
        assert refusal_code(guards.check_port, port) == "PORT_FORBIDDEN"
    assert refusal_code(guards.check_port, 80) == "PORT_INVALID"
    assert guards.check_port(45123) == 45123
    assert refusal_code(guards.check_loopback_host, "0.0.0.0") == "HOST_NOT_LOOPBACK"
    assert refusal_code(guards.check_loopback_host, "10.12.0.2") == "HOST_NOT_LOOPBACK"
    for _ in range(5):
        assert factory.free_port() not in guards.FORBIDDEN_PORTS


def test_a_settings_naming_a_forbidden_port_or_host_is_refused(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base)
    settings = subject.settings()
    bad_port = settings.model_copy(update={"api": settings.api.model_copy(update={"port": 8080})})
    assert refusal_code(guards.check_settings_network, bad_port) == "PORT_FORBIDDEN"
    bad_host = settings.model_copy(update={"api": settings.api.model_copy(update={"host": "0.0.0.0"})})
    assert refusal_code(guards.check_settings_network, bad_host) == "HOST_NOT_LOOPBACK"
    origin = settings.r18.model_copy(update={"approval_origins": ["http://localhost:8000"]})
    bad_origin = settings.model_copy(update={"r18": origin})
    assert refusal_code(guards.check_settings_network, bad_origin) == "PORT_FORBIDDEN"


def test_refuses_an_inherited_config_path_or_pythonpath_outside_the_repository() -> None:
    assert refusal_code(guards.check_inherited_environment, {"SUZKA_CONFIG_PATH": "/etc/suzka/config.yaml"}) == "INHERITED_CONFIG_PATH"
    assert refusal_code(guards.check_inherited_environment, {"PYTHONPATH": "/opt/project-suzka"}) == "INHERITED_PYTHONPATH"
    guards.check_inherited_environment({"PYTHONPATH": str(ROOT)})
    guards.check_inherited_environment({})


def test_the_factory_refuses_an_inherited_config_path(r20_base: Path) -> None:
    assert (
        refusal_code(create_subject, "NEG-plain", base=r20_base, environ={"SUZKA_CONFIG_PATH": "/x/config.yaml"})
        == "INHERITED_CONFIG_PATH"
    )
    assert list(r20_base.iterdir()) == []  # refused before anything was created


def test_refuses_a_symlink_anywhere_in_the_root(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base)
    os.symlink("/etc", subject.root / "link")
    assert refusal_code(factory.verify_prepared, subject) == "SYMLINK_IN_ROOT"


def test_refuses_a_settings_path_that_leaves_the_root(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base)
    settings = subject.settings()
    for bad in (Path("~/x"), Path("/etc/passwd"), subject.root.parent / "other" / "j.jsonl"):
        moved = settings.model_copy(
            update={"agent_state": settings.agent_state.model_copy(update={"path": bad})}
        )
        assert refusal_code(guards.check_settings_paths, moved, subject.root, relative=False) == "SETTINGS_PATH"
    escaped = settings.model_copy(
        update={"agent_state": settings.agent_state.model_copy(update={"path": subject.root / ".." / "x.json"})}
    )
    assert guards.settings_path_problems(escaped, subject.root, relative=False)


# ---------------------------------------------------------------- the marker


def test_the_wrapper_refuses_a_root_without_the_marker_or_with_another_run(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    args = serve.parse([str(subject.root), "--run-id", subject.run_id])
    serve.prepare(args)  # the happy path
    (subject.root / guards.MARKER_NAME).unlink()
    assert refusal_code(serve.prepare, args) == "MARKER_MISSING"
    guards.write_marker(subject.root, "someotherrun")
    assert refusal_code(serve.prepare, args) == "MARKER_MISMATCH"
    # exit status 2, nothing started
    assert serve.main([str(subject.root), "--run-id", subject.run_id]) == 2


def test_the_wrapper_refuses_a_production_looking_root() -> None:
    args = serve.parse(["/var/lib/suzka", "--run-id", "x"])
    assert refusal_code(serve.prepare, args) == "PRODUCTION_PATH"


# -------------------------------------------------------------- provider modes


def test_scripted_approval_is_refused_with_a_real_provider() -> None:
    assert (
        refusal_code(check_modes, ProviderMode.REAL, ApprovalActor.SCRIPTED, human_launch=True)
        == "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER"
    )


def test_a_real_provider_is_refused_without_the_humans_own_launch() -> None:
    assert (
        refusal_code(check_modes, ProviderMode.REAL, ApprovalActor.HUMAN, human_launch=False)
        == "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH"
    )
    check_modes(ProviderMode.SCRIPTED, ApprovalActor.SCRIPTED, human_launch=False)
    check_modes(ProviderMode.DUMMY, ApprovalActor.SCRIPTED, human_launch=False)


def test_serve_refuses_a_real_provider_launched_by_a_process_not_a_human(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    # the flag alone is not enough: the launching process has no terminal
    for extra in ([], ["--human-launch"]):
        args = serve.parse([str(subject.root), "--run-id", subject.run_id, "--provider", "real", "--approval-actor", "HUMAN", *extra])
        if not sys.stdin.isatty():
            assert refusal_code(serve.prepare, args) == "REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH"
    args = serve.parse([str(subject.root), "--run-id", subject.run_id, "--provider", "real"])
    assert refusal_code(serve.prepare, args) == "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER"


def test_the_scripted_provider_is_non_neutral_and_digest_stable() -> None:
    first = provider.ScriptedProvider(provider.r15_protocol_script())
    second = provider.ScriptedProvider(provider.r15_protocol_script())
    assert first.identity == second.identity and first.identity["fixture_only"] is True
    assert first.calculate_loss("c", "t") == provider.NON_NEUTRAL_LOSS
    assert [first.generate("p") for _ in range(3)][2].startswith("{")  # the S1 JSON at call 2
    assert provider.ScriptedProvider(["x"]).identity != first.identity


# ---------------------------------------------------------------- profiles


@pytest.mark.parametrize(
    ("name", "receipts", "r18", "r19"),
    [("S0", True, True, True), ("F1", True, False, False), ("S1", False, False, True),
     ("NEG-plain", False, False, False), ("NEG-r15", False, False, False)],
)
def test_profiles_set_plan_hold_receipts_and_the_r18_r19_sections(
    r20_base: Path, name: str, receipts: bool, r18: bool, r19: bool
) -> None:
    subject = create_subject(name, base=r20_base)
    settings = subject.settings()
    assert settings.subject_history.plan_hold_receipts is receipts
    assert settings.subject_history.enabled is True
    assert settings.r18.enabled is r18 and settings.r19.enabled is r19
    assert settings.r18.approval_origins == [f"http://localhost:{subject.port}"]
    assert settings.r18.approval_trusted_networks == []
    assert (subject.target_path is not None) is r18
    assert settings.api.host == "127.0.0.1" and settings.api.port == subject.port
    if r18:
        assert subject.target_path is not None and subject.target_path.exists()
        assert (subject.target_path.parent / "obligation.key").exists()
        assert oct(subject.target_path.parent.stat().st_mode & 0o777) == "0o700"
    else:
        assert not (subject.root / ".suzka" / "private" / "r18").exists()
    assert settings.emotion.timer_enabled is False and settings.emotion.appraisal_response_rate == 1.0
    assert PROFILES[name].self_goal is (name == "S1")  # HD-22-4 (a)


def test_a_neg_profile_can_turn_receipts_off_and_r18_on_per_case(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, r18=True)
    assert subject.settings().r18.enabled and subject.target_path is not None


def test_the_evidence_directory_is_a_sibling_of_the_subject_root(r20_base: Path) -> None:
    subject = create_subject("S0", base=r20_base)
    assert subject.root.parent == subject.evidence_dir.parent == subject.control_dir.parent
    assert subject.root not in subject.evidence_dir.parents and subject.evidence_dir != subject.root
    assert (subject.root / guards.MARKER_NAME).is_file()
    assert refusal_code(factory.write_identity_record, _inside(subject), None) == "EVIDENCE_INSIDE_SUBJECT"


def _inside(subject: factory.GenesisSubject) -> factory.GenesisSubject:
    import dataclasses

    return dataclasses.replace(subject, evidence_dir=subject.root / "evidence")


# ------------------------------------------------------------ path rules


def test_t1_writes_absolute_paths_and_t2_relative_with_cwd_equal_root(r20_base: Path) -> None:
    one = create_subject("NEG-plain", base=r20_base, tier=Tier.T1)
    two = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    for where, path in guards._paths(one.settings(), "settings"):
        assert path.is_absolute() and path.is_relative_to(one.root), where
    for where, path in guards._paths(two.settings(), "settings"):
        assert not path.is_absolute() and not str(path).startswith(".."), where
    assert not guards.settings_path_problems(two.settings(), two.root, relative=True)
    assert guards.settings_path_problems(two.settings(), two.root, relative=False)  # T2 is not absolute
    assert guards.settings_path_problems(one.settings(), one.root, relative=True)  # T1 is not relative
    absolute = two.resolved_settings()
    assert all(path.is_absolute() and path.is_relative_to(two.root) for _w, path in guards._paths(absolute, "s"))


def _lab_prepare():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("r20_lab_prepare", ROOT / "scripts" / "lab_prepare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mutation",
    [None, ("event_journal", "path", "../escape.jsonl"), ("agent_state", "path", "/tmp/abs.json"),
     ("agent_state", "path", "~/state.json"), ("state_wal", "directory", ".suzka/../../out")],
)
def test_the_path_check_agrees_with_lab_prepare_on_samples(
    r20_base: Path, mutation: tuple[str, str, str] | None, capsys: pytest.CaptureFixture[str]
) -> None:
    lab_prepare = _lab_prepare()
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    if mutation is not None:
        raw[mutation[0]][mutation[1]] = mutation[2]
    prod = r20_base / "prod.yaml"
    prod.write_text(yaml.safe_dump(raw))
    lab = r20_base / "lab"
    lab.mkdir()
    lab_ok = lab_prepare.main(str(prod), str(lab), "45678") == 0
    capsys.readouterr()
    from suzka.config.settings import load_settings

    mine = not guards.settings_path_problems(load_settings(lab / "config.yaml"), lab.resolve(), relative=True)
    assert lab_ok is mine is (mutation is None)


# ----------------------------------------------------------- architecture


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_nothing_under_suzka_imports_the_harness() -> None:
    for path in (ROOT / "suzka").rglob("*.py"):
        text = path.read_text()
        assert not any(name.split(".")[0] == "experiments" for name in _imports(path)), path
        assert "experiments.r20" not in text, path


def test_the_harness_never_calls_the_approval_owner_and_stays_outside_suzka_state() -> None:
    forbidden = {"confirm", "claim", "issue", "_submit_challenge", "_submit_claim"}
    for path in (ROOT / "experiments" / "r20").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"approval_owner"}:
                # naming the owner is only allowed to set its clock (tier1.FakeClock)
                assert path.name == "tier1.py", path
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr in forbidden
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "approval_owner"
            for node in ast.walk(tree)
        ), path
    # the stand-in imports no R18 contract: its recomputation is independent
    assert not any(name.startswith("suzka.r18") for name in _imports(ROOT / "experiments" / "r20" / "approval.py"))


def test_the_operators_own_state_directory_is_refused_even_without_a_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert refusal_code(guards.check_production_prefix, tmp_path / ".suzka" / "lab") == "HOME_STATE_PATH"
    assert refusal_code(guards.resolve_base, tmp_path / ".suzka" / "base") == "HOME_STATE_PATH"
    assert not (tmp_path / ".suzka").exists()


def test_an_unparsable_approval_origin_port_is_a_guard_refusal(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base)
    settings = subject.settings()
    bad = settings.model_copy(update={"r18": settings.r18.model_copy(update={"approval_origins": ["http://localhost:99999"]})})
    assert refusal_code(guards.check_settings_network, bad) == "PORT_INVALID"
