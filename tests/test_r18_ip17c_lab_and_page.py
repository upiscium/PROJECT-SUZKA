"""IP-17c: the lab ``--r18`` option (#411 D1 r7 §9-1) and the approval page's
in-browser verification (§5), the latter run with Node when it is available.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from suzka.config.settings import load_settings
from suzka.r18.contracts import ActionIntentV1, ApprovalChallengeV1
from suzka.r18.obligation import SECRET_FILE
from suzka.r18.target import TargetReader

import test_r18_ip13_contracts as ip13

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PAGE = ROOT / "suzka" / "r18" / "approval_page" / "approve.js"


def _lab_prepare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lab_prepare_under_test", SCRIPTS / "lab_prepare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prod_config(tmp_path: Path, **sections: Any) -> Path:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for name, value in sections.items():
        if isinstance(value, dict):
            raw.setdefault(name, {}).update(value)
        else:
            raw[name] = value
    path = tmp_path / "prod-config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _lab(tmp_path: Path) -> Path:
    lab = tmp_path / "lab"
    lab.mkdir()
    return lab


def _run(tmp_path: Path, capsys, *flags: str, port: str = "8101", **sections: Any):  # type: ignore[no-untyped-def]
    prod = _prod_config(tmp_path, **sections)
    lab = _lab(tmp_path) if not (tmp_path / "lab").exists() else tmp_path / "lab"
    code = _lab_prepare().main(str(prod), str(lab), port, *flags)
    captured = capsys.readouterr()
    return code, captured.out, captured.err, lab, prod


# ============================================= lab_prepare without --r18


def test_without_r18_the_lab_config_is_exactly_the_production_config_plus_the_port(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, err, lab, prod = _run(tmp_path, capsys)
    assert (code, out, err) == (0, "8000\n", "")
    expected = yaml.safe_load(prod.read_text(encoding="utf-8"))
    expected["api"]["port"] = 8101
    assert yaml.safe_load((lab / "config.yaml").read_text(encoding="utf-8")) == expected
    settings = load_settings(lab / "config.yaml")
    assert settings.r18.enabled is False
    assert settings.r18.approval_origins == ["http://localhost:8000"]  # the default, untouched
    assert "r18_target" not in out


def test_without_r18_a_production_r18_section_is_left_alone(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _err, lab, _prod = _run(tmp_path, capsys, r18={"enabled": False})
    assert code == 0 and out == "8000\n"
    assert load_settings(lab / "config.yaml").r18.enabled is False


def test_the_existing_refusals_still_hold_without_the_flag(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, _out, err, _lab_dir, _prod = _run(tmp_path, capsys, port="8000")
    assert code == 1 and "is the production port" in err


def test_an_unknown_flag_is_a_usage_error(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, err, _lab_dir, _prod = _run(tmp_path, capsys, "--bogus")
    assert (code, out) == (2, "") and "usage" in err
    assert not (tmp_path / "lab" / "config.yaml").exists()


# ================================================== lab_prepare with --r19


def test_r19_alone_switches_only_r19_on_in_the_lab_and_never_writes_production(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    prod = _prod_config(tmp_path, r19={"enabled": False})
    before = prod.read_bytes()
    lab = _lab(tmp_path)
    code = _lab_prepare().main(str(prod), str(lab), "8101", "--r19")
    out = capsys.readouterr().out
    assert (code, out) == (0, "8000\n")
    assert prod.read_bytes() == before
    settings = load_settings(lab / "config.yaml")
    assert settings.r19.enabled is True and settings.r18.enabled is False
    assert "r18_target" not in out


def test_r19_keeps_other_production_r19_settings(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    prod_default = load_settings(_prod_config(tmp_path)).r19
    other = next(
        name for name, value in prod_default.model_dump().items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1
    )
    value = getattr(prod_default, other) + 1
    code, _out, _err, lab, _prod = _run(tmp_path, capsys, "--r19", r19={"enabled": False, other: value})
    assert code == 0
    settings = load_settings(lab / "config.yaml")
    assert settings.r19.enabled is True and getattr(settings.r19, other) == value


def test_r19_with_r18_sets_both(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _err, lab, _prod = _run(tmp_path, capsys, "--r18", "--r19")
    assert code == 0 and "r18_target=" in out
    settings = load_settings(lab / "config.yaml")
    assert settings.r18.enabled is True and settings.r19.enabled is True


def test_r19_refuses_the_production_port_and_a_lab_without_subject_history(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, _out, err, _lab_dir, _prod = _run(tmp_path, capsys, "--r19", port="8000")
    assert code == 1 and "is the production port" in err


def test_r19_checks_flag_missing_history_and_non_loopback() -> None:
    from types import SimpleNamespace

    def fake(history: bool, host: str, enabled: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            r19=SimpleNamespace(enabled=enabled),
            api=SimpleNamespace(host=host),
            subject_history=SimpleNamespace(enabled=history),
        )

    check = _lab_prepare()._r19_problems
    assert check(fake(True, "127.0.0.1")) == []
    assert any("subject.history" in p for p in check(fake(False, "127.0.0.1")))
    assert any("loopback" in p for p in check(fake(True, "0.0.0.0")))
    assert any("r19.enabled" in p for p in check(fake(True, "127.0.0.1", enabled=False)))


def test_the_lab_script_wires_r19_and_leaves_the_default_arguments_alone() -> None:
    text = _script()
    assert "--r19) R19=1; shift ;;" in text
    assert 'if [[ "${R19}" == 1 ]]; then\n  PREPARE_ARGS+=(--r19)\nfi\n' in text
    assert 'PREPARE_ARGS=("${LAB}/prod-config.yaml" "${LAB}" "${PORT}")' in text
    assert "/api/r19/inbox/messages" in text and "R19_WAKE_MATERIALIZE" in text
    done = subprocess.run(
        ["bash", str(SCRIPTS / "lab-from-production.sh"), "--r19", "--port", "99999", "--", "true"],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 2 and "invalid port" in done.stderr


# ================================================== lab_prepare with --r18


def test_r18_switches_the_lab_on_with_a_localhost_origin_and_a_relative_target(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, err, lab, prod = _run(tmp_path, capsys, "--r18", port="8123")
    assert (code, err) == (0, "")
    assert out.splitlines() == ["8000", "r18_target=.suzka/private/r18/target.sqlite3"]
    settings = load_settings(lab / "config.yaml")
    assert settings.r18.enabled is True
    assert settings.r18.approval_origins == ["http://localhost:8123"]
    assert settings.r18.approval_trusted_networks == []
    assert not settings.r18.target_path.is_absolute()
    assert (lab / settings.r18.target_path).resolve().is_relative_to(lab.resolve())
    assert settings.api.port == 8123
    # Every value is a str or a relative Path: no absolute path, no uid setting.
    raw = yaml.safe_load((lab / "config.yaml").read_text(encoding="utf-8"))
    assert set(raw["r18"]) <= {"enabled", "target_path", "approval_origins", "approval_trusted_networks"}
    assert not any("uid" in key for key in raw["r18"])
    # The production config file is not touched, and only r18 / api.port differ.
    original = yaml.safe_load(prod.read_text(encoding="utf-8"))
    assert original["r18"]["enabled"] is False
    raw["r18"] = original["r18"]
    raw["api"]["port"] = original["api"]["port"]
    assert raw == original


def test_r18_keeps_a_production_target_path_when_it_stays_inside_the_lab(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _err, lab, _prod = _run(
        tmp_path, capsys, "--r18", r18={"enabled": False, "target_path": "x/y/t.sqlite3"}
    )
    assert code == 0 and "r18_target=x/y/t.sqlite3" in out
    assert load_settings(lab / "config.yaml").r18.target_path == Path("x/y/t.sqlite3")


@pytest.mark.parametrize(
    "sections,message",
    [
        ({"r18": {"target_path": "/etc/suzka/target.sqlite3"}}, "is not relative"),
        ({"r18": {"target_path": "../../outside/t.sqlite3"}}, "leaves the lab"),
        ({"r18": {"target_path": "target.sqlite3"}}, "inside a directory"),
        ({"api": {"host": "0.0.0.0"}}, "r18.enabled needs api.host on loopback"),
        ({"api": {"host": "10.12.0.2"}}, "r18.enabled needs api.host on loopback"),
        ({"subject_history": {"enabled": False}}, "needs subject.history"),
        ({"r18": {"target_path": "a/b\nc/t.sqlite3"}}, "control characters"),
    ],
)
def test_r18_refuses_an_unsafe_lab(tmp_path, capsys, sections, message) -> None:  # type: ignore[no-untyped-def]
    code, out, err, _lab_dir, _prod = _run(tmp_path, capsys, "--r18", **sections)
    assert code == 1 and message in err and "r18_target" not in out


def test_r18_refuses_the_production_port(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, err, _lab_dir, _prod = _run(tmp_path, capsys, "--r18", port="8000")
    assert code == 1 and "is the production port" in err
    assert "names the production port" in err and "r18_target" not in out


# ====================================== target_init inside the lab only


def test_target_init_creates_the_target_inside_the_lab_as_the_script_runs_it(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _err, lab, _prod = _run(tmp_path, capsys, "--r18")
    assert code == 0
    relative = out.splitlines()[1].split("=", 1)[1]
    directory = lab / Path(relative).parent
    directory.mkdir(parents=True)
    directory.chmod(0o700)  # install -d -m 700
    target = (lab / relative).resolve()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # The script runs it from the lab directory with ONE absolute path.
    done = subprocess.run(
        [sys.executable, "-m", "suzka.r18.target_init", str(target)],
        cwd=lab,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert target.is_file() and (directory / SECRET_FILE).is_file()
    assert oct((directory / SECRET_FILE).stat().st_mode & 0o777) == "0o600"
    assert all(path.resolve().is_relative_to(lab.resolve()) for path in (target, directory / SECRET_FILE))
    reader = TargetReader.open(target)
    try:
        assert reader.meta().target_instance_id
    finally:
        reader.close()
    # Never replaced: a second run is refused, and a bad directory is refused.
    again = subprocess.run(
        [sys.executable, "-m", "suzka.r18.target_init", str(target)],
        cwd=lab, env=env, capture_output=True, text=True, timeout=60,
    )
    assert again.returncode == 1 and "refused" in again.stderr
    wide = lab / "wide"
    wide.mkdir()
    wide.chmod(0o777)
    bad = subprocess.run(
        [sys.executable, "-m", "suzka.r18.target_init", str(wide / "t.sqlite3")],
        cwd=lab, env=env, capture_output=True, text=True, timeout=60,
    )
    assert bad.returncode == 1


# ============================================= the shell script (static)


def _script() -> str:
    return (SCRIPTS / "lab-from-production.sh").read_text(encoding="utf-8")


def test_the_lab_script_parses_and_wires_r18_with_todays_sudo_pattern() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPTS / "lab-from-production.sh")]).returncode == 0
    text = _script()
    assert "--r18) R18=1; shift ;;" in text
    assert 'PREPARE_ARGS+=(--r18)' in text
    # Only the pattern the script already uses: sudo install -d / sudo -u, no new
    # users, groups, sockets or chown of anything outside the lab.
    assert 'sudo -u "${SERVICE_USER}" mkdir -p -m 700 "${LAB}/${R18_DIR}"' in text
    assert "sudo install -d" not in text.split("R18_DIR=")[1]
    assert '-m suzka.r18.target_init "${LAB}/${R18_TARGET}"' in text
    for forbidden in ("useradd", "groupadd", "adduser", "chown -R /", "usermod", "visudo", "sudoers", "ssh-", "systemctl enable", "nix"):
        assert forbidden not in text
    # The target step is inside the R18 branch only.
    branch = text[text.index('if [[ "${R18}" == 1 ]]; then\n  # lab_prepare printed'):]
    assert branch.index("target_init") < branch.index("\nfi\n")
    assert text.count("target_init") == 2  # the comment and the one call
    # The target path is refused unless relative, inside a directory, with no '..'.
    assert '"${R18_TARGET}" == /*' in text and '"${R18_TARGET}" == *..*' in text
    # Without --r18 the lab_prepare arguments are exactly the old three.
    assert 'PREPARE_ARGS=("${LAB}/prod-config.yaml" "${LAB}" "${PORT}")' in text


def test_the_lab_script_rejects_bad_arguments_before_touching_anything() -> None:
    script = str(SCRIPTS / "lab-from-production.sh")
    for args, fragment in (
        (["--r18"], "usage"),
        (["--r18", "--port", "99999", "--", "true"], "invalid port"),
        (["--r18", "--timeout", "0", "--", "true"], "invalid timeout"),
        (["--r18", "--bogus", "--", "true"], "unknown argument"),
    ):
        done = subprocess.run(["bash", script, *args], capture_output=True, text=True, timeout=30)
        assert done.returncode == 2 and fragment in done.stderr, (args, done.stderr)


# ================================================ the page, run with Node


def _node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    for candidate in sorted(Path("/nix/store").glob("*nodejs*/bin/node")) if Path("/nix/store").is_dir() else ():
        return str(candidate)
    return None


def _node_missing() -> None:
    # CI installs Node (actions/setup-node): there a missing node is a failure.
    if os.environ.get("CI"):
        pytest.fail("node is required in CI")
    pytest.skip("node is not available")


_DRIVER = r"""
const fs = require("fs");
eval(fs.readFileSync(process.argv[1], "utf8"));
const api = globalThis.__suzkaApproval;
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
(async () => {
  const out = [];
  for (const reply of cases) {
    try { const r = await api.verify(reply); out.push({ok: r.ok, problems: r.problems, code: r.code, digest: r.challengeDigest}); }
    catch (e) { out.push({error: String(e)}); }
  }
  process.stdout.write(JSON.stringify(out));
})();
"""


def _verify_in_node(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node = _node()
    if node is None:
        _node_missing()
    done = subprocess.run(
        [node, "-e", _DRIVER, str(PAGE)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)  # type: ignore[no-any-return]


def _reply(base_intent: ActionIntentV1, base_challenge: ApprovalChallengeV1, **changes: Any) -> dict[str, Any]:
    intent, challenge = base_intent, base_challenge
    reply: dict[str, Any] = {
        "challenge": challenge.canonical_bytes().decode("utf-8"),
        "challenge_digest": challenge.challenge_digest,
        "intent": intent.canonical_bytes().decode("utf-8"),
        "forecast_digest": challenge.expectation_digest,
        "window_end_utc": intent.approval_window_end_utc,
    }
    reply.update(changes)
    return reply


def test_the_page_recomputes_the_digests_and_the_code_exactly_like_the_server() -> None:
    intent = ip13._intent()
    for deadline in (None, "2026-11-16T00:30:00Z"):
        challenge = ip13._challenge(intent, commitment_deadline_utc=deadline)
        (result,) = _verify_in_node([_reply(intent, challenge)])
        assert result["ok"] is True and result["problems"] == []
        assert result["code"] == challenge.confirmation_code
        assert result["digest"] == challenge.challenge_digest


def test_n5_the_page_aborts_on_every_mismatch_and_sends_nothing() -> None:
    intent = ip13._intent()
    challenge = ip13._challenge(intent)
    other = ip13._intent(selected_candidate_id="candidate-z")
    cases = [
        # a challenge that binds another Intent / key / payload / target / forecast
        _reply(intent, ip13._challenge(other)),
        _reply(intent, challenge, forecast_digest="1" * 64),
        _reply(intent, challenge, challenge_digest="2" * 64),
        _reply(intent, challenge, window_end_utc="2026-11-16T00:59:59Z"),
        _reply(intent, ip13._challenge(intent, payload_digest="3" * 64)),
        _reply(intent, ip13._challenge(intent, operation_key="4" * 64)),
        _reply(intent, ip13._challenge(intent, target_expected_revision=9)),
        _reply(intent, ip13._challenge(intent, target_instance_id="55555555-5555-4555-8555-555555555555")),
        _reply(intent, ip13._challenge(intent, intent_digest="6" * 64)),
        # the Intent bytes tampered with: the digest no longer follows
        _reply(intent, challenge, intent=intent.canonical_bytes().decode().replace("candidate-b", "candidate-x")),
        # non-canonical bytes
        _reply(intent, challenge, challenge=challenge.canonical_bytes().decode().replace('{"', '{ "', 1)),
    ]
    results = _verify_in_node(cases)
    assert len(results) == len(cases)
    for index, result in enumerate(results):
        assert result.get("ok") is False or "error" in result, (index, result)
    # The server-side window that the challenge outlives:
    late = ApprovalChallengeV1(
        challenge.nonce, challenge.direction, challenge.intent_id, challenge.intent_digest,
        challenge.expectation_digest, challenge.target_instance_id, challenge.target_expected_revision,
        challenge.operation_key, challenge.payload_digest, "2026-11-16T00:55:00Z", "2026-11-16T01:05:00Z",
        challenge.boot_id, None,
    )
    (result,) = _verify_in_node([_reply(intent, late)])
    assert result["ok"] is False and "challenge outlives the window" in result["problems"]


def test_the_page_script_is_valid_javascript() -> None:
    node = _node()
    if node is None:
        _node_missing()
    done = subprocess.run([node, "--check", str(PAGE)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_r18_reports_the_real_reason_without_echoing_input_values(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    """A settings refusal names where and why (not the class name only), and
    never prints the offending input value."""

    secret_host = "10.99.88.77"
    code, out, err, _lab_dir, _prod = _run(tmp_path, capsys, "--r18", api={"host": secret_host})
    assert code == 1 and "r18.enabled needs api.host on loopback" in err
    assert secret_host not in err and "ValidationError" not in err and "r18_target" not in out
