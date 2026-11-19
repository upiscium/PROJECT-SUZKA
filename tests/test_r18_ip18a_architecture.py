"""IP-18a: structural pins (no path from a caller's ``approved`` to the
executor, inverse stays dormant, in-process executor, stop order, deploy)."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
R18 = ROOT / "suzka" / "r18"
SERVER = ROOT / "suzka" / "api" / "server.py"
NEW_MODULES = ("executor.py", "sweep.py", "outcome_owner.py")


def _source(name: str) -> str:
    return (R18 / name).read_text()


def test_enqueue_is_reached_only_from_the_approval_owner_and_the_server_wiring() -> None:
    users = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "suzka").rglob("*.py")
        if re.search(r"\.enqueue\b", path.read_text())
    )
    # The executor's own backstop calls ``self.enqueue``; the server hands the
    # bound method to the approval owner; nothing else may feed the executor.
    assert users == ["suzka/api/server.py", "suzka/r18/executor.py"]
    owner = _source("approval_owner.py")
    assert "on_claimed" in owner and ".enqueue" not in owner


def test_no_model_or_caller_approved_flag_reaches_the_executor() -> None:
    for name in ("executor.py", "outcome_owner.py", "sweep.py"):
        assert not re.search(r"\bapproved\b", _source(name)), name
    # ``enqueue`` takes only an ExecutionClaimV1 (a CLAIMED event's value).
    assert "claim: ExecutionClaimV1" in _source("executor.py")


def test_the_inverse_stays_dormant() -> None:
    for name in NEW_MODULES:
        text = _source(name)
        assert "inverse_write" not in text and "verify_inverse" not in text, name
        assert "INVERSE_" not in text.replace("INVERSE_INTENT", ""), name


def test_the_executor_is_in_process_no_subprocess_no_second_unit() -> None:
    for name in NEW_MODULES:
        text = _source(name)
        assert not re.search(r"\b(subprocess|os\.system|os\.fork|multiprocessing|popen)\b", text), name


def test_the_deploy_units_never_weaken_the_kill_mode() -> None:
    for unit in (ROOT / "deploy" / "systemd").glob("*"):
        text = unit.read_text()
        assert not re.search(r"^\s*KillMode\s*=\s*(process|none|mixed)\b", text, re.M), unit.name


def test_the_server_stops_in_the_design_order_and_cleans_up_a_failed_start() -> None:
    text = SERVER.read_text()
    final = text[text.index("        try:\n            yield") :]
    order = [
        "approval_owner.stop()",
        "r18_executor.stop()",
        "r16_driver.stop()",
        "emotion_timer.stop()",
        "agent_runtime.shutdown()",
        "event_journal.close()",
    ]
    positions = [final.index(item) for item in order]
    assert positions == sorted(positions), positions
    # The start-failure cleanup (before ``yield``) also stops the executor,
    # and it starts only after the approval owner.
    start = text[text.index("app.state.approval_owner.start()") : text.index("        try:\n            yield")]
    assert start.index("r18_executor.start()") < start.index("r18_executor.stop()")
    assert start.index("approval_owner.stop()") < start.index("r18_executor.stop()") < start.index(
        "r16_driver.stop()"
    )


def test_the_real_gate_replaces_the_always_open_gate_at_the_install_sites() -> None:
    text = SERVER.read_text()
    assert "ALWAYS_OPEN_UNTIL_IP18" not in text
    assert text.count("admission_gate=r18_gate") == 3
    # The gate is built BEFORE the Decision issuer and handed a read-only view.
    assert text.index("r18_gate = SweepGate(") < text.index("app.state.decision_issuer = _DecisionIssuer(")
    assert "blocked_target=r18_gate.blocked_target" in text
    assert "target_facts=r18_facts.get" in text


def test_one_runtime_instance_id_per_process_reaches_the_owner_and_the_executor() -> None:
    text = SERVER.read_text()
    assert text.count("new_runtime_instance_id()") == 1
    assert text.count("runtime_instance_id=runtime_instance_id") == 2


def test_only_the_outcome_owner_submits_the_outcome_event() -> None:
    users = sorted(
        path.name
        for path in R18.glob("*.py")
        if "AgentEventType.R18_OUTCOME_COMMIT" in path.read_text()
    )
    assert users == ["outcome_owner.py"]


def test_the_executor_never_waits_on_the_runtime_thread_work_of_cognition() -> None:
    """Cognition never waits on the effect write: the worker owns the target,
    the Runtime thread only runs the Outcome event."""

    text = _source("executor.py")
    assert "TargetStore" in text and "forward_write" in text
    assert "agent_runtime" not in text.lower().replace("from suzka.runtime.agent_runtime", "")
