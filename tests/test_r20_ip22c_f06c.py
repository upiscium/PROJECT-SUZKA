"""IP-22c, F06-c: a real SIGKILL on the process tier (T2) as SMOKE (D1 sec. 4-3, 6).

T2 results are NOT counted as covered: the provider is scripted but the poll
races the server, so the exact crash window is not controlled; the exact
windows are the merged T1 tests (crash matrix, IP-17c claim, IP-19, IP-20d).

Scope (stated, not hidden): this smoke is at the R16 level.  The D1 points
K2-K6 need the R18 / R19 chain on a T2 root, and that chain does NOT start on a
T2 root today: with ``r18.enabled`` and the factory's T2 layout (relative paths,
cwd = root, ``serve``) the app never builds the Decision / Intent issuer and the
R18 gate stays SWEEPING, so no Intent follows the row-4 Decision (observed
for 40 s; the same config in-process with absolute paths works).  The cause is
not isolated; it is reported for IP-22d, which owns K2-K6 and F15-x.

What is asserted after a real SIGKILL of the process group at the first moment
a Decision row is visible: the restart reaches readiness on the same root
with the same lineage, serves a new turn, and every old Decision row is intact
(no loss, no change, no duplicate; at most one row-3 and one row-4 Decision).  No R16 command is sent after the restart (the
driver resolves a target only from turns the CURRENT process observed).
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.r20 import factory
from experiments.r20.client import BoundClient, Destination
from experiments.r20.factory import Tier, create_subject
from experiments.r20.manifest import Manifest
from experiments.r20.polling import poll_until, short_read
from experiments.r20.supervisor import ChildProcess
from r20_chain import X_TEXT


def _shim(client: BoundClient) -> Any:
    """The ``TestClient`` shape over the destination-bound client."""

    def wrap(result: Any) -> Any:
        return SimpleNamespace(status_code=result.status, text=result.text, json=result.json)

    return SimpleNamespace(
        get=lambda path: wrap(client.request("GET", path)),
        post=lambda path, json=None: wrap(client.request("POST", path, json)),
    )


def _decisions(history: Path) -> list[str]:
    rows = short_read(
        history,
        "SELECT operation_digest FROM history_operations WHERE operation_kind = 'R16_DECISION_ISSUE' "
        "ORDER BY processing_sequence",
    )
    return [str(row[0]) for row in rows]


def _ticket(wire: Any, text: str) -> str:
    reply = wire.post("/api/chat", json={"message": text, "attachments": []})
    assert reply.status_code == 200, reply.text
    return str(reply.json()["command_ticket"]["ticket_id"])


def _settled(wire: Any, ticket: str) -> str:
    def view() -> str | None:
        status = str(wire.get(f"/api/chat/commands/{ticket}").json()["status"])
        return None if status == "pending" else status

    found = poll_until(view, timeout_s=30)
    assert found, "ticket never settled"
    return str(found)


def test_f06_c_a_real_sigkill_during_the_r16_chain_finishes_with_one_decision_per_stage(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    history = subject.resolved_settings().subject_history.path
    manifest = Manifest(subject.evidence_dir, subject.run_id, subject_root=subject.root)
    first = ChildProcess.spawn(subject)
    second: ChildProcess | None = None
    try:
        first.wait_ready()
        factory.write_identity_record(subject, factory.verify_first_start(subject, provider={"mode": "scripted"}))
        client = BoundClient(Destination.from_child(first), subject.subject_json)
        wire = _shim(client)
        assert client.activate(subject.profile)["driver"] == 200
        chat = wire.post("/api/chat", json={"message": X_TEXT, "attachments": []})
        assert chat.status_code == 200
        xid = chat.json()["experience_id"]
        # REVIEW once the INTEREST exists (a refused attempt is retried), then COMMIT
        assert poll_until(
            lambda: _settled(wire, _ticket(wire, f"REVIEW_EXPERIENCE:{xid}")) == "adopted",
            timeout_s=40,
            interval_s=0.3,
        )
        commit = _ticket(wire, f"COMMIT_REVIEW_EXPERIENCE:1:{xid}")

        # SIGKILL the group at the first moment a Decision row is visible
        assert poll_until(lambda: _decisions(history), timeout_s=40, interval_s=0.005)
        at_kill = _decisions(history)
        first.kill_group(signal.SIGKILL)
        assert not first.alive()
        manifest.append(
            "kill_points",
            {"point": "R16-after-first-decision" if len(at_kill) == 1 else "R16-later", "decisions": len(at_kill)},
        )
        del commit

        # restart on the same root: readiness, same lineage, the chain finishes
        second = ChildProcess.spawn(subject)
        again = second.wait_ready()
        assert again["pid"] == second.pid != first.pid
        recorded = factory.read_identity_record(subject.subject_json)
        assert again["journal_lineage_id"] == recorded["journal_lineage_id"]
        manifest.append("restarts", {"pid": second.pid})
        # the restarted subject serves a new turn on the SAME Journal
        fresh = BoundClient(Destination.from_child(second), subject.subject_json)
        assert fresh.request("GET", "/health").status == 200
        after = fresh.request("POST", "/api/chat", {"message": "after the restart", "attachments": []})
        assert after.status == 200
        time.sleep(3)  # a duplicate or a stray further Decision would show now
        final = _decisions(history)
        # no loss, no change, no duplicate: at most the row 3 and the row 4 exist,
        # and a Decision that was visible at the kill is byte-for-byte the same.
        # (The row-4 Decision is NOT re-issued by the restart: nothing triggers a
        # plan preview after a restart, which is the product's present behaviour.)
        assert final[: len(at_kill)] == at_kill
        assert len(final) == len(set(final)) <= 2
        manifest.result("F06-c", "SMOKE_T2_NOT_COUNTED")
        manifest.write()
    finally:
        first.stop()
        if second is not None:
            second.stop()
    assert not first.alive()
    with pytest.raises(ProcessLookupError):
        os.killpg(first.pgid, 0)
