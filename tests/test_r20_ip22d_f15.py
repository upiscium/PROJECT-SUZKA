"""IP-22d, F15-x (the rewritten F15 binding, F15 slice D1 r3 FINAL sec. 10).

The Human rule: a safety stop must never stop the subject; every stop ends within
bounded time without the Human in a terminal or re-evaluable state.  These are T2
SMOKE tests on a real S0 root served by a real child process with
``degraded_continuation.mode = "continue"`` and the SCRIPTED provider (provider
deterministic, kill window non-deterministic): they are not counted as covered.
What F15 "covers" is carried by the deterministic T1 DC tests the F-matrix cites;
these tests show the same behaviour end to end after a real SIGKILL.

* F15-x-1: kill, apply exactly ONE covered fault (an Experience pending artifact
  corrupted while a transaction is open), restart; the process stays
  alive, ``/health`` and ``/health/dc`` answer throughout, the subject serves and
  answers chat within the bound, no new Action is admitted on unproven history, the
  damaged artifact is byte-preserved in the quarantine, no row continues an
  identifier that only the quarantined tree holds.
* F15-x-2: put a copy of the quarantined directory back over the live root and
  restart: the process does NOT use it (``RECONCILING(domain_store_replaced)``), no
  row of it is read, no Action is admitted.  This is a known violation (F-15): the
  subject does not serve while it lasts.
* F15-x-3 (negative, the KNOWN-VIOLATION BOUNDARY): one uncovered fault (the
  SubjectHistory file removed; a Journal torn tail) must stay ``REFUSED_RUNNING``
  with the process alive and ``/health`` answering, nothing recreated, repaired or
  admitted.  The subject does NOT serve in these cases; that is a stated known
  violation of "a safety stop must never stop the subject", not a pass of it.

Not shown here (stated): that the live process never reads a quarantined path is
proved by the T1 access-spy test; here only the open descriptors of the child are
sampled, which is not exhaustive.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import pytest

from dc_kit import assert_trees_equal, tree_snapshot
from experiments.r20.client import BoundClient, DestinationMismatch, DestinationNoAnswer
from experiments.r20.control import ControlServer
from experiments.r20.factory import Tier, create_subject
from experiments.r20.manifest import Manifest
from experiments.r20.s0 import CONTINUE, S0Session
from test_dc4_real_app import arm_open_transaction
from test_r20_ip22a_process import LINEAGE, _destination, _Listener, _post, _record

# D1 sec. 3 defaults of the supervised mode (the settings' own defaults).
T_BOOT_MAX_S = 240.0
NOT_SERVING = {"REFUSED_RUNNING", "RESTART_LIMIT", "HARD_STUCK"}
# What counts as "an Action admitted": a new Decision, Intent, Policy verdict,
# challenge, claim or Outcome, or any change of the target's rows.
ACTION_KINDS = (
    "R16_DECISION_ISSUE", "R18_INTENT_ISSUE", "R18_POLICY_COMMIT", "R18_CHALLENGE_ISSUE",
    "R18_EXECUTION_CLAIM", "R18_OUTCOME_COMMIT",
)


def _actions(session: S0Session) -> dict[str, Any]:
    counters = session.counters()
    return {
        "counts": {kind: counters[kind] for kind in ACTION_KINDS},
        "target": session.evidence.target_digest(),
        "effects": session.evidence.target_effects(),
    }


def _open_paths(pid: int) -> list[str]:
    found: list[str] = []
    for entry in Path(f"/proc/{pid}/fd").iterdir():
        try:
            found.append(os.readlink(entry))
        except OSError:
            continue
    return found


@dataclass
class Quarantined:
    session: S0Session
    root: Path
    snapshot: dict[str, Any]
    transaction_id: str
    before: dict[str, Any]
    epoch: str
    elapsed_s: float


def _fault(session: S0Session) -> tuple[Path, dict[str, Any], str]:
    """Offline (the child is dead, its Journal lease released by the OS): an open
    PRE_INTERNAL transaction needing ``memory.experience`` whose pending artifact
    holds a conflicting body (the abort path raises a divergence, DC-4 quarantines).

    Returns the store root, the byte snapshot of the damaged tree, and the id.

    Not armed here (stated): the finalize path with a REMOVED pending artifact.  The
    ``Rig`` cannot commit internally on a real S0 root (its internal commit is not
    schema-10 aware: "ordinary v8-or-newer internal commit cannot downgrade
    schema"), and a PRE_INTERNAL transaction with an absent pending artifact is a plain
    ``ALREADY_ABSENT`` abort, no fault.  Those shapes are the deterministic T1 tests
    the F-matrix cites (evidence absent, the finalize path)."""

    return arm_open_transaction(session.subject.resolved_settings())


def _t2_quarantined_subject(base: Path) -> Quarantined:
    session = S0Session(base, dc=CONTINUE)
    session.start()
    session.drive_to_open_approval()  # Policy ALLOW on the Intent, not approved
    before = _actions(session)
    assert before["counts"]["R18_INTENT_ISSUE"] == 1 and before["effects"] == 0
    session.kill()  # a real SIGKILL of the process group
    root, snapshot, transaction = _fault(session)
    begun = time.monotonic()
    session.start()  # restart, supervised, mode = continue
    elapsed = time.monotonic() - begun
    epoch = session.health_dc()["dc_epoch_id"]
    return Quarantined(session, root, snapshot, transaction, before, str(epoch), elapsed)


def test_f15_x_1_a_covered_experience_fault_after_a_kill_is_quarantined_and_the_subject_serves(
    r20_base: Path,
) -> None:
    q = _t2_quarantined_subject(r20_base)
    session = q.session
    try:
        child = session.child
        assert child is not None and child.alive()  # the process never went away
        # (4) every stop ended in a re-evaluable or serving state, within the bound, without the Human
        states = {sample.get("dc_state") for sample in session.observed}
        assert "SERVING" in states and not states & NOT_SERVING, states
        assert q.elapsed_s <= T_BOOT_MAX_S and (session.serving_after_s or 0) <= T_BOOT_MAX_S
        # /health and /health/dc answered at every sample of the start, once the port was bound
        answered = [sample for sample in session.observed if "error" not in sample]
        assert answered and all(sample["health_status"] == 200 for sample in answered)
        health = session.health_dc()
        assert health["dc_state"] == "SERVING" and health["dc_reason"] == "domain_quarantined"
        assert health["dc_participants"] == "memory.experience"

        # the subject keeps answering chat on the quarantined domain's empty store
        reply = session.chat("hello after the quarantine")
        assert reply["response"] and reply["experience_id"]

        # (3) no new Action on unproven history (a window in which an Action could appear)
        time.sleep(8)
        after = _actions(session)
        assert after == q.before, (q.before, after)

        # (1) the damaged artifact is byte-preserved beside its origin
        quarantined = q.root.with_name(q.root.name + ".quarantine-" + q.epoch)
        assert_trees_equal(q.snapshot, tree_snapshot(quarantined))
        manifest_path = session.subject.root / ".suzka" / "quarantine" / q.epoch / "manifest.json"
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert recorded["state"] == "COMMITTED" and recorded["participant_id"] == "memory.experience"
        assert q.transaction_id in recorded["transaction_ids"]
        # the Journal says so: an explicit QUARANTINED outcome through the existing terminal records
        journal = session.evidence.journal.read_bytes()
        assert journal.count(b"\"quarantined\"") >= 1  # the enum value of AbortOutcome.QUARANTINED

        # (2) the live store holds nothing that only the quarantined tree has
        live_files = {p.relative_to(q.root).as_posix() for p in q.root.rglob("*") if p.is_file()}
        old_files = {p.relative_to(quarantined).as_posix() for p in quarantined.rglob("*") if p.is_file()}
        assert not {name for name in live_files & old_files if name != ".dc_epoch.json"}
        assert (q.root / ".dc_epoch.json").is_file()

        # the child holds no descriptor under a quarantined path (a sample, not a proof)
        assert not [path for path in _open_paths(child.pid) if ".quarantine-" in path]

        manifest = Manifest(session.subject.evidence_dir, session.subject.run_id, subject_root=session.subject.root)
        manifest.append("kill_points", {"point": "F15-x-1:experience_pending_corrupted", "tier": "T2", "label": "SMOKE_T2_NOT_COUNTED"})
        manifest.append("restarts", {"pid": child.pid, "serving_after_s": session.serving_after_s})
        manifest.result("F15-x-1", "SMOKE_T2_NOT_COUNTED")
        manifest.write()
    finally:
        session.stop()


def test_f15_x_2_a_restored_copy_of_the_quarantined_tree_is_detected_blocks_and_is_never_used(
    r20_base: Path,
) -> None:
    q = _t2_quarantined_subject(r20_base)
    session = q.session
    try:
        quarantined = q.root.with_name(q.root.name + ".quarantine-" + q.epoch)
        qdir = session.subject.root / ".suzka" / "quarantine"
        epochs_before = sorted(path.name for path in qdir.iterdir() if path.is_dir())
        assert epochs_before == [q.epoch]
        session.kill()
        # the Human puts a copy of the quarantined directory back over the live root
        aside = q.root.with_name(q.root.name + ".live-aside")
        q.root.rename(aside)
        shutil.copytree(quarantined, q.root)
        restored = tree_snapshot(q.root)
        old = tree_snapshot(quarantined)
        actions = _actions(session)

        session.start(expect=("RECONCILING", "REFUSED_RUNNING"))
        child = session.child
        assert child is not None and child.alive()
        health = session.health_dc()
        # F15 slice D1 predicts RECONCILING(domain_store_replaced).  The real server path
        # differs when no transaction is open at the restart: ``_subject_cycle`` calls
        # ``inspect_degraded_startup``, which refuses with StateRecoveryError("degraded
        # transaction authority is ambiguous") when there is nothing open to prove, so the
        # observed state is REFUSED_RUNNING.  Both are the same SAFE class (process alive,
        # state visible, re-evaluated, restored tree unused); the deviation from the D1
        # text is recorded in the manifest and reported, not hidden, and not fixed here
        # (IP-22 changes no product code).
        observed = (health["dc_state"], health["dc_reason"])
        assert observed in {
            ("RECONCILING", "domain_store_replaced"),
            ("REFUSED_RUNNING", "suzka.runtime.state_recovery.StateRecoveryError"),
        }, observed
        d1_state = observed == ("RECONCILING", "domain_store_replaced")
        # the process answers, but the subject does NOT serve while this lasts (known violation F-15)
        assert session.health()["status"] == "degraded"
        assert session.request("GET", "/api/contexts", timeout_s=10).status == 503
        # re-evaluated, not resolved by itself: still the same state a while later
        time.sleep(6)
        assert (session.health_dc()["dc_state"], session.health_dc()["dc_reason"]) == observed
        # nothing of the restored tree was used, moved or changed; no second epoch; no Action
        assert_trees_equal(restored, tree_snapshot(q.root))
        assert_trees_equal(old, tree_snapshot(quarantined))
        assert sorted(path.name for path in qdir.iterdir() if path.is_dir()) == epochs_before
        assert _actions(session) == actions
        assert not [path for path in _open_paths(child.pid) if ".quarantine-" in path]
        manifest = Manifest(session.subject.evidence_dir, session.subject.run_id, subject_root=session.subject.root)
        manifest.result("F15-x-2", "SMOKE_T2_NOT_COUNTED")
        manifest.result(
            "F15-x-2.d1_predicted_state_observed",
            "YES" if d1_state else f"NO: observed {observed[0]}({observed[1]}) instead of RECONCILING(domain_store_replaced)",
        )
        manifest.result("F15-x-2.known_violation", "NOT_COVERED_KNOWN_VIOLATION: stale restore over a quarantined store (F-15)")
        manifest.write()
    finally:
        session.stop()


@pytest.mark.parametrize("fault", ["subject_history_removed", "journal_torn_tail"])
def test_f15_x_3_an_uncovered_fault_stays_refused_with_the_process_alive_and_nothing_repaired(
    r20_base: Path, fault: str
) -> None:
    session = S0Session(r20_base, dc=CONTINUE)
    session.start()
    try:
        session.activate()
        session.chat("a turn before the kill")
        rows_before = session.counters() if fault == "journal_torn_tail" else {}
        history = session.subject.resolved_settings().subject_history.path
        journal = session.evidence.journal
        session.kill()
        if fault == "subject_history_removed":
            for sidecar in history.parent.glob(history.name + "*"):
                sidecar.unlink()
            assert not history.exists()
        else:
            data = journal.read_bytes()
            assert data.endswith(b"\n")
            journal.write_bytes(data[:-23])  # a record cut in the middle: a torn tail
            torn = journal.read_bytes()
        session.start(expect="REFUSED_RUNNING")
        child = session.child
        assert child is not None and child.alive()  # the process stays alive (R-B: not dead, not hidden)
        health = session.health_dc()
        assert health["dc_state"] == "REFUSED_RUNNING" and health["dc_reason"]
        assert session.health()["status"] == "degraded"  # /health answers
        # known violation: the subject does not serve (chat is refused), and nothing recovers by itself
        # (a POST cannot even be addressed: the child has no Journal lineage to verify, so the
        # destination-bound client sends nothing; a GET shows the phase middleware's 503)
        assert session.request("GET", "/api/contexts", timeout_s=10).status == 503
        later: list[Any] = []
        for _ in range(8):
            time.sleep(1)
            later.append(session.health_dc()["dc_state"])
        assert set(later) <= {"REFUSED_RUNNING", "STARTING", "RESTARTING"} and "SERVING" not in later
        assert child.alive()
        if fault == "subject_history_removed":
            assert not history.exists()  # never recreated as a fresh DB under the adopted identity
            assert not list(history.parent.glob(history.name + "*"))
        else:
            assert journal.read_bytes() == torn  # never repaired, truncated or extended
            assert session.counters() == rows_before  # and nothing was admitted on the unproven Journal
        manifest = Manifest(session.subject.evidence_dir, session.subject.run_id, subject_root=session.subject.root)
        manifest.result(f"F15-x-3.{fault}", "REFUSED_RUNNING")
        manifest.result(
            f"F15-x-3.{fault}.known_violation",
            "NOT_COVERED_KNOWN_VIOLATION: the subject does not serve while this lasts; process alive, state visible",
        )
        manifest.write()
    finally:
        session.stop()


def test_a_child_whose_journal_is_not_built_yet_is_a_retryable_no_answer_and_gets_nothing(
    r20_base: Path,
) -> None:
    """A supervised child binds its port before its first cycle has built the
    Journal, so its lineage is not known YET.  The destination check says "no answer
    yet" (retryable, nothing is sent, the client is not marked) instead of marking a
    mismatch for good; with the lineage known and equal it sends."""

    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    _record(subject)
    answer: dict[str, Any] = {"lineage": None}
    server = ControlServer(
        subject.control_socket,
        {
            "status": lambda _a: {
                "port": subject.port, "pid": os.getpid(), "run_id": subject.run_id,
                "journal_lineage_id": answer["lineage"], "ready": True,
            },
            "r15_review": lambda _a: {},
            "quit": lambda _a: {},
        },
    )
    server.start()
    listener = _Listener(subject.port)
    try:
        client = BoundClient(_destination(subject), subject.subject_json)
        with pytest.raises(DestinationNoAnswer):
            _post(client)
        assert listener.connections == 0 and client.label is None
        answer["lineage"] = LINEAGE
        assert _post(client).status == 200 and listener.connections == 1
        # a WRONG lineage is still a permanent mismatch: only "not known yet" is retryable
        answer["lineage"] = "22222222-2222-4222-8222-222222222222"
        with pytest.raises(DestinationMismatch):
            _post(client)
        assert client.label == "DESTINATION_MISMATCH" and listener.connections == 1
    finally:
        listener.close()
        server.stop()
