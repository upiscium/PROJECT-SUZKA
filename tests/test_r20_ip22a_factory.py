"""IP-22a: the genesis-subject factory on real apps (D1 sec. 3, sec. 8-1):
identity uniqueness, no shared object / path / inode, first-start adoption,
the T1 runtime, the process-wide fake clock and the R15 bridge."""

from __future__ import annotations

from datetime import timedelta
import hashlib
from pathlib import Path

import pytest

from experiments.r20 import bridge, factory
from experiments.r20.factory import Tier, create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.provider import ScriptedProvider, r15_protocol_script
from experiments.r20.tier1 import FakeClock, T1Runtime
from r20_fixtures import started_record


def test_three_subjects_share_no_identity_object_path_or_inode(r20_base: Path) -> None:
    subjects = [create_subject("S0", base=r20_base) for _ in range(3)]
    runtimes = [T1Runtime(subject, ScriptedProvider()) for subject in subjects]
    try:
        for runtime in runtimes:
            runtime.start()
        records = [started_record(s, r) for s, r in zip(subjects, runtimes, strict=True)]
        factory.assert_pairwise_distinct(records)
        # every component of the id tuple is present for an r18 subject, and distinct
        for index in range(5):
            values = [record.id_tuple()[index] for record in records]
            assert None not in values and len(set(values)) == 3
        assert len({record.history_subject_id for record in records}) == 3
        # the obligation secret of each subject is its own
        keys = {hashlib.sha256((s.target_path.parent / "obligation.key").read_bytes()).hexdigest() for s in subjects if s.target_path}
        assert len(keys) == 3
        factory.assert_disjoint_roots(subjects)
        # leak test: one process, three live apps, no shared object
        states = [runtime.state for runtime in runtimes]
        for name in ("event_journal", "agent_runtime", "subject_history_store", "experience_store", "memory_system"):
            assert len({id(getattr(state, name)) for state in states}) == 3, name
        # a turn on subject 0 appears in no other subject's Journal
        before = [len(state.event_journal.inspect().records) for state in states]
        assert runtimes[0].client is not None
        reply = runtimes[0].client.post("/api/chat", json={"message": "hello", "attachments": []})
        assert reply.status_code == 200
        after = [len(state.event_journal.inspect().records) for state in states]
        assert after[0] > before[0] and after[1:] == before[1:]
    finally:
        for runtime in runtimes:
            runtime.stop()


def test_a_collision_or_a_shared_inode_is_detected(r20_base: Path) -> None:
    first, second = (create_subject("NEG-plain", base=r20_base) for _ in range(2))
    runtime_a, runtime_b = T1Runtime(first, ScriptedProvider()), T1Runtime(second, ScriptedProvider())
    with runtime_a, runtime_b:
        a, b = started_record(first, runtime_a), started_record(second, runtime_b)
        assert a.target_instance_id is None and a.obligation_key_sha256 is None  # r18 off: no target
        factory.assert_pairwise_distinct([a, b])
        import dataclasses

        clone = dataclasses.replace(b, journal_lineage_id=a.journal_lineage_id)
        with pytest.raises(GuardRefusal) as caught:
            factory.assert_pairwise_distinct([a, clone])
        assert caught.value.code == "IDENTITY_COLLISION"
    # a hard link between two roots is a shared inode
    (second.root / "linked").hardlink_to(first.root / "config.yaml")
    with pytest.raises(GuardRefusal) as shared:
        factory.assert_disjoint_roots([first, second])
    assert shared.value.code == "SHARED_INODE"


def test_first_start_adopts_subject_history_and_the_factory_verifies_it(r20_base: Path) -> None:
    subject = create_subject("S0", base=r20_base)
    # before the first start there is no store: the factory only prepared preconditions
    assert not subject.resolved_settings().subject_history.path.exists()
    with pytest.raises(Exception):
        factory.verify_first_start(subject)
    with T1Runtime(subject, ScriptedProvider()) as runtime:
        record = started_record(subject, runtime)
        meta = factory.read_history_meta(subject.resolved_settings().subject_history.path)
        assert meta["phase"] == "ACTIVE" and meta["adopted_baseline_id"] and meta["genesis_incarnation_id"]
        path = factory.write_identity_record(subject, record)
        assert path.parent == subject.evidence_dir
        stored = factory.read_identity_record(path)
        assert stored["port"] == subject.port and stored["run_id"] == subject.run_id
        assert stored["config_sha256"] == hashlib.sha256(subject.config_path.read_bytes()).hexdigest()
        assert "token" not in " ".join(stored)  # no admin token layer any more
        with pytest.raises(FileExistsError):
            factory.write_identity_record(subject, record)  # written once


def test_t1_restart_is_a_new_app_on_the_same_root_with_the_same_identity(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base)
    runtime = T1Runtime(subject, ScriptedProvider())
    with runtime:
        first_app, first = runtime.client.app, started_record(subject, runtime)  # type: ignore[union-attr]
        runtime.restart()
        assert runtime.client.app is not first_app  # type: ignore[union-attr]
        assert started_record(subject, runtime) == first


def test_t2_subjects_are_not_started_in_process_and_t1_not_by_the_supervisor(r20_base: Path) -> None:
    t2 = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    with pytest.raises(GuardRefusal) as caught:
        T1Runtime(t2, ScriptedProvider())
    assert caught.value.code == "NOT_A_T1_SUBJECT"
    from experiments.r20.supervisor import ChildProcess

    t1 = create_subject("NEG-plain", base=r20_base, tier=Tier.T1)
    with pytest.raises(GuardRefusal) as other:
        ChildProcess.spawn(t1)
    assert other.value.code == "NOT_A_T2_SUBJECT"


def test_the_fake_clock_is_process_wide_so_only_one_may_be_installed() -> None:
    from suzka.runtime import agent_runtime

    real = agent_runtime.datetime
    with FakeClock() as clock:
        assert agent_runtime.datetime is not real
        with pytest.raises(GuardRefusal) as caught:
            FakeClock().install()
        assert caught.value.code == "FAKE_CLOCK_ALREADY_INSTALLED"
        before = agent_runtime.datetime.now(tz=__import__("datetime").timezone.utc)
        clock.advance(3600)
        assert agent_runtime.datetime.now(tz=__import__("datetime").timezone.utc) - before >= timedelta(seconds=3599)
        clock.set_back(7200)  # the backward-clock cases
        assert agent_runtime.datetime.now(tz=__import__("datetime").timezone.utc) < before
        with pytest.raises(ValueError):
            clock.advance(-1)
    assert agent_runtime.datetime is real
    FakeClock().install().uninstall()  # reusable after uninstall


def test_the_fake_clock_drives_the_runtime_stamp_of_a_real_app(r20_base: Path) -> None:
    """A backward wall clock is refused by the real runtime (fail closed) and
    the same app is usable again once the clock is back.  As in the existing
    R19 fixtures, a FORWARD advance is for owner / runtime driven events: the
    chat route reads the real clock too, so HTTP chat after ``advance`` is not
    part of the T1 clock contract."""

    subject = create_subject("NEG-plain", base=r20_base)
    with FakeClock() as clock, T1Runtime(subject, ScriptedProvider(), clock=clock) as runtime:
        client = runtime.client
        assert client is not None
        assert client.post("/api/chat", json={"message": "one", "attachments": []}).status_code == 200
        records = len(runtime.state.event_journal.inspect().records)
        clock.set_back(5)
        refused = client.post("/api/chat", json={"message": "two", "attachments": []})
        assert refused.status_code == 503
        assert len(runtime.state.event_journal.inspect().records) == records  # nothing admitted
        clock.advance(5)
        assert client.post("/api/chat", json={"message": "three", "attachments": []}).status_code == 200


def test_the_r15_bridge_admits_s1_then_s2_through_the_apps_own_owner(r20_base: Path) -> None:
    """The protocol turns are the real R15 P1 runner's (sealed fixture); only
    the trigger is the bridge's, as in ``tests/test_r15_p1bb_wal.py``."""

    from experiments.r15_p0.pilot import Case, seal_case
    from experiments.r15_p1.protocol import run_trial

    subject = create_subject("NEG-r15", base=r20_base)
    provider = ScriptedProvider(r15_protocol_script())
    manifest, capture = subject.evidence_dir / "p1.manifest.json", subject.evidence_dir / "p1.capture.json"
    seal_case(Case("r20-22a-bridge", "A", "forced", 0, "normal"), manifest)
    with T1Runtime(subject, provider) as runtime:
        client = runtime.client
        assert client is not None
        state = runtime.state
        owner = state.r15_state_port.narrative
        with pytest.raises(bridge.BridgeRefused):  # no turns yet
            bridge.review(state, "S1")
        with pytest.raises(bridge.BridgeRefused):
            bridge.review(state, "S3")
        admitted: list[dict[str, object]] = []
        calls = 0

        def transport(message: str) -> dict[str, str]:
            nonlocal calls
            if calls == 3:  # after E and the S1 request, BEFORE E2
                with pytest.raises(bridge.BridgeRefused):  # S2 needs five turns
                    bridge.review(state, "S2")
                admitted.append(bridge.review(state, "S1"))
            calls += 1
            reply = client.post("/api/chat", json={"message": message, "attachments": []})
            assert reply.status_code == 200
            data = reply.json()
            return {"episode_id": data["episode_id"], "response": data["response"]}

        run_trial(manifest, capture, transport)
        assert admitted == [{"stage": "S1", "changed": True, "revision": 1, "path": "HARNESS_BRIDGE"}]
        second = bridge.review(state, "S2")
        assert second["revision"] == 2 and second["changed"] is True
        assert owner.snapshot().revision == 2
        assert provider.calls == 5
