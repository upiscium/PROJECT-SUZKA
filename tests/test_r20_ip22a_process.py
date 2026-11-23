"""IP-22a: the control channel, the destination-bound client, short read
transactions and the T2 ``serve`` wrapper (D1 sec. 4-2, 4-3, 4-6b)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import threading
import time
from typing import Any

import pytest

from experiments.r20 import factory, polling
from experiments.r20.client import (
    BoundClient,
    Destination,
    DestinationMismatch,
    Recorder,
)
from experiments.r20.control import COMMANDS, ControlError, ControlServer, send, validate_request
from experiments.r20.factory import Tier, create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.supervisor import ChildProcess, child_environment

LINEAGE = "11111111-1111-4111-8111-111111111111"


# ------------------------------------------------------- the closed command set


def test_the_command_set_is_closed_and_its_arguments_are_validated() -> None:
    assert COMMANDS == {"r15_review", "status", "quit"}
    assert validate_request({"cmd": "status"}) == ("status", {})
    assert validate_request({"cmd": "r15_review", "stage": "S1"}) == ("r15_review", {"stage": "S1"})
    for bad in (
        {"cmd": "exec"}, {"cmd": "status", "extra": 1}, {"cmd": "r15_review"},
        {"cmd": "r15_review", "stage": "S9"}, {"cmd": "r15_review", "stage": "S1", "x": 1},
        {"cmd": "quit", "force": True}, {}, [], "status", {"cmd": 3},
    ):
        with pytest.raises(ControlError):
            validate_request(bad)


def test_a_server_needs_exactly_the_closed_handler_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ControlServer(tmp_path / "c.sock", {"status": lambda _a: {}})
    with pytest.raises(ValueError):
        ControlServer(tmp_path / "c.sock", {**{name: (lambda _a: {}) for name in COMMANDS}, "shell": lambda _a: {}})


def _fake_child_control(subject: factory.GenesisSubject, *, lineage: str = LINEAGE, pid: int | None = None, port: int | None = None) -> ControlServer:
    server = ControlServer(
        subject.control_socket,
        {
            "status": lambda _a: {
                "port": subject.port if port is None else port,
                "pid": os.getpid() if pid is None else pid,
                "run_id": subject.run_id,
                "journal_lineage_id": lineage,
                "ready": True,
            },
            "r15_review": lambda _a: {},
            "quit": lambda _a: {},
        },
    )
    server.start()
    return server


def test_the_control_socket_is_private_and_unknown_commands_get_closed_replies(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    server = _fake_child_control(subject)
    try:
        assert oct(subject.control_dir.stat().st_mode & 0o777) == "0o700"
        assert oct(subject.control_socket.stat().st_mode & 0o777) == "0o600"
        assert subject.root not in subject.control_socket.parents  # outside the subject root
        reply, pid = send(subject.control_socket, {"cmd": "status"})
        assert reply["ok"] is True and pid == os.getpid()
        for request in ({"cmd": "rm"}, {"cmd": "status", "x": 1}, {"cmd": "r15_review", "stage": "S0"}):
            reply, _ = send(subject.control_socket, request)
            assert reply["ok"] is False and set(reply) == {"ok", "error"}
        raw = socket.socket(socket.AF_UNIX)
        raw.connect(str(subject.control_socket))
        raw.sendall(b"not json\n")
        assert json.loads(raw.makefile().readline())["ok"] is False
        raw.close()
    finally:
        server.stop()
    assert not os.path.lexists(subject.control_socket)


def test_a_stale_socket_of_a_killed_child_is_replaced_but_a_live_one_is_not(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(str(subject.control_socket))
    stale.close()  # a SIGKILLed server leaves the file behind
    assert os.path.lexists(subject.control_socket)
    server = _fake_child_control(subject)
    try:
        assert send(subject.control_socket, {"cmd": "status"})[0]["ok"] is True
        with pytest.raises(ControlError):
            ControlServer(
                subject.control_socket, {name: (lambda _a: {}) for name in COMMANDS}
            ).start()
    finally:
        server.stop()


# ------------------------------------------------------------ destination binding


class _Listener:
    """A TCP listener that counts connections and bytes (a decoy or the child)."""

    def __init__(self, port: int = 0) -> None:
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(8)
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        self.received = b""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            self.connections += 1
            connection.settimeout(0.5)
            try:
                self.received += connection.recv(65536)
                connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            except OSError:
                pass
            finally:
                connection.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.sock.close()


def _record(subject: factory.GenesisSubject, lineage: str = LINEAGE) -> factory.IdentityRecord:
    record = factory.IdentityRecord(
        run_id=subject.run_id, profile=subject.profile.name, tier=subject.tier.value, port=subject.port,
        journal_lineage_id=lineage, genesis_incarnation_id="g", history_store_id="s",
        history_subject_id="x", target_instance_id=None, obligation_key_sha256=None,
        config_sha256="c", git_head="h",
    )
    factory.write_identity_record(subject, record)
    return record


def _destination(subject: factory.GenesisSubject, **changes: Any) -> Destination:
    base = Destination(
        run_id=subject.run_id, port=subject.port, pid=os.getpid(), ticks=None,
        control_socket=subject.control_socket, alive=lambda: True,
    )
    return dataclasses.replace(base, **changes)


@pytest.fixture
def bound(r20_base: Path):  # type: ignore[no-untyped-def]
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    _record(subject)
    control = _fake_child_control(subject)
    child = _Listener(subject.port)
    yield subject, child
    child.close()
    control.stop()


def _post(client: BoundClient) -> Any:
    return client.request("POST", "/api/r16/driver/activate")


def test_a_bound_client_sends_to_the_child_it_started(bound) -> None:  # type: ignore[no-untyped-def]
    subject, child = bound
    client = BoundClient(_destination(subject), subject.subject_json)
    assert _post(client).status == 200
    assert child.connections == 1
    head = child.received.decode("latin-1")
    assert head.startswith("POST /api/r16/driver/activate ")
    assert f"Host: localhost:{subject.port}".lower() in head.lower()
    assert client.label is None


def test_a_decoy_on_another_port_receives_zero_bytes(bound, r20_base: Path) -> None:  # type: ignore[no-untyped-def]
    subject, child = bound
    decoy = _Listener()
    try:
        client = BoundClient(_destination(subject, port=decoy.port), subject.subject_json)
        with pytest.raises(DestinationMismatch):
            _post(client)
        assert decoy.connections == 0 and decoy.received == b""
        assert child.connections == 0
        assert client.label == "DESTINATION_MISMATCH"
        # the mark is permanent: even a now-correct destination is not trusted again
        client.destination = _destination(subject)
        with pytest.raises(DestinationMismatch):
            _post(client)
        assert child.connections == 0
    finally:
        decoy.close()


def test_a_different_lineage_in_the_control_answer_sends_nothing(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    _record(subject)
    control = _fake_child_control(subject, lineage="22222222-2222-4222-8222-222222222222")
    child = _Listener(subject.port)
    try:
        client = BoundClient(_destination(subject), subject.subject_json)
        with pytest.raises(DestinationMismatch):
            _post(client)
        assert child.connections == 0 and client.label == "DESTINATION_MISMATCH"
    finally:
        child.close()
        control.stop()


def test_a_pid_or_run_or_port_disagreeing_with_the_records_sends_nothing(bound) -> None:  # type: ignore[no-untyped-def]
    subject, child = bound
    for changes in ({"pid": os.getpid() + 1}, {"run_id": "other"}):
        client = BoundClient(_destination(subject, **changes), subject.subject_json)
        with pytest.raises(DestinationMismatch):
            _post(client)
    assert child.connections == 0


def test_a_child_that_is_not_alive_right_before_the_send_gets_nothing(bound) -> None:  # type: ignore[no-untyped-def]
    subject, child = bound
    client = BoundClient(_destination(subject, alive=lambda: False), subject.subject_json)
    with pytest.raises(DestinationMismatch):
        _post(client)
    assert child.connections == 0


def test_a_missing_or_foreign_factory_record_sends_nothing(r20_base: Path) -> None:
    mine = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    theirs = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    _record(theirs)
    control = _fake_child_control(mine)
    child = _Listener(mine.port)
    try:
        with pytest.raises(DestinationMismatch):  # no subject.json yet
            _post(BoundClient(_destination(mine), mine.subject_json))
        # a client built for subject B refuses subject A's port
        with pytest.raises(DestinationMismatch):
            _post(BoundClient(_destination(mine), theirs.subject_json))
        assert child.connections == 0
    finally:
        child.close()
        control.stop()


@pytest.mark.parametrize("port", [8000, 8080, 8101])
def test_the_client_refuses_the_production_ports_for_every_method(bound, port: int) -> None:  # type: ignore[no-untyped-def]
    subject, _child = bound
    for method in ("POST", "GET"):
        client = BoundClient(_destination(subject, port=port), subject.subject_json)
        with pytest.raises(DestinationMismatch) as caught:
            client.request(method, "/health")
        assert caught.value.detail == "PORT_FORBIDDEN"


def test_reads_do_not_need_the_control_round_trip(bound) -> None:  # type: ignore[no-untyped-def]
    subject, child = bound
    client = BoundClient(_destination(subject, pid=os.getpid() + 1), subject.subject_json)
    assert client.request("GET", "/health").status == 200  # a GET changes nothing
    assert child.connections == 1


def test_the_recorder_keeps_status_and_a_closed_projection_and_finds_canaries(bound) -> None:  # type: ignore[no-untyped-def]
    subject, _child = bound

    class Fake:
        def request(self, method: str, path: str, *a: Any, **k: Any) -> Any:
            from experiments.r20.client import HttpResult

            return HttpResult(200, {}, json.dumps({"answer": "ok CANARY-123", "z": 1}).encode())

    recorder = Recorder(Fake(), {"obligation": "CANARY-123", "thought": "R20-HIDDEN"})
    recorder.request("GET", "/api/chat")
    (entry,) = recorder.entries
    assert entry["status"] == 200 and entry["keys"] == ["answer", "z"] and entry["canary_hits"] == ["obligation"]
    assert "CANARY-123" not in json.dumps(entry)  # the projection never stores the text
    assert recorder.sweep("server-log", "line R20-HIDDEN here") == ["thought"]


# -------------------------------------------------------- short read transactions


def test_short_read_is_read_only_one_select_and_releases_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    connection = sqlite3.connect(path, isolation_level=None)
    assert connection.execute("PRAGMA journal_mode = DELETE").fetchone() == ("delete",)
    connection.execute("CREATE TABLE t (a INTEGER)")
    connection.execute("INSERT INTO t VALUES (1)")
    assert polling.short_read(path, "SELECT a FROM t") == [(1,)]
    for statement in ("INSERT INTO t VALUES (2)", "DELETE FROM t", "PRAGMA journal_mode = WAL"):
        with pytest.raises(ValueError):
            polling.short_read(path, statement)
    # the reader holds nothing: an EXCLUSIVE writer starts at once
    connection.execute("BEGIN EXCLUSIVE")
    connection.execute("INSERT INTO t VALUES (2)")
    connection.execute("COMMIT")
    assert polling.short_read(path, "SELECT count(*) FROM t") == [(2,)]
    connection.close()


def test_a_locked_store_is_a_retry_not_a_failure(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    writer = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    writer.execute("PRAGMA journal_mode = DELETE")
    writer.execute("CREATE TABLE t (a INTEGER)")
    writer.execute("BEGIN EXCLUSIVE")
    with pytest.raises(sqlite3.OperationalError):
        polling.short_read(path, "SELECT a FROM t", busy_timeout_s=0.05)
    threading.Timer(0.4, lambda: writer.execute("COMMIT")).start()
    started = time.monotonic()
    assert polling.poll_until(lambda: polling.short_read(path, "SELECT 1 FROM sqlite_master", busy_timeout_s=0.05), timeout_s=5)
    assert time.monotonic() - started >= 0.3
    assert polling.poll_until(lambda: False, timeout_s=0.1) is None
    writer.close()


# ----------------------------------------------------------------- T2 serve


def test_the_t2_child_is_its_own_process_group_bound_and_restartable(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    config = {"script": ["first scripted answer"], "default": "scripted default"}
    child = ChildProcess.spawn(subject, provider_config=config)
    second: ChildProcess | None = None
    try:
        status = child.wait_ready()
        assert status["pid"] == child.pid and status["port"] == subject.port and status["provider"] == "scripted"
        assert status["journal_lineage_id"]
        # a normal child in its OWN process group, cwd = the root, allow-list environment
        assert os.getpgid(child.pid) == child.pid != os.getpgid(0)
        assert os.path.realpath(f"/proc/{child.pid}/cwd") == os.path.realpath(subject.root)
        environment = Path(f"/proc/{child.pid}/environ").read_bytes().split(b"\0")
        assert {item.split(b"=")[0] for item in environment if item.startswith(b"SUZKA_")} == {b"SUZKA_CONFIG_PATH"}
        assert child_environment(subject, {"SUZKA_SECRET": "x", "PATH": "/bin"}).keys() >= {"PATH", "PYTHONPATH"}
        assert "SUZKA_SECRET" not in child_environment(subject, {"SUZKA_SECRET": "x", "PATH": "/bin"})
        # the factory verifies adoption on the running child through read-only polls
        record = factory.verify_first_start(subject, provider={"mode": "scripted"})
        assert record.journal_lineage_id == status["journal_lineage_id"]
        factory.write_identity_record(subject, record)
        # control: the closed set answers; the bridge refuses a bad stage state
        reply, pid = send(subject.control_socket, {"cmd": "r15_review", "stage": "S1"})
        assert pid == child.pid and reply == {"ok": False, "error": "TURN_COUNT_MISMATCH"}
        assert send(subject.control_socket, {"cmd": "format"})[0]["ok"] is False
        # the bound client talks to this child only
        client = BoundClient(Destination.from_child(child), subject.subject_json)
        assert client.request("GET", "/health").status == 200
        chat = client.request("POST", "/api/chat", {"message": "hi", "attachments": []})
        assert chat.status == 200 and "scripted" in chat.text
        assert client.activate(subject.profile) == {"driver": 200, "self_goal": None}
        assert client.request("POST", "/api/r16/driver/activate").status == 409  # unique
        # SIGKILL of the group, then a new child on the same root
        old_pid = child.pid
        child.kill_group(signal.SIGKILL)
        assert not child.alive()
        with pytest.raises(DestinationMismatch):
            client.request("POST", "/api/chat", {"message": "after", "attachments": []})
        second = ChildProcess.spawn(subject, provider_config=config)
        again = second.wait_ready()
        assert again["pid"] == second.pid != old_pid and again["journal_lineage_id"] == status["journal_lineage_id"]
        assert client.label == "DESTINATION_MISMATCH"  # the old client stays refused
        fresh = BoundClient(Destination.from_child(second), subject.subject_json)
        assert fresh.request("POST", "/api/chat", {"message": "again", "attachments": []}).status == 200
    finally:
        child.stop()
        if second is not None:
            second.stop()
    for stopped in (child, second):
        assert stopped is None or not stopped.alive()
    with pytest.raises(ProcessLookupError):
        os.killpg(child.pgid, 0)


def test_serve_refuses_to_start_without_the_marker_as_a_real_process(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, tier=Tier.T2)
    (subject.root / ".r20_throwaway").unlink()
    with pytest.raises(GuardRefusal):
        ChildProcess.spawn(subject)  # the parent refuses before spawning
