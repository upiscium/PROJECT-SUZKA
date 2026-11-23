"""The harness control channel (D1 sec. 4-2; risk 9).

A Unix socket under ``<base>/<run_id>/control/`` (directory 0700, socket 0600,
OUTSIDE the subject root), one JSON line per request, a CLOSED command set
``r15_review(stage)`` / ``status`` / ``quit``.  It is not a product route and
cannot be reached over the network.  The server accepts only peers of its own
uid; a client can read the server's pid from the socket credentials, so a
reply cannot come from another process unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import socket
import struct
import threading
from typing import Any

COMMANDS = frozenset({"r15_review", "status", "quit"})
STAGES = frozenset({"S1", "S2"})
MAX_LINE = 4096
IO_TIMEOUT_S = 60.0

Handler = Callable[[Mapping[str, Any]], dict[str, Any]]


class ControlError(Exception):
    pass


def validate_request(request: object) -> tuple[str, dict[str, Any]]:
    """The closed grammar: ``{"cmd": <one of COMMANDS>, ...}`` with exactly the
    arguments of that command.  Raises ``ControlError`` otherwise."""

    if not isinstance(request, dict) or not isinstance(request.get("cmd"), str):
        raise ControlError("bad_request")
    command = request["cmd"]
    if command not in COMMANDS:
        raise ControlError("unknown_command")
    arguments = {key: value for key, value in request.items() if key != "cmd"}
    if command == "r15_review":
        if set(arguments) != {"stage"} or arguments["stage"] not in STAGES:
            raise ControlError("bad_arguments")
    elif arguments:
        raise ControlError("bad_arguments")
    return command, arguments


def _peer_pid_uid(connection: socket.socket) -> tuple[int, int]:
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    pid, uid, _gid = struct.unpack("3i", raw)
    return pid, uid


def _read_line(connection: socket.socket) -> bytes:
    data = b""
    while b"\n" not in data:
        chunk = connection.recv(1024)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_LINE:
            raise ControlError("too_long")
    return data.split(b"\n", 1)[0]


class ControlServer:
    """Serves the closed command set from a daemon thread."""

    def __init__(self, path: Path, handlers: Mapping[str, Handler]) -> None:
        if set(handlers) != COMMANDS:
            raise ValueError("handlers must be exactly the closed command set")
        self.path = Path(path)
        self._handlers = dict(handlers)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        if os.path.lexists(self.path):
            # A SIGKILLed child leaves its socket behind: a refused connection
            # means nobody listens, so the file is stale; a live answer is not.
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(1.0)
            try:
                probe.connect(str(self.path))
            except (ConnectionRefusedError, FileNotFoundError):
                os.unlink(self.path)
            except OSError as error:
                raise ControlError("socket_exists") from error
            else:
                raise ControlError("socket_in_use")
            finally:
                probe.close()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        previous = os.umask(0o177)
        try:
            server.bind(str(self.path))
        finally:
            os.umask(previous)
        os.chmod(self.path, 0o600)
        server.listen(4)
        server.settimeout(0.2)
        self._socket = server
        self._thread = threading.Thread(target=self._serve, name="r20-control", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                connection.settimeout(5.0)
                try:
                    self._handle(connection)
                except (OSError, ControlError):
                    continue

    def _handle(self, connection: socket.socket) -> None:
        _pid, uid = _peer_pid_uid(connection)
        if uid != os.getuid():
            reply: dict[str, Any] = {"ok": False, "error": "peer_uid"}
        else:
            try:
                command, arguments = validate_request(json.loads(_read_line(connection)))
                reply = {"ok": True, **self._handlers[command](arguments)}
            except (ValueError, ControlError) as error:
                reply = {"ok": False, "error": str(error) if isinstance(error, ControlError) else "bad_request"}
            except Exception as error:  # a handler failure is a closed reply, not a crash
                reply = {"ok": False, "error": type(error).__name__}
        connection.sendall(json.dumps(reply, separators=(",", ":")).encode("utf-8") + b"\n")

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if os.path.lexists(self.path):
            os.unlink(self.path)


def send(path: Path, request: Mapping[str, Any], *, timeout_s: float = IO_TIMEOUT_S) -> tuple[dict[str, Any], int]:
    """One request; returns ``(reply, server_pid)``.  The pid comes from the
    socket credentials, not from the reply."""

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout_s)
    try:
        connection.connect(str(path))
        pid, _uid = _peer_pid_uid(connection)
        connection.sendall(json.dumps(dict(request), separators=(",", ":")).encode("utf-8") + b"\n")
        line = _read_line(connection)
    finally:
        connection.close()
    reply = json.loads(line)
    if not isinstance(reply, dict):
        raise ControlError("bad_reply")
    return reply, pid


__all__ = ["COMMANDS", "STAGES", "ControlError", "ControlServer", "send", "validate_request"]
