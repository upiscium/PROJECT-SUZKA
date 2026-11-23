"""The destination-bound HTTP client and the recorder (D1 sec. 4-6, 4-6b).

Without the admin token layer, ``POST /api/r16/driver/activate`` and
``.../self-goal/activate`` are unauthenticated and irreversible, and a chat
turn is a state change too.  ``BoundClient`` therefore sends a state-changing
request (every method but GET / HEAD) ONLY to the port of the child this
harness started, and before EVERY such request:

1. the port equals the factory record (``subject.json``) and is not 8000 /
   8080 / 8101 (refused client-side as well as in the factory);
2. ``status`` on the control channel of that child answers with the same port,
   the same ``journal_lineage_id`` and the child's pid as the factory / spawn
   record, and the socket credentials say the answering process IS that pid;
3. right before the send the client re-checks that the pid is alive and still
   the process it spawned.

On any mismatch NOTHING is sent and the client is marked ``DESTINATION_MISMATCH``
for good; a restart means a new child, a new record and a new client.  The one
residual race (the child dies and another process takes the port between check
3 and the connect) is not closable client-side and is stated in the report.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import http.client
import json
from pathlib import Path
from typing import Any, Protocol

from experiments.r20 import guards
from experiments.r20.control import send as control_send
from experiments.r20.factory import Profile, read_identity_record
from experiments.r20.guards import GuardRefusal

LOOPBACK = "127.0.0.1"
SAFE_METHODS = frozenset({"GET", "HEAD"})
TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class HttpResult:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class Transport(Protocol):
    def send(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes | None
    ) -> HttpResult: ...


class SocketTransport:
    """A plain HTTP connection to the loopback port, one per request."""

    def __init__(self, port: int) -> None:
        self.port = guards.check_port(port)

    def send(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes | None
    ) -> HttpResult:
        connection = http.client.HTTPConnection(LOOPBACK, self.port, timeout=TIMEOUT_S)
        try:
            connection.request(method, path, body=body, headers=dict(headers))
            response = connection.getresponse()
            return HttpResult(response.status, dict(response.getheaders()), response.read())
        finally:
            connection.close()


class AsgiTransport:
    """T1: the in-process ``TestClient`` (no socket, so no destination)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def send(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes | None
    ) -> HttpResult:
        response = self._client.request(method, path, headers=dict(headers), content=body)
        return HttpResult(response.status_code, dict(response.headers), response.content)


class HttpClient:
    """Builds requests the way a browser-less local client does: ``Host:
    localhost:<port>``, JSON bodies, no ``Origin`` unless given."""

    def __init__(self, transport: Transport, port: int) -> None:
        self._transport = transport
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResult:
        merged = {"Host": f"localhost:{self.port}", "Accept": "application/json"}
        payload: bytes | None
        if body is None:
            payload = None
        elif isinstance(body, bytes):
            payload = body
        else:
            payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
            merged["Content-Type"] = "application/json"
        if payload is not None:
            merged["Content-Length"] = str(len(payload))
        elif method not in SAFE_METHODS:
            merged["Content-Length"] = "0"
        merged.update(headers or {})
        return self._transport.send(method, path, merged, payload)


@dataclass(frozen=True, slots=True)
class Destination:
    """The spawn record of the child this harness started."""

    run_id: str
    port: int
    pid: int
    ticks: int | None
    control_socket: Path
    alive: Callable[[], bool]

    @classmethod
    def from_child(cls, child: Any) -> Destination:
        return cls(
            run_id=child.subject.run_id,
            port=child.port,
            pid=child.pid,
            ticks=child.ticks,
            control_socket=child.subject.control_socket,
            alive=child.alive,
        )


class DestinationMismatch(GuardRefusal):
    def __init__(self, detail: str) -> None:
        super().__init__("DESTINATION_MISMATCH", detail)


class DestinationNoAnswer(GuardRefusal):
    """The child did not answer in time or is not ready yet: retryable, the
    client stays usable, and nothing was sent meanwhile."""

    def __init__(self, detail: str) -> None:
        super().__init__("DESTINATION_NO_ANSWER", detail)


class BoundClient:
    """The only HTTP client the harness uses against a T2 child."""

    def __init__(
        self,
        destination: Destination,
        factory_record: Path,
        *,
        transport_factory: Callable[[int], Transport] = SocketTransport,
    ) -> None:
        self.destination = destination
        self.factory_record = Path(factory_record)
        self._transport_factory = transport_factory
        self.label: str | None = None  # set to DESTINATION_MISMATCH, never cleared

    def _refuse(self, detail: str) -> DestinationMismatch:
        self.label = "DESTINATION_MISMATCH"
        return DestinationMismatch(detail)

    def verify_destination(self) -> None:
        """Checks 1 and 2 (the control round trip); raises and marks on failure."""

        dest = self.destination
        if self.label is not None:
            raise self._refuse("already marked")
        try:
            guards.check_port(dest.port)
        except GuardRefusal as refusal:
            raise self._refuse(refusal.code) from refusal
        try:
            record = read_identity_record(self.factory_record)
        except (OSError, ValueError, GuardRefusal) as error:
            raise self._refuse("factory record unreadable") from error
        if record.get("port") != dest.port or record.get("run_id") != dest.run_id:
            raise self._refuse("port or run differs from the factory record")
        try:
            reply, server_pid = control_send(dest.control_socket, {"cmd": "status"}, timeout_s=5)
        except TimeoutError as error:  # slow status: retryable, still sends nothing
            raise DestinationNoAnswer("status timed out") from error
        except (OSError, ValueError, GuardRefusal) as error:
            raise self._refuse("control channel does not answer") from error
        if reply.get("ok") is True and reply.get("ready") is False:
            raise DestinationNoAnswer("the child is not ready yet")
        if (
            reply.get("ok") is not True
            or reply.get("ready") is not True  # the child has bound its port
            or reply.get("port") != dest.port
            or reply.get("journal_lineage_id") is None
            or reply.get("journal_lineage_id") != record.get("journal_lineage_id")
            or reply.get("pid") != dest.pid
            or server_pid != dest.pid
            or reply.get("run_id") != dest.run_id
        ):
            raise self._refuse("control status differs from the records")

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResult:
        dest = self.destination
        try:
            guards.check_port(dest.port)  # every request, reads included
        except GuardRefusal as refusal:
            raise self._refuse(refusal.code) from refusal
        if method.upper() not in SAFE_METHODS:
            self.verify_destination()
            if not dest.alive():  # check 3, right before the send
                raise self._refuse("child is not alive")
        elif self.label is not None:
            raise self._refuse("already marked")
        http_client = HttpClient(self._transport_factory(dest.port), dest.port)
        return http_client.request(method.upper(), path, body, headers)

    def activate(self, profile: Profile) -> dict[str, int | None]:
        """The activations of a profile through the real routes: the driver
        for a profile that uses it, the self-goal path ONLY for S1 (HD-22-4)."""

        out: dict[str, int | None] = {"driver": None, "self_goal": None}
        if profile.driver:
            out["driver"] = self.request("POST", "/api/r16/driver/activate").status
        if profile.self_goal:
            out["self_goal"] = self.request("POST", "/api/r16/self-goal/activate").status
        return out


class Recorder:
    """Wraps any client; stores status plus a closed-field projection of every
    response (never its text) and which planted canaries it contained."""

    def __init__(self, inner: Any, canaries: Mapping[str, str] | None = None) -> None:
        self._inner = inner
        self._canaries = dict(canaries or {})
        self.entries: list[dict[str, Any]] = []

    def hits(self, text: str) -> list[str]:
        return sorted(name for name, value in self._canaries.items() if value and value in text)

    def request(self, method: str, path: str, *args: Any, **kwargs: Any) -> HttpResult:
        result: HttpResult = self._inner.request(method, path, *args, **kwargs)
        keys: list[str] = []
        try:
            decoded = result.json()
            if isinstance(decoded, dict):
                keys = sorted(str(key) for key in decoded)
        except ValueError:
            pass
        self.entries.append(
            {
                "method": method.upper(),
                "path": path,
                "status": result.status,
                "keys": keys,
                "length": len(result.body),
                "sha256": hashlib.sha256(result.body).hexdigest(),
                "canary_hits": self.hits(result.text),
            }
        )
        return result

    def sweep(self, label: str, text: str) -> list[str]:
        """Scan a log or an Outbox text for canaries (F13); names only."""

        found = self.hits(text)
        self.entries.append({"sweep": label, "length": len(text), "canary_hits": found})
        return found


__all__ = [
    "AsgiTransport",
    "BoundClient",
    "Destination",
    "DestinationMismatch",
    "DestinationNoAnswer",
    "HttpClient",
    "HttpResult",
    "Recorder",
    "SocketTransport",
    "Transport",
]
