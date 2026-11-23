"""The approval operator stand-in (D1 sec. 4-5).

A scripted operator that is a CLIENT OF THE REAL ROUTE: it sends what a browser
sends (``Host``, ``Origin``, ``X-SUZKA-R18: 1``, JSON; optionally
``Sec-Fetch-Site: same-origin``; never a forwarding header), so
``R18RequestGuard`` runs unmodified, and the new ``ApiOriginGuard`` exempts
``/api/r18/*``.

It RECOMPUTES the digests and the typed code independently from the displayed
challenge (a Python port of ``suzka/r18/approval_page/approve.js``); it never
imports the R18 contracts and never calls the approval owner's ``confirm`` /
``claim`` / ``issue``.  A port that disagrees with the server fails the test
instead of being accommodated.  It runs in the harness process, never in the
app process, and nothing the provider can say reaches it.  ``SCRIPTED`` is
refused with a real provider (HD-22-3); ``HUMAN`` only announces the URL.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol

from experiments.r20.guards import GuardRefusal
from experiments.r20.provider import ApprovalActor, ProviderMode, check_modes

CHALLENGE_DOMAIN = "PROJECT-SUZKA:R18:APPROVAL-CHALLENGE:V1\0"
CODE_DOMAIN = "PROJECT-SUZKA:R18:CONFIRMATION-CODE:V1\0"
INTENT_DOMAIN = "PROJECT-SUZKA:R18:INTENT:V1\0"
ARTIFACT_DOMAIN = "PROJECT-SUZKA:R18:REVIEW-ARTIFACT:V1\0"
CODE_LENGTH = 8
MAX_SAFE_INTEGER = 2**53 - 1


class _Client(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Any = None,
    ) -> Any: ...


def canon(value: Any) -> str:
    """The page's canonical JSON: sorted keys, no spaces, integers only."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("not canonical")
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(canon(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(key, ensure_ascii=False) + ":" + canon(value[key]) for key in sorted(value)
        ) + "}"
    raise ValueError("not canonical")


def _digest(domain: str, text: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + text.encode("utf-8")).hexdigest()


def code_from(raw: bytes) -> str:
    """The first ``CODE_LENGTH`` characters of RFC 4648 base32 = the first 40 bits."""

    return base64.b32encode(raw[:5]).decode("ascii")[:CODE_LENGTH]


@dataclass(frozen=True, slots=True)
class Recomputed:
    ok: bool
    problems: tuple[str, ...]
    code: str
    challenge_digest: str
    nonce: str


def recompute(reply: dict[str, Any]) -> Recomputed:
    """Verify challenge <-> intent <-> forecast and recompute digests and code
    from the bytes the server returned (the page's ``verify``)."""

    problems: list[str] = []
    challenge_text, intent_text = reply["challenge"], reply["intent"]
    challenge, intent = json.loads(challenge_text), json.loads(intent_text)
    if canon(challenge) != challenge_text:
        problems.append("challenge bytes are not canonical")
    if canon(intent) != intent_text:
        problems.append("intent bytes are not canonical")
    challenge_digest = _digest(CHALLENGE_DOMAIN, challenge_text)
    body = {key: value for key, value in intent.items() if key not in {"intent_digest", "intent_id"}}
    intent_digest = _digest(INTENT_DOMAIN, canon(body))
    payload_digest = _digest(ARTIFACT_DOMAIN, canon(intent["artifact"]))
    code = code_from(hashlib.sha256(CODE_DOMAIN.encode() + challenge_digest.encode()).digest())

    def same(name: str, first: Any, second: Any) -> None:
        if first != second:
            problems.append(f"{name} differs")

    same("intent_digest", intent.get("intent_digest"), intent_digest)
    same("challenge intent_digest", challenge.get("intent_digest"), intent_digest)
    same("challenge_digest", reply.get("challenge_digest"), challenge_digest)
    same("intent_id", challenge.get("intent_id"), intent.get("intent_id"))
    same("operation_key", challenge.get("operation_key"), intent.get("forward_key"))
    same("payload_digest", challenge.get("payload_digest"), payload_digest)
    same("target_instance_id", challenge.get("target_instance_id"), intent.get("target_instance_id"))
    same("target_expected_revision", challenge.get("target_expected_revision"), intent.get("target_expected_revision"))
    same("expectation_digest", challenge.get("expectation_digest"), reply.get("forecast_digest"))
    same("direction", challenge.get("direction"), "FORWARD")
    same("window_end", reply.get("window_end_utc"), intent.get("approval_window_end_utc"))
    if challenge["expires_utc"] > intent["approval_window_end_utc"]:
        problems.append("challenge outlives the window")
    deadline = challenge.get("commitment_deadline_utc")
    if deadline is not None and challenge["expires_utc"] > deadline:
        problems.append("challenge outlives the Commitment deadline")
    return Recomputed(not problems, tuple(problems), code, challenge_digest, str(challenge["nonce"]))


class ApprovalOperator:
    """The scripted operator.  ``same_origin_fetch`` adds the header browsers
    send (``Sec-Fetch-Site: same-origin``); the guard accepts it when present."""

    def __init__(
        self,
        client: _Client,
        port: int,
        *,
        provider: ProviderMode = ProviderMode.SCRIPTED,
        actor: ApprovalActor = ApprovalActor.SCRIPTED,
        same_origin_fetch: bool = False,
    ) -> None:
        # HD-22-3 (a): a scripted operator never goes with a real provider.
        check_modes(provider, ApprovalActor.SCRIPTED, human_launch=False)
        if actor is ApprovalActor.HUMAN:
            raise GuardRefusal("HUMAN_ACTOR_APPROVES_BY_HAND", self.url_for(port))
        self._client = client
        self.port = port
        self.same_origin_fetch = same_origin_fetch
        self.actor = actor

    @staticmethod
    def url_for(port: int) -> str:
        """What the HUMAN actor is shown: the page URL for the Human's tunnel."""

        return f"http://localhost:{port}/api/r18/approve"

    def headers(self, *, post: bool, **changes: str | None) -> dict[str, str]:
        """Browser-equivalent headers; ``changes`` replace or (``None``) drop
        one, so the guard tests can send exactly one deviation."""

        headers = {"Host": f"localhost:{self.port}", "X-SUZKA-R18": "1"}
        if post:
            headers["Origin"] = f"http://localhost:{self.port}"
            headers["Content-Type"] = "application/json"
        if self.same_origin_fetch:
            headers["Sec-Fetch-Site"] = "same-origin"
        for name, value in changes.items():
            key = name.replace("_", "-")
            for existing in [k for k in headers if k.lower() == key.lower()]:
                del headers[existing]
            if value is not None:
                headers[key] = value
        return headers

    def listing(self) -> Any:
        return self._client.request("GET", "/api/r18/approvals", None, self.headers(post=False))

    def review(self, intent_id: str) -> Any:
        return self._client.request(
            "POST", f"/api/r18/approvals/{intent_id}/review", {}, self.headers(post=True)
        )

    def confirm(self, intent_id: str, nonce: str, code: str, **header_changes: str | None) -> Any:
        return self._client.request(
            "POST",
            f"/api/r18/approvals/{intent_id}/confirm",
            {"nonce": nonce, "code": code},
            self.headers(post=True, **header_changes),
        )

    def approve(self, intent_id: str) -> Any:
        """review -> recompute -> confirm.  Aborts, sending nothing, when the
        recomputation finds a problem (as the page does)."""

        shown = self.review(intent_id)
        if shown.status != 200:
            return shown
        reply = shown.json()
        if reply.get("status") != "review":
            return shown
        checked = recompute(reply)
        if not checked.ok:
            raise GuardRefusal("APPROVAL_RECOMPUTE_MISMATCH", "; ".join(checked.problems[:3]))
        return self.confirm(intent_id, checked.nonce, checked.code)


__all__ = [
    "ARTIFACT_DOMAIN",
    "CHALLENGE_DOMAIN",
    "CODE_DOMAIN",
    "INTENT_DOMAIN",
    "ApprovalOperator",
    "Recomputed",
    "canon",
    "code_from",
    "recompute",
]
