"""IP-17c: the approval API, its request guards and the page (#411 D1 r7 §4,
§5, §9, §10 N1 / N2).

Every refusal asserts that nothing was written and no approval event ran.
"""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import re
from typing import Any

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from suzka.api.server import create_app
from suzka.config.schema import (
    R18Settings,
    approval_host_header,
    is_loopback_host,
    require_approval_binding,
)
from suzka.r18 import approval_api
from suzka.r18.approval_api import (
    MAX_BODY_BYTES,
    GuardConfig,
    R18RequestGuard,
    Refusal,
    check_request,
)
from suzka.r18.contracts import ApprovalChallengeV1, ExecutionClaimV1
from suzka.runtime.agent_runtime import AgentEventType

import test_r16_ip10_decision_issuer as ip10
import test_r16_ip16b_ledger_reader as ip16b
import test_r18_ip16a2_intent_issuer as ip16a2
import test_r18_ip17b_approval as ip17b
from test_r16_u2a_source_owner import _app

ROOT = Path(__file__).resolve().parents[1]
LOCAL = "http://localhost:8000"
LAN = "http://10.12.0.2:8080"
NAMED = "http://melchior:8080"
NETWORKS = ["10.12.0.0/16", "10.255.255.0/24", "192.168.100.0/24"]
_APPROVAL_TYPES = frozenset(
    {
        AgentEventType.R18_CHALLENGE_ISSUE,
        AgentEventType.R18_APPROVAL_EXPIRE,
        AgentEventType.R18_EXECUTION_CLAIM,
    }
)
_allowed = ip17b._allowed
_adopted = ip17b._adopted
_fake_timers = ip17b._fake_timers
_no_failed_events = ip16a2._no_failed_events


def _settings(root: Path, *, r18: bool = True, **r18_changes: Any):  # type: ignore[no-untyped-def]
    base = ip16a2._settings(root, r18=r18)
    changes: dict[str, Any] = {
        "approval_origins": [LOCAL, LAN, NAMED],
        "approval_trusted_networks": NETWORKS,
    }
    changes.update(r18_changes)
    return base.model_copy(update={"r18": base.r18.model_copy(update=changes)})


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    return _app(_settings(tmp_path / "api"))


def _client(app, **kwargs: Any) -> TestClient:  # type: ignore[no-untyped-def]
    return TestClient(app, base_url=LOCAL, **kwargs)


def _events(state) -> int:  # type: ignore[no-untyped-def]
    return sum(1 for row in state.event_journal.records if row.event_type in _APPROVAL_TYPES)


def _snapshot(state) -> tuple[int, int, int]:  # type: ignore[no-untyped-def]
    return (
        len(ip10._history_transactions(state)),
        _events(state),
        len(state.approval_owner._pending_challenges),
    )


DATA = {"X-SUZKA-R18": "1"}


def _local(path_origin: str = LOCAL, post: bool = False, **extra: str) -> dict[str, str]:
    headers = dict(DATA)
    if post:
        headers["Origin"] = path_origin
        headers["Content-Type"] = "application/json"
    headers.update(extra)
    return headers


def _lan(post: bool = False, forwarded: str = "10.12.5.5", **extra: str) -> dict[str, str]:
    headers = _local(LAN, post, **extra)
    headers["Host"] = "10.12.0.2:8080"
    if forwarded:
        headers["X-Forwarded-For"] = forwarded
    return headers


def _id(intent) -> str:  # type: ignore[no-untyped-def]
    return intent.intent_id


# ===================================================== settings (§4 / §9)


def test_the_settings_defaults_are_localhost_only() -> None:
    settings = R18Settings()
    assert settings.approval_origins == ["http://localhost:8000"]
    assert settings.approval_trusted_networks == []
    assert settings.approval_ttl_seconds == 300 and settings.approval_window_seconds == 3600
    assert settings.approval_reviews_per_minute == 10


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:8000",  # a relay's Host must never match
        "http://127.1.2.3:8000",
        "http://[::1]:8000",
        "http://localhost:8000/",
        "http://localhost:8000/x",
        "http://user@localhost:8000",
        "ftp://localhost:8000",
        "HTTP://localhost:8000",
        "http://LOCALHOST:8000",
        "localhost:8000",
        "http://localhost:99999",
        "",
        "http://localhost:8000?x=1",
    ],
)
def test_a_bad_or_loopback_ip_origin_is_rejected_by_the_settings_validator(origin: str) -> None:
    with pytest.raises(ValidationError):
        R18Settings(approval_origins=[origin])


def test_origins_and_networks_are_closed_lists() -> None:
    with pytest.raises(ValidationError):
        R18Settings(approval_origins=[])
    with pytest.raises(ValidationError):
        R18Settings(approval_origins=[LOCAL, LOCAL])
    for network in ("10.12.0.1/16", "0.0.0.0/0", "nonsense", "10.12.0.0/33"):
        with pytest.raises(ValidationError):
            R18Settings(approval_trusted_networks=[network])
    for value in (0, 61, "10", 10.0):
        with pytest.raises(ValidationError):
            R18Settings(approval_reviews_per_minute=value)  # type: ignore[arg-type]
    ok = R18Settings(approval_origins=[LOCAL, LAN, NAMED], approval_trusted_networks=NETWORKS)
    assert [approval_host_header(origin) for origin in ok.approval_origins] == [
        "localhost:8000",
        "10.12.0.2:8080",
        "melchior:8080",
    ]
    assert approval_host_header("http://melchior") == "melchior"
    # The default port is never spelled (a browser omits it), so such an origin is refused.
    for spelled in ("https://melchior:443", "http://melchior:80"):
        with pytest.raises(ValueError, match="default port"):
            approval_host_header(spelled)
        with pytest.raises(ValueError):
            R18Settings(approval_origins=[spelled])  # type: ignore[list-item]
    assert approval_host_header("https://melchior") == "melchior"
    assert approval_host_header("https://melchior:8443") == "melchior:8443"
    assert approval_host_header("http://melchior:443") == "melchior:443"


def test_a_trusted_network_origin_needs_a_loopback_listen_address() -> None:
    lan = R18Settings(enabled=True, approval_origins=[LAN], approval_trusted_networks=NETWORKS)
    for host in ("127.0.0.1", "localhost", "::1", "127.0.0.2"):
        assert is_loopback_host(host)
        require_approval_binding(host, lan)
    for host in ("0.0.0.0", "10.12.0.2", "192.168.1.5", "::", "melchior"):
        assert not is_loopback_host(host)
        with pytest.raises(ValueError, match="loopback"):
            require_approval_binding(host, lan)
    # A localhost origin needs loopback too: on a wider listen address any LAN
    # host could send ``Host: localhost:8000`` and pass the direct transport.
    for host in ("0.0.0.0", "10.12.0.2", "::"):
        with pytest.raises(ValueError, match="loopback"):
            require_approval_binding(host, R18Settings(enabled=True))
    require_approval_binding("127.0.0.1", R18Settings(enabled=True))
    # R18 off asks nothing.
    require_approval_binding("0.0.0.0", R18Settings(enabled=False, approval_origins=[LAN]))


def test_startup_refuses_a_lan_origin_with_a_non_loopback_api_host(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "bind")
    wide = settings.model_copy(update={"api": settings.api.model_copy(update={"host": "0.0.0.0"})})
    with pytest.raises(ValueError, match="loopback"):
        create_app(wide)
    # model_copy skips the validators, so the full Settings model re-checks it too.
    with pytest.raises(ValidationError):
        type(wide).model_validate(wide.model_dump(mode="json"))
    # A localhost-only origin on a wide listen address is refused as well.
    local_only = _settings(tmp_path / "bind-local", approval_origins=[LOCAL], approval_trusted_networks=[])
    wide_local = local_only.model_copy(
        update={"api": local_only.api.model_copy(update={"host": "0.0.0.0"})}
    )
    with pytest.raises(ValueError, match="loopback"):
        create_app(wide_local)
    with pytest.raises(ValidationError):
        type(wide_local).model_validate(wide_local.model_dump(mode="json"))
    # Loopback is accepted; r18 off mounts nothing and asks nothing.
    create_app(settings)
    off = _settings(tmp_path / "bind-off", r18=False)
    create_app(off.model_copy(update={"api": off.api.model_copy(update={"host": "0.0.0.0"})}))


# =================================================== R18 off: unchanged


def test_r18_off_mounts_nothing_and_the_app_is_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    off = create_app(_settings(tmp_path / "off", r18=False))
    on = create_app(_settings(tmp_path / "on"))
    assert not [r for r in off.routes if str(getattr(r, "path", "")).startswith("/api/r18")]
    assert all(m.cls is not R18RequestGuard for m in off.user_middleware)
    assert any(m.cls is R18RequestGuard for m in on.user_middleware)
    assert not hasattr(off.state, "approval_guard_config")
    # Exactly the same routes and middleware apart from the R18 additions.
    base_routes = [(r.path, sorted(getattr(r, "methods", ()) or ())) for r in off.routes]
    on_routes = [(r.path, sorted(getattr(r, "methods", ()) or ())) for r in on.routes]
    assert on_routes[: len(base_routes)] == base_routes
    assert [m.cls for m in on.user_middleware[1:]] == [m.cls for m in off.user_middleware]
    with TestClient(off, base_url=LOCAL) as client:
        for method, path in (
            ("GET", "/api/r18/approve"),
            ("GET", "/api/r18/approve.js"),
            ("GET", "/api/r18/approvals"),
            ("POST", "/api/r18/approvals/" + "intent-" + "0" * 64 + "/review"),
            ("POST", "/api/r18/approvals/" + "intent-" + "0" * 64 + "/confirm"),
        ):
            assert client.request(method, path, headers=_local(post=method == "POST")).status_code == 404
        assert client.get("/health").status_code == 200
        # Nothing about /api/r18 is special: the app-wide CORS answers as it
        # does for any path.
        assert client.options(
            "/api/r18/approvals",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        ).headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert _events(client.app.state) == 0


# ========================================================= the page (§5)


def test_the_static_page_and_its_headers(app) -> None:  # type: ignore[no-untyped-def]
    with _client(app) as client:
        state = client.app.state
        before = _snapshot(state)
        for path in ("/api/r18/approve", "/api/r18/approve.js"):
            # Rule 1 only: no X-SUZKA-R18 and no Origin on a navigation / script load.
            response = client.get(path)
            assert response.status_code == 200
            csp = response.headers["content-security-policy"]
            for directive in (
                "frame-ancestors 'none'", "script-src 'self'", "default-src 'none'",
                "connect-src 'self'", "base-uri 'none'", "object-src 'none'",
            ):
                assert directive in csp
            assert response.headers["x-frame-options"] == "DENY"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["cache-control"] == "no-store"
        page = client.get("/api/r18/approve")
        assert page.headers["content-type"].startswith("text/html")
        assert 'src="/api/r18/approve.js"' in page.text and "<script>" not in page.text
        assert "onclick" not in page.text.lower()
        script = client.get("/api/r18/approve.js")
        assert script.headers["content-type"].startswith("application/javascript")
        # The page is subject to rule 1: a wrong Host, or forwarding headers, refuse it.
        assert client.get("/api/r18/approve", headers={"Host": "evil.example"}).status_code == 403
        assert client.get("/api/r18/approve", headers={"X-Forwarded-For": "10.12.5.5"}).status_code == 403
        assert client.get("/api/r18/approve", headers={"Host": "10.12.0.2:8080"}).status_code == 403
        assert client.get("/api/r18/approve", headers={"Host": "10.12.0.2:8080", "X-Forwarded-For": "10.12.5.5"}).status_code == 200
        # POST on the page path is a data request.
        assert client.post("/api/r18/approve", headers=_local(post=True), content=b"{}").status_code in (404, 405)
        assert _snapshot(state) == before


def test_the_page_script_has_no_html_sink_and_computes_what_the_server_does() -> None:
    text = (ROOT / "suzka" / "r18" / "approval_page" / "approve.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "dangerouslySetInnerHTML"):
        assert sink not in text
    from suzka.r18 import contracts

    for name, constant in (
        ("CHALLENGE_DOMAIN", contracts._CHALLENGE_DOMAIN),
        ("CODE_DOMAIN", contracts._CODE_DOMAIN),
        ("INTENT_DOMAIN", contracts._INTENT_DOMAIN),
        ("ARTIFACT_DOMAIN", contracts._ARTIFACT_DOMAIN),
    ):
        match = re.search(rf'var {name} = "([^"]*)";', text)
        assert match is not None
        assert match.group(1).replace("\\0", "\0").encode("ascii") == constant
    assert f"var CODE_LENGTH = {contracts.CONFIRMATION_CODE_LENGTH};" in text
    assert "Cache" not in text and "localStorage" not in text and "sessionStorage" not in text
    html = (ROOT / "suzka" / "r18" / "approval_page" / "approve.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "javascript:" not in html


# ============================================ guards (§4) over RAW headers


def _h(host: str, **headers: str) -> list[tuple[bytes, bytes]]:
    pairs = [(b"host", host.encode())]
    pairs += [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return pairs


def test_check_request_is_pure_and_uses_the_last_forwarded_entry() -> None:
    config = GuardConfig.from_settings(
        R18Settings(approval_origins=[LOCAL, LAN, NAMED], approval_trusted_networks=NETWORKS)
    )
    data = {"x-suzka-r18": "1"}
    ok = check_request(config, "GET", "/api/r18/approvals", _h("localhost:8000", **data))
    assert ok is None
    lan = lambda value: check_request(  # noqa: E731
        config, "GET", "/api/r18/approvals", _h("10.12.0.2:8080", **{**data, "x-forwarded-for": value})
    )
    assert lan("10.12.5.5") is None
    assert lan("8.8.8.8, 10.12.5.5") is None  # the client is the last entry
    assert lan("10.12.5.5, 8.8.8.8") is Refusal.NETWORK  # spoofed first entry, real client outside
    assert lan("192.168.100.7") is None and lan("10.255.255.9") is None
    assert lan("10.255.254.1") is Refusal.NETWORK
    assert lan("") is Refusal.NETWORK and lan("garbage") is Refusal.NETWORK
    assert lan("::ffff:10.12.5.5") is None  # IPv4-mapped
    assert lan("2001:db8::1") is Refusal.NETWORK
    # Two Host headers, a missing Host, an unknown Host, a Host with a trailing dot.
    assert check_request(config, "GET", "/api/r18/approvals", [(b"host", b"localhost:8000")] * 2) is Refusal.HOST
    assert check_request(config, "GET", "/api/r18/approvals", [(b"x-suzka-r18", b"1")]) is Refusal.HOST
    for host in ("127.0.0.1:8000", "localhost", "localhost:8001", "localhost.:8000", "10.12.0.2", "evil:8080"):
        assert check_request(config, "GET", "/api/r18/approvals", _h(host, **data)) is Refusal.HOST
    # Forwarding headers with a localhost Host.
    for name in ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto"):
        assert (
            check_request(config, "GET", "/api/r18/approvals", _h("localhost:8000", **data, **{name: "x"}))
            is Refusal.FORWARDING
        )
    # Methods.
    assert check_request(config, "OPTIONS", "/api/r18/approvals", _h("localhost:8000")) is Refusal.OPTIONS
    for method in ("PUT", "DELETE", "PATCH", "HEAD", "TRACE"):
        assert check_request(config, method, "/api/r18/approvals", _h("localhost:8000", **data)) is Refusal.METHOD


def test_transfer_encoding_and_a_foreign_origin_authority_are_refused() -> None:
    config = GuardConfig.from_settings(
        R18Settings(approval_origins=[LOCAL, LAN], approval_trusted_networks=NETWORKS)
    )
    good = {"x-suzka-r18": "1", "origin": LOCAL, "content-type": "application/json", "content-length": "2"}
    assert check_request(config, "POST", "/api/r18/x", _h("localhost:8000", **good)) is None
    assert (
        check_request(config, "POST", "/api/r18/x", _h("localhost:8000", **good, **{"transfer-encoding": "chunked"}))
        is Refusal.BODY_SIZE
    )
    assert check_request(config, "GET", "/api/r18/approve", _h("localhost:8000", **{"transfer-encoding": "chunked"})) is Refusal.BODY_SIZE
    # A listed Origin that is not this request's Host (LAN origin on a localhost Host and back).
    assert check_request(config, "POST", "/api/r18/x", _h("localhost:8000", **{**good, "origin": LAN})) is Refusal.ORIGIN
    lan = {**good, "origin": LAN, "x-forwarded-for": "10.12.5.5"}
    assert check_request(config, "POST", "/api/r18/x", _h("10.12.0.2:8080", **lan)) is None
    assert check_request(config, "POST", "/api/r18/x", _h("10.12.0.2:8080", **{**lan, "origin": LOCAL})) is Refusal.ORIGIN


def test_a_root_path_cannot_bypass_the_guard_and_bodies_are_read_bounded() -> None:
    import asyncio

    config = GuardConfig.from_settings(R18Settings())
    limits = approval_api.ApprovalLimits.create(config)
    reached: list[str] = []

    async def inner(scope, receive, send):  # type: ignore[no-untyped-def]
        reached.append(scope["path"])

    guard = R18RequestGuard(inner, config, limits)

    def call(path: str, root_path: str, headers: dict[str, str]) -> int | None:
        sent: list[dict[str, Any]] = []

        async def send(message):  # type: ignore[no-untyped-def]
            sent.append(message)

        scope = {
            "type": "http", "method": "GET", "path": path, "root_path": root_path,
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
        asyncio.run(guard(scope, None, send))  # type: ignore[arg-type]
        return int(sent[0]["status"]) if sent else None

    bad = {"Host": "evil.example"}
    assert call("/api/r18/approvals", "", bad) == 403
    assert call("/app/api/r18/approvals", "/app", bad) == 403  # stripped path is r18
    assert call("/api/r18/approvals", "/api", bad) == 403  # raw path is r18
    assert call("/health", "", bad) is None and reached == ["/health"]
    good = {"Host": "localhost:8000", **DATA}
    assert call("/app/api/r18/approvals", "/app", good) is None


def test_the_oversized_streamed_body_is_not_buffered(app) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    from starlette.requests import Request

    async def run() -> object:
        chunks = [b"x" * 10_000, b"x" * 10_000]

        async def receive():  # type: ignore[no-untyped-def]
            if chunks:
                return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request({"type": "http", "method": "POST", "headers": []}, receive)
        return await approval_api._json_body(request)

    assert asyncio.run(run()) is None


def test_mounting_logs_the_closed_warning_code(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    with caplog.at_level(logging.WARNING, logger="suzka.r18.approval_api"):
        create_app(_settings(tmp_path / "warn"))
    assert "R18_ENABLED_BEFORE_EXECUTOR" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="suzka.r18.approval_api"):
        create_app(_settings(tmp_path / "warn-off", r18=False))
    assert "R18_ENABLED_BEFORE_EXECUTOR" not in caplog.text


def test_check_request_data_header_origin_fetch_site_content_type_and_size() -> None:
    config = GuardConfig.from_settings(R18Settings(approval_origins=[LOCAL]))
    path = "/api/r18/approvals/intent-x/confirm"
    good = {
        "x-suzka-r18": "1",
        "origin": LOCAL,
        "content-type": "application/json",
        "content-length": "10",
    }
    assert check_request(config, "POST", path, _h("localhost:8000", **good)) is None
    assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "content-type": "application/json; charset=utf-8"})) is None
    assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "sec-fetch-site": "same-origin"})) is None

    def without(name: str) -> dict[str, str]:
        return {k: v for k, v in good.items() if k != name}

    assert check_request(config, "POST", path, _h("localhost:8000", **without("x-suzka-r18"))) is Refusal.DATA_HEADER
    for bad in ("0", "true", "", "1 ", "11"):
        assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "x-suzka-r18": bad})) is Refusal.DATA_HEADER
    assert check_request(config, "POST", path, _h("localhost:8000", **without("origin"))) is Refusal.ORIGIN
    for origin in ("http://evil.example", "http://localhost:3000", "null", "http://127.0.0.1:8000", LOCAL + "/"):
        assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "origin": origin})) is Refusal.ORIGIN
    for site in ("cross-site", "same-site", "none", ""):
        assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "sec-fetch-site": site})) is Refusal.FETCH_SITE
    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "application/jsonx"):
        assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "content-type": content_type})) is Refusal.CONTENT_TYPE
    assert check_request(config, "POST", path, _h("localhost:8000", **without("content-type"))) is Refusal.CONTENT_TYPE
    assert check_request(config, "POST", path, _h("localhost:8000", **without("content-length"))) is Refusal.BODY_SIZE
    assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "content-length": str(MAX_BODY_BYTES + 1)})) is Refusal.BODY_SIZE
    assert check_request(config, "POST", path, _h("localhost:8000", **{**good, "content-length": str(MAX_BODY_BYTES)})) is None
    # A GET: Origin is optional but checked when present.
    get = {"x-suzka-r18": "1"}
    assert check_request(config, "GET", path, _h("localhost:8000", **get)) is None
    assert check_request(config, "GET", path, _h("localhost:8000", **get, origin=LOCAL)) is None
    assert check_request(config, "GET", path, _h("localhost:8000", **get, origin="http://evil.example")) is Refusal.ORIGIN
    assert check_request(config, "GET", path, _h("localhost:8000", **get, **{"sec-fetch-site": "cross-site"})) is Refusal.FETCH_SITE


def test_the_guard_reads_raw_headers_not_the_rewritten_client() -> None:
    """uvicorn rewrites ``request.client`` (``scope["client"]``) from
    X-Forwarded-For for loopback peers; the guard never looks at it."""

    import asyncio

    config = GuardConfig.from_settings(
        R18Settings(approval_origins=[LOCAL, LAN], approval_trusted_networks=NETWORKS)
    )
    limits = approval_api.ApprovalLimits.create(config)
    reached: list[str] = []

    async def inner(scope, receive, send):  # type: ignore[no-untyped-def]
        reached.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = R18RequestGuard(inner, config, limits)

    def call(headers: dict[str, str], client: tuple[str, int]) -> int:
        sent: list[dict[str, Any]] = []

        async def send(message):  # type: ignore[no-untyped-def]
            sent.append(message)

        async def receive():  # type: ignore[no-untyped-def]
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/r18/approvals",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": client,
        }
        asyncio.run(guard(scope, receive, send))
        return int(sent[0]["status"])

    direct = {"Host": "localhost:8000", **DATA}
    outside = {"Host": "10.12.0.2:8080", "X-Forwarded-For": "8.8.8.8", **DATA}
    inside = {"Host": "10.12.0.2:8080", "X-Forwarded-For": "10.12.5.5", **DATA}
    for peer in (("127.0.0.1", 5000), ("10.12.5.5", 5000), ("8.8.8.8", 5000), ("::1", 5000)):
        # Accepted or refused by the raw headers alone, whatever the peer is.
        assert call(direct, peer) == 200
        assert call(inside, peer) == 200
        assert call(outside, peer) == 403
    assert len(reached) == 8
    # Non-r18 paths and non-http scopes pass through untouched.
    assert call({"Host": "evil"}, ("8.8.8.8", 1)) == 403

    async def other() -> None:
        passed: list[str] = []

        async def inner2(scope, receive, send):  # type: ignore[no-untyped-def]
            passed.append(scope["type"] + scope.get("path", ""))

        wrapped = R18RequestGuard(inner2, config, limits)
        for scope in (
            {"type": "http", "method": "GET", "path": "/health", "headers": []},
            {"type": "http", "method": "OPTIONS", "path": "/api/r180", "headers": []},
            {"type": "lifespan"},
        ):
            await wrapped(scope, None, None)  # type: ignore[arg-type]
        assert passed == ["http/health", "http/api/r180", "lifespan"]

    asyncio.run(other())


# ================================ N2: refused with nothing written (HTTP)


def _refused(response, reason: Refusal) -> None:  # type: ignore[no-untyped-def]
    assert response.status_code in (403, 405, 413, 429), response.text
    assert response.json() == {"status": "refused", "reason": reason.value}
    assert "access-control-allow-origin" not in response.headers


def test_n2_every_refusal_writes_nothing_and_submits_no_event(app) -> None:  # type: ignore[no-untyped-def]
    with _client(app) as client:
        state = client.app.state
        _fake_timers(state)
        _gid, intent, _forecast = _allowed(state)
        review = state.approval_owner.review(intent.intent_id)
        assert review.challenge is not None
        before = _snapshot(state)
        base = f"/api/r18/approvals/{_id(intent)}"
        body = json.dumps({"nonce": review.challenge.nonce, "code": review.challenge.confirmation_code})

        def post(path: str, headers: dict[str, str], content: str = body):  # type: ignore[no-untyped-def]
            return client.post(path, headers=headers, content=content)

        # forwarding headers with a localhost Host
        for name in ("X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "Forwarded"):
            _refused(post(base + "/confirm", _local(post=True, **{name: "10.12.5.5"})), Refusal.FORWARDING)
            _refused(client.get("/api/r18/approvals", headers=_local(**{name: "10.12.5.5"})), Refusal.FORWARDING)
        # no forwarding headers with a LAN Host
        no_xff = _lan(post=True, forwarded="")
        _refused(post(base + "/confirm", no_xff), Refusal.NETWORK)
        # X-Forwarded-For outside the CIDRs
        _refused(post(base + "/confirm", _lan(post=True, forwarded="8.8.8.8")), Refusal.NETWORK)
        _refused(post(base + "/confirm", _lan(post=True, forwarded="10.12.5.5, 8.8.8.8")), Refusal.NETWORK)
        # a 127.0.0.1 Host (no origin may name it)
        _refused(post(base + "/confirm", {**_local(post=True), "Host": "127.0.0.1:8000"}), Refusal.HOST)
        # a Host name that is not listed
        _refused(post(base + "/confirm", {**_lan(post=True), "Host": "10.12.0.9:8080"}), Refusal.HOST)
        _refused(client.get("/api/r18/approvals", headers={**_lan(), "Host": "other:8080"}), Refusal.HOST)
        # a missing X-SUZKA-R18
        _refused(post(base + "/confirm", {k: v for k, v in _local(post=True).items() if k != "X-SUZKA-R18"}), Refusal.DATA_HEADER)
        _refused(client.get("/api/r18/approvals"), Refusal.DATA_HEADER)
        # a wrong or missing Origin, a wrong Sec-Fetch-Site
        _refused(post(base + "/confirm", _local("http://evil.example", post=True)), Refusal.ORIGIN)
        _refused(post(base + "/confirm", {k: v for k, v in _local(post=True).items() if k != "Origin"}), Refusal.ORIGIN)
        _refused(client.get("/api/r18/approvals", headers=_local(Origin="http://evil.example")), Refusal.ORIGIN)
        _refused(post(base + "/confirm", _local(post=True, **{"Sec-Fetch-Site": "cross-site"})), Refusal.FETCH_SITE)
        _refused(client.get("/api/r18/approvals", headers=_local(**{"Sec-Fetch-Site": "same-site"})), Refusal.FETCH_SITE)
        # a non-JSON POST
        for content_type in ("text/plain", "application/x-www-form-urlencoded"):
            _refused(
                post(base + "/confirm", {**_local(post=True), "Content-Type": content_type}),
                Refusal.CONTENT_TYPE,
            )
        # a body over 16 KiB
        _refused(post(base + "/confirm", _local(post=True), "x" * (MAX_BODY_BYTES + 1)), Refusal.BODY_SIZE)
        # the admin proxy's exact request shape: Host 127.0.0.1:8000 / localhost:8000,
        # only Content-Type (no token any more), on GET and on POST.
        for host, reason in (("127.0.0.1:8000", Refusal.HOST), ("localhost:8000", Refusal.DATA_HEADER)):
            proxy = {"Host": host, "Content-Type": "application/json"}
            _refused(client.get("/api/r18/approvals", headers=proxy), reason)
            _refused(client.get(base + "/status", headers=proxy), reason)
            _refused(client.post(base + "/review", headers=proxy, content="{}"), reason)
            post_reason = Refusal.DATA_HEADER if host == "localhost:8000" else Refusal.HOST
            _refused(client.post(base + "/confirm", headers=proxy, content=body), post_reason)
        # a wrong code is accepted by the guards but writes nothing (a 409 HOLD).
        wrong = post(base + "/confirm", _local(post=True), json.dumps({"nonce": review.challenge.nonce, "code": "WRONG000"}))
        assert wrong.status_code == 409 and wrong.json() == {"status": "hold", "reason": "code_mismatch"}
        assert _snapshot(state) == before
        assert _adopted(state, ApprovalChallengeV1) == [] and _adopted(state, ExecutionClaimV1) == []
        _no_failed_events(state)


def test_options_on_r18_is_refused_before_the_cors_middleware(app) -> None:  # type: ignore[no-untyped-def]
    with _client(app) as client:
        cors = client.app.state.settings.api.cors_origins
        assert cors  # the app-wide CORS middleware would answer these origins
        origin = cors[0]
        preflight = {"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-suzka-r18"}
        for path in ("/api/r18/approve", "/api/r18/approvals", "/api/r18/approvals/intent-x/confirm", "/api/r18/nothing"):
            for headers in (preflight, {}, {**preflight, "Host": "10.12.0.2:8080", "X-Forwarded-For": "10.12.5.5"}):
                response = client.options(path, headers=headers)
                assert response.status_code == 405
                assert response.json() == {"status": "refused", "reason": Refusal.OPTIONS.value}
                assert not any(name.startswith("access-control-") for name in response.headers)
        # Everything else is still answered by CORS exactly as before.
        health = client.options("/health", headers=preflight)
        assert health.status_code == 200 and health.headers["access-control-allow-origin"] == origin
        assert client.get("/health").status_code == 200
        assert client.get("/health", headers={"Host": "anything"}).status_code == 200


def test_health_and_other_routes_are_unaffected_by_the_guards(app) -> None:  # type: ignore[no-untyped-def]
    with TestClient(app, base_url="http://testserver") as client:
        # A Host the approval guards would refuse still reaches every other route.
        assert client.get("/health").status_code == 200
        assert client.get("/api/contexts").status_code == 200  # no auth dependency any more
        assert client.get("/api/r18/approvals").status_code == 403  # the guard answers


# ============================================ accepted transports and flow


def test_both_transports_are_accepted_and_the_flow_claims_once(app, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with _client(app) as client:
        state = client.app.state
        owner = state.approval_owner
        _fake_timers(state)
        gid, intent, forecast = _allowed(state)
        target = Path(state.settings.r18.target_path).read_bytes()
        base = f"/api/r18/approvals/{_id(intent)}"
        # list / status: ledger facts only, both transports.
        for headers in (_local(), _lan()):
            listed = client.get("/api/r18/approvals", headers=headers)
            assert listed.status_code == 200
            (item,) = listed.json()["approvals"]
            assert item["intent_id"] == intent.intent_id and item["ledger_state"] == "INTENT_ISSUED"
            assert item["attempts_used"] == 0 and item["live_challenge"] is False
            assert item["max_attempts"] == 3
            assert "nonce" not in json.dumps(listed.json()) and "code" not in item
        status = client.get(base + "/status", headers=_lan())
        assert status.status_code == 200 and status.json()["approvals"][0]["intent_id"] == intent.intent_id
        before = _snapshot(state)
        # review: nothing is written and no event is submitted; the same challenge twice.
        first = client.post(base + "/review", headers=_local(post=True), content="{}")
        assert first.status_code == 200, first.text
        shown = first.json()
        again = client.post(base + "/review", headers=_lan(post=True), content="{}")
        assert again.status_code == 200 and again.json()["challenge"] == shown["challenge"]
        assert again.json()["source"] == "review_pending_live" and shown["source"] == "review_built"
        assert _events(state) == before[1] and len(ip10._history_transactions(state)) == before[0]
        challenge = ApprovalChallengeV1.from_json(shown["challenge"].encode())
        assert shown["challenge_digest"] == challenge.challenge_digest
        assert shown["forecast_digest"] == forecast.forecast_digest == challenge.expectation_digest
        assert shown["attempt"] == 1 and shown["max_attempts"] == 3
        assert shown["window_end_utc"] == intent.approval_window_end_utc
        from suzka.r18.contracts import ActionIntentV1

        assert ActionIntentV1.from_json(shown["intent"].encode()) == intent
        assert "confirmation_code" not in json.dumps(shown) and challenge.confirmation_code not in json.dumps(shown)
        # confirm: issue + CLAIM, from the LAN transport.
        confirm = client.post(
            base + "/confirm",
            headers=_lan(post=True),
            content=json.dumps({"nonce": challenge.nonce, "code": challenge.confirmation_code}),
        )
        assert confirm.status_code == 200, confirm.text
        done = confirm.json()
        (claim,) = _adopted(state, ExecutionClaimV1)
        assert done == {"status": "claimed", "intent_id": intent.intent_id, "claim_id": claim.claim_id, "attempt": 1}
        # N3: a second confirm after the CLAIM is refused; one claim.
        second = client.post(
            base + "/confirm",
            headers=_local(post=True),
            content=json.dumps({"nonce": challenge.nonce, "code": challenge.confirmation_code}),
        )
        assert second.status_code == 409 and second.json()["status"] == "hold"
        assert len(_adopted(state, ExecutionClaimV1)) == 1
        # The ledger facts now say CLAIMED, and nothing executed: the target is unchanged.
        owner._facts_cache.clear()
        assert client.get(base + "/status", headers=_local()).json()["approvals"][0]["ledger_state"] == "CLAIMED"
        assert client.get("/api/r18/approvals", headers=_local()).json()["approvals"] == []
        assert Path(state.settings.r18.target_path).read_bytes() == target
        assert ip16b._preview(state, gid).row.value == "2"
        _no_failed_events(state)


def test_unknown_ids_bad_bodies_and_closed_reasons(app) -> None:  # type: ignore[no-untyped-def]
    with _client(app) as client:
        state = client.app.state
        _fake_timers(state)
        _gid, intent, _forecast = _allowed(state)
        base = f"/api/r18/approvals/{_id(intent)}"
        for bad in ("intent-x", "inverse-intent-" + "0" * 64, "intent-" + "G" * 64, "x" * 300):
            assert client.post(f"/api/r18/approvals/{bad}/review", headers=_local(post=True), content="{}").status_code == 404
            assert client.get(f"/api/r18/approvals/{bad}/status", headers=_local()).status_code == 404
        for content in ("", "[]", "{}", '{"nonce":"a"}', '{"nonce":1,"code":"x"}', '{"nonce":"a","code":"b","approved":true}', "not json", '{"nonce":"' + "a" * 200 + '","code":"b"}'):
            response = client.post(base + "/confirm", headers=_local(post=True), content=content)
            assert response.status_code == 400 and response.json() == {"status": "bad_request"}
        # An unreviewed Intent with no known challenge: UNKNOWN_CHALLENGE, nothing written.
        response = client.post(base + "/confirm", headers=_local(post=True), content=json.dumps({"nonce": "0" * 64, "code": "AAAAAAAA"}))
        assert response.status_code == 409 and response.json() == {"status": "hold", "reason": "unknown_challenge"}
        # No route accepts an "approved" field (N1): there is no such route.
        for path in (base + "/approve", base, "/api/r18/approvals/approve", "/api/r18/approve/confirm"):
            assert client.post(path, headers=_local(post=True), content='{"approved": true}').status_code in (404, 405)
        # A review of an unknown Intent is a closed HOLD reason.
        unknown = client.post("/api/r18/approvals/intent-" + "0" * 64 + "/review", headers=_local(post=True), content="{}")
        assert unknown.status_code == 409 and unknown.json() == {"status": "hold", "reason": "intent_unknown"}


# ================================================================ limits


def test_review_rate_limit_and_the_global_in_flight_event(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _settings(tmp_path / "limits", approval_reviews_per_minute=2)
    with _client(_app(settings)) as client:
        state = client.app.state
        _fake_timers(state)
        _gid, intent, _forecast = _allowed(state)
        base = f"/api/r18/approvals/{_id(intent)}"
        codes = [client.post(base + "/review", headers=_local(post=True), content="{}").status_code for _ in range(4)]
        assert codes == [200, 200, 429, 429]
        review = state.approval_owner.review(intent.intent_id)
        assert review.challenge is not None
        before = _snapshot(state)
        confirm = json.dumps({"nonce": review.challenge.nonce, "code": review.challenge.confirmation_code})
        # One approval event in flight in total: while one runs, a further confirm is a 429.
        limits = state.approval_limits
        assert limits.event_lock.acquire(blocking=False)
        try:
            busy = client.post(base + "/confirm", headers=_local(post=True), content=confirm)
        finally:
            limits.event_lock.release()
        assert busy.status_code == 429
        assert busy.json() == {"status": "refused", "reason": Refusal.RATE.value}
        assert _snapshot(state) == before
        # Free again: the confirm goes through.
        assert client.post(base + "/confirm", headers=_local(post=True), content=confirm).status_code == 200


def test_the_global_request_limit_and_the_confirm_limit(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(approval_api, "MAX_DATA_REQUESTS_PER_MINUTE", 5)
    monkeypatch.setattr(approval_api, "MAX_CONFIRMS_PER_MINUTE", 2)
    with _client(_app(_settings(tmp_path / "hard"))) as client:
        state = client.app.state
        _gid, intent, _forecast = _allowed(state)
        base = f"/api/r18/approvals/{_id(intent)}"
        bad = json.dumps({"nonce": "0" * 64, "code": "AAAAAAAA"})
        statuses = [client.post(base + "/confirm", headers=_local(post=True), content=bad).status_code for _ in range(4)]
        assert statuses == [409, 409, 429, 429]
        # The global window: five accepted data requests per minute; page loads and refusals do not count.
        for _ in range(10):
            client.get("/api/r18/approvals", headers=_local())
        assert client.get("/api/r18/approvals", headers=_local()).status_code == 429
        assert client.get("/api/r18/approve").status_code in (200, 429)
        assert client.get("/health").status_code == 200  # /health is untouched


def test_the_limit_windows_slide() -> None:
    clock = [0.0]
    window = approval_api._Window(2, lambda: clock[0])
    assert [window.allow() for _ in range(3)] == [True, True, False]
    clock[0] = 59.0
    assert window.allow() is False
    clock[0] = 60.5
    assert window.allow() is True


# ============================================ N1: architecture (§1, §10)


def _imports(path: Path) -> set[str]:
    import ast

    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _module_path(name: str) -> Path | None:
    candidate = ROOT.joinpath(*name.split("."))
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def _closure(start: str) -> set[str]:
    seen: set[str] = set()
    todo = [start]
    while todo:
        name = todo.pop()
        if name in seen or not name.startswith("suzka"):
            continue
        path = _module_path(name)
        if path is None:
            continue
        seen.add(name)
        todo.extend(_imports(path))
    return seen


def test_n1_only_the_approval_api_calls_the_challenge_and_claim_submit_methods() -> None:
    root = ROOT / "suzka"
    private = re.compile(r"\b_submit_(issue|claim)\b|\b_issue_event\b")
    public = re.compile(r"\.(confirm|claim|issue)\(")
    types = re.compile(r"AgentEventType\.R18_(CHALLENGE_ISSUE|EXECUTION_CLAIM)")
    source = re.compile(r"AgentEventSource\.API_R18_APPROVAL")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(root).as_posix()
        if name != "r18/approval_owner.py":
            assert not private.search(text), f"{name} calls a private approval submit"
            assert not source.search(text), f"{name} names the approval source"
        if name not in {"r18/approval_owner.py", "api/server.py", "runtime/event_journal.py", "runtime/agent_runtime.py"}:
            assert not types.search(text), f"{name} names an approval event type"
        # The owner's confirm / claim / issue are called by approval_api only.
        if name not in {"r18/approval_owner.py", "r18/approval_api.py"} and "approval_owner" in text:
            assert not public.search(text) or name in {"r16/decision_issuer.py"}, name
    api_text = (root / "r18" / "approval_api.py").read_text(encoding="utf-8")
    assert "owner.confirm" in api_text and "_submit_" not in api_text
    server = (root / "api" / "server.py").read_text(encoding="utf-8")
    assert "owner.confirm" not in server and ".claim(" not in server and ".issue(" not in server
    assert "approval_owner.review" not in server


def test_n1_no_other_module_names_the_approval_owner() -> None:
    """The driver, chat, models and every other route never import or reach the
    Approval owner or the approval API: server.py wires them and nothing else."""

    root = ROOT / "suzka"
    allowed = {
        "r18/approval_owner.py",
        "r18/approval_api.py",
        "api/server.py",
        "r18/__init__.py",
    }
    for path in root.rglob("*.py"):
        name = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if name in allowed:
            continue
        assert "approval_owner" not in text and "approval_api" not in text, name
        assert "ApprovalChallengeOwnerV1" not in text, name


def test_n1_model_and_chat_paths_cannot_reach_the_approval_api() -> None:
    forbidden = {"suzka.r18.approval_api", "suzka.r18.approval_owner", "suzka.api.server"}
    for start in (
        "suzka.api.routes.chat",
        "suzka.api.routes.debug",
        "suzka.api.routes.memory",
        "suzka.api.routes.contexts",
        "suzka.api.routes.values",
        "suzka.api.routes.sleep",
        "suzka.api.routes.adapters",
        "suzka.api.routes.r16",
        "suzka.r16.driver",
        "suzka.models",
    ):
        reached = _closure(start)
        assert reached, start
        assert not (reached & forbidden), (start, sorted(reached & forbidden))
    for route in (ROOT / "suzka" / "api" / "routes").glob("*.py"):
        assert "/api/r18" not in route.read_text(encoding="utf-8"), route
    # The frontend's only way in is the admin proxy; its allowlist has no /api/r18.
    for path in (ROOT / "suzka").rglob("*.py"):
        if path.name not in {"approval_api.py", "server.py"}:
            assert "/api/r18" not in path.read_text(encoding="utf-8") or path.name in {
                "approval_owner.py",
                "lab_prepare.py",
            }, path


def test_the_executor_boundary_no_target_writer_is_importable_from_17() -> None:
    """§7: IP-17 has no TargetWriter, no rotate_runtime_instance, no writable
    target: the approval modules import only the read-only target API."""

    writers = {"TargetStore", "TargetWriter", "rotate_runtime_instance", "target_init", "initialize"}
    for name in ("approval_owner", "approval_api", "policy_owner", "requalify", "admission"):
        path = ROOT / "suzka" / "r18" / f"{name}.py"
        text = path.read_text(encoding="utf-8")
        for writer in writers:
            assert not re.search(rf"\b{writer}\b", text), (name, writer)
        for module in _imports(path):
            assert module != "suzka.r18.target_init", name
    api_text = (ROOT / "suzka" / "r18" / "approval_api.py").read_text(encoding="utf-8")
    assert "suzka.r18.target" not in api_text


def test_the_guards_are_ipv6_and_ipv4_network_aware() -> None:
    config = GuardConfig.from_settings(R18Settings(approval_origins=[LAN], approval_trusted_networks=["fd00::/8", "10.0.0.0/8"]))
    headers = lambda value: _h("10.12.0.2:8080", **{"x-suzka-r18": "1", "x-forwarded-for": value})  # noqa: E731
    assert check_request(config, "GET", "/api/r18/approvals", headers("fd00::5")) is None
    assert check_request(config, "GET", "/api/r18/approvals", headers("10.1.2.3")) is None
    assert check_request(config, "GET", "/api/r18/approvals", headers("fe80::5")) is Refusal.NETWORK
    assert ipaddress.ip_address("10.1.2.3") in ipaddress.ip_network("10.0.0.0/8")
