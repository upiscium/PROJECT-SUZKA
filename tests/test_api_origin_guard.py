"""Origin / Host guard for /api outside /api/r18 (CSRF and DNS rebinding).

A refusal asserts the route handler was NOT called (a spy on every route's
callable), so no side effect can have happened.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from suzka.api import origin_guard
from suzka.api.origin_guard import (
    ApiOriginGuard,
    OriginGuardConfig,
    Refusal,
    check_request,
    is_guarded_path,
    parse_authority,
)
from suzka.api.server import create_app
from suzka.config.schema import ApiSettings

from test_fastapi_backend import _settings

ROOT = Path(__file__).resolve().parents[1]
R18 = "/api/r18"
LAN_HOST = "10.12.0.2:8080"
LAN_ORIGIN = f"http://{LAN_HOST}"
SAFE = {"GET", "HEAD", "OPTIONS"}
FORMER_TOKEN_ROUTES = [
    ("POST", "/api/chat/debug", {"message": "hi", "attachments": [], "debug": True}),
    ("POST", "/api/sleep/run", None),
    ("POST", "/api/adapters/a/approve", None),
    ("POST", "/api/adapters/a/activate", None),
    ("POST", "/api/values/v/freeze", None),
    ("POST", "/api/values/v/unfreeze", None),
    ("POST", "/api/values/v/rollback", None),
    ("POST", "/api/values/config-seeds/v/adopt", None),
    ("POST", "/api/contexts/c/suspend", None),
    ("POST", "/api/contexts/c/resume", None),
    ("POST", "/api/contexts/c/close", None),
    ("POST", "/api/r16/driver/activate", None),
    ("POST", "/api/r16/self-goal/activate", None),
]


class Spy:
    """Counts the calls of every API route's handler."""

    def __init__(self, app: Any) -> None:
        self.calls = 0
        for route in app.routes:
            if isinstance(route, APIRoute):
                original = route.dependant.call

                def wrapped(*args: Any, __orig: Any = original, **kwargs: Any) -> Any:
                    self.calls += 1
                    return __orig(*args, **kwargs)

                route.dependant.call = wrapped  # type: ignore[assignment]


@pytest.fixture
def app(tmp_path: Path):  # type: ignore[no-untyped-def]
    app = create_app(_settings(tmp_path / "guard"))
    with TestClient(app) as client:
        app.state.guard_test_client = client
        yield app


def _send(app, method: str, path: str, host: str, headers=None, body=None):  # type: ignore[no-untyped-def]
    # One lifespan per app: the Host is set per request.
    client = app.state.guard_test_client
    return client.request(method, path, headers={**(headers or {}), "Host": host}, json=body)


def _refused(response, code: Refusal) -> None:  # type: ignore[no-untyped-def]
    assert response.status_code == 403
    assert response.json() == {"status": "refused", "reason": code.value}
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("method,path,body", FORMER_TOKEN_ROUTES)
@pytest.mark.parametrize(
    "headers,code",
    [
        ({"Origin": "https://evil.example"}, Refusal.ORIGIN),
        ({"Origin": "null"}, Refusal.ORIGIN),
        ({"Sec-Fetch-Site": "cross-site"}, Refusal.FETCH_SITE),
        ({"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}, Refusal.ORIGIN),
    ],
)
def test_cross_site_post_is_refused_without_a_side_effect(app, method, path, body, headers, code) -> None:  # type: ignore[no-untyped-def]
    spy = Spy(app)
    response = _send(app, method, path, LAN_HOST, headers, body)
    _refused(response, code)
    assert spy.calls == 0


@pytest.mark.parametrize("method,path,body", FORMER_TOKEN_ROUTES)
def test_rebinding_host_is_refused_for_post(app, method, path, body) -> None:  # type: ignore[no-untyped-def]
    spy = Spy(app)
    _refused(_send(app, method, path, "evil.example:8080", {}, body), Refusal.HOST)
    _refused(
        _send(app, method, path, "evil.example", {"Origin": "http://evil.example"}, body),
        Refusal.HOST,
    )
    assert spy.calls == 0


@pytest.mark.parametrize(
    "path",
    ["/api/state/emotion", "/api/memory/search?query=x", "/api/adapters", "/api/values", "/api/contexts", "/api/chat/sessions"],
)
def test_rebinding_host_and_foreign_origin_are_refused_for_get(app, path) -> None:  # type: ignore[no-untyped-def]
    spy = Spy(app)
    _refused(_send(app, "GET", path, "evil.example"), Refusal.HOST)
    _refused(
        _send(app, "GET", path, LAN_HOST, {"Origin": "https://evil.example"}),
        Refusal.ORIGIN,
    )
    assert spy.calls == 0


def test_every_non_safe_api_route_is_refused_cross_origin(app) -> None:  # type: ignore[no-untyped-def]
    spy = Spy(app)
    seen = 0
    for route in app.routes:
        if not isinstance(route, APIRoute) or not is_guarded_path(route.path, R18):
            continue
        for method in sorted(route.methods - SAFE):
            path = re.sub(r"\{[^}]+\}", "x", route.path)
            response = _send(app, method, path, LAN_HOST, {"Origin": "https://evil.example"})
            _refused(response, Refusal.ORIGIN)
            seen += 1
    assert seen >= len(FORMER_TOKEN_ROUTES) and spy.calls == 0


def test_same_origin_post_and_caddy_forwarded_pattern_reach_the_handler(app) -> None:  # type: ignore[no-untyped-def]
    spy = Spy(app)
    caddy = {
        "Origin": LAN_ORIGIN,
        "X-Forwarded-For": "10.12.0.9",
        "X-Forwarded-Proto": "http",
        "X-Forwarded-Host": LAN_HOST,
        "Sec-Fetch-Site": "same-origin",
    }
    assert _send(app, "POST", "/api/sleep/run", LAN_HOST, caddy).status_code == 200
    assert spy.calls == 1
    for host in ("localhost:8080", "127.0.0.1:8080", "[::1]:8080"):
        headers = {"Origin": f"http://{host}", "Sec-Fetch-Site": "same-origin"}
        assert _send(app, "POST", "/api/contexts/x/suspend", host, headers).status_code != 403
    assert _send(app, "GET", "/api/state/emotion", LAN_HOST, {"Origin": LAN_ORIGIN}).status_code == 200


def test_requests_without_origin_are_allowed_for_every_method(app) -> None:  # type: ignore[no-untyped-def]
    spy = Spy(app)
    # curl / smoke script / Next.js admin proxy: no Origin, loopback Host.
    assert _send(app, "POST", "/api/sleep/run", "127.0.0.1:8000").status_code == 200
    assert _send(app, "GET", "/api/state/emotion", "127.0.0.1:8000").status_code == 200
    assert spy.calls == 2
    # TestClient default Host (no Host override).
    client = app.state.guard_test_client
    assert client.get("/api/state/emotion").status_code == 200
    assert client.post("/api/contexts/x/suspend").status_code != 403


def test_health_and_non_api_paths_are_untouched(app) -> None:  # type: ignore[no-untyped-def]
    assert _send(app, "GET", "/health", "evil.example").status_code == 200


def test_the_guard_adds_no_cors_headers_and_a_dev_cors_origin_still_works(app) -> None:  # type: ignore[no-untyped-def]
    response = _send(app, "GET", "/api/state/emotion", LAN_HOST)
    assert "access-control-allow-origin" not in response.headers
    # api.cors_origins (the dev frontend) stays usable cross-origin.
    dev = {"Origin": "http://localhost:3000"}
    ok = _send(app, "GET", "/api/state/emotion", "127.0.0.1:8000", dev)
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    preflight = _send(
        app,
        "OPTIONS",
        "/api/sleep/run",
        "127.0.0.1:8000",
        {**dev, "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 200
    evil = _send(
        app,
        "OPTIONS",
        "/api/sleep/run",
        "127.0.0.1:8000",
        {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    _refused(evil, Refusal.ORIGIN)


def test_r18_paths_are_left_to_their_own_guard(app) -> None:  # type: ignore[no-untyped-def]
    # r18 is disabled in this app: 404, never this guard's 403.
    assert _send(app, "POST", "/api/r18/approvals/x/confirm", "evil.example").status_code == 404
    assert not is_guarded_path("/api/r18", R18) and not is_guarded_path("/api/r18/approve", R18)
    assert is_guarded_path("/api", R18) and is_guarded_path("/api/chat", R18)
    assert is_guarded_path("/api/r18x", R18)
    assert not is_guarded_path("/health", R18) and not is_guarded_path("/apix", R18)


def test_configured_allowed_hosts_and_origins(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "cfg")
    api = settings.api.model_copy(
        update={
            "allowed_hosts": ["suzka.lan"],
            "allowed_origins": ["https://ui.suzka.lan"],
        }
    )
    built = create_app(settings.model_copy(update={"api": api}))
    with TestClient(built) as client:
        built.state.guard_test_client = client
        _configured(built)


def _configured(app) -> None:  # type: ignore[no-untyped-def]
    assert _send(app, "GET", "/api/state/emotion", "suzka.lan:8080").status_code == 200
    ok = _send(app, "POST", "/api/sleep/run", "suzka.lan", {"Origin": "https://ui.suzka.lan"})
    assert ok.status_code == 200
    _refused(
        _send(app, "POST", "/api/sleep/run", "suzka.lan", {"Origin": "https://other.lan"}),
        Refusal.ORIGIN,
    )
    # a listed origin does not make its host an acceptable Host
    _refused(_send(app, "GET", "/api/state/emotion", "ui.suzka.lan"), Refusal.HOST)


def test_config_validation_and_defaults() -> None:
    base = {"host": "127.0.0.1", "port": 8000, "cors_origins": []}
    api = ApiSettings(**base)
    assert api.allowed_hosts == [] and api.allowed_origins == []
    for bad_host in ("EVIL.example", "evil.example:80", "*", "a..b", "evil.example.", "user@evil", ""):
        with pytest.raises(ValidationError):
            ApiSettings(**base, allowed_hosts=[bad_host])
    for bad_origin in ("*", "null", "http://evil.example/", "ftp://x", "http://u@x", "HTTP://x"):
        with pytest.raises(ValidationError):
            ApiSettings(**base, allowed_origins=[bad_origin])


CONFIG = OriginGuardConfig(frozenset({"localhost", "testserver", "suzka.lan"}), frozenset({"http://localhost:3000"}))


def _check(method: str, **headers: str) -> Refusal | None:
    raw = [(k.replace("_", "-").encode(), v.encode("utf-8")) for k, v in headers.items()]
    return check_request(CONFIG, method, raw)


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8080",
        "localhost.",
        "localhost.evil.example",
        "10.12.0.2.evil.example",
        "user@localhost",
        "localhost@evil.example",
        "localhost:8080@evil.example",
        "localhost/",
        "localhost:0",
        "localhost:65536",
        "localhost:08080",
        "localhost:",
        " localhost",
        "localhost ",
        "0x7f.1",
        "010.0.0.1",
        "256.1.1.1",
        "[::1",
        "[::1]x",
        "[fe80::1%eth0]",
        "[evil]",
        "lоcalhost",  # Cyrillic o
        "",
    ],
)
def test_host_parsing_refuses_tricks(host: str) -> None:
    assert _check("GET", host=host) == Refusal.HOST


@pytest.mark.parametrize(
    "host",
    ["localhost", "LocalHost:8080", "127.0.0.1", "10.12.0.2:8080", "[::1]:8000", "[0:0:0:0:0:0:0:1]", "testserver", "suzka.lan:9", "SUZKA.LAN"],
)
def test_host_parsing_accepts_authorities(host: str) -> None:
    assert _check("GET", host=host) is None


def test_missing_or_duplicate_host_is_refused() -> None:
    assert check_request(CONFIG, "GET", []) == Refusal.HOST
    assert check_request(CONFIG, "GET", [(b"host", b"localhost"), (b"host", b"localhost")]) == Refusal.HOST


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "",
        "http://evil.example",
        "http://localhost:3000.evil.example",
        "http://localhost:8081",  # other port: another origin
        "https://localhost:8080@evil.example",
        "http://user@localhost:8080",
        "http://localhost:8080/",
        "http://localhost:8080/x",
        "ftp://localhost:8080",
        "chrome-extension://localhost:8080",
        "HTTP://localhost:8080",
        "http://localhost.:8080",
        "http://LOCALHOST:8080 ",
        "localhost:8080",
    ],
)
def test_origin_parsing_refuses_tricks(origin: str) -> None:
    assert _check("POST", host="localhost:8080", origin=origin) == Refusal.ORIGIN
    assert _check("GET", host="localhost:8080", origin=origin) == Refusal.ORIGIN


def test_origin_accepts_same_authority_and_listed_origins() -> None:
    assert _check("POST", host="localhost:8080", origin="http://localhost:8080") is None
    assert _check("POST", host="localhost:8080", origin="https://localhost:8080") is None
    assert _check("POST", host="[::1]:8080", origin="http://[0:0::1]:8080") is None
    assert _check("POST", host="127.0.0.1:8000", origin="http://localhost:3000") is None


def test_duplicate_origin_and_forwarded_host_are_not_trusted() -> None:
    raw = [(b"host", b"localhost"), (b"origin", b"http://localhost"), (b"origin", b"http://localhost")]
    assert check_request(CONFIG, "POST", raw) == Refusal.ORIGIN
    # X-Forwarded-Host can neither grant nor deny
    assert _check("GET", host="evil.example", x_forwarded_host="localhost") == Refusal.HOST
    assert _check("GET", host="localhost", x_forwarded_host="evil.example") is None
    assert _check("POST", host="localhost", origin="http://evil.example", x_forwarded_host="evil.example") == Refusal.ORIGIN


def test_fetch_metadata_only_gates_non_safe_methods() -> None:
    assert _check("POST", host="localhost", sec_fetch_site="cross-site") == Refusal.FETCH_SITE
    assert _check("DELETE", host="localhost", sec_fetch_site="Cross-Site") == Refusal.FETCH_SITE
    assert _check("POST", host="localhost", sec_fetch_site="same-origin") is None
    assert _check("POST", host="localhost", sec_fetch_site="none") is None
    assert _check("GET", host="localhost", sec_fetch_site="cross-site") is None


def test_refusal_logging_is_rate_limited_per_class(caplog: pytest.LogCaptureFixture) -> None:
    now = [0.0]
    log = origin_guard._RefusalLog(lambda: now[0])
    with caplog.at_level("WARNING", logger=origin_guard.LOGGER.name):
        for _ in range(5):
            log.note(Refusal.HOST)
        log.note(Refusal.ORIGIN)
        now[0] = 61.0
        log.note(Refusal.HOST)
    assert [r.getMessage() for r in caplog.records] == [
        "api request refused: api_host_not_accepted",
        "api request refused: api_origin_not_accepted",
        "api request refused: api_host_not_accepted",
    ]


def test_refusal_is_closed_and_never_echoes_headers(app) -> None:  # type: ignore[no-untyped-def]
    response = _send(app, "POST", "/api/sleep/run", LAN_HOST, {"Origin": "https://canary-evil.example"})
    assert "canary" not in response.text and "canary" not in str(response.headers)


def test_unexpected_guard_error_fails_closed(monkeypatch: pytest.MonkeyPatch, app) -> None:  # type: ignore[no-untyped-def]
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    spy = Spy(app)
    monkeypatch.setattr(origin_guard, "check_request", boom)
    response = _send(app, "POST", "/api/sleep/run", LAN_HOST)
    assert response.status_code == 403 and spy.calls == 0


def test_parse_authority_is_total() -> None:
    for value in ("\x00", "a" * 1000, "[" * 5, "::1", "1.2.3.4:5:6", "\n"):
        assert parse_authority(value, frozenset()) is None


# ------------------------------------------------------- architecture guard


def test_every_api_route_is_covered_by_this_guard_or_its_own_guard(app) -> None:  # type: ignore[no-untyped-def]
    allowed_outside_api = {"/health", "/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"}
    for route in app.routes:
        path = getattr(route, "path", "")
        assert (
            path.startswith("/api/") or path in allowed_outside_api
        ), f"route {path} is neither under /api nor a known non-API route"
        if path.startswith("/api/"):
            assert is_guarded_path(path, R18) or path.startswith("/api/r18/"), path
    # the middleware is installed on the app, outside CORS, and the r18 guard
    # (own guard) is the only other thing that may own an /api prefix
    classes = [m.cls.__name__ for m in app.user_middleware]
    assert classes.index("ApiOriginGuard") < classes.index("CORSMiddleware")
    assert app.user_middleware[0].cls is ApiOriginGuard


def test_every_route_module_is_mounted_under_the_api_prefix() -> None:
    for path in (ROOT / "suzka" / "api" / "routes").glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text()
        assert re.search(r'APIRouter\(\s*prefix="/api', text), path.name


def test_r18_mount_keeps_its_guard_outermost_and_this_guard_inside(tmp_path: Path) -> None:
    from test_r18_ip17c_api import _settings as r18_settings

    app = create_app(r18_settings(tmp_path / "r18"))
    names = [m.cls.__name__ for m in app.user_middleware]
    assert names[0] == "R18RequestGuard"
    assert names.index("ApiOriginGuard") < names.index("CORSMiddleware")


def test_no_websocket_routes_and_the_guard_closes_a_websocket_handshake(app) -> None:  # type: ignore[no-untyped-def]
    from fastapi.routing import APIWebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    assert not [r for r in app.routes if isinstance(r, APIWebSocketRoute)]
    client = app.state.guard_test_client
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/api/ws", headers={"Origin": "https://evil.example"}):
            pass
    assert closed.value.code == 1008


@pytest.mark.parametrize("prefix", ["", "/", "/api", "/api/", "/api/x/", "/other", "api/x"])
def test_a_malformed_exempt_prefix_is_refused_at_construction(prefix: str) -> None:
    async def noop(*args: Any) -> None:  # pragma: no cover
        return None

    with pytest.raises(ValueError):
        ApiOriginGuard(noop, CONFIG, prefix)


def test_scheme_and_port_edge_cases() -> None:
    assert _check("GET", host="10.0.0.1:65536") == Refusal.HOST
    assert _check("GET", host="[::1]:0") == Refusal.HOST
    assert _check("POST", host="suzka.lan:443", origin="https://suzka.lan") == Refusal.ORIGIN
    wild = OriginGuardConfig.from_settings(
        ApiSettings(host="127.0.0.1", port=8000, cors_origins=["*"])
    )
    assert wild.origins == frozenset()
