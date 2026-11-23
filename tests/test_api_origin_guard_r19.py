"""The Origin/Host guard covers the R19 Inbox routes (only /api/r18 is exempt)."""

from __future__ import annotations

from test_r19_ip21c_inbox_api import (  # noqa: F401  (``world`` is a fixture)
    BASE,
    _AckSubmits,
    _ack_body,
    _ack_rows,
    _acks,
    _journal_len,
    _message,
    world,
)

LAN = "10.12.0.2:8080"


def _post(w, message, headers):  # type: ignore[no-untyped-def]
    return w.client.post(
        f"{BASE}/messages/{message.message_id}/ack",
        json=_ack_body(message.context_id),
        headers=headers,
    )


def test_cross_origin_ack_is_refused_with_no_event_and_no_row(world) -> None:  # type: ignore[no-untyped-def]
    message = _message(world)
    submits = _AckSubmits(world.state, world.monkeypatch)
    journal = _journal_len(world.state)
    for extra in (
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
    ):
        response = _post(world, message, {"Host": LAN, **extra})
        assert response.status_code == 403
        assert response.json()["status"] == "refused"
    # A rebinding Host: refused for the POST and for the read.
    assert _post(world, message, {"Host": "evil.example"}).status_code == 403
    read = world.client.get(
        f"{BASE}/messages?context_id={message.context_id}", headers={"Host": "evil.example"}
    )
    assert read.status_code == 403
    assert submits.count == 0
    assert not _ack_rows(world.state) and not _acks(world.state)
    assert _journal_len(world.state) == journal


def test_same_origin_and_no_origin_acks_still_work(world) -> None:  # type: ignore[no-untyped-def]
    first, second = _message(world), _message(world, another=True)
    same = _post(
        world, first, {"Host": LAN, "Origin": f"http://{LAN}", "Sec-Fetch-Site": "same-origin"}
    )
    assert (same.status_code, same.json()["result"]) == (201, "COMMITTED")
    assert _post(world, second, {}).status_code == 201  # no Origin, TestClient Host
    read = world.client.get(
        f"{BASE}/messages?context_id={first.context_id}",
        headers={"Host": LAN, "Origin": f"http://{LAN}"},
    )
    assert read.status_code == 200
