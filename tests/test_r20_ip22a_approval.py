"""IP-22a: the approval operator stand-in (D1 sec. 4-5, sec. 8-2, F05 guards).

The stand-in is a client of the REAL ``/api/r18`` route of a factory subject
(profile S0, T1): the real guards run unmodified, the typed code is recomputed
independently, and the guards still refuse every deviation, with and without
the ``Sec-Fetch-Site: same-origin`` header browsers send.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Any

import pytest

from experiments.r20 import approval
from experiments.r20.approval import ApprovalOperator, canon, code_from, recompute
from experiments.r20.client import AsgiTransport, HttpClient, HttpResult
from experiments.r20.factory import create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.provider import ApprovalActor, ProviderMode, ScriptedProvider
from experiments.r20.tier1 import T1Runtime

import test_r16_ip10_decision_issuer as ip10
import test_r18_ip17b_approval as ip17b
from suzka.r18 import contracts
from suzka.r18.contracts import ApprovalChallengeV1, ExecutionClaimV1

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------- the recomputation


def test_the_domains_and_the_code_length_are_the_pages_and_the_servers() -> None:
    text = (ROOT / "suzka" / "r18" / "approval_page" / "approve.js").read_text(encoding="utf-8")
    for name in ("CHALLENGE_DOMAIN", "CODE_DOMAIN", "INTENT_DOMAIN", "ARTIFACT_DOMAIN"):
        match = re.search(rf'var {name} = "([^"]*)";', text)
        assert match is not None
        page = match.group(1).replace("\\0", "\0")
        assert page == getattr(approval, name)
        assert page.encode("ascii") == getattr(contracts, f"_{name}")
    assert f"var CODE_LENGTH = {approval.CODE_LENGTH};" in text
    assert approval.CODE_LENGTH == contracts.CONFIRMATION_CODE_LENGTH


def _js_code_from(raw: bytes) -> str:
    """A literal port of the page's ``codeFrom`` bit loop (the oracle)."""

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    bits = acc = 0
    out = ""
    for index in range(5):
        acc = (acc << 8) | raw[index]
        bits += 8
        while bits >= 5 and len(out) < 8:
            out += alphabet[(acc >> (bits - 5)) & 31]
            bits -= 5
        acc &= (1 << bits) - 1
    return out


def test_the_code_port_matches_the_pages_bit_loop() -> None:
    assert code_from(bytes(5)) == "AAAAAAAA" and code_from(b"\xff" * 5) == "77777777"
    rng = random.Random(22)
    for _ in range(200):
        raw = bytes(rng.randrange(256) for _ in range(32))
        assert code_from(raw) == _js_code_from(raw)


def test_the_canonical_json_is_the_pages_and_refuses_what_it_refuses() -> None:
    assert canon({"b": [1, None, True], "a": "x"}) == '{"a":"x","b":[1,null,true]}'
    for bad in (1.5, 2**53, object()):
        with pytest.raises(ValueError):
            canon(bad)


# --------------------------------------------------------------- a real flow


def _world(base: Path, *, same_origin: bool):  # type: ignore[no-untyped-def]
    subject = create_subject("S0", base=base)
    runtime = T1Runtime(subject, ScriptedProvider(), executor=False, scheduler=False)
    runtime.start()
    state = runtime.state
    ip17b._fake_timers(state)
    gid, intent, forecast = ip17b._allowed(state)
    client = HttpClient(AsgiTransport(runtime.client), subject.port)
    operator = ApprovalOperator(client, subject.port, same_origin_fetch=same_origin)
    return runtime, state, intent, operator


def _snapshot(state: Any) -> tuple[int, int, int]:
    return (
        len(ip10._history_transactions(state)),
        len(ip17b._adopted(state, ExecutionClaimV1)),
        len(state.approval_owner._pending_challenges),
    )


@pytest.mark.parametrize("same_origin", [False, True])
def test_the_stand_in_approves_through_the_real_route_and_every_guard_still_refuses(
    r20_base: Path, same_origin: bool
) -> None:
    runtime, state, intent, operator = _world(r20_base, same_origin=same_origin)
    try:
        target = Path(state.settings.r18.target_path).read_bytes()
        listed = operator.listing()
        assert listed.status == 200 and listed.json()["approvals"][0]["intent_id"] == intent.intent_id

        # the independently recomputed values equal the server's, for a real challenge
        shown = operator.review(intent.intent_id)
        assert shown.status == 200
        reply = shown.json()
        checked = recompute(reply)
        assert checked.ok, checked.problems
        server_side = ApprovalChallengeV1.from_json(reply["challenge"].encode())
        assert checked.code == server_side.confirmation_code
        assert checked.challenge_digest == server_side.challenge_digest == reply["challenge_digest"]
        assert checked.nonce == server_side.nonce
        before = _snapshot(state)

        # every deviation is refused by the REAL guard, and nothing changes
        good = {"nonce": checked.nonce, "code": checked.code}
        deviations: list[tuple[str, dict[str, str | None]]] = [
            ("wrong Origin", {"Origin": "http://evil.example"}),
            ("another port as Origin", {"Origin": "http://localhost:9"}),
            ("forwarding header", {"X-Forwarded-For": "10.12.5.5"}),
            ("forwarded host", {"Forwarded": "for=10.12.5.5"}),
            ("missing X-SUZKA-R18", {"X-SUZKA-R18": None}),
            ("missing Origin on a POST", {"Origin": None}),
            ("cross-site fetch metadata", {"Sec-Fetch-Site": "cross-site"}),
            ("same-site fetch metadata", {"Sec-Fetch-Site": "same-site"}),
            ("wrong Host", {"Host": "evil.example"}),
        ]
        for label, change in deviations:
            result = operator.confirm(intent.intent_id, good["nonce"], good["code"], **change)
            assert result.status in (400, 403, 409, 421), (label, result.status)
            assert _snapshot(state) == before, label
        # a wrong code and an unknown nonce are HOLDs, not approvals
        for nonce, code in ((checked.nonce, "AAAAAAAA"), ("f" * 64, checked.code)):
            held = operator.confirm(intent.intent_id, nonce, code)
            assert held.status == 409 and held.json()["status"] == "hold"
        assert _snapshot(state)[1] == 0  # still no claim

        # a recomputation problem aborts WITHOUT sending a confirm
        confirms: list[str] = []
        real_request = operator._client.request

        def spy(method: str, path: str, *args: Any, **kwargs: Any) -> HttpResult:
            if path.endswith("/confirm"):
                confirms.append(path)
            return real_request(method, path, *args, **kwargs)

        tampered = dict(reply, forecast_digest="0" * 64)
        operator._client.request = spy  # type: ignore[method-assign]
        operator.review = lambda _id: HttpResult(200, {}, json.dumps(tampered).encode())  # type: ignore[method-assign]
        with pytest.raises(GuardRefusal) as aborted:
            operator.approve(intent.intent_id)
        assert aborted.value.code == "APPROVAL_RECOMPUTE_MISMATCH" and confirms == []
        del operator.review  # restore the class method

        # the real approval
        done = operator.approve(intent.intent_id)
        assert done.status == 200 and done.json()["status"] == "claimed"
        (claim,) = ip17b._adopted(state, ExecutionClaimV1)
        assert done.json()["claim_id"] == claim.claim_id
        # a replayed nonce is refused; still one claim; nothing executed (no executor here)
        replay = operator.confirm(intent.intent_id, checked.nonce, checked.code)
        assert replay.status == 409 and len(ip17b._adopted(state, ExecutionClaimV1)) == 1
        assert Path(state.settings.r18.target_path).read_bytes() == target
    finally:
        runtime.stop()


def test_a_scripted_operator_is_refused_with_a_real_provider_and_human_never_approves_by_script() -> None:
    client = HttpClient(AsgiTransport(None), 45000)
    with pytest.raises(GuardRefusal) as real:
        ApprovalOperator(client, 45000, provider=ProviderMode.REAL)
    assert real.value.code == "SCRIPTED_APPROVAL_WITH_REAL_PROVIDER"
    with pytest.raises(GuardRefusal) as human:
        ApprovalOperator(client, 45000, actor=ApprovalActor.HUMAN)
    assert human.value.code == "HUMAN_ACTOR_APPROVES_BY_HAND"
    assert human.value.detail == "http://localhost:45000/api/r18/approve"


def test_the_stand_in_sends_what_a_browser_sends() -> None:
    operator = ApprovalOperator(HttpClient(AsgiTransport(None), 45000), 45000)
    post = operator.headers(post=True)
    assert post == {
        "Host": "localhost:45000", "X-SUZKA-R18": "1",
        "Origin": "http://localhost:45000", "Content-Type": "application/json",
    }
    assert operator.headers(post=False) == {"Host": "localhost:45000", "X-SUZKA-R18": "1"}
    variant = ApprovalOperator(HttpClient(AsgiTransport(None), 45000), 45000, same_origin_fetch=True)
    assert variant.headers(post=True)["Sec-Fetch-Site"] == "same-origin"
    assert not any(name.lower().startswith(("x-forwarded", "forwarded")) for name in post)
    assert "Sec-Fetch-Site" not in operator.headers(post=True, Sec_Fetch_Site=None)
