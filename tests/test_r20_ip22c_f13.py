"""IP-22c, F13-c: a canary sweep over a whole S0-shaped run (D1 sec. 4-6, 6).

The admin-token canary is gone with the token layer.  The canaries are:
the obligation secret (hex and the raw file text), an ``interlocutor_key``
the client sends with every turn, and a "hidden thought" marker the scripted
provider emits inside ``<think>``.  The run is the full chain over the real
routes (driver chain, approval by the stand-in, executor effect, Outcome,
Feedback, Wake, Outbox, Inbox); every response goes through the harness
``Recorder``; the server log, the Inbox / Outbox text and the approval page
are swept too.  Controls prove the sweep is not blind: the canaries really
are inside the subject (the secret file, the persisted interlocutor key; the
hidden thought is produced by the provider and stripped before storage) and a
deliberately leaking text is detected.
"""

from __future__ import annotations

import logging
from pathlib import Path
import secrets

import pytest

from experiments.r20.approval import ApprovalOperator
from experiments.r20.client import AsgiTransport, HttpClient
from experiments.r20.factory import create_subject
from experiments.r20.provider import HIDDEN_THOUGHT_MARKER, ScriptedProvider
from experiments.r20.tier1 import T1Runtime
from r20_chain import FAST_R19, Wire, chat, run_full_chain
from r20_fixtures import r20_base  # noqa: F401


def _tree_text(root: Path) -> bytes:
    blobs: list[bytes] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            try:
                blobs.append(path.read_bytes())
            except OSError:
                continue
    return b"\n".join(blobs)


def test_f13_c_no_canary_reaches_any_response_the_inbox_the_page_or_the_log(
    r20_base: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    subject = create_subject("S0", base=r20_base, r19_overrides=FAST_R19)
    assert subject.target_path is not None
    secret_file = subject.target_path.parent / "obligation.key"
    raw_secret = secret_file.read_bytes()
    interlocutor = f"r20-canary-interlocutor-{secrets.token_hex(8)}"
    marker = f"{HIDDEN_THOUGHT_MARKER}-{secrets.token_hex(4)}"
    canaries = {
        "obligation_secret_hex": raw_secret.hex(),
        "obligation_secret_text": raw_secret.decode("latin-1"),
        "interlocutor_key": interlocutor,
        "hidden_thought": marker,
    }
    provider = ScriptedProvider(default=f"<think>{marker}</think>synthetic visible answer (fixture only)")
    with T1Runtime(subject, provider, executor=True, scheduler=True) as runtime:
        wire = Wire(runtime, subject.port, canaries)
        run_full_chain(runtime, subject, client=wire)
        state = runtime.state
        context = state.main_loop.context_registry.current_context_id
        # an untrusted interlocutor key does not authenticate an actor (so it
        # cannot drive the chain): it is planted on its own turns, own session
        for index in range(2):
            turn = chat(wire, f"hello {index}", interlocutor_key=interlocutor, client_session_id="r20-canary-session")
            assert turn["response"]

        # surfaces beyond the chain: Inbox / Outbox text, the context, the page, a listing
        inbox = wire.get(f"/api/r19/inbox/messages?context_id={context}")
        assert inbox.status_code == 200 and inbox.json()["messages"], "an Outbox message exists"
        wire.get(f"/api/contexts/{context}")
        operator = ApprovalOperator(HttpClient(AsgiTransport(runtime.client), subject.port), subject.port)
        listing = operator.listing()
        assert listing.status == 200
        for page in ("/api/r18/approve", "/api/r18/approvals"):
            wire.get(page)

        entries = wire.recorder.entries
        paths = {entry["path"].split("?")[0] for entry in entries}
        assert {"/api/chat", "/api/r16/driver/activate", "/api/r19/inbox/messages"} <= paths
        assert len(entries) >= 8
        assert [entry for entry in entries if entry["canary_hits"]] == []
        # the log, the Inbox text and the operator's own listing
        for label, text in (
            ("server log", caplog.text),
            ("inbox text", inbox.text),
            ("approval listing", listing.text),
        ):
            assert wire.recorder.sweep(label, text) == [], label
        assert caplog.text, "the log sweep saw log lines"

        # controls: the canaries are really inside the subject ...
        stored = _tree_text(subject.root)
        assert raw_secret in stored  # the secret file exists on disk
        assert interlocutor.encode() in stored  # the client's key IS persisted in the subject
        # the hidden thought is stripped before storage (never persisted raw), so
        # its only exposure is the response of the turn that produced it
        assert marker.encode() not in stored and provider.calls >= 5
        assert len([e for e in entries if e.get("path") == "/api/chat"]) >= 5  # the key rode on real turns
        # ... and the sweep is not blind
        for name, value in canaries.items():
            assert name in wire.recorder.sweep(f"control-{name}", f"leak: {value} ."), name
