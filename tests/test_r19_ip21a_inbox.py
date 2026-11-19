"""IP-21a: the pure Inbox / Outbox functions (IP-21 D1 r3 §4, §5, §10-1..3).

The closed projection, the derived message state, the ACK pre-check (status
only), the emission plan (dedupe first, record-byte lower bound, TTL) and the
purity of the module. No writer, no route, no store.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest

from suzka.r16.cycle_contracts import (
    ResultKind,
)
from suzka.r16.history_codec import PolicyRow
from suzka.r16.history_reader import AdoptedRecord
from suzka.r16.history_records import HistoryOperationKind, RecordKind
from suzka.r19 import inbox
from suzka.r19.contracts import (
    AckStatus,
    InboxAckV1,
    OutboxMessageV1,
    OutboxReason,
    OutboxTemplate,
    TerminalReason,
    WakeTerminalKind,
    WakeTerminalV1,
)
from suzka.r19.inbox import (
    AckEligibility,
    AckHold,
    MessageState,
    OutboxHold,
    ack_eligibility,
    derive_inbox_view,
    emission_candidates,
    plan_emission,
)
from suzka.r19.ledger import WakeEntry, WakeState, derive_wake_view

from test_r19_ip13b_contracts import EVENT_A, EVENT_T, FP, INC, _wake
from test_r19_ip20a_ledger import (
    LEASE_END,
    _claim,
    _claim_rec,
    _closure_rec,
    _terminal_rec,
    _wake_rec,
)

_K = HistoryOperationKind
CTX = "context-1"
CLOSED_AT = "2026-11-16T02:00:30Z"  # inside the first lease (02:00:00 .. 02:02:00)
L = "2026-11-16T02:00:00Z"  # LEASE_END[0] - 120 s


def _ts(base: str, *, seconds: int = 0, days: int = 0) -> str:
    from datetime import datetime, timedelta, timezone

    value = datetime.strptime(base, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (value + timedelta(seconds=seconds, days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rec(kind: RecordKind, record: object, sequence: int) -> AdoptedRecord:
    op = _K.R19_OUTBOX_EMIT if kind is RecordKind.OUTBOX_MESSAGE else _K.R19_INBOX_ACK
    return AdoptedRecord(op, kind, record, f"tx-{sequence}", EVENT_A, sequence)


def _msg(
    wake_changes: dict[str, Any] | None = None,
    *,
    created: str = CLOSED_AT,
    ttl_days: int = 7,
) -> OutboxMessageV1:
    wake = _wake(**(wake_changes or {}))
    claim = _claim(wake)
    rows = [_wake_rec(wake), _claim_rec(claim), _closure_rec(claim)]
    entry = derive_wake_view(rows, created, clock_floor=None).entries[0]
    plan = plan_emission(entry, created, outbox_ttl_s=ttl_days * 86_400)
    assert plan.message is not None
    return plan.message


def _ack(
    message: OutboxMessageV1,
    status: AckStatus = AckStatus.ACKNOWLEDGED,
    *,
    context: str | None = None,
    selector: str = "context-only",
    at: str = "2026-11-16T03:00:00Z",
) -> InboxAckV1:
    return InboxAckV1(
        message.message_id,
        context or message.context_id,
        selector,
        status,
        EVENT_T,
        90,
        at,
    )


def _mrec(message: OutboxMessageV1, sequence: int = 80) -> AdoptedRecord:
    return _rec(RecordKind.OUTBOX_MESSAGE, message, sequence)


def _arec(ack: InboxAckV1, sequence: int = 90) -> AdoptedRecord:
    return _rec(RecordKind.INBOX_ACK, ack, sequence)


# ------------------------------------------------------------------- texts


def test_template_text_is_a_closed_set_of_five_plain_strings() -> None:
    table = inbox.TEMPLATE_TEXT
    assert set(table) == {
        (OutboxTemplate.DECISION_ISSUED, OutboxReason.PLAN_SELECTED),
        (OutboxTemplate.DECISION_ISSUED, OutboxReason.NO_ACTION),
        (OutboxTemplate.DECISION_CONFIRMED, OutboxReason.CONFIRMED),
        (OutboxTemplate.DECISION_HELD, OutboxReason.HELD),
        (OutboxTemplate.CYCLE_FAILED, OutboxReason.FAILED),
    }
    assert len(set(table.values())) == 5
    for text in table.values():
        assert text.isascii() and text.endswith(".")
        assert not any(mark in text for mark in "{}<>/%$\\")
        assert not any(char.isdigit() for char in text)
        assert "http" not in text and "://" not in text
    with pytest.raises(TypeError):
        table[(OutboxTemplate.DECISION_HELD, OutboxReason.HELD)] = "x"  # type: ignore[index]


def test_every_pair_the_dto_accepts_has_a_text() -> None:
    """The DTO's own pairing table and the text table cannot drift apart."""

    from suzka.r19 import contracts

    accepted = {
        (template, reason)
        for template, reasons in contracts._TEMPLATE_REASONS.items()
        for reason in reasons
    }
    assert accepted == set(inbox.TEMPLATE_TEXT)


# ------------------------------------------------------------------- state


def test_message_state_derivation_and_the_expiry_boundary() -> None:
    message = _msg()
    expires = message.expires_utc
    pending = derive_inbox_view([_mrec(message)], _ts(expires, seconds=-1), context_id=CTX)
    assert [item.state for item in pending.entries] == [MessageState.PENDING]
    expired = derive_inbox_view([_mrec(message)], expires, context_id=CTX)  # now >= expires
    assert [item.state for item in expired.entries] == [MessageState.EXPIRED]
    for status, state in (
        (AckStatus.ACKNOWLEDGED, MessageState.ACKNOWLEDGED),
        (AckStatus.REJECTED, MessageState.REJECTED),
    ):
        rows = [_mrec(message), _arec(_ack(message, status))]
        # The ACK row is immutable: the state stays after the message expires.
        for now in (_ts(expires, seconds=-1), _ts(expires, days=3)):
            view = derive_inbox_view(rows, now, context_id=CTX)
            assert [item.state for item in view.entries] == [state]


def test_the_view_is_for_exactly_one_context_and_reads_nothing_else() -> None:
    mine = _msg()
    other = _msg({"context_id": "context-2"})
    rows = [_mrec(mine, 80), _mrec(other, 81)]
    assert [e.message for e in derive_inbox_view(rows, CLOSED_AT, context_id=CTX).entries] == [mine]
    assert [e.message for e in derive_inbox_view(rows, CLOSED_AT, context_id="context-2").entries] == [other]
    nothing = derive_inbox_view(rows, CLOSED_AT, context_id="unknown-context")
    assert nothing.entries == () and nothing.public_value()["messages"] == []
    with pytest.raises(ValueError):
        derive_inbox_view(rows, CLOSED_AT, context_id="")
    with pytest.raises(ValueError):
        derive_inbox_view(rows, "2026-11-16 02:00:30", context_id=CTX)
    with pytest.raises(TypeError):
        derive_inbox_view(
            [AdoptedRecord(_K.R19_OUTBOX_EMIT, RecordKind.OUTBOX_MESSAGE, object(), "tx", EVENT_A, 1)],
            CLOSED_AT,
            context_id=CTX,
        )
    # Total over every other record kind: ignored.
    wake = _wake()
    assert derive_inbox_view([_wake_rec(wake)], CLOSED_AT, context_id=CTX).entries == ()


def test_an_ack_ties_to_its_message_only_by_id_and_context() -> None:
    message = _msg()
    stray = _ack(message, context="context-2")
    unknown = replace(_ack(message), message_id=_msg({"context_id": "context-3"}).message_id)
    view = derive_inbox_view([_mrec(message), _arec(stray, 90), _arec(unknown, 91)], CLOSED_AT, context_id=CTX)
    assert [item.state for item in view.entries] == [MessageState.PENDING]
    assert {kind for kind, _ in view.orphans} == {RecordKind.INBOX_ACK}
    assert len(view.orphans) == 2
    # The first adopted ACK wins; a second row of the same message is an orphan.
    first, second = _ack(message, AckStatus.REJECTED), _ack(message, AckStatus.ACKNOWLEDGED)
    twice = derive_inbox_view([_mrec(message), _arec(first, 90), _arec(second, 91)], CLOSED_AT, context_id=CTX)
    assert [item.state for item in twice.entries] == [MessageState.REJECTED]
    assert len(twice.orphans) == 1


def test_newest_first_and_the_list_limit() -> None:
    messages = [
        _msg(created=_ts(CLOSED_AT, seconds=n), wake_changes={"evidence_fingerprint": f"{n:x}" * 64})
        for n in range(1, 4)
    ]
    rows = [_mrec(m, 80 + i) for i, m in enumerate(messages)]
    view = derive_inbox_view(rows, "2026-11-16T03:00:00Z", context_id=CTX)
    assert [e.message.created_utc for e in view.entries] == sorted(
        (m.created_utc for m in messages), reverse=True
    )
    assert len(view.items(2)) == 2 and len(view.items()) == 3
    for bad in (0, 51, -1, True, 1.5, "5", None):
        with pytest.raises(ValueError):
            view.items(bad)  # type: ignore[arg-type]
    assert len(view.items(1)) == 1 and inbox.MAX_LIST_LIMIT == 50


def test_the_view_is_deterministic_and_never_mutates_its_input() -> None:
    message = _msg()
    rows = [_mrec(message), _arec(_ack(message))]
    snapshot = list(rows)
    first = derive_inbox_view(rows, CLOSED_AT, context_id=CTX)
    assert derive_inbox_view(iter(rows), CLOSED_AT, context_id=CTX) == first
    assert rows == snapshot


def test_outbox_and_ack_records_leave_the_wake_view_unchanged() -> None:
    """D1 §4-3 / A-13: ``derive_wake_view`` ignores them; no orphan appears."""

    wake = _wake()
    claim = _claim(wake)
    base = [_wake_rec(wake), _claim_rec(claim), _closure_rec(claim)]
    message = _msg()
    extra = [_mrec(message), _arec(_ack(message))]
    plain = derive_wake_view(base, CLOSED_AT, clock_floor=None)
    mixed = derive_wake_view([*base, *extra], CLOSED_AT, clock_floor=None)
    assert mixed == plain and mixed.orphans == ()
    assert [e.state for e in mixed.entries] == [WakeState.CLOSED]


# --------------------------------------------------------------- projection


FORBIDDEN_SUBSTRINGS = (
    "source_id",
    "result_digest",
    "dedupe",
    "incarnation",
    "wakeup",
    "logical_cycle",
    "cycle",
    "context_id",
    "event_id",
    "event_sequence",
    "session_selector",
    "policy_row",
    "claim",
    "terminal",
    "digest",
)


def test_the_projection_has_exactly_the_closed_fields() -> None:
    message = _msg()
    view = derive_inbox_view(
        [_mrec(message), _arec(_ack(message, selector="my-session"))], CLOSED_AT, context_id=CTX
    )
    body = view.public_value()
    assert set(body) == {"messages", "as_of_utc"}
    assert body["as_of_utc"] == CLOSED_AT
    [item] = body["messages"]  # type: ignore[misc]
    assert set(item) == {
        "message_id",
        "template",
        "reason_code",
        "text",
        "created_utc",
        "expires_utc",
        "state",
        "acked_utc",
    }
    assert item["message_id"] == message.message_id
    assert (item["template"], item["reason_code"]) == ("DECISION_HELD", "HELD")
    assert item["text"] == inbox.TEMPLATE_TEXT[(OutboxTemplate.DECISION_HELD, OutboxReason.HELD)]
    assert item["state"] == "ACKNOWLEDGED" and item["acked_utc"] == "2026-11-16T03:00:00Z"
    blob = json.dumps(body)
    for forbidden in (
        message.source_id,
        message.result_digest,
        message.dedupe_key,
        message.incarnation_id,
        message.context_id,
        "my-session",
        EVENT_T,
        "context-only",
        "STALE_SOURCE",
        "ROW_",
    ):
        assert forbidden not in blob
    # The InboxItem dataclass itself has no field named like an omitted one.
    names = set(inbox.InboxItem.__dataclass_fields__)
    assert not any(bad in name for name in names for bad in FORBIDDEN_SUBSTRINGS)
    # A pending message has no ACK time.
    pending = derive_inbox_view([_mrec(message)], CLOSED_AT, context_id=CTX).public_value()
    assert pending["messages"][0]["acked_utc"] is None  # type: ignore[index]


def test_message_id_does_not_contain_the_cycle_or_any_digest() -> None:
    wake = _wake()
    message = _msg()
    for secret in (
        wake.logical_cycle_id,
        message.source_id,
        message.result_digest,
        message.dedupe_key,
        INC,
        FP,
    ):
        assert secret not in message.message_id
    assert message.message_id == derive_inbox_view(
        [_mrec(message)], CLOSED_AT, context_id=CTX
    ).entries[0].message_id


# --------------------------------------------------------------------- ACK


def _check(
    rows: list[AdoptedRecord],
    message: OutboxMessageV1,
    status: AckStatus = AckStatus.ACKNOWLEDGED,
    now: str = CLOSED_AT,
    **kw: Any,
) -> inbox.AckCheck:
    view = derive_inbox_view(rows, now, context_id=message.context_id)
    return ack_eligibility(view, message.message_id, status, now, **kw)


def test_ack_eligibility_order() -> None:
    message = _msg()
    rows = [_mrec(message)]
    assert _check(rows, message) == inbox.AckCheck(AckEligibility.WRITE)
    # Nothing / another Context: one answer.
    view = derive_inbox_view(rows, CLOSED_AT, context_id="context-2")
    unknown = ack_eligibility(view, message.message_id, AckStatus.ACKNOWLEDGED, CLOSED_AT)
    assert unknown == inbox.AckCheck(AckEligibility.NOT_FOUND)
    assert _check([], message).eligibility is AckEligibility.NOT_FOUND
    assert (
        ack_eligibility(derive_inbox_view(rows, CLOSED_AT, context_id=CTX), "outbox-" + "0" * 64, AckStatus.ACKNOWLEDGED, CLOSED_AT).eligibility
        is AckEligibility.NOT_FOUND
    )
    # Expiry is half-open: now >= expires.
    assert _check(rows, message, now=_ts(message.expires_utc, seconds=-1)).eligibility is AckEligibility.WRITE
    assert _check(rows, message, now=message.expires_utc).eligibility is AckEligibility.EXPIRED
    # A PREPARED ACK not yet adopted.
    assert _check(rows, message, ack_key_live=True) == inbox.AckCheck(AckEligibility.HOLD, AckHold.IN_FLIGHT)
    # Clock: before the message was created, or before the floor.
    early = _ts(message.created_utc, seconds=-1)
    assert _check(rows, message, now=early) == inbox.AckCheck(AckEligibility.HOLD, AckHold.CLOCK_UNVERIFIABLE)
    assert _check(rows, message, clock_floor=_ts(CLOSED_AT, seconds=1)) == inbox.AckCheck(
        AckEligibility.HOLD, AckHold.CLOCK_UNVERIFIABLE
    )
    assert _check(rows, message, clock_floor=CLOSED_AT).eligibility is AckEligibility.WRITE
    # Expired wins over the clock holds; EXPIRED needs no clock.
    assert _check(rows, message, now=_ts(message.expires_utc, days=1), clock_floor=_ts(message.expires_utc, days=9)).eligibility is AckEligibility.EXPIRED


def test_the_ack_compare_is_by_status_only() -> None:
    message = _msg()
    stored = _ack(message, AckStatus.ACKNOWLEDGED, selector="first-session")
    rows = [_mrec(message), _arec(stored)]
    # Same status, whatever the selector: EXISTING (nothing to write).
    assert inbox.same_ack_status(stored, AckStatus.ACKNOWLEDGED)
    assert _check(rows, message, AckStatus.ACKNOWLEDGED).eligibility is AckEligibility.EXISTING
    assert not inbox.same_ack_status(stored, AckStatus.REJECTED)
    assert _check(rows, message, AckStatus.REJECTED).eligibility is AckEligibility.CONFLICT
    # And the reverse.
    reverse = [_mrec(message), _arec(_ack(message, AckStatus.REJECTED, selector="x"))]
    assert _check(reverse, message, AckStatus.REJECTED).eligibility is AckEligibility.EXISTING
    assert _check(reverse, message, AckStatus.ACKNOWLEDGED).eligibility is AckEligibility.CONFLICT
    # A replay after the message expired still answers from the immutable row.
    late = _ts(message.expires_utc, days=2)
    assert _check(rows, message, AckStatus.ACKNOWLEDGED, now=late).eligibility is AckEligibility.EXISTING
    assert _check(rows, message, AckStatus.REJECTED, now=late).eligibility is AckEligibility.CONFLICT
    # An ACK in flight does not hide an adopted one.
    assert _check(rows, message, ack_key_live=True).eligibility is AckEligibility.EXISTING


def test_the_ack_compare_never_calls_conflicts_with(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: InboxAckV1, other: InboxAckV1) -> bool:
        raise AssertionError("conflicts_with must not be used")

    monkeypatch.setattr(InboxAckV1, "conflicts_with", boom)
    message = _msg()
    rows = [_mrec(message), _arec(_ack(message))]
    assert _check(rows, message, AckStatus.REJECTED).eligibility is AckEligibility.CONFLICT
    assert _check(rows, message, AckStatus.ACKNOWLEDGED).eligibility is AckEligibility.EXISTING
    source = inspect.getsource(inbox)
    assert [n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute) and n.attr == "conflicts_with"] == []


def test_ack_check_and_inputs_are_strict() -> None:
    message = _msg()
    view = derive_inbox_view([_mrec(message)], CLOSED_AT, context_id=CTX)
    with pytest.raises(TypeError):
        ack_eligibility(view, message.message_id, "ACKNOWLEDGED", CLOSED_AT)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ack_eligibility(view, message.message_id, AckStatus.REJECTED, CLOSED_AT, ack_key_live=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ack_eligibility(view, message.message_id, AckStatus.REJECTED, "now")
    with pytest.raises(ValueError):
        inbox.AckCheck(AckEligibility.HOLD)
    with pytest.raises(ValueError):
        inbox.AckCheck(AckEligibility.WRITE, AckHold.IN_FLIGHT)
    with pytest.raises(TypeError):
        inbox.same_ack_status(object(), AckStatus.REJECTED)  # type: ignore[arg-type]


def test_the_session_selector_is_the_client_id_or_the_closed_literal() -> None:
    assert inbox.ack_selector(None) == inbox.DEFAULT_SESSION_SELECTOR == "context-only"
    assert inbox.ack_selector("my-session.1") == "my-session.1"
    for bad in ("", " x", "x" * 129, "a/b", "-x"):
        with pytest.raises(ValueError):
            inbox.ack_selector(bad)
    # The literal is a valid ACK selector (the DTO accepts it).
    _ack(_msg(), selector=inbox.DEFAULT_SESSION_SELECTOR)


def test_is_message_id_matches_the_dto_format() -> None:
    message = _msg()
    assert inbox.is_message_id(message.message_id)
    for bad in (
        "",
        None,
        5,
        "outbox-message-" + "a" * 64,
        "outbox-" + "A" * 64,
        "outbox-" + "a" * 63,
        "outbox-" + "a" * 65,
        "outbox-" + "a" * 64 + "\n",
        "x" + message.message_id,
    ):
        assert not inbox.is_message_id(bad)


def test_ack_vocabularies_are_closed() -> None:
    assert {item.value for item in inbox.AckResult} == {
        "COMMITTED", "EXISTING", "CONFLICT", "EXPIRED", "NOT_FOUND", "HOLD",
    }
    assert {item.value for item in AckHold} == {
        "R19_DISABLED", "HISTORY_UNAVAILABLE", "IN_FLIGHT", "CONTEXT_CONFLICT",
        "CLOCK_UNVERIFIABLE", "HISTORY_CAPACITY", "COLLISION",
    }
    assert {item.value for item in OutboxHold} >= {
        "R19_DISABLED", "HISTORY_UNAVAILABLE", "WAKE_UNKNOWN", "WAKE_CORRUPT", "NOT_SETTLED",
        "NO_MESSAGE", "NONE_LEFT", "IN_FLIGHT", "TOO_OLD", "CLOCK_UNVERIFIABLE",
        "UNREPRESENTABLE", "HISTORY_CAPACITY", "COLLISION",
    }
    assert inbox.PERMANENT_OUTBOX_HOLDS == {OutboxHold.TOO_OLD, OutboxHold.UNREPRESENTABLE}


# ---------------------------------------------------------------- emission


def _closed_entry(at: str = CLOSED_AT) -> WakeEntry:
    wake = _wake()
    claim = _claim(wake)
    rows = [_wake_rec(wake), _claim_rec(claim), _closure_rec(claim)]
    entry = derive_wake_view(rows, at, clock_floor=None).entries[0]
    assert entry.state is WakeState.CLOSED
    return entry


def _failed_entry(
    reason: TerminalReason = TerminalReason.EXPIRED_WAKE, *, claims: int = 1
) -> WakeEntry:
    wake = _wake()
    rows = [_wake_rec(wake), *(_claim_rec(_claim(wake, n)) for n in range(1, claims + 1))]
    terminal = WakeTerminalV1.for_wake(
        wake, WakeTerminalKind.FAILED_TERMINAL, reason, event_id=EVENT_T, event_sequence=70
    )
    rows.append(_terminal_rec(terminal))
    entry = derive_wake_view(rows, "2026-11-23T00:01:00Z", clock_floor=None).entries[0]
    assert entry.state is WakeState.FAILED
    return entry


def _plan(entry: WakeEntry, now: str, **kw: Any) -> inbox.EmissionPlan:
    kw.setdefault("outbox_ttl_s", 604_800)
    return plan_emission(entry, now, **kw)


def test_lower_bound_comes_from_record_bytes() -> None:
    claim_entry = _closed_entry()
    assert claim_entry.latest_claim is not None
    assert claim_entry.latest_claim.claim_lease_expires_utc == LEASE_END[0]
    assert inbox.closing_lower_bound(claim_entry) is not None
    assert inbox.closing_lower_bound(claim_entry).strftime("%Y-%m-%dT%H:%M:%SZ") == L  # type: ignore[union-attr]
    exhausted = _failed_entry(TerminalReason.ATTEMPTS_EXHAUSTED, claims=3)
    assert exhausted.latest_claim is not None
    assert inbox.closing_lower_bound(exhausted).strftime("%Y-%m-%dT%H:%M:%SZ") == "2026-11-16T02:20:00Z"  # type: ignore[union-attr]
    expired = _failed_entry(TerminalReason.EXPIRED_WAKE)
    assert inbox.closing_lower_bound(expired).strftime("%Y-%m-%dT%H:%M:%SZ") == expired.expiry_utc  # type: ignore[union-attr]
    # No claim and no EXPIRED_WAKE terminal: no bound.
    assert inbox.closing_lower_bound(replace(claim_entry, claims=())) is None


def test_the_lease_bound_matches_the_settings_cap_and_the_raw_state_cap() -> None:
    from suzka.config.schema import R19Settings
    from suzka.r19.raw_state import CLAIM_LEASE_CAP_S

    field = R19Settings.model_fields["claim_lease_s"]
    assert any(getattr(m, "le", None) == inbox.LEASE_BOUND_S for m in field.metadata)
    assert inbox.LEASE_BOUND_S == CLAIM_LEASE_CAP_S == 120


def test_a_closed_cycle_gets_one_message_with_the_ttl_rule() -> None:
    entry = _closed_entry()
    plan = _plan(entry, CLOSED_AT)
    assert plan.ready and plan.hold is None and plan.lower_bound_utc == L
    message = plan.message
    assert message is not None
    assert message.created_utc == CLOSED_AT
    # min(T + 7 d, L + 7 d) = L + 7 d: anchored at the result, not at T.
    assert message.expires_utc == _ts(L, days=7)
    assert (message.template, message.reason_code) == (OutboxTemplate.DECISION_HELD, OutboxReason.HELD)
    assert (message.context_id, message.incarnation_id) == (entry.wake.context_id, entry.wake.incarnation_id)
    assert OutboxMessageV1.from_json(message.canonical_bytes()) == message
    # A short TTL wins: min(T + ttl, L + 7 d).
    short = _plan(entry, CLOSED_AT, outbox_ttl_s=60).message
    assert short is not None and short.expires_utc == _ts(CLOSED_AT, seconds=60)


def test_the_message_never_lives_more_than_seven_days_after_the_result() -> None:
    entry = _closed_entry()
    late = _ts(L, days=3)  # emitted three days late
    message = _plan(entry, late).message
    assert message is not None
    assert message.expires_utc == _ts(L, days=7)  # not late + 7 d
    # At or past L + 7 d: TOO_OLD, a derived "never notified".
    assert _plan(entry, _ts(L, days=7, seconds=-1)).ready
    too_old = _plan(entry, _ts(L, days=7))
    assert too_old.hold is OutboxHold.TOO_OLD and too_old.message is None
    assert too_old.message_id is not None and too_old.lower_bound_utc == L


def test_dedupe_comes_first_and_needs_no_time() -> None:
    entry = _closed_entry()
    message = _plan(entry, CLOSED_AT).message
    assert message is not None
    adopted = frozenset({message.message_id})
    # Adopted: NONE_LEFT even when the message would now be too old, or the
    # clock cannot be verified (no time is needed to know it exists).
    for now, floor in (
        (CLOSED_AT, None),
        (_ts(L, days=30), None),
        (_ts(L, seconds=-5000), None),
        (CLOSED_AT, _ts(CLOSED_AT, days=1)),
    ):
        plan = _plan(entry, now, adopted_ids=adopted, clock_floor=floor)
        assert plan.hold is OutboxHold.NONE_LEFT and plan.message_id == message.message_id
    # The identity has no time in it.
    later = _plan(entry, _ts(CLOSED_AT, days=2)).message
    assert later is not None and later.message_id == message.message_id
    assert later.dedupe_key == message.dedupe_key and later.created_utc != message.created_utc


def test_clock_holds() -> None:
    entry = _closed_entry()
    rolled_back = _plan(entry, _ts(L, seconds=-1))
    assert rolled_back.hold is OutboxHold.CLOCK_UNVERIFIABLE and rolled_back.message is None
    assert _plan(entry, L).ready  # T >= L
    assert _plan(entry, CLOSED_AT, clock_floor=_ts(CLOSED_AT, seconds=1)).hold is OutboxHold.CLOCK_UNVERIFIABLE
    assert _plan(entry, CLOSED_AT, clock_floor=CLOSED_AT).ready
    # TOO_OLD is checked before the clock holds (D1 §4-5 step 5).
    assert _plan(entry, _ts(L, days=8), clock_floor=_ts(L, days=9)).hold is OutboxHold.TOO_OLD


def test_failed_cycles_get_cycle_failed_messages() -> None:
    for reason, claims in (
        (TerminalReason.EXPIRED_WAKE, 1),
        (TerminalReason.ATTEMPTS_EXHAUSTED, 3),
    ):
        entry = _failed_entry(reason, claims=claims)
        bound = inbox.closing_lower_bound(entry)
        assert bound is not None
        now = bound.strftime("%Y-%m-%dT%H:%M:%SZ")
        plan = _plan(entry, _ts(now, seconds=5))
        assert plan.ready, plan.hold
        message = plan.message
        assert message is not None
        assert (message.template, message.reason_code) == (OutboxTemplate.CYCLE_FAILED, OutboxReason.FAILED)
        assert entry.terminal is not None and message.source_id == entry.terminal.terminal_id
        assert message.expires_utc == _ts(now, days=7)
    # EXPIRED_WAKE: the terminal needs T past the Wake's expiry, so a time before
    # it cannot be the emitter's time.
    expired = _failed_entry(TerminalReason.EXPIRED_WAKE)
    assert _plan(expired, _ts(expired.expiry_utc, seconds=-1)).hold is OutboxHold.CLOCK_UNVERIFIABLE


def test_a_cancelled_corrupt_or_unsettled_wake_has_no_message() -> None:
    wake = _wake()
    claim = _claim(wake)
    cancelled_terminal = WakeTerminalV1.for_wake(
        wake,
        WakeTerminalKind.CANCELLED,
        TerminalReason.SUPERSEDED,
        event_id=EVENT_T,
        event_sequence=70,
        superseding=_wake(evidence_fingerprint="e" * 64),
    )
    cancelled = derive_wake_view([_wake_rec(wake), _terminal_rec(cancelled_terminal)], CLOSED_AT, clock_floor=None).entries[0]
    assert cancelled.state is WakeState.CANCELLED
    assert _plan(cancelled, CLOSED_AT).hold is OutboxHold.NO_MESSAGE
    for state in (WakeState.DUE, WakeState.SCHEDULED, WakeState.CLAIMED_LIVE, WakeState.EXHAUSTED, WakeState.RETRYABLE):
        assert _plan(replace(_closed_entry(), state=state), CLOSED_AT).hold is OutboxHold.NOT_SETTLED
    corrupt = replace(_closed_entry(), state=WakeState.CORRUPT, corrupt_reason="X")
    assert _plan(corrupt, CLOSED_AT).hold is OutboxHold.WAKE_CORRUPT
    # A closed state with the records missing is unrepresentable, never raised.
    assert _plan(replace(_closed_entry(), closure=None), CLOSED_AT).hold is OutboxHold.UNREPRESENTABLE
    assert _plan(replace(_failed_entry(), terminal=None), CLOSED_AT).hold is OutboxHold.UNREPRESENTABLE
    # ... and so is a closure without a claim to bound its time.
    no_claim = _plan(replace(_closed_entry(), claims=()), CLOSED_AT)
    assert no_claim.hold is OutboxHold.UNREPRESENTABLE and no_claim.message_id is not None
    assert claim  # the claim fixture is part of the closed entry above


def test_a_policy_row_the_builder_refuses_is_unrepresentable_not_raised() -> None:
    entry = _closed_entry()
    resolution = replace(
        entry.closure.resolution,  # type: ignore[union-attr]
        result_kind=ResultKind.DECISION_ISSUED,
        result_id="decision-" + "d" * 64,
    )
    bad = replace(entry, closure=SimpleNamespace(resolution=resolution, decision=SimpleNamespace(policy_row="ROW_4")))
    plan = _plan(bad, CLOSED_AT)  # type: ignore[arg-type]
    assert plan.hold is OutboxHold.UNREPRESENTABLE and plan.message is None


@pytest.mark.parametrize("row", list(PolicyRow))
def test_every_policy_row_maps_to_one_of_the_five_codes(row: PolicyRow) -> None:
    entry = _closed_entry()
    resolution = replace(
        entry.closure.resolution,  # type: ignore[union-attr]
        result_kind=ResultKind.DECISION_ISSUED,
        result_id="decision-" + "d" * 64,
    )
    issued = replace(
        entry,
        closure=SimpleNamespace(resolution=resolution, decision=SimpleNamespace(policy_row=row)),
    )
    plan = _plan(issued, CLOSED_AT)  # type: ignore[arg-type]
    assert plan.message is not None
    expected = OutboxReason.NO_ACTION if row in (PolicyRow.ROW_5, PolicyRow.ROW_5B) else OutboxReason.PLAN_SELECTED
    assert (plan.message.template, plan.message.reason_code) == (OutboxTemplate.DECISION_ISSUED, expected)
    body = json.loads(plan.message.canonical_bytes())
    assert not any("row" in key for key in body)
    assert body["template"] == "DECISION_ISSUED" and body["reason_code"] == expected.value
    assert (plan.message.template, plan.message.reason_code) in inbox.TEMPLATE_TEXT
    # Within one code the bytes do not depend on the row (2b / 4b / 5b collapse).
    twin = PolicyRow.ROW_5 if expected is OutboxReason.NO_ACTION else PolicyRow.ROW_4
    again = replace(
        issued,
        closure=SimpleNamespace(
            resolution=resolution, decision=SimpleNamespace(policy_row=twin)
        ),
    )
    assert _plan(again, CLOSED_AT).message == plan.message  # type: ignore[arg-type]


def test_an_existing_confirmation_is_a_separate_message() -> None:
    entry = _closed_entry()
    held = _plan(entry, CLOSED_AT).message
    resolution = replace(
        entry.closure.resolution,  # type: ignore[union-attr]
        result_kind=ResultKind.EXISTING_DECISION,
        result_id="cycle-existing-" + "e" * 64,
        result_digest="3" * 64,
    )
    confirmed = replace(entry, closure=SimpleNamespace(resolution=resolution, decision=None))
    message = _plan(confirmed, CLOSED_AT).message  # type: ignore[arg-type]
    assert held is not None and message is not None
    assert message.template is OutboxTemplate.DECISION_CONFIRMED and message.reason_code is OutboxReason.CONFIRMED
    assert message.message_id != held.message_id


def test_the_message_context_is_the_wakes_context() -> None:
    wake = _wake(context_id="context-9")
    claim = _claim(wake)
    rows = [_wake_rec(wake), _claim_rec(claim), _closure_rec(claim)]
    entry = derive_wake_view(rows, CLOSED_AT, clock_floor=None).entries[0]
    message = _plan(entry, CLOSED_AT).message
    assert message is not None and message.context_id == "context-9"
    failed = _failed_entry()
    failed_message = _plan(failed, _ts(failed.expiry_utc, seconds=5)).message
    assert failed_message is not None and failed_message.context_id == failed.wake.context_id


def test_emission_candidates_are_a_derived_ordered_list() -> None:
    first_wake = _wake()
    second_wake = _wake(evidence_fingerprint="e" * 64)
    third_wake = _wake(evidence_fingerprint="d" * 64)
    rows: list[AdoptedRecord] = []
    for index, wake in enumerate((first_wake, second_wake)):
        claim = _claim(wake)
        rows += [_wake_rec(wake, 20 + index), _claim_rec(claim), _closure_rec(claim)]
    rows.append(_wake_rec(third_wake, 30))  # DUE: not a candidate
    view = derive_wake_view(rows, CLOSED_AT, clock_floor=None)
    states = {e.wakeup_id: e.state for e in view.entries}
    assert states[third_wake.wakeup_id] is WakeState.DUE
    plans = emission_candidates(view, frozenset(), outbox_ttl_s=604_800)
    closed = [e.wakeup_id for e in view.entries if e.state is WakeState.CLOSED]
    assert [p.wakeup_id for p in plans] == closed and len(plans) == 2
    assert all(p.ready for p in plans)
    # An adopted message is skipped; the rest keep their order.
    skip = plans[0].message_id
    assert skip is not None
    rest = emission_candidates(view, frozenset({skip}), outbox_ttl_s=604_800)
    assert [p.wakeup_id for p in rest] == [plans[1].wakeup_id]
    # Nothing left to do when all are adopted: no periodic work is invented.
    everything = frozenset(p.message_id for p in plans if p.message_id)
    assert emission_candidates(view, everything, outbox_ttl_s=604_800) == ()
    # The instant and the floor are the view's own.
    floored = derive_wake_view(rows, CLOSED_AT, clock_floor=_ts(CLOSED_AT, seconds=5))
    assert {p.hold for p in emission_candidates(floored, frozenset(), outbox_ttl_s=604_800)} == {
        OutboxHold.CLOCK_UNVERIFIABLE
    }
    assert emission_candidates(replace(view, entries=()), frozenset(), outbox_ttl_s=604_800) == ()


def test_adopted_message_ids_reads_outbox_records_only() -> None:
    message = _msg()
    other = _msg({"context_id": "context-2"})
    rows = [_mrec(message), _arec(_ack(message)), _wake_rec(_wake()), _mrec(other, 81)]
    assert inbox.adopted_message_ids(rows) == {message.message_id, other.message_id}
    assert inbox.adopted_message_ids([]) == frozenset()


def test_emission_inputs_are_strict() -> None:
    entry = _closed_entry()
    with pytest.raises(TypeError):
        plan_emission(object(), CLOSED_AT, outbox_ttl_s=60)  # type: ignore[arg-type]
    for ttl in (0, -1, 604_801, True, 1.5):
        with pytest.raises(ValueError):
            plan_emission(entry, CLOSED_AT, outbox_ttl_s=ttl)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        plan_emission(entry, "tomorrow", outbox_ttl_s=60)
    with pytest.raises(ValueError):
        inbox.EmissionPlan("w", WakeState.CLOSED, None, None, None, None)
    with pytest.raises(ValueError):
        inbox.EmissionPlan("w", WakeState.CLOSED, "x", None, _msg(), OutboxHold.TOO_OLD)


# ------------------------------------------------------------------- purity


def test_inbox_module_is_pure() -> None:
    tree = ast.parse(inspect.getsource(inbox))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    # Only pure values: no store, Journal, runtime, API, registry or owner.
    for module in imported:
        assert module is not None
        assert module.split(".")[0] in {
            "__future__", "collections", "dataclasses", "datetime", "enum",
            "re", "types", "typing", "suzka",
        }, module
        assert not module.startswith(
            ("suzka.runtime", "suzka.api", "suzka.r16.history_store", "suzka.r16.history_reader_impl")
        ), module
    assert imported <= {
        m
        for m in imported
        if m.startswith(("suzka.r16.", "suzka.r19.contracts", "suzka.r19.ledger"))
        or "." not in m
        or m in {"collections.abc"}
    }
    plain = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert plain <= {"re"}
    # No clock read, no I/O: the instant is always an argument.
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in {"now", "utcnow", "today"}
    ]
    # The reader / codec names are annotations only (under TYPE_CHECKING).
    top = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "suzka.r16.history_reader" not in top
    assert "suzka.r16.history_codec" not in top
    source = inspect.getsource(inbox)
    for name in ("sqlite3", "threading", "logging", "subprocess"):
        assert name not in source
    assert "os" not in {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}


def test_scaffold_no_api_module_reads_outbox_and_ack_records() -> None:
    """The architecture guard of D1 §5-2 (scope amendment A-14), scaffold form:
    among ``suzka``'s API-facing modules none touches the record kinds.
    IP-21c extends it to the routes."""

    import pathlib

    root = pathlib.Path(inspect.getfile(inbox)).resolve().parents[1]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        if "RecordKind.OUTBOX_MESSAGE" in text or "RecordKind.INBOX_ACK" in text:
            offenders.append(path.relative_to(root).as_posix())
    api = [item for item in offenders if item.startswith("api/")]
    assert api == []
    assert "r19/inbox.py" in offenders
    for forbidden in ("api/context_read_model.py", "api/routes/debug.py"):
        text = (root / forbidden).read_text()
        assert "suzka.r19.inbox" not in text and "OutboxMessageV1" not in text
