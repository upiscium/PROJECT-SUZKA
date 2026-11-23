"""IP-22c, F05-w: the approval window and the stand-in (D1 sec. 4-4, 4-5, 6).

T1, process-wide fake clock (22a ``FakeClock``).  The chain is the real driver
chain over the real routes up to a Policy ALLOW on the forward Intent; the
stand-in operator is a client of the real ``/api/r18`` route.

* never approves: the Intent closes at its window end as row 2c with NO claim,
  the target bytes are unchanged, a late approval is refused, and the subject
  went on deciding (a safety stop never blocks the subject).
* replays a consumed nonce over HTTP: refused, one claim, one target state.

Scope: profile NEG-plain with R18 on (A-12: F05 needs no R15 admission); the
whole S0 chain with the R15 admission is IP-22d's dry run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import pytest

from experiments.r20.approval import ApprovalOperator, recompute
from experiments.r20.client import AsgiTransport, HttpClient
from experiments.r20.factory import create_subject
from experiments.r20.provider import ScriptedProvider
from experiments.r20.tier1 import FakeClock, T1Runtime
from r20_chain import adopted, drive_to_allowed, goal_id, operations
from suzka.r16.history_records import HistoryOperationKind as K
from suzka.r16.successor_policy import SuccessorRow
from suzka.r18.contracts import ExecutionClaimV1
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import EventLifecycle


def _target_digest(state: Any) -> str:
    return hashlib.sha256(Path(state.settings.r18.target_path).read_bytes()).hexdigest()


def _no_failed_events(state: Any) -> None:
    assert not [r for r in state.event_journal.inspect().records if r.lifecycle is EventLifecycle.FAILED]


def _claims(state: Any) -> int:
    return len(operations(state, K.R18_EXECUTION_CLAIM)) + len(adopted(state, ExecutionClaimV1))


def test_f05_w_an_unapproved_intent_closes_as_row_2c_with_no_claim_and_the_subject_goes_on(
    r20_base: Path,
) -> None:
    subject = create_subject("NEG-plain", base=r20_base, r18=True)
    with FakeClock() as clock, T1Runtime(subject, ScriptedProvider(), clock=clock, scheduler=False) as runtime:
        client, state = runtime.client, runtime.state
        assert client is not None
        ids = drive_to_allowed(client, state)
        intent = ids["intent"]
        operator = ApprovalOperator(HttpClient(AsgiTransport(client), subject.port), subject.port)
        target_before = _target_digest(state)

        # the operator can SEE the open approval, and never approves it
        listed = operator.listing()
        assert listed.status == 200
        assert [item["intent_id"] for item in listed.json()["approvals"]] == [intent.intent_id]
        assert _claims(state) == 0

        # the window ends: forward past it (T1 clock; windows are >= 600 s, D1 4-4)
        end = datetime.strptime(intent.approval_window_end_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        clock.advance((end - clock.now()).total_seconds() + 2)
        closing = state._plan_preview_owner._submit_preview(ids["goal_id"]).result(timeout=30).value
        assert closing.row is SuccessorRow.ROW_2C  # the unapproved Intent closed, Plan A
        assert goal_id(state) == ids["goal_id"]

        # a late approval is refused: still no claim, no effect
        late = operator.approve(intent.intent_id)
        assert late.status in (404, 409), late.text  # the window closed: a hold, not a claim
        assert late.json().get("status") in ("hold", "refused", "not_found"), late.text
        assert _claims(state) == 0
        assert _target_digest(state) == target_before
        assert operations(state, K.R18_OUTCOME_COMMIT) == []
        # the subject went on: the closing Decision was issued by the live
        # runtime, no event failed, the server is healthy
        assert client.get("/health").status_code == 200
        _no_failed_events(state)
        assert any(
            row.event_type is AgentEventType.PLAN_PREVIEW_REVIEW and row.lifecycle is EventLifecycle.COMPLETED
            for row in state.event_journal.inspect().records
        )


def test_f05_w_a_replayed_nonce_over_http_is_refused_one_claim_one_state(r20_base: Path) -> None:
    subject = create_subject("NEG-plain", base=r20_base, r18=True)
    with T1Runtime(subject, ScriptedProvider(), executor=False, scheduler=False) as runtime:
        client, state = runtime.client, runtime.state
        assert client is not None
        ids = drive_to_allowed(client, state)
        operator = ApprovalOperator(HttpClient(AsgiTransport(client), subject.port), subject.port)
        shown = operator.review(ids["intent"].intent_id)
        checked = recompute(shown.json())
        assert checked.ok, checked.problems
        first = operator.confirm(ids["intent"].intent_id, checked.nonce, checked.code)
        assert first.status == 200 and first.json()["status"] == "claimed"
        assert _claims(state) == 2  # one claim row and its adopted record
        target_after_first = _target_digest(state)
        for _ in range(2):
            replay = operator.confirm(ids["intent"].intent_id, checked.nonce, checked.code)
            assert replay.status == 409 and replay.json()["status"] == "hold"
        # a fresh review of the claimed Intent shows no new challenge to confirm
        again = operator.review(ids["intent"].intent_id)
        assert again.status != 200 or again.json().get("status") != "review"
        assert _claims(state) == 2
        assert _target_digest(state) == target_after_first
        _no_failed_events(state)


@pytest.mark.parametrize("same_origin", [False, True])
def test_f05_w_the_stand_in_is_a_guarded_client_with_and_without_fetch_metadata(
    r20_base: Path, same_origin: bool
) -> None:
    """The stand-in can never approve through a route the guards refuse: with a
    foreign Origin nothing is claimed (the full guard matrix is IP-22a's)."""

    subject = create_subject("NEG-plain", base=r20_base, r18=True)
    with T1Runtime(subject, ScriptedProvider(), executor=False, scheduler=False) as runtime:
        client, state = runtime.client, runtime.state
        assert client is not None
        ids = drive_to_allowed(client, state)
        operator = ApprovalOperator(
            HttpClient(AsgiTransport(client), subject.port), subject.port, same_origin_fetch=same_origin
        )
        checked = recompute(operator.review(ids["intent"].intent_id).json())
        refused = operator.confirm(ids["intent"].intent_id, checked.nonce, checked.code, Origin="http://evil.example")
        assert refused.status in (400, 403, 409, 421)
        assert _claims(state) == 0
        # CONTROL: the same request with the right Origin (and, in the second
        # variant, Sec-Fetch-Site: same-origin) is the one that claims
        accepted = operator.confirm(ids["intent"].intent_id, checked.nonce, checked.code)
        assert accepted.status == 200 and accepted.json()["status"] == "claimed"
        assert _claims(state) == 2
