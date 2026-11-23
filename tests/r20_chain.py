"""Shared drivers for the IP-22c F-matrix tests (imported, not collected).

Everything goes through the real routes of a real app built by the IP-22a
factory (``T1Runtime``); the helpers only wait and read.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from fastapi.testclient import TestClient

from suzka.r16.history_records import HistoryOperationKind
from suzka.runtime.agent_runtime import AgentEventType
from suzka.runtime.event_journal import EventLifecycle

X_TEXT = "something wonderful happened today"


def until(predicate: Callable[[], Any], timeout: float = 40.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def chat(client: Any, text: str, **selectors: str) -> dict[str, Any]:
    body: dict[str, Any] = {"message": text, "attachments": []}
    body.update(selectors)
    response = client.post("/api/chat", json=body)
    assert response.status_code == 200, response.text
    data: dict[str, Any] = response.json()
    return data


def settled(client: Any, ticket_id: str) -> dict[str, Any]:
    def view() -> dict[str, Any] | None:
        body: dict[str, Any] = client.get(f"/api/chat/commands/{ticket_id}").json()
        return None if body["status"] == "pending" else body

    found: dict[str, Any] = until(view)
    return found


def wait_interest(state: Any, experience_id: str) -> None:
    until(
        lambda: any(
            record.target.reference == experience_id
            for record in state.main_loop.export_motivation_state().records
        )
    )


def incarnation(state: Any) -> str:
    from suzka.r16.history_adoption import current_incarnation

    return str(current_incarnation(state.event_journal, state.subject_history_store))


def operations(state: Any, kind: HistoryOperationKind) -> list[Any]:
    return [
        row
        for row in state.subject_history_store.operations(incarnation(state))
        if row.operation_kind is kind
    ]


def completed_events(state: Any, event_type: AgentEventType) -> int:
    return sum(
        1
        for row in state.event_journal.records
        if row.event_type is event_type and row.lifecycle is EventLifecycle.COMPLETED
    )


def adopted(state: Any, record_type: type, *, strict: bool = False) -> list[Any]:
    """Adopted records of one type.  While events are in flight the reader may
    report a transient block: polling callers get ``[]``, ``strict`` asserts."""

    from suzka.r16.history_reader import SubjectHistoryReader

    found = SubjectHistoryReader(state.event_journal, state.subject_history_store).adopted_records()
    if found.blocked is not None:
        assert not strict, found.blocked
        return []
    return [item.record for item in found.records if type(item.record) is record_type]


def goal_id(state: Any) -> str:
    (record,) = state.main_loop.export_goal_state().records
    return str(record.goal_id)


def drive_to_allowed(
    client: Any, state: Any, selectors: dict[str, str] | None = None
) -> dict[str, Any]:
    """The S0 driver chain through the real routes up to a Policy ALLOW on the
    row-4 forward Intent: activate, X, INTEREST, REVIEW, COMMIT.  The forecast
    and the Policy follow on their own.  Returns the ids."""

    from suzka.r18.contracts import ActionIntentV1, PolicyVerdict, PolicyVerdictV1

    selected = selectors or {}
    assert client.post("/api/r16/driver/activate").status_code == 200
    x = chat(client, X_TEXT, **selected)
    wait_interest(state, x["experience_id"])
    review = chat(client, f"REVIEW_EXPERIENCE:{x['experience_id']}", **selected)
    assert settled(client, review["command_ticket"]["ticket_id"])["status"] == "adopted"
    commit = chat(client, f"COMMIT_REVIEW_EXPERIENCE:1:{x['experience_id']}", **selected)
    assert settled(client, commit["command_ticket"]["ticket_id"])["status"] == "accepted"
    verdict = until(lambda: (adopted(state, PolicyVerdictV1) or [None])[0])
    assert verdict.verdict is PolicyVerdict.ALLOW
    (intent,) = adopted(state, ActionIntentV1)
    return {"x": x["experience_id"], "intent": intent, "goal_id": goal_id(state)}


# R19 timings of the T1 chain: the Feedback Wake is due at once and the tick is
# one second, so a whole S0-shaped cycle runs in seconds on the real clock.
# Goal / Commitment Wakes stay a week away (they are not the Feedback successor).
FAST_R19 = {
    "feedback_wake_delay_s": 0,
    "goal_wake_delay_s": 604_800,
    "commitment_wake_delay_s": 604_800,
    "tick_s": 1,
}


def run_full_chain(
    runtime: Any, subject: Any, *, client: Any = None, selectors: dict[str, str] | None = None
) -> dict[str, Any]:
    """The S0-shaped chain through real routes and the stand-in operator, up to
    the Outbox message of the Feedback Wake: driver chain, Policy ALLOW, human
    approval (stand-in), claim, executor effect, Outcome, Feedback, Wake,
    cycle Decision, Outbox."""

    from experiments.r20.approval import ApprovalOperator
    from experiments.r20.client import AsgiTransport, HttpClient
    from suzka.r18.contracts import VerifiedOutcomeV1

    client = client if client is not None else runtime.client
    state = runtime.state
    ids = drive_to_allowed(client, state, selectors)
    operator = ApprovalOperator(HttpClient(AsgiTransport(runtime.client), subject.port), subject.port)
    done = operator.approve(ids["intent"].intent_id)
    assert done.status == 200 and done.json()["status"] == "claimed", done.text
    until(lambda: adopted(state, VerifiedOutcomeV1))
    until(lambda: operations(state, HistoryOperationKind.R19_OUTBOX_EMIT), timeout=90)
    ids["operator"] = operator
    return ids


def true_rollback(settings: Any, root: Any) -> str:
    """The true-rollback recipe of ``tests/test_startup_reconciliation.py``
    (corrupt the LAST transition record of the ACTIVE StateWAL generation and
    delete the AgentState snapshot), applied to a STOPPED subject on a harness
    root.  Test-side fault injection on real files; it refuses any root that
    is not a marked throwaway root.  Returns the corrupted record's event type."""

    import json
    from pathlib import Path

    from experiments.r20 import guards

    directory = Path(settings.state_wal.directory)
    snapshot = Path(settings.agent_state.path)
    root_path = Path(root)
    for path in (directory, snapshot):
        assert root_path in path.resolve().parents, "not inside the subject root"
    assert (root_path / guards.MARKER_NAME).is_file(), "not a throwaway root"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    active = str(manifest["active_generation_id"])
    generation = next(directory.rglob(f"{active}.jsonl"))
    lines = generation.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[-1])
    assert record["record_type"] == "transition"
    event_type = str(record["event_type"])
    record["record_hash"] = "0" * 64
    lines[-1] = json.dumps(record, separators=(",", ":")).encode() + b"\n"
    generation.write_bytes(b"".join(lines))
    snapshot.unlink()
    return event_type


class Wire:
    """A ``TestClient``-shaped shim that sends every request through the
    harness ``Recorder`` (status + closed-field projection + canary hits)."""

    def __init__(self, runtime: Any, port: int, canaries: dict[str, str]) -> None:
        from types import SimpleNamespace

        from experiments.r20.client import AsgiTransport, HttpClient, Recorder

        self.recorder = Recorder(HttpClient(AsgiTransport(runtime.client), port), canaries)
        self.app = runtime.client.app
        self._wrap = lambda result: SimpleNamespace(
            status_code=result.status, text=result.text, json=result.json
        )

    def get(self, path: str) -> Any:
        return self._wrap(self.recorder.request("GET", path))

    def post(self, path: str, json: Any = None) -> Any:
        return self._wrap(self.recorder.request("POST", path, json))
