"""S0 on the process tier (T2): the chain driver and the read-only evidence
readers (IP-22d; IP-22 D1 r3 sec. 4-3 and sec. 7).

One ``S0Session`` is one genesis subject served by a normal ``serve`` child
(``ChildProcess``) with the SCRIPTED provider.  Every state-changing request
goes through the destination-bound client; every read of a store is ONE short
``mode=ro`` read transaction (``polling.short_read``).  Nothing here touches a
production path or a real model, and nothing under ``suzka/`` imports it.

Honest limits (kept in the docstrings of the callers, repeated here):

* The R15 S1 / S2 admission is pulled by the harness (``HARNESS_BRIDGE``,
  HD-22-2 (a)); the admission itself is the real gate and the real owner.
* T2 runs on the real wall clock: R19 delays are set small, windows are real.
  The provider is deterministic, the kill window is not.
* After a restart NO command to X can be sent (``SOURCE_NOT_OBSERVED``): a
  session therefore runs the R16 commands only in its first process life.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from experiments.r15_p0.pilot import Case, seal_case
from experiments.r15_p1.protocol import TrialRefused, run_trial
from experiments.r20 import factory
from experiments.r20.approval import ApprovalOperator
from experiments.r20 import artifacts
from experiments.r20.client import BoundClient, Destination, DestinationNoAnswer, SocketTransport
from experiments.r20.control import send
from experiments.r20.factory import GenesisSubject, Tier, create_subject
from experiments.r20.guards import GuardRefusal
from experiments.r20.launch import HumanLaunch
from experiments.r20.manifest import Manifest
from experiments.r20.model_config import ModelOverride
from experiments.r20.polling import poll_until, short_read
from experiments.r20.provider import S0_TARGET, r15_protocol_script
from experiments.r20.supervisor import READY_TIMEOUT_S, ChildProcess

# R19 timings of the T2 chain: the Feedback Wake is due at once, the tick and the
# claim lease are short, Goal / Commitment Wakes stay a week away (they are not the
# Feedback successor).  ``claim_lease_s`` 15 s keeps a killed claim's recovery
# (K5) within seconds, and is far above the one-tick review the scripted provider
# needs (item 11 of the checklist computes the inequality from these numbers).
T2_R19: dict[str, int] = {
    "feedback_wake_delay_s": 0,
    "goal_wake_delay_s": 604_800,
    "commitment_wake_delay_s": 604_800,
    "tick_s": 1,
    "claim_lease_s": 15,
}
# A REAL model answers in seconds, not milliseconds: the Feedback Wake review is up to 2 model calls, so
# the claim lease takes its maximum (120 s, ``schema.py``) and the tick is 2 s.
REAL_R19: dict[str, int] = {**T2_R19, "tick_s": 2, "claim_lease_s": 120}
REAL_STEP_TIMEOUT_S = 300.0
REAL_LOAD_TIMEOUT_S = 600.0  # model load time is unmeasured (assume 1-3 min); G1 measures it
# F15: the supervised mode with its stated defaults (D1 sec. 3), spelled out.
CONTINUE: dict[str, Any] = {"mode": "continue"}
X_TEXT = "something wonderful happened today"
STEP_TIMEOUT_S = 90.0
FINALIZED = "FINALIZED"
KINDS = (
    "R16_DECISION_ISSUE", "R18_INTENT_ISSUE", "R17_EXPECTATION_COMMIT", "R18_POLICY_COMMIT",
    "R18_CHALLENGE_ISSUE", "R18_EXECUTION_CLAIM", "R18_OUTCOME_COMMIT", "R17_ASSESSMENT_COMMIT",
    "R17_FEEDBACK_COMMIT", "R19_WAKE_MATERIALIZE", "R19_WAKE_CLAIM", "R19_WAKE_ATTEMPT_OUTCOME",
    "R16_PLAN_HOLD", "R16_CYCLE_DECISION", "R16_CYCLE_RECEIPT", "R19_WAKE_TERMINAL",
    "R19_OUTBOX_EMIT", "R19_INBOX_ACK",
)


# ----------------------------------------------------------------- evidence


@dataclass(frozen=True, slots=True)
class DecisionRow:
    decision_id: str
    operation_kind: str
    sequence: int
    row: str
    predecessor_key: str
    sealed_input: bytes
    sealed_digest: str


class Evidence:
    """Read-only views of one subject's stores.  Safe against a live writer:
    every call is a short read transaction; a locked store is the caller's retry."""

    def __init__(self, subject: GenesisSubject) -> None:
        settings = subject.resolved_settings()
        self.history = settings.subject_history.path
        self.target = settings.r18.target_path if settings.r18.enabled else None
        self.journal = settings.event_journal.path
        self.root = subject.root

    # one transaction -> one consistent snapshot of the progress counters
    def counts(self) -> dict[tuple[str, str], int]:
        rows = short_read(
            self.history,
            "SELECT operation_kind, phase, count(*) FROM history_operations GROUP BY operation_kind, phase",
        )
        return {(str(kind), str(phase)): int(count) for kind, phase, count in rows}

    def finalized(self, kind: str) -> int:
        return self.counts().get((kind, FINALIZED), 0)

    def table_count(self, table: str, where: str = "1 = 1") -> int:
        if table not in _TABLES:
            raise ValueError(table)
        rows = short_read(self.history, f"SELECT count(*) FROM {table} WHERE {where}")
        return int(rows[0][0])

    def decisions(self) -> list[DecisionRow]:
        rows = short_read(
            self.history,
            "SELECT d.decision_id, o.operation_kind, o.processing_sequence, d.policy_row, d.predecessor_key, "
            "d.policy_input_bytes, d.policy_input_digest FROM history_decisions d "
            "JOIN history_operations o ON o.incarnation_id = d.incarnation_id AND o.transaction_id = d.transaction_id "
            "WHERE o.phase = 'FINALIZED' ORDER BY o.processing_sequence",
        )
        return [
            DecisionRow(str(r[0]), str(r[1]), int(r[2]), str(r[3]), str(r[4]), bytes(r[5]), str(r[6])) for r in rows
        ]

    def outcome_states(self) -> list[str]:
        return [str(r[0]) for r in short_read(self.history, "SELECT outcome_state FROM verified_outcomes")]

    def target_effects(self) -> int:
        if self.target is None:
            return 0
        return int(short_read(self.target, "SELECT count(*) FROM target_operations WHERE kind = 'EFFECT'")[0][0])

    def target_artifacts(self) -> int:
        if self.target is None:
            return 0
        return int(short_read(self.target, "SELECT count(*) FROM target_artifacts")[0][0])

    def target_digest(self) -> str:
        """Order-stable digest of the target's rows (the file bytes change with
        every SQLite page touch; the rows do not)."""

        import hashlib

        if self.target is None:
            return ""
        digest = hashlib.sha256()
        for sql in (
            "SELECT * FROM target_operations ORDER BY operation_key",
            "SELECT * FROM target_artifacts ORDER BY obligation_group_key",
        ):
            for row in short_read(self.target, sql):
                digest.update(repr(row).encode("utf-8"))
        return digest.hexdigest()

    def wake_causes(self) -> list[str]:
        """The cause of every materialized Wake, in storage order."""

        causes: list[str] = []
        for (blob,) in short_read(self.history, "SELECT record_bytes FROM wakeups ORDER BY rowid"):
            record = json.loads(bytes(blob))
            causes.append(str(record.get("cause")))
        return causes

    def sequences(self, kind: str) -> list[int]:
        rows = short_read(
            self.history,
            "SELECT processing_sequence FROM history_operations WHERE operation_kind = ? AND phase = 'FINALIZED' "
            "ORDER BY processing_sequence",
            (kind,),
        )
        return [int(row[0]) for row in rows]

    def records(self, table: str) -> list[dict[str, Any]]:
        """The decoded ``record_bytes`` of one record table, in storage order."""

        if table not in _TABLES:
            raise ValueError(table)
        rows = short_read(self.history, f"SELECT record_bytes FROM {table} ORDER BY rowid")
        decoded = [json.loads(bytes(row[0])) for row in rows]
        return [item for item in decoded if isinstance(item, dict)]

    def journal_events(self) -> list[dict[str, Any]]:
        """``completed`` Journal events in processing order (an ordering
        witness; a torn last line while a writer runs is ignored)."""

        found: list[dict[str, Any]] = []
        if not self.journal.is_file():
            return found
        for line in self.journal.read_bytes().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("lifecycle") == "completed":
                found.append(
                    {
                        "sequence": record.get("processing_sequence"),
                        "event_type": record.get("event_type"),
                        "event_id": record.get("event_id"),
                    }
                )
        return sorted(found, key=lambda item: int(item["sequence"] or 0))

    def journal_counts(self) -> dict[str, Any]:
        """Live Journal bytes and per-event-type / lifecycle counts (the
        budget measurement of #370 D1 "Retention").  A torn last line while a
        writer is running is ignored."""

        counts: dict[str, int] = {}
        total = 0
        paths = [self.journal, *sorted(self.journal.parent.glob(self.journal.name + ".*"))]
        live = 0
        for path in paths:
            if not path.is_file() or path.name.endswith(".lock"):
                continue
            live += path.stat().st_size
            for line in path.read_bytes().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    key = f"{record.get('event_type', record.get('record_type', '?'))}:{record.get('lifecycle', '-')}"
                    counts[key] = counts.get(key, 0) + 1
                    total += 1
        return {"live_bytes": live, "records": total, "by_event_type": dict(sorted(counts.items()))}


_TABLES = frozenset(
    {
        "history_operations", "history_decisions", "verified_outcomes", "outcome_assessments",
        "decision_feedback", "wakeups", "wake_claims", "wake_attempt_outcomes",
        "logical_cycle_resolutions", "wake_terminals", "outbox_messages", "outbox_acks",
        "plan_hold_receipts", "action_intents", "action_forecasts", "policy_verdicts",
        "approval_challenges", "execution_claims",
    }
)


# --------------------------------------------------------------------- kill


class KillPoint:
    """The coarse kill points of D1 sec. 4-3 as predicates over one consistent
    snapshot of the FINALIZED counters.  K1 is T1 only (``confirm`` writes the
    challenge and the claim in one request): it has no T2 predicate."""

    ORDER = ("K2", "K3", "K4", "K5", "K6")

    @staticmethod
    def holds(point: str, counts: Mapping[tuple[str, str], int]) -> bool:
        def done(kind: str) -> bool:
            return counts.get((kind, FINALIZED), 0) >= 1

        if point == "K2":  # EXECUTION_CLAIM visible, no Outcome row
            return done("R18_EXECUTION_CLAIM") and not done("R18_OUTCOME_COMMIT")
        if point == "K3":  # Outcome visible, no Feedback
            return done("R18_OUTCOME_COMMIT") and not done("R17_FEEDBACK_COMMIT")
        if point == "K4":  # Feedback visible, no Wake claim
            return done("R17_FEEDBACK_COMMIT") and not done("R19_WAKE_CLAIM")
        if point == "K5":  # Wake claim visible, no Decision of the cycle
            return done("R19_WAKE_CLAIM") and not done("R16_CYCLE_DECISION")
        if point == "K6":  # cycle Decision visible, no Outbox message
            return done("R16_CYCLE_DECISION") and not done("R19_OUTBOX_EMIT")
        raise ValueError(f"{point} has no T2 predicate")


# ------------------------------------------------------------------ session


def _r15_script(contradictory: bool) -> tuple[str, ...]:
    """The protocol answers (setup, E, S1, E2, S2).  ``contradictory`` makes the
    S2 answer KEEP the original referent while LIKELY_ALIGNED: the R15 status
    table (``interpretation_status_v2``) then gives CONTRADICTORY, i.e. row 2b."""

    script = list(r15_protocol_script())
    if contradictory:
        answer = json.loads(script[4])
        answer["assumed_referent"] = "B"
        answer["assessment"] = "LIKELY_ALIGNED"
        script[4] = json.dumps(answer, separators=(",", ":"), sort_keys=True)
    return tuple(script)


class TurnCapExceeded(GuardRefusal):
    """A chat turn, command or review took longer than the turn cap (a stop
    reason of a real run; the attempt counts)."""

    def __init__(self, detail: str) -> None:
        super().__init__("TURN_CAP_EXCEEDED", detail)


class S0Session:
    """One genesis subject on the process tier, driven through real routes.

    ``model`` (a real ``ModelOverride``) switches the session to a REAL model: it
    then needs the attestation of a Human launch (``launch``) and spawns its child
    through ``ChildProcess.spawn_real``; the plain ``spawn`` still refuses a real
    provider for every caller.  In a real session the R15 capture (real model text)
    goes to the private directory, never to the closed-key evidence directory.
    """

    def __init__(
        self,
        base: Path,
        *,
        profile: str = "S0",
        run_id: str | None = None,
        contradictory: bool = False,
        r19: Mapping[str, int] | None = None,
        dc: Mapping[str, Any] | None = None,
        default_answer: str | None = None,
        script: tuple[str, ...] | None = None,
        model: ModelOverride | None = None,
        launch: HumanLaunch | None = None,
        turn_cap_s: float = 120.0,
        record_calls: bool = False,
        chat_roles: bool = False,
        bare_chat: bool = False,
        bare_restore: str | None = None,
        restore_placement: str = "system",
        history_window: int | None = None,
        state_block_max_bytes: int | None = None,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.record_calls = record_calls
        self.chat_roles = chat_roles
        self.bare_chat = bare_chat
        self.bare_restore = bare_restore
        self.restore_placement = restore_placement
        self.history_window = history_window
        self.state_block_max_bytes = state_block_max_bytes
        self.real = model is not None and model.is_real
        if self.real and (launch is None or not isinstance(launch, HumanLaunch) or not launch.valid_here()):
            raise GuardRefusal("REAL_PROVIDER_WITHOUT_HUMAN_LAUNCH")
        self.model = model
        self.launch = launch
        self.turn_cap_s = turn_cap_s
        self.step_timeout_s = REAL_STEP_TIMEOUT_S if self.real else STEP_TIMEOUT_S
        self.ticket_timeout_s = max(30.0, turn_cap_s) if self.real else 30.0
        self.turn_seconds: list[float] = []
        self.subject = create_subject(
            profile,
            base=base,
            run_id=run_id,
            tier=Tier.T2,
            r19_overrides=dict(r19 if r19 is not None else (REAL_R19 if self.real else T2_R19)),
            degraded_continuation=None if dc is None else dict(dc),
            model=model,
            **({"config_overrides": dict(config_overrides)} if config_overrides else {}),
        )
        self.supervised = dc is not None and dc.get("mode") == "continue"
        self.provider_config: dict[str, Any] = {"script": list(script if script is not None else _r15_script(contradictory))}
        if default_answer is not None:
            self.provider_config["default"] = default_answer
        self.capture_dir = artifacts.private_dir(self.subject.run_dir) if self.real else self.subject.evidence_dir
        self.manifest = Manifest(self.subject.evidence_dir, self.subject.run_id, subject_root=self.subject.root)
        self.evidence = Evidence(self.subject)
        self.child: ChildProcess | None = None
        self.client: BoundClient | None = None
        self.experience_ids: list[str] = []
        self.started_at: float | None = None
        self.serving_after_s: float | None = None
        self.observed: list[dict[str, Any]] = []
        self._lives: list[ChildProcess] = []

    # -- process
    def _spawn(self) -> ChildProcess:
        if self.real:
            assert self.launch is not None and self.model is not None
            return ChildProcess.spawn_real(
                self.subject, self.launch, hf_home=self.model.hf_home, turn_cap_s=self.turn_cap_s,
                **({"record_calls": True} if self.record_calls else {}),
                **({"chat_roles": True} if self.chat_roles else {}),
                **({"bare_chat": True} if self.bare_chat else {}),
                **self._restore_arguments(),
            )
        return ChildProcess.spawn(
            self.subject, provider_config=self.provider_config, **({"record_calls": True} if self.record_calls else {}),
            **({"chat_roles": True} if self.chat_roles else {}),
            **({"bare_chat": True} if self.bare_chat else {}),
            **self._restore_arguments(),
        )

    def _restore_arguments(self) -> dict[str, Any]:
        found: dict[str, Any] = {}
        if self.bare_restore:
            found.update(bare_restore=self.bare_restore, restore_placement=self.restore_placement)
        if self.history_window is not None:
            found["history_window"] = self.history_window
        if self.state_block_max_bytes is not None:
            found["state_block_max_bytes"] = self.state_block_max_bytes
        return found

    def provider_identity(self) -> dict[str, object]:
        if self.real:
            assert self.model is not None
            return self.model.record()
        return {"mode": "scripted", "fixture_only": True}

    def start(self, *, expect: str | tuple[str, ...] = "SERVING", timeout_s: float | None = None) -> ChildProcess:
        """Spawn a child and wait until it serves (in ``continue`` mode: until
        ``/health/dc`` says ``expect``, default ``SERVING``; ``/health`` and
        ``/health/dc`` are sampled meanwhile into ``self.observed``).  The first
        call also verifies the adoption and writes the identity record."""

        first = not self._lives
        timeout_s = self.step_timeout_s if timeout_s is None else timeout_s
        child = self._spawn()
        self.child = child
        self._lives.append(child)
        self.started_at = time.monotonic()
        child.wait_ready(REAL_LOAD_TIMEOUT_S if self.real else READY_TIMEOUT_S)
        self.client = BoundClient(
            Destination.from_child(child),
            self.subject.subject_json,
            transport_factory=(lambda port: SocketTransport(port, timeout_s=self.turn_cap_s)) if self.real else SocketTransport,
        )
        self.observed = []
        if self.supervised:
            self.observed = self.observe_until(expect, timeout_s)
            wanted = (expect,) if isinstance(expect, str) else tuple(expect)
            if not self.observed or self.observed[-1].get("dc_state") not in wanted:
                raise GuardRefusal("NOT_" + "_OR_".join(wanted), str(self.observed[-3:]))
        self.serving_after_s = time.monotonic() - self.started_at
        if first:
            record = factory.verify_first_start(self.subject, provider=self.provider_identity())
            factory.write_identity_record(self.subject, record)
        return child

    def kill(self) -> None:
        assert self.child is not None
        self.child.kill_group()

    def stop(self) -> None:
        for child in self._lives:
            child.stop()

    # -- plain requests (a retryable "not yet" is retried, nothing is sent meanwhile)
    def request(self, method: str, path: str, body: Any = None, *, timeout_s: float | None = None) -> Any:
        assert self.client is not None
        deadline = time.monotonic() + (self.step_timeout_s if timeout_s is None else timeout_s)
        while True:
            try:
                return self.client.request(method, path, body)
            except DestinationNoAnswer:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
            except (TimeoutError, ConnectionError):
                # A busy child (GIL starved while it commits) can miss a 30 s connect.  Only a
                # read-only GET is retried: nothing was changed, and a POST that timed out may have been applied.
                if method.upper() != "GET" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

    def health(self) -> dict[str, Any]:
        result = self.request("GET", "/health", timeout_s=5)
        assert result.status == 200, result.text
        return dict(result.json())

    def health_dc(self) -> dict[str, Any]:
        assert self.client is not None
        result = self.client.request("GET", "/health/dc")
        assert result.status == 200, result.text
        return dict(result.json())

    def observe_until(self, state: str | tuple[str, ...], timeout_s: float) -> list[dict[str, Any]]:
        """Sample ``/health/dc`` and ``/health`` every 0.25 s until ``dc_state``
        is ``state`` (or the time is up): the answers DURING the start are the
        evidence that the process stays up and answers while a cycle builds."""

        assert self.client is not None and self.started_at is not None
        states = (state,) if isinstance(state, str) else tuple(state)
        samples: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            moment = round(time.monotonic() - self.started_at, 2)
            try:
                dc = self.health_dc()
                answer = self.client.request("GET", "/health")
                samples.append(
                    {
                        "t": moment,
                        "dc_state": dc.get("dc_state"),
                        "dc_reason": dc.get("dc_reason"),
                        "health_status": answer.status,
                    }
                )
            except (OSError, ValueError, AssertionError) as error:
                samples.append({"t": moment, "error": type(error).__name__})
            if samples[-1].get("dc_state") in states:
                break
            time.sleep(0.25)
        return samples

    def wait_state(self, state: str, timeout_s: float) -> bool:
        def reached() -> bool:
            try:
                return self.health_dc().get("dc_state") == state
            except (OSError, ValueError):
                return False

        return bool(poll_until(reached, timeout_s=timeout_s, interval_s=0.25))

    def chat(self, text: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            reply = self.request("POST", "/api/chat", {"message": text, "attachments": []})
        except TimeoutError as error:
            if self.real:
                raise TurnCapExceeded(f"no answer within {self.turn_cap_s:.0f} s") from error
            raise
        took = time.monotonic() - started
        self.turn_seconds.append(round(took, 3))
        if self.real and took > self.turn_cap_s:
            raise TurnCapExceeded(f"{took:.0f} s > {self.turn_cap_s:.0f} s")
        assert reply.status == 200, reply.text
        data: dict[str, Any] = reply.json()
        return data

    def ticket_status(self, ticket: str) -> str:
        return str(self.request("GET", f"/api/chat/commands/{ticket}").json()["status"])

    # -- the S0 chain (D1 sec. 7 items 1-5)
    def activate(self) -> dict[str, int | None]:
        assert self.client is not None
        return self.client.activate(self.subject.profile)

    def run_r15_protocol(self) -> dict[str, Any]:
        """Items 1-2: the five protocol turns with S1 admitted after the third and
        S2 after the fifth.  Returns the two reviews (revision, path label)."""

        reviews: dict[str, Any] = {}
        seal = self.subject.evidence_dir / "p1.manifest.json"  # the case parameters: no model text
        capture = self.capture_dir / "p1.capture.json"  # real runs: the private directory
        seal_case(Case(f"r20-{self.subject.run_id}", S0_TARGET, "forced", 0, "normal"), seal)

        def transport(message: str) -> dict[str, str]:
            if len(self.experience_ids) == 3:  # after E and the S1 request, BEFORE E2
                reviews["S1"] = self.review("S1")
            data = self.chat(message)
            self.experience_ids.append(str(data["experience_id"]))
            reply = {"episode_id": str(data["episode_id"]), "response": str(data["response"])}
            if self.real:  # the stored text, one line per turn, as it arrives (the model's literal text is p1.model_raw.jsonl)
                self._keep_turn(len(self.experience_ids), message, reply["response"])
            return reply

        try:
            run_trial(seal, capture, transport)
        except TrialRefused as refusal:  # the reason is coarse by contract (no transcript, path or id)
            raise GuardRefusal("P1_PROTOCOL_REFUSED", f"{refusal}; turns_answered={len(self.experience_ids)}") from refusal
        reviews["S2"] = self.review("S2")
        return reviews

    def _keep_turn(self, ordinal: int, request: str, response: str) -> None:
        path = self.capture_dir / "p1.turns.jsonl"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ordinal": ordinal, "request": request, "response": response}, ensure_ascii=False) + "\n")

    def review(self, stage: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            reply, _pid = send(
                self.subject.control_socket,
                {"cmd": "r15_review", "stage": stage},
                timeout_s=self.turn_cap_s + 10 if self.real else 60.0,
            )
        except TimeoutError as error:
            raise TurnCapExceeded(f"review {stage} did not answer") from error
        self.turn_seconds.append(round(time.monotonic() - started, 3))
        if reply.get("ok") is not True:
            raise GuardRefusal("R15_REVIEW_REFUSED", str(reply.get("error")))
        return reply

    @property
    def x(self) -> str:
        """The experience id of the E turn (the protocol's second turn)."""

        if len(self.experience_ids) < 2:
            raise GuardRefusal("X_NOT_YET")
        return self.experience_ids[1]

    def _command(self, text: str) -> str:
        reply = self.chat(text)
        return str(reply["command_ticket"]["ticket_id"])

    def adopt_goal(self, max_s: float | None = None) -> dict[str, Any]:
        """U2a INTEREST to Goal adoption: ``REVIEW_EXPERIENCE`` is re-sent (each
        attempt is its own turn) until the INTEREST exists and the ticket is
        ``adopted``.  Returns the number of refused attempts and the last reason."""

        count = 0
        reasons: list[str] = []

        def once() -> bool:
            nonlocal count
            count += 1
            ticket = self._command(f"REVIEW_EXPERIENCE:{self.x}")

            def settled() -> str | None:
                status = self.ticket_status(ticket)
                return None if status == "pending" else status

            status = poll_until(settled, timeout_s=self.ticket_timeout_s)
            if status != "adopted":
                reasons.append(str(status))
            return status == "adopted"

        if not poll_until(once, timeout_s=self.step_timeout_s if max_s is None else max_s, interval_s=0.5):
            raise GuardRefusal("GOAL_NOT_ADOPTED", ",".join(reasons))
        return {"attempts": count, "refused": reasons}

    def accept_commitment(self) -> str:
        """U2c: ``COMMIT_REVIEW_EXPERIENCE:1:<X>``; returns the settled status."""

        ticket = self._command(f"COMMIT_REVIEW_EXPERIENCE:1:{self.x}")
        status = poll_until(
            lambda: (lambda s: None if s == "pending" else s)(self.ticket_status(ticket)), timeout_s=self.ticket_timeout_s
        )
        return str(status)

    # -- approval (item 5), through the real route with the stand-in operator
    def operator(self) -> ApprovalOperator:
        assert self.client is not None
        return ApprovalOperator(self.client, self.subject.port)

    def open_approval(self) -> dict[str, Any]:
        operator = self.operator()

        def listed() -> dict[str, Any] | None:
            result = operator.listing()
            if result.status != 200:
                return None
            items = result.json().get("approvals", [])
            return dict(items[0]) if items else None

        found = poll_until(listed, timeout_s=self.step_timeout_s, interval_s=0.3)
        if not found:
            raise GuardRefusal("NO_OPEN_APPROVAL")
        return dict(found)

    def approve(self, intent_id: str) -> Any:
        return self.operator().approve(intent_id)

    # -- waiting on the store (the kill poll lives in ``kill.py``)
    def wait_for(self, predicate: Callable[[], Any], timeout_s: float, interval_s: float = 0.2) -> Any:
        return poll_until(predicate, timeout_s=timeout_s, interval_s=interval_s)

    def wait_finalized(self, kind: str, timeout_s: float | None = None) -> bool:
        return bool(self.wait_for(lambda: self.evidence.finalized(kind) >= 1, timeout_s or self.step_timeout_s))

    def drive_to_open_approval(self) -> dict[str, Any]:
        """Items 1-3 and 5 up to the open approval: genesis (already started),
        activation, R15 S1 / S2, Goal adoption, U2c acceptance, Policy ALLOW."""

        out: dict[str, Any] = {"activation": self.activate()}
        out["r15"] = self.run_r15_protocol()
        out["adopt"] = self.adopt_goal()
        out["commit"] = self.accept_commitment()
        if out["commit"] != "accepted":
            raise GuardRefusal("COMMIT_NOT_ACCEPTED", out["commit"])
        out["approval"] = self.open_approval()
        return out

    def drive_to_claim(self) -> dict[str, Any]:
        out = self.drive_to_open_approval()
        done = self.approve(out["approval"]["intent_id"])
        out["confirm"] = {"status": done.status, "body": dict(done.json()) if done.status < 500 else {}}
        return out

    def counters(self) -> dict[str, int]:
        """The FINALIZED counters that the invariants compare."""

        snapshot = self.evidence.counts()
        return {kind: snapshot.get((kind, FINALIZED), 0) for kind in KINDS}

    def finished(self) -> bool:
        return self.evidence.finalized("R19_OUTBOX_EMIT") >= 1


def new_session(base: Path, **options: Any) -> S0Session:
    return S0Session(base, **options)


__all__ = [
    "CONTINUE", "FINALIZED", "KINDS", "T2_R19", "X_TEXT", "DecisionRow", "Evidence", "KillPoint",
    "REAL_R19", "S0Session", "TurnCapExceeded", "new_session",
]
