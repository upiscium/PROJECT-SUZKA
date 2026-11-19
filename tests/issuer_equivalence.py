"""IP-20a: the PLAN_PREVIEW issuer equivalence gate (IP-20 D1 r3 §5-1, §9-1).

``install`` replaces ``_DecisionIssuer._issue`` with a dual run: the PRE-refactor
path (``legacy_issuer_path._Legacy``, verbatim) and the refactored one run on
the same event, against the same state, with ONE fixed clock reading shared by
both. Their results must be identical: status, Decision ID, policy row, HOLD
assessment or reason, the Decision's canonical bytes (so the sealed policy
input, ``review_intent_key`` and ``decision_id``), the participant type, the
pending state the commit validation reads, and the exception when one is
raised.  The clock is read at most once by the refactored path (it was read up
to twice, in the ledger and in the deadline check) and never more often than
before when the old path read it at all.

Byte-identity holds only with a fixed clock (D1 §5-1): with a live clock the two
reads could straddle a window or deadline boundary, as they can today.

Used by ``test_r19_ip20a_issuer_equivalence`` on the issuer's own fixtures, and
by ``conftest`` for the whole suite when ``SUZKA_ISSUER_EQUIVALENCE=1``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pytest

from legacy_issuer_path import _Legacy
from suzka.r16.decision_issuer import (
    _DecisionIssuer,
    _DecisionIssueResult,
    _PendingIssue,
)
from suzka.runtime.transaction_coordinator import CoordinatedResult


@dataclass
class Recorder:
    """One entry per dual run: ``(outcome, legacy_clock_reads, new_clock_reads)``."""

    runs: list[tuple[str, int, int]] = field(default_factory=list)

    def outcomes(self) -> set[str]:
        return {item[0] for item in self.runs}


def _normalize(value: object) -> object:
    if isinstance(value, CoordinatedResult):
        return (
            "coordinated",
            _normalize(value.value),
            tuple(type(item).__name__ for item in value.participants),
            value.transaction_kind,
        )
    if isinstance(value, _DecisionIssueResult):
        return (
            "result",
            value.status,
            value.decision_id,
            value.row,
            value.hold,
            value.owner_reason,
            None if value.decision is None else value.decision.canonical_bytes(),
            value.lineage_scope,
        )
    return ("other", repr(value))


def _pending(value: _PendingIssue | None) -> object:
    if value is None:
        return None
    return (
        value.event,
        value.qualification,
        value.incarnation,
        value.snapshot_hash,
        value.rows_digest,
        value.transaction_id,
        value.committed.canonical_bytes(),
        value.initial_state,
    )


def _label(value: object) -> str:
    if isinstance(value, CoordinatedResult):
        return "ISSUED"
    if isinstance(value, _DecisionIssueResult):
        return value.status.name
    return "RAISED"


def install(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    refactored = _DecisionIssuer._issue

    def dual(self: _DecisionIssuer, event: object, goal_id: str) -> object:
        if "_issue_qualified" in self.__dict__:
            # A test that wraps the seam itself (IP-16b's byte-identity "twin")
            # makes its own before / after comparison; a copy would inherit the
            # wrapper and compare the refactored path with itself.
            return refactored(self, event, goal_id)
        moment = self._clock()
        reads = {"legacy": 0, "new": 0}

        def clock_for(name: str):  # type: ignore[no-untyped-def]
            def read():  # type: ignore[no-untyped-def]
                reads[name] += 1
                return moment

            return read

        legacy = copy.copy(self)
        legacy.__class__ = _Legacy
        legacy._clock = clock_for("legacy")
        legacy._pending = None
        expected: tuple[str, object]
        try:
            expected = ("value", legacy._issue(event, goal_id))
        except Exception as error:  # noqa: BLE001 - compared below
            expected = ("raised", (type(error), str(error)))
        saved = self._clock
        self._clock = clock_for("new")  # type: ignore[method-assign]
        raised: Exception | None = None
        actual: tuple[str, object]
        try:
            try:
                actual = ("value", refactored(self, event, goal_id))
            except Exception as error:  # noqa: BLE001 - compared, then re-raised
                raised = error
                actual = ("raised", (type(error), str(error)))
        finally:
            self._clock = saved  # type: ignore[method-assign]
        assert expected[0] == actual[0], (expected, actual)
        if expected[0] == "value":
            assert _normalize(expected[1]) == _normalize(actual[1])
            assert _pending(legacy._pending) == _pending(self._pending)
        else:
            assert expected[1] == actual[1]
        assert reads["new"] <= 1, reads
        if reads["legacy"]:
            assert reads["new"] <= reads["legacy"], reads
        recorder.runs.append(
            (
                _label(actual[1]) if actual[0] == "value" else "RAISED",
                reads["legacy"],
                reads["new"],
            )
        )
        if raised is not None:
            raise raised
        return actual[1]

    monkeypatch.setattr(_DecisionIssuer, "_issue", dual)
    return recorder
