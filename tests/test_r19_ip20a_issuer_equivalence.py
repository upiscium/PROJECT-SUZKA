"""IP-20a: the PLAN_PREVIEW issuer refactor is byte-identical (IP-20 D1 r3 §5-1, §9-1).

The gate. Every PLAN_PREVIEW issuer call of the IP-10 fixtures (initial Decision,
EXISTING, row 4 successor, interpretation change, row 2b / row 1 HOLDs, source
drift, capacity, unverifiable head) is run twice on the same event and state with
ONE fixed clock reading: once through the pre-refactor path (``tests/
legacy_issuer_path.py``, verbatim) and once through the refactored issuer, and the
two must agree on the Decision's canonical bytes (sealed policy input, decision
id, review-intent key), the result, the pending state and any exception.  The
whole suite can be run under the same gate:

    SUZKA_ISSUER_EQUIVALENCE=1 pytest tests/

(``tests/conftest.py``), which also covers the ledger, Feedback, V2 / V3 head and
``TARGET_UNAVAILABLE`` fixtures of IP-14a / IP-16b / IP-17 / IP-18 / IP-19.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

from issuer_equivalence import Recorder, install
from suzka.r16.decision_issuer import _DecisionIssuer
from suzka.runtime.agent_runtime import AgentRuntimeExecutionError
from test_r16_ip10_decision_issuer import (  # noqa: F401  (re-run under the gate)
    _issue,
    _settings,
    app,
    test_capacity_and_slot_refusals_hold_before_prepare,
    test_commitment_acceptance_issues_the_row_4_successor,
    test_contradictory_interpretation_holds_without_a_transaction,
    test_initial_decision_is_committed_and_adopted,
    test_interpretation_change_is_a_successor_witness,
    test_source_drift_without_new_witness_holds,
    test_unavailable_interpretation_is_row_1,
    test_unverifiable_head_is_a_conflicting_predecessor,
)
from test_r16_u2a_source_owner import _app
from test_r16_u3a_plan_preview import _qualified_goal

_ALL: list[tuple[str, int, int]] = []


@pytest.fixture(autouse=True)
def _dual_run(monkeypatch: pytest.MonkeyPatch) -> Iterator[Recorder]:
    recorder = install(monkeypatch)
    yield recorder
    _ALL.extend(recorder.runs)


def test_the_gate_covered_issued_existing_and_hold_paths() -> None:
    """Runs after the re-run fixtures above (definition order): the dual run
    really saw every outcome the IP-10 fixtures produce."""

    outcomes = {item[0] for item in _ALL}
    assert {"ISSUED", "EXISTING", "HOLD"} <= outcomes, outcomes
    assert len(_ALL) >= 10
    # The refactor reads the clock exactly once per call (the old path read it 0, 1 or 2 times).
    assert {item[2] for item in _ALL} == {1}


def test_the_gate_detects_a_changed_byte(
    tmp_path, monkeypatch: pytest.MonkeyPatch, _dual_run: Recorder
) -> None:  # type: ignore[no-untyped-def]
    """The harness is not vacuous: a one-bit change in the refactored path's
    review-intent key fails the dual run."""

    real = _DecisionIssuer._committed

    def mutated(
        self, event, q, value, result, incarnation, scope, head, review_intent_key=None
    ):  # type: ignore[no-untyped-def]
        return real(self, event, q, value, result, incarnation, scope, head, "f" * 64)

    with TestClient(_app(_settings(tmp_path / "mutation"))) as client:
        state = client.app.state
        _x, goal_id = _qualified_goal(state)
        monkeypatch.setattr(_DecisionIssuer, "_committed", mutated)
        with pytest.raises(AgentRuntimeExecutionError) as error:
            _issue(state, goal_id)
        assert isinstance(error.value.__cause__, AssertionError)
