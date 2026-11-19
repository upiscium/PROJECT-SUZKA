"""IP-20a gate, part 4: the IP-18 TARGET_UNAVAILABLE / facts ledgers under the dual run."""

from __future__ import annotations

from issuer_equivalence import gate_fixture
from test_r18_ip18b_facts_sweep import (  # noqa: F401
    settings,
    test_a_blocked_target_gives_plan_a_over_an_empty_ledger_and_clears_with_hysteresis,
    test_a_row_4_head_sealed_before_facts_gets_a_new_row_2c_decision_not_a_failure,
)

_dual_run = gate_fixture()
