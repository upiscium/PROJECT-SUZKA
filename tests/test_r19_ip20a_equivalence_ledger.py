"""IP-20a gate, part 2: fixtures that reach ``derive_ledger(now)`` and
``read_feedback(before_sequence)`` (IP-20 D1 r3 §9-1; see
``test_r19_ip20a_issuer_equivalence`` and ``tests/issuer_equivalence.py``).

IP-17a: the approval window closes an unclaimed Intent (row 2c).
IP-18:  TARGET_UNAVAILABLE / facts ledgers and a head sealed before the facts.
IP-19:  the Feedback witness (row 5) through the driver.
``SUZKA_ISSUER_EQUIVALENCE=1 pytest tests/`` runs the same gate over the whole suite.
"""

from __future__ import annotations

from issuer_equivalence import gate_fixture
from test_r17_ip19_flow import (  # noqa: F401
    settings,
    test_an_active_driver_turns_an_executed_outcome_into_feedback_and_row_5,
)
from test_r17_ip19_feedback_owner import (  # noqa: F401
    app,
    test_an_authorized_outcome_yields_match_self_feedback_and_row_5,
)

_dual_run = gate_fixture()
