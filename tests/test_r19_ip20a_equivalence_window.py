"""IP-20a gate, part 3: the IP-17a window closure under the dual run."""

from __future__ import annotations

from issuer_equivalence import gate_fixture
from test_r18_ip17a_policy import (  # noqa: F401
    app,
    test_never_reviewed_closes_at_the_window_end_to_row_2c,
    test_three_expiries_close_to_row_2c_inside_the_window,
)

_dual_run = gate_fixture()
