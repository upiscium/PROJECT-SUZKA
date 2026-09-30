"""Shared finite bounds for persisted authoritative-state counters."""

from typing import Final

MAX_PERSISTED_REVISION: Final = 2**31 - 1
MAX_PERSISTED_EVENT_SEQUENCE: Final = 2**63 - 1
MAX_CALIBRATION_SAMPLE_COUNT: Final = 2**63 - 1

__all__ = [
    "MAX_CALIBRATION_SAMPLE_COUNT",
    "MAX_PERSISTED_EVENT_SEQUENCE",
    "MAX_PERSISTED_REVISION",
]
