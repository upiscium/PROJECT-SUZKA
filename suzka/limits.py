"""Shared finite bounds for persisted authoritative-state counters."""

from typing import Final

MAX_PERSISTED_REVISION: Final = 2**31 - 1
MAX_PERSISTED_EVENT_SEQUENCE: Final = 2**63 - 1
MAX_CALIBRATION_SAMPLE_COUNT: Final = 2**63 - 1

# #417 ConversationTrace schema bounds, shared by the settings and the owner.
MAX_TRACE_ITEMS_TOTAL: Final = 512
MAX_TRACE_ITEMS_PER_CONTEXT: Final = 32
MAX_TRACE_CONTEXTS: Final = 64
MAX_TRACE_STABILITY_HOURS: Final = 720.0


__all__ = [
    "MAX_CALIBRATION_SAMPLE_COUNT",
    "MAX_PERSISTED_EVENT_SEQUENCE",
    "MAX_PERSISTED_REVISION",
    "MAX_TRACE_CONTEXTS",
    "MAX_TRACE_ITEMS_PER_CONTEXT",
    "MAX_TRACE_ITEMS_TOTAL",
    "MAX_TRACE_STABILITY_HOURS",
]
