"""Dependency-neutral Context contracts."""

from enum import StrEnum


class ContextRelation(StrEnum):
    """Pure compatibility classifications in precedence order."""

    SAME_CONTEXT = "same_context"
    PARENT_CHILD = "parent_child"
    RELATED = "related"
    SHARED_INTERLOCUTOR = "shared_interlocutor"
    LEGACY_UNKNOWN = "legacy_unknown"
    UNKNOWN_CONTEXT = "unknown_context"
    UNRELATED = "unrelated"
