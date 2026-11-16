"""The exact R16 Chat command texts (#370 D0 §2), shared without importing the driver.

The R16 driver and ConversationTrace (#417) both need to know whether a CHAT
text is a command turn. The driver imports the R16 owners, so the matcher lives
here, free of runtime imports.
"""

from __future__ import annotations

from enum import Enum
import re


_EXPERIENCE = r"experience-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_REVIEW = re.compile(rf"REVIEW_EXPERIENCE:(?P<target>{_EXPERIENCE})")
_COMMIT = re.compile(rf"COMMIT_REVIEW_EXPERIENCE:1:(?P<target>{_EXPERIENCE})")


class R16CommandKind(str, Enum):
    REVIEW_EXPERIENCE = "REVIEW_EXPERIENCE"
    COMMIT_REVIEW_EXPERIENCE = "COMMIT_REVIEW_EXPERIENCE"


def parse_command(message: str) -> tuple[R16CommandKind, str] | None:
    """The command a CHAT text is, by exact fullmatch only (#370 D0 §2)."""

    if type(message) is not str:
        return None
    match = _REVIEW.fullmatch(message)
    if match is not None:
        return R16CommandKind.REVIEW_EXPERIENCE, match.group("target")
    match = _COMMIT.fullmatch(message)
    if match is not None:
        return R16CommandKind.COMMIT_REVIEW_EXPERIENCE, match.group("target")
    return None
