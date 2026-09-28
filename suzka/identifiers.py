"""Dependency-neutral validation for shared opaque identifiers."""

import re
from typing import Final


MAX_IDENTIFIER_CODEPOINTS: Final = 128
_IDENTIFIER = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._:-]{{0,{MAX_IDENTIFIER_CODEPOINTS - 1}}}\Z"
)


def validate_identifier(value: object) -> str:
    """Return an opaque identifier, rejecting every noncanonical value."""

    if type(value) is not str:
        raise TypeError("identifier must be an exact string")
    if _IDENTIFIER.fullmatch(value) is None or ".." in value:
        raise ValueError("invalid identifier")
    return value
