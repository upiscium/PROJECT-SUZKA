"""C10 SQLite-layer digests shared by the attestation provider and the reader.

#376 C10 D3 §1/§2.  Pure functions of SubjectHistory rows and adopted chain
tips; the provider computes them inside ``compact()``, the reader recomputes
them from SQLite after the archive and requires equality.  No authority.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

from suzka.r16.history_codec import CommittedDecisionV1
from suzka.r16.history_store import HistoryOperationRow

_OPERATIONS_DOMAIN: Final = b"PROJECT-SUZKA:R16:HISTORY-ATTESTED-OPERATIONS:V1\0"
_SCOPES_DOMAIN: Final = b"PROJECT-SUZKA:R16:HISTORY-ATTESTED-SCOPES:V1\0"


@dataclass(frozen=True, slots=True)
class ChainNode:
    decision: CommittedDecisionV1
    transaction_id: str
    sequence: int


def _digest(domain: bytes, value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(domain + payload).hexdigest()


def operations_digest(rows: Iterable[HistoryOperationRow]) -> tuple[int, str]:
    """Count and digest of the attested operation rows (any phase/kind)."""

    entries = sorted(
        [
            row.processing_sequence,
            row.transaction_id,
            row.event_id,
            row.operation_kind.value,
            row.phase.value,
            row.operation_digest,
        ]
        for row in rows
    )
    return len(entries), _digest(_OPERATIONS_DOMAIN, entries)


def scopes_digest(tips: dict[str, tuple[ChainNode, int]]) -> tuple[int, str]:
    """Count and digest of (scope, tip identity, depth) for attested prefixes."""

    entries = sorted(
        [scope, node.decision.decision_id, node.decision.content_digest, depth]
        for scope, (node, depth) in tips.items()
    )
    return len(entries), _digest(_SCOPES_DOMAIN, entries)


def prefix_tips(
    chains: dict[str, tuple[ChainNode, ...]], high_water: int
) -> dict[str, tuple[ChainNode, int]]:
    """The tip and depth of each chain's prefix with sequence <= high_water.

    A child is always issued after its predecessor, so the prefix is the
    chain cut at the first node above the high-water.
    """

    tips: dict[str, tuple[ChainNode, int]] = {}
    for scope, chain in chains.items():
        prefix = [node for node in chain if node.sequence <= high_water]
        if prefix:
            if prefix != list(chain[: len(prefix)]):
                raise ValueError("attested chain prefix is not contiguous")
            tips[scope] = (prefix[-1], len(prefix))
    return tips


__all__ = ["ChainNode", "operations_digest", "prefix_tips", "scopes_digest"]
