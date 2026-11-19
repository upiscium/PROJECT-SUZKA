"""R18 IP-18b: target facts and the ledger witnesses derived from them
(#411 D1 r4 §9-2, §9-4, §9-5), as pure functions.

Nothing here reads a clock, a file or process state.  ``classify_forward``
maps one group's target rows to a fact; ``compute_target_facts`` maps a
target snapshot plus the keys that have a Claim in the current incarnation to
``TargetFacts``; ``facts_ledger`` turns a fact into the ``LedgerRead`` the
Decision issuer seals (V3 only, row 2c); ``with_target_blocked`` is the
last-step TARGET_UNAVAILABLE substitution.  The sweep that feeds these, and
the issuer that consults them, follow IP-18a's merge.

Facts take precedence over the prior-claim path once complete: a key present
in ``TargetFacts`` is answered from the facts and ``derive_ledger`` is
skipped; a key with a prior claim and no fact keeps CLAIMED(prior), row 2.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from suzka.r16.contracts import checksum

from suzka.r16.successor_policy import LedgerState, PolicyRef
from suzka.r18.contracts import Direction, operation_key
from suzka.r18.ledger import NO_LEDGER, LedgerRead
from suzka.r18.target import ArtifactRow, LedgerKind, LedgerRow
from suzka.r18.verifier import fenced_without_artifact

TARGET_FENCE_REF_KIND: Final = "r18-target-fence"
TARGET_EFFECT_REF_KIND: Final = "r18-target-effect"
TARGET_BLOCKED_REF_KIND: Final = "r18-target-blocked"
# Constant on purpose (§9-5 "no churn"): nothing in the blocked ref is time- or
# process-based, so the evidence fingerprint does not move while it holds.
TARGET_BLOCKED_REF_ID: Final = "target"
_REF_REVISION: Final = 1
_DOMAIN: Final = b"suzka.r18.target-facts.v1\x00"


class TargetFact(str, Enum):
    EFFECT_NO_CLAIM = "EFFECT_NO_CLAIM"
    FENCED_NO_CLAIM = "FENCED_NO_CLAIM"


@dataclass(frozen=True, slots=True)
class TargetFacts:
    """Forward key -> fact, for keys the target decides without a Claim of
    this incarnation.  A key that is absent has no fact (NONE, undecidable /
    mixed, or still a prior claim awaiting the sweep)."""

    by_key: Mapping[str, TargetFact]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_key", dict(self.by_key))

    def get(self, forward_key: str) -> TargetFact | None:
        return self.by_key.get(forward_key)

    def __bool__(self) -> bool:
        return bool(self.by_key)

    def __len__(self) -> int:
        return len(self.by_key)


EMPTY_FACTS: Final = TargetFacts({})


def classify_forward(forward: LedgerRow | None, artifact: ArtifactRow | None) -> TargetFact | None:
    """One group's forward rows -> a fact, or None (no row, or undecidable).

    FENCED + no artifact -> FENCED_NO_CLAIM, through the verifier's own
    predicate.  EFFECT + an ACTIVE-or-reverted artifact -> EFFECT_NO_CLAIM; an
    EFFECT row whose artifact is gone and a FENCED row with an artifact are
    mixed: None (the verifier's "key-level BLOCKED", never a guess)."""

    if forward is None or forward.direction is not Direction.FORWARD:
        return None
    if forward.kind is LedgerKind.FENCED:
        return TargetFact.FENCED_NO_CLAIM if fenced_without_artifact(forward, artifact) else None
    if forward.kind is LedgerKind.EFFECT and artifact is not None:
        return TargetFact.EFFECT_NO_CLAIM
    return None


def compute_target_facts(
    snapshot: Iterable[tuple[LedgerRow, ArtifactRow | None]],
    claimed_keys: Collection[str],
) -> TargetFacts:
    """The facts of a settled target snapshot.

    ``claimed_keys`` are the forward keys with a Claim in the CURRENT
    incarnation (adopted or not, with or without an Outcome): those are never
    a fact (their state comes from the Outcome path, so a freshly written
    effect of this process is never misread as prior).  Keys whose only
    Claim belongs to a prior incarnation are classified by the target row
    like any other.  The result depends on the rows and the key set only."""

    claimed = frozenset(claimed_keys)
    found: dict[str, TargetFact] = {}
    for row, artifact in snapshot:
        if row.direction is not Direction.FORWARD:
            continue
        if row.operation_key != operation_key(row.obligation_group_key, Direction.FORWARD):
            continue  # not this group's forward key: no fact from a foreign row
        if row.operation_key in claimed or row.operation_key in found:
            continue
        fact = classify_forward(row, artifact)
        if fact is not None:
            found[row.operation_key] = fact
    return TargetFacts(found)


def _ref(kind: str, forward_key: str, *parts: object) -> PolicyRef:
    return PolicyRef(
        kind,
        forward_key,
        _REF_REVISION,
        checksum(_DOMAIN, {"kind": kind, "forward_key": forward_key, "parts": list(parts)}),
    )


def facts_ledger(facts: TargetFacts, forward_key: str) -> LedgerRead | None:
    """§9-2: the ledger witness for a key the facts decide, else None.

    FENCED_NO_CLAIM -> VERIFIED_ABSENT (ref ``r18-target-fence``, no Outcome);
    EFFECT_NO_CLAIM -> PRIOR_INCARNATION_EFFECT_PRESENT (ref
    ``r18-target-effect``).  Both are row 2c under V3."""

    fact = facts.get(forward_key)
    if fact is TargetFact.FENCED_NO_CLAIM:
        return LedgerRead(
            LedgerState.VERIFIED_ABSENT,
            (_ref(TARGET_FENCE_REF_KIND, forward_key, fact.value),),
            None,
            None,
        )
    if fact is TargetFact.EFFECT_NO_CLAIM:
        return LedgerRead(
            LedgerState.PRIOR_INCARNATION_EFFECT_PRESENT,
            (_ref(TARGET_EFFECT_REF_KIND, forward_key, fact.value),),
            None,
            None,
        )
    return None


def target_blocked_ref() -> PolicyRef:
    """The constant ``r18-target-blocked`` ref (§9-5): id ``target``,
    revision 1, digest of the fixed reason code."""

    return PolicyRef(
        TARGET_BLOCKED_REF_KIND,
        TARGET_BLOCKED_REF_ID,
        _REF_REVISION,
        checksum(_DOMAIN, {"kind": TARGET_BLOCKED_REF_KIND, "reason": "TARGET_BLOCKED"}),
    )


TARGET_UNAVAILABLE_LEDGER: Final = LedgerRead(
    LedgerState.TARGET_UNAVAILABLE, (target_blocked_ref(),), None, None
)


def with_target_blocked(
    real: LedgerRead, *, blocked: bool, has_obligation: bool, has_prior_claim: bool
) -> LedgerRead:
    """§9-5, substitute LAST and only over an empty ledger: the real ledger is
    returned unchanged unless it is ``NONE`` with no prior claim, the target is
    blocked and the Decision has an obligation.  Never replaces CLAIMED,
    ACTIVE_VERIFIED_AUTHORIZED or a ledger that carries Feedback."""

    if blocked and has_obligation and not has_prior_claim and real == NO_LEDGER:
        return TARGET_UNAVAILABLE_LEDGER
    return real


__all__ = [
    "EMPTY_FACTS",
    "TARGET_BLOCKED_REF_KIND",
    "TARGET_EFFECT_REF_KIND",
    "TARGET_FENCE_REF_KIND",
    "TARGET_UNAVAILABLE_LEDGER",
    "TargetFact",
    "TargetFacts",
    "classify_forward",
    "compute_target_facts",
    "facts_ledger",
    "target_blocked_ref",
    "with_target_blocked",
]
