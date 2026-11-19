"""IP-16b: the Decision issuer's operation ledger (#411 D1 r3 §5, §10).

Pure: from the adopted R17 / R18 records of the current incarnation (one
``SubjectHistoryReader.classify()``, the same one the lineage head came from),
one obligation group key and one clock read, derive the SuccessorPolicy
ledger witness: ``LedgerState``, ``ledger_refs`` (at most
``MAX_LEDGER_REFS``, sorted and unique by (kind, id), §10-3),
``inverse_authorization`` and ``verified_outcome_ref``.  No store or Journal;
the clock is the caller's (IP-17a, #411 D1 r7 §3a / §12-3).

Records of *every* lineage count: the group is the obligation, not the
lineage (r2 P1-3).  A Claim of another incarnation (its ``claim:`` key is
live but not this incarnation's) is CLAIMED with one ``r18-prior-claim`` ref
built from the raw ``execution_claims`` row behind the key; a missing or
mismatched row is CORRUPT (§10-2).  An unclaimed Intent of another
incarnation is never seen here (A1-5).

Forward table (latest state):

- no record of the group                          -> NONE
- Intent (+ forecast, + ALLOW, + open challenge)  -> INTENT_ISSUED
- + DENY                                          -> POLICY_DENIED
- + expiries < MAX_CHALLENGES_PER_INTENT          -> INTENT_ISSUED
- + expiries = MAX_CHALLENGES_PER_INTENT          -> APPROVAL_EXPIRED_UNCLAIMED
- no Claim and ``now >= approval_window_end_utc`` -> APPROVAL_EXPIRED_UNCLAIMED
  (IP-17a, HD-13 (a): closure derived from the records and the clock, with
  no event; a DENY stays POLICY_DENIED)
- + Claim, no Outcome                             -> CLAIMED
- + forward Outcome                               -> its state (latest revision)
- an inconsistent set                             -> CORRUPT

Inverse: the same table after a forward ACTIVE_VERIFIED_AUTHORIZED or
UNAUTHORIZED_EFFECT, with each state's INVERSE_* counterpart; REVERTED takes
``inverse_authorization`` from the inverse Outcome's ``authorization``.

``ledger_refs``: forward = the Intent, the latest of (verdict, challenge,
expiry) and the Claim; inverse = the forward Claim, the inverse Intent, the
inverse Claim and the latest inverse (verdict, challenge, expiry).  The
forward Outcome is only ever ``verified_outcome_ref`` (kind ``r18-outcome``),
never a ledger ref, so ``reference_identity_reused`` cannot fire (r2 P1-2).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, TypeVar

from suzka.r16.history_records import (
    MAX_CHALLENGES_PER_INTENT,
    ApprovalExpirationV1,
)
from suzka.r16.successor_policy import (
    MAX_LEDGER_REFS,
    FeedbackState,
    InverseAuthorization,
    LedgerState,
    PolicyRef,
)
from suzka.r17.contracts import ExpectedObservationV1
from suzka.r18.contracts import (
    OUTCOME_REF_KIND,
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    ExecutionClaimV1,
    InverseIntentV1,
    PolicyVerdict,
    PolicyVerdictV1,
    VerifiedOutcomeV1,
    ledger_state_of,
    operation_key,
)

INTENT_REF_KIND: Final = "r18-intent"
VERDICT_REF_KIND: Final = "r18-verdict"
CHALLENGE_REF_KIND: Final = "r18-challenge"
EXPIRY_REF_KIND: Final = "r18-expiry"
CLAIM_REF_KIND: Final = "r18-claim"
PRIOR_CLAIM_REF_KIND: Final = "r18-prior-claim"
# Immutable records carry no revision of their own; a ref names revision 1.
_RECORD_REVISION: Final = 1
_T = TypeVar("_T")

_INVERSE_OF: Final[Mapping[LedgerState, LedgerState]] = {
    LedgerState.INTENT_ISSUED: LedgerState.INVERSE_INTENT_ISSUED,
    LedgerState.POLICY_DENIED: LedgerState.INVERSE_POLICY_DENIED,
    LedgerState.APPROVAL_EXPIRED_UNCLAIMED: LedgerState.INVERSE_APPROVAL_EXPIRED_UNCLAIMED,
    LedgerState.CLAIMED: LedgerState.INVERSE_CLAIMED,
}
_INVERTIBLE: Final = frozenset(
    {LedgerState.ACTIVE_VERIFIED_AUTHORIZED, LedgerState.UNAUTHORIZED_EFFECT}
)


@dataclass(frozen=True, slots=True)
class LedgerRead:
    """The ledger witness the Decision issuer seals."""

    state: LedgerState
    refs: tuple[PolicyRef, ...]
    inverse_authorization: InverseAuthorization | None
    verified_outcome_ref: PolicyRef | None
    # IP-19: the R17 Feedback witness.  ``derive_ledger`` never sets it (the
    # ledger is R18's); the Decision issuer fills it from ``read_feedback``,
    # so every pre-IP-19 value keeps its ABSENT / None default.
    feedback: FeedbackState = FeedbackState.ABSENT
    feedback_ref: PolicyRef | None = None


# The production witness while the store holds no R17 / R18 data (§5): the
# exact values the issuer sealed before IP-16b.
NO_LEDGER: Final = LedgerRead(LedgerState.NONE, (), None, None)
CORRUPT_LEDGER: Final = LedgerRead(LedgerState.CORRUPT, (), None, None)


@dataclass(frozen=True, slots=True)
class RecordAt:
    """One adopted record and the processing sequence that ordered it."""

    record: object
    sequence: int


class _Corrupt(Exception):
    """An inconsistent record set for the group (row 0)."""


def prior_claim_ref(row: Mapping[str, object] | None, forward_key: str) -> PolicyRef | None:
    """§5 / §10-2: the ``r18-prior-claim`` ref from the raw ``execution_claims``
    row behind another incarnation's live ``claim:`` key, or None when that
    row is missing or does not match the key (the caller maps None to
    CORRUPT)."""

    if row is None:
        return None
    try:
        payload = row.get("record_bytes")
        if type(payload) is not bytes:
            return None
        claim = ExecutionClaimV1.from_json(payload)
    except (TypeError, ValueError):
        return None
    if (
        claim.direction is not Direction.FORWARD
        or claim.operation_key != forward_key
        or row.get("operation_key") != forward_key
        or row.get("record_id") != claim.claim_id
        or row.get("record_digest") != claim.claim_digest
    ):
        return None
    return PolicyRef(PRIOR_CLAIM_REF_KIND, claim.claim_id, _RECORD_REVISION, claim.claim_digest)


def derive_ledger(
    records: Iterable[RecordAt],
    group_key: str,
    now: datetime,
    *,
    prior_claim: PolicyRef | None = None,
) -> LedgerRead:
    """The §5 table for one obligation group (pure, total, fail-closed).

    ``now`` (timezone-aware) closes an unclaimed forward Intent whose
    approval window has ended (IP-17a §3a)."""

    if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("the ledger clock must be timezone-aware")
    try:
        return _derive(tuple(records), group_key, prior_claim, now)
    except _Corrupt:
        return CORRUPT_LEDGER


@dataclass(frozen=True, slots=True)
class _Side:
    state: LedgerState
    refs: tuple[PolicyRef, ...]
    claim: PolicyRef | None
    outcome: VerifiedOutcomeV1 | None
    outcomes: tuple[VerifiedOutcomeV1, ...]


def _derive(
    records: tuple[RecordAt, ...],
    group_key: str,
    prior_claim: PolicyRef | None,
    now: datetime,
) -> LedgerRead:
    forward_key = operation_key(group_key, Direction.FORWARD)
    inverse_key = operation_key(group_key, Direction.INVERSE)
    forward = _side(records, Direction.FORWARD, group_key, forward_key, now)
    inverse = _side(records, Direction.INVERSE, group_key, inverse_key, now)
    if prior_claim is not None:
        # Another incarnation claimed this group's forward effect.  This
        # incarnation's writers refuse a second forward obligation for it, so
        # any forward record here is inconsistent.  This incarnation's
        # inverse records are deliberately ignored here (IP-16b
        # interpretation 3): a successor incarnation may compensate the prior
        # effect (the inverse gate is global), and CLAIMED holds the lineage at
        # row 2 / row 0 just as CORRUPT would, so both are a HOLD.  Without a
        # prior Claim, an inverse with no forward side is CORRUPT (below).
        if forward is not None:
            raise _Corrupt
        return LedgerRead(LedgerState.CLAIMED, (prior_claim,), None, None)
    if forward is None:
        if inverse is not None:
            raise _Corrupt
        return NO_LEDGER
    outcome_ref = (
        None
        if forward.outcome is None
        else PolicyRef(
            OUTCOME_REF_KIND,
            forward.outcome.verified_outcome_id,
            forward.outcome.outcome_revision,
            forward.outcome.outcome_digest,
        )
    )
    if inverse is None:
        return LedgerRead(forward.state, _sorted(forward.refs), None, outcome_ref)
    if forward.state not in _INVERTIBLE or forward.claim is None:
        raise _Corrupt
    inverse_intent = _of(
        records, InverseIntentV1, lambda item: item.obligation_group_key == group_key
    )[0][0]
    if not any(
        (
            outcome.verified_outcome_id,
            outcome.outcome_revision,
            outcome.outcome_digest,
            outcome.outcome_state,
        )
        == (
            inverse_intent.forward_outcome_id,
            inverse_intent.forward_outcome_revision,
            inverse_intent.forward_outcome_digest,
            inverse_intent.forward_outcome_state,
        )
        for outcome in forward.outcomes
    ):
        raise _Corrupt
    if inverse.outcome is not None:
        state = ledger_state_of(inverse.outcome)
    else:
        state = _INVERSE_OF[inverse.state]
    authorization = None
    if state is LedgerState.REVERTED:
        assert inverse.outcome is not None
        authorization = inverse.outcome.authorization
    refs = (forward.claim, *inverse.refs)
    return LedgerRead(state, _sorted(refs), authorization, outcome_ref)


def _of(
    records: tuple[RecordAt, ...], kind: type[_T], match: Callable[[_T], bool]
) -> list[tuple[_T, int]]:
    """The records of one exact type that match, with their sequence."""

    found: list[tuple[_T, int]] = []
    for item in records:
        record = item.record
        if type(record) is kind:
            assert isinstance(record, kind)
            if match(record):
                found.append((record, item.sequence))
    return found


def _side(
    records: tuple[RecordAt, ...],
    direction: Direction,
    group_key: str,
    key: str,
    now: datetime,
) -> _Side | None:
    """One direction of the group, or None when it has no record at all."""

    intents: list[tuple[ActionIntentV1 | InverseIntentV1, int]]
    if direction is Direction.FORWARD:
        intents = list(_of(records, ActionIntentV1, lambda item: item.obligation_group_key == group_key))
    else:
        intents = list(_of(records, InverseIntentV1, lambda item: item.obligation_group_key == group_key))
    ids = {intent.intent_id for intent, _ in intents}

    def ours(item: PolicyVerdictV1 | ApprovalChallengeV1 | ExecutionClaimV1) -> bool:
        return item.direction is direction and (item.operation_key == key or item.intent_id in ids)

    verdicts = _of(records, PolicyVerdictV1, ours)
    challenges = _of(records, ApprovalChallengeV1, ours)
    claims = _of(records, ExecutionClaimV1, ours)
    outcomes = _of(
        records,
        VerifiedOutcomeV1,
        lambda item: item.direction is direction and item.operation_key == key,
    )
    expiries = _of(
        records,
        ApprovalExpirationV1,
        lambda item: item.direction is direction and item.intent_id in ids,
    )
    if not intents:
        if verdicts or challenges or claims or outcomes or expiries:
            raise _Corrupt
        return None
    if len(intents) != 1:
        raise _Corrupt
    intent = intents[0][0]
    intent_key = intent.inverse_key if isinstance(intent, InverseIntentV1) else intent.forward_key
    if intent_key != key:
        raise _Corrupt
    linked: list[PolicyVerdictV1 | ApprovalChallengeV1 | ExecutionClaimV1] = [
        *(record for record, _ in verdicts),
        *(record for record, _ in challenges),
        *(record for record, _ in claims),
    ]
    for record in linked:
        if (
            record.intent_id != intent.intent_id
            or record.intent_digest != intent.intent_digest
            or record.operation_key != key
        ):
            raise _Corrupt
    if direction is Direction.FORWARD:
        forecasts = _of(
            records,
            ExpectedObservationV1,
            lambda item: item.action_intent_id == intent.intent_id,
        )
        if len(forecasts) > 1:
            raise _Corrupt
    if len(verdicts) > 1 or len(claims) > 1:
        raise _Corrupt
    verdict = verdicts[0][0] if verdicts else None
    allowed = verdict is not None and verdict.verdict is PolicyVerdict.ALLOW
    issued = {record.challenge_digest for record, _ in challenges}
    expired = {record.challenge_digest for record, _ in expiries}
    if (
        len(issued) != len(challenges)
        or len(expired) != len(expiries)
        or len(challenges) > MAX_CHALLENGES_PER_INTENT
        or (challenges and not allowed)
        or not expired <= issued
    ):
        raise _Corrupt
    claim = claims[0][0] if claims else None
    if claim is not None and (
        not allowed or claim.challenge_digest not in issued or claim.challenge_digest in expired
    ):
        raise _Corrupt
    if outcomes and claim is None:
        raise _Corrupt
    revisions = [record.outcome_revision for record, _ in outcomes]
    if len(revisions) != len(set(revisions)):
        raise _Corrupt
    ordered = tuple(record for record, _ in sorted(outcomes, key=lambda pair: pair[0].outcome_revision))
    latest_outcome = ordered[-1] if ordered else None
    if latest_outcome is not None:
        state = ledger_state_of(latest_outcome)
    elif claim is not None:
        state = LedgerState.CLAIMED
    elif verdict is not None and not allowed:
        state = LedgerState.POLICY_DENIED
    elif len(expiries) >= MAX_CHALLENGES_PER_INTENT:
        state = LedgerState.APPROVAL_EXPIRED_UNCLAIMED
    elif isinstance(intent, ActionIntentV1) and not intent.window_open(now):
        # IP-17a §3a (HD-13 (a)): the approval window ended unclaimed.
        state = LedgerState.APPROVAL_EXPIRED_UNCLAIMED
    else:
        state = LedgerState.INTENT_ISSUED
    refs: list[PolicyRef] = [
        PolicyRef(INTENT_REF_KIND, intent.intent_id, _RECORD_REVISION, intent.intent_digest)
    ]
    # "Latest of (verdict, challenge, expiry)": the highest processing
    # sequence; on an equal sequence the later stage wins, pinned as
    # expiry > challenge > verdict.
    stages: list[tuple[object, int, int]] = [
        *((record, sequence, 0) for record, sequence in verdicts),
        *((record, sequence, 1) for record, sequence in challenges),
        *((record, sequence, 2) for record, sequence in expiries),
    ]
    if stages:
        refs.append(_stage_ref(max(stages, key=lambda item: (item[1], item[2]))[0]))
    claim_ref = None
    if claim is not None:
        claim_ref = PolicyRef(CLAIM_REF_KIND, claim.claim_id, _RECORD_REVISION, claim.claim_digest)
        refs.append(claim_ref)
    return _Side(state, tuple(refs), claim_ref, latest_outcome, ordered)


def _stage_ref(record: object) -> PolicyRef:
    if type(record) is PolicyVerdictV1:
        return PolicyRef(VERDICT_REF_KIND, record.verdict_digest, _RECORD_REVISION, record.verdict_digest)
    if type(record) is ApprovalChallengeV1:
        return PolicyRef(
            CHALLENGE_REF_KIND, record.challenge_digest, _RECORD_REVISION, record.challenge_digest
        )
    assert type(record) is ApprovalExpirationV1
    return PolicyRef(
        EXPIRY_REF_KIND, record.challenge_digest, _RECORD_REVISION, record.expiration_digest
    )


def _sorted(refs: Iterable[PolicyRef]) -> tuple[PolicyRef, ...]:
    """§10-3: sorted and unique by (kind, id), within MAX_LEDGER_REFS."""

    items = tuple(refs)
    keys = [ref.sort_key for ref in items]
    if len(set(keys)) != len(keys) or len(items) > MAX_LEDGER_REFS:
        raise _Corrupt
    return tuple(sorted(items, key=lambda ref: ref.sort_key))


__all__ = [
    "CHALLENGE_REF_KIND",
    "CLAIM_REF_KIND",
    "CORRUPT_LEDGER",
    "EXPIRY_REF_KIND",
    "INTENT_REF_KIND",
    "NO_LEDGER",
    "PRIOR_CLAIM_REF_KIND",
    "VERDICT_REF_KIND",
    "LedgerRead",
    "RecordAt",
    "derive_ledger",
    "prior_claim_ref",
]
