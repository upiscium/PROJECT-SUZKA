"""R18 IP-15: the independent read-only verifier (#380 D2 P2-2, #368 D0 B).

It reads the target only through ``TargetReader`` (``mode=ro``) and decides
from what is physically there:

- It recomputes the payload digest from the stored effect bytes itself. The
  executor's ``payload_digest`` column and its receipt are never the basis.
- AUTHORIZED needs the stored bytes, the approved digest and the ledger row
  to agree, and the ledger's ``approval_ref`` to be the nonce SubjectHistory
  recorded as consumed (the caller passes that record; IP-14a stores it).
- VERIFIED_ABSENT needs a FENCED row and no effect row; a fence that the
  sweep merely inserted is not proof by itself (#380 D2 P2-2).
- ``foreign_change`` is a target change that is not in the ledger, in the
  revision range (expected, observed] (R17 ``VerifiedOutcomeView``). It makes
  an effect UNAUTHORIZED (R20-F09: the effect cannot be attributed to SELF).

Anything it cannot classify returns None: the operation stays IN_DOUBT (or
BLOCKED) and is never retried or called ABSENT by default.
"""

from __future__ import annotations

from typing import Final

from suzka.r16.successor_policy import InverseAuthorization
from suzka.r17.contracts import ObservedState
from suzka.r18.contracts import (
    ActionIntentV1,
    ApprovalChallengeV1,
    Direction,
    InverseIntentV1,
    VerifiedOutcomeV1,
    VerifiedState,
    payload_digest,
)
from suzka.r18.target import ArtifactState, GroupStatus, LedgerKind, TargetReader

_FORWARD_REVISION: Final = 1


def _status(reader: TargetReader, group_key: str, target_instance_id: str) -> GroupStatus | None:
    if type(reader) is not TargetReader:
        raise TypeError("the verifier reads only through a TargetReader")
    status = reader.status(group_key)
    if status.meta.target_instance_id != target_instance_id:
        return None  # another target: nothing here can be about this intent
    return status


def _check_challenge(challenge: ApprovalChallengeV1, direction: Direction, intent_id: str) -> None:
    if type(challenge) is not ApprovalChallengeV1:
        raise TypeError("challenge must be an ApprovalChallengeV1")
    if challenge.direction is not direction or challenge.intent_id != intent_id:
        raise ValueError("the challenge is not this intent's")


def verify_forward(
    reader: TargetReader,
    intent: ActionIntentV1,
    challenge: ApprovalChallengeV1,
    *,
    consumed_nonce: str | None,
    outcome_revision: int,
    verifier_event_id: str,
    verifier_sequence: int,
) -> VerifiedOutcomeV1 | None:
    """The terminal forward Outcome, or None when it cannot be decided."""

    if type(intent) is not ActionIntentV1:
        raise TypeError("intent must be an ActionIntentV1")
    _check_challenge(challenge, Direction.FORWARD, intent.intent_id)
    status = _status(reader, intent.obligation_group_key, intent.target_instance_id)
    if status is None or status.forward is None:
        return None  # no effect and no fence: undecidable until the sweep fences
    forward, artifact = status.forward, status.artifact
    key = intent.forward_key

    def outcome(state: VerifiedState, **fields: object) -> VerifiedOutcomeV1:
        return VerifiedOutcomeV1(
            operation_key=key,
            direction=Direction.FORWARD,
            outcome_revision=outcome_revision,
            outcome_state=state,
            target_instance_id=intent.target_instance_id,
            verifier_event_id=verifier_event_id,
            verifier_sequence=verifier_sequence,
            **fields,  # type: ignore[arg-type]
        )

    if forward.operation_key != key:
        return None
    if forward.kind is LedgerKind.FENCED:
        if artifact is not None:
            return None  # an artifact without an effect row: mixed, not ABSENT
        return outcome(
            VerifiedState.VERIFIED_ABSENT,
            authorization=None,
            observed_state=ObservedState.ABSENT,
            observed_payload_digest=None,
            observed_revision=intent.target_expected_revision,
            effect_operation_key=None,
            foreign_change=False,
        )
    if artifact is None or artifact.state is not ArtifactState.ACTIVE or artifact.payload is None:
        return None  # an effect row whose effect is gone (or already reverted)
    stored = payload_digest(artifact.payload)
    foreign = artifact.revision != forward.revision_after or stored != forward.payload_digest
    authorized = (
        not foreign
        and consumed_nonce is not None
        and consumed_nonce == forward.approval_ref
        and stored == challenge.payload_digest
        and forward.revision_after == _FORWARD_REVISION
    )
    return outcome(
        VerifiedState.ACTIVE_VERIFIED_AUTHORIZED if authorized else VerifiedState.UNAUTHORIZED_EFFECT,
        authorization=InverseAuthorization.AUTHORIZED if authorized else InverseAuthorization.UNPROVEN,
        observed_state=ObservedState.ACTIVE,
        observed_payload_digest=stored,
        observed_revision=artifact.revision,
        effect_operation_key=key,
        foreign_change=foreign,
    )


def verify_inverse(
    reader: TargetReader,
    intent: InverseIntentV1,
    challenge: ApprovalChallengeV1,
    *,
    consumed_nonce: str | None,
    outcome_revision: int,
    verifier_event_id: str,
    verifier_sequence: int,
) -> VerifiedOutcomeV1 | None:
    """The terminal inverse Outcome (REVERTED / INVERSE_VERIFIED_ABSENT)."""

    if type(intent) is not InverseIntentV1:
        raise TypeError("intent must be an InverseIntentV1")
    _check_challenge(challenge, Direction.INVERSE, intent.intent_id)
    status = _status(reader, intent.obligation_group_key, intent.target_instance_id)
    if status is None or status.inverse is None or status.forward is None:
        return None
    forward, inverse, artifact = status.forward, status.inverse, status.artifact
    if (
        forward.kind is not LedgerKind.EFFECT
        or forward.operation_key != intent.forward_key
        or inverse.operation_key != intent.inverse_key
        or artifact is None
    ):
        return None

    def outcome(state: VerifiedState, **fields: object) -> VerifiedOutcomeV1:
        return VerifiedOutcomeV1(
            operation_key=intent.inverse_key,
            direction=Direction.INVERSE,
            outcome_revision=outcome_revision,
            outcome_state=state,
            target_instance_id=intent.target_instance_id,
            verifier_event_id=verifier_event_id,
            verifier_sequence=verifier_sequence,
            **fields,  # type: ignore[arg-type]
        )

    if inverse.kind is LedgerKind.FENCED:
        if artifact.state is not ArtifactState.ACTIVE or artifact.payload is None:
            return None
        stored = payload_digest(artifact.payload)
        return outcome(
            VerifiedState.INVERSE_VERIFIED_ABSENT,
            authorization=None,
            observed_state=ObservedState.ACTIVE,
            observed_payload_digest=stored,
            observed_revision=artifact.revision,
            effect_operation_key=intent.forward_key,
            foreign_change=artifact.revision != forward.revision_after or stored != forward.payload_digest,
        )
    if artifact.state is not ArtifactState.REVERTED:
        return None  # an inverse effect row while the effect is still ACTIVE
    authorized = (
        consumed_nonce is not None
        and consumed_nonce == inverse.approval_ref
        and inverse.payload_digest == challenge.payload_digest == intent.forward_payload_digest
    )
    return outcome(
        VerifiedState.REVERTED,
        authorization=InverseAuthorization.AUTHORIZED if authorized else InverseAuthorization.UNPROVEN,
        observed_state=ObservedState.ABSENT,
        observed_payload_digest=None,
        observed_revision=artifact.revision,
        effect_operation_key=None,
        foreign_change=artifact.revision != inverse.revision_after,
    )


__all__ = ["verify_forward", "verify_inverse"]
