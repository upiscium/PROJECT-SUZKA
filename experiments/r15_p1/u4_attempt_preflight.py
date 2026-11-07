"""TEST ONLY: source-backed first-person task-attempt candidate for R15 U4.

The existing U1 SelfModel schema can hold "I attempted referent resolution"
without claiming task success, capability, verified competence, personality or
identity. This evaluator constructs a DETACHED ROOT only: no U4 owner method,
AgentState v10 commit, WAL, event/RevisionProof, production claim or setter.

Source owner qualification is cooperative; it is not historical SourceEpoch.
"""

from __future__ import annotations

from dataclasses import dataclass

from suzka.r15.common import (
    Accessibility,
    Inertia,
    Interpretation,
    InterpretationStatus,
    SCORE_SCALE,
    checksum,
)
from suzka.r15.contracts import (
    CapabilityHypothesis,
    SelfModelState,
    VerifiedCompetence,
)
from suzka.r15.narrative_self_system import NarrativeSelfSystem
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
    QualifiedTurnRef,
    _LATER,
    _PRIOR,
    _SETUP,
    _parse_candidate,
    restore_scoped_option,
)


TASK_CLASS = "task:referent-resolution"
_ATTEMPT_DOMAIN = b"PROJECT-SUZKA:R15:U4:TEST-ONLY-OBSERVED-ATTEMPT:V1\x00"


class SelfAttemptUnavailable(ValueError):
    """Source or narrow meaning not qualified: do not materialize candidate."""


@dataclass(frozen=True, slots=True)
class DetachedSelfAttempt:
    """Not a trustworthy U4 owner state or real learned ability."""

    root: SelfModelState
    event_count: int
    original_interpretation: str
    current_interpretation: str
    corrected_by_user_statement: bool
    admission_to_u4: bool = False


def inspect_bounded_referent_attempt(
    gate: QualifiedNarrativeSourceGate,
    narrative_owner: NarrativeSelfSystem,
    source_refs: tuple[QualifiedTurnRef, ...],
) -> DetachedSelfAttempt:
    """Qualify one observable attempted answer and detached SelfModel U1 value.

    The correct admission boundary still requires a distinct, Human-approved
    subject-owned U4 event + WAL internal commit. This is NOT that boundary.
    """
    if (
        type(gate) is not QualifiedNarrativeSourceGate
        or type(narrative_owner) is not NarrativeSelfSystem
        or narrative_owner._source_gate is not gate
        or type(source_refs) is not tuple
        or len(source_refs) != 5
        or any(type(ref) is not QualifiedTurnRef for ref in source_refs)
    ):
        raise SelfAttemptUnavailable("source-qualified candidate unavailable")
    try:
        narrative = narrative_owner.snapshot()
        if narrative.revision != 2:
            raise SelfAttemptUnavailable("narrative has not admitted later evidence")
        observations = tuple(gate._read(ref) for ref in source_refs)
        sequences = tuple(item.event.event_sequence for item in observations)
        if (
            sequences != tuple(sorted(set(sequences)))
            or len({item.context_id for item in observations}) != 1
            or not _SETUP.fullmatch(observations[0].request)
        ):
            raise SelfAttemptUnavailable("source sequence or Context inconsistent")
        original_answer = observations[1].response.replace("*", "")
        attempted = {
            name for pattern in _PRIOR for name in pattern.findall(original_answer)
        }
        if len(attempted) != 1:
            raise SelfAttemptUnavailable("actual first-person attempt is ambiguous")
        chosen = next(iter(attempted))
        if _parse_candidate(observations[2].response, "S1") != chosen:
            raise SelfAttemptUnavailable("S1 differs from actual observed answer")
        reported = _LATER.fullmatch(observations[3].request)
        latest = _parse_candidate(observations[4].response, "S2")
        if reported is None or latest == chosen or latest != reported.group(1):
            raise SelfAttemptUnavailable("no bounded later correction")
        first = tuple(ep for ep in narrative.episodes if not ep.links)
        last = tuple(ep for ep in narrative.episodes if ep.links)
        if (
            len(first) != 1
            or len(last) != 1
            or first[0].experience != observations[1].witness
            or last[0].experience != observations[3].witness
            or restore_scoped_option(
                observations[0].episode_id,
                first[0].interpretation.statement_id,
            )
            != chosen
            or restore_scoped_option(
                observations[0].episode_id,
                last[0].interpretation.statement_id,
            )
            != latest
        ):
            raise SelfAttemptUnavailable("U3 semantic claim lacks actual sources")
        witness = observations[1].witness
        # SCORE_SCALE for occurrence means "a Chat attempt WAS observed", not
        # the task success rate or the model's competence.
        statement_id = checksum(
            _ATTEMPT_DOMAIN,
            {
                "task_class": TASK_CLASS,
                "observed_attempt": witness.reference,
                "later_reported_correction": observations[3].witness.reference,
            },
        )
        attempt = Interpretation(
            statement_id,
            InterpretationStatus.PRESENT,
            SCORE_SCALE,
            SCORE_SCALE,
            (witness,),
            (),
            Accessibility.CURRENT,
            Inertia.TENTATIVE,
        )
        candidate = CapabilityHypothesis(
            TASK_CLASS, attempt, VerifiedCompetence.UNKNOWN
        )
        # revision=0 is INTENTIONAL: no synthetic RevisionProof is presented
        # as a real U4 actor event. A future trusted U4 producer must perform
        # an actual serialized review/WAL commit, beginning revision=1.
        detached = SelfModelState(1, 0, (), (candidate,), None, ())
        recovered = SelfModelState.from_json(detached.canonical_bytes())
        if recovered != detached:
            raise SelfAttemptUnavailable("U1 SelfModel candidate roundtrip failed")
        return DetachedSelfAttempt(recovered, 5, chosen, latest, True, False)
    except (QualifiedProposalUnavailable, ValueError, TypeError) as error:
        raise SelfAttemptUnavailable("observed attempt cannot be qualified") from error
