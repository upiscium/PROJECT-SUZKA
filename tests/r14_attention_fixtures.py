"""Coherent source fixtures shared by legacy production integration tests."""

from datetime import UTC, datetime, timedelta

from suzka.motivation import CommitmentSystem, GoalSystem, MotivationSystem
from suzka.motivation.common import R13Reference, R13ReferenceKind
from test_commitment_system import (
    admission_for as commitment_admission_for,
    ingest as ingest_commitment,
    make_proposal as make_commitment_proposal,
)
from test_goal_system import (
    admission_for as goal_admission_for,
    ingest as ingest_goal,
    make_proposal as make_goal_proposal,
)
from test_motivation_system import make_evidence, source_event


_SOURCE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def coherent_r13_systems(
    count: int = 2,
) -> tuple[MotivationSystem, GoalSystem, CommitmentSystem]:
    """Build real active R13 records on one globally coherent event stream."""

    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("count must be an exact integer in [1, 32]")

    motivation = MotivationSystem()
    for index in range(count):
        sequence = index + 1
        source = R13Reference(
            R13ReferenceKind.EXPERIENCE,
            f"r14-attention:motivation-source:{index:03}",
        )
        evidence = make_evidence(
            source,
            target=R13Reference(
                R13ReferenceKind.VALUE,
                f"value:r14-attention:{index:03}",
            ),
        )
        event = source_event(
            evidence,
            sequence,
            event_id=f"event:r14-attention:motivation:create:{index:03}",
            recorded_at=_SOURCE_TIME + timedelta(seconds=sequence),
        )
        motivation.apply_evidence(evidence, event)

    goals = GoalSystem()
    for index in range(count):
        genesis_sequence = count + 2 * index + 1
        genesis_event_id = f"event:r14-attention:goal:create:{index:03}"
        proposal = make_goal_proposal(
            key=f"r14-attention-goal-{index:03}",
            genesis_event_id=genesis_event_id,
            genesis_sequence=genesis_sequence,
            created_at=_SOURCE_TIME + timedelta(seconds=genesis_sequence),
        )
        ingest_goal(goals, proposal, genesis_sequence)
        admission, event = goal_admission_for(
            proposal,
            genesis_sequence + 1,
            event_id=f"event:r14-attention:goal:adopt:{index:03}",
        )
        goals.adopt(proposal.goal_id, admission, event)

    commitments = CommitmentSystem()
    for index in range(count):
        genesis_sequence = 3 * count + 1 + 2 * index
        genesis_event_id = f"event:r14-attention:commitment:propose:{index:03}"
        proposal = make_commitment_proposal(
            f"r14-attention-commitment-{index:03}",
            genesis_id=genesis_event_id,
            genesis_sequence=genesis_sequence,
            genesis_at=_SOURCE_TIME + timedelta(seconds=genesis_sequence),
        )
        ingest_commitment(commitments, proposal, genesis_sequence)
        admission, event = commitment_admission_for(
            proposal,
            genesis_sequence + 1,
            event_id=f"event:r14-attention:commitment:accept:{index:03}",
        )
        commitments.accept(proposal.commitment_id, admission, event)

    return motivation, goals, commitments
