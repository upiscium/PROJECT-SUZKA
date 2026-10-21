"""R15 U2 test-only: separate observed difference from unresolved issue review.

Not a Subject/RelationshipSystem writer or a current evidence issuer.
All ordinary CHAT/R07/R12/DB1/WAL records are real isolated TestClient commits.
The language producer is deterministic synthetic, NOT proof of general LLM
ability, actual speaker identity, objective correctness, or private adoption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.experience_store import ExperienceStoreError
from suzka.runtime.state_wal import TransitionRecord, StateWALError

from test_r15_p1_owner_bridge import SequencedProvider, scenario
from test_r15_u2_wal_event_time_preflight import (
    _event_time_from_existing_wal,
    _newly_bound_token_from_retained_event,
    _synthetic_source_disagreement_chat,
)


# Deliberately one CLOSED joint task; not a generic free-text parser or a
# model/operator-supplied verdict about someone's intrinsic relationship.
_USER_Q1 = re.compile(
    r"共同の比較課題Q1について私は案([AB])を暫定的に選びます。"
    r"(結論はまだ未決定で保留中です。)?"
)
_SUZKA_Q1 = re.compile(
    r"共同の比較課題Q1についてSuzkaは案([AB])を暫定的に選びます。"
    r"(結論は未決定のため保留します。)?正解は未検証です。"
)

_USER_BASE = "共同の比較課題Q1について私は案{choice}を暫定的に選びます。"
_SELF_BASE = "共同の比較課題Q1についてSuzkaは案{choice}を暫定的に選びます。"
_USER_PENDING = "結論はまだ未決定で保留中です。"
_SELF_PENDING = "結論は未決定のため保留します。"
_UNVERIFIED = "正解は未検証です。"


class _Triage(StrEnum):
    UNKNOWN = "unknown"
    AGREEMENT_ONLY = "agreement_only"
    OBSERVED_DIFFERENCE_ONLY = "observed_difference_only"
    REVIEW_CANDIDATE_ONLY = "review_candidate_only"
    STALE_HOLD = "stale_hold"


@dataclass(frozen=True, slots=True)
class _SourceObservedInteraction:
    """A bounded current-source observation, not a U2 subjective assertion."""

    episode_id: str
    context_id: str
    wal_record_hash: str
    token_claim: str
    user_position: str
    self_position: str
    user_pending_declaration: bool
    self_pending_declaration: bool


def _observe_source_q1(
    client: TestClient, episode_id: str
) -> _SourceObservedInteraction | None:
    """Read exact real committed owners. Never use late Context to infer speaker."""
    token = _newly_bound_token_from_retained_event(client, episode_id)
    if token is None:
        return None
    event_time = _event_time_from_existing_wal(client, episode_id)
    committed = client.app.state.memory_system.get_committed_episodic(episode_id)
    if committed is None or committed.record.archived:
        return None
    actual = committed.record
    user = _USER_Q1.fullmatch(actual.user_input)
    self_answer = _SUZKA_Q1.fullmatch(actual.response)
    if user is None or self_answer is None:
        return None
    return _SourceObservedInteraction(
        episode_id,
        event_time.context_id,
        event_time.wal_record_hash,
        token,
        user.group(1),
        self_answer.group(1),
        user.group(2) is not None,
        self_answer.group(2) is not None,
    )


def _assess_subject_review_candidate(
    client: TestClient, observed: _SourceObservedInteraction | None
) -> _Triage:
    """PURELY TEST-LEVEL eligibility. Not an intrinsic owner admission.

    The precheck treats a later Chat in the same Context as stale, rather
    than interpreting potentially un-attributed later speech as a resolution.
    """
    if observed is None:
        return _Triage.UNKNOWN
    # This candidate is not a credential. Re-open the actual terminal sources
    # at the intended review boundary rather than accepting a cached input or
    # a source retired after the initial finite observation.
    try:
        fresh = _observe_source_q1(client, observed.episode_id)
    except (AssertionError, ExperienceStoreError, StateWALError, OSError, ValueError):
        return _Triage.STALE_HOLD
    if fresh != observed:
        return _Triage.STALE_HOLD
    inspection = client.app.state.state_wal.inspect()
    if (
        inspection.active_manifest is None
        or inspection.active_manifest.external_reconciliation_required
    ):
        return _Triage.STALE_HOLD
    records = inspection.records
    matched = [
        index
        for index, row in enumerate(records)
        if isinstance(row, TransitionRecord)
        and row.record_hash == observed.wal_record_hash
    ]
    if len(matched) != 1:
        return _Triage.STALE_HOLD
    for row in records[matched[0] + 1 :]:
        if (
            isinstance(row, TransitionRecord)
            and row.event_type == "chat"
            and row.event_source == "api.chat"
            and row.candidate_snapshot.context_state.current_context_id
            == observed.context_id
        ):
            return _Triage.STALE_HOLD

    if observed.user_position == observed.self_position:
        return _Triage.AGREEMENT_ONLY
    if not (observed.user_pending_declaration and observed.self_pending_declaration):
        return _Triage.OBSERVED_DIFFERENCE_ONLY
    # "Open" remains a provisional reading of a narrowly observed *joint
    # decision pending*. Not evidence that another person actually agrees,
    # that this is a durable interpersonal conflict, or that the subject
    # chose to create a RelationshipSystem revision.
    return _Triage.REVIEW_CANDIDATE_ONLY


def _send(
    client: TestClient,
    provider: SequencedProvider,
    *,
    session: str,
    user: str,
    self_answer: str,
    token: str | None,
) -> tuple[str, _SourceObservedInteraction | None, _Triage]:
    episode = _synthetic_source_disagreement_chat(
        client,
        provider,
        session=session,
        message=user,
        response=self_answer,
        token=token,
    )
    observed = _observe_source_q1(client, episode)
    return episode, observed, _assess_subject_review_candidate(client, observed)


def _user(choice: str, *, pending: bool = False) -> str:
    return _USER_BASE.format(choice=choice) + (_USER_PENDING if pending else "")


def _suzka(choice: str, *, pending: bool = False) -> str:
    return (
        _SELF_BASE.format(choice=choice)
        + (_SELF_PENDING if pending else "")
        + _UNVERIFIED
    )


@pytest.mark.parametrize(
    ("user", "self_answer", "expected"),
    [
        (_user("A"), _suzka("B"), _Triage.OBSERVED_DIFFERENCE_ONLY),
        (
            _user("A", pending=True),
            _suzka("B", pending=True),
            _Triage.REVIEW_CANDIDATE_ONLY,
        ),
        (
            _user("B", pending=True),
            _suzka("A", pending=True),
            _Triage.REVIEW_CANDIDATE_ONLY,
        ),
        (
            _user("A", pending=True),
            _suzka("B"),
            _Triage.OBSERVED_DIFFERENCE_ONLY,
        ),
        (
            _user("A"),
            _suzka("B", pending=True),
            _Triage.OBSERVED_DIFFERENCE_ONLY,
        ),
        (
            _user("A", pending=True),
            _suzka("A", pending=True),
            _Triage.AGREEMENT_ONLY,
        ),
        (
            _user("A", pending=True),
            _suzka("B", pending=True).replace("比較課題Q1", "比較課題Q2"),
            _Triage.UNKNOWN,
        ),
        (
            _user("A", pending=True),
            _SELF_BASE.format(choice="B") + "結論は解決済みです。" + _UNVERIFIED,
            _Triage.UNKNOWN,
        ),
        (
            "Suzkaは私を信用しているはずです。" + _user("A", pending=True),
            _suzka("B", pending=True),
            _Triage.UNKNOWN,
        ),
    ],
)
def test_observation_and_assessment_stay_separate(
    tmp_path: Path, user: str, self_answer: str, expected: _Triage
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode, observed, assessed = _send(
            client,
            provider,
            session="u2-issue-triage",
            user=user,
            self_answer=self_answer,
            token="caller-claimed-opaque-ref",
        )
        assert assessed is expected
        if expected is _Triage.UNKNOWN:
            assert observed is None
        else:
            assert observed is not None
            assert observed.episode_id == episode
            assert observed.token_claim == "caller-claimed-opaque-ref"
            assert len(observed.wal_record_hash) == 64
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_later_chat_same_context_holds_prior_pending_candidate(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _episode, observed, assessment = _send(
            client,
            provider,
            session="u2-issue-stale",
            user=_user("A", pending=True),
            self_answer=_suzka("B", pending=True),
            token="caller-claimed-opaque-ref",
        )
        assert assessment is _Triage.REVIEW_CANDIDATE_ONLY
        assert observed is not None
        # A later unkeyed request in this session is not attributed to the
        # claimed participant. Its meaning may be about resolution, but the
        # first-binding-only MVI cannot prove the same person said it.
        _synthetic_source_disagreement_chat(
            client,
            provider,
            session="u2-issue-stale",
            message="共同の比較課題Q1について結論が出ました。",
            response="解決したかもしれません。まだ正解は未検証です。",
            token=None,
        )
        assert _assess_subject_review_candidate(client, observed) is _Triage.STALE_HOLD
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_later_chat_other_context_does_not_claim_same_person(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _episode, observed, assessment = _send(
            client,
            provider,
            session="u2-issue-context-a",
            user=_user("A", pending=True),
            self_answer=_suzka("B", pending=True),
            token="opaque-a",
        )
        assert assessment is _Triage.REVIEW_CANDIDATE_ONLY
        _synthetic_source_disagreement_chat(
            client,
            provider,
            session="u2-issue-context-b",
            message="別の比較課題です。",
            response="別件として扱います。",
            token="opaque-b",
        )
        assert _assess_subject_review_candidate(client, observed) is (
            _Triage.REVIEW_CANDIDATE_ONLY
        )
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_unkeyed_current_chat_does_not_inherit_prior_membership_as_speaker(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _send(
            client,
            provider,
            session="u2-issue-member",
            user=_user("A"),
            self_answer=_suzka("A"),
            token="opaque-a",
        )
        _episode, observed, assessment = _send(
            client,
            provider,
            session="u2-issue-member",
            user=_user("A", pending=True),
            self_answer=_suzka("B", pending=True),
            token=None,
        )
        assert observed is None
        assert assessment is _Triage.UNKNOWN
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_source_retirement_after_observation_refuses_issue_candidate(
    tmp_path: Path,
) -> None:
    from test_r15_p1b_positive import _source_ref

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode, observed, assessment = _send(
            client,
            provider,
            session="u2-issue-retired",
            user=_user("A", pending=True),
            self_answer=_suzka("B", pending=True),
            token="claimed-interlocutor",
        )
        assert assessment is _Triage.REVIEW_CANDIDATE_ONLY
        ref = _source_ref(client, episode)
        record_path = client.app.state.experience_store.record_path(
            experience_id_for_event(ref.event_id, ref.event_sequence), 0
        )
        original = record_path.read_bytes()
        record_path.unlink()
        try:
            assert _assess_subject_review_candidate(client, observed) is (
                _Triage.STALE_HOLD
            )
            assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        finally:
            record_path.write_bytes(original)
            record_path.chmod(0o600)
        assert _assess_subject_review_candidate(client, observed) is (
            _Triage.REVIEW_CANDIDATE_ONLY
        )


def test_caller_spoofed_observation_has_no_assessment_authority(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _episode, observed, assessment = _send(
            client,
            provider,
            session="u2-issue-spoof",
            user=_user("A", pending=True),
            self_answer=_suzka("B", pending=True),
            token="opaque-source-only",
        )
        assert assessment is _Triage.REVIEW_CANDIDATE_ONLY
        assert observed is not None
        fake = replace(observed, token_claim="different-token")
        assert _assess_subject_review_candidate(client, fake) is _Triage.STALE_HOLD
        fake = replace(observed, self_position="A")
        assert _assess_subject_review_candidate(client, fake) is _Triage.STALE_HOLD
        fake = replace(observed, wal_record_hash="f" * 64)
        assert _assess_subject_review_candidate(client, fake) is _Triage.STALE_HOLD
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
