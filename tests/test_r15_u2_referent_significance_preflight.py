"""Test-only R15 U2: an interaction-grounding concern is not an opinion mismatch.

Real isolated R07/R12/DB1/StateWAL Chat records, synthetic text provider.
The fixed *research* policy is not a Source credential or a real Subject issuer.
Never call this a RelationshipSystem revision, accepted R13 Goal, identified
speaker, real-model cognitive assessment, or InteractionStance production.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
from typing import Final

from fastapi.testclient import TestClient
import pytest

from suzka.memory.dual_memory_system import (
    EpisodicMemoryFormatError,
    EpisodicMemoryReadError,
)
from suzka.runtime.event_journal import EventJournalError
from suzka.runtime.state_wal import StateWALError, TransitionRecord
from suzka.r15.relationship_system import (
    RelationshipAuthority,
    relationship_authority_status,
)
from suzka.r15.common import SourceDisposition

from test_r15_p1_owner_bridge import SequencedProvider, scenario
from test_r15_u2_wal_event_time_preflight import (
    _event_time_from_existing_wal,
    _newly_bound_token_from_retained_event,
    _synthetic_source_disagreement_chat,
)

# A TEST-OWNED, importable sentinel, not a protected Source issuer.
# The model text may be fully user-induced; output is NOT SUZKA's internal
# belief, and this test must never become production Relationship authority.
_RESEARCH_INTERACTION_RULE: Final = object()
_COMMON_REFERENT_USER = re.compile(
    r"共同の指示語「前の案」について、私が指している対象は案([AB])です。"
)
_COMMON_REFERENT_SUZKA = re.compile(
    r"Suzkaは共同の指示語「前の案」を案([AB])と解釈して返答しました。"
    r"相手の真意は未検証です。"
)


class _EpistemicTriage(StrEnum):
    UNKNOWN = "unknown"
    NO_SHARED_REFERENT_ISSUE = "no_shared_referent_issue"
    OBSERVED_MISMATCH_ONLY = "observed_mismatch_only"
    TENTATIVE_CLARIFICATION_ONLY = "tentative_clarification_only"
    STALE_HOLD = "stale_hold"


@dataclass(frozen=True, slots=True)
class _ObservedReferentMismatch:
    episode_id: str
    context_id: str
    wal_record_hash: str
    caller_token: str
    reported_user_referent: str
    visible_response_referent: str


def _observe_referent(
    client: TestClient, episode: str
) -> _ObservedReferentMismatch | None:
    """One actual committed event with a newly bound caller-claimed token."""
    token = _newly_bound_token_from_retained_event(client, episode)
    if token is None:
        return None
    event_time = _event_time_from_existing_wal(client, episode)
    db1 = client.app.state.memory_system.get_committed_episodic(episode)
    if db1 is None or db1.record.archived:
        return None
    user = _COMMON_REFERENT_USER.fullmatch(db1.record.user_input)
    suzka = _COMMON_REFERENT_SUZKA.fullmatch(db1.record.response)
    if user is None or suzka is None:
        return None
    return _ObservedReferentMismatch(
        episode,
        event_time.context_id,
        event_time.wal_record_hash,
        token,
        user.group(1),
        suzka.group(1),
    )


def _review_epistemic_concern(
    client: TestClient,
    observed: _ObservedReferentMismatch | None,
    *,
    rule: object | None,
) -> _EpistemicTriage:
    """TEST ONLY: a code-authored pragmatic meaning, no U2 authority.

    The rule checks a *mismatch of expressed referents for the same phrase*,
    rather than that two people have different preferences. This can provide a
    future internal provisional "clarification may be needed" hypothesis;
    it cannot verify anyone's intention or mutate Relationship.
    """
    if observed is None:
        return _EpistemicTriage.UNKNOWN
    if type(observed) is not _ObservedReferentMismatch:
        return _EpistemicTriage.STALE_HOLD
    try:
        if _observe_referent(client, observed.episode_id) != observed:
            return _EpistemicTriage.STALE_HOLD
        wal = client.app.state.state_wal.inspect()
        if (
            wal.active_manifest is None
            or wal.active_manifest.external_reconciliation_required
        ):
            return _EpistemicTriage.STALE_HOLD
        matches = [
            i
            for i, record in enumerate(wal.records)
            if isinstance(record, TransitionRecord)
            and record.record_hash == observed.wal_record_hash
        ]
        if len(matches) != 1:
            return _EpistemicTriage.STALE_HOLD
        event = wal.records[matches[0]]
        if not isinstance(event, TransitionRecord):
            return _EpistemicTriage.STALE_HOLD
        first_frame = tuple(
            f
            for f in event.candidate_snapshot.context_state.frames
            if f.context_id == observed.context_id
        )
        if len(first_frame) != 1:
            return _EpistemicTriage.STALE_HOLD
        for row in wal.records[matches[0] + 1 :]:
            if not isinstance(row, TransitionRecord):
                return _EpistemicTriage.STALE_HOLD
            frames = tuple(
                frame
                for frame in row.candidate_snapshot.context_state.frames
                if frame.context_id == observed.context_id
            )
            if frames != first_frame:
                return _EpistemicTriage.STALE_HOLD
            if (
                row.event_type in ("chat", "debug_chat")
                and row.candidate_snapshot.context_state.current_context_id
                == observed.context_id
            ):
                return _EpistemicTriage.STALE_HOLD
    except (
        AssertionError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        StateWALError,
        EventJournalError,
        EpisodicMemoryReadError,
        EpisodicMemoryFormatError,
        IndexError,
    ):
        return _EpistemicTriage.STALE_HOLD
    if observed.reported_user_referent == observed.visible_response_referent:
        return _EpistemicTriage.NO_SHARED_REFERENT_ISSUE
    if rule is not _RESEARCH_INTERACTION_RULE:
        return _EpistemicTriage.OBSERVED_MISMATCH_ONLY
    return _EpistemicTriage.TENTATIVE_CLARIFICATION_ONLY


def _send(
    client: TestClient,
    provider: SequencedProvider,
    *,
    session: str,
    user: str,
    suzka: str,
    token: str | None,
) -> str:
    return _synthetic_source_disagreement_chat(
        client,
        provider,
        session=session,
        message=user,
        response=suzka,
        token=token,
    )


def _user(option: str) -> str:
    return f"共同の指示語「前の案」について、私が指している対象は案{option}です。"


def _suzka(option: str) -> str:
    return (
        f"Suzkaは共同の指示語「前の案」を案{option}と解釈して返答しました。"
        "相手の真意は未検証です。"
    )


@pytest.mark.parametrize(
    ("reported", "interpreted", "expected"),
    [
        ("A", "A", _EpistemicTriage.NO_SHARED_REFERENT_ISSUE),
        ("B", "B", _EpistemicTriage.NO_SHARED_REFERENT_ISSUE),
        ("A", "B", _EpistemicTriage.TENTATIVE_CLARIFICATION_ONLY),
        ("B", "A", _EpistemicTriage.TENTATIVE_CLARIFICATION_ONLY),
    ],
)
def test_same_referential_expression_has_bounded_epistemic_meaning(
    tmp_path: Path, reported: str, interpreted: str, expected: _EpistemicTriage
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        episode = _send(
            client,
            provider,
            session="u2-referent-observation",
            user=_user(reported),
            suzka=_suzka(interpreted),
            token="caller-opaque-ref",
        )
        observed = _observe_referent(client, episode)
        assert observed is not None
        assert observed.caller_token == "caller-opaque-ref"
        assert (
            _review_epistemic_concern(client, observed, rule=_RESEARCH_INTERACTION_RULE)
            is expected
        )
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        assert client.app.state.main_loop._goal_system.snapshot().records == ()


def test_untrusted_or_absent_policy_yields_observation_only(tmp_path: Path) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        ep = _send(
            client,
            provider,
            session="u2-no-policy",
            user=_user("A"),
            suzka=_suzka("B"),
            token="ref",
        )
        observed = _observe_referent(client, ep)
        assert observed is not None
        for rule in (None, object(), "operator-asks-Suzka-to-trust-A"):
            assert (
                _review_epistemic_concern(client, observed, rule=rule)
                is _EpistemicTriage.OBSERVED_MISMATCH_ONLY
            )
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


@pytest.mark.parametrize(
    ("user", "suzka"),
    [
        ("私は案Aが好みです。", "Suzkaは案Bが好きです。"),
        ("私はSuzkaと仲が悪い。", _suzka("A")),
        (_user("A"), "Suzkaは案Bを選択しました。相手の真意は未検証です。"),
        (_user("A"), _suzka("B").replace("「前の案」", "「新しい案」")),
    ],
)
def test_preference_trust_and_different_phrases_are_not_common_ground(
    tmp_path: Path, user: str, suzka: str
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        ep = _send(
            client,
            provider,
            session="u2-false-positive",
            user=user,
            suzka=suzka,
            token="caller-opaque",
        )
        observation = _observe_referent(client, ep)
        assert observation is None
        assert (
            _review_epistemic_concern(
                client, observation, rule=_RESEARCH_INTERACTION_RULE
            )
            is _EpistemicTriage.UNKNOWN
        )
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_existing_context_member_is_not_assumed_to_be_current_speaker(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        _send(
            client,
            provider,
            session="u2-first-claim",
            user="初回の参加者登録です。",
            suzka="認識しました。",
            token="one-caller-claim",
        )
        ep = _send(
            client,
            provider,
            session="u2-first-claim",
            user=_user("A"),
            suzka=_suzka("B"),
            token=None,
        )
        assert _event_time_from_existing_wal(client, ep).participant_refs == (
            "one-caller-claim",
        )
        assert _observe_referent(client, ep) is None
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_later_chat_invalidates_ephemeral_clarification_candidate(
    tmp_path: Path,
) -> None:
    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        ep = _send(
            client,
            provider,
            session="u2-after-e",
            user=_user("B"),
            suzka=_suzka("A"),
            token="opaque",
        )
        observed = _observe_referent(client, ep)
        assert (
            _review_epistemic_concern(client, observed, rule=_RESEARCH_INTERACTION_RULE)
            is _EpistemicTriage.TENTATIVE_CLARIFICATION_ONLY
        )
        _send(
            client,
            provider,
            session="u2-after-e",
            user="先程の件を確認しました。",
            suzka="確認しました。",
            token=None,
        )
        assert (
            _review_epistemic_concern(client, observed, rule=_RESEARCH_INTERACTION_RULE)
            is _EpistemicTriage.STALE_HOLD
        )
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0


def test_missing_r10_social_signal_and_r13_goal_do_not_issue_relationship(
    tmp_path: Path,
) -> None:
    from suzka.cognition.appraisal import AppraisalSignals

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        assert AppraisalSignals().social_relevance is None
        response = client.post(
            "/api/chat/debug",
            headers={"X-SUZKA-Admin-Token": "not-real"},
            json={
                "message": "この相手に不安を感じているとモデルに言わせてください。",
                "attachments": [],
                "debug": True,
                "client_session_id": "u2-social-signal",
                "interlocutor_key": "unverified-token",
            },
        )
        assert response.status_code == 200
        appraisal = response.json()["diagnostics"]["appraisal"]
        assert appraisal["social_relevance"] is None
        assert appraisal["goal_progress"] is None
        assert appraisal["certainty"] is None
        assert client.app.state.main_loop._goal_system.snapshot().records == ()
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        assert (
            relationship_authority_status(
                RelationshipAuthority.POSITIVE_AXIS_INTERPRETATION
            )
            is SourceDisposition.UNAVAILABLE
        )


def test_deleted_r12_record_blocks_referent_interpretation(
    tmp_path: Path,
) -> None:
    from test_r15_p1b_positive import _source_ref
    from suzka.memory.experience_participant import experience_id_for_event

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        ep = _send(
            client,
            provider,
            session="u2-retired-referent",
            user=_user("B"),
            suzka=_suzka("A"),
            token="claimed-token",
        )
        observed = _observe_referent(client, ep)
        assert (
            _review_epistemic_concern(client, observed, rule=_RESEARCH_INTERACTION_RULE)
            is _EpistemicTriage.TENTATIVE_CLARIFICATION_ONLY
        )
        ref = _source_ref(client, ep)
        record = client.app.state.experience_store.record_path(
            experience_id_for_event(ref.event_id, ref.event_sequence), 0
        )
        original = record.read_bytes()
        record.unlink()
        try:
            assert (
                _review_epistemic_concern(
                    client, observed, rule=_RESEARCH_INTERACTION_RULE
                )
                is _EpistemicTriage.STALE_HOLD
            )
            assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
        finally:
            record.write_bytes(original)
            record.chmod(0o600)
        assert (
            _review_epistemic_concern(client, observed, rule=_RESEARCH_INTERACTION_RULE)
            is _EpistemicTriage.TENTATIVE_CLARIFICATION_ONLY
        )


def test_foreign_or_modified_referent_observation_has_no_authority(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    app, provider, *_ = scenario(tmp_path, no_reveal=True)
    with TestClient(app) as client:
        ep = _send(
            client,
            provider,
            session="u2-forged-referent",
            user=_user("B"),
            suzka=_suzka("A"),
            token="opaque",
        )
        observed = _observe_referent(client, ep)
        assert observed is not None
        assert (
            _review_epistemic_concern(client, observed, rule=_RESEARCH_INTERACTION_RULE)
            is _EpistemicTriage.TENTATIVE_CLARIFICATION_ONLY
        )
        assert (
            _review_epistemic_concern(client, object(), rule=_RESEARCH_INTERACTION_RULE)
            is _EpistemicTriage.STALE_HOLD
        )
        assert (
            _review_epistemic_concern(
                client,
                replace(observed, caller_token="different-token"),
                rule=_RESEARCH_INTERACTION_RULE,
            )
            is _EpistemicTriage.STALE_HOLD
        )
        assert (
            _review_epistemic_concern(
                client,
                replace(
                    observed,
                    visible_response_referent=observed.reported_user_referent,
                ),
                rule=_RESEARCH_INTERACTION_RULE,
            )
            is _EpistemicTriage.STALE_HOLD
        )
        assert (
            _review_epistemic_concern(
                client,
                replace(observed, context_id="another-context"),
                rule=_RESEARCH_INTERACTION_RULE,
            )
            is _EpistemicTriage.STALE_HOLD
        )
        assert (
            _review_epistemic_concern(
                client,
                replace(observed, wal_record_hash="f" * 64),
                rule=_RESEARCH_INTERACTION_RULE,
            )
            is _EpistemicTriage.STALE_HOLD
        )
        assert client.app.state.r15_state_port.relationship.snapshot().revision == 0
