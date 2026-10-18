"""R15 P1b TEST-ONLY bridge: owner-checked trial evidence to reconstructable U1 roots.

Not a production admission API, not a true NarrativeSelfSystem transition, not
a general G2 coherence proof. The app/test owner supplies its own live,
exclusive EventJournal + Memory + ExperienceStore in a disposable Source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.r15_p0.pilot import canonical_bytes, read_sealed
from experiments.r15_p1.protocol import (
    AUTHORITY,
    audit_trial,
    parse_candidate,
    s1_request,
)
from suzka.experience import ExperienceLifecycle, experience_record_digest
from suzka.memory import DualMemorySystem, ExperienceStore
from suzka.memory.dual_memory_system import (
    EpisodicMemoryReadError,
    EpisodicMemoryFormatError,
)
from suzka.memory.experience_store import ExperienceStoreError
from suzka.memory.experience_participant import experience_id_for_event
from suzka.memory.memory_schema import MemoryRecordType
from suzka.r15.common import (
    Accessibility,
    ClaimedOrigin,
    EventRef,
    Inertia,
    Interpretation,
    InterpretationStatus,
    RevisionProof,
    RevisionReason,
    SourceKind,
    SourceLifecycle,
    SourceWitness,
)
from suzka.r15.contracts import (
    NarrativeEpisode,
    NarrativeLink,
    NarrativeLinkKind,
    NarrativeSelfState,
)
from suzka.runtime.event_journal import EventJournal, EventLifecycle, EventJournalError

ZERO_DIGEST = "0" * 64
DOMAIN = b"SUZKA:R15:P1b:EXPERIMENTAL-OPTION:V1\0"
AUTHORITATIVE = False  # deliberately never an R15 owner or SourceSetVerified
MAX_TEST_EXPERIENCES = 16


class SourceBridgeRefused(ValueError):
    """Only non-sensitive failure classifications leave the test bridge."""


@dataclass(frozen=True)
class CheckedTurn:
    step: str
    event: EventRef
    episode_id: str
    experience_id: str
    experience_digest: str
    operation_digest: str
    context_id: str
    request: str
    response: str


@dataclass(frozen=True)
class TrialOwnerRead:
    """Production owners' real read methods, injected by disposable test app."""

    journal: EventJournal
    experience: ExperienceStore
    memory: DualMemorySystem

    def __post_init__(self) -> None:
        if not isinstance(self.journal, EventJournal):
            raise SourceBridgeRefused("missing actual R07 Journal owner")
        if not isinstance(self.experience, ExperienceStore):
            raise SourceBridgeRefused("missing actual R12 Experience owner")
        if not isinstance(self.memory, DualMemorySystem):
            raise SourceBridgeRefused("missing actual DB1 owner")


def option_key(setup_episode: str, option: str) -> str:
    """A CLOSED option mapping recoverable only with known checked setup."""
    if option not in ("A", "B", "UNKNOWN") or type(option) is not str:
        raise SourceBridgeRefused("unrecognized experimental option")
    return hashlib.sha256(
        DOMAIN + canonical_bytes({"setup_episode": setup_episode, "option": option})
    ).hexdigest()


def decode_key(setup_episode: str, key: str) -> str:
    candidates = [
        choice
        for choice in ("A", "B", "UNKNOWN")
        if option_key(setup_episode, choice) == key
    ]
    if len(candidates) != 1:
        raise SourceBridgeRefused("meaning cannot be recovered from scoped source")
    return candidates[0]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _assert_row(row: object, expected_step: str, index: int) -> dict[str, Any]:
    if (
        type(row) is not dict
        or set(row) != {"step", "ordinal", "request", "episode_id", "response"}
        or row["step"] != expected_step
        or type(row["ordinal"]) is not int
        or row["ordinal"] != index
        or any(type(row[k]) is not str for k in ("request", "episode_id", "response"))
        or not row["episode_id"]
    ):
        raise SourceBridgeRefused("malformed experiment stage")
    return row


def _owner_turn(owners: TrialOwnerRead, row: dict[str, Any]) -> CheckedTurn:
    """Source read, not subject-admission. No caller-asserted EventRef accepted."""
    episode_id = row["episode_id"]
    try:
        inspection = owners.journal.inspect()
    except EventJournalError as exc:
        raise SourceBridgeRefused("Journal unavailable") from exc
    # This bounded scan is a test-only adapter. Production already has an
    # event ID in AgentRuntime; it must not enumerate an unbounded store.
    try:
        paths = tuple(owners.experience.records_root.iterdir())
    except OSError as exc:
        raise SourceBridgeRefused("Experience storage unavailable") from exc
    if len(paths) > MAX_TEST_EXPERIENCES:
        raise SourceBridgeRefused("test Experience inventory exceeds cap")
    matches = []
    for path in paths:
        if not path.is_dir() or path.is_symlink():
            raise SourceBridgeRefused("invalid Experience directory")
        try:
            entry = owners.experience.load_committed_current(path.name)
        except ExperienceStoreError as exc:
            raise SourceBridgeRefused("Experience owner read unavailable") from exc
        if entry.record.source_episode_id == episode_id:
            matches.append(entry)
    if len(matches) != 1:
        raise SourceBridgeRefused("missing or ambiguous committed Experience")
    entry = matches[0]
    rec = entry.record
    if rec.lifecycle is not ExperienceLifecycle.ACTIVE:
        raise SourceBridgeRefused("Experience not active")
    if rec.experience_id != experience_id_for_event(
        rec.source_event_id, rec.source_event_sequence
    ):
        raise SourceBridgeRefused("Experience source event identity mismatches")
    records = tuple(r for r in inspection.records if r.event_id == rec.source_event_id)
    completed = [
        r
        for r in records
        if r.lifecycle is EventLifecycle.COMPLETED
        and r.processing_sequence == rec.source_event_sequence
    ]
    accepted = [r for r in records if r.lifecycle is EventLifecycle.ACCEPTED]
    txn = [
        r
        for r in records
        if r.lifecycle is EventLifecycle.TRANSACTION_COMPLETED
        and r.processing_sequence == rec.source_event_sequence
    ]
    if (
        len(completed) != 1
        or len(accepted) != 1
        or len(txn) != 1
        or any(r.lifecycle is EventLifecycle.FAILED for r in records)
    ):
        raise SourceBridgeRefused("source Journal is nonterminal or ambiguous")
    event = EventRef(
        rec.source_event_id, rec.source_event_sequence, accepted[0].timestamp
    )
    try:
        committed = owners.memory.get_committed_episodic(episode_id)
    except (EpisodicMemoryReadError, EpisodicMemoryFormatError) as exc:
        raise SourceBridgeRefused("DB1 owner read unavailable") from exc
    if committed is None:
        raise SourceBridgeRefused("DB1 episode unavailable")
    db1 = committed.record
    if (
        db1.record_type is not MemoryRecordType.EPISODIC_LOG
        or db1.archived
        or db1.user_input != row["request"]
        or db1.response != row["response"]
        or db1.context_id != rec.context_id
        or not db1.context_id
    ):
        raise SourceBridgeRefused("DB1/Experience source content mismatch")
    return CheckedTurn(
        step=row["step"],
        event=event,
        episode_id=episode_id,
        experience_id=rec.experience_id,
        experience_digest=experience_record_digest(rec),
        operation_digest=entry.operation_digest,
        context_id=rec.context_id,
        request=row["request"],
        response=row["response"],
    )


def _stage(manifest_path: Path, stage_path: Path) -> list[dict[str, Any]]:
    case, manifest = read_sealed(manifest_path)
    raw = stage_path.read_bytes()
    if len(raw) > 65536:
        raise SourceBridgeRefused("stage exceeds test limit")
    try:
        stage = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise SourceBridgeRefused("malformed stage") from exc
    if (
        type(stage) is not dict
        or set(stage) != {"schema", "authority", "case_id", "manifest_seal", "turns"}
        or stage["authority"] != AUTHORITY
        or stage["schema"] != 1
        or stage["case_id"] != case.case_id
        or stage["manifest_seal"] != manifest["seal"]
        or type(stage["turns"]) is not list
        or len(stage["turns"]) != 3
        or raw != canonical_bytes(stage) + b"\n"
    ):
        raise SourceBridgeRefused("stage does not match sealed trial")
    rows = [
        _assert_row(row, name, index)
        for index, (row, name) in enumerate(
            zip(stage["turns"], ("setup", "E", "S1"), strict=True)
        )
    ]
    prompts = case.prompts()
    if (
        rows[0]["request"] != prompts["setup"]
        or rows[1]["request"] != prompts["E"]
        or rows[2]["request"] != s1_request(rows[1]["request"], rows[1]["response"])
        or len({r["episode_id"] for r in rows}) != 3
    ):
        raise SourceBridgeRefused("pre-E2 trial source was altered")
    parse_candidate(rows[2]["response"], phase="S1", has_reveal=False)
    return rows


def _witness(turn: CheckedTurn) -> SourceWitness:
    # U1 SourceWitness is explicitly declarative, never a trusted source issuer.
    # operation_digest is an operation token, NOT a certified historical root.
    # Domain-hashed test snapshot is used only for a roundtrip spike.
    return SourceWitness(
        SourceKind.EXPERIENCE,
        turn.experience_id,
        0,
        turn.experience_digest,
        _digest({"test_only": True, "operation": turn.operation_digest}),
        SourceLifecycle.ACTIVE,
        ClaimedOrigin.SUBJECT_OBSERVATION,
        turn.event,
    )


def _score(uncertainty: str) -> int:
    return {"LOW": 8000, "MEDIUM": 5000, "HIGH": 2000}[uncertainty]


def _episode_id(scope: str, stage: str) -> str:
    return _digest({"scope": scope, "stage": stage, "domain": "P1b-test"})


def _episode_digest(episode: NarrativeEpisode) -> str:
    return _digest(
        {
            "episode_id": episode.episode_id,
            "statement_id": episode.interpretation.statement_id,
            "support": [w.source_digest for w in episode.interpretation.support],
        }
    )


def create_s1_root(
    manifest_path: Path,
    stage_path: Path,
    owners: TrialOwnerRead,
) -> NarrativeSelfState:
    """Test-only S1 projection using actual committed source reads, BEFORE E2."""
    rows = _stage(manifest_path, stage_path)
    checked = [_owner_turn(owners, row) for row in rows]
    if (
        sorted({r.event.event_sequence for r in checked})
        != [r.event.event_sequence for r in checked]
        or len({r.context_id for r in checked}) != 1
    ):
        raise SourceBridgeRefused("stage source ordering/context mismatch")
    c = parse_candidate(rows[2]["response"], phase="S1", has_reveal=False)
    scope = checked[0].episode_id
    ref = _witness(checked[1])
    known = c.assumed_referent != "UNKNOWN"
    interpretation = Interpretation(
        option_key(scope, c.assumed_referent),
        InterpretationStatus.PRESENT if known else InterpretationStatus.UNKNOWN,
        _score(c.uncertainty) if known else None,
        _score(c.uncertainty) if known else None,
        (ref,) if known else (),
        (),
        Accessibility.CURRENT,
        Inertia.TENTATIVE,
    )
    ep = NarrativeEpisode(_episode_id(scope, "S1"), ref, interpretation, ())
    eid = _episode_id(scope, "HYPOTHESIS")
    proof = RevisionProof(
        1,
        checked[2].event,
        eid,
        ZERO_DIGEST,
        _episode_digest(ep),
        (ref.source_digest,) if known else (),
        (),
        RevisionReason.INITIAL_INTERPRETATION,
        None,
    )
    root = NarrativeSelfState(1, 1, (ep,), (), None, (proof,))
    if root != NarrativeSelfState.from_json(root.canonical_bytes()):
        raise SourceBridgeRefused("S1 roundtrip failed")
    return root


def create_s2_root(
    manifest_path: Path,
    capture_path: Path,
    s1_root: NarrativeSelfState,
    owners: TrialOwnerRead,
) -> NarrativeSelfState:
    """A retrospective test reconstruction; NOT a production R15 mutation."""
    audit_trial(manifest_path, capture_path)
    stage_path = capture_path.with_name(capture_path.name + ".s1.json")
    expected_s1 = create_s1_root(manifest_path, stage_path, owners)
    if s1_root != expected_s1:
        raise SourceBridgeRefused("S1 supplied root is stale or altered")
    case, _ = read_sealed(manifest_path)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    rows = capture["turns"]
    if case.control == "no_reveal":
        return s1_root  # No new source: do not promote confidence/wording drift.
    if len(rows) != 5:
        raise SourceBridgeRefused("E2 or S2 missing")
    checked = [_owner_turn(owners, row) for row in rows]
    sequences = [r.event.event_sequence for r in checked]
    if sequences != sorted(set(sequences)):
        raise SourceBridgeRefused("E2/S2 chronological source mismatch")
    if len({r.context_id for r in checked}) != 1:
        raise SourceBridgeRefused("cross-context interpretation unqualified")
    old = parse_candidate(rows[2]["response"], phase="S1", has_reveal=False)
    new = parse_candidate(rows[4]["response"], phase="S2", has_reveal=True)
    if new.assumed_referent in ("UNKNOWN", old.assumed_referent):
        return s1_root
    # Benchmark-only objective tie to a real E2 USER STATEMENT. It does not
    # establish objective human intent or independently subject-owned belief.
    if f"案{new.assumed_referent}" not in checked[3].request:
        raise SourceBridgeRefused("S2 has no observed corresponding E2 statement")
    scope = checked[0].episode_id
    source = _witness(checked[3])
    prior = _witness(checked[1])
    interpretation = Interpretation(
        option_key(scope, new.assumed_referent),
        InterpretationStatus.CONTESTED,
        _score(new.uncertainty),
        _score(new.uncertainty),
        (source,),
        (prior,),
        Accessibility.CURRENT,
        Inertia.REVISABLE,
    )
    old_episode = s1_root.episodes[0]
    ep = NarrativeEpisode(
        _episode_id(scope, "S2"),
        source,
        interpretation,
        (NarrativeLink(NarrativeLinkKind.EPISODE, old_episode.episode_id),),
    )
    first = s1_root.revision_history[-1]
    proof = RevisionProof(
        2,
        checked[4].event,
        first.target_id,
        first.after_digest,
        _episode_digest(ep),
        (source.source_digest,),
        (prior.source_digest,),
        RevisionReason.QUALIFIED_REINTERPRETATION,
        first.record_digest,
    )
    root = NarrativeSelfState(
        1,
        2,
        tuple(sorted((old_episode, ep), key=lambda item: item.episode_id)),
        (),
        None,
        (first, proof),
    )
    if root != NarrativeSelfState.from_json(root.canonical_bytes()):
        raise SourceBridgeRefused("S2 roundtrip failed")
    return root


def recover_option(
    state: NarrativeSelfState,
    *,
    setup_episode_id: str,
    stage: str,
) -> str:
    """Reconstruct A/B/UNKNOWN from retained scope, not from a bare hash."""
    desired = _episode_id(setup_episode_id, stage)
    matches = [ep for ep in state.episodes if ep.episode_id == desired]
    if len(matches) != 1:
        raise SourceBridgeRefused("recorded staged episode absent")
    return decode_key(setup_episode_id, matches[0].interpretation.statement_id)


def projected_current_referent(
    state: NarrativeSelfState,
    *,
    setup_episode_id: str,
) -> dict[str, object]:
    """Read-only post-restart decision CONTEXT PROJECTION, never authority.

    The caller must independently qualify actual source lifecycle before
    treating this candidate as a production R15 decision input.
    """
    stage = (
        "S2"
        if any(
            ep.episode_id == _episode_id(setup_episode_id, "S2")
            for ep in state.episodes
        )
        else "S1"
    )
    choice = recover_option(state, setup_episode_id=setup_episode_id, stage=stage)
    return {
        "authority": "P1b_TEST_ONLY_UNVERIFIED_SEMANTIC_PROJECTION",
        "selected_stage": stage,
        "current_referent": choice,
        "source_scope": setup_episode_id,
        "revision": state.revision,
        "production_decision_authority": False,
    }


def save_test_root(path: Path, state: NarrativeSelfState) -> None:
    """Create-only local test snapshot. Never call on production AgentState."""
    data = state.canonical_bytes()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    dfd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def load_test_root(path: Path) -> NarrativeSelfState:
    return NarrativeSelfState.from_json(path.read_bytes())
