"""Test rig for the F15 DC-4 participant quarantine.

Builds the real durable graph (Journal, StateWAL, AgentState, Chroma-backed
``DualMemorySystem``, Experience and Semantic stores) and open transactions that
need ``memory.experience`` and / or ``memory.semantic``, so a test can damage a
store and run the real ``StartupReconciliationCoordinator`` on it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from suzka.config import Settings, load_settings
from suzka.experience import (
    ExperienceAppraisalEvidence,
    ExperienceAppraisalReasonCode,
    ExperienceEmotionContributions,
    ExperienceEmotionProjection,
    ExperienceEmotionUpdateReasonCode,
    ExperienceLifecycle,
    ExperienceMeasurementEvidence,
    ExperienceRecord,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    ExperienceRevisionRecord,
)
from suzka.memory import DualMemorySystem, MemoryRecordType
from suzka.memory.episodic_participant import EpisodicWrite, MemoryEpisodicParticipant
from suzka.memory.experience_participant import (
    ExperienceCreateIntent,
    MemoryExperienceParticipant,
    experience_id_for_event,
)
from suzka.memory.experience_store import ExperienceStore
from suzka.memory.semantic_lifecycle import SemanticRevision, semantic_content_digest
from suzka.memory.semantic_participant import (
    MemorySemanticParticipant,
    SemanticBatchEntry,
    SemanticBatchOperation,
    SemanticCreateIntent,
    semantic_id_for_batch_entry,
)
from suzka.memory.semantic_store import SemanticStore
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from suzka.runtime.agent_state import (
    AgentStateSnapshotV2,
    AgentStateStore,
    EmotionStateSnapshot,
    WorkingMemorySnapshot,
)
from suzka.runtime.degraded_continuation import DegradedContinuationStatus
from suzka.runtime.domain_quarantine import DomainQuarantinePolicy
from suzka.runtime.event_journal import EventJournal, TransactionKind
from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator
from suzka.runtime.state_recovery import StateRecoveryCoordinator
from suzka.runtime.state_wal import StateWAL
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionBinding,
    TransactionCoordinator,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def settings_for(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "dc4_db1",
                    "db2_collection": "dc4_db2",
                }
            )
        }
    )


def snapshot(sequence: int) -> AgentStateSnapshotV2:
    return AgentStateSnapshotV2(
        saved_at=NOW,
        last_processed_event_sequence=sequence,
        emotion_state=EmotionStateSnapshot(valence=0.4, arousal=0.2, optimal_loss=1.0),
        working_memory=WorkingMemorySnapshot(revision=0, items=()),
    )


@dataclass
class Prepared:
    """One prepared transaction: the event, its participants and their bindings."""

    event: AgentEvent
    transaction_id: str
    episode: MemoryEpisodicParticipant | None
    experience: MemoryExperienceParticipant | None
    semantic: MemorySemanticParticipant | None
    experience_id: str | None
    evidence: object | None = None


class Rig:
    """The durable graph under ``tmp_path``.  Stores live beside the Chroma
    directory exactly as in production (``experience/``, ``semantic/``)."""

    def __init__(
        self, tmp_path: Path, *, settings: Settings | None = None, attach: bool = False
    ) -> None:
        """``attach=True`` opens the files an application already wrote (the
        application must be stopped: its Journal lease is released) and
        initialises nothing; the paths come from ``settings``."""

        self.tmp = tmp_path
        self.settings = settings or settings_for(tmp_path)
        self.memory = DualMemorySystem(self.settings)
        if attach:
            self.state = AgentStateStore(
                self.settings.agent_state.path,
                self.settings.emotion.baseline_surprisal,
            )
            cfg = self.settings.event_journal
            self.journal = EventJournal(
                cfg.path,
                cfg.max_bytes,
                cfg.retained_files,
                archive_dir=cfg.archive_dir,
                max_total_bytes=cfg.max_total_bytes,
                soft_band_bytes=cfg.soft_band_bytes,
            )
            self.wal = StateWAL(self.settings.state_wal.directory)
            self.recovery = StateRecoveryCoordinator(self.state, self.journal, self.wal)
        else:
            self.state = AgentStateStore(
                tmp_path / "state.json", baseline_surprisal=1.0, clock=lambda: NOW
            )
            self.journal = EventJournal(
                tmp_path / "events.jsonl", 1_000_000, 4, clock=lambda: NOW
            )
            self.wal = StateWAL(tmp_path / "wal")
            self.recovery = StateRecoveryCoordinator(self.state, self.journal, self.wal)
            boot = self.recovery.prepare_startup()
            self.journal.append_v3_migration_checkpoint()
            StartupReconciliationCoordinator(
                self.journal, self.recovery, self.memory
            ).ensure_adoption_baseline(boot)
            self.journal.append_v4_migration_checkpoint()
        self.experience_store = ExperienceStore.from_memory_root(
            self.settings.memory.persist_directory
        )
        self.semantic_store = SemanticStore.from_memory_root(
            self.settings.memory.persist_directory
        )
        self.qdir = (
            self.settings.degraded_continuation.resolved_quarantine_dir(
                self.settings.event_journal.path
            )
            if attach
            else tmp_path / "quarantine"
        )
        self.status = DegradedContinuationStatus()
        self.transactions = TransactionCoordinator(
            self.journal, lambda _event, _evidence: None
        )

    # -- coordinators

    def policy(self, **overrides: object) -> DomainQuarantinePolicy:
        options: dict[str, object] = {
            "memory_root": self.settings.memory.persist_directory,
            "quarantine_dir": self.qdir,
            "has_exclusive_authority": lambda: self.journal.has_exclusive_authority,
            "status": self.status,
            "confirm_delay_s": 0.0,
            # both domains: Semantic is OFF by default (S0-c), tests enable it
            "participants": frozenset({"memory.experience", "memory.semantic"}),
        }
        options.update(overrides)
        return DomainQuarantinePolicy(**options)  # type: ignore[arg-type]

    def coordinator(
        self, policy: DomainQuarantinePolicy | None = None
    ) -> StartupReconciliationCoordinator:
        return StartupReconciliationCoordinator(
            self.journal,
            self.recovery,
            self.memory,
            self.experience_store,
            self.semantic_store,
            resolution_policy=policy,
        )

    # -- transactions

    def event(self, name: str, sequence: int, *, sleep: bool = False) -> AgentEvent:
        return AgentEvent(
            str(uuid5(NAMESPACE_URL, name)),
            AgentEventType.SLEEP if sleep else AgentEventType.CHAT,
            AgentEventSource.API_SLEEP_RUN if sleep else AgentEventSource.API_CHAT,
            NOW + timedelta(seconds=sequence),
            sequence,
        )

    def prepare(
        self,
        name: str,
        sequence: int,
        *,
        experience: bool = False,
        semantic: bool = False,
        commit_internal: bool = True,
    ) -> Prepared:
        """Accept, start and prepare one transaction needing the chosen memory
        participants; with ``commit_internal`` also publish the internal commit
        (so the open transaction classifies INTERNALLY_COMMITTED, else
        PRE_INTERNAL)."""

        item = self.event(name, sequence, sleep=semantic and not experience)
        transaction_id = TransactionCoordinator.derive_transaction_id(
            item, TransactionKind.EVENT_MUTATION
        )
        participants: list[object] = []
        episode: MemoryEpisodicParticipant | None = None
        experience_participant: MemoryExperienceParticipant | None = None
        semantic_participant: MemorySemanticParticipant | None = None
        experience_id: str | None = None
        if experience:
            episode = MemoryEpisodicParticipant(
                self.memory,
                EpisodicWrite(
                    user_input=f"input {name}",
                    response=f"response {name}",
                    loss=0.2,
                    emotion_valence=0.3,
                    emotion_arousal=0.4,
                    record_type=MemoryRecordType.EPISODIC_LOG,
                    created_at=NOW.isoformat(),
                    context_id="context:1",
                    source_channel="chat",
                    source_session_id="session:1",
                    schema_version=3,
                ),
            )
            experience_id = experience_id_for_event(item.event_id, sequence)
            record = _experience_record(
                experience_id, item.event_id, sequence, episode.episode_id(transaction_id)
            )
            experience_participant = MemoryExperienceParticipant(
                self.memory,
                self.experience_store,
                ExperienceCreateIntent(record, episode.operation_digest),
            )
            participants += [episode, experience_participant]
        if semantic:
            semantic_participant = _semantic_participant(
                self.memory, self.semantic_store, item, transaction_id, name
            )
            participants.append(semantic_participant)
        self.journal.append_accepted(item)
        self.journal.append_started(item)
        self.transactions.prepare_result(
            item, CoordinatedResult("public", tuple(participants))  # type: ignore[arg-type]
        )
        evidence = None
        if commit_internal:
            evidence = self.recovery.commit_internal_candidate(
                item, self.state.load(), snapshot(sequence)
            )
        return Prepared(
            item,
            transaction_id,
            episode,
            experience_participant,
            semantic_participant,
            experience_id,
            evidence,
        )



def _experience_record(
    experience_id: str, event_id: str, sequence: int, episode_id: str
) -> ExperienceRecord:
    genesis = ExperienceRevisionRecord(
        experience_id,
        0,
        ExperienceRevisionOperation.CREATE,
        ExperienceRevisionReason.CREATION,
        NOW,
        event_id,
        sequence,
        evidence_refs=(event_id,),
    )
    return ExperienceRecord(
        experience_id=experience_id,
        revision=0,
        lifecycle=ExperienceLifecycle.ACTIVE,
        source_event_id=event_id,
        source_event_sequence=sequence,
        source_episode_id=episode_id,
        context_id="context:1",
        measurement=ExperienceMeasurementEvidence(
            "model." + "a" * 64, True, calibrated_novelty=0.25
        ),
        appraisal=ExperienceAppraisalEvidence(
            novelty=0.25,
            novelty_valid=True,
            reason_codes=(ExperienceAppraisalReasonCode.NOVELTY_MEASURED,),
        ),
        pre_appraisal_emotion=ExperienceEmotionProjection(0.0, 0.0),
        temporal_update_reasons=(ExperienceEmotionUpdateReasonCode.TIMELINE_INITIALIZED,),
        post_appraisal_emotion=ExperienceEmotionProjection(0.5, 0.25),
        emotion_contributions=ExperienceEmotionContributions(),
        emotion_update_reasons=(ExperienceEmotionUpdateReasonCode.APPRAISAL_APPLIED,),
        subjective_salience=0.25,
        created_at=NOW,
        revision_history=(genesis,),
    )


def _semantic_participant(
    memory: DualMemorySystem,
    store: SemanticStore,
    item: AgentEvent,
    transaction_id: str,
    name: str,
) -> MemorySemanticParticipant:
    semantic_id = semantic_id_for_batch_entry(transaction_id, 0)
    revision = SemanticRevision(
        semantic_id=semantic_id,
        revision=0,
        semantic_content=f"semantic fact {name}",
        content_digest=semantic_content_digest(f"semantic fact {name}"),
        created_at=item.requested_at,
        event_id=item.event_id,
        event_sequence=item.processing_sequence or 0,
    )
    revision = replace(revision, source_edges=(), provenance_class=None)
    operation = SemanticBatchOperation(
        transaction_id, (SemanticBatchEntry(0, SemanticCreateIntent(revision)),)
    )
    return MemorySemanticParticipant(memory, store, operation)


def binding_for(
    prepared: Prepared, participant_id: str, digest: str
) -> TransactionBinding:
    return TransactionBinding(
        prepared.transaction_id,
        prepared.event.event_id,
        prepared.event.processing_sequence or 0,
        participant_id,
        digest,
        TransactionKind.EVENT_MUTATION,
    )


def vanish(path: Path) -> None:
    """Move a store directory out of the way WITHOUT the writer (a test that
    needs the 'directory is gone' fault shape)."""

    path.rename(path.with_name(path.name + ".gone"))


