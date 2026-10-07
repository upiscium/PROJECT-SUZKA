"""Integrated runtime main loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
import inspect
from typing import TYPE_CHECKING, Callable, cast

from suzka.attention.adapters import (
    project_attention_candidates,
    project_global_emotion,
)
from suzka.attention.contracts import (
    AttentionContinuity,
    AttentionEvent,
    AttentionTarget,
)
from suzka.attention.system import (
    AttentionRefreshResult,
    AttentionSelectedView,
    AttentionSystem,
)
from suzka.body import EmotionEngineAllostasis, EmotionState, EmotionUpdate
from suzka.belief import BeliefDomainError, BeliefSystem, BeliefSystemSnapshot
from suzka.cognition import (
    AppraisalResult,
    AppraisalSignals,
    CognitiveAppraiser,
    LossCalibration,
    LossMeasurement,
    SurprisalCalculator,
    model_key,
)
from suzka.config import Settings
from suzka.identity import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    ValueAdmissionStatus,
    ValueConflictDefinition,
    ValueDomainError,
    ValueMutationEvidence,
    ValueMutationResult,
    ValueNotFound,
    ValueOriginReviewDecision,
    ValuePromptView,
    ValueRevisionOperation,
    ValueSystem,
    ValueSystemSnapshot,
)
from suzka.memory import DualMemorySystem, MemoryContext, MemoryRecordType
from suzka.experience import (
    ExperienceAppraisalEvidence,
    ExperienceEmotionContributions,
    ExperienceEmotionProjection,
    ExperienceEmotionUpdateReasonCode,
    ExperienceMeasurementEvidence,
    ExperienceRecord,
    ExperienceLifecycle,
    ExperienceRevisionOperation,
    ExperienceRevisionReason,
    ExperienceRevisionRecord,
    calculate_subjective_salience,
)
from suzka.memory.experience_store import ExperienceStore
from suzka.models import ModelProvider
from suzka.motivation.commitment_system import (
    CommitmentSystem,
    CommitmentSystemSnapshot,
)
from suzka.motivation.goal_system import GoalSystem, GoalSystemSnapshot
from suzka.motivation.projection import R13PromptView
from suzka.motivation.system import MotivationSystem, MotivationSystemSnapshot
from suzka.persona import (
    ConsciousAgent,
    ContextPromptView,
    PromptBuilder,
    ResponsePostprocessor,
)
from suzka.persona.attention_prompt import (
    AttentionPromptPayload,
    build_attention_prompt_payload,
)
from suzka.runtime.chat_context import ChatContextSelectors, resolve_chat_context
from suzka.runtime.agent_runtime import (
    AgentEvent,
    AgentEventSource,
    AgentEventType,
    AgentRuntime,
)
from suzka.runtime.session_participant import (
    SessionTurnOperation,
    SessionTurnParticipant,
)
from suzka.runtime.session_state import SessionState
from suzka.runtime.context import (
    ContextFrame,
    ContextRegistry,
    ContextRegistryState,
    ContextStatus,
    validate_context_registry_state,
)
from suzka.runtime.attention_state_codec import validated_attention_continuity
from suzka.runtime.attention_state_port import AttentionStatePort
from suzka.runtime.transaction_coordinator import (
    CoordinatedResult,
    TransactionBoundValue,
    TransactionParticipant,
    TransactionKind,
    TransactionCoordinator,
)
from suzka.runtime.working_memory import (
    WorkingMemory,
    WorkingMemorySourceKind,
    WorkingMemoryView,
)
from suzka.runtime.r13_codec import (
    R13_CODEC_SCHEMA_VERSION,
    R13GraphSnapshot,
    R13StateSnapshot,
)

if TYPE_CHECKING:
    from suzka.memory.episodic_participant import MemoryEpisodicParticipant
    from suzka.memory.experience_participant import MemoryExperienceParticipant
    from suzka.runtime.agent_state import (
        CommitmentStatePort,
        GoalStatePort,
        MotivationStatePort,
    )


@dataclass(frozen=True)
class ChatResult:
    """Visible/structured result safe to pass to ordinary callers."""

    episode_id: str
    response: str
    loss: float | None
    valence: float
    arousal: float
    optimal_loss: float
    model_id: str
    adapter_id: str | None


@dataclass(frozen=True)
class DebugChatTrace:
    """Request-scoped diagnostic data that must never become durable authority."""

    hidden_thought: str
    prompt: str
    memory_context: MemoryContext
    working_memory_view: WorkingMemoryView
    diagnostics: ChatDiagnostics


@dataclass(frozen=True)
class ChatDiagnostics:
    """Request-scoped structured cognition diagnostics."""

    measurement: LossMeasurement
    appraisal: AppraisalResult
    temporal_update: EmotionUpdate
    emotion_update: EmotionUpdate


@dataclass(frozen=True)
class _ComputedChat:
    response: str
    loss: float | None
    valence: float
    arousal: float
    optimal_loss: float
    trace: DebugChatTrace | None
    diagnostics: ChatDiagnostics
    memory_participant: MemoryEpisodicParticipant
    experience_participant: MemoryExperienceParticipant | None
    session_participant: SessionTurnParticipant


@dataclass(frozen=True, slots=True)
class _CommittedR13Bundle:
    """One atomically published, detached R13 view and its source states."""

    motivation_state: MotivationSystemSnapshot
    goal_state: GoalSystemSnapshot
    commitment_state: CommitmentSystemSnapshot
    view: R13PromptView


@dataclass(frozen=True, slots=True)
class _CommittedReadViews:
    """Atomically published ordinary read views across all owned domains."""

    value_snapshot: ValueSystemSnapshot
    belief_snapshot: BeliefSystemSnapshot
    r13_bundle: _CommittedR13Bundle
    attention_view: AttentionSelectedView


@dataclass(frozen=True, slots=True)
class _AgentStatePorts:
    """The existing state ports, plus this loop's concrete Attention owner."""

    motivation_state_port: MotivationStatePort
    goal_state_port: GoalStatePort
    commitment_state_port: CommitmentStatePort
    attention_state_port: AttentionStatePort


def _validated_attention_view(value: object) -> AttentionSelectedView:
    """Return a detached, revalidated copy of an Attention metadata view."""

    if type(value) is not AttentionSelectedView:
        raise TypeError("attention view must be an exact AttentionSelectedView")
    value.__post_init__()
    event = (
        None
        if value.event is None
        else AttentionEvent.from_canonical_value(value.event.canonical_value())
    )
    focused_targets = tuple(
        AttentionTarget.from_canonical_value(target.canonical_value())
        for target in value.focused_targets
    )
    unfinished_targets = tuple(
        AttentionTarget.from_canonical_value(target.canonical_value())
        for target in value.unfinished_targets
    )
    competition = value.competition
    if competition is not None:
        competition_value = competition.canonical_value()
        competition = copy.deepcopy(competition)
        competition.__post_init__()
        if competition.canonical_value() != competition_value:
            raise ValueError("Attention competition changed while copying its view")
    prompt = value.prompt
    if prompt is not None:
        prompt_value = prompt.canonical_value()
        prompt = copy.deepcopy(prompt)
        prompt.__post_init__()
        if prompt.canonical_value() != prompt_value:
            raise ValueError("Attention prompt changed while copying its view")
    result = AttentionSelectedView(
        revision=value.revision,
        event=event,
        state_digest=value.state_digest,
        authority_digest=value.authority_digest,
        focused_targets=focused_targets,
        unfinished_targets=unfinished_targets,
        competition=competition,
        prompt=prompt,
    )
    result.__post_init__()
    return result


def _attention_view_matches_state(
    view: AttentionSelectedView, snapshot: AttentionContinuity
) -> bool:
    """Check that the selection metadata is bound to an exact continuity root."""

    target_by_id = {
        item.candidate_id: item.target for item in snapshot.candidates
    }
    return (
        view.revision == snapshot.revision
        and view.event == snapshot.last_event
        and view.state_digest == snapshot.state_digest
        and view.authority_digest == snapshot.authority_digest
        and tuple(target.candidate_id for target in view.focused_targets)
        == snapshot.focused_ids
        and all(
            target_by_id.get(target.candidate_id) == target
            for target in view.focused_targets
        )
        and tuple(target.candidate_id for target in view.unfinished_targets)
        == snapshot.unfinished_ids
        and all(
            target_by_id.get(target.candidate_id) == target
            for target in view.unfinished_targets
        )
    )


class SuzkaMainLoop:
    """Connect prediction error, emotion, memory, generation, and storage."""

    def __init__(
        self,
        settings: Settings,
        provider: ModelProvider,
        memory_system: DualMemorySystem,
        *,
        session_state: SessionState | None = None,
        working_memory: WorkingMemory | None = None,
        emotion_engine: EmotionEngineAllostasis | None = None,
        loss_calibration: LossCalibration | None = None,
        prompt_builder: PromptBuilder | None = None,
        agent: ConsciousAgent | None = None,
        postprocessor: ResponsePostprocessor | None = None,
        adapter_id: str | None = None,
        context_registry: ContextRegistry | None = None,
        value_system: ValueSystem | None = None,
        belief_system: BeliefSystem | None = None,
        experience_store: ExperienceStore | None = None,
        motivation_system: MotivationSystem | None = None,
        goal_system: GoalSystem | None = None,
        commitment_system: CommitmentSystem | None = None,
        attention_system: AttentionSystem | None = None,
    ) -> None:
        from suzka.memory.working_memory_resolver import MemoryWorkingMemoryResolver

        self.settings = settings
        self.provider = provider
        self.memory_system = memory_system
        self.experience_store = experience_store or ExperienceStore.from_memory_root(
            memory_system.settings.memory.persist_directory
        )
        self._runtime: object | None = None
        self.working_memory_resolver = MemoryWorkingMemoryResolver(memory_system)
        self.session_state = session_state or SessionState()
        self.working_memory = (
            working_memory
            if working_memory is not None
            else WorkingMemory(
                item_capacity=settings.working_memory.item_capacity,
                projection_max_bytes=settings.working_memory.projection_max_bytes,
            )
        )
        self.surprisal_calculator = SurprisalCalculator(provider)
        self.appraiser = CognitiveAppraiser()
        self.primary_model_key = model_key(
            settings.model.provider, settings.model.primary_id
        )
        approved_keys = tuple(
            sorted(
                {
                    self.primary_model_key,
                    model_key(settings.model.provider, settings.model.fallback_id),
                }
            )
        )
        if loss_calibration is None:
            self.loss_calibration = LossCalibration(
                approved_keys,
                initial_baseline=settings.emotion.baseline_surprisal,
                initial_scale=settings.appraisal.initial_loss_scale,
                minimum_scale=settings.appraisal.minimum_loss_scale,
            )
        else:
            if not isinstance(loss_calibration, LossCalibration):
                raise TypeError("loss_calibration must be LossCalibration")
            if loss_calibration.approved_keys != approved_keys:
                raise ValueError(
                    "loss_calibration approved keys do not match settings"
                )
            self.loss_calibration = loss_calibration
        self.emotion_engine = emotion_engine or EmotionEngineAllostasis(
            EmotionState(optimal_loss=settings.emotion.baseline_surprisal),
            adaptation_rate=settings.emotion.decay_rate,
            appraisal_response_rate=settings.emotion.appraisal_response_rate,
            resting_valence=settings.emotion.resting_valence,
            resting_arousal=settings.emotion.resting_arousal,
            valence_recovery_rate=settings.emotion.valence_recovery_rate,
            arousal_recovery_rate=settings.emotion.arousal_recovery_rate,
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.agent = agent or ConsciousAgent(provider)
        self.postprocessor = postprocessor or ResponsePostprocessor()
        self.adapter_id = adapter_id
        self.context_registry = (
            context_registry if context_registry is not None else ContextRegistry()
        )
        configured_seeds = tuple(seed.to_declaration() for seed in settings.values.seeds)
        configured_conflicts = tuple(
            ValueConflictDefinition(
                left_value_id=conflict.left_value_id,
                right_value_id=conflict.right_value_id,
            )
            for conflict in settings.values.conflicts
        )
        if value_system is None:
            authority = ValueSystem.from_seed_declarations(
                configured_seeds, configured_conflicts
            )
        else:
            if not isinstance(value_system, ValueSystem):
                raise TypeError("value_system must be ValueSystem")
            authority = ValueSystem.restore_snapshot(value_system.snapshot())
        self._value_system = authority
        if belief_system is None:
            belief_authority = BeliefSystem(event_provider=self._current_belief_event)
        else:
            if not isinstance(belief_system, BeliefSystem):
                raise TypeError("belief_system must be BeliefSystem")
            belief_authority = BeliefSystem.restore_snapshot(
                belief_system.snapshot(), event_provider=self._current_belief_event
            )
        self._belief_system = belief_authority
        if motivation_system is None:
            motivation_authority = MotivationSystem()
        else:
            if not isinstance(motivation_system, MotivationSystem):
                raise TypeError("motivation_system must be MotivationSystem")
            motivation_authority = MotivationSystem()
            motivation_authority.restore_motivation_state(
                motivation_system.snapshot()
            )
        self._motivation_system = motivation_authority
        if goal_system is None:
            goal_authority = GoalSystem()
        else:
            if not isinstance(goal_system, GoalSystem):
                raise TypeError("goal_system must be GoalSystem")
            goal_authority = GoalSystem()
            goal_authority.restore_goal_state(goal_system.snapshot())
        self._goal_system = goal_authority
        if commitment_system is None:
            commitment_authority = CommitmentSystem()
        else:
            if not isinstance(commitment_system, CommitmentSystem):
                raise TypeError("commitment_system must be CommitmentSystem")
            commitment_authority = CommitmentSystem()
            commitment_authority.restore_commitment_state(
                commitment_system.snapshot()
            )
        self._commitment_system = commitment_authority
        if attention_system is None:
            attention_authority = AttentionSystem()
        else:
            if type(attention_system) is not AttentionSystem:
                raise TypeError("attention_system must be AttentionSystem")
            attention_authority = AttentionSystem(attention_system.snapshot())
        self._attention_system = attention_authority
        self._attention_state_port = AttentionStatePort(attention_authority)
        self._agent_state_ports = _AgentStatePorts(
            self,
            self,
            self,
            self._attention_state_port,
        )
        initial_r13_state = R13StateSnapshot(
            schema_version=R13_CODEC_SCHEMA_VERSION,
            motivation=R13GraphSnapshot.capture(
                self._motivation_system.snapshot()
            ),
            goal=R13GraphSnapshot.capture(self._goal_system.snapshot()),
            commitment=R13GraphSnapshot.capture(
                self._commitment_system.snapshot()
            ),
        )
        self._committed_read_views = self._prepare_committed_attention_read_views(
            authority.snapshot(),
            belief_authority.snapshot(),
            initial_r13_state,
            attention_authority.snapshot(),
            retain_selection=False,
        )

    @property
    def agent_state_ports(self) -> _AgentStatePorts:
        """Expose the exact state-port topology consumed by AgentState only."""

        return self._agent_state_ports

    @property
    def value_system(self) -> ValueSystem:
        """Return a detached read-only-by-convention Value projection."""

        return ValueSystem.restore_snapshot(
            self._committed_read_views.value_snapshot
        )

    @property
    def _committed_value_snapshot(self) -> ValueSystemSnapshot:
        """Compatibility read for event validation from the shared bundle."""

        return self._committed_read_views.value_snapshot

    @property
    def _committed_belief_snapshot(self) -> BeliefSystemSnapshot:
        """Compatibility read for event validation from the shared bundle."""

        return self._committed_read_views.belief_snapshot

    @property
    def _committed_r13_bundle(self) -> _CommittedR13Bundle:
        """Compatibility read from the shared ordinary-view bundle."""

        return self._committed_read_views.r13_bundle

    def _value_system_for_state(self) -> ValueSystem:
        """Return the internal Value authority to AgentState only."""

        return self._value_system

    def _replace_value_system_for_state(self, value_system: ValueSystem) -> None:
        if not isinstance(value_system, ValueSystem):
            raise TypeError("value_system must be ValueSystem")
        self._value_system = value_system

    def _publish_committed_value_view(self) -> None:
        """Publish the Value state after the runtime commit protocol succeeds."""

        snapshot = self._value_system.snapshot()
        current = self._committed_read_views
        self._committed_read_views = _CommittedReadViews(
            snapshot,
            current.belief_snapshot,
            current.r13_bundle,
            current.attention_view,
        )

    def _committed_read_view_checkpoint(self) -> _CommittedReadViews:
        """Return the opaque last-published handle for trusted server recovery."""

        return self._committed_read_views

    @property
    def belief_system(self) -> BeliefSystem:
        """Return a detached read-only-by-convention Belief projection."""

        return BeliefSystem.restore_snapshot(
            self._committed_read_views.belief_snapshot
        )

    def export_belief_state(self) -> BeliefSystemSnapshot:
        """Export an immutable Belief snapshot for AgentState capture."""

        return self._belief_system.snapshot()

    def restore_belief_state(self, snapshot: BeliefSystemSnapshot) -> None:
        """Replace Belief authority through the explicit AgentState port."""

        if not isinstance(snapshot, BeliefSystemSnapshot):
            raise TypeError("snapshot must be BeliefSystemSnapshot")
        restored = BeliefSystem.restore_snapshot(
            snapshot, event_provider=self._current_belief_event
        )
        self._belief_system = restored

    def _current_belief_event(self) -> object | None:
        runtime = self._runtime
        if not isinstance(runtime, AgentRuntime):
            return None
        return runtime.current_event()

    def _publish_committed_belief_view(self) -> None:
        """Publish Belief state after the runtime commit protocol succeeds."""

        snapshot = self._belief_system.snapshot()
        current = self._committed_read_views
        self._committed_read_views = _CommittedReadViews(
            current.value_snapshot,
            snapshot,
            current.r13_bundle,
            current.attention_view,
        )

    def export_motivation_state(self) -> MotivationSystemSnapshot:
        """Export a complete immutable Motivation snapshot for AgentState."""

        return self._motivation_system.snapshot()

    def restore_motivation_state(self, snapshot: MotivationSystemSnapshot) -> None:
        """Replace only Motivation authority through the AgentState port."""

        self._motivation_system.restore_motivation_state(snapshot)

    def export_goal_state(self) -> GoalSystemSnapshot:
        """Export a complete immutable Goal snapshot for AgentState."""

        return self._goal_system.snapshot()

    def restore_goal_state(self, snapshot: GoalSystemSnapshot) -> None:
        """Replace only Goal authority through the AgentState port."""

        self._goal_system.restore_goal_state(snapshot)

    def export_commitment_state(self) -> CommitmentSystemSnapshot:
        """Export a complete immutable Commitment snapshot for AgentState."""

        return self._commitment_system.snapshot()

    def restore_commitment_state(
        self, snapshot: CommitmentSystemSnapshot
    ) -> None:
        """Replace only Commitment authority through the AgentState port."""

        self._commitment_system.restore_commitment_state(snapshot)

    def r13_view(self) -> R13PromptView:
        """Return the one atomically committed immutable R13 prompt view."""

        return self._committed_read_views.r13_bundle.view

    def attention_view(self) -> AttentionSelectedView:
        """Return a detached copy of only the last committed Attention metadata."""

        return _validated_attention_view(
            self._committed_read_views.attention_view
        )

    @staticmethod
    def _prepare_committed_r13_view(
        r13_state: R13StateSnapshot,
    ) -> _CommittedR13Bundle:
        """Decode, validate, and project R13 without changing runtime state."""

        if type(r13_state) is not R13StateSnapshot:
            raise TypeError("r13_state must be an exact R13StateSnapshot")
        validated = R13StateSnapshot(
            schema_version=r13_state.schema_version,
            motivation=R13GraphSnapshot.model_validate(
                r13_state.motivation.model_dump(mode="python")
            ),
            goal=R13GraphSnapshot.model_validate(
                r13_state.goal.model_dump(mode="python")
            ),
            commitment=R13GraphSnapshot.model_validate(
                r13_state.commitment.model_dump(mode="python")
            ),
        )
        motivation_state = validated.motivation.restore()
        goal_state = validated.goal.restore()
        commitment_state = validated.commitment.restore()
        if type(motivation_state) is not MotivationSystemSnapshot:
            raise ValueError("R13 motivation graph has an invalid root type")
        if type(goal_state) is not GoalSystemSnapshot:
            raise ValueError("R13 goal graph has an invalid root type")
        if type(commitment_state) is not CommitmentSystemSnapshot:
            raise ValueError("R13 commitment graph has an invalid root type")
        view = R13PromptView.from_snapshots(
            motivation_state, goal_state, commitment_state
        )
        return _CommittedR13Bundle(
            motivation_state,
            goal_state,
            commitment_state,
            view,
        )

    def _publish_committed_r13_view(
        self, bundle: _CommittedR13Bundle
    ) -> None:
        """Publish R13 while preserving Value/Belief views for compatibility."""

        current = self._committed_read_views
        self._committed_read_views = _CommittedReadViews(
            current.value_snapshot,
            current.belief_snapshot,
            bundle,
            current.attention_view,
        )

    def _prepare_committed_read_views(
        self,
        value_snapshot: ValueSystemSnapshot,
        belief_snapshot: BeliefSystemSnapshot,
        r13_state: R13StateSnapshot,
    ) -> _CommittedReadViews:
        """Validate every ordinary read projection before atomic publication."""

        (
            validated_value_snapshot,
            validated_belief_snapshot,
            r13_bundle,
        ) = self._prepare_committed_domain_read_views(
            value_snapshot,
            belief_snapshot,
            r13_state,
        )
        current = getattr(self, "_committed_read_views", None)
        attention_view = (
            self._attention_system.selected_view()
            if current is None
            else current.attention_view
        )
        return _CommittedReadViews(
            validated_value_snapshot,
            validated_belief_snapshot,
            r13_bundle,
            _validated_attention_view(attention_view),
        )

    def _prepare_committed_domain_read_views(
        self,
        value_snapshot: ValueSystemSnapshot,
        belief_snapshot: BeliefSystemSnapshot,
        r13_state: R13StateSnapshot,
    ) -> tuple[ValueSystemSnapshot, BeliefSystemSnapshot, _CommittedR13Bundle]:
        """Stage the existing Value, Belief, and R13 roots without side reads."""

        if type(value_snapshot) is not ValueSystemSnapshot:
            raise TypeError("value_snapshot must be an exact ValueSystemSnapshot")
        if type(belief_snapshot) is not BeliefSystemSnapshot:
            raise TypeError("belief_snapshot must be an exact BeliefSystemSnapshot")
        validated_value_snapshot = ValueSystem.restore_snapshot(
            value_snapshot
        ).snapshot()
        validated_belief_snapshot = BeliefSystem.restore_snapshot(
            belief_snapshot
        ).snapshot()
        r13_bundle = self._prepare_committed_r13_view(r13_state)
        return (
            validated_value_snapshot,
            validated_belief_snapshot,
            r13_bundle,
        )

    def _prepare_committed_attention_read_views(
        self,
        value_snapshot: ValueSystemSnapshot,
        belief_snapshot: BeliefSystemSnapshot,
        r13_state: R13StateSnapshot,
        attention_state: AttentionContinuity,
        *,
        retain_selection: bool = False,
    ) -> _CommittedReadViews:
        """Stage all committed read projections for one explicit Attention root."""

        if type(retain_selection) is not bool:
            raise TypeError("retain_selection must be an exact bool")
        validated_attention = validated_attention_continuity(attention_state)
        (
            validated_value_snapshot,
            validated_belief_snapshot,
            r13_bundle,
        ) = self._prepare_committed_domain_read_views(
            value_snapshot,
            belief_snapshot,
            r13_state,
        )
        if retain_selection:
            selected_view = self._attention_system.selected_view()
            if not _attention_view_matches_state(selected_view, validated_attention):
                raise ValueError(
                    "live Attention selection does not match the committed root"
                )
        else:
            # Restore/constructor staging must use the supplied target root, not
            # speculative metadata belonging to the current live owner.
            selected_view = AttentionSystem(validated_attention).selected_view()
        return _CommittedReadViews(
            validated_value_snapshot,
            validated_belief_snapshot,
            r13_bundle,
            _validated_attention_view(selected_view),
        )

    def _publish_committed_read_views(self, bundle: _CommittedReadViews) -> None:
        """Publish all ordinary domain reads with one non-fallible ref swap."""

        self._committed_read_views = bundle

    def _validate_belief_event_commit(self, event: AgentEvent) -> None:
        """Require every newly published Belief revision to bind this event."""

        if event.processing_sequence is None or event.processing_sequence <= 0:
            raise BeliefDomainError("Belief commit event has no processing sequence")
        self._belief_system.validate()
        current = self._belief_system.snapshot()
        previous = self._committed_belief_snapshot
        if current == previous:
            return
        previous_records = {record.belief_id: record for record in previous.records}
        current_records = {record.belief_id: record for record in current.records}
        if set(previous_records) - set(current_records):
            raise BeliefDomainError("Belief authority removed a committed record")

        for belief_id, record in current_records.items():
            prior = previous_records.get(belief_id)
            if prior == record:
                continue
            prior_digests = (
                {revision.record_digest for revision in prior.revision_history}
                if prior is not None
                else set()
            )
            new_revisions = tuple(
                revision
                for revision in record.revision_history
                if revision.record_digest not in prior_digests
            )
            if not new_revisions:
                raise BeliefDomainError(
                    "Belief change has no committing revision"
                )
            for revision in new_revisions:
                if (
                    revision.event_id != event.event_id
                    or revision.event_sequence != event.processing_sequence
                ):
                    raise BeliefDomainError(
                        "Belief revision is not bound to the committing event"
                    )

    def _validate_value_event_commit(self, event: AgentEvent) -> None:
        """Require newly introduced Value history to name the committing event."""

        self._value_system.validate()
        current = self._value_system.snapshot()
        previous = self._committed_value_snapshot
        if current == previous:
            return
        if current.conflicts != previous.conflicts:
            raise ValueDomainError("Value conflicts changed outside their authority")
        previous_values = {value.value_id: value for value in previous.values}
        current_values = {value.value_id: value for value in current.values}
        previous_histories = {
            history.value_id: history for history in previous.histories
        }
        current_histories = {
            history.value_id: history for history in current.histories
        }
        previous_ledgers = dict(previous.evidence_ledgers)
        current_ledgers = dict(current.evidence_ledgers)
        previous_ledger_digests = dict(previous.evidence_ledger_digests)
        current_ledger_digests = dict(current.evidence_ledger_digests)
        if not (
            set(current_values)
            == set(current_histories)
            == set(current_ledgers)
            == set(current_ledger_digests)
        ):
            raise ValueDomainError("Value authority keys are inconsistent at commit")
        if set(previous_values) - set(current_values):
            raise ValueDomainError("Value state was removed outside its authority")

        for value_id, value in current_values.items():
            previous_value = previous_values.get(value_id)
            previous_history = previous_histories.get(value_id)
            current_history = current_histories.get(value_id)
            if current_history is None:
                raise ValueDomainError("Value history is missing at commit")
            prior_record_digests = (
                {record.record_digest for record in previous_history.records}
                if previous_history is not None
                else set()
            )
            new_records = tuple(
                record
                for record in current_history.records
                if record.record_digest not in prior_record_digests
            )
            changed = (
                previous_value != value
                or previous_history != current_history
                or previous_ledgers.get(value_id) != current_ledgers.get(value_id)
                or previous_ledger_digests.get(value_id)
                != current_ledger_digests.get(value_id)
            )
            if not changed:
                continue
            if not new_records:
                raise ValueDomainError("Value change has no committing revision")
            for record in new_records:
                expected_source = {
                    ValueRevisionOperation.ADMISSION: AgentEventSource.API_VALUES_SEED_ADOPT,
                    ValueRevisionOperation.FREEZE: AgentEventSource.API_VALUES_FREEZE,
                    ValueRevisionOperation.UNFREEZE: AgentEventSource.API_VALUES_UNFREEZE,
                    ValueRevisionOperation.ORIGIN_REVIEW: AgentEventSource.API_VALUES_ORIGIN_REVIEW,
                    ValueRevisionOperation.ROLLBACK: AgentEventSource.API_VALUES_ROLLBACK,
                }.get(record.operation)
                if (
                    event.event_type is not AgentEventType.VALUE_GOVERNANCE
                    or expected_source is None
                    or event.source is not expected_source
                ):
                    raise ValueDomainError(
                        "Value revision is not authorized by a governance event"
                    )
                governance_origin_id = IdentityOrigin(
                    OriginActor.OPERATOR,
                    OriginInputKind.CONSTRAINT,
                    ValueAdmissionStatus.PENDING,
                    source_ref=event.source.value,
                    event_id=event.event_id,
                    event_sequence=event.processing_sequence,
                ).origin_id
                if record.origin_id != governance_origin_id:
                    raise ValueDomainError(
                        "Value revision origin is not bound to governance provenance"
                    )
                if (
                    record.event_id != event.event_id
                    or record.event_sequence != event.processing_sequence
                ):
                    raise ValueDomainError(
                        "Value revision is not bound to the committing event"
                    )

    def bind_runtime(self, runtime: object) -> None:
        """Bind the one application runtime allowed to govern this authority."""

        if self._runtime is not None and self._runtime is not runtime:
            raise RuntimeError("SuzkaMainLoop is already bound to another runtime")
        self._runtime = runtime

    def _validated_chat_event(self, *, capture_debug: bool) -> AgentEvent | None:
        """Validate the chat method against the active runtime event, when bound."""

        if self._runtime is None:
            return None
        if not isinstance(self._runtime, AgentRuntime):
            raise RuntimeError("chat runtime binding is invalid")
        event = self._runtime.current_event()
        if event is None:
            raise RuntimeError("chat requires the active runtime event")
        expected_type = AgentEventType.DEBUG_CHAT if capture_debug else AgentEventType.CHAT
        expected_source = (
            AgentEventSource.API_CHAT_DEBUG
            if capture_debug
            else AgentEventSource.API_CHAT
        )
        if event.event_type is not expected_type or event.source is not expected_source:
            raise RuntimeError("chat method does not match the active runtime event")
        if event.processing_sequence is None or event.processing_sequence <= 0:
            raise RuntimeError("chat event has no processing sequence")
        return event

    def _governance_context(
        self,
        runtime: AgentRuntime, expected_source: AgentEventSource
    ) -> tuple[ValueMutationEvidence, IdentityOrigin]:
        if not isinstance(runtime, AgentRuntime):
            raise TypeError("governance runtime must be an AgentRuntime")
        if self._runtime is not runtime:
            raise ValueDomainError("governance runtime is not the bound application runtime")
        event = runtime.current_event()
        if event is None:
            raise ValueDomainError("Value governance requires the active runtime event")
        if event.event_type is not AgentEventType.VALUE_GOVERNANCE:
            raise ValueDomainError("Value governance requires a governance event")
        if event.source is not expected_source:
            raise ValueDomainError("Value governance source is not authorized")
        if event.processing_sequence is None:
            raise ValueDomainError("Value governance event has no processing sequence")
        evidence = ValueMutationEvidence(
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            recorded_at=event.requested_at,
        )
        origin = IdentityOrigin(
            OriginActor.OPERATOR,
            OriginInputKind.CONSTRAINT,
            ValueAdmissionStatus.PENDING,
            source_ref=event.source.value,
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            confidence=1.0,
        )
        return evidence, origin

    def freeze_value(self, runtime: AgentRuntime, value_id: str) -> ValueMutationResult:
        evidence, origin = self._governance_context(
            runtime, AgentEventSource.API_VALUES_FREEZE
        )
        return self._value_system.freeze(
            value_id, evidence, governance_origin=origin
        )

    def unfreeze_value(self, runtime: AgentRuntime, value_id: str) -> ValueMutationResult:
        evidence, origin = self._governance_context(
            runtime, AgentEventSource.API_VALUES_UNFREEZE
        )
        return self._value_system.unfreeze(
            value_id, evidence, governance_origin=origin
        )

    def rollback_value(
        self, runtime: AgentRuntime, value_id: str, target_revision: int
    ) -> ValueMutationResult:
        evidence, origin = self._governance_context(
            runtime, AgentEventSource.API_VALUES_ROLLBACK
        )
        return self._value_system.rollback(
            value_id,
            target_revision,
            evidence,
            governance_origin=origin,
        )

    def adopt_configured_value_seed(
        self, runtime: AgentRuntime, value_id: str
    ) -> ValueMutationResult:
        evidence, origin = self._governance_context(
            runtime, AgentEventSource.API_VALUES_SEED_ADOPT
        )
        for configured_seed in self.settings.values.seeds:
            if configured_seed.value_id == value_id:
                return self._value_system.adopt_seed(
                    configured_seed.to_declaration(),
                    evidence,
                    governance_origin=origin,
                )
        raise ValueNotFound("configured Value seed not found")

    def review_value_origin(
        self,
        runtime: AgentRuntime,
        value_id: str,
        decision: ValueOriginReviewDecision,
    ) -> ValueMutationResult:
        evidence, origin = self._governance_context(
            runtime, AgentEventSource.API_VALUES_ORIGIN_REVIEW
        )
        return self._value_system.review_origin(
            value_id,
            decision,
            evidence,
            governance_origin=origin,
        )

    def chat(
        self,
        user_input: str,
        selectors: ChatContextSelectors | None = None,
    ) -> CoordinatedResult[ChatResult]:
        """Compute one live turn inside the serialized AgentRuntime handler."""

        return self._chat_plan(
            user_input,
            capture_debug=False,
            selectors=selectors,
        )

    def _chat_plan(
        self,
        user_input: str,
        *,
        capture_debug: bool,
        selectors: ChatContextSelectors | None,
    ) -> CoordinatedResult[ChatResult]:
        computed = self._run_chat(
            user_input,
            capture_debug=capture_debug,
            selectors=selectors,
            context_registry=self.context_registry,
            working_memory=self.working_memory,
            emotion_engine=self.emotion_engine,
        )
        return CoordinatedResult(
            TransactionBoundValue(
                lambda transaction_id: self._chat_result(computed, transaction_id)
            ),
            self._participants(computed),
        )

    def chat_debug(
        self,
        user_input: str,
        selectors: ChatContextSelectors | None = None,
    ) -> CoordinatedResult[tuple[ChatResult, DebugChatTrace]]:
        """Compute one live debug turn inside the serialized AgentRuntime handler."""

        return self._debug_chat_plan(
            user_input,
            selectors=selectors,
        )

    def emotion_tick(self) -> None:
        """Advance idle emotion state inside the serialized runtime handler."""

        self.emotion_engine.advance_to()
        return None

    def _debug_chat_plan(
        self,
        user_input: str,
        *,
        selectors: ChatContextSelectors | None,
    ) -> CoordinatedResult[tuple[ChatResult, DebugChatTrace]]:
        computed = self._run_chat(
            user_input,
            capture_debug=True,
            selectors=selectors,
            context_registry=self.context_registry,
            working_memory=self.working_memory,
            emotion_engine=self.emotion_engine,
        )
        if computed.trace is None:  # pragma: no cover - internal invariant
            raise RuntimeError("Debug trace was not captured")
        trace = computed.trace
        return CoordinatedResult(
            TransactionBoundValue(
                lambda transaction_id: (
                    self._chat_result(computed, transaction_id),
                    trace,
                )
            ),
            self._participants(computed),
        )

    def _run_chat(
        self,
        user_input: str,
        *,
        capture_debug: bool,
        selectors: ChatContextSelectors | None,
        context_registry: ContextRegistry,
        working_memory: WorkingMemory,
        emotion_engine: EmotionEngineAllostasis,
    ) -> _ComputedChat:
        from suzka.memory.episodic_participant import (
            EpisodicWrite,
            MemoryEpisodicParticipant,
            episodic_episode_id,
        )
        from suzka.memory.experience_participant import (
            ExperienceCreateIntent,
            MemoryExperienceParticipant,
            experience_id_for_event,
        )

        event = self._validated_chat_event(capture_debug=capture_debug)
        current_context = resolve_chat_context(context_registry, selectors)
        provenance = (
            current_context.context_id,
            current_context.source_channel,
            current_context.source_session_id,
        )
        context_view = ContextPromptView.from_frame(current_context)
        value_view = self.value_system.prompt_view(current_context.context_id)
        context_text = self.session_state.context_text()
        temporal_update = emotion_engine.advance_to()
        measurement = self.surprisal_calculator.measure(
            context_text,
            user_input,
            model_key=self.primary_model_key,
            calibration=self.loss_calibration,
        )
        appraisal = self.appraiser.appraise(measurement, AppraisalSignals())
        emotion_update = emotion_engine.update_from_appraisal(
            appraisal,
            primary_loss=measurement.raw_loss if measurement.valid else None,
        )
        emotion_state = emotion_update.state
        memory_context = self.memory_system.retrieve_context(user_input)
        working_memory.advance()
        candidates = [
            (WorkingMemorySourceKind.EPISODIC, record.id, rank)
            for rank, record in enumerate(memory_context.db1_results)
        ] + [
            (WorkingMemorySourceKind.SEMANTIC, record.id, rank)
            for rank, record in enumerate(memory_context.db2_results)
        ]
        # Admit larger rank numbers first (lower retrieval priority). Equal-rank
        # references use ascending source kind and source ID as the total tie-break.
        for source_kind, source_id, rank in sorted(
            candidates,
            key=lambda candidate: (-candidate[2], candidate[0].value, candidate[1]),
        ):
            working_memory.admit(
                source_kind,
                source_id,
                activation=1.0,
                salience=1.0 / (rank + 1),
            )
        working_memory_view = working_memory.select_contextual(
            self.working_memory_resolver,
            context_registry,
            current_context.context_id,
        )
        attention_payload = (
            self._refresh_attention_prompt(
                event,
                current_context,
                emotion_state,
                working_memory_view,
                working_memory,
            )
            if event is not None
            else None
        )
        prompt = self._build_prompt(
            user_input,
            emotion_state,
            working_memory_view,
            context_view,
            value_view,
            attention_payload=attention_payload,
        )
        raw_response = self.agent.generate(prompt)
        processed_response = self.postprocessor.process(raw_response)
        memory_participant = MemoryEpisodicParticipant(
            self.memory_system,
            EpisodicWrite(
                user_input=user_input,
                response=processed_response.visible_response,
                loss=measurement.raw_loss,
                emotion_valence=emotion_state.valence,
                emotion_arousal=emotion_state.arousal,
                record_type=MemoryRecordType.EPISODIC_LOG,
                created_at=datetime.now(UTC).isoformat(),
                context_id=provenance[0],
                source_channel=provenance[1],
                source_session_id=provenance[2],
                schema_version=3,
            ),
        )
        experience_participant: MemoryExperienceParticipant | None = None
        if not capture_debug and event is not None:
            event_sequence = event.processing_sequence
            if event_sequence is None:  # pragma: no cover - validated above
                raise RuntimeError("chat event has no processing sequence")
            transaction_id = TransactionCoordinator.derive_transaction_id(
                event, TransactionKind.EVENT_MUTATION
            )
            measurement_evidence = ExperienceMeasurementEvidence.from_measurement(
                measurement
            )
            appraisal_evidence = ExperienceAppraisalEvidence.from_result(appraisal)
            pre_appraisal_emotion = ExperienceEmotionProjection.from_state(
                temporal_update.state
            )
            post_appraisal_emotion = ExperienceEmotionProjection.from_state(
                emotion_update.state
            )
            emotion_contributions = ExperienceEmotionContributions.from_update(
                emotion_update
            )
            temporal_reasons = tuple(
                ExperienceEmotionUpdateReasonCode(reason.value)
                for reason in temporal_update.reasons
            )
            emotion_reasons = tuple(
                ExperienceEmotionUpdateReasonCode(reason.value)
                for reason in emotion_update.reasons
            )
            experience_record = ExperienceRecord(
                experience_id=experience_id_for_event(
                    event.event_id, event_sequence
                ),
                revision=0,
                lifecycle=ExperienceLifecycle.ACTIVE,
                source_event_id=event.event_id,
                source_event_sequence=event_sequence,
                source_episode_id=episodic_episode_id(
                    transaction_id,
                    memory_participant.participant_id,
                    memory_participant.operation_digest,
                ),
                context_id=provenance[0],
                measurement=measurement_evidence,
                appraisal=appraisal_evidence,
                pre_appraisal_emotion=pre_appraisal_emotion,
                temporal_update_reasons=temporal_reasons,
                post_appraisal_emotion=post_appraisal_emotion,
                emotion_contributions=emotion_contributions,
                emotion_update_reasons=emotion_reasons,
                subjective_salience=calculate_subjective_salience(
                    measurement_evidence,
                    pre_appraisal_emotion,
                    post_appraisal_emotion,
                ),
                created_at=event.requested_at,
                revision_history=(
                    ExperienceRevisionRecord(
                        experience_id=experience_id_for_event(
                            event.event_id, event_sequence
                        ),
                        revision=0,
                        operation=ExperienceRevisionOperation.CREATE,
                        reason=ExperienceRevisionReason.CREATION,
                        created_at=event.requested_at,
                        event_id=event.event_id,
                        event_sequence=event_sequence,
                        evidence_refs=(event.event_id,),
                    ),
                ),
            )
            experience_participant = MemoryExperienceParticipant(
                self.memory_system,
                self.experience_store,
                ExperienceCreateIntent(
                    experience_record,
                    memory_participant.operation_digest,
                ),
            )
        session_participant = SessionTurnParticipant(
            self.session_state,
            SessionTurnOperation(
                user_input=user_input,
                response=processed_response.visible_response,
            ),
        )
        trace = None
        diagnostics = ChatDiagnostics(
            measurement=measurement,
            appraisal=appraisal,
            temporal_update=temporal_update,
            emotion_update=emotion_update,
        )
        if capture_debug:
            trace = DebugChatTrace(
                hidden_thought=processed_response.hidden_thought,
                prompt=prompt,
                memory_context=memory_context,
                working_memory_view=working_memory_view,
                diagnostics=diagnostics,
            )
        return _ComputedChat(
            response=processed_response.visible_response,
            loss=measurement.raw_loss,
            valence=emotion_state.valence,
            arousal=emotion_state.arousal,
            optimal_loss=emotion_state.optimal_loss,
            trace=trace,
            diagnostics=diagnostics,
            memory_participant=memory_participant,
            experience_participant=experience_participant,
            session_participant=session_participant,
        )

    def _refresh_attention_prompt(
        self,
        event: AgentEvent,
        current_context: ContextFrame,
        emotion_state: EmotionState,
        working_memory_view: WorkingMemoryView,
        working_memory: WorkingMemory,
    ) -> AttentionPromptPayload:
        """Refresh Attention once from exact sources inside this active chat."""

        runtime = self._runtime
        if not isinstance(runtime, AgentRuntime):
            raise RuntimeError(
                "Attention refresh requires the exact bound handler event"
            )
        if runtime.current_event() is not event:
            raise RuntimeError(
                "Attention refresh requires the exact bound handler event"
            )
        if type(event) is not AgentEvent or (
            type(event.processing_sequence) is not int
            or event.processing_sequence <= 0
        ):
            raise RuntimeError("Attention refresh requires a validated runtime event")
        if type(current_context) is not ContextFrame:
            raise TypeError("Attention refresh requires the resolved Context frame")

        context_state = self.context_registry.state
        if type(context_state) is not ContextRegistryState:
            raise RuntimeError("Context authority returned an invalid state")
        validate_context_registry_state(context_state)
        registered_context = next(
            (
                frame
                for frame in context_state.frames
                if frame.context_id == context_state.current_context_id
            ),
            None,
        )
        if (
            context_state.current_context_id != current_context.context_id
            or registered_context != current_context
            or registered_context is None
            or registered_context.status is not ContextStatus.ACTIVE
            or self.context_registry.current_context != registered_context
        ):
            raise RuntimeError(
                "Attention refresh Context does not match the current R09 authority"
            )

        attention_event = AttentionEvent(
            event.event_id,
            event.processing_sequence,
            event.requested_at,
        )
        working_memory_revision = working_memory.revision
        working_memory_items = working_memory.items
        if (
            type(working_memory_items) is not tuple
            or working_memory_view.revision != working_memory_revision
            or working_memory.revision != working_memory_revision
            or working_memory.items != working_memory_items
        ):
            raise RuntimeError(
                "Attention refresh requires one coherent complete Working Memory view"
            )

        belief_snapshot = self._belief_system.snapshot()
        if type(belief_snapshot) is not BeliefSystemSnapshot:
            raise RuntimeError("Belief authority returned an invalid current snapshot")
        motivation_state = self._motivation_system.snapshot()
        goal_state = self._goal_system.snapshot()
        commitment_state = self._commitment_system.snapshot()
        global_emotion = project_global_emotion(
            emotion_state,
            event=attention_event,
        )
        projections = project_attention_candidates(
            working_memory_items=working_memory_items,
            working_memory_revision=working_memory_revision,
            working_memory_view=working_memory_view,
            motivation_snapshot=motivation_state,
            goal_snapshot=goal_state,
            commitment_snapshot=commitment_state,
            event=attention_event,
        )
        if (
            runtime.current_event() is not event
            or self.context_registry.state != context_state
            or working_memory.revision != working_memory_revision
            or working_memory.items != working_memory_items
        ):
            raise RuntimeError("Attention source capture changed before refresh")
        refresh = self._attention_system.refresh(
            projections,
            attention_event,
            global_emotion=global_emotion,
        )
        if type(refresh) is not AttentionRefreshResult:
            raise RuntimeError("Attention owner returned an invalid refresh result")
        return build_attention_prompt_payload(
            refresh,
            working_memory_items=working_memory_items,
            working_memory_revision=working_memory_revision,
            working_memory_view=working_memory_view,
            motivation_state=motivation_state,
            goal_state=goal_state,
            commitment_state=commitment_state,
        )

    def _build_prompt(
        self,
        user_input: str,
        emotion_state: EmotionState,
        working_memory_view: WorkingMemoryView,
        context_view: ContextPromptView,
        value_view: ValuePromptView,
        *,
        attention_payload: AttentionPromptPayload | None = None,
    ) -> str:
        """Pass prompt projections while retaining older injected builders."""

        build = cast(Callable[..., str], self.prompt_builder.build)
        if attention_payload is not None:
            if type(attention_payload) is not AttentionPromptPayload:
                raise TypeError(
                    "attention_payload must be an exact AttentionPromptPayload"
                )
            if type(self.prompt_builder) is not PromptBuilder:
                raise TypeError(
                    "Attention prompt payload requires the exact PromptBuilder"
                )
            return build(
                user_input,
                emotion_state,
                working_memory_view,
                context_view=context_view,
                value_view=value_view,
                attention_payload=attention_payload,
            )
        parameters = tuple(inspect.signature(build).parameters.values())
        keyword_names = {
            parameter.name
            for parameter in parameters
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        keyword_arguments: dict[str, object] = {}
        r13_view = self.r13_view()
        if accepts_kwargs or "context_view" in keyword_names:
            keyword_arguments["context_view"] = context_view
        if accepts_kwargs or "value_view" in keyword_names:
            keyword_arguments["value_view"] = value_view
        if accepts_kwargs or "r13_view" in keyword_names:
            keyword_arguments["r13_view"] = r13_view
        return build(
            user_input,
            emotion_state,
            working_memory_view,
            **keyword_arguments,
        )

    def _chat_result(
        self, computed: _ComputedChat, transaction_id: str
    ) -> ChatResult:
        return ChatResult(
            episode_id=computed.memory_participant.episode_id(transaction_id),
            response=computed.response,
            loss=computed.loss,
            valence=computed.valence,
            arousal=computed.arousal,
            optimal_loss=computed.optimal_loss,
            model_id=self.settings.model.primary_id,
            adapter_id=self.adapter_id,
        )

    @staticmethod
    def _participants(computed: _ComputedChat) -> tuple[TransactionParticipant, ...]:
        if computed.experience_participant is None:
            return (computed.memory_participant, computed.session_participant)
        return (
            computed.memory_participant,
            computed.experience_participant,
            computed.session_participant,
        )
