"""FastAPI startup foundation for PROJECT-SUZKA."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    request_validation_exception_handler as default_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from suzka.api.routes import adapters, chat, contexts, debug, memory, sleep, values
from suzka.api.context_read_model import (
    CommittedContextReadModel,
    PublishedContextView,
)
from suzka.config import Settings, get_settings
from suzka.identity import ValueConflictDefinition
from suzka.learning import AdapterRegistry, SleepCycleManager
from suzka.memory import DualMemorySystem, ExperienceStore
from suzka.models import load_model_provider
from suzka.r15.qualified_proposal import (
    QualifiedNarrativeSourceGate,
    QualifiedProposalUnavailable,
)
from suzka.runtime import (
    AgentEvent,
    AgentRuntime,
    AgentRuntimeStatus,
    AgentStateStore,
    AgentStateSnapshotV9,
    CompatibleAgentStateSnapshot,
    EventJournal,
    EventJournalError,
    EventJournalLease,
    EmotionTimer,
    InternalCommitEvidence,
    SuzkaMainLoop,
    StateRecoveryCoordinator,
    StateRecoveryError,
    StateRecoveryResult,
    StateWAL,
    TransactionCoordinator,
    WorkingMemory,
)
from suzka.runtime.agent_state import (
    _domain_belief_system,
    _domain_value_system,
)
from suzka.runtime.startup_reconciliation import StartupReconciliationCoordinator
from suzka.runtime.semantic_receipt_retention import (
    SemanticReceiptRetentionCoordinator,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application from typed settings."""

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        existing_journal = getattr(app.state, "event_journal", None)
        journal_lease: EventJournalLease | None = None
        initial_committed_read_views: object | None = None
        committed_read_view_publisher: Callable[[object], None] | None = None
        try:
            if existing_journal is None:
                journal_lease = EventJournalLease(app_settings.event_journal.path)
                app.state.event_journal = EventJournal(
                    app_settings.event_journal.path,
                    app_settings.event_journal.max_bytes,
                    app_settings.event_journal.retained_files,
                    lease=journal_lease,
                )
            else:
                if not existing_journal.has_exclusive_authority:
                    raise RuntimeError(
                        "Injected EventJournal has no exclusive authority"
                    )
                app.state.event_journal = existing_journal

            configured_value_seeds = tuple(
                seed.to_declaration() for seed in app_settings.values.seeds
            )
            configured_value_conflicts = tuple(
                ValueConflictDefinition(
                    left_value_id=conflict.left_value_id,
                    right_value_id=conflict.right_value_id,
                )
                for conflict in app_settings.values.conflicts
            )
            app.state.agent_state_store = getattr(
                app.state, "agent_state_store", None
            ) or AgentStateStore(
                app_settings.agent_state.path,
                app_settings.emotion.baseline_surprisal,
            )
            app.state.agent_state_store.configure_value_contract(
                configured_value_seeds, configured_value_conflicts
            )
            app.state.state_wal = getattr(app.state, "state_wal", None) or StateWAL(
                app_settings.state_wal.directory
            )
            app.state.state_recovery = StateRecoveryCoordinator(
                app.state.agent_state_store,
                app.state.event_journal,
                app.state.state_wal,
            )
            journal_schema = app.state.event_journal.inspect().schema_version
            app.state.memory_system = getattr(
                app.state, "memory_system", None
            ) or DualMemorySystem(app_settings)
            app.state.semantic_store = app.state.memory_system.semantic_store
            app.state.semantic_receipt_retention = SemanticReceiptRetentionCoordinator(
                app.state.event_journal, app.state.semantic_store
            )
            injected_main_loop = getattr(app.state, "main_loop", None)
            if injected_main_loop is not None and not isinstance(
                injected_main_loop, SuzkaMainLoop
            ):
                raise RuntimeError("Injected main loop has an invalid type")
            injected_experience_store = getattr(app.state, "experience_store", None)
            if injected_main_loop is not None:
                loop_experience_store = injected_main_loop.experience_store
                if injected_experience_store is None:
                    injected_experience_store = loop_experience_store
                elif injected_experience_store is not loop_experience_store:
                    raise RuntimeError(
                        "Injected main loop and Experience store do not match"
                    )
            if injected_experience_store is None:
                injected_experience_store = ExperienceStore.from_memory_root(
                    app.state.memory_system.settings.memory.persist_directory
                )
            elif not isinstance(injected_experience_store, ExperienceStore):
                raise RuntimeError("Injected Experience store has an invalid type")
            app.state.experience_store = injected_experience_store
            app.state.startup_reconciliation = StartupReconciliationCoordinator(
                app.state.event_journal,
                app.state.state_recovery,
                app.state.memory_system,
                app.state.experience_store,
                app.state.semantic_store,
                semantic_checkpoint_covers=(
                    app.state.semantic_receipt_retention.checkpoint_covers
                ),
            )
            app.state.startup_reconciliation.resume_prepared_gate_clear()
            participants_consistent = True
            degraded_reason = None
            if journal_schema == 3:
                participants_consistent, degraded_reason = (
                    app.state.startup_reconciliation.reconcile_open_transactions()
                )
            recovery: StateRecoveryResult | None = None
            if participants_consistent:
                recovery = app.state.state_recovery.prepare_startup()
                journal_schema = app.state.event_journal.inspect().schema_version
                if journal_schema == 2:
                    app.state.event_journal.append_v3_migration_checkpoint()
                    journal_schema = app.state.event_journal.inspect().schema_version
                if journal_schema != 3:
                    raise StateRecoveryError("Runtime requires EventJournal schema 3")
                if recovery.external_reconciliation_required:
                    reconciliation = (
                        app.state.startup_reconciliation.reconcile_recovery_gate(
                            recovery
                        )
                    )
                    recovery = reconciliation.recovery
                    participants_consistent = reconciliation.participants_consistent
                    degraded_reason = reconciliation.degraded_reason
                else:
                    app.state.startup_reconciliation.ensure_adoption_baseline(recovery)
                if participants_consistent:
                    participants_consistent, degraded_reason = (
                        app.state.startup_reconciliation.reconcile_terminal_semantic_projections()
                    )
                if participants_consistent:
                    try:
                        app.state.semantic_receipt_retention.after_terminal_completion()
                    except Exception:
                        participants_consistent = False
                        degraded_reason = "semantic_receipt_retention_unavailable"
                startup_state = recovery
            else:
                if journal_schema != 3:
                    raise StateRecoveryError("Degraded startup requires schema 3")
                startup_state = app.state.state_recovery.inspect_degraded_startup()
            app.state.external_reconciliation_required = (
                recovery.external_reconciliation_required if recovery else False
            ) or not participants_consistent
            app.state.reconciliation_degraded_reason = degraded_reason
            retention_status = app.state.event_journal.admission_status()
            app.state.startup_retention_admission_available = retention_status.available
            app.state.startup_retention_reason = retention_status.reason
            snapshot = startup_state.snapshot
            snapshot_hash = startup_state.snapshot_hash

            app.state.model_provider = getattr(
                app.state, "model_provider", None
            ) or load_model_provider(app_settings)
            app.state.adapter_registry = getattr(
                app.state, "adapter_registry", None
            ) or AdapterRegistry(app_settings)
            injected_working_memory = getattr(app.state, "working_memory", None)
            app.state.working_memory = (
                injected_working_memory
                if injected_working_memory is not None
                else WorkingMemory(
                    item_capacity=app_settings.working_memory.item_capacity,
                    projection_max_bytes=app_settings.working_memory.projection_max_bytes,
                )
            )
            app.state.main_loop = getattr(
                app.state, "main_loop", None
            ) or SuzkaMainLoop(
                app_settings,
                app.state.model_provider,
                app.state.memory_system,
                working_memory=app.state.working_memory,
                experience_store=app.state.experience_store,
            )
            if app.state.main_loop.experience_store is not app.state.experience_store:
                raise RuntimeError(
                    "Main loop and Experience store do not match after startup"
                )
            app.state.working_memory = app.state.main_loop.working_memory
            # U5 technical port exists BEFORE v10/v9 snapshot restore.
            app.state.r15_state_port = app.state.main_loop._enable_r15_state_port()
            app.state.agent_state_store.restore_into(app.state.main_loop, snapshot)
            # Only a reconciled, externally terminal AgentState snapshot may
            # initialize public Context GETs. Never seed from a worker's live
            # (potentially speculative) registry at query time. No B capture
            # is installed by G1-P; this projection is NOT G2 source evidence.
            restored_context = app.state.main_loop.context_registry.state
            historical_context = getattr(snapshot, "context_state", None)
            if historical_context is None:
                # Older legal v1/v2 snapshots have no R09 Context root. A
                # clean, empty restored registry is the only public projection.
                if (
                    restored_context.revision != 0
                    or restored_context.frames
                    or restored_context.current_context_id is not None
                    or restored_context.interlocutor_bindings
                ):
                    raise StateRecoveryError("Legacy Context restoration is not empty")
            elif restored_context != historical_context.to_registry_state():
                raise StateRecoveryError("Context bootstrap does not match AgentState")
            initial_context_view = PublishedContextView(
                restored_context,
                snapshot.last_processed_event_sequence,
                snapshot_hash,
            )
            app.state.context_committed_reads = CommittedContextReadModel(
                initial_context_view,
                ready=not app.state.external_reconciliation_required,
            )
            if isinstance(app.state.main_loop, SuzkaMainLoop):
                checkpoint_read_views = getattr(
                    app.state.main_loop,
                    "_committed_read_view_checkpoint",
                    None,
                )
                if not callable(checkpoint_read_views):
                    raise StateRecoveryError(
                        "Main loop committed read-view checkpoint is unavailable"
                    )
                publish_read_views = getattr(
                    app.state.main_loop, "_publish_committed_read_views", None
                )
                if not callable(publish_read_views):
                    raise StateRecoveryError(
                        "Main loop committed read-view publisher is unavailable"
                    )
                committed_read_view_publisher = cast(
                    Callable[[object], None], publish_read_views
                )
                initial_committed_read_views = checkpoint_read_views()
                if initial_committed_read_views is None:
                    raise StateRecoveryError(
                        "Main loop committed read-view checkpoint is invalid"
                    )
            app.state.sleep_cycle_manager = getattr(
                app.state, "sleep_cycle_manager", None
            ) or SleepCycleManager(
                app_settings,
                app.state.memory_system,
                app.state.model_provider,
                app.state.adapter_registry,
            )
        except BaseException:
            if journal_lease is not None:
                journal_lease.close()
            journal = getattr(app.state, "event_journal", None)
            if journal is not None:
                journal.close()
            raise
        committed_snapshot: CompatibleAgentStateSnapshot = snapshot
        committed_snapshot_hash = snapshot_hash
        committed_context_view = initial_context_view
        committed_read_views = initial_committed_read_views
        app.state.transaction_coordinator = TransactionCoordinator(
            app.state.event_journal,
            app.state.state_recovery.verify_internal_commit,
            before_prepare=app.state.semantic_receipt_retention.before_prepare,
            after_participant_finalized=(
                app.state.semantic_receipt_retention.after_participant_finalized
            ),
        )

        def admission_checkpoint(event: AgentEvent) -> None:
            del event

        def pre_admission_guard(event: AgentEvent) -> bool:
            return app.state.event_journal.append_accepted_if_admission_available(event)

        def started_checkpoint(event: AgentEvent) -> None:
            # STARTED append errors occur before any other G1-P callback.
            # Fence the short Journal append against GET admission, so no
            # new READY read can linearize after the append has failed but
            # before AgentRuntime enters FAILED.
            app.state.context_committed_reads.append_started_checkpoint(
                lambda: app.state.event_journal.append_started(event)
            )

        def preparation_checkpoint(event: AgentEvent, value: object) -> object:
            try:
                return app.state.transaction_coordinator.prepare_result(event, value)
            except BaseException:
                # Runtime treats preparation uncertainty as fail-stop; the
                # last public terminal state must not claim healthy freshness.
                app.state.context_committed_reads.degrade()
                raise

        def _internal_commit_checkpoint(event: AgentEvent) -> InternalCommitEvidence:
            nonlocal committed_snapshot, committed_snapshot_hash
            nonlocal committed_read_views
            sequence = event.processing_sequence
            assert sequence is not None
            # Before the first WAL mutation, failure is definitively not a
            # committed R15 revision. Restore all domains including U3 now.
            # Once commit_internal_candidate begins, WAL publication may be
            # indeterminate: do NOT roll back a possibly committed revision.
            try:
                validate_belief_commit = getattr(
                    app.state.main_loop, "_validate_belief_event_commit", None
                )
                if callable(validate_belief_commit):
                    validate_belief_commit(event)
                app.state.main_loop._validate_value_event_commit(event)
                candidate = app.state.agent_state_store.capture(
                    app.state.main_loop, sequence
                )
                candidate_hash = app.state.agent_state_store.snapshot_hash(candidate)
                prepared_read_views: tuple[object, Callable[[object], None]] | None = (
                    None
                )
                if isinstance(app.state.main_loop, SuzkaMainLoop):
                    assert isinstance(candidate, AgentStateSnapshotV9)
                    value_snapshot = _domain_value_system(
                        candidate.value_state
                    ).snapshot()
                    belief_snapshot = _domain_belief_system(
                        candidate.belief_state
                    ).snapshot()
                    prepare_attention_read_views = getattr(
                        app.state.main_loop,
                        "_prepare_committed_attention_read_views",
                        None,
                    )
                    if not callable(prepare_attention_read_views):
                        raise StateRecoveryError(
                            "Committed Attention read-view hooks are unavailable"
                        )
                    publisher = committed_read_view_publisher
                    if publisher is None:
                        raise StateRecoveryError(
                            "Committed Attention read-view publisher is unavailable"
                        )
                    prepared_read_views = prepare_attention_read_views(
                        value_snapshot,
                        belief_snapshot,
                        candidate.r13_state,
                        candidate.attention_state,
                        retain_selection=True,
                    )
                    if prepared_read_views is None:
                        raise StateRecoveryError(
                            "Committed read views were not prepared"
                        )
                    prepared_read_views = (prepared_read_views, publisher)
            except Exception:
                app.state.agent_state_store.restore_into(
                    app.state.main_loop, committed_snapshot
                )
                raise
            evidence = app.state.state_recovery.commit_internal_candidate(
                event, committed_snapshot, candidate
            )
            if prepared_read_views is not None:
                read_view_bundle, publish_read_views = prepared_read_views
                publish_read_views(read_view_bundle)
                committed_read_views = read_view_bundle
            committed_snapshot = candidate
            committed_snapshot_hash = candidate_hash
            return evidence

        def internal_commit_checkpoint(event: AgentEvent) -> InternalCommitEvidence:
            try:
                return _internal_commit_checkpoint(event)
            except BaseException:
                # A failed or uncertain internal commit cannot leave a READY
                # public view that implies the runtime remains healthy.
                app.state.context_committed_reads.degrade()
                raise

        def finalization_checkpoint(event: AgentEvent, evidence: object) -> None:
            try:
                if not isinstance(evidence, InternalCommitEvidence):
                    raise StateRecoveryError("Internal commit evidence is unavailable")
                app.state.transaction_coordinator.finalize_event(event, evidence)
            except BaseException:
                app.state.context_committed_reads.degrade()
                raise

        def terminal_completion_checkpoint(event: AgentEvent, evidence: object) -> None:
            nonlocal committed_context_view
            try:
                if not isinstance(evidence, InternalCommitEvidence):
                    raise StateRecoveryError("Internal commit evidence is unavailable")
                sequence = event.processing_sequence
                if type(sequence) is not int or sequence <= 0:
                    raise StateRecoveryError("Context terminal sequence unavailable")
                if (
                    not isinstance(committed_snapshot, AgentStateSnapshotV9)
                    or evidence.snapshot_sequence != sequence
                    or evidence.snapshot_hash != committed_snapshot_hash
                    or committed_snapshot.last_processed_event_sequence != sequence
                ):
                    raise StateRecoveryError("Context terminal snapshot mismatch")
                # Stage the immutable result before any terminal publication.
                candidate = PublishedContextView(
                    committed_snapshot.context_state.to_registry_state(),
                    sequence,
                    committed_snapshot_hash,
                )
                public_context = app.state.context_committed_reads
                # From this linearization point until the full-view swap,
                # new GETs receive 503. A GET admitted earlier may finish.
                public_context.begin_terminal(
                    event_id=event.event_id,
                    sequence=sequence,
                    expected_previous=committed_context_view,
                )
                app.state.state_recovery.complete_committed_event(event, evidence)
                app.state.semantic_receipt_retention.after_terminal_completion()
                # This atomic swap is before AgentRuntime resolves the future.
                public_context.publish_terminal(
                    event_id=event.event_id,
                    sequence=sequence,
                    candidate=candidate,
                )
                committed_context_view = candidate
            except BaseException:
                app.state.context_committed_reads.degrade()
                raise

        def failure_checkpoint(event: AgentEvent) -> None:
            public_context = app.state.context_committed_reads
            prior = public_context.begin_handler_restore()
            try:
                app.state.agent_state_store.restore_into(
                    app.state.main_loop, committed_snapshot
                )
                if (
                    committed_read_view_publisher is not None
                    and committed_read_views is not None
                ):
                    committed_read_view_publisher(committed_read_views)
                app.state.event_journal.append_failed(
                    event,
                    committed_snapshot.last_processed_event_sequence,
                    committed_snapshot_hash,
                )
                if (
                    prior is not None
                    and app.state.main_loop.context_registry.state != prior.state
                ):
                    raise StateRecoveryError("Context failure restore diverged")
                # Only a normal pre-commit handler failure re-enables the
                # previous terminal view. A preparation/internal/finalization
                # error has already made the projection permanently DEGRADED.
                public_context.complete_handler_restore(prior)
            except BaseException:
                public_context.degrade()
                raise

        try:
            if getattr(app.state, "agent_runtime", None) is None:
                app.state.agent_runtime = AgentRuntime(
                    app_settings.runtime.queue_capacity,
                    initial_sequence=startup_state.processing_high_water,
                    pre_admission_guard=pre_admission_guard,
                    admission_checkpoint=admission_checkpoint,
                    started_checkpoint=started_checkpoint,
                    preparation_checkpoint=preparation_checkpoint,
                    internal_commit_checkpoint=internal_commit_checkpoint,
                    finalization_checkpoint=finalization_checkpoint,
                    terminal_completion_checkpoint=terminal_completion_checkpoint,
                    failure_checkpoint=failure_checkpoint,
                )
            else:
                app.state.agent_runtime.configure_durability(
                    initial_sequence=startup_state.processing_high_water,
                    pre_admission_guard=pre_admission_guard,
                    admission_checkpoint=admission_checkpoint,
                    started_checkpoint=started_checkpoint,
                    preparation_checkpoint=preparation_checkpoint,
                    internal_commit_checkpoint=internal_commit_checkpoint,
                    finalization_checkpoint=finalization_checkpoint,
                    terminal_completion_checkpoint=terminal_completion_checkpoint,
                    failure_checkpoint=failure_checkpoint,
                )
            app.state.main_loop.bind_runtime(app.state.agent_runtime)
            app.state.sleep_cycle_manager.bind_runtime(app.state.agent_runtime)
            try:
                narrative_reader = QualifiedNarrativeSourceGate(
                    app.state.agent_runtime,
                    app.state.event_journal,
                    app.state.experience_store,
                    app.state.memory_system,
                    app.state.main_loop.context_registry,
                )
            except QualifiedProposalUnavailable:
                # Older authorized injected MainLoops can use a distinct
                # Experience storage root. They retain their original
                # functionality, but NEVER gain positive R15 source/admission
                # authority from these mismatched owners. If this instance
                # has a restored subjective revision, refuse startup instead
                # of silently running with unqualified biography.
                if app.state.r15_state_port.narrative.snapshot().revision != 0:
                    raise StateRecoveryError(
                        "Persisted R15 narrative Source owners are inconsistent"
                    ) from None
            else:
                app.state.r15_state_port.narrative._bind_source_gate_for_state(
                    narrative_reader
                )
                # U6 uses the already-authoritative WAL/Runtime, no new owner.
                app.state.r15_state_port._bind_private_read_sources(
                    app.state.agent_state_store, app.state.agent_runtime
                )
                # U2b: the same Subject/Journal/Experience/DB1/Context/WAL
                # owners, with no new external Source or schema migration.
                if (
                    type(app.state.agent_state_store) is AgentStateStore
                    and type(app.state.event_journal) is EventJournal
                    and type(app.state.experience_store) is ExperienceStore
                    and type(app.state.memory_system) is DualMemorySystem
                    and type(app.state.state_wal) is StateWAL
                ):
                    app.state.r15_state_port._bind_private_relationship_sources(
                        app.state.agent_state_store,
                        app.state.agent_runtime,
                        app.state.event_journal,
                        app.state.experience_store,
                        app.state.memory_system,
                        app.state.state_wal,
                        app.state.main_loop.context_registry,
                    )
        except BaseException:
            app.state.event_journal.close()
            raise
        try:
            app.state.emotion_timer = getattr(
                app.state, "emotion_timer", None
            ) or EmotionTimer(
                app.state.agent_runtime,
                app_settings.emotion.timer_interval_seconds,
                app.state.main_loop.emotion_tick,
            )
        except BaseException:
            app.state.agent_runtime.shutdown()
            app.state.event_journal.close()
            raise
        if (
            not app.state.external_reconciliation_required
            and app.state.startup_retention_admission_available
        ):
            try:
                assert recovery is not None
                app.state.agent_runtime.start()
                app.state.state_recovery.publish_boot_anchor(recovery)
                if app_settings.emotion.timer_enabled:
                    app.state.emotion_timer.start()
            except BaseException:
                app.state.emotion_timer.stop()
                app.state.agent_runtime.shutdown()
                app.state.event_journal.close()
                raise
        try:
            yield
        finally:
            app.state.emotion_timer.stop()
            app.state.agent_runtime.shutdown()
            app.state.event_journal.close()

    app = FastAPI(title=app_settings.project.name, lifespan=lifespan)
    app.state.settings = app_settings

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        """Return bounded validation details without echoing request payloads."""

        is_context_path = request.url.path == "/api/contexts" or (
            request.url.path.startswith("/api/contexts/")
        )
        is_values_path = request.url.path == "/api/values" or (
            request.url.path.startswith("/api/values/")
        )
        if (
            request.url.path not in {"/api/chat", "/api/chat/debug"}
            and not is_context_path
            and not is_values_path
        ):
            return await default_validation_exception_handler(request, error)
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "type": str(item.get("type", "validation_error")),
                        "loc": list(item.get("loc", ())),
                        "msg": "Invalid request",
                    }
                    for item in error.errors()
                ]
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        if app.state.external_reconciliation_required:
            return {
                "status": "degraded",
                "project": app_settings.project.name,
                "reason": app.state.reconciliation_degraded_reason
                or "external_reconciliation_required",
            }
        try:
            retention_available = app.state.event_journal.admission_status().available
        except EventJournalError:
            retention_available = False
        if (
            not retention_available
            or not app.state.startup_retention_admission_available
        ):
            return {
                "status": "degraded",
                "project": app_settings.project.name,
                "reason": "event_journal_retention_exhausted",
            }
        if app.state.agent_runtime.status is not AgentRuntimeStatus.ACCEPTING:
            return {
                "status": "degraded",
                "project": app_settings.project.name,
                "reason": "agent_runtime_unavailable",
            }
        return {"status": "ok", "project": app_settings.project.name}

    app.include_router(chat.router)
    app.include_router(contexts.router)
    app.include_router(debug.router)
    app.include_router(memory.router)
    app.include_router(sleep.router)
    app.include_router(adapters.router)
    app.include_router(values.router)

    return app


# No module-level ASGI app: plain imports must not select ambient settings.


def main() -> None:
    """Run the development API server."""

    settings = get_settings()
    uvicorn.run(
        "suzka.api.dev_asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
