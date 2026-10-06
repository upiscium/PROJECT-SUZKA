"""Deterministic, bounded, reference-only Working Memory authority."""

from __future__ import annotations

import math
from collections.abc import Iterable
from threading import RLock
from typing import TYPE_CHECKING

from suzka.context_contracts import ContextRelation
from suzka.limits import MAX_PERSISTED_REVISION
from suzka.working_memory_contracts import (
    MAX_ITEM_CAPACITY,
    MAX_PROJECTION_BYTES,
    MAX_SOURCE_ID_BYTES as MAX_SOURCE_ID_BYTES,
    WorkingMemoryAdmission,
    WorkingMemoryAdmissionReason,
    WorkingMemoryContextProjection,
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemoryResolver,
    WorkingMemoryResolverResult as WorkingMemoryResolverResult,
    WorkingMemoryRetentionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
    _ITEM_ID_DOMAIN as _ITEM_ID_DOMAIN,
    _SOURCE_ID as _SOURCE_ID,
    _validate_source,
    working_memory_item_id,
)

if TYPE_CHECKING:
    from suzka.runtime.context import ContextRegistry


REACTIVATION_BONUS = 0.2


class WorkingMemory:
    """Finite authoritative membership with an explicit monotonic revision."""

    def __init__(self, item_capacity: int, projection_max_bytes: int) -> None:
        self._item_capacity = _bounded_int(
            item_capacity, "item_capacity", MAX_ITEM_CAPACITY
        )
        self._projection_max_bytes = _bounded_int(
            projection_max_bytes, "projection_max_bytes", MAX_PROJECTION_BYTES
        )
        self._revision = 0
        self._items: dict[str, WorkingMemoryItem] = {}
        self._lock = RLock()
        self._selecting = False

    @property
    def item_capacity(self) -> int:
        return self._item_capacity

    @property
    def projection_max_bytes(self) -> int:
        return self._projection_max_bytes

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def items(self) -> tuple[WorkingMemoryItem, ...]:
        """Return deterministic immutable authoritative state."""

        with self._lock:
            return tuple(self._items[item_id] for item_id in sorted(self._items))

    def restore_exact(
        self, revision: int, items: Iterable[WorkingMemoryItem]
    ) -> None:
        """Restore canonical membership exactly without replaying mutations."""

        with self._lock:
            self._require_mutation_available()
            # Validate and publish under one lock: a concurrent admission must
            # not be able to slip between validation and exact replacement.
            revision_value = _nonnegative_revision(revision, "revision")
            restored: dict[str, WorkingMemoryItem] = {}
            source_references: set[tuple[WorkingMemorySourceKind, str]] = set()
            for item in items:
                if not isinstance(item, WorkingMemoryItem):
                    raise ValueError("Working Memory item is invalid")
                _validate_source(item.source_kind, item.source_id)
                if item.item_id != working_memory_item_id(
                    item.source_kind, item.source_id
                ):
                    raise ValueError("Working Memory item identity is invalid")
                _strict_unit_float(item.activation, "activation")
                _strict_unit_float(item.salience, "salience")
                if type(item.retention_reason) is not WorkingMemoryRetentionReason:
                    raise ValueError("Working Memory retention reason is invalid")
                created_revision = _nonnegative_revision(
                    item.created_revision, "created_revision"
                )
                last_activated_revision = _nonnegative_revision(
                    item.last_activated_revision, "last_activated_revision"
                )
                if (
                    created_revision > revision_value
                    or last_activated_revision > revision_value
                ):
                    raise ValueError("Working Memory item revision is invalid")
                source_reference = (item.source_kind, item.source_id)
                if item.item_id in restored or source_reference in source_references:
                    raise ValueError("Working Memory item is duplicated")
                restored[item.item_id] = item
                source_references.add(source_reference)
            if len(restored) > MAX_ITEM_CAPACITY:
                raise ValueError("Working Memory state exceeds maximum capacity")
            if len(restored) > self._item_capacity:
                raise ValueError("Working Memory state exceeds configured capacity")
            self._items = restored
            self._revision = revision_value

    @staticmethod
    def score(item: WorkingMemoryItem) -> float:
        """Return the minimal R08 score: 0.6A + 0.4S + retention bonus."""

        retention_bonus = (
            0.1
            if item.retention_reason is WorkingMemoryRetentionReason.REACTIVATED
            else 0.0
        )
        return 0.6 * item.activation + 0.4 * item.salience + retention_bonus

    def admit(
        self,
        source_kind: WorkingMemorySourceKind,
        source_id: str,
        activation: float,
        salience: float,
    ) -> WorkingMemoryAdmission:
        """Admit or reactivate one source reference and enforce hard capacity."""

        activation_value = _unit_float(activation, "activation")
        salience_value = _unit_float(salience, "salience")
        item_id = working_memory_item_id(source_kind, source_id)
        with self._lock:
            self._require_mutation_available()
            self._require_revision_available()
            self._revision += 1
            existing = self._items.get(item_id)
            if existing is None:
                item = WorkingMemoryItem(
                    item_id=item_id,
                    source_kind=source_kind,
                    source_id=source_id,
                    activation=activation_value,
                    salience=salience_value,
                    retention_reason=WorkingMemoryRetentionReason.RECENT,
                    created_revision=self._revision,
                    last_activated_revision=self._revision,
                )
                reason = WorkingMemoryAdmissionReason.ADMITTED
            else:
                item = WorkingMemoryItem(
                    item_id=existing.item_id,
                    source_kind=existing.source_kind,
                    source_id=existing.source_id,
                    activation=min(
                        1.0,
                        max(existing.activation, activation_value)
                        + REACTIVATION_BONUS,
                    ),
                    salience=max(existing.salience, salience_value),
                    retention_reason=WorkingMemoryRetentionReason.REACTIVATED,
                    created_revision=existing.created_revision,
                    last_activated_revision=self._revision,
                )
                reason = WorkingMemoryAdmissionReason.REACTIVATED
            self._items[item_id] = item
            evicted_item_id: str | None = None
            while len(self._items) > self._item_capacity:
                evicted = min(self._items.values(), key=self._rank_key)
                evicted_item_id = evicted.item_id
                del self._items[evicted.item_id]
            retained = item_id in self._items
            if not retained:
                reason = WorkingMemoryAdmissionReason.CAPACITY_EVICTED
            return WorkingMemoryAdmission(
                item, retained, reason, evicted_item_id
            )

    def advance(
        self, decay: float = 0.15, forget_below: float = 0.1
    ) -> tuple[WorkingMemoryItem, ...]:
        """Apply explicit activation decay and forget weak membership."""

        decay_value = _unit_float(decay, "decay")
        threshold = _unit_float(forget_below, "forget_below")
        with self._lock:
            self._require_mutation_available()
            updated: dict[str, WorkingMemoryItem] = {}
            for item_id, item in self._items.items():
                activation = max(0.0, item.activation - decay_value)
                if activation < threshold:
                    continue
                updated[item_id] = (
                    item
                    if activation == item.activation
                    else WorkingMemoryItem(
                        item_id=item.item_id,
                        source_kind=item.source_kind,
                        source_id=item.source_id,
                        activation=activation,
                        salience=item.salience,
                        retention_reason=item.retention_reason,
                        created_revision=item.created_revision,
                        last_activated_revision=item.last_activated_revision,
                    )
                )
            if updated != self._items:
                self._require_revision_available()
                self._items = updated
                self._revision += 1
            return tuple(self._items[item_id] for item_id in sorted(self._items))

    def forget(self, item_id: str) -> bool:
        """Remove membership idempotently; missing identity is a no-op."""

        if not isinstance(item_id, str):
            raise TypeError("item_id must be a string")
        with self._lock:
            self._require_mutation_available()
            if item_id not in self._items:
                return False
            self._require_revision_available()
            del self._items[item_id]
            self._revision += 1
            return True

    def select(self, resolver: WorkingMemoryResolver) -> WorkingMemoryView:
        """Resolve and pack an immutable byte-bounded view without mutation."""

        if not callable(resolver):
            raise TypeError("resolver must be callable")
        with self._lock:
            if self._selecting:
                raise RuntimeError("Working Memory selection is already active")
            self._selecting = True
            ranked = tuple(
                sorted(self._items.values(), key=self._rank_key, reverse=True)
            )
            revision = self._revision
            item_capacity = self._item_capacity
            projection_max_bytes = self._projection_max_bytes
        try:
            selected: list[WorkingMemorySelection] = []
            decisions: list[WorkingMemoryDecision] = []
            projected_bytes = 0
            for item in ranked:
                score = self.score(item)
                reason: WorkingMemoryDecisionReason
                rendered: str | None = None
                source_context_id: str | None = None
                context_projection: WorkingMemoryContextProjection | None = None
                try:
                    resolved = resolver(item)
                    if resolved is None:
                        reason = WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE
                    elif isinstance(resolved, str):
                        rendered = resolved
                    elif not isinstance(resolved, WorkingMemoryResolution):
                        reason = WorkingMemoryDecisionReason.RESOLVER_FAILURE
                    else:
                        resolution_reasons = {
                            WorkingMemoryResolutionStatus.MISSING: (
                                WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE
                            ),
                            WorkingMemoryResolutionStatus.ARCHIVED: (
                                WorkingMemoryDecisionReason.SOURCE_ARCHIVED
                            ),
                            WorkingMemoryResolutionStatus.UNAVAILABLE: (
                                WorkingMemoryDecisionReason.SOURCE_UNAVAILABLE
                            ),
                            WorkingMemoryResolutionStatus.MALFORMED: (
                                WorkingMemoryDecisionReason.SOURCE_MALFORMED
                            ),
                        }
                        if (
                            resolved.status
                            is not WorkingMemoryResolutionStatus.RESOLVED
                        ):
                            reason = resolution_reasons[resolved.status]
                        else:
                            rendered = resolved.rendered_content
                            source_context_id = resolved.source_context_id
                            context_projection = resolved.context_projection
                    if rendered is not None:
                        resolved_bytes = rendered.encode("utf-8")
                        if projected_bytes + len(resolved_bytes) > projection_max_bytes:
                            reason = WorkingMemoryDecisionReason.PROJECTION_BUDGET
                            rendered = None
                        else:
                            reason = WorkingMemoryDecisionReason.SELECTED
                            projected_bytes += len(resolved_bytes)
                except Exception:
                    reason = WorkingMemoryDecisionReason.RESOLVER_FAILURE
                is_selected = reason is WorkingMemoryDecisionReason.SELECTED
                decisions.append(
                    WorkingMemoryDecision(
                        item.item_id,
                        item.source_kind,
                        item.source_id,
                        is_selected,
                        score,
                        reason,
                        context_projection=context_projection,
                    )
                )
                if is_selected:
                    assert rendered is not None
                    selected.append(
                        WorkingMemorySelection(
                            item.item_id,
                            item.source_kind,
                            item.source_id,
                            rendered,
                            score,
                            reason,
                            source_context_id,
                            context_projection=context_projection,
                        )
                    )
            return WorkingMemoryView(
                tuple(selected),
                tuple(decisions),
                projected_bytes,
                item_capacity,
                projection_max_bytes,
                revision,
            )
        finally:
            with self._lock:
                self._selecting = False

    def select_contextual(
        self,
        resolver: WorkingMemoryResolver,
        context_registry: ContextRegistry,
        current_context_id: str,
    ) -> WorkingMemoryView:
        """Select a pure view ranked by Context compatibility-adjusted score."""

        if not callable(resolver):
            raise TypeError("resolver must be callable")
        with self._lock:
            if self._selecting:
                raise RuntimeError("Working Memory selection is already active")
            self._selecting = True
            items = tuple(self._items.values())
            item_capacity = self._item_capacity
            projection_max_bytes = self._projection_max_bytes
            revision = self._revision

        contextual_resolve = getattr(resolver, "resolve_contextual", None)
        try:
            candidates: list[
                tuple[
                    WorkingMemoryItem,
                    float,
                    str | None,
                    WorkingMemoryDecisionReason,
                    str | None,
                    ContextRelation | None,
                    float | None,
                    float | None,
                    WorkingMemoryContextProjection | None,
                ]
            ] = []
            for item in items:
                base_score = self.score(item)
                rendered: str | None = None
                source_context_id: str | None = None
                relation: ContextRelation | None = None
                compatibility_score: float | None = None
                effective_score: float | None = None
                context_projection: WorkingMemoryContextProjection | None = None
                try:
                    resolved = (
                        contextual_resolve(item, context_registry, current_context_id)
                        if callable(contextual_resolve)
                        else resolver(item)
                    )
                    if resolved is None:
                        reason = WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE
                    elif isinstance(resolved, str):
                        rendered = resolved
                        reason = WorkingMemoryDecisionReason.SELECTED
                    elif not isinstance(resolved, WorkingMemoryResolution):
                        reason = WorkingMemoryDecisionReason.RESOLVER_FAILURE
                    else:
                        resolution_reasons = {
                            WorkingMemoryResolutionStatus.MISSING: (
                                WorkingMemoryDecisionReason.UNRESOLVED_REFERENCE
                            ),
                            WorkingMemoryResolutionStatus.ARCHIVED: (
                                WorkingMemoryDecisionReason.SOURCE_ARCHIVED
                            ),
                            WorkingMemoryResolutionStatus.UNAVAILABLE: (
                                WorkingMemoryDecisionReason.SOURCE_UNAVAILABLE
                            ),
                            WorkingMemoryResolutionStatus.MALFORMED: (
                                WorkingMemoryDecisionReason.SOURCE_MALFORMED
                            ),
                        }
                        if (
                            resolved.status
                            is not WorkingMemoryResolutionStatus.RESOLVED
                        ):
                            reason = resolution_reasons[resolved.status]
                        else:
                            rendered = resolved.rendered_content
                            source_context_id = resolved.source_context_id
                            context_projection = resolved.context_projection
                            reason = WorkingMemoryDecisionReason.SELECTED
                except Exception:
                    reason = WorkingMemoryDecisionReason.RESOLVER_FAILURE

                if rendered is not None:
                    if context_projection is not None:
                        compatibility_score = (
                            context_projection.aggregate_compatibility
                        )
                        if source_context_id is not None:
                            compatibility = context_registry.compatibility(
                                source_context_id, current_context_id
                            )
                            relation = compatibility.relation
                    else:
                        compatibility = context_registry.compatibility(
                            source_context_id, current_context_id
                        )
                        relation = compatibility.relation
                        compatibility_score = compatibility.score
                    effective_score = base_score * compatibility_score
                candidates.append(
                    (
                        item,
                        base_score,
                        rendered,
                        reason,
                        source_context_id,
                        relation,
                        compatibility_score,
                        effective_score,
                        context_projection,
                    )
                )

            candidates.sort(
                key=lambda candidate: (
                    candidate[7] if candidate[7] is not None else float("-inf"),
                    candidate[1],
                    candidate[0].activation,
                    candidate[0].salience,
                    candidate[0].last_activated_revision,
                    candidate[0].created_revision,
                    candidate[0].item_id,
                ),
                reverse=True,
            )
            selected: list[WorkingMemorySelection] = []
            decisions: list[WorkingMemoryDecision] = []
            projected_bytes = 0
            for (
                item,
                base_score,
                rendered,
                reason,
                source_context_id,
                relation,
                compatibility_score,
                effective_score,
                context_projection,
            ) in candidates:
                if rendered is not None:
                    rendered_bytes = len(rendered.encode("utf-8"))
                    if projected_bytes + rendered_bytes > projection_max_bytes:
                        reason = WorkingMemoryDecisionReason.PROJECTION_BUDGET
                        rendered = None
                    else:
                        projected_bytes += rendered_bytes
                is_selected = reason is WorkingMemoryDecisionReason.SELECTED
                decisions.append(
                    WorkingMemoryDecision(
                        item_id=item.item_id,
                        source_kind=item.source_kind,
                        source_id=item.source_id,
                        selected=is_selected,
                        score=base_score,
                        reason=reason,
                        context_relation=relation,
                        context_compatibility=compatibility_score,
                        effective_score=effective_score,
                        context_projection=context_projection,
                    )
                )
                if is_selected:
                    assert rendered is not None
                    selected.append(
                        WorkingMemorySelection(
                            item_id=item.item_id,
                            source_kind=item.source_kind,
                            source_id=item.source_id,
                            rendered_content=rendered,
                            score=base_score,
                            reason=reason,
                            source_context_id=source_context_id,
                            context_relation=relation,
                            context_compatibility=compatibility_score,
                            effective_score=effective_score,
                            context_projection=context_projection,
                        )
                    )
            return WorkingMemoryView(
                tuple(selected),
                tuple(decisions),
                projected_bytes,
                item_capacity,
                projection_max_bytes,
                revision,
            )
        finally:
            with self._lock:
                self._selecting = False

    def _require_mutation_available(self) -> None:
        if self._selecting:
            raise RuntimeError("Working Memory cannot mutate during selection")

    def _require_revision_available(self) -> None:
        if self._revision >= MAX_PERSISTED_REVISION:
            raise ValueError("Working Memory revision capacity is exhausted")

    def _rank_key(
        self, item: WorkingMemoryItem
    ) -> tuple[float, float, float, int, int, str]:
        return (
            self.score(item),
            item.activation,
            item.salience,
            item.last_activated_revision,
            item.created_revision,
            item.item_id,
        )


def _bounded_int(value: int, name: str, maximum: int) -> int:
    if (
        type(value) is not int
        or not 0 < value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _unit_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _strict_unit_float(value: object, name: str) -> float:
    """Validate the already-decoded float representation of durable state."""

    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _nonnegative_revision(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if value > MAX_PERSISTED_REVISION:
        raise ValueError(f"{name} exceeds the persisted revision bound")
    return value
