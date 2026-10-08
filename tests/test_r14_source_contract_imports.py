"""R14 U1 neutral source-contract relocation compatibility tests."""

from dataclasses import fields
import os
from pathlib import Path
import subprocess
import sys

import pytest

from suzka.context_contracts import ContextRelation
from suzka.emotion_contracts import EmotionState
from suzka.working_memory_contracts import (
    MAX_ITEM_CAPACITY,
    MAX_PROJECTION_BYTES,
    MAX_SOURCE_ID_BYTES,
    WorkingMemoryAdmission,
    WorkingMemoryAdmissionReason,
    WorkingMemoryContextProjection,
    WorkingMemoryDecision,
    WorkingMemoryDecisionReason,
    WorkingMemoryItem,
    WorkingMemoryResolution,
    WorkingMemoryResolutionStatus,
    WorkingMemoryResolver,
    WorkingMemoryResolverResult,
    WorkingMemoryRetentionReason,
    WorkingMemorySelection,
    WorkingMemorySourceKind,
    WorkingMemoryView,
    _ITEM_ID_DOMAIN,
    _SOURCE_ID,
    _validate_source as source_validate_working_memory_source,
    working_memory_item_id,
)


def test_source_contract_modules_import_without_runtime_or_model_dependencies() -> None:
    script = """
import sys
from suzka.context_contracts import ContextRelation
from suzka.emotion_contracts import EmotionState
from suzka.working_memory_contracts import WorkingMemoryItem, working_memory_item_id
from suzka.working_memory_contracts import WorkingMemorySourceKind

assert ContextRelation.SAME_CONTEXT.value == 'same_context'
assert EmotionState().optimal_loss == 1.0
assert WorkingMemoryItem.__dataclass_fields__
assert working_memory_item_id(
    WorkingMemorySourceKind.EPISODIC, 'episode-independent'
) == 'wm-5a2ef2f7e237f8fae733f1e3e562b9e84c3809fd9484e1493547e2b7376c22fa'
for prefix in ('suzka.runtime', 'suzka.cognition', 'suzka.models', 'torch', 'transformers'):
    assert not any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    ), prefix
"""
    environment = os.environ.copy()
    environment.pop("SUZKA_CONFIG_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr


def test_legacy_emotion_imports_keep_exact_type_defaults_and_validation() -> None:
    from suzka.body import EmotionState as BodyEmotionState
    from suzka.body.emotion_engine import EmotionState as LegacyEmotionState

    assert BodyEmotionState is LegacyEmotionState is EmotionState
    assert tuple(field.name for field in fields(EmotionState)) == (
        "valence", "arousal", "optimal_loss"
    )
    assert EmotionState() == EmotionState(0.0, 0.0, 1.0)
    assert EmotionState(-1, 1, -2).optimal_loss == -2.0
    with pytest.raises(TypeError, match="must be a number"):
        EmotionState(arousal=True)
    with pytest.raises(ValueError, match="must be finite"):
        EmotionState(optimal_loss=float("nan"))
    with pytest.raises(ValueError, match="must be in"):
        EmotionState(arousal=1.01)


def test_legacy_modules_reexport_identical_source_contracts() -> None:
    from suzka.runtime.context import ContextRelation as LegacyContextRelation
    from suzka.runtime.working_memory import (
        MAX_ITEM_CAPACITY as LegacyMaxItemCapacity,
        MAX_PROJECTION_BYTES as LegacyMaxProjectionBytes,
        MAX_SOURCE_ID_BYTES as LegacyMaxSourceIdBytes,
        _ITEM_ID_DOMAIN as legacy_item_id_domain,
        _SOURCE_ID as legacy_source_id_pattern,
        _validate_source as legacy_validate_working_memory_source,
        WorkingMemoryAdmission as LegacyWorkingMemoryAdmission,
        WorkingMemoryAdmissionReason as LegacyWorkingMemoryAdmissionReason,
        WorkingMemoryContextProjection as LegacyWorkingMemoryContextProjection,
        WorkingMemoryDecision as LegacyWorkingMemoryDecision,
        WorkingMemoryDecisionReason as LegacyWorkingMemoryDecisionReason,
        WorkingMemoryItem as LegacyWorkingMemoryItem,
        WorkingMemoryResolution as LegacyWorkingMemoryResolution,
        WorkingMemoryResolutionStatus as LegacyWorkingMemoryResolutionStatus,
        WorkingMemoryResolver as LegacyWorkingMemoryResolver,
        WorkingMemoryResolverResult as LegacyWorkingMemoryResolverResult,
        WorkingMemoryRetentionReason as LegacyWorkingMemoryRetentionReason,
        WorkingMemorySelection as LegacyWorkingMemorySelection,
        WorkingMemorySourceKind as LegacyWorkingMemorySourceKind,
        WorkingMemoryView as LegacyWorkingMemoryView,
        working_memory_item_id as legacy_working_memory_item_id,
    )

    assert LegacyContextRelation is ContextRelation
    assert LegacyMaxItemCapacity is MAX_ITEM_CAPACITY == 4_096
    assert LegacyMaxProjectionBytes is MAX_PROJECTION_BYTES == 16 * 1024 * 1024
    assert LegacyMaxSourceIdBytes is MAX_SOURCE_ID_BYTES == 128
    assert legacy_item_id_domain is _ITEM_ID_DOMAIN == b"suzka-working-memory-item-v1\0"
    assert legacy_source_id_pattern is _SOURCE_ID
    assert (
        legacy_validate_working_memory_source
        is source_validate_working_memory_source
    )
    assert tuple(relation.value for relation in ContextRelation) == (
        "same_context",
        "parent_child",
        "related",
        "shared_interlocutor",
        "legacy_unknown",
        "unknown_context",
        "unrelated",
    )
    assert tuple(kind.value for kind in WorkingMemorySourceKind) == (
        "episodic",
        "semantic",
    )
    assert tuple(reason.value for reason in WorkingMemoryRetentionReason) == (
        "recent",
        "reactivated",
    )
    assert tuple(reason.value for reason in WorkingMemoryDecisionReason) == (
        "selected",
        "unresolved_reference",
        "resolver_failure",
        "source_archived",
        "source_unavailable",
        "source_malformed",
        "projection_budget",
    )
    assert tuple(reason.value for reason in WorkingMemoryAdmissionReason) == (
        "admitted",
        "reactivated",
        "capacity_evicted",
    )
    assert tuple(status.value for status in WorkingMemoryResolutionStatus) == (
        "resolved",
        "missing",
        "archived",
        "unavailable",
        "malformed",
    )

    for legacy, contract in (
        (LegacyWorkingMemoryAdmission, WorkingMemoryAdmission),
        (LegacyWorkingMemoryAdmissionReason, WorkingMemoryAdmissionReason),
        (LegacyWorkingMemoryContextProjection, WorkingMemoryContextProjection),
        (LegacyWorkingMemoryDecision, WorkingMemoryDecision),
        (LegacyWorkingMemoryDecisionReason, WorkingMemoryDecisionReason),
        (LegacyWorkingMemoryItem, WorkingMemoryItem),
        (LegacyWorkingMemoryResolution, WorkingMemoryResolution),
        (LegacyWorkingMemoryResolutionStatus, WorkingMemoryResolutionStatus),
        (LegacyWorkingMemoryResolver, WorkingMemoryResolver),
        (LegacyWorkingMemoryResolverResult, WorkingMemoryResolverResult),
        (LegacyWorkingMemoryRetentionReason, WorkingMemoryRetentionReason),
        (LegacyWorkingMemorySelection, WorkingMemorySelection),
        (LegacyWorkingMemorySourceKind, WorkingMemorySourceKind),
        (LegacyWorkingMemoryView, WorkingMemoryView),
        (legacy_working_memory_item_id, working_memory_item_id),
    ):
        assert legacy is contract

    assert legacy_working_memory_item_id(
        LegacyWorkingMemorySourceKind.EPISODIC,
        "episode-independent",
    ) == "wm-5a2ef2f7e237f8fae733f1e3e562b9e84c3809fd9484e1493547e2b7376c22fa"
    layouts = (
        (
            LegacyWorkingMemoryItem,
            (
                "item_id",
                "source_kind",
                "source_id",
                "activation",
                "salience",
                "retention_reason",
                "created_revision",
                "last_activated_revision",
            ),
        ),
        (
            LegacyWorkingMemoryAdmission,
            ("item", "retained", "reason", "evicted_item_id"),
        ),
        (
            LegacyWorkingMemorySelection,
            (
                "item_id",
                "source_kind",
                "source_id",
                "rendered_content",
                "score",
                "reason",
                "source_context_id",
                "context_relation",
                "context_compatibility",
                "effective_score",
                "context_projection",
            ),
        ),
        (
            LegacyWorkingMemoryDecision,
            (
                "item_id",
                "source_kind",
                "source_id",
                "selected",
                "score",
                "reason",
                "context_relation",
                "context_compatibility",
                "effective_score",
                "context_projection",
            ),
        ),
        (
            LegacyWorkingMemoryView,
            (
                "selected",
                "decisions",
                "projected_bytes",
                "item_capacity",
                "projection_max_bytes",
                "revision",
            ),
        ),
        (
            LegacyWorkingMemoryResolution,
            ("status", "rendered_content", "source_context_id", "context_projection"),
        ),
    )
    for model, expected_fields in layouts:
        assert tuple(field.name for field in fields(model)) == expected_fields
