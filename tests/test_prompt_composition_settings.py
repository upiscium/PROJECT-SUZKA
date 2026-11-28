"""#500 slice 3b-1: shape and cross-validation of ``prompt_composition``."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from suzka.config import Settings, load_settings
from suzka.config.schema import PromptCompositionSettings
from suzka.persona.chat_composition import CompositionMode

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _raw() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _settings(composition: dict[str, Any] | None = None, **trace: Any) -> Settings:
    raw = _raw()
    raw.pop("prompt_composition", None)
    if composition is not None:
        raw["prompt_composition"] = composition
    raw["conversation_trace"] = {**raw.get("conversation_trace", {}), **trace}
    return Settings.model_validate(raw)


def test_defaults_are_the_documented_provisional_values() -> None:
    cfg = PromptCompositionSettings()
    assert cfg.mode == "legacy"
    assert cfg.history_window == 6
    assert cfg.max_history_bytes == 32768
    assert cfg.max_state_block_bytes == 32768
    assert cfg.evidence_top_k == 3
    assert cfg.evidence_item_max_chars == 200


def test_config_without_the_section_loads_with_defaults() -> None:
    raw = _raw()
    raw.pop("prompt_composition")
    assert Settings.model_validate(raw).prompt_composition == PromptCompositionSettings()


def test_repo_config_yaml_loads_and_matches_defaults() -> None:
    settings = load_settings(CONFIG)
    assert settings.prompt_composition == PromptCompositionSettings()
    assert settings.conversation_trace.enabled is False


def test_mode_literal_equals_composition_mode_values() -> None:
    literal = PromptCompositionSettings.model_fields["mode"].annotation
    assert set(literal.__args__) == {m.value for m in CompositionMode}  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [
        ("history_window", 0, 64),
        ("max_history_bytes", 0, 16_777_216),
        ("max_state_block_bytes", 0, 16_777_216),
        ("evidence_top_k", 1, 50),
        ("evidence_item_max_chars", 20, 2000),
    ],
)
def test_integer_bounds(field: str, low: int, high: int) -> None:
    assert getattr(PromptCompositionSettings(**{field: low}), field) == low
    assert getattr(PromptCompositionSettings(**{field: high}), field) == high
    for bad in (low - 1, high + 1):
        with pytest.raises(ValidationError):
            PromptCompositionSettings(**{field: bad})
    for bad_type in (True, 3.0, "3"):
        with pytest.raises(ValidationError):
            PromptCompositionSettings(**{field: bad_type})


@pytest.mark.parametrize("field", ["evidence_top_k", "evidence_item_max_chars"])
def test_evidence_limits_accept_none_independently(field: str) -> None:
    cfg = PromptCompositionSettings(**{field: None})
    assert getattr(cfg, field) is None
    other = "evidence_item_max_chars" if field == "evidence_top_k" else "evidence_top_k"
    assert getattr(cfg, other) is not None


@pytest.mark.parametrize("field", ["history_window", "max_history_bytes", "max_state_block_bytes"])
def test_non_nullable_fields_reject_none(field: str) -> None:
    with pytest.raises(ValidationError):
        PromptCompositionSettings(**{field: None})


def test_unknown_mode_and_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        PromptCompositionSettings(mode="chat")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PromptCompositionSettings(bogus=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("mode", ["chat_roles", "chat_roles_split"])
def test_non_legacy_mode_requires_enabled_trace(mode: str) -> None:
    with pytest.raises(ValidationError, match=r"conversation_trace\.enabled"):
        _settings({"mode": mode})
    assert _settings({"mode": mode}, enabled=True).prompt_composition.mode == mode


def test_history_window_must_fit_trace_capacity() -> None:
    with pytest.raises(ValidationError, match="max_items_per_context"):
        _settings(
            {"mode": "chat_roles", "history_window": 21},
            enabled=True,
            max_items_per_context=20,
        )
    ok = _settings(
        {"mode": "chat_roles", "history_window": 20},
        enabled=True,
        max_items_per_context=20,
    )
    assert ok.prompt_composition.history_window == 20


def test_legacy_ignores_both_cross_checks() -> None:
    settings = _settings(
        {"mode": "legacy", "history_window": 64}, enabled=False, max_items_per_context=1
    )
    assert settings.prompt_composition.history_window == 64
