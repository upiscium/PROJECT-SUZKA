"""DC-1: settings (defaults, validation, safety against an existing production
config) and the health fields (F15 slice D1 r3 sec. 5.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest
import yaml

from suzka.api.server import create_app
from suzka.config.schema import DegradedContinuationSettings, Settings
from suzka.config.settings import load_settings
from suzka.runtime.agent_runtime import AgentRuntimeStatus
from suzka.runtime.degraded_continuation import DcState
from suzka.runtime.event_journal import (
    EventJournalAdmissionBand,
    EventJournalError,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "config.yaml", ROOT / "deploy" / "config" / "config.fhs.yaml.example"]


def _raw(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict)
    return raw


def test_defaults_match_the_approved_design() -> None:
    d = DegradedContinuationSettings()
    # DC-5: off by default so a deploy does not change eve; "continue" is opt-in.
    assert d.mode == "refuse"
    assert d.quarantine_dir is None
    assert (d.t_retry_s, d.t_hash_s) == (60, 30)
    assert d.t_boot_max_s == 240  # HD-new-3, not the older 120
    assert (d.t_teardown_max_s, d.t_hard_s, d.t_stable_s, d.t_shutdown_join_s) == (
        240,
        600,
        600,
        60,
    )
    assert (d.restart_limit, d.restart_window_s) == (3, 600)
    assert (d.reeval_max_backoff_s, d.cycle_restart_cap) == (300, 3)
    assert d.exit_on_threads_not_stopped is False
    assert d.quarantine_unavailable == "never"
    assert d.quarantine_unavailable_after_s == 86_400
    assert d.hard_band_auto_shrink is True
    assert d.hard_ladder_days == [14, 7, 3, 1, 0]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_an_existing_config_without_the_section_stays_valid(path: Path) -> None:
    raw = _raw(path)
    assert "degraded_continuation" not in raw  # as an installed config is today
    settings = Settings.model_validate(raw)
    assert settings.degraded_continuation == DegradedContinuationSettings()
    assert load_settings(path).degraded_continuation.mode == "refuse"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
@pytest.mark.parametrize(
    "retention_days", [0, 1, 2, 3, 4, 7, 8, 14, 15, 30, 90, 36_500]
)
def test_no_retention_days_makes_the_default_ladder_a_config_error(
    path: Path, retention_days: int
) -> None:
    # eve's real retention_days is not visible: whatever it is, it validates.
    raw = _raw(path)
    raw["event_journal"] = {
        **raw.get("event_journal", {}),
        "retention_days": retention_days,
    }
    settings = Settings.model_validate(raw)
    ladder = settings.degraded_continuation.effective_hard_ladder(
        settings.event_journal.retention_days
    )
    assert ladder == tuple(step for step in (14, 7, 3, 1, 0) if step < retention_days)
    assert all(step < retention_days for step in ladder)


@pytest.mark.parametrize(
    "retention_days, expected",
    [
        (30, (14, 7, 3, 1, 0)),
        (15, (14, 7, 3, 1, 0)),
        (14, (7, 3, 1, 0)),
        (10, (7, 3, 1, 0)),
        (2, (1, 0)),
        (1, (0,)),
        (0, ()),
        (-5, ()),  # never validated away; the field itself rejects it elsewhere
    ],
)
def test_ladder_is_filtered_to_steps_below_the_window(
    retention_days: int, expected: tuple[int, ...]
) -> None:
    assert (
        DegradedContinuationSettings().effective_hard_ladder(retention_days) == expected
    )


def test_a_custom_ladder_is_filtered_the_same_way() -> None:
    d = DegradedContinuationSettings(hard_ladder_days=[20, 10, 5])
    assert d.effective_hard_ladder(12) == (10, 5)
    assert d.effective_hard_ladder(5) == ()
    assert (
        DegradedContinuationSettings(hard_ladder_days=[]).effective_hard_ladder(30)
        == ()
    )


def test_the_shrink_switch_removes_every_rung() -> None:
    d = DegradedContinuationSettings(hard_band_auto_shrink=False)
    assert d.effective_hard_ladder(30) == ()
    assert d.hard_ladder_days == [14, 7, 3, 1, 0]  # the list itself is kept


@pytest.mark.parametrize(
    "update",
    [
        {"hard_ladder_days": [3, 7]},  # not descending
        {"hard_ladder_days": [7, 7]},
        {"hard_ladder_days": [7, -1]},
        {"hard_ladder_days": [7.5]},
        {"hard_ladder_days": [40_000]},
        {"mode": "off"},
        {"quarantine_unavailable": "always"},
        {"t_retry_s": 0},
        {"t_hash_s": -1},
        {"t_boot_max_s": float("nan")},
        {"t_hard_s": float("inf")},
        {"restart_limit": 0},
        {"cycle_restart_cap": 0},
        {"restart_window_s": 0},
        {"unknown_key": 1},
    ],
)
def test_explicit_nonsense_is_rejected(update: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DegradedContinuationSettings(**update)


def test_quarantine_dir_resolution(tmp_path: Path) -> None:
    journal = tmp_path / "state" / "event_journal.jsonl"
    assert DegradedContinuationSettings().resolved_quarantine_dir(journal) == (
        tmp_path / "state" / "quarantine"
    )
    explicit = tmp_path / "elsewhere"
    assert (
        DegradedContinuationSettings(quarantine_dir=explicit).resolved_quarantine_dir(
            journal
        )
        == explicit
    )


# ------------------------------------------------------------------ health


def _app() -> Any:
    app = create_app(load_settings(CONFIG_PATH))
    return app


CONFIG_PATH = ROOT / "config.yaml"


class _Status:
    def __init__(self, available: bool, band: EventJournalAdmissionBand) -> None:
        self.available = available
        self.band = band


class _Journal:
    def __init__(self, result: Any) -> None:
        self._result = result

    def admission_status(self) -> Any:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def _prepared(
    journal: Any, *, runtime: AgentRuntimeStatus = AgentRuntimeStatus.ACCEPTING
) -> Any:
    """An app without its lifespan: only what ``/health`` reads is set."""

    app = _app()
    app.state.external_reconciliation_required = False
    app.state.reconciliation_degraded_reason = None
    app.state.event_journal = journal
    app.state.startup_retention_admission_available = True
    app.state.startup_retention_reason = None

    class _Runtime:
        status = runtime

    app.state.agent_runtime = _Runtime()
    return app


def test_health_dc_answers_with_no_app_state_at_all() -> None:
    app = _app()  # no lifespan, no app.state beyond settings and dc_status
    client = TestClient(app)
    response = client.get("/health/dc")
    assert response.status_code == 200
    assert response.json() == {"dc_state": "INACTIVE"}
    app.state.dc_status.publish(
        DcState.REFUSED_RUNNING, reason="suzka.x.Boom", participants=["memory.semantic"]
    )
    body = client.get("/health/dc").json()
    assert body["dc_state"] == "REFUSED_RUNNING"
    assert body["dc_reason"] == "suzka.x.Boom"
    assert body["dc_participants"] == "memory.semantic"
    assert set(body) <= {
        "dc_state",
        "dc_reason",
        "dc_since",
        "dc_participants",
        "dc_epoch_id",
    }


def test_health_is_unchanged_while_degraded_continuation_is_inactive() -> None:
    ok = _prepared(_Journal(_Status(True, EventJournalAdmissionBand.NORMAL)))
    name = ok.state.settings.project.name
    assert TestClient(ok).get("/health").json() == {"status": "ok", "project": name}
    ok.state.dc_status.publish(DcState.SERVING)  # plain SERVING adds nothing either
    assert TestClient(ok).get("/health").json() == {"status": "ok", "project": name}
    soft = _prepared(_Journal(_Status(True, EventJournalAdmissionBand.SOFT)))
    assert TestClient(soft).get("/health").json() == {
        "status": "degraded",
        "project": name,
        "reason": "journal_soft_limit",
    }


def test_health_carries_the_dc_fields_when_there_is_something_to_say() -> None:
    app = _prepared(_Journal(_Status(True, EventJournalAdmissionBand.NORMAL)))
    app.state.dc_status.publish(
        DcState.SERVING,
        reason="domain_quarantined",
        participants=["memory.experience"],
        epoch_id="e1",
    )
    body = TestClient(app).get("/health").json()
    assert body["status"] == "ok"  # the existing keys are untouched
    assert body["dc_state"] == "SERVING"
    assert body["dc_reason"] == "domain_quarantined"
    assert body["dc_participants"] == "memory.experience"
    assert body["dc_epoch_id"] == "e1"
    assert all(isinstance(v, str) for v in body.values())


def test_a_journal_that_cannot_be_inspected_is_not_called_retention_exhausted() -> None:
    name = _app().state.settings.project.name
    unreadable = _prepared(_Journal(EventJournalError("cannot read")))
    assert TestClient(unreadable).get("/health").json() == {
        "status": "degraded",
        "project": name,
        "reason": "event_journal_unavailable",
    }
    # a real retention exhaustion keeps its label
    exhausted = _prepared(_Journal(_Status(False, EventJournalAdmissionBand.NORMAL)))
    assert TestClient(exhausted).get("/health").json()["reason"] == (
        "event_journal_retention_exhausted"
    )
    hard = _prepared(_Journal(_Status(False, EventJournalAdmissionBand.HARD)))
    assert TestClient(hard).get("/health").json()["reason"] == "journal_hard_limit"
    # startup said HARD and the journal is unreadable now: still the hard label
    startup_hard = _prepared(_Journal(EventJournalError("cannot read")))
    startup_hard.state.startup_retention_admission_available = False
    startup_hard.state.startup_retention_reason = "journal_hard_limit"
    assert (
        TestClient(startup_hard).get("/health").json()["reason"] == "journal_hard_limit"
    )
