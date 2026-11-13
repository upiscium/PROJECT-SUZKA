"""IP-4b (#383 / #381 §6, D2 P1-4): soft and hard limits over live bytes.

Live bytes are the active segment plus the rotated segments; archives are
not counted. Above the soft limit only cognition is admitted (a closed
allow-list over the 16 AgentEventTypes); above the hard limit no new event
is admitted. Neither band blocks the lifecycle of an already admitted event.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from suzka.config.schema import EventJournalSettings
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from suzka.runtime.event_journal import (
    SOFT_BAND_ADMITTED_EVENT_TYPES,
    EventJournal,
    EventJournalAdmissionBand,
    EventLifecycle,
)

from tests.test_event_journal import HASH_0, HASH_1
from tests.test_event_journal_compaction import _Clock, _fill, _segments


def _journal(
    path: Path,
    clock: _Clock,
    *,
    max_total_bytes: int | None,
    soft_band_bytes: int = 0,
    max_bytes: int = 3_000,
) -> EventJournal:
    value = EventJournal(
        path,
        max_bytes,
        2,
        clock=clock,
        max_total_bytes=max_total_bytes,
        soft_band_bytes=soft_band_bytes,
    )
    if not value.has_history:
        value.append_v2_bootstrap_checkpoint(
            0,
            HASH_0,
            str(uuid5(NAMESPACE_URL, "bands-generation")),
            str(uuid5(NAMESPACE_URL, "bands-wal")),
            HASH_1,
        )
        value.append_v3_migration_checkpoint()
        value.append_v4_migration_checkpoint()
    return value


def _live_bytes(path: Path) -> int:
    return path.stat().st_size + sum(item.stat().st_size for item in _segments(path))


def _typed(name: str, event_type: AgentEventType) -> AgentEvent:
    return AgentEvent(
        str(uuid5(NAMESPACE_URL, name)),
        event_type,
        AgentEventSource.API_CHAT,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        None,
    )


def _at_band(
    tmp_path: Path, band: EventJournalAdmissionBand
) -> tuple[Path, _Clock, EventJournal, int]:
    """A Journal filled past several segments, then limited into ``band``."""
    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock, max_total_bytes=None)
    following = _fill(value, 1, 8)
    assert len(_segments(path)) >= 2
    live = _live_bytes(path)
    value.close()
    if band is EventJournalAdmissionBand.SOFT:
        limits = {"max_total_bytes": live + 1_000, "soft_band_bytes": 2_000}
    elif band is EventJournalAdmissionBand.HARD:
        limits = {"max_total_bytes": live - 1, "soft_band_bytes": 0}
    else:
        limits = {"max_total_bytes": live * 10, "soft_band_bytes": live}
    value = _journal(path, clock, **limits)
    return path, clock, value, following


# --- the allow-list ------------------------------------------------------------


def test_soft_band_allow_list_matches_the_design_table() -> None:
    """#381 D2 P1-4: 13 cognitive types admitted, the other 3 refused."""
    refused = set(AgentEventType) - SOFT_BAND_ADMITTED_EVENT_TYPES
    assert refused == {
        AgentEventType.DEBUG_CHAT,
        AgentEventType.ADAPTER_EVALUATE,
        AgentEventType.ADAPTER_UPDATE,
    }
    assert len(AgentEventType) == 16
    assert SOFT_BAND_ADMITTED_EVENT_TYPES <= set(AgentEventType)


# --- the bands -----------------------------------------------------------------


def test_normal_band_admits_everything(tmp_path: Path) -> None:
    _path, _clock, value, _following = _at_band(
        tmp_path, EventJournalAdmissionBand.NORMAL
    )
    status = value.admission_status()
    assert status.band is EventJournalAdmissionBand.NORMAL
    assert status.available and status.reason is None
    assert all(status.admits(item) for item in AgentEventType)
    assert value.append_accepted_if_admission_available(
        _typed("normal-debug", AgentEventType.DEBUG_CHAT)
    )
    value.close()


def test_soft_band_admits_only_cognition_and_writes_nothing_on_refusal(
    tmp_path: Path,
) -> None:
    path, _clock, value, _following = _at_band(tmp_path, EventJournalAdmissionBand.SOFT)
    status = value.admission_status()
    assert status.band is EventJournalAdmissionBand.SOFT
    assert status.available
    assert status.reason == "journal_soft_limit"
    assert status.live_bytes == _live_bytes(path)
    assert status.soft_limit_bytes == status.max_total_bytes - 2_000  # type: ignore[operator]
    for item in AgentEventType:
        assert status.admits(item) is (item in SOFT_BAND_ADMITTED_EVENT_TYPES)
    for item in (
        AgentEventType.DEBUG_CHAT,
        AgentEventType.ADAPTER_EVALUATE,
        AgentEventType.ADAPTER_UPDATE,
    ):
        before = value.records
        assert not value.append_accepted_if_admission_available(
            _typed(f"soft-{item.value}", item)
        )
        assert value.records == before
    assert value.append_accepted_if_admission_available(
        _typed("soft-chat", AgentEventType.CHAT)
    )
    assert value.records[-1].lifecycle is EventLifecycle.ACCEPTED
    value.close()


def test_hard_band_refuses_every_new_event(tmp_path: Path) -> None:
    _path, _clock, value, _following = _at_band(tmp_path, EventJournalAdmissionBand.HARD)
    status = value.admission_status()
    assert status.band is EventJournalAdmissionBand.HARD
    assert not status.available
    assert status.reason == "event_journal_retention_exhausted"
    before = value.records
    for item in AgentEventType:
        assert not status.admits(item)
        assert not value.append_accepted_if_admission_available(
            _typed(f"hard-{item.value}", item)
        )
    assert value.records == before
    value.close()


def test_admitted_lifecycle_completes_in_the_hard_band(tmp_path: Path) -> None:
    """#376 D2-4: terminal records of accepted events are still appended."""
    path = tmp_path / "events.jsonl"
    clock = _Clock()
    value = _journal(path, clock, max_total_bytes=None)
    following = _fill(value, 1, 8)
    live = _live_bytes(path)
    value.close()
    # One ordinary turn (four records) pushes the Journal over the hard limit
    # while its own lifecycle is still being written.
    value = _journal(path, clock, max_total_bytes=live + 200)
    assert value.admission_status().band is EventJournalAdmissionBand.NORMAL
    _fill(value, following, 1)
    assert value.admission_status().band is EventJournalAdmissionBand.HARD
    assert value.inspect().open_events == ()
    # A rotation checkpoint may follow; the event's terminal record is there.
    assert [
        record.lifecycle
        for record in value.records
        if record.lifecycle is not EventLifecycle.CHECKPOINT
    ][-1] is EventLifecycle.COMPLETED
    value.close()


def test_compaction_returns_the_journal_to_the_normal_band(tmp_path: Path) -> None:
    path, clock, value, _following = _at_band(tmp_path, EventJournalAdmissionBand.HARD)
    assert value.admission_status().band is EventJournalAdmissionBand.HARD
    clock.advance(days=31)
    assert value.compact(retention_days=30).performed
    status = value.admission_status()
    assert status.band is EventJournalAdmissionBand.NORMAL
    assert status.available
    assert status.live_bytes == _live_bytes(path)
    # Archives are not live bytes.
    assert any(value.archive_dir.iterdir())
    value.close()


def test_no_total_limit_keeps_the_previous_behaviour(tmp_path: Path) -> None:
    _path, _clock, value, _following = _at_band(tmp_path, EventJournalAdmissionBand.NORMAL)
    value.close()
    path = tmp_path / "events.jsonl"
    unlimited = EventJournal(path, 3_000, 2, clock=_Clock())
    status = unlimited.admission_status()
    assert status.band is EventJournalAdmissionBand.NORMAL
    assert status.max_total_bytes is None and status.soft_limit_bytes is None
    unlimited.close()


# --- configuration -------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_total_bytes", "soft_band_bytes"),
    [(0, 0), (-1, 0), (True, 0), (100, 100), (100, 101), (100, -1), (None, 1)],
)
def test_invalid_limits_are_refused(
    tmp_path: Path, max_total_bytes: object, soft_band_bytes: object
) -> None:
    with pytest.raises(ValueError):
        EventJournal(
            tmp_path / "events.jsonl",
            3_000,
            2,
            max_total_bytes=max_total_bytes,  # type: ignore[arg-type]
            soft_band_bytes=soft_band_bytes,  # type: ignore[arg-type]
        )


def test_settings_defaults_and_validation() -> None:
    settings = EventJournalSettings()
    assert settings.max_total_bytes == 72 * 1024 * 1024
    assert settings.soft_band_bytes == 16 * 1024 * 1024
    assert EventJournalSettings(max_total_bytes=None, soft_band_bytes=0)
    with pytest.raises(ValueError):
        EventJournalSettings(max_total_bytes=100, soft_band_bytes=100)
    with pytest.raises(ValueError):
        EventJournalSettings(max_total_bytes=None, soft_band_bytes=1)
