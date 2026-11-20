"""IP-21a: settings, event kinds, soft / hard band, no schema change
(IP-21 D1 r3 §2, §3, §10-3, A-8).  No writer, no route and no Scheduler step
exist yet, so none is tested here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from suzka.config.schema import R19Settings, Settings
from suzka.r16.history_store import HISTORY_SCHEMA_VERSION, SCHEMA_HASH
from suzka.r19.contracts import MAX_MESSAGE_TTL
from suzka.runtime.agent_runtime import AgentEventSource, AgentEventType
from suzka.runtime.event_journal import (
    SOFT_BAND_ADMITTED_EVENT_TYPES,
    EventJournalAdmissionBand,
)

from tests.test_event_journal_capacity_bands import _at_band, _typed

OUTBOX_TYPES = (AgentEventType.R19_WAKE_COMPLETE, AgentEventType.R19_INBOX_ACK)
OUTBOX_SOURCES = (
    AgentEventSource.RUNTIME_R19_WAKE_COMPLETE,
    AgentEventSource.API_R19_INBOX_ACK,
)

# The schema hash of v4 as of IP-14b-1. IP-21 needs no v5 (D1 §2): a change
# here is a schema change and needs its own design.
V4_SCHEMA_HASH = "5d1504ffc421ce17abd1d05b09111a8ffee55666c85baaaec7df172c0bc22de8"


# ----------------------------------------------------------------- settings


def test_outbox_ttl_default_is_the_message_maximum_and_r19_stays_off() -> None:
    settings = R19Settings()
    assert settings.enabled is False
    assert settings.outbox_ttl_s == 604_800 == int(MAX_MESSAGE_TTL.total_seconds())
    # A configuration with no ``r19`` section (eve's /etc) takes the defaults.
    default_factory = Settings.model_fields["r19"].default_factory
    assert default_factory is not None and default_factory() == settings  # type: ignore[call-arg]


@pytest.mark.parametrize("value", [60, 3600, 604_800])
def test_outbox_ttl_accepts_its_bounds(value: int) -> None:
    assert R19Settings(outbox_ttl_s=value).outbox_ttl_s == value


@pytest.mark.parametrize(
    "changes",
    [
        {"outbox_ttl_s": 59},
        {"outbox_ttl_s": 0},
        {"outbox_ttl_s": -1},
        {"outbox_ttl_s": 604_801},  # above the 7-day message maximum
        {"outbox_ttl_s": True},
        {"outbox_ttl_s": 60.5},
        {"outbox_ttl_s": "3600"},
        {"outbox_ttl": 3600},  # an unknown name is refused (strict model)
    ],
)
def test_outbox_ttl_is_validated(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        R19Settings(**changes)


def test_the_wake_settings_are_unchanged_by_the_outbox_ttl() -> None:
    before = R19Settings().model_dump()
    assert set(before) - {"outbox_ttl_s"} == {
        "enabled",
        "goal_wake_delay_s",
        "commitment_wake_delay_s",
        "feedback_wake_delay_s",
        "deadline_margin_s",
        "claim_lease_s",
        "tick_s",
        "rescan_s",
        "materialize_max_age_s",
    }
    # The ttl does not interact with the timing validation either way.
    assert R19Settings(outbox_ttl_s=60, claim_lease_s=120).claim_lease_s == 120
    with pytest.raises(ValidationError):
        R19Settings(outbox_ttl_s=60, claim_lease_s=121)


# -------------------------------------------------------------- event kinds


def test_the_two_event_kinds_and_sources_exist() -> None:
    assert [item.value for item in OUTBOX_TYPES] == ["r19_wake_complete", "r19_inbox_ack"]
    assert [item.value for item in OUTBOX_SOURCES] == [
        "runtime.r19.wake_complete",
        "api.r19.inbox_ack",
    ]
    assert len({item.value for item in AgentEventType}) == len(AgentEventType)
    assert len({item.value for item in AgentEventSource}) == len(AgentEventSource)


@pytest.mark.parametrize("event_type", OUTBOX_TYPES)
def test_each_event_is_admitted_in_the_soft_band_and_refused_in_the_hard_band(
    tmp_path: Path, event_type: AgentEventType
) -> None:
    assert event_type in SOFT_BAND_ADMITTED_EVENT_TYPES
    _path, _clock, soft, _following = _at_band(
        tmp_path / "soft", EventJournalAdmissionBand.SOFT
    )
    assert soft.admission_status().admits(event_type)
    assert soft.append_accepted_if_admission_available(
        _typed(f"soft-{event_type.value}", event_type)
    )
    soft.close()
    _path, _clock, hard, _following = _at_band(
        tmp_path / "hard", EventJournalAdmissionBand.HARD
    )
    assert not hard.admission_status().admits(event_type)
    before = hard.records
    assert not hard.append_accepted_if_admission_available(
        _typed(f"hard-{event_type.value}", event_type)
    )
    assert hard.records == before  # a refusal writes nothing
    hard.close()


# --------------------------------------------------------------- no schema


def test_there_is_no_schema_change() -> None:
    assert HISTORY_SCHEMA_VERSION == 4
    assert SCHEMA_HASH == V4_SCHEMA_HASH
