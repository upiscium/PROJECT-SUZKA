"""The current repository uses SUZKA for all product and durable namespaces."""

from pathlib import Path
import re
import subprocess

from suzka.belief import records as belief_records
from suzka.cognition import surprisal_calculator
from suzka.experience import records as experience_records
from suzka.identity import origin, value_system
from suzka.memory import episodic_participant, experience_participant
from suzka.memory import semantic_lifecycle, semantic_participant
from suzka.runtime import chat_context, event_journal, session_participant, working_memory


def test_authoritative_domain_constants_use_suzka_namespace() -> None:
    assert (
        belief_records.BELIEF_PROPOSITION_DOMAIN,
        belief_records.BELIEF_ADMISSION_DOMAIN,
        belief_records.BELIEF_REVISION_DOMAIN,
        belief_records.BELIEF_RECORD_DOMAIN,
        surprisal_calculator.MODEL_KEY_DOMAIN,
        experience_records._EXPERIENCE_DOMAIN,
        experience_records._EXPERIENCE_REVISION_DOMAIN,
        origin._ORIGIN_DOMAIN,
        value_system._SEED_DOMAIN,
        value_system._STATE_DOMAIN,
        value_system._RECORD_DOMAIN,
        value_system._GENESIS_DOMAIN,
        value_system._LEDGER_DOMAIN,
        episodic_participant._OPERATION_HASH_DOMAINS,
        experience_participant._OPERATION_DOMAIN,
        semantic_lifecycle.SEMANTIC_CONTENT_DOMAIN,
        semantic_lifecycle.SEMANTIC_PROVENANCE_DOMAIN,
        semantic_lifecycle.SEMANTIC_REVISION_DOMAIN,
        semantic_participant._OPERATION_DOMAIN,
        chat_context._CHAT_SESSION_DOMAIN,
        event_journal._HASH_DOMAIN_V1,
        event_journal._HASH_DOMAIN_V2,
        event_journal._HASH_DOMAIN_V3,
        event_journal._STARTUP_AGGREGATE_DOMAIN,
        session_participant._SESSION_HASH_DOMAIN,
        working_memory._ITEM_ID_DOMAIN,
    ) == (
        b"PROJECT-SUZKA:R12:BELIEF-PROPOSITION:V1\0",
        b"PROJECT-SUZKA:R12:BELIEF-ADMISSION:V1\0",
        b"PROJECT-SUZKA:R12:BELIEF-REVISION:V1\0",
        b"PROJECT-SUZKA:R12:BELIEF-RECORD:V1\0",
        b"PROJECT-SUZKA:R10:CALIBRATION-MODEL:V1\0",
        b"PROJECT-SUZKA:R12:EXPERIENCE:V1\0",
        b"PROJECT-SUZKA:R12:EXPERIENCE-REVISION:V1\0",
        "suzka.identity.origin/v1",
        "suzka.identity.value-seed/v1",
        "suzka.identity.value-state/v1",
        "suzka.identity.value-revision/v1",
        "suzka.identity.value-genesis/v1",
        "suzka.identity.value-evidence-ledger/v1",
        {
            1: b"PROJECT-SUZKA:R07:MEMORY-EPISODIC:V1\x00",
            2: b"PROJECT-SUZKA:R07:MEMORY-EPISODIC:V2\x00",
            3: b"PROJECT-SUZKA:R07:MEMORY-EPISODIC:V3\x00",
        },
        b"PROJECT-SUZKA:R12:EXPERIENCE-PARTICIPANT:V1\0",
        b"PROJECT-SUZKA:R12:SEMANTIC-CONTENT:V1\0",
        b"PROJECT-SUZKA:R12:SEMANTIC-PROVENANCE:V1\0",
        b"PROJECT-SUZKA:R12:SEMANTIC-REVISION:V1\0",
        b"PROJECT-SUZKA:R12:SEMANTIC-PARTICIPANT:V1\0",
        b"PROJECT-SUZKA:R09:CHAT-SESSION:V1\0",
        b"PROJECT-SUZKA:event-journal:v1\0",
        b"PROJECT-SUZKA:event-journal:v2\0",
        b"PROJECT-SUZKA:event-journal:v3\0",
        b"PROJECT-SUZKA:R07:STARTUP-PARTICIPANT-AGGREGATE:V1\x00",
        b"PROJECT-SUZKA:R07:SESSION-TURN:V1\x00",
        b"suzka-working-memory-item-v1\0",
    )


def test_tracked_tree_has_no_pre_rename_namespace() -> None:
    repository = Path(__file__).resolve().parents[1]
    product_marker = "PROJECT-" + "K" + "AGYA"
    forbidden = (
        product_marker,
        "K" + "AGYA_",
        "X-" + "K" + "AGYA",
        "." + "k" + "agya",
        "k" + "agya" + ".",
        "k" + "agya" + "-",
    )
    pattern = "|".join(re.escape(marker) for marker in forbidden)
    result = subprocess.run(
        ["git", "grep", "-n", "-E", pattern, "--", "."],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode in (0, 1)
    assert result.stdout == ""
