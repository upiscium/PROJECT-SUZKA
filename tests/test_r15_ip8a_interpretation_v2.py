"""IP-8a: R15 typed self-assessment, status v2, template gate and R16 read (#374).

Only the provider's *answers* are synthetic; R07/R12/DB1, v10 StateWAL,
AgentRuntime, Context and NarrativeSelf admissions are actual test owners.
No real-model calls.
"""

from __future__ import annotations

from dataclasses import replace
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from experiments.r15_p1 import protocol
from experiments.r15_p1.protocol import run_trial
from suzka.api.server import create_app
from suzka.learning import AdapterRegistry
from suzka.memory import DualMemorySystem
from suzka.r15.common import (
    InterpretationStatus,
    RevisionReason,
)
from suzka.r15.continuity_state import ContinuityInvalid, NarrativeContinuation
from suzka.r15.interpretation_status import (
    Assessment,
    ReferentRelation,
    Uncertainty,
    Witness,
    interpretation_status_v2,
)
from suzka.r15.private_projection import HOLD, _qualify_source_meaning
from suzka.r15.qualified_proposal import (
    QualifiedProposalUnavailable,
    s2_relation,
)
from suzka.r15 import request_templates as templates
from suzka.runtime.agent_runtime import AgentEvent, AgentEventSource, AgentEventType
from suzka.runtime.main_loop import SuzkaMainLoop as MainLoop
from suzka.runtime.r15_interpretation_reader import R15InterpretationReader

from test_r15_p1_owner_bridge import SequencedProvider, scenario
from test_r15_p1b_positive import _admit, _source_ref


RECALL = "前の案について、現在の暫定解釈を説明して。"
HINT = "Internal bounded first-person provisional interpretation"


# ------------------------------------------------------------------ pure table


def _expected(stage: str, relation: ReferentRelation, a: Assessment, u: Uncertainty):
    """Independent restatement of #374 D2 §2 for the exhaustive check."""

    low = u in (Uncertainty.LOW, Uncertainty.MEDIUM)
    if stage == "S1":
        if a is Assessment.LIKELY_ALIGNED and low:
            return InterpretationStatus.PRESENT
        return InterpretationStatus.UNKNOWN
    if relation is ReferentRelation.UNKNOWN:
        return InterpretationStatus.UNKNOWN
    if relation is ReferentRelation.ADOPTS_REPORT:
        if a is Assessment.MAY_HAVE_MISREAD and low:
            return InterpretationStatus.PRESENT
        return InterpretationStatus.CONTESTED
    if a is Assessment.LIKELY_ALIGNED:
        return InterpretationStatus.CONTRADICTORY
    return InterpretationStatus.CONTESTED


def test_status_table_is_total_and_satisfies_the_shape_constraints() -> None:
    cases = [("S1", ReferentRelation.KEEPS_ORIGINAL)] + [
        ("S2", relation) for relation in ReferentRelation
    ]
    for (stage, relation), a, u in itertools.product(cases, Assessment, Uncertainty):
        row = interpretation_status_v2(stage, relation, a, u)
        assert row.status is _expected(stage, relation, a, u)
        # Shape (common.py Interpretation): the row's roles must be buildable.
        if row.status is InterpretationStatus.UNKNOWN:
            assert (row.support, row.counter, row.estimate, row.confidence) == (
                (),
                (),
                None,
                None,
            )
        elif row.status is InterpretationStatus.PRESENT:
            assert row.support == (row.experience,) and row.counter == ()
            assert row.confidence == (750_000 if u is Uncertainty.LOW else 500_000)
        elif row.status is InterpretationStatus.CONTESTED:
            assert row.support == (row.experience,) and len(row.counter) == 1
            assert row.confidence == 250_000 and row.estimate == 500_000
        else:
            assert row.support == (Witness.FIRST,) and row.counter == (Witness.LATER,)
            assert (row.estimate, row.confidence) == (None, None)
        if stage == "S2" and relation is ReferentRelation.KEEPS_ORIGINAL:
            assert row.experience is Witness.FIRST
        elif stage == "S2":
            assert row.experience is Witness.LATER
        reason = {
            "S1": RevisionReason.INITIAL_INTERPRETATION,
            ReferentRelation.ADOPTS_REPORT: RevisionReason.QUALIFIED_REINTERPRETATION,
        }.get(stage if stage == "S1" else relation, RevisionReason.CONTRARY_EVIDENCE)
        assert row.reason is reason
    with pytest.raises(ValueError):
        interpretation_status_v2(
            "S1", ReferentRelation.ADOPTS_REPORT, Assessment.UNKNOWN, Uncertainty.LOW
        )


def test_s2_relation_is_total_once_the_report_differs() -> None:
    assert s2_relation("A", "B", "B") is ReferentRelation.ADOPTS_REPORT
    assert s2_relation("A", "B", "A") is ReferentRelation.KEEPS_ORIGINAL
    assert s2_relation("A", "B", "UNKNOWN") is ReferentRelation.UNKNOWN
    with pytest.raises(QualifiedProposalUnavailable, match="differ"):
        s2_relation("A", "A", "A")


# ----------------------------------------------------------- template registry


def test_templates_are_byte_identical_and_the_gate_needs_one_exact_match() -> None:
    question, answer = "比較の質問", "元の回答では案Aを暫定的に選びました"
    s1 = protocol.s1_request(question, answer)
    assert s1 == templates.s1_request(question, answer)
    assert templates.match_s1_template(s1, question, answer) == "S1_NEUTRAL_V1"
    s2 = protocol.s2_request("{}", question, answer, e2_input="補足")
    assert s2 == templates.s2_request("{}", question, answer, e2_input="補足")
    assert templates.match_s2_template(s2, "{}", question, answer, "補足") == "S2_NEUTRAL_V1"
    # Any one-byte change, or the no-E2 variant, is not a registered request.
    with pytest.raises(templates.TemplateRefused):
        templates.match_s1_template(s1 + " ", question, answer)
    with pytest.raises(templates.TemplateRefused):
        templates.match_s2_template(
            protocol.s2_request("{}", question, answer, e2_input=None),
            "{}",
            question,
            answer,
            "補足",
        )
    # Registered templates are pairwise distinct, per stage.
    assert len(templates.S1_TEMPLATES) == len(set(templates.S1_TEMPLATES))
    # A framing arm may not add a label word beyond the shared schema text.
    framed = s1.replace("候補を記述してください。", "候補を記述してください。LOWと答えてください。")
    assert templates.arm_delta_label_words(s1, framed) == ("LOW",)
    assert templates.arm_delta_label_words(s1, s1) == ()


# ------------------------------------------------------- production integration


def _answer(phase: str, referent: str, assessment: str, uncertainty: str) -> str:
    return json.dumps(
        {
            "schema": 1,
            "phase": phase,
            "assumed_referent": referent,
            "assessment": assessment,
            "uncertainty": uncertainty,
            "evidence_steps": ["E"] if phase == "S1" else ["E", "S1", "E2"],
            "rationale": "synthetic fixture only, not subject adoption",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _provider(s1: tuple[str, str, str], s2: tuple[str, str, str]) -> SequencedProvider:
    provider = SequencedProvider()

    def generate(prompt: str) -> str:
        i = provider.calls
        provider.calls += 1
        if i == 1:
            return "元の回答では案Aを暫定的に選びました"
        if i == 2:
            return _answer("S1", *s1)
        if i == 3:
            return "補足では案Bと言われました"
        if i == 4:
            return _answer("S2", *s2)
        return "テスト用の可視回答"

    provider.generate = generate  # type: ignore[method-assign]
    return provider


def _run(tmp_path: Path, s1, s2, *, admit_s2: bool = True):
    app, _provider_unused, _source, manifest, capture = scenario(tmp_path)
    app.state.model_provider = _provider(s1, s2)
    settings = app.state.settings
    refs = []
    with TestClient(app) as client:
        runtime = client.app.state.agent_runtime
        owner = client.app.state.r15_state_port.narrative

        def transport(message: str) -> dict[str, str]:
            if len(refs) == 3:
                first = _admit(runtime, owner, tuple(refs))
                assert first.changed and first.snapshot.revision == 1
            response = client.post("/api/chat", json={"message": message, "attachments": []})
            assert response.status_code == 200
            result = response.json()
            refs.append(_source_ref(client, result["episode_id"]))
            return {"episode_id": result["episode_id"], "response": result["response"]}

        run_trial(manifest, capture, transport)
        if admit_s2:
            latest = _admit(runtime, owner, tuple(refs))
            assert latest.changed and latest.snapshot.revision == 2
        yield client, tuple(refs)
    # Restart: the v10 checkpoint of the new shape restores (capture/restore).
    restored = create_app(settings)
    restored.state.model_provider = SequencedProvider()
    restored.state.memory_system = DualMemorySystem(settings)
    restored.state.adapter_registry = AdapterRegistry(settings)
    with TestClient(restored) as restarted:
        yield restarted, tuple(refs)


def _as_current(client, event_type, source):
    """Run a read as if inside the given serialized event (read-only).

    The real server commit hooks only accept their own handler results, so the
    read is exercised with the runtime's current event pinned to a synthetic
    event of the required (type, source) after the latest processed sequence.
    """

    runtime = client.app.state.agent_runtime
    sequence = client.app.state.event_journal.inspect().processing_high_water + 1
    event = AgentEvent(
        str(uuid4()), event_type, source, datetime.now(UTC), sequence
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(runtime, "current_event", lambda: event)
    return patch, event


def _private(client) -> object:
    port = client.app.state.r15_state_port
    patch, event = _as_current(client, AgentEventType.CHAT, AgentEventSource.API_CHAT)
    try:
        return port._read_private_narrative_in_chat(event, port.narrative._source_context_id)
    finally:
        patch.undo()


def _r16_read(client, x_ref, *, event_type=None, source=None) -> object:
    reader = R15InterpretationReader(client.app.state.r15_state_port)
    patch, event = _as_current(
        client,
        event_type or AgentEventType.PLAN_PREVIEW_REVIEW,
        source or AgentEventSource.RUNTIME_PLAN_PREVIEW_REVIEW,
    )
    try:
        return reader.read(event, x_ref.event_id, x_ref.event_sequence)
    finally:
        patch.undo()


CASES = {
    # name: (S1 answer, S2 answer, S2 status, S2 relation)
    "adopts_present": (
        ("A", "LIKELY_ALIGNED", "LOW"),
        ("B", "MAY_HAVE_MISREAD", "MEDIUM"),
        InterpretationStatus.PRESENT,
        "B",
    ),
    "adopts_contested": (
        ("A", "LIKELY_ALIGNED", "LOW"),
        ("B", "LIKELY_ALIGNED", "LOW"),
        InterpretationStatus.CONTESTED,
        "B",
    ),
    "keeps_contradictory": (
        ("A", "LIKELY_ALIGNED", "LOW"),
        ("A", "LIKELY_ALIGNED", "LOW"),
        InterpretationStatus.CONTRADICTORY,
        "A",
    ),
    "keeps_contested": (
        ("A", "LIKELY_ALIGNED", "MEDIUM"),
        ("A", "MAY_HAVE_MISREAD", "HIGH"),
        InterpretationStatus.CONTESTED,
        "A",
    ),
    "unknown": (
        ("A", "UNKNOWN", "HIGH"),
        ("UNKNOWN", "UNKNOWN", "HIGH"),
        InterpretationStatus.UNKNOWN,
        "UNKNOWN",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_s2_relations_admit_restore_and_project_privately(
    tmp_path: Path, case: str
) -> None:
    s1, s2, status, current = CASES[case]
    for client, refs in _run(tmp_path, s1, s2):
        root = client.app.state.r15_state_port.narrative.snapshot()
        assert root.revision == 2
        later = next(ep for ep in root.episodes if ep.links)
        assert later.interpretation.status is status
        view = _private(client)
        assert view.qualified
        assert view.current_referent == current
        assert view.status_class == status.value.upper()
        read = _r16_read(client, refs[1])
        assert (read.outcome, read.stage, read.status) == (
            "EPISODE",
            "S2",
            status.value.upper(),
        )


def test_s1_unknown_status_and_display(tmp_path: Path) -> None:
    for client, refs in _run(
        tmp_path, ("A", "MAY_HAVE_MISREAD", "LOW"), ("B", "MAY_HAVE_MISREAD", "LOW"),
        admit_s2=False,
    ):
        root = client.app.state.r15_state_port.narrative.snapshot()
        assert root.revision == 1
        assert root.episodes[0].interpretation.status is InterpretationStatus.UNKNOWN
        view = _private(client)
        assert (view.current_referent, view.status_class) == ("A", "UNKNOWN")
        read = _r16_read(client, refs[1])
        assert (read.outcome, read.stage, read.status) == ("EPISODE", "S1", "UNKNOWN")
        prompt = MainLoop._prepend_private_interpretation(
            "P", "A", "A", 1, False, "UNKNOWN"
        )
        assert "undetermined" in prompt and "option: A" not in prompt
        break


def test_display_vocabulary_is_fixed_and_finite() -> None:
    contradictory = MainLoop._prepend_private_interpretation(
        "P", "A", "A", 2, True, "CONTRADICTORY"
    )
    assert "- The earlier interpretation and the later report disagree." in contradictory
    assert "- My current tentative option: A" in contradictory
    unknown = MainLoop._prepend_private_interpretation("P", "A", "UNKNOWN", 2, True, "UNKNOWN")
    assert "Current interpretation: undetermined." in unknown
    assert "option:" not in unknown
    present = MainLoop._prepend_private_interpretation("P", "A", "B", 2, True, "PRESENT")
    assert "disagree" not in present
    for bad in (
        ("A", "UNKNOWN", 2, True, "PRESENT"),
        ("A", "B", 2, True, "SETTLED"),
        ("A", "B", 2, True, None),
    ):
        with pytest.raises(ValueError):
            MainLoop._prepend_private_interpretation("P", *bad)


def test_legacy_v1_root_restores_but_reads_hold(tmp_path: Path) -> None:
    """D4 §2 / D5 §2: no migration; S2 may still match, S1 does not -> HOLD."""

    for client, refs in _run(
        tmp_path, ("A", "LIKELY_ALIGNED", "LOW"), ("B", "MAY_HAVE_MISREAD", "HIGH")
    ):
        owner = client.app.state.r15_state_port.narrative
        root, continuation = owner._technical_state_for_agent_state()
        later = next(ep for ep in root.episodes if ep.links)
        # The v2 S2 here equals the v1 S2 shape (CONTESTED, 250k).
        assert later.interpretation.status is InterpretationStatus.CONTESTED
        first = next(ep for ep in root.episodes if not ep.links)
        legacy_first = replace(
            first, interpretation=replace(first.interpretation, confidence=250_000)
        )
        legacy = replace(
            root,
            episodes=tuple(
                legacy_first if ep.episode_id == first.episode_id else ep
                for ep in root.episodes
            ),
        )
        legacy_continuation = replace(
            continuation, root_digest=str(legacy.canonical_value()["state_digest"])
        )
        # Shape-only checkpoint validation accepts the legacy confidence.
        checked = NarrativeContinuation.from_value(
            legacy_continuation.canonical_value(legacy), legacy
        )
        assert checked.revision == 2
        assert _private(client).qualified
        # The private re-validation re-derives S1 *and* S2: the S2 matches, the
        # legacy S1 confidence does not -> refused (HOLD), never a stale value.
        gate = owner._source_gate
        with pytest.raises(ContinuityInvalid, match="original interpretation differs"):
            _qualify_source_meaning(
                legacy,
                legacy_continuation,
                gate,
                context_id=legacy_continuation.context_id,
                event_sequence=client.app.state.event_journal.inspect().processing_high_water
                + 1,
            )
        # Technical restore of the legacy root still succeeds (WAL recovery).
        owner._restore_technical_for_agent_state(legacy, legacy_continuation)
        assert owner.snapshot() == legacy
        # The owner root now differs from the committed AgentState: HOLD and
        # UNAVAILABLE for R16.
        assert _private(client) == HOLD
        assert _r16_read(client, refs[1]).outcome == "UNAVAILABLE"
        break


def test_r16_read_is_not_eligible_for_ordinary_turns_and_event_bound(
    tmp_path: Path,
) -> None:
    for client, refs in _run(
        tmp_path, ("A", "LIKELY_ALIGNED", "LOW"), ("B", "MAY_HAVE_MISREAD", "LOW")
    ):
        # The setup turn's own answer is not R15-shaped: NOT_ELIGIBLE.
        assert _r16_read(client, refs[0]).outcome == "NOT_ELIGIBLE"
        # Outside the R16 plan-preview event: UNAVAILABLE.
        outside = _r16_read(
            client,
            refs[1],
            event_type=AgentEventType.CHAT,
            source=AgentEventSource.API_CHAT,
        )
        assert outside.outcome == "UNAVAILABLE"
        # D5 P2-2: called after the event opened an R07 transaction, the source
        # read refuses, so the result is UNAVAILABLE (never a stale value).
        journal = client.app.state.event_journal
        real = journal.inspect()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                journal,
                "inspect",
                lambda: replace(real, open_transactions=(object(),)),
            )
            assert _r16_read(client, refs[1]).outcome == "UNAVAILABLE"
        assert _r16_read(client, refs[1]).outcome == "EPISODE"
        break


def test_r15_shaped_turn_without_episode_is_unavailable(tmp_path: Path) -> None:
    """D5 §3: R15-shaped answer after a _SETUP turn but no episode -> UNAVAILABLE."""

    for client, _refs in _run(
        tmp_path, ("A", "LIKELY_ALIGNED", "LOW"), ("B", "MAY_HAVE_MISREAD", "LOW")
    ):
        provider = client.app.state.model_provider
        setup = (
            "比較用の案は3つです。案Aは「甲」、案Bは「乙」、新案Cは「丙」です。"
            "分類はまだしないでください。"
        )
        assert client.post("/api/chat", json={"message": setup, "attachments": []}).status_code == 200
        provider.generate = lambda _prompt: "暫定的に案Bを選びます"  # type: ignore[method-assign]
        response = client.post("/api/chat", json={"message": "どれにしますか", "attachments": []})
        assert response.status_code == 200
        x = _source_ref(client, response.json()["episode_id"])
        assert _r16_read(client, x).outcome == "UNAVAILABLE"
        break
