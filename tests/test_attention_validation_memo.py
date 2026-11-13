"""IP-1b: memoized Attention continuity validation is exact.

One AgentState commit validates the same Attention continuity many times.
The memo is keyed by the exact canonical JSON text and requires exact JSON
literal types, so it only ever returns what full validation would return
for that very input.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from suzka.attention.contracts import AttentionContinuity
from suzka.runtime import attention_state_codec as codec
from suzka.runtime.attention_state_codec import validated_attention_continuity
from tests.test_agent_state_v9 import _continuity


@pytest.fixture(autouse=True)
def _fresh_memo():
    codec._validated_memo.clear()
    yield
    codec._validated_memo.clear()


@pytest.fixture
def continuity() -> AttentionContinuity:
    return _continuity()


def _count_full(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    original = AttentionContinuity.from_canonical_value.__func__  # type: ignore[attr-defined]

    def counted(cls, value):  # type: ignore[no-untyped-def]
        calls[0] += 1
        return original(cls, value)

    monkeypatch.setattr(AttentionContinuity, "from_canonical_value", classmethod(counted))
    return calls


def test_repeated_validation_is_served_from_the_memo(
    continuity: AttentionContinuity, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_full(monkeypatch)
    canonical = continuity.canonical_value()
    first = validated_attention_continuity(canonical)
    assert calls[0] == 1
    again = validated_attention_continuity(deepcopy(canonical))
    typed = validated_attention_continuity(first)
    assert calls[0] == 1
    assert again is first and typed is first
    assert first == continuity


def test_the_memo_never_accepts_what_full_validation_rejects(
    continuity: AttentionContinuity,
) -> None:
    canonical = continuity.canonical_value()
    validated_attention_continuity(canonical)

    # A tuple where the canonical form has a list: same JSON text, but not
    # an exact literal, so it is validated in full and refused as before.
    tupled = dict(canonical)
    tupled["receipts"] = tuple(canonical["receipts"])  # type: ignore[arg-type]
    assert codec._memo_key(tupled) is None
    with pytest.raises((TypeError, ValueError)):
        validated_attention_continuity(tupled)

    # Changed content with stale digests: a different key, refused in full.
    changed = deepcopy(canonical)
    changed["revision"] = int(canonical["revision"]) + 1  # type: ignore[call-overload]
    with pytest.raises((TypeError, ValueError)):
        validated_attention_continuity(changed)

    # A bool where an integer is expected encodes differently in JSON.
    flagged = deepcopy(canonical)
    flagged["schema_version"] = True
    with pytest.raises((TypeError, ValueError)):
        validated_attention_continuity(flagged)


def test_a_tampered_typed_value_is_validated_in_full(
    continuity: AttentionContinuity, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated_attention_continuity(continuity)
    tampered = replace(continuity)
    object.__setattr__(tampered, "focused_ids", ("forged-id",))
    expected: object
    try:
        codec._validated_memo.clear()
        expected = validated_attention_continuity(tampered)
    except (TypeError, ValueError) as error:
        expected = type(error)
    codec._validated_memo.clear()
    validated_attention_continuity(continuity)
    calls = _count_full(monkeypatch)
    try:
        actual: object = validated_attention_continuity(tampered)
    except (TypeError, ValueError) as error:
        actual = type(error)
    assert calls[0] == 1
    assert actual == expected


def test_the_memo_is_bounded(continuity: AttentionContinuity) -> None:
    for index in range(codec._VALIDATED_MEMO_SIZE):
        codec._validated_memo[json.dumps({"probe": index})] = continuity
    validated_attention_continuity(continuity.canonical_value())
    assert len(codec._validated_memo) == codec._VALIDATED_MEMO_SIZE
    assert json.dumps({"probe": 0}) not in codec._validated_memo
