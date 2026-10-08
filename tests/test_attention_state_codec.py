"""Bounded R14 Attention persistence codec tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import suzka.runtime.attention_state_codec as codec
from suzka.attention.bounds import (
    derive_attention_schema_size_budget,
    maximum_attention_continuity_fixture,
)
from suzka.attention.contracts import AttentionContinuity


def _attention_root(**overrides: object) -> dict[str, object]:
    value = AttentionContinuity.bootstrap().canonical_value()
    value.update(overrides)
    return value


def _agent_state(attention_json: bytes | None = None) -> bytes:
    encoded = (
        AttentionContinuity.bootstrap().canonical_bytes()
        if attention_json is None
        else attention_json
    )
    return b'{"schema_version":9,"attention_state":' + encoded + b"}"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")


def _legacy_v8_snapshot(tmp_path: Path) -> bytes:
    from test_agent_state_v8 import _capture_v8, _store

    snapshot, _ = _capture_v8(_store(tmp_path / "legacy-v8.json"), populated=False)
    return json.dumps(
        snapshot.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def test_derived_runtime_field_bound_is_the_exact_attention_schema_budget() -> None:
    assert codec.AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES == 4_193_496
    assert codec.AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES == (
        derive_attention_schema_size_budget().attention_state_max_bytes
    )


def test_typed_and_canonical_dict_inputs_are_detached_checksum_checked_copies() -> None:
    original = AttentionContinuity.bootstrap()
    copied_typed = codec.validated_attention_continuity(original)
    copied_dict = codec.validated_attention_continuity(original.canonical_value())

    assert copied_typed is not original
    assert copied_dict is not original
    assert copied_typed.canonical_bytes() == original.canonical_bytes()
    assert copied_dict.canonical_bytes() == original.canonical_bytes()

    tampered = original.canonical_value()
    tampered["authority_digest"] = "0" * 64
    with pytest.raises(ValueError, match="authority digest"):
        codec.validated_attention_continuity(tampered)


def test_full_legal_candidate_universe_is_preserved_without_truncation() -> None:
    full = maximum_attention_continuity_fixture()
    copied = codec.validated_attention_continuity(full)

    assert len(copied.candidates) == 4_192
    assert len(copied.revision_history) == 16
    assert len(copied.receipts) == 256
    assert copied.canonical_bytes() == full.canonical_bytes()
    assert len(copied.canonical_bytes()) == codec.AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES
    copied_mapping = codec.validated_attention_continuity(full.canonical_value())
    assert copied_mapping.canonical_bytes() == full.canonical_bytes()
    codec.preflight_attention_json(_agent_state(full.canonical_bytes()))


def test_programmatic_mapping_count_bounds_precede_nested_value_parsing() -> None:
    too_many = _attention_root(candidates=[None] * 4_193)
    with pytest.raises(ValueError, match="candidates exceeds"):
        codec.validated_attention_continuity(too_many)

    per_kind = [
        {"target": {"kind": "working_memory"}}
        for _ in range(4_097)
    ]
    with pytest.raises(ValueError, match="source-authority bound"):
        codec.validated_attention_continuity(_attention_root(candidates=per_kind))


def test_v9_raw_attention_byte_limit_precedes_attention_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_decoder(cls: type[AttentionContinuity], value: bytes | str) -> AttentionContinuity:
        raise AssertionError("oversized Attention field reached the decoder")

    monkeypatch.setattr(AttentionContinuity, "from_json", classmethod(fail_decoder))
    oversized_value = b'"' + b"a" * (
        codec.AGENT_STATE_ATTENTION_MAX_SERIALIZED_BYTES + 1
    ) + b'"'
    with pytest.raises(ValueError, match="field byte bound"):
        codec.preflight_attention_json(_agent_state(oversized_value))


def test_raw_array_caps_precede_decoder_and_row_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_decoder(cls: type[AttentionContinuity], value: bytes | str) -> AttentionContinuity:
        raise AssertionError("over-cap Attention array reached the decoder")

    monkeypatch.setattr(AttentionContinuity, "from_json", classmethod(fail_decoder))
    root = _attention_root(candidates=[None] * 4_193)
    raw_attention = _json_bytes(root)

    with pytest.raises(ValueError, match="array exceeds"):
        codec.preflight_attention_json(_agent_state(raw_attention))


def test_raw_attention_root_is_closed_bounded_and_strict_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_decoder(cls: type[AttentionContinuity], value: bytes | str) -> AttentionContinuity:
        raise AssertionError("invalid Attention root reached the decoder")

    monkeypatch.setattr(AttentionContinuity, "from_json", classmethod(fail_decoder))
    with pytest.raises(ValueError, match="unknown field"):
        codec.preflight_attention_json(
            _agent_state(_json_bytes(_attention_root(unrecognized=None)))
        )

    non_integer_schema = _attention_root(schema_version=1.0)
    with pytest.raises(TypeError, match="wrong literal type|cannot be floats"):
        codec.preflight_attention_json(_agent_state(_json_bytes(non_integer_schema)))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":9,"schema_version":8,"attention_state":{}}',
        b'{"schema_version":8,"schema_version":9,"attention_state":{}}',
    ],
)
def test_v9_duplicate_root_schema_version_cannot_be_downgraded(raw: bytes) -> None:
    with pytest.raises(ValueError, match="duplicate schema_version"):
        codec.preflight_attention_json(raw)


@pytest.mark.parametrize(
    "token",
    [
        b"9.0",
        b"9e0",
        b"90e-1",
        b"9.0000000000000000001",
        b"8.9999999999999999999",
        b"0.89999999999999999999e1",
    ],
)
def test_float_tokens_rounded_to_nine_are_rejected_before_attention_decode(
    monkeypatch: pytest.MonkeyPatch,
    token: bytes,
) -> None:
    assert float(token) == 9.0

    def fail_decoder(cls: type[AttentionContinuity], value: bytes | str) -> AttentionContinuity:
        raise AssertionError("rounded v9 discriminator reached Attention decoding")

    monkeypatch.setattr(AttentionContinuity, "from_json", classmethod(fail_decoder))
    raw = (
        b'{"schema_version":'
        + token
        + b',"schema_version":8,"attention_state":{}}'
    )
    with pytest.raises(ValueError, match="exact integer literal 9"):
        codec.preflight_attention_json(raw)


@pytest.mark.parametrize("record_type_field", [
    ("baseline", "baseline_snapshot"),
    ("transition", "candidate_snapshot"),
])
def test_wal_snapshots_reject_float_tokens_rounded_to_v9_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    record_type_field: tuple[str, str],
) -> None:
    def fail_decoder(cls: type[AttentionContinuity], value: bytes | str) -> AttentionContinuity:
        raise AssertionError("rounded WAL v9 discriminator reached Attention decoding")

    monkeypatch.setattr(AttentionContinuity, "from_json", classmethod(fail_decoder))
    record_type, field = record_type_field
    raw = (
        b'{"record_type":"'
        + record_type.encode("ascii")
        + b'","'
        + field.encode("ascii")
        + b'":{"schema_version":9.0000000000000000001,'
        + b'"schema_version":8,"attention_state":{}}}'
    )
    with pytest.raises(ValueError, match="exact integer literal 9"):
        codec.preflight_attention_json(raw)


@pytest.mark.parametrize("token", [b"1.0", b"4.0", b"8e0", b"-0", b"9e999"])
def test_non_v9_float_and_negative_zero_legacy_discriminators_are_not_reclassified(
    token: bytes,
) -> None:
    raw = b'{"schema_version":' + token + b',"schema_version":8}'
    codec.preflight_attention_json(raw)


def test_v9_requires_one_actual_attention_key_and_detects_escaped_duplicates() -> None:
    with pytest.raises(ValueError, match="requires exactly one"):
        codec.preflight_attention_json(b'{"schema_version":9}')

    state = AttentionContinuity.bootstrap().canonical_bytes()
    duplicate = (
        b'{"schema_version":9,"attention_state":'
        + state
        + b',"\\u0061ttention_state":'
        + state
        + b"}"
    )
    with pytest.raises(ValueError, match="requires exactly one"):
        codec.preflight_attention_json(duplicate)


def test_escaped_attention_key_is_recognized_but_quoted_text_is_not_a_key() -> None:
    state = AttentionContinuity.bootstrap().canonical_bytes()
    escaped = (
        b'{"schema_version":9,"\\u0061ttention_state":'
        + state
        + b"}"
    )
    codec.preflight_attention_json(escaped)

    old_state = b'{"schema_version":8,"payload":"attention_state"}'
    codec.preflight_attention_json(old_state)
    with pytest.raises(ValueError, match="requires exactly one"):
        codec.preflight_attention_json(
            b'{"schema_version":9,"payload":"attention_state"}'
        )


def test_legacy_duplicate_policy_is_not_changed_for_schema_versions_below_nine() -> None:
    codec.preflight_attention_json(b'{"schema_version":8,"schema_version":8}')


def test_wal_preflight_checks_both_embedded_snapshot_fields() -> None:
    snapshot = _agent_state().removeprefix(b'{"schema_version":9,"attention_state":')[:-1]
    embedded = b'{"schema_version":9,"attention_state":' + snapshot + b"}"
    wal = (
        b'{"record_type":"baseline","baseline_snapshot":'
        + embedded
        + b',"candidate_snapshot":'
        + embedded
        + b"}"
    )
    codec.preflight_attention_json(wal)

@pytest.mark.parametrize(
    ("record_type", "field"),
    [("baseline", "baseline_snapshot"), ("transition", "candidate_snapshot")],
)
def test_wal_embedded_v9_duplicates_are_detected_before_record_decoding(
    record_type: str,
    field: str,
) -> None:
    broken = (
        b'{"record_type":"'
        + record_type.encode("ascii")
        + b'","'
        + field.encode("ascii")
        + b'":{"schema_version":9,"schema_version":8,"attention_state":{}}}'
    )
    with pytest.raises(ValueError, match="duplicate schema_version"):
        codec.preflight_attention_json(broken)


@pytest.mark.parametrize(
    ("record_type", "field"),
    [("baseline", "baseline_snapshot"), ("transition", "candidate_snapshot")],
)
@pytest.mark.parametrize("v9_first", [True, False])
@pytest.mark.parametrize("escaped_first", [True, False])
def test_duplicate_wal_snapshot_envelopes_cannot_hide_a_v9_value(
    tmp_path: Path,
    record_type: str,
    field: str,
    v9_first: bool,
    escaped_first: bool,
) -> None:
    field_bytes = field.encode("ascii")
    escaped_field = (
        b"\\u0062aseline_snapshot"
        if field == "baseline_snapshot"
        else b"\\u0063andidate_snapshot"
    )
    first_key = escaped_field if escaped_first else field_bytes
    second_key = field_bytes if escaped_first else escaped_field
    v9_snapshot = _agent_state()
    v8_snapshot = _legacy_v8_snapshot(tmp_path)
    first_snapshot, second_snapshot = (
        (v9_snapshot, v8_snapshot) if v9_first else (v8_snapshot, v9_snapshot)
    )
    raw = (
        b'{"record_type":"'
        + record_type.encode("ascii")
        + b'","'
        + first_key
        + b'":'
        + first_snapshot
        + b',"'
        + second_key
        + b'":'
        + second_snapshot
        + b"}"
    )

    with pytest.raises(ValueError, match=f"duplicate v9 {field} envelope"):
        codec.preflight_attention_json(raw)


def test_duplicate_legacy_wal_snapshot_envelopes_keep_legacy_policy(
    tmp_path: Path,
) -> None:
    from suzka.runtime.agent_state import AgentStateSnapshotV8

    legacy = _legacy_v8_snapshot(tmp_path)
    for field, escaped in (
        ("baseline_snapshot", b"\\u0062aseline_snapshot"),
        ("candidate_snapshot", b"\\u0063andidate_snapshot"),
    ):
        raw = (
            b'{"record_type":"baseline","'
            + field.encode("ascii")
            + b'":'
            + legacy
            + b',"'
            + escaped
            + b'":'
            + legacy
            + b"}"
        )
        codec.preflight_attention_json(raw)
        parsed = json.loads(raw)
        assert AgentStateSnapshotV8.model_validate(parsed[field]).schema_version == 8


def test_attention_nesting_and_malformed_outer_json_fail_closed_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_decoder(cls: type[AttentionContinuity], value: bytes | str) -> AttentionContinuity:
        raise AssertionError("invalid raw Attention reached the decoder")

    monkeypatch.setattr(AttentionContinuity, "from_json", classmethod(fail_decoder))
    nested: object = None
    for _ in range(70):
        nested = {"x": nested}
    too_deep = _attention_root(last_event=nested)
    with pytest.raises(ValueError, match="nesting bound"):
        codec.preflight_attention_json(_agent_state(_json_bytes(too_deep)))

    with pytest.raises(ValueError, match="malformed|object key"):
        codec.preflight_attention_json(b'{"schema_version":9,')


def test_outer_scanner_keeps_legacy_depth_and_has_a_deterministic_cap() -> None:
    legacy_depth = 1_100
    payload = (
        b'{"schema_version":8,"payload":'
        + b"[" * legacy_depth
        + b"null"
        + b"]" * legacy_depth
        + b"}"
    )
    codec.preflight_attention_json(payload)

    over_limit = codec._MAX_RAW_JSON_NESTING + 1
    deeply_nested = b"[" * over_limit + b"null" + b"]" * over_limit
    with pytest.raises(ValueError, match="scanner bound"):
        codec.preflight_attention_json(deeply_nested)
