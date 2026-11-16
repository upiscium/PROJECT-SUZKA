"""IP-16a-1 (#411 IP-16 D1 r3 §3a, §8-1, §8-2): obligation identity and target_init."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from suzka.config.schema import R18Settings
from suzka.r18.obligation import (
    SECRET_BYTES,
    SECRET_FILE,
    ObligationSecretError,
    create_obligation_secret,
    load_obligation_secret,
    normalize_request_text,
    obligation_identity,
)
from suzka.r18.target import TargetCorrupt, TargetStore
from suzka.r18.target_init import initialize, main

SECRET = bytes(range(SECRET_BYTES))
SUBJECT = "subject-1"
SCOPE = ("review-artifact", "v1")


def _private(tmp_path: Path) -> Path:
    directory = tmp_path / "r18"
    directory.mkdir(mode=0o700)
    return directory


def _identity(text: str, *, secret: bytes = SECRET, context: str = "context-1", scope: tuple[str, ...] = SCOPE):
    return obligation_identity(secret, subject_id=SUBJECT, context_id=context, request_text=text, scope=scope)


def test_normalization_is_nfc_and_whitespace_only() -> None:
    assert normalize_request_text("  á\tb \n c  ") == "á b c"
    assert normalize_request_text("Review") != normalize_request_text("review")
    for bad in ("", " \n\t "):
        with pytest.raises(ValueError):
            normalize_request_text(bad)
    with pytest.raises(ValueError):
        normalize_request_text("x" * 16_385)
    with pytest.raises(TypeError):
        normalize_request_text(b"x")  # type: ignore[arg-type]


def test_same_request_same_context_is_one_obligation() -> None:
    first = _identity("Please review  the draft.")
    assert first == _identity("Please review the draft.\n")
    assert first.experience_content_digest != _identity("Please review the draft!").experience_content_digest
    assert first.group_key != _identity("Please review the draft.", context="context-2").group_key
    assert first.group_key != _identity("Please review the draft.", scope=("other",)).group_key
    assert first.group_key != _identity("Please review the draft.", secret=bytes(SECRET_BYTES)).group_key
    # The text never appears in the identity.
    assert "review" not in repr(first)


def test_identity_refuses_bad_inputs() -> None:
    with pytest.raises(ObligationSecretError):
        _identity("x", secret=b"short")
    with pytest.raises(ValueError):
        _identity("x", scope=())
    with pytest.raises(ValueError):
        _identity("x", scope=("",))


def test_secret_is_created_once_and_loaded_only_with_private_custody(tmp_path: Path) -> None:
    directory = _private(tmp_path)
    with pytest.raises(ObligationSecretError, match="missing"):
        load_obligation_secret(directory)
    path = create_obligation_secret(directory)
    assert path == directory / SECRET_FILE
    assert os.stat(path).st_mode & 0o777 == 0o600
    secret = load_obligation_secret(directory)
    assert len(secret) == SECRET_BYTES
    with pytest.raises(ObligationSecretError, match="never replaced"):
        create_obligation_secret(directory)
    assert load_obligation_secret(directory) == secret

    os.chmod(path, 0o644)
    with pytest.raises(ObligationSecretError, match="custody"):
        load_obligation_secret(directory)
    os.chmod(path, 0o600)
    os.link(path, directory / "extra-link")
    with pytest.raises(ObligationSecretError, match="custody"):
        load_obligation_secret(directory)
    os.unlink(directory / "extra-link")
    os.chmod(directory, 0o750)
    assert load_obligation_secret(directory) == secret
    os.chmod(directory, 0o770)
    with pytest.raises(ObligationSecretError, match="directory"):
        load_obligation_secret(directory)


def test_a_symlinked_or_truncated_secret_is_refused(tmp_path: Path) -> None:
    directory = _private(tmp_path)
    elsewhere = tmp_path / "elsewhere.key"
    elsewhere.write_bytes(SECRET)
    os.chmod(elsewhere, 0o600)
    os.symlink(elsewhere, directory / SECRET_FILE)
    with pytest.raises(ObligationSecretError, match="custody"):
        load_obligation_secret(directory)
    os.unlink(directory / SECRET_FILE)
    short = directory / SECRET_FILE
    short.write_bytes(SECRET[:-1])
    os.chmod(short, 0o600)
    with pytest.raises(ObligationSecretError, match="custody"):
        load_obligation_secret(directory)


def test_target_init_creates_both_and_refuses_a_second_run(tmp_path: Path) -> None:
    directory = _private(tmp_path)
    path = directory / "target.sqlite3"
    instance, secret = initialize(path)
    assert secret == directory / SECRET_FILE
    store = TargetStore.open(path)
    try:
        assert store.meta().target_instance_id == instance
    finally:
        store.close()
    assert len(load_obligation_secret(directory)) == SECRET_BYTES

    with pytest.raises(ObligationSecretError):
        initialize(path)
    os.unlink(secret)
    with pytest.raises(TargetCorrupt):
        initialize(path)
    # Neither run replaced anything that existed.
    assert not (directory / SECRET_FILE).exists()
    store = TargetStore.open(path)
    try:
        assert store.meta().target_instance_id == instance
    finally:
        store.close()


def test_target_init_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = _private(tmp_path)
    assert main([]) == 2
    assert main(["relative/target.sqlite3"]) == 1
    assert main([str(directory / "target.sqlite3")]) == 0
    assert "created target" in capsys.readouterr().out
    assert main([str(directory / "target.sqlite3")]) == 1
    assert "refused" in capsys.readouterr().err


def test_target_init_refuses_a_shared_directory(tmp_path: Path) -> None:
    directory = tmp_path / "shared"
    directory.mkdir()
    os.chmod(directory, 0o775)
    assert main([str(directory / "target.sqlite3")]) == 1
    assert list(directory.iterdir()) == []


def test_the_target_path_setting_defaults_under_private() -> None:
    assert R18Settings().target_path == Path(".suzka/private/r18/target.sqlite3")
    assert R18Settings().enabled is False
