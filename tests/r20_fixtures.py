"""Shared helpers for the IP-22a tests (no test here; imported by them)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import shutil
import tempfile

import pytest

from experiments.r20 import factory
from experiments.r20.guards import BASE_ENV


@pytest.fixture
def r20_base(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A short throwaway base under /tmp (AF_UNIX paths are length-limited);
    the inherited variables the factory refuses are removed."""

    for name in ("SUZKA_CONFIG_PATH", "PYTHONPATH", BASE_ENV):
        monkeypatch.delenv(name, raising=False)
    base = Path(tempfile.mkdtemp(prefix="r20t-", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def started_record(subject: factory.GenesisSubject, runtime: object) -> factory.IdentityRecord:
    provider = getattr(runtime, "provider", None)
    identity = getattr(provider, "identity", {}) if provider is not None else {}
    return factory.verify_first_start(subject, provider=identity)
