"""The model-configuration override of a lab subject (HD-23-8).

The factory's default stays ``provider: dummy`` (the repository config).  A real
run names its model EXPLICITLY: the model id is an argument of the Human's
launch, never read from a production host, ``/etc`` or the repository's own
choice, and the HF cache is an explicit absolute directory.  The revision
digest is read (read-only) from that cache's ``refs/main`` and checked against
the digest the Human pinned; the weight / revision actually loaded is recorded
in the manifest when the Human-launched run loads it.

This module imports no ML library: package versions come from
``importlib.metadata``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
from typing import Any

from experiments.r20.guards import GuardRefusal

# ``org/name``: one slash, no traversal; the same shape as an HF repo id.
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_REVISION = re.compile(r"[0-9a-f]{40}")
PROVIDERS = ("dummy", "transformers")
MAX_TEMPERATURE = 1.5
VERSIONED_PACKAGES = ("torch", "transformers", "bitsandbytes", "accelerate", "peft")


@dataclass(frozen=True, slots=True)
class ModelOverride:
    """What a subject's ``model`` / ``generation`` sections are overridden with.

    ``provider = "dummy"`` is the default and changes nothing.  A real model
    needs an explicit ``model_id`` and (for the Human-launched path) an
    explicit ``hf_home`` and a pinned ``revision`` (the 40-hex commit digest).
    """

    provider: str = "dummy"
    model_id: str | None = None
    load_in_4bit: bool = True
    max_new_tokens: int | None = None
    device: str = "auto"
    dtype: str = "auto"
    hf_home: Path | None = None
    revision: str | None = None
    # Lab override of generation.temperature; 0.0 means greedy (do_sample false); None keeps the config.
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise GuardRefusal("MODEL_PROVIDER_UNKNOWN", self.provider)
        if self.temperature is not None and not 0.0 <= self.temperature <= MAX_TEMPERATURE:
            raise GuardRefusal("MODEL_TEMPERATURE", f"0.0..{MAX_TEMPERATURE}")
        if self.max_new_tokens is not None and not 1 <= self.max_new_tokens <= 4096:
            raise GuardRefusal("MODEL_MAX_NEW_TOKENS", str(self.max_new_tokens))
        if self.provider == "dummy":
            return
        if self.model_id is None or not _MODEL_ID.fullmatch(self.model_id):
            raise GuardRefusal("MODEL_ID_REQUIRED", "an explicit org/name model id")
        if self.revision is not None and not _REVISION.fullmatch(self.revision):
            raise GuardRefusal("MODEL_REVISION_INVALID", "a 40-hex commit digest")
        if self.hf_home is not None:
            text = str(self.hf_home)
            if text.startswith("~") or not os.path.isabs(text):
                raise GuardRefusal("HF_HOME_NOT_ABSOLUTE", text)

    @property
    def is_real(self) -> bool:
        return self.provider == "transformers"

    def apply(self, raw: dict[str, Any]) -> None:
        """Override the ``model`` and ``generation`` sections of a raw config."""

        if not self.is_real:
            return
        assert self.model_id is not None
        model = raw.setdefault("model", {})
        model.update(
            {
                "provider": "transformers",
                "primary_id": self.model_id,
                "fallback_id": self.model_id,  # no silent fallback to another model
                "device": self.device,
                "dtype": self.dtype,
                "load_in_4bit": self.load_in_4bit,
            }
        )
        if self.max_new_tokens is not None:
            raw.setdefault("generation", {})["max_new_tokens"] = self.max_new_tokens
        if self.temperature is not None:
            generation = raw.setdefault("generation", {})
            if self.temperature == 0.0:
                generation["do_sample"] = False  # greedy; the provider then passes no temperature / top_p
            else:
                generation["do_sample"] = True
                generation["temperature"] = self.temperature

    def record(self) -> dict[str, object]:
        """Identifiers only (the manifest / identity ``provider`` entry)."""

        return {
            "mode": "real" if self.is_real else "dummy",
            "provider": self.provider,
            "model_id": self.model_id,
            "revision_pinned": self.revision,
            "load_in_4bit": self.load_in_4bit,
            "device": self.device,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature_override": self.temperature,  # None = the config's own value
            "do_sample_override": None if self.temperature is None else self.temperature != 0.0,
            "hf_home": None if self.hf_home is None else str(self.hf_home),
            "fixture_only": not self.is_real,
        }


# The ONLY config keys a lab caller may override beyond the model section (an allow-list, never a
# free-form mapping).  Every other value of the section stays the repository default.
ALLOWED_CONFIG_OVERRIDES: dict[str, type] = {"conversation_trace.enabled": bool}


def validate_config_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """The overrides as a plain dict; any key outside the allow-list or any wrong type is refused."""

    checked: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        wanted = ALLOWED_CONFIG_OVERRIDES.get(key)
        if wanted is None:
            raise GuardRefusal("CONFIG_OVERRIDE_NOT_ALLOWED", str(key))
        if type(value) is not wanted:
            raise GuardRefusal("CONFIG_OVERRIDE_TYPE", str(key))
        checked[key] = value
    return checked


def apply_config_overrides(raw: dict[str, Any], overrides: Mapping[str, Any] | None) -> None:
    """Apply validated overrides to a raw config (``a.b`` sets ``raw['a']['b']``)."""

    for key, value in validate_config_overrides(overrides).items():
        section, name = key.split(".")
        raw.setdefault(section, {})[name] = value


def cache_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def cached_revision(hf_home: Path, model_id: str) -> str:
    """The commit digest the cache's ``refs/main`` names (a read-only file
    read; nothing is downloaded and nothing is written)."""

    refs = Path(hf_home) / "hub" / cache_dir_name(model_id) / "refs" / "main"
    try:
        text = refs.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise GuardRefusal("MODEL_NOT_IN_CACHE", model_id) from error
    if not _REVISION.fullmatch(text):
        raise GuardRefusal("MODEL_REVISION_UNREADABLE", model_id)
    snapshot = Path(hf_home) / "hub" / cache_dir_name(model_id) / "snapshots" / text
    if not snapshot.is_dir():
        raise GuardRefusal("MODEL_SNAPSHOT_MISSING", model_id)
    return text


def check_pinned_revision(model: ModelOverride) -> str:
    """The cached revision must equal the revision the Human pinned (HD-23-8)."""

    if not model.is_real:
        raise GuardRefusal("MODEL_NOT_REAL")
    if model.hf_home is None or model.revision is None:
        raise GuardRefusal("MODEL_PIN_REQUIRED", "--hf-home and --model-revision")
    assert model.model_id is not None
    found = cached_revision(model.hf_home, model.model_id)
    if found != model.revision:
        raise GuardRefusal("MODEL_REVISION_MISMATCH", f"cache {found[:12]} != pinned {model.revision[:12]}")
    return found


def model_digest(model: ModelOverride) -> str:
    """The digest a Human launch binds: model id, pinned revision and the 4-bit flag (identifiers only)."""

    payload = json.dumps(
        {"model_id": model.model_id, "revision": model.revision, "load_in_4bit": model.load_in_4bit},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


__all__ = [
    "ALLOWED_CONFIG_OVERRIDES",
    "apply_config_overrides",
    "validate_config_overrides",
    "PROVIDERS",
    "ModelOverride",
    "cache_dir_name",
    "cached_revision",
    "check_pinned_revision",
    "model_digest",
    "package_versions",
]
