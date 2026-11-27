"""Lab-only allocator observation of a real-model ``serve`` child (observation only).

Tests one hypothesis: that the gap between this process's peak ALLOCATED memory and its
RESERVED memory (the allocator's cache) is what leaves the device-wide free memory near zero.

* ``ALLOWED_ALLOC_CONF`` is the closed list of ``PYTORCH_CUDA_ALLOC_CONF`` values a launch may ask
  for.  The value reaches ONLY the child's environment (``supervisor.child_environment``), and only
  when the flag is given.
* ``EmptyCacheProvider`` wraps the provider the app's loader builds.  It reads reserved / free
  memory and, per the mode, calls ``torch.cuda.empty_cache()`` right after the load and (``per-turn``)
  after every model call.  A failing read or ``empty_cache`` is recorded as an error kind and never
  breaks a turn.  It never touches the text.
* One 0600 row of NUMBERS per hook event goes to ``<private dir>/alloc_probe.jsonl``.

OPT-IN: without ``serve --empty-cache`` nothing here is imported or installed.  Nothing under
``suzka/`` is touched; production allocator settings are a separate Human decision.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import threading
from typing import Any

from experiments.r20.guards import GuardRefusal

PROBE_NAME = "alloc_probe.jsonl"
ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
ALLOWED_ALLOC_CONF = (
    "expandable_segments:True",
    "garbage_collection_threshold:0.6",
    "expandable_segments:True,garbage_collection_threshold:0.6",
)
EMPTY_CACHE_MODES = ("none", "after-load", "per-turn")


def check_alloc_conf(value: object) -> str | None:
    """``None`` stays ``None``; anything outside the allow-list is refused."""

    if value is None:
        return None
    if not isinstance(value, str) or value not in ALLOWED_ALLOC_CONF:
        raise GuardRefusal("ALLOC_CONF_NOT_ALLOWED", "--alloc-conf: " + " | ".join(ALLOWED_ALLOC_CONF))
    return value


def check_empty_cache(value: object) -> str:
    if value is None:
        return "none"
    if not isinstance(value, str) or value not in EMPTY_CACHE_MODES:
        raise GuardRefusal("EMPTY_CACHE_MODE_INVALID", "--empty-cache: " + " | ".join(EMPTY_CACHE_MODES))
    return value


def _torch_cuda() -> Any:  # pragma: no cover - needs a GPU
    import torch

    return torch.cuda


class EmptyCacheProvider:
    """Delegates everything to ``inner``; records and (per mode) empties the cache around the calls.

    ``generate_chat`` exists on the wrapper only when the inner provider has it (the app probes it).
    """

    def __init__(self, inner: Any, mode: str, path: Path, cuda: Callable[[], Any] = _torch_cuda) -> None:
        self._inner = inner
        self._mode = check_empty_cache(mode)
        self._path = Path(path)
        self._cuda = cuda
        self._lock = threading.Lock()
        self._sequence = 0

    def __getattr__(self, name: str) -> Any:
        if name in {"_inner", "_mode", "_path", "_cuda", "_lock", "_sequence"}:
            raise AttributeError(name)
        target = getattr(self._inner, name)
        return self._generate_chat if name == "generate_chat" else target

    @staticmethod
    def _read(cuda: Any) -> tuple[int | None, int | None]:
        """(reserved, device-wide free via mem_get_info) now; ``None`` for a reading that failed."""

        try:
            reserved: int | None = int(cuda.memory_reserved())
        except Exception:  # noqa: BLE001 - a missing reading is a recorded gap
            reserved = None
        try:
            free: int | None = int(cuda.mem_get_info()[0])
        except Exception:  # noqa: BLE001
            free = None
        return reserved, free

    def observe(self, event: str, *, empty: bool) -> None:
        """One row; never raises."""

        row: dict[str, Any] = {"event": event, "mode": self._mode, "empty_cache_called": False, "error": None}
        try:
            cuda = self._cuda()
            before, free_before = self._read(cuda)
            error: str | None = None
            called = False
            if empty:
                try:
                    cuda.empty_cache()
                    called = True
                except Exception as failure:  # noqa: BLE001 - a failed empty_cache must not crash a turn
                    error = type(failure).__name__
            after, free_after = self._read(cuda)
            row.update(
                empty_cache_called=called, error=error, reserved_before_bytes=before, reserved_after_bytes=after,
                free_before_bytes=free_before, free_after_bytes=free_after,
            )
        except Exception as failure:  # noqa: BLE001
            row["error"] = type(failure).__name__
        try:
            with self._lock:
                self._sequence += 1
                descriptor = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"seq": self._sequence, **row}, sort_keys=True) + "\n")
        except OSError:
            pass

    def _after_call(self) -> None:
        self.observe("per_turn", empty=self._mode == "per-turn")

    def generate(self, prompt: str) -> str:
        try:
            return str(self._inner.generate(prompt))
        finally:
            self._after_call()

    def _generate_chat(self, system_text: str, user_text: str) -> str:
        try:
            return str(self._inner.generate_chat(system_text, user_text))
        finally:
            self._after_call()


def wrap_loader(
    loader: Callable[..., Any], mode: str, private_dir: Path, cuda: Callable[[], Any] = _torch_cuda,
) -> Callable[..., Any]:
    """A drop-in for ``load_model_provider``: reads (and for after-load / per-turn empties) right after the load."""

    def load(*arguments: Any, **keywords: Any) -> EmptyCacheProvider:
        provider = EmptyCacheProvider(loader(*arguments, **keywords), mode, Path(private_dir) / PROBE_NAME, cuda)
        provider.observe("after_load", empty=mode in {"after-load", "per-turn"})
        return provider

    return load


def read_rows(private_dir: Path) -> list[dict[str, Any]]:
    path = Path(private_dir) / PROBE_NAME
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Numbers only: the calls, the errors, reserved after the load, the lowest device-wide free seen."""

    loads = [row for row in rows if row.get("event") == "after_load"]
    free = [row[key] for row in rows for key in ("free_before_bytes", "free_after_bytes") if isinstance(row.get(key), int)]
    reserved = [
        row[key] for row in rows for key in ("reserved_before_bytes", "reserved_after_bytes") if isinstance(row.get(key), int)
    ]
    return {
        "empty_cache_calls": sum(1 for row in rows if row.get("empty_cache_called") is True),
        "empty_cache_errors": sum(1 for row in rows if row.get("error")),
        "reserved_after_load_bytes": loads[0].get("reserved_after_bytes") if loads else None,
        "probe_max_reserved_bytes": max(reserved) if reserved else None,
        "probe_min_free_bytes": min(free) if free else None,
        "calls": [
            {key: row.get(key) for key in ("event", "empty_cache_called", "reserved_before_bytes", "reserved_after_bytes")}
            for row in rows
        ],
    }


__all__ = [
    "ALLOWED_ALLOC_CONF", "ALLOC_CONF_ENV", "EMPTY_CACHE_MODES", "EmptyCacheProvider", "PROBE_NAME",
    "check_alloc_conf", "check_empty_cache", "read_rows", "summarize", "wrap_loader",
]
