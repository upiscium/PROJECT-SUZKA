"""The child-side GPU sampler of a real-model ``serve`` (IP-23 sec. 2).

A daemon thread appends one JSON line of NUMBERS ONLY (time, device-wide free /
total, this process's reserved bytes) to ``evidence/gpu.jsonl`` before the
server starts and then every ``INTERVAL_S`` seconds, so the model load, every
turn and any supervisor cycle show in the log.  It never touches the model.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import threading
import time

INTERVAL_S = 5.0
Reader = Callable[[], tuple[int, int, int]]


def torch_reader() -> tuple[int, int, int]:  # pragma: no cover - needs a GPU
    import torch

    free, total = torch.cuda.mem_get_info()
    return int(free), int(total), int(torch.cuda.memory_reserved())


class GpuSampler:
    def __init__(self, path: Path, reader: Reader = torch_reader, interval_s: float = INTERVAL_S) -> None:
        self.path = Path(path)
        self._reader = reader
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self, label: str) -> None:
        try:
            free, total, reserved = self._reader()
            row: dict[str, object] = {"free": free, "total": total, "reserved": reserved}
        except Exception as error:  # a failed read is a recorded gap, never a crash
            row = {"error": type(error).__name__}
        row.update({"t": round(time.time(), 3), "pid": os.getpid(), "label": label})
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def start(self) -> None:
        self.sample("before_load")
        self._thread = threading.Thread(target=self._loop, name="r20-gpu", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample("tick")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.sample("stopped")


def read_log(path: Path) -> list[dict[str, object]]:
    if not Path(path).is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Numbers for the report: free before load, minimum free seen, peak reserved."""

    good = [row for row in rows if "free" in row]
    if not good:
        return {"samples": len(rows), "readable": 0}
    first = good[0]
    return {
        "samples": len(rows),
        "readable": len(good),
        "free_before_load": first["free"],
        "min_free": min(int(str(row["free"])) for row in good),
        "peak_reserved": max(int(str(row["reserved"])) for row in good),
        "total": first["total"],
    }


__all__ = ["GpuSampler", "read_log", "summarize"]
