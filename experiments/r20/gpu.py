"""GPU observation without NVML (IP-23 sec. 2 "GPU probe").

``nvidia-smi`` is never used (NVML is broken on the target host).  Free and
total device memory come from ``torch.cuda.mem_get_info()`` (the CUDA runtime),
read in a SHORT-LIVED subprocess so the launcher never holds a CUDA context.
``mem_get_info`` is device-wide: "free before our load" is what production
leaves us.  Numbers only; nothing here loads a model.

``MIN_FREE_BYTES`` is the Human's cap (HD-23-2: free >= 14 GB at launch).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
import subprocess
import sys

from experiments.r20.guards import GuardRefusal

GIB = 1024**3
MIN_FREE_BYTES = 14 * GIB
PROBE_TIMEOUT_S = 120.0
# a CUDA runtime on NixOS may need these; no secret, no SUZKA_* variable
GPU_ENV = ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")

PROBE_CODE = (
    "import json,sys\n"
    "try:\n"
    "    import torch\n"
    "    if not torch.cuda.is_available():\n"
    "        print(json.dumps({'error': 'CUDA_UNAVAILABLE'})); sys.exit(0)\n"
    "    free, total = torch.cuda.mem_get_info()\n"
    "    print(json.dumps({'free': int(free), 'total': int(total), 'reserved': int(torch.cuda.memory_reserved())}))\n"
    "except Exception as error:\n"
    "    print(json.dumps({'error': 'CUDA_PROBE_FAILED', 'kind': type(error).__name__}))\n"
)


@dataclass(frozen=True, slots=True)
class GpuReading:
    free_bytes: int
    total_bytes: int
    reserved_bytes: int

    def as_record(self) -> dict[str, int]:
        return {"free": self.free_bytes, "total": self.total_bytes, "reserved": self.reserved_bytes}


Runner = Callable[[], str]


def _subprocess_runner() -> str:
    env = {key: os.environ[key] for key in GPU_ENV if key in os.environ}
    done = subprocess.run(
        [sys.executable, "-c", PROBE_CODE],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_S,
        check=False,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    return done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""


def read_gpu(runner: Runner | None = None) -> GpuReading:
    """One reading; a clear refusal when CUDA is unavailable or the probe fails."""

    try:
        text = (runner or _subprocess_runner)()
        data = json.loads(text)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise GuardRefusal("CUDA_PROBE_FAILED", type(error).__name__) from error
    if not isinstance(data, dict):
        raise GuardRefusal("CUDA_PROBE_FAILED", "unreadable")
    if "error" in data:
        raise GuardRefusal(str(data["error"]), "CUDA is not usable; no real-model run can start")
    try:
        return GpuReading(int(data["free"]), int(data["total"]), int(data.get("reserved", 0)))
    except (KeyError, TypeError, ValueError) as error:
        raise GuardRefusal("CUDA_PROBE_FAILED", "unreadable") from error


def check_free(minimum: int = MIN_FREE_BYTES, runner: Runner | None = None) -> GpuReading:
    """Refuse unless at least ``minimum`` bytes are free (device-wide)."""

    reading = read_gpu(runner)
    if reading.free_bytes < minimum:
        raise GuardRefusal(
            "GPU_FREE_BELOW_MINIMUM", f"free {reading.free_bytes / GIB:.1f} GiB < {minimum / GIB:.1f} GiB"
        )
    return reading


__all__ = ["GIB", "MIN_FREE_BYTES", "GpuReading", "check_free", "read_gpu"]
