"""Attachment points for IP-22b / 22c / 22d (hooks, NOT implementations).

IP-22a builds the factory and the harness core only.  The units below plug in
through these names; nothing here runs a case.

* 22b ablation: ``ABLATION`` receives sealed policy-input bytes and returns an
  AblationResult-like value; the module ``experiments.r20.ablation`` is 22b's.
* 22c F-matrix: ``F_CASES`` maps ``"F01-m"`` ... to a callable taking the
  harness context; the matrix table / runner is 22c's.
* 22d S0 dry run and the SIGKILL tier: ``KILL_POINTS`` names the coarse points
  (D1 sec. 4-3); the poll that decides when to kill is ``kill.py`` (using
  ``polling.short_read`` and ``supervisor.ChildProcess.kill_group``), the dry run is
  ``dry_run.py`` and the process-tier chain driver is ``s0.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

KILL_POINTS = ("K1", "K2", "K3", "K4", "K5", "K6")
# K1 is T1 only (confirm writes the challenge and the claim in one request).
T1_ONLY_KILL_POINTS = frozenset({"K1"})

ABLATION: dict[str, Callable[[bytes], Any]] = {}
F_CASES: dict[str, Callable[..., Any]] = {}
DRY_RUN_STEPS: dict[str, Callable[..., Any]] = {}


def register(table: dict[str, Callable[..., Any]], name: str, function: Callable[..., Any]) -> None:
    if name in table:
        raise ValueError(f"{name} is already registered")
    table[name] = function


__all__ = ["ABLATION", "DRY_RUN_STEPS", "F_CASES", "KILL_POINTS", "T1_ONLY_KILL_POINTS", "register"]
