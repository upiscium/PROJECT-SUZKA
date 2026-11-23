"""The T2 wrapper: ``python -m experiments.r20.serve <root> --run-id ID``.

A NORMAL child process: it builds ``create_app`` on the factory's config
(paths relative, cwd = the root), injects the provider, starts the control
channel and runs uvicorn on the loopback port.  It refuses to start a root that
lacks the throwaway marker (or whose marker names another run), a production
path, a forbidden port or a non-loopback host.  It runs only on a root the
factory created; it is not part of the product and nothing under ``suzka/``
imports it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from experiments.r20 import bridge, guards
from experiments.r20.control import ControlError, ControlServer
from experiments.r20.guards import GuardRefusal
from experiments.r20.provider import (
    DEFAULT_ANSWER,
    NON_NEUTRAL_LOSS,
    ApprovalActor,
    ProviderMode,
    ScriptedProvider,
    check_modes,
)


def load_provider(mode: ProviderMode, config: Path | None) -> Any:
    """The provider object to inject, or ``None`` to keep the app's own
    ``load_model_provider`` (``dummy`` config / the real tier)."""

    if mode is not ProviderMode.SCRIPTED:
        return None
    if config is None:
        return ScriptedProvider()
    data = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GuardRefusal("PROVIDER_CONFIG_INVALID")
    return ScriptedProvider(
        data.get("script", ()),
        default=data.get("default", DEFAULT_ANSWER),
        loss_value=float(data.get("loss", NON_NEUTRAL_LOSS)),
    )


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="experiments.r20.serve")
    parser.add_argument("root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--provider", choices=[m.value for m in ProviderMode], default="scripted")
    parser.add_argument("--approval-actor", choices=[a.value for a in ApprovalActor], default="SCRIPTED")
    parser.add_argument("--provider-config")
    parser.add_argument("--human-launch", action="store_true")
    return parser.parse_args(argv)


def prepare(args: argparse.Namespace) -> tuple[Path, Any, ProviderMode]:
    """Every refusal, before anything is created or started."""

    root = Path(os.path.realpath(args.root))
    guards.check_production_prefix(root)
    guards.check_no_foreign_journal(root.parent)  # the root holds its own journal
    guards.check_marker(root, args.run_id)
    if root.parent.name != args.run_id:
        raise GuardRefusal("MARKER_MISMATCH", "run directory name")
    guards.check_no_symlinks(root)
    mode = ProviderMode(args.provider)
    # The Human's own launch needs a terminal; a process started by the harness
    # or another agent has none.  A misuse guard, not an approval.
    check_modes(
        mode,
        ApprovalActor(args.approval_actor),
        human_launch=bool(args.human_launch) and sys.stdin.isatty(),
    )
    return root, load_provider(mode, Path(args.provider_config) if args.provider_config else None), mode


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        root, provider, mode = prepare(args)
        os.chdir(root)  # T2: every path in the config is relative to the root
        from suzka.config.settings import load_settings

        settings = load_settings(root / "config.yaml")
        guards.check_settings_paths(settings, root, relative=True)
        port = guards.check_settings_network(settings)
    except GuardRefusal as refusal:
        print(f"serve refused: {refusal.code}", file=sys.stderr)
        return 2

    import uvicorn

    from suzka.api.server import create_app

    app = create_app(settings)
    if provider is not None:
        app.state.model_provider = provider
    server = uvicorn.Server(
        uvicorn.Config(app, host=settings.api.host, port=port, log_level="warning", lifespan="on")
    )

    def status(_arguments: Any) -> dict[str, Any]:
        journal = getattr(app.state, "event_journal", None)
        lineage = None if journal is None else journal.inspect().journal_lineage_id
        return {
            "port": port,
            "pid": os.getpid(),
            "run_id": args.run_id,
            "journal_lineage_id": None if lineage is None else str(lineage),
            "ready": bool(server.started),
            "provider": mode.value,
        }

    def review(arguments: Any) -> dict[str, Any]:
        try:
            return bridge.review(app.state, str(arguments["stage"]))
        except bridge.BridgeRefused as refusal:
            raise ControlError(str(refusal)) from refusal

    def quit_(_arguments: Any) -> dict[str, Any]:
        server.should_exit = True
        return {}

    control = ControlServer(
        Path(os.path.realpath(root.parent / "control" / "ctl.sock")),
        {"status": status, "r15_review": review, "quit": quit_},
    )
    control.start()
    try:
        server.run()
    finally:
        control.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
