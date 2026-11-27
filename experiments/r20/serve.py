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
from experiments.r20.launch import check_launch_record
from experiments.r20.measure import GpuSampler
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
    parser.add_argument("--launch-nonce")
    parser.add_argument("--turn-cap-s", type=float, default=float(bridge.TIMEOUT_S))
    # Lab diagnostics, OPT-IN (the smoke sets it): record every model call to the private directory.
    parser.add_argument("--record-calls", action="store_true")
    # Lab diagnostics, OPT-IN (the smoke sets it): send the recent conversation as real chat roles.
    parser.add_argument("--chat-roles", action="store_true")
    # Lab diagnostics, OPT-IN (the smoke sets it): the model sees a bare chat (no SUZKA prompt); not with --chat-roles.
    parser.add_argument("--bare-chat", action="store_true")
    # Lab diagnostics, OPT-IN (needs --bare-chat): restore these sections of the real prompt into the bare chat.
    parser.add_argument("--bare-restore")
    parser.add_argument("--restore-placement", choices=["system", "user"], default="system")
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
    if mode is ProviderMode.REAL:
        # The launcher (the Human-launch entry) is this child's parent and holds the Human's attestation.
        check_launch_record(root.parent / "control" / "launch.json", args.launch_nonce, os.getppid())
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
        # A real model is loaded ONLY in a Human-launched real run, and a real run loads ONLY a real config.
        if (mode is ProviderMode.REAL) != (settings.model.provider != "dummy"):
            raise GuardRefusal("MODEL_CONFIG_MODE_MISMATCH", f"{mode.value} vs {settings.model.provider}")
    except GuardRefusal as refusal:
        print(f"serve refused: {refusal.code}", file=sys.stderr)
        return 2

    import uvicorn

    from suzka.api.server import create_app

    sampler: GpuSampler | None = None
    if mode is ProviderMode.REAL:
        sampler = GpuSampler(root.parent / "evidence" / "gpu.jsonl")
        sampler.start()  # numbers only: free / total / reserved, before the load and then every few seconds
    record = bool(args.record_calls)
    chat_roles = bool(args.chat_roles)
    bare_chat = bool(args.bare_chat)
    if chat_roles and bare_chat:
        print("serve refused: CHAT_ROLES_AND_BARE_CHAT", file=sys.stderr)  # the smoke refuses this earlier
        return 2
    restore: tuple[str, ...] = ()
    if args.bare_restore:
        from experiments.r20.bare_chat_adapter import parse_restore_list

        try:
            restore = parse_restore_list(args.bare_restore)
        except ValueError:
            print("serve refused: BARE_RESTORE_INVALID", file=sys.stderr)  # the smoke refuses this earlier
            return 2
        if not bare_chat:
            print("serve refused: BARE_RESTORE_NEEDS_BARE_CHAT", file=sys.stderr)
            return 2
    placement = str(args.restore_placement)
    if bare_chat:
        # Lab diagnostics only (`bare_chat_adapter`): inside the recorder, like the chat-role adapter.
        from experiments.r20.bare_chat_adapter import BareChatAdapter, wrap_loader as wrap_bare_loader
        from suzka.api import server as app_server

        if mode is ProviderMode.REAL:
            app_server.load_model_provider = wrap_bare_loader(app_server.load_model_provider, restore, placement)
    if chat_roles:
        # Lab diagnostics only (`chat_role_adapter`): inside the recorder, so the recorded call is the one sent.
        from experiments.r20.chat_role_adapter import ChatRoleAdapter, wrap_loader as wrap_adapter_loader
        from suzka.api import server as app_server

        if mode is ProviderMode.REAL:
            app_server.load_model_provider = wrap_adapter_loader(app_server.load_model_provider)
    if record:
        # Lab diagnostics only (``recording_provider``): raw prompts / outputs go to <run dir>/private (0700, 0600 file).
        from experiments.r20 import artifacts
        from experiments.r20.recording_provider import RecordingProvider, wrap_loader
        from suzka.api import server as app_server

        private = artifacts.private_dir(root.parent)
        if mode is ProviderMode.REAL:
            # The model is built lazily in the app's lifespan, once per DC cycle, through this module-level
            # name.  Replacing it wraps every provider a cycle builds (a restart cycle included).
            app_server.load_model_provider = wrap_loader(app_server.load_model_provider, private)
    app = create_app(settings)
    if provider is not None:
        if chat_roles:
            provider = ChatRoleAdapter(provider)
        if bare_chat:
            provider = BareChatAdapter(provider, restore, placement)
        app.state.model_provider = RecordingProvider(provider, private) if record else provider
        # Mode "continue": the supervisor drops every ``app.state`` key added after
        # ``create_app`` before a new cycle.  The injected scripted provider is the
        # harness's, not the cycle's: part of the baseline, so a restart cycle in
        # this process keeps it (a model provider the CYCLE builds is still rebuilt).
        supervisor = getattr(app.state, "dc_supervisor", None)
        if supervisor is not None:
            supervisor.capture_baseline()
    server = uvicorn.Server(
        uvicorn.Config(app, host=settings.api.host, port=port, log_level="warning", lifespan="on")
    )

    lineage_cache: list[str] = []

    def status(_arguments: Any) -> dict[str, Any]:
        # The lineage never changes in a life: read it once (inspect() is heavy).
        journal = getattr(app.state, "event_journal", None)
        if not lineage_cache and journal is not None and server.started:
            found = journal.inspect().journal_lineage_id
            if found is not None:
                lineage_cache.append(str(found))
        lineage = lineage_cache[0] if lineage_cache else None
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
            return bridge.review(app.state, str(arguments["stage"]), timeout_s=float(args.turn_cap_s))
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
        if sampler is not None:
            sampler.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
