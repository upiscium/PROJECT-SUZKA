"""Write and check the lab config for scripts/lab-from-production.sh.

Run with the deployed interpreter, so the checks see the same Settings the
lab server will load. Usage: lab_prepare.py PROD_CONFIG LAB_DIR PORT [--r18]

Refuses (exit 1) unless every Path in the parsed lab settings stays inside
LAB_DIR and the lab port differs from the production port.

``--r18`` (#411 D1 r7 §9-1, IP-17c) additionally switches R18 on in the LAB
config only, for an approval-relevant lab run: ``r18.enabled: true``, the
approval origins ``["http://localhost:<port>"]`` (a lab is localhost / tunnel
only, never behind Caddy) with no trusted networks, and ``r18.target_path``
relative in the lab.  Every value is a ``str`` or a relative ``Path``, so the
checks above still apply.  It also refuses a lab that does not listen on
loopback, a production port among the approval origins, and a lab without
subject.history.  A second output line, ``r18_target=<relative path>``, tells
the shell where ``python -m suzka.r18.target_init`` creates the target.
Without ``--r18`` the lab is exactly as before.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from suzka.config.settings import load_settings

# The lab target when the production config does not name one (R18Settings).
DEFAULT_R18_TARGET = ".suzka/private/r18/target.sqlite3"


def _paths(value: Any, where: str):  # type: ignore[no-untyped-def]
    if isinstance(value, Path):
        yield where, value
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _paths(getattr(value, name), f"{where}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _paths(item, f"{where}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _paths(item, f"{where}.{key}")


def _strings(value: Any):  # type: ignore[no-untyped-def]
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _r18_overrides(raw: dict[str, Any], port: int) -> str:
    """Set the lab's R18 section in the raw dict; return the relative target."""

    section = raw.get("r18")
    section = dict(section) if isinstance(section, dict) else {}
    target = section.get("target_path", DEFAULT_R18_TARGET)
    section["enabled"] = True
    section["target_path"] = str(target)
    section["approval_origins"] = [f"http://localhost:{port}"]
    section["approval_trusted_networks"] = []
    raw["r18"] = section
    return str(target)


def _r18_problems(settings: Any, lab: Path, port: int, prod_port: int) -> list[str]:
    """The ``--r18`` checks, over the parsed lab settings."""

    # Imported here: a lab without ``--r18`` must not depend on the deployed
    # code having these helpers.
    from suzka.config.schema import is_loopback_host, parse_approval_origin

    problems = []
    if not settings.r18.enabled:
        problems.append("lab r18.enabled is not true")
    if settings.r18.approval_origins != [f"http://localhost:{port}"]:
        problems.append("lab approval origins are not exactly the lab's localhost origin")
    if settings.r18.approval_trusted_networks:
        problems.append("a lab has no trusted networks")
    for origin in settings.r18.approval_origins:
        if parse_approval_origin(origin)[1] == prod_port:
            problems.append(f"approval origin {origin} names the production port")
    if not is_loopback_host(settings.api.host):
        problems.append(f"lab api.host {settings.api.host} is not loopback")
    if not settings.subject_history.enabled:
        problems.append("r18 needs subject.history in the lab config")
    target = settings.r18.target_path
    # The shell extracts the path with sed: no control characters or newlines.
    if any(ord(char) < 32 or ord(char) == 127 for char in str(target)):
        problems.append("r18.target_path has control characters")
    if target.is_absolute() or target.parent == Path("."):
        problems.append(f"r18.target_path {target} is not a relative path inside a directory")
    elif not (lab / target).resolve().is_relative_to(lab):
        problems.append(f"r18.target_path {target} leaves the lab directory")
    return problems



def _refusal_reasons(error: ValueError) -> str:
    """The location and message of each settings problem, without input values."""

    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            items = errors(include_input=False, include_url=False)
        except TypeError:  # an older pydantic without those switches
            items = [{"loc": item.get("loc", ()), "msg": item.get("msg", "")} for item in errors()]
        lines = [
            f"{'.'.join(str(part) for part in item.get('loc', ())) or 'settings'}: {item.get('msg', '')}"
            for item in items
        ]
        return "; ".join(lines[:8]) or error.__class__.__name__
    return str(error)[:200] or error.__class__.__name__


def main(prod_config: str, lab_dir: str, port_text: str, *flags: str) -> int:
    if any(flag != "--r18" for flag in flags):
        print("usage: lab_prepare.py PROD_CONFIG LAB_DIR PORT [--r18]", file=sys.stderr)
        return 2
    r18 = "--r18" in flags
    lab = Path(lab_dir).resolve()
    port = int(port_text)
    prod_port = load_settings(prod_config).api.port

    with open(prod_config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.setdefault("api", {})["port"] = port
    r18_target = _r18_overrides(raw, port) if r18 else None
    lab_config = lab / "config.yaml"
    lab_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    try:
        settings = load_settings(lab_config)
    except ValueError as error:
        if not r18:
            raise
        # e.g. r18.enabled with a non-loopback api.host (the settings refuse it).
        # Print where and why only, never the offending input values.
        print(f"lab_prepare: the lab settings are refused: {_refusal_reasons(error)}", file=sys.stderr)
        return 1

    problems = []
    if port == prod_port:
        problems.append(f"lab port {port} is the production port")
    if settings.api.port != port:
        problems.append(f"lab api.port is {settings.api.port}, not {port}")
    for where, path in _paths(settings, "settings"):
        text = str(path)
        if path.is_absolute() or text.startswith("~"):
            problems.append(f"{where} = {text} is not relative")
        elif not (lab / path).resolve().is_relative_to(lab):
            problems.append(f"{where} = {text} leaves the lab directory")
    if r18:
        problems.extend(_r18_problems(settings, lab, port, prod_port))
    registry = lab / settings.adapter_registry.path
    if registry.exists():
        for text in _strings(json.loads(registry.read_text(encoding="utf-8"))):
            if text.startswith(("/", "~")):
                problems.append(f"adapter registry names an absolute path: {text}")
            elif "/" in text and not (lab / text).resolve().is_relative_to(lab):
                problems.append(f"adapter registry path leaves the lab: {text}")
    for problem in problems:
        print(f"lab_prepare: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(prod_port)
    if r18_target is not None:
        print(f"r18_target={r18_target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
