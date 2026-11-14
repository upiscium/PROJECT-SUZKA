"""Write and check the lab config for scripts/lab-from-production.sh.

Run with the deployed interpreter, so the checks see the same Settings the
lab server will load. Usage: lab_prepare.py PROD_CONFIG LAB_DIR PORT

Refuses (exit 1) unless every Path in the parsed lab settings stays inside
LAB_DIR and the lab port differs from the production port.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from suzka.config.settings import load_settings


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


def main(prod_config: str, lab_dir: str, port_text: str) -> int:
    lab = Path(lab_dir).resolve()
    port = int(port_text)
    prod_port = load_settings(prod_config).api.port

    with open(prod_config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw.setdefault("api", {})["port"] = port
    lab_config = lab / "config.yaml"
    lab_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    settings = load_settings(lab_config)

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
    registry = lab / settings.adapter_registry.path
    if registry.exists():
        for text in _strings(json.loads(registry.read_text(encoding="utf-8"))):
            if text.startswith(("/", "~")):
                problems.append(f"adapter registry names an absolute path: {text}")
    for problem in problems:
        print(f"lab_prepare: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(prod_port)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
