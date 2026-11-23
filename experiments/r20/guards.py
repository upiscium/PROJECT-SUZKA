"""Isolation guards: the harness must never touch production (D1 sec. 3-4).

Pure checks over paths, ports, settings and an environment mapping.  Each
refusal raises ``GuardRefusal`` with a closed ``code``; a test proves each one.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

# The ports of the production service, its Caddy front and its neighbour.
FORBIDDEN_PORTS = frozenset({8000, 8080, 8101})
# Directories a harness root must never be, or be inside.
PRODUCTION_ROOTS = (Path("/opt/project-suzka"), Path("/var/lib/suzka"), Path("/etc"))
MARKER_NAME = ".r20_throwaway"
BASE_ENV = "SUZKA_R20_BASE"
JOURNAL_NAME = "event_journal.jsonl"
CONFIG_ENV = "SUZKA_CONFIG_PATH"


class GuardRefusal(Exception):
    """A closed refusal; ``code`` never carries a request or a secret."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_port(port: object) -> int:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise GuardRefusal("PORT_INVALID", str(port))
    if port in FORBIDDEN_PORTS:
        raise GuardRefusal("PORT_FORBIDDEN", str(port))
    return port


def check_loopback_host(host: str) -> str:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise GuardRefusal("HOST_NOT_LOOPBACK", host)
    return host


def _is_inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def check_production_prefix(path: Path, *, repo: Path | None = None) -> Path:
    """Refuse a path that is, or is inside, a production location or the
    repository's own ``.suzka``."""

    resolved = Path(os.path.realpath(path))
    for production in PRODUCTION_ROOTS:
        if _is_inside(resolved, Path(os.path.realpath(production))):
            raise GuardRefusal("PRODUCTION_PATH", str(production))
    checkout = Path(os.path.realpath((repo or repo_root()) / ".suzka"))
    if _is_inside(resolved, checkout):
        raise GuardRefusal("REPOSITORY_STATE_PATH", str(checkout))
    return resolved


def check_no_foreign_journal(path: Path) -> None:
    """Refuse a path at or under a directory that already holds an
    ``event_journal.jsonl`` (an existing subject).  A started subject root holds
    its OWN journal, so a started root is checked through its run directory."""

    resolved = Path(os.path.realpath(path))
    for ancestor in (resolved, *resolved.parents):
        for candidate in (ancestor / JOURNAL_NAME, ancestor / ".suzka" / JOURNAL_NAME):
            if os.path.lexists(candidate):
                raise GuardRefusal("EXISTING_JOURNAL", str(candidate))


def check_not_production(path: Path, *, repo: Path | None = None) -> Path:
    """Both checks (for a location that must be new)."""

    resolved = check_production_prefix(path, repo=repo)
    check_no_foreign_journal(resolved)
    return resolved


def resolve_base(base: Path | str | None, environ: Mapping[str, str] | None = None) -> Path:
    """The explicit base: argument, then ``$SUZKA_R20_BASE``, else a new
    ``mkdtemp`` under ``/tmp``.  Never a production location."""

    import tempfile

    env = os.environ if environ is None else environ
    chosen = base if base is not None else env.get(BASE_ENV)
    if chosen is None:
        resolved = Path(tempfile.mkdtemp(prefix="suzka-r20-", dir="/tmp"))
    else:
        resolved = Path(chosen)
        if not resolved.is_absolute():
            raise GuardRefusal("BASE_NOT_ABSOLUTE", str(resolved))
        check_not_production(resolved)  # BEFORE anything is created
        resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_not_production(resolved)
    return Path(os.path.realpath(resolved))


def check_root_location(root: Path, base: Path) -> Path:
    """The root must be new and strictly under the explicit base."""

    if os.path.lexists(root):
        raise GuardRefusal("ROOT_EXISTS", str(root))
    parent = Path(os.path.realpath(root.parent))
    resolved_base = Path(os.path.realpath(base))
    if not _is_inside(parent, resolved_base):
        raise GuardRefusal("ROOT_OUTSIDE_BASE", str(root))
    check_not_production(parent)
    return parent / root.name


def check_inherited_environment(
    environ: Mapping[str, str], *, repo: Path | None = None
) -> None:
    """An inherited config path or ``PYTHONPATH`` outside the repository is a
    refusal (the child environment is built from an allow-list anyway)."""

    repository = Path(os.path.realpath(repo or repo_root()))
    config = environ.get(CONFIG_ENV)
    if config:
        raise GuardRefusal("INHERITED_CONFIG_PATH", CONFIG_ENV)
    for entry in filter(None, environ.get("PYTHONPATH", "").split(os.pathsep)):
        if not _is_inside(Path(os.path.realpath(entry)), repository):
            raise GuardRefusal("INHERITED_PYTHONPATH", entry)


def _paths(value: Any, where: str) -> Iterator[tuple[str, Path]]:
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


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def settings_path_problems(settings: Any, root: Path, *, relative: bool) -> list[str]:
    """Every ``Path`` in the parsed settings must stay inside ``root``.

    Re-implements the walk of ``scripts/lab_prepare.py``.  ``relative=True``
    (T2: cwd = root) additionally requires every path to be relative; T1
    (shared process cwd) writes absolute paths and requires exactly that.
    """

    resolved_root = Path(os.path.realpath(root))
    problems: list[str] = []
    for where, path in _paths(settings, "settings"):
        text = str(path)
        if text.startswith("~"):
            problems.append(f"{where} = {text} starts with ~")
        elif relative and path.is_absolute():
            problems.append(f"{where} = {text} is not relative")
        elif not relative and not path.is_absolute():
            problems.append(f"{where} = {text} is not absolute")
        elif not _is_inside(Path(os.path.realpath(path if path.is_absolute() else root / path)), resolved_root):
            problems.append(f"{where} = {text} leaves the subject root")
    registry = root / settings.adapter_registry.path if relative else settings.adapter_registry.path
    if registry.exists():
        for text in _strings(json.loads(registry.read_text(encoding="utf-8"))):
            if text.startswith(("/", "~")):
                problems.append(f"adapter registry names an absolute path: {text}")
            elif "/" in text and not _is_inside(Path(os.path.realpath(root / text)), resolved_root):
                problems.append(f"adapter registry path leaves the subject root: {text}")
    return problems


def check_settings_paths(settings: Any, root: Path, *, relative: bool) -> None:
    problems = settings_path_problems(settings, root, relative=relative)
    if problems:
        raise GuardRefusal("SETTINGS_PATH", "; ".join(problems[:4]))


def check_settings_network(settings: Any) -> int:
    check_loopback_host(settings.api.host)
    port = check_port(settings.api.port)
    for origin in settings.r18.approval_origins:
        if urlsplit(origin).port in FORBIDDEN_PORTS:
            raise GuardRefusal("PORT_FORBIDDEN", f"approval origin {origin}")
    return port


def check_no_symlinks(root: Path) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            if os.path.islink(os.path.join(current, name)):
                raise GuardRefusal("SYMLINK_IN_ROOT", name)


def write_marker(root: Path, run_id: str) -> None:
    (root / MARKER_NAME).write_text(
        json.dumps({"run_id": run_id, "created_by": "experiments.r20.factory"}),
        encoding="utf-8",
    )


def check_marker(root: Path, run_id: str) -> None:
    """The wrapper starts only a root the factory created for this run."""

    marker = root / MARKER_NAME
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GuardRefusal("MARKER_MISSING", str(root)) from error
    if (
        not isinstance(data, dict)
        or data.get("created_by") != "experiments.r20.factory"
        or data.get("run_id") != run_id
    ):
        raise GuardRefusal("MARKER_MISMATCH", str(root))


__all__ = [
    "BASE_ENV",
    "FORBIDDEN_PORTS",
    "MARKER_NAME",
    "GuardRefusal",
    "check_inherited_environment",
    "check_loopback_host",
    "check_marker",
    "check_no_foreign_journal",
    "check_no_symlinks",
    "check_production_prefix",
    "check_not_production",
    "check_port",
    "check_root_location",
    "check_settings_network",
    "check_settings_paths",
    "repo_root",
    "resolve_base",
    "settings_path_problems",
    "write_marker",
]
