"""Architecture guards of the F15 degraded-continuation slice (D1 r3 sec. 10).

DC-1 state: the primitives exist and nothing reads them yet.  Later units
(DC-5 supervisor, DC-4 participant quarantine, DC-6 ladder) extend the
allow-lists below; extending one is a deliberate act that a reviewer sees.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "suzka"
MODULE = PACKAGE / "runtime" / "degraded_continuation.py"

# Modules that import the primitives.  DC-5: the server and the supervisor.  DC-4
# adds the two modules of the participant quarantine: ``domain_quarantine`` (the
# policy) and ``startup_reconciliation`` (it raises ``CycleRestart``).  DC-6 adds
# ``hard_band_watcher`` (the HARD-band ladder).
IMPORTERS_OF_PRIMITIVES = {
    "suzka/api/server.py",
    "suzka/api/supervisor.py",
    "suzka/runtime/domain_quarantine.py",
    "suzka/runtime/hard_band_watcher.py",
    "suzka/runtime/startup_reconciliation.py",
}
# Modules that read ``settings.degraded_continuation``: create_app (DC-5: it
# chooses the supervised lifespan with ``mode``).
READERS_OF_SETTINGS: set[str] = {"suzka/api/server.py"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _calls(tree: ast.AST) -> list[tuple[str, ast.Call]]:
    found: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                owner = ast.unparse(func.value)
                found.append((f"{owner}.{func.attr}", node))
            elif isinstance(func, ast.Name):
                found.append((func.id, node))
    return found


def _enclosing_functions(tree: ast.Module) -> dict[ast.AST, str]:
    owners: dict[ast.AST, str] = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                owners.setdefault(node, function.name)
    return owners


FORBIDDEN_CALL_SUFFIXES = (
    "rmtree",
    "unlink",
    "remove",
    "removedirs",
    "rmdir",
    "truncate",
    "ftruncate",
    "copy",
    "copy2",
    "copyfile",
    "copyfileobj",
    "copytree",
    "move",
    "write_text",
    "write_bytes",
    "touch",
    "symlink",
    "link",
    "chmod",
    "chown",
    "utime",
    "setxattr",
    "sendfile",
    "copy_file_range",
    "renames",
    "makedirs",
    "mkfifo",
    "mknod",
    "fdopen",
    "mkstemp",
    "mkdtemp",
    "popen",
    "system",
    "spawn",
    "exec",
    "run",
    "check_call",
    "check_output",
)


def test_the_primitive_module_never_deletes_repairs_or_copies() -> None:
    tree = _tree(MODULE)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    banned = {
        "shutil",
        "tempfile",
        "subprocess",
        "ctypes",
        "importlib",
        "io",
        "builtins",
    }
    assert not banned & {name.split(".")[0] for name in imported}, imported
    # every rename / replace / open / mkdir is one of the named primitives
    # (Path.rename, Path.open, io.open... are not among them)
    owners = _enclosing_functions(tree)
    primitive = {
        "os.rename": "_rename_directory",
        "os.replace": "_publish_file",
        "os.mkdir": "_create_manifest_dir",
        "manifest_dir.parent.mkdir": "_create_manifest_dir",
    }
    for name, call in _calls(tree):
        suffix = name.rsplit(".", 1)[-1]
        if suffix in {"rename", "replace", "mkdir", "open"} and name not in {
            "os.open",
            "replace",  # dataclasses.replace, imported by that bare name
        }:
            assert name in primitive and owners.get(call) == primitive[name], (
                name,
                call.lineno,
            )
    # __import__ / getattr(os, ...) indirection is not used
    for name, call in _calls(tree):
        assert name not in {"__import__", "exec", "eval"}, (name, call.lineno)
        if name == "getattr":
            assert "os" not in ast.unparse(call.args[0]) or ast.unparse(
                call.args[1]
            ) in {
                "'O_NOFOLLOW'",
                "'O_NONBLOCK'",
                "'O_NOCTTY'",
            }, call.lineno
    for name, call in _calls(tree):
        suffix = name.rsplit(".", 1)[-1]
        assert suffix not in FORBIDDEN_CALL_SUFFIXES, (name, call.lineno)
    # nothing is opened with the builtin open() for writing, and no flag that
    # truncates or appends exists anywhere in the module
    for name, call in _calls(tree):
        if name == "open":
            raise AssertionError(f"builtin open() at line {call.lineno}")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not names & {"O_TRUNC", "O_APPEND", "O_RDWR"}


def test_every_mutation_goes_through_one_named_primitive() -> None:
    tree = _tree(MODULE)
    owners = _enclosing_functions(tree)
    allowed = {
        "os.rename": {"_rename_directory"},
        "os.replace": {"_publish_file"},
        "os.write": {"_write_new_file"},
        "os.mkdir": {"_create_manifest_dir"},
        "manifest_dir.parent.mkdir": {"_create_manifest_dir"},
        "os.fsync": {"_fsync_directory", "_write_new_file"},
    }
    seen: dict[str, set[str]] = {key: set() for key in allowed}
    for name, call in _calls(tree):
        if name in allowed:
            seen[name].add(owners.get(call, "<module>"))
    for name, functions in seen.items():
        assert functions, f"{name} is expected to be used"
        assert functions <= allowed[name], (name, functions)
    # os.open: read-only descriptors (hash, fsync of a directory) and the one
    # create-new writer (O_EXCL), nowhere else
    opens = {
        owners.get(call, "<module>") for name, call in _calls(tree) if name == "os.open"
    }
    assert opens == {"_hash_file", "_fsync_directory", "_write_new_file"}
    writer = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_write_new_file"
    )
    assert "O_EXCL" in ast.unparse(writer) and "O_CREAT" in ast.unparse(writer)


def test_the_module_guards_the_quarantined_path_before_it_writes() -> None:
    tree = _tree(MODULE)
    for function in ("_write_new_file", "_publish_file"):
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == function
        )
        assert "_assert_writable_path" in ast.unparse(node)


def _imports(path: Path) -> set[str]:
    tree = _tree(path)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            modules.add(base)
            modules |= {f"{base}.{alias.name}" for alias in node.names}
    return modules


def test_slice_modules_never_create_a_subject_history_store() -> None:
    # A-2 (iii): no module of the slice imports or calls ``open_or_create_store``.
    slice_modules = [MODULE, PACKAGE / "runtime" / "thread_liveness.py"] + [
        path
        for path in (
            PACKAGE / "api" / "supervisor.py",
            PACKAGE / "runtime" / "hard_band_watcher.py",
        )
        if path.exists()
    ]
    for path in slice_modules:
        text = path.read_text()
        assert "open_or_create_store" not in text, path
        assert not any("history_store" in name for name in _imports(path)), path


def test_primitives_stay_inert_until_a_unit_wires_them_in() -> None:
    importers: set[str] = set()
    readers: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if path == MODULE:
            continue
        if any("degraded_continuation" in name for name in _imports(path)):
            if rel != "suzka/config/schema.py":  # the settings class lives there
                importers.add(rel)
        reads_setting = any(
            isinstance(node, ast.Attribute) and node.attr == "degraded_continuation"
            for node in ast.walk(_tree(path))
        )
        if reads_setting and rel != "suzka/config/schema.py":
            readers.add(rel)
    assert importers == IMPORTERS_OF_PRIMITIVES
    assert readers == READERS_OF_SETTINGS


def test_every_thread_site_is_covered_by_the_termination_proof() -> None:
    # D1 sec. 12: an architecture test that every ``Thread(`` / ``Timer(`` site
    # in suzka/ is covered.  A file that creates a thread must expose
    # ``threads_alive`` and register what it starts in a ``ThreadLedger``.
    sites: dict[str, int] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = _tree(path)
        count = sum(
            1
            for name, _ in _calls(tree)
            if name.rsplit(".", 1)[-1] in {"Thread", "Timer"}
        )
        if count:
            sites[path.relative_to(ROOT).as_posix()] = count
    # DC-5: the supervisor's own control / cycle threads are proven by the
    # cycle-thread check of the proof itself, not by a ``threads_alive`` owner.
    sites.pop("suzka/api/supervisor.py", None)
    assert set(sites) == {
        "suzka/runtime/agent_runtime.py",
        "suzka/runtime/emotion_timer.py",
        "suzka/r16/driver.py",
        "suzka/r18/approval_owner.py",
        "suzka/r18/executor.py",
        "suzka/r19/scheduler.py",
        "suzka/runtime/hard_band_watcher.py",
    }, "a new thread owner must be added to the termination proof (and here)"
    for rel, count in sites.items():
        text = (ROOT / rel).read_text()
        assert "def threads_alive" in text, rel
        assert "ThreadLedger()" in text, rel
        # every started thread is registered after its start()
        assert text.count("_thread_ledger.register(") >= count, rel


# ------------------------------------------------------------------ DC-4

DOMAIN_MODULE = PACKAGE / "runtime" / "domain_quarantine.py"


def test_the_domain_quarantine_module_never_deletes_repairs_or_copies() -> None:
    tree = _tree(DOMAIN_MODULE)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not {"shutil", "tempfile", "subprocess", "ctypes", "importlib"} & {
        name.split(".")[0] for name in imported
    }
    for name, call in _calls(tree):
        suffix = name.rsplit(".", 1)[-1]
        if name == "open":
            raise AssertionError(f"builtin open() at line {call.lineno}")
        # the ONLY filesystem effects: mkdir / chmod of the new temporary root,
        # the marker write, one rename, fsync.  Nothing that removes or rewrites.
        assert suffix not in set(FORBIDDEN_CALL_SUFFIXES) - {"chmod"}, (name, call.lineno)
    owners = _enclosing_functions(tree)
    allowed = {
        "os.rename": {"_publish_root"},
        "os.mkdir": {"_mkdir_private"},
        "os.chmod": {"_mkdir_private"},
        "os.write": {"_write_marker"},
    }
    for name, call in _calls(tree):
        if name in allowed:
            assert owners.get(call) in allowed[name], (name, call.lineno)
    # one writer of the marker: exclusive create, never truncate or append
    writer = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_write_marker"
    )
    assert "O_EXCL" in ast.unparse(writer) and "O_CREAT" in ast.unparse(writer)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not names & {"O_TRUNC", "O_APPEND", "O_RDWR"}


def test_the_domain_quarantine_module_never_creates_a_subject_history_store() -> None:
    text = DOMAIN_MODULE.read_text()
    assert "open_or_create_store" not in text
    assert not any("history_store" in name for name in _imports(DOMAIN_MODULE))


def test_only_the_two_memory_domains_can_ever_be_quarantined() -> None:
    from suzka.runtime.domain_quarantine import (
        DEFAULT_QUARANTINE_PARTICIPANTS,
        STORE_DIRECTORY_NAMES,
    )
    from suzka.runtime.degraded_continuation import QUARANTINABLE_PARTICIPANTS

    assert set(STORE_DIRECTORY_NAMES) == {"memory.experience", "memory.semantic"}
    assert QUARANTINABLE_PARTICIPANTS == set(STORE_DIRECTORY_NAMES)
    # Semantic is implemented but OFF by default (S0-c, D1 F-7)
    assert DEFAULT_QUARANTINE_PARTICIPANTS == {"memory.experience"}


def test_no_ordinary_module_reads_the_quarantine_policy_or_its_floor() -> None:
    """The policy is built only by the server (mode "continue"); the floor only
    by startup reconciliation."""

    users: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel == "suzka/runtime/domain_quarantine.py":
            continue
        if any("domain_quarantine" in name for name in _imports(path)):
            users.add(rel)
    assert users == {"suzka/api/server.py", "suzka/runtime/startup_reconciliation.py"}


def test_the_server_builds_the_policy_only_in_continue_mode() -> None:
    tree = _tree(PACKAGE / "api" / "server.py")
    source = (PACKAGE / "api" / "server.py").read_text()
    assert source.count("DomainQuarantinePolicy(") == 1
    # the construction sits under ``if dc_config.mode == "continue":``
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "dc_config.mode" in ast.unparse(node.test):
            assert "DomainQuarantinePolicy(" in ast.unparse(node)
            break
    else:  # pragma: no cover
        raise AssertionError("no mode guard around the policy")


def test_the_empty_root_is_created_before_any_store_object_and_before_reconciliation() -> None:
    """D1 sec. 4.2 step 4: ``start_cycle`` (a JOURNALED manifest creates the empty
    root + marker) runs BEFORE ``DualMemorySystem`` (which builds the Semantic
    store), ``ExperienceStore.from_memory_root`` and any reconciliation."""

    source = (PACKAGE / "api" / "server.py").read_text()
    start = source.index("domain_policy.start_cycle()")
    for later in (
        "app.state.memory_system = getattr(",
        "ExperienceStore.from_memory_root(",
        "StartupReconciliationCoordinator(\n                app.state.event_journal",
        "reconcile_open_transactions()",
        "resolve_open_wal_rollover()",
    ):
        assert source.index(later) > start, later


# ------------------------------------------------------------------ DC-6

WATCHER_MODULE = PACKAGE / "runtime" / "hard_band_watcher.py"


def test_the_hard_band_watcher_deletes_nothing_and_only_appends_its_log() -> None:
    # The ladder archives only through ``compact_with_window`` (the Journal's
    # own, verified compaction): this module has no way to delete, move, copy
    # or rewrite anything.
    for name, _call in _calls(_tree(WATCHER_MODULE)):
        assert name.rsplit(".", 1)[-1] not in FORBIDDEN_CALL_SUFFIXES, name
    text = WATCHER_MODULE.read_text()
    assert "os.O_APPEND" in text and "os.O_TRUNC" not in text
    assert "open(" not in text.replace("os.open(", "")


def test_the_server_builds_the_watcher_only_in_continue_mode_with_the_switch_on() -> None:
    source = (PACKAGE / "api" / "server.py").read_text()
    assert source.count("HardBandWatcher(") == 1
    tree = _tree(PACKAGE / "api" / "server.py")
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_hard_band_watcher"
    )
    guard = next(node for node in function.body if isinstance(node, ast.If))
    text = ast.unparse(guard.test)
    assert "mode" in text and "hard_band_auto_shrink" in text
    assert isinstance(guard.body[0], ast.Return)
    # and the supervisor is the only owner of the origin of the HARD band
    # (P1-6): the watcher holds no origin attribute of its own.
    names = {
        node.attr
        for node in ast.walk(_tree(WATCHER_MODULE))
        if isinstance(node, ast.Attribute)
    }
    assert not {name for name in names if "hard_since" in name}
