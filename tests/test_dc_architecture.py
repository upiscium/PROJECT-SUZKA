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

# Modules that import the primitives.  Inert in DC-1: only the health routes.
IMPORTERS_OF_PRIMITIVES = {"suzka/api/server.py"}
# Modules that read ``settings.degraded_continuation``.  None in DC-1.
READERS_OF_SETTINGS: set[str] = set()


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
    assert "shutil" not in imported and "tempfile" not in imported
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
    assert set(sites) == {
        "suzka/runtime/agent_runtime.py",
        "suzka/runtime/emotion_timer.py",
        "suzka/r16/driver.py",
        "suzka/r18/approval_owner.py",
        "suzka/r18/executor.py",
        "suzka/r19/scheduler.py",
    }, "a new thread owner must be added to the termination proof (and here)"
    for rel, count in sites.items():
        text = (ROOT / rel).read_text()
        assert "def threads_alive" in text, rel
        assert "ThreadLedger()" in text, rel
        # every started thread is registered after its start()
        assert text.count("_thread_ledger.register(") >= count, rel
