"""Negative-test kit for the F15 degraded-continuation slice (DC-1).

* ``tree_snapshot`` / ``assert_trees_equal``: byte-for-byte tree equality
  (relative path -> kind, mode-independent content).  A quarantined artifact
  must equal its pre-fault snapshot.
* ``QuarantineAccessSpy``: fails the test when the code under test OPENS, lists
  or writes a path that has a ``*.quarantine-*`` component.  Later units wrap
  the live process in it ("the live process never opens a quarantined tree").
  The quarantine action itself reads the tree it moved (the re-hash), so a test
  of the writer must not run under the spy.
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator

from suzka.runtime.degraded_continuation import QUARANTINE_NAME_INFIX

Snapshot = dict[str, tuple[str, bytes | str | None]]


def tree_snapshot(root: Path) -> Snapshot:
    """``{relpath: (kind, content)}``: file bytes, symlink target, ``None`` for a
    directory.  Never follows a symlink."""

    snapshot: Snapshot = {}
    for path in sorted(root.rglob("*")):
        relpath = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relpath] = ("symlink", os.readlink(path))
        elif path.is_dir():
            snapshot[relpath] = ("dir", None)
        elif path.is_file():
            snapshot[relpath] = ("file", path.read_bytes())
        else:
            snapshot[relpath] = ("other", None)
    return snapshot


def assert_trees_equal(expected: Snapshot, actual: Snapshot) -> None:
    assert expected.keys() == actual.keys(), (
        sorted(expected.keys() - actual.keys()),
        sorted(actual.keys() - expected.keys()),
    )
    for relpath, value in expected.items():
        assert actual[relpath] == value, relpath


class QuarantineAccessViolation(AssertionError):
    pass


def _under_quarantine(path: object) -> bool:
    if isinstance(path, int):  # a file descriptor: nothing to inspect
        return False
    try:
        text = os.fspath(path)  # type: ignore[call-overload]
    except TypeError:
        return False
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    return QUARANTINE_NAME_INFIX in text


@contextmanager
def QuarantineAccessSpy() -> Iterator[list[str]]:
    """Patch ``open`` / ``os.open`` / ``os.scandir`` / ``os.listdir`` so that
    touching a ``*.quarantine-*`` path raises ``QuarantineAccessViolation``.
    Yields the list of refused paths."""

    refused: list[str] = []
    real_open = builtins.open
    real_os_open = os.open
    real_scandir = os.scandir
    real_listdir = os.listdir

    def guard(path: object, how: str) -> None:
        if _under_quarantine(path):
            refused.append(f"{how}:{os.fspath(path)}")  # type: ignore[call-overload]
            raise QuarantineAccessViolation(f"{how} of a quarantined path")

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        guard(file, "open")
        return real_open(file, *args, **kwargs)

    def spy_os_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        guard(path, "os.open")
        return real_os_open(path, *args, **kwargs)

    def spy_scandir(path: Any = ".") -> Any:
        guard(path, "os.scandir")
        return real_scandir(path)

    def spy_listdir(path: Any = ".") -> Any:
        guard(path, "os.listdir")
        return real_listdir(path)

    builtins.open = spy_open  # type: ignore[assignment]
    os.open = spy_os_open  # type: ignore[assignment]
    os.scandir = spy_scandir  # type: ignore[assignment]
    os.listdir = spy_listdir  # type: ignore[assignment]
    try:
        yield refused
    finally:
        builtins.open = real_open
        os.open = real_os_open
        os.scandir = real_scandir
        os.listdir = real_listdir
