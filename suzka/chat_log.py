"""Display-only Chat transcript store, grouped into owner-visible sessions.

A Chat session here is a grouping of turns for the Chat screen. It is not a
Context: every session's turns still run in the same durable Context, so the
subject's state stays continuous across sessions. Nothing in this store is
read by the runtime, the prompt, memory, or learning, and deleting a session
removes only its transcript, never anything the subject remembers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import sqlite3
import threading
import uuid


logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    episode_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_turns_by_session ON chat_turns(session_id, turn_id);
"""


class ChatSessionNotFound(LookupError):
    """The requested Chat session does not exist."""


class ChatLogUnavailable(RuntimeError):
    """The transcript store cannot be opened or used."""


@dataclass(frozen=True, slots=True)
class ChatSession:
    session_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ChatLogTurn:
    role: str
    content: str
    episode_id: str | None
    created_at: str


class ChatLogStore:
    """SQLite-backed transcript store; safe to call from request threads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent = self.path.parent.stat()
            if parent.st_uid == os.getuid() and parent.st_mode & 0o077:
                if self.path.parent.name == "chat_log":
                    # Our own dedicated directory: make it owner-only.
                    os.chmod(self.path.parent, 0o700)
                else:
                    logger.warning("chat log directory %s is readable by others", self.path.parent)
            # Create the file owner-only before SQLite opens it, without
            # following a symlink planted at its path.
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)
            with self._connect() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, SCHEMA_VERSION):
                    raise ChatLogUnavailable(f"unknown chat log schema version {version}")
                connection.executescript(_SCHEMA)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except (OSError, sqlite3.Error) as error:
            raise ChatLogUnavailable(str(error)) from error

    def create_session(self, title: str = "") -> ChatSession:
        now = _now()
        session = ChatSession(str(uuid.uuid4()), _title(title), now, now)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?)",
                (session.session_id, session.title, session.created_at, session.updated_at),
            )
        return session

    def list_sessions(self) -> list[ChatSession]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id, title, created_at, updated_at FROM chat_sessions "
                "ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [ChatSession(*row) for row in rows]

    def get_session(self, session_id: str) -> ChatSession:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id, title, created_at, updated_at FROM chat_sessions "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise ChatSessionNotFound(session_id)
        return ChatSession(*row)

    def turns(self, session_id: str) -> list[ChatLogTurn]:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone() is None:
                raise ChatSessionNotFound(session_id)
            rows = connection.execute(
                "SELECT role, content, episode_id, created_at FROM chat_turns "
                "WHERE session_id = ? ORDER BY turn_id",
                (session_id,),
            ).fetchall()
        return [ChatLogTurn(*row) for row in rows]

    def append_exchange(
        self, session_id: str, user_text: str, response: str, episode_id: str | None
    ) -> None:
        """Append one user turn and its reply; an untitled session takes its title from it."""

        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise ChatSessionNotFound(session_id)
            connection.executemany(
                "INSERT INTO chat_turns (session_id, role, content, episode_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (session_id, "user", user_text, None, now),
                    (session_id, "assistant", response, episode_id, now),
                ],
            )
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (row[0] or _title(user_text), now, session_id),
            )

    def rename_session(self, session_id: str, title: str) -> ChatSession:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                (_title(title), session_id),
            )
            if cursor.rowcount == 0:
                raise ChatSessionNotFound(session_id)
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
            )
            if cursor.rowcount == 0:
                raise ChatSessionNotFound(session_id)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                connection = sqlite3.connect(self.path, timeout=5)
            except sqlite3.Error as error:
                raise ChatLogUnavailable(str(error)) from error
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                with connection:
                    yield connection
            except sqlite3.Error as error:
                raise ChatLogUnavailable(str(error)) from error
            finally:
                connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _title(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= TITLE_MAX_CHARS else line[: TITLE_MAX_CHARS - 1] + "…"
