"""
memory.py — Two-layer memory system.

Layer 1 — Short-term (in-process list)
    A plain Python list of HumanMessage / AIMessage / SystemMessage objects
    capped to MEMORY_SIZE turns, returned to the LLM each turn.

Layer 2 — Long-term (SQLite)
    - facts        : key/value facts about the user
    - preferences  : key/value user preferences
    - fts_memory   : FTS5 full-text index of every conversation turn

Compared with the original version:
- memory retrieval runs on every turn
- the FTS query is normalized for better recall
- SQLite is tuned for local performance
- history compaction preserves a summary marker cleanly
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# ── Config ────────────────────────────────────────────────────────────────────

MEMORY_SIZE: int = int(os.getenv("MEMORY_SIZE", "6"))
STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", "./storage"))

MEMORY_SNIPPET_LIMIT: int = int(os.getenv("MEMORY_SNIPPET_LIMIT", "3"))
MEMORY_SNIPPET_MAX_CHARS: int = int(os.getenv("MEMORY_SNIPPET_MAX_CHARS", "900"))

MEMORY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do",
    "does", "for", "from", "has", "have", "how", "i", "if", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "please", "should", "the", "to",
    "was", "we", "what", "when", "where", "who", "why", "will", "with", "you",
    "your", "about", "into", "this", "that", "these", "those", "then", "than",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_memory
    USING fts5(content, tags, ts UNINDEXED);
"""

# ── Cache helper ──────────────────────────────────────────────────────────────

@dataclass
class _TTLCache:
    """Simple time-to-live cache for a single value."""
    ttl: float = 5.0
    _value: Any = field(default=None, repr=False)
    _expires: float = field(default=0.0, repr=False)

    def get(self) -> Any | None:
        if time.monotonic() < self._expires:
            return self._value
        return None

    def set(self, value: Any) -> None:
        self._value = value
        self._expires = time.monotonic() + self.ttl

    def invalidate(self) -> None:
        self._expires = 0.0

# ── MemoryManager ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Manages both short-term and long-term memory.

    Usage
    -----
    mm = await MemoryManager.create()
    ctx = await mm.get_context(user_query)
    mm.add_message(HumanMessage(...))
    mm.add_message(AIMessage(...))
    await mm.save_interaction(user_text, ai_text)
    history = mm.get_history()
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._history: list[BaseMessage] = []
        self._facts_cache = _TTLCache(ttl=5.0)
        self._prefs_cache = _TTLCache(ttl=5.0)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, storage_dir: Path = STORAGE_DIR) -> "MemoryManager":
        storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = storage_dir / "memory.db"
        mm = cls(db_path)
        await asyncio.to_thread(mm._init_db)
        return mm

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA busy_timeout=3000;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryManager not initialised — call create() first")
        return self._conn

    # ── Short-term history ────────────────────────────────────────────────────

    def add_message(self, msg: BaseMessage) -> None:
        self._history.append(msg)

    def get_history(self) -> list[BaseMessage]:
        """
        Return recent history, preserving a compacted summary message at the front
        when one exists.
        """
        cap = MEMORY_SIZE * 2
        if not self._history:
            return []

        first = self._history[0]
        has_summary = isinstance(first, SystemMessage) and isinstance(first.content, str) and first.content.startswith("[Conversation summary]")

        if has_summary:
            tail = self._history[1:]
            if len(tail) > cap:
                tail = tail[-cap:]
            return [first, *tail]

        return self._history[-cap:] if len(self._history) > cap else list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def undo_last_exchange(self) -> bool:
        """Remove the last Human+AI pair. Returns True if something was removed."""
        if len(self._history) >= 2:
            self._history = self._history[:-2]
            return True
        if len(self._history) == 1:
            self._history.pop()
            return True
        return False

    def history_token_estimate(self) -> int:
        """Rough token count: 1 token ≈ 4 chars."""
        total = sum(len(m.content) for m in self._history if isinstance(m.content, str))
        return total // 4

    # ── Facts ─────────────────────────────────────────────────────────────────

    def add_fact(self, key: str, value: str) -> None:
        self._db().execute(
            "INSERT OR REPLACE INTO facts (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._db().commit()
        self._facts_cache.invalidate()

    def get_all_facts(self) -> dict[str, str]:
        cached = self._facts_cache.get()
        if cached is not None:
            return cached

        rows = self._db().execute("SELECT key, value FROM facts").fetchall()
        result = {k: v for k, v in rows}
        self._facts_cache.set(result)
        return result

    def remove_fact(self, key: str) -> None:
        self._db().execute("DELETE FROM facts WHERE key = ?", (key,))
        self._db().commit()
        self._facts_cache.invalidate()

    # ── Preferences ───────────────────────────────────────────────────────────

    def add_preference(self, key: str, value: str) -> None:
        self._db().execute(
            "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._db().commit()
        self._prefs_cache.invalidate()

    def get_all_preferences(self) -> dict[str, str]:
        cached = self._prefs_cache.get()
        if cached is not None:
            return cached

        rows = self._db().execute("SELECT key, value FROM preferences").fetchall()
        result = {k: v for k, v in rows}
        self._prefs_cache.set(result)
        return result

    # ── FTS5 memory search ────────────────────────────────────────────────────

    def _normalize_query(self, query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
        filtered = [t for t in tokens if len(t) > 2 and t not in MEMORY_STOPWORDS]
        if not filtered:
            filtered = [t for t in tokens if len(t) > 1]
        return " ".join(filtered[:12]).strip()

    def _clip(self, text: str, max_chars: int = MEMORY_SNIPPET_MAX_CHARS) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    async def search(self, query: str, limit: int = MEMORY_SNIPPET_LIMIT) -> list[str]:
        """Full-text search over past interactions. Returns relevant snippets."""

        def _search() -> list[str]:
            safe_q = self._normalize_query(query)
            if not safe_q:
                return []

            try:
                rows = self._db().execute(
                    """
                    SELECT content
                    FROM fts_memory
                    WHERE fts_memory MATCH ?
                    ORDER BY bm25(fts_memory)
                    LIMIT ?
                    """,
                    (safe_q, limit),
                ).fetchall()
                return [self._clip(r[0]) for r in rows]
            except sqlite3.OperationalError:
                return []

        return await asyncio.to_thread(_search)

    async def save_interaction(self, user_text: str, ai_text: str) -> None:
        """Persist a turn to the FTS5 index for future recall."""
        content = f"User: {user_text}\nAssistant: {ai_text}"
        ts = str(int(time.time()))

        def _save() -> None:
            self._db().execute(
                "INSERT INTO fts_memory (content, tags, ts) VALUES (?, ?, ?)",
                (content, "dialogue", ts),
            )
            self._db().commit()

        await asyncio.to_thread(_save)

    # ── Context assembly ──────────────────────────────────────────────────────

    async def get_context(self, query: str = "") -> str:
        """
        Assemble the context block injected into every system prompt.

        Always includes facts + preferences.
        Also searches FTS5 on every turn for better continuity.
        """
        parts: list[str] = []

        facts = self.get_all_facts()
        if facts:
            lines = "\n".join(f"  {k}: {v}" for k, v in facts.items())
            parts.append(f"[User Facts]\n{lines}")

        prefs = self.get_all_preferences()
        if prefs:
            lines = "\n".join(f"  {k}: {v}" for k, v in prefs.items())
            parts.append(f"[User Preferences]\n{lines}")

        if query.strip():
            snippets = await self.search(query, limit=MEMORY_SNIPPET_LIMIT)
            if snippets:
                joined = "\n---\n".join(snippets)
                parts.append(f"[Relevant Past Interactions]\n{joined}")

        return "\n\n".join(parts)

    # ── Compact ───────────────────────────────────────────────────────────────

    async def compact_history(self, summarise_fn) -> str:
        """
        Summarise the current history via summarise_fn (async str->str),
        then replace the history with a single SystemMessage summary.
        """
        if not self._history:
            return ""

        history_to_summarise = self._history
        if (
            isinstance(self._history[0], SystemMessage)
            and isinstance(self._history[0].content, str)
            and self._history[0].content.startswith("[Conversation summary]")
        ):
            history_to_summarise = self._history[1:]

        if not history_to_summarise:
            return ""

        raw = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history_to_summarise
            if isinstance(m.content, str)
        )

        summary = (await summarise_fn(raw)).strip()
        if not summary:
            return ""

        self._history = [SystemMessage(content=f"[Conversation summary]\n{summary}")]
        return summary

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None