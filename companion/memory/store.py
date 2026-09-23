import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .retriever import Weights, hybrid_scores, keywords, mmr
from .structure import CATEGORIES, categorize, expires_at

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,            -- 'fact' (lasting) or 'episode' (a past exchange)
    text        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    last_used   REAL,
    use_count   INTEGER NOT NULL DEFAULT 0,
    embedding   BLOB NOT NULL             -- float32 unit vector
);
"""
# Phase 5 columns, added to older databases in place (nothing is lost).
MIGRATIONS = [("category", "ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'other'"),
              ("expires_at", "ALTER TABLE memories ADD COLUMN expires_at REAL")]


class EmbedderLike(Protocol):
    dim: int

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray: ...


@dataclass
class Memory:
    id: int
    kind: str
    text: str
    created_at: float
    score: float = 0.0
    category: str = "other"
    expires_at: float | None = None


class MemoryStore:
    def __init__(self, path: Path | str, embedder: EmbedderLike, clock: Callable[[], float] = time.time,
                 weights: Weights = Weights()):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(memories)")}
        for column, sql in MIGRATIONS:
            if column not in have:
                self.conn.execute(sql)
        self.conn.commit()
        self.embedder = embedder
        if not have or "category" not in have:
            self._backfill()
        self.clock = clock
        self.weights = weights
        self._lock = threading.Lock()

    def add(self, text: str, kind: str = "fact", dedupe_threshold: float = 0.92) -> tuple[int, bool]:
        """Store a memory. A fact nearly identical to an existing one replaces it. Returns (id, is_new)."""
        text = text.strip()
        vec = self.embedder.embed([text])[0]
        now = self.clock()
        category, expires = ("other", None) if kind != "fact" else (categorize(text), expires_at(text, now))
        with self._lock:
            if kind == "fact":
                rows = self._rows(["fact"], include_expired=True)
                if rows:
                    sims = np.stack([r[5] for r in rows]) @ vec
                    best = int(np.argmax(sims))
                    if sims[best] >= dedupe_threshold:
                        mem_id = rows[best][0]
                        self.conn.execute("UPDATE memories SET text=?, created_at=?, embedding=?, category=?, "
                                          "expires_at=? WHERE id=?",
                                          (text, now, vec.tobytes(), category, expires, mem_id))
                        self.conn.commit()
                        return mem_id, False
            cur = self.conn.execute(
                "INSERT INTO memories(kind, text, created_at, embedding, category, expires_at) VALUES (?,?,?,?,?,?)",
                (kind, text, now, vec.astype(np.float32).tobytes(), category, expires))
            self.conn.commit()
            return cur.lastrowid, True

    def search(self, query: str, k: int = 4, kinds: list[str] | None = None,
               min_score: float = 0.35) -> list[Memory]:
        qvec = self.embedder.embed([query], query=True)[0]
        now = self.clock()
        with self._lock:
            rows = self._rows(kinds)
            if not rows:
                return []
            vecs = np.stack([r[5] for r in rows])
            ages = np.array([(now - r[3]) / 86400 for r in rows])
            scores = hybrid_scores(qvec, vecs, keywords(query), [keywords(r[2]) for r in rows], ages, self.weights)
            picked = mmr(scores, vecs, k, min_score=min_score)
            hits = [Memory(rows[i][0], rows[i][1], rows[i][2], rows[i][3], float(scores[i])) for i in picked]
            if hits:
                self.conn.executemany("UPDATE memories SET last_used=?, use_count=use_count+1 WHERE id=?",
                                      [(now, h.id) for h in hits])
                self.conn.commit()
            return hits

    def delete(self, mem_id: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            self.conn.commit()

    def count(self, kind: str | None = None) -> int:
        with self._lock:
            if kind:
                return self.conn.execute("SELECT COUNT(*) FROM memories WHERE kind=?", (kind,)).fetchone()[0]
            return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def recent(self, n: int = 10, kind: str | None = None) -> list[Memory]:
        with self._lock:
            sql = "SELECT id, kind, text, created_at FROM memories"
            args: tuple = ()
            if kind:
                sql += " WHERE kind=?"
                args = (kind,)
            rows = self.conn.execute(sql + " ORDER BY created_at DESC LIMIT ?", args + (n,)).fetchall()
            return [Memory(*r) for r in rows]

    def overview(self, include_expired: bool = False) -> dict[str, list[Memory]]:
        """Every remembered fact, grouped by category (the 'what do you remember about me' view)."""
        now = self.clock()
        with self._lock:
            rows = self.conn.execute("SELECT id, kind, text, created_at, category, expires_at FROM memories "
                                     "WHERE kind='fact' ORDER BY created_at DESC").fetchall()
        groups: dict[str, list[Memory]] = {c: [] for c in CATEGORIES}
        for r in rows:
            if r[5] is not None and r[5] < now and not include_expired:
                continue
            groups.setdefault(r[4], []).append(Memory(r[0], r[1], r[2], r[3], 0.0, r[4], r[5]))
        return {c: ms for c, ms in groups.items() if ms}

    def _backfill(self) -> None:
        """Give facts saved before phase 5 a category and expiry, dated from when they were saved."""
        rows = self.conn.execute("SELECT id, text, created_at FROM memories WHERE kind='fact'").fetchall()
        self.conn.executemany("UPDATE memories SET category=?, expires_at=? WHERE id=?",
                              [(categorize(t), expires_at(t, c), i) for i, t, c in rows])
        self.conn.commit()

    def _rows(self, kinds: list[str] | None, include_expired: bool = False):
        sql = "SELECT id, kind, text, created_at, use_count, embedding FROM memories"
        where, args = [], []
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            args = list(kinds)
        if not include_expired:
            # An expired fact ("exam tomorrow", three days later) is kept but no longer recalled.
            where.append("(expires_at IS NULL OR expires_at >= ?)")
            args.append(self.clock())
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [(r[0], r[1], r[2], r[3], r[4], np.frombuffer(r[5], dtype=np.float32))
                for r in self.conn.execute(sql, args).fetchall()]
