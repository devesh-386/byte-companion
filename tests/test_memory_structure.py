"""Phase 5: categories, expiry of dated facts, the grouped overview, and upgrading an old database."""
import datetime as dt
import sqlite3

import numpy as np
import pytest

from companion.memory.store import MemoryStore
from companion.memory.structure import categorize, expires_at, find_date
from tests.test_memory import HashEmbedder, make_store

TUE = dt.date(2026, 9, 22)  # a Tuesday


@pytest.mark.parametrize("text,day", [
    ("The user has an exam tomorrow.", dt.date(2026, 9, 23)),
    ("The user has a test the day after tomorrow.", dt.date(2026, 9, 24)),
    ("The user's exam is on Friday.", dt.date(2026, 9, 25)),
    ("The user has a meeting on Tuesday.", TUE),                       # said on a Tuesday = today
    ("The user has a meeting next Tuesday.", dt.date(2026, 9, 29)),
    ("The user's sister is visiting next week.", dt.date(2026, 10, 4)),
    ("The user's CN exam is on 3 October.", dt.date(2026, 10, 3)),
    ("The user's project demo is on October 18th, 2026.", dt.date(2026, 10, 18)),
    ("The user's trip starts 2026-12-20.", dt.date(2026, 12, 20)),
    ("The user's visa appointment is on 5 January.", dt.date(2027, 1, 5)),  # month already gone -> next year
    ("The user likes Valorant.", None),
])
def test_find_date(text, day):
    assert find_date(text, TUE) == day


@pytest.mark.parametrize("text,cat", [
    ("The user's name is Devesh.", "identity"),
    ("The user has a sister named Priya.", "people"),
    ("The user's sister Priya is visiting next week.", "schedule"),
    ("The user has an exam tomorrow.", "schedule"),
    ("The user studies computer science at SRM.", "work"),
    ("The user loves biryani.", "preferences"),
    ("The user has an RTX 5050 laptop.", "setup"),
    ("The user is left handed.", "other"),
])
def test_categorize(text, cat):
    assert categorize(text) == cat


def ts(day: dt.date, hour: int = 12) -> float:
    return dt.datetime.combine(day, dt.time(hour)).timestamp()


def test_birthdays_in_the_past_do_not_expire():
    assert expires_at("The user was born on 3 May 2004.", ts(TUE)) is None


def test_exam_tomorrow_expires_after_the_date():
    """Phase 5 done-when: 'exam tomorrow' stops being recalled once the day is over."""
    clock = type("C", (), {"t": ts(TUE), "__call__": lambda self: self.t})()
    store = make_store(clock)
    store.add("The user has a Computer Networks exam tomorrow.")
    store.add("The user studies computer science at SRM.")
    assert any("exam" in h.text for h in store.search("exam", k=3, min_score=0.1))

    clock.t = ts(TUE + dt.timedelta(days=2), 10)   # Thursday morning: the day after the exam, still fine
    assert any("exam" in h.text for h in store.search("exam", k=3, min_score=0.1))

    clock.t = ts(TUE + dt.timedelta(days=3), 10)   # Friday: out of date
    assert not any("exam" in h.text for h in store.search("exam", k=3, min_score=0.1))
    assert "schedule" not in store.overview() and "schedule" in store.overview(include_expired=True)
    assert store.count("fact") == 2                # kept, only hidden


def test_overview_groups_by_category():
    store = make_store()
    for f in ("The user's name is Devesh.", "The user loves biryani.", "The user plays Valorant a lot, loves it."):
        store.add(f)
    groups = store.overview()
    assert list(groups) == ["identity", "preferences"]
    assert len(groups["preferences"]) == 2


def test_old_database_is_upgraded_in_place(tmp_path):
    db = tmp_path / "memory.db"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
        text TEXT NOT NULL, created_at REAL NOT NULL, last_used REAL, use_count INTEGER NOT NULL DEFAULT 0,
        embedding BLOB NOT NULL);""")
    vec = np.zeros(64, dtype=np.float32).tobytes()
    con.execute("INSERT INTO memories(kind, text, created_at, embedding) VALUES ('fact', ?, ?, ?)",
                ("The user's name is Devesh.", ts(TUE), vec))
    con.execute("INSERT INTO memories(kind, text, created_at, embedding) VALUES ('fact', ?, ?, ?)",
                ("The user has an exam tomorrow.", ts(TUE), vec))
    con.commit()
    con.close()
    store = MemoryStore(db, HashEmbedder(), clock=lambda: ts(TUE))
    groups = store.overview()
    assert groups["identity"][0].text.startswith("The user's name")
    assert groups["schedule"][0].expires_at is not None
    MemoryStore(db, HashEmbedder(), clock=lambda: ts(TUE))  # opening again is harmless
