import datetime as dt

import pytest

from companion.scheduler import Scheduler, Task, cron_matches, load_tasks, parse_at, quote_is_backed

SUN_1100 = dt.datetime(2026, 9, 27, 11, 0)   # a Sunday
WED_0830 = dt.datetime(2026, 9, 23, 8, 30)   # a Wednesday


def test_parse_at():
    assert parse_at("08:30") == (None, 8, 30)
    assert parse_at("Sun 11:00") == (6, 11, 0)
    with pytest.raises(ValueError):
        parse_at("25:00")


@pytest.mark.parametrize("spec,now,expected", [
    ("0 11 * * 0", SUN_1100, True),
    ("0 11 * * 7", SUN_1100, True),
    ("0 11 * * 1-5", SUN_1100, False),
    ("*/15 8-9 * * *", WED_0830, True),
    ("*/20 8-9 * * *", WED_0830, False),
    ("30 8 23 9 3", WED_0830, True),
])
def test_cron(spec, now, expected):
    assert cron_matches(spec, now) is expected


def make(tmp_path, tasks, now):
    ran = []
    s = Scheduler(tmp_path / "tasks.yaml", tmp_path / "state.json", ran.append, clock=lambda: now[0])
    s.tasks = tasks
    s.reload = lambda: None
    return s, ran


def test_daily_at_runs_once_and_catches_up_later_same_day(tmp_path):
    now = [dt.datetime(2026, 9, 23, 7, 0)]
    s, _ = make(tmp_path, [Task("brief", "p", at=(None, 8, 30))], now)
    assert s.check() == []
    now[0] = dt.datetime(2026, 9, 23, 14, 5)       # laptop was off at 08:30
    assert [t.name for t in s.check()] == ["brief"]
    assert s.check() == []                          # not twice the same day
    now[0] = dt.datetime(2026, 9, 24, 8, 30)
    assert [t.name for t in s.check()] == ["brief"]


def test_weekly_and_disabled(tmp_path):
    now = [WED_0830]
    s, _ = make(tmp_path, [Task("sun", "p", at=(6, 11, 0)), Task("off", "p", at=(None, 0, 0), enabled=False)], now)
    assert s.check() == []
    now[0] = SUN_1100
    assert [t.name for t in s.check()] == ["sun"]


def test_cron_runs_once_per_minute_and_state_persists(tmp_path):
    now = [SUN_1100]
    s, _ = make(tmp_path, [Task("weekly", "p", cron="0 11 * * 0")], now)
    assert len(s.check()) == 1 and s.check() == []
    again = Scheduler(tmp_path / "tasks.yaml", tmp_path / "state.json", lambda t: None)
    assert again.last_run["weekly"] == "2026-09-27T11:00"


def test_load_tasks(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text("tasks:\n  a:\n    at: '08:30'\n    require_quote: true\n    prompt: >\n      do it\n"
                 "  b:\n    cron: '0 11 * * 0'\n    enabled: false\n    prompt: x\n", encoding="utf-8")
    a, b = load_tasks(p)
    assert (a.name, a.at, a.require_quote, a.prompt) == ("a", (None, 8, 30), True, "do it")
    assert (b.cron, b.enabled) == ("0 11 * * 0", False)


def test_gather_open_items_and_status(tmp_path):
    from companion.scheduler import gather
    (tmp_path / "a.md").write_text("---\nstatus: active\n---\n# A\nintro line\n- [ ] record the demo video\n"
                                   "- [x] done thing\nexam on 3 September\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\nstatus: paused\n---\n## Resume\nNext: retrain the model.\n## Other\nx\n",
                                   encoding="utf-8")
    now = dt.datetime.now()
    items, n = gather(str(tmp_path / "*.md"), "open-items", now)
    assert "record the demo video" in items and "exam on 3 September" in items and n == 2
    assert "done thing" not in items and "intro line" not in items and "last edited 0 days ago" in items
    status, n_status = gather(str(tmp_path / "*.md"), "status", now)
    assert "Next: retrain the model." in status and "status: paused" in status and n_status == 1
    (tmp_path / "a.md").write_text("---\nstarted: 2026-09-15\n---\nall done\n", encoding="utf-8")
    (tmp_path / "b.md").unlink()
    assert gather(str(tmp_path / "*.md"), "open-items", now)[1] == 0  # frontmatter dates don't count


def test_morning_brief_rules(tmp_path):
    from companion.scheduler import morning_brief
    today = dt.date(2026, 9, 23)
    (tmp_path / "exam.md").write_text("---\nstatus: active\ndate: 2026-09-24\n---\n"
                                      "- [ ] revise unit 2\nCN exam on 26 September\nold demo on 2026-08-01\n",
                                      encoding="utf-8")
    (tmp_path / "done.md").write_text("---\nstatus: complete\n---\n- [ ] demo due 2026-09-24\n", encoding="utf-8")
    msg = morning_brief(str(tmp_path / "*.md"), today)
    # The metadata date (24th) and the finished project are ignored; the real exam (26th) wins.
    assert msg == "In 3 days (Sat 26 Sep), from exam: “CN exam on 26 September”"
    (tmp_path / "exam.md").write_text("---\nstatus: active\n---\n- [x] done\n- [ ] record the demo video\n",
                                      encoding="utf-8")
    assert morning_brief(str(tmp_path / "*.md"), today) == \
        "No dated deadlines. Outstanding in exam: “record the demo video”"
    (tmp_path / "exam.md").write_text("---\nstatus: active\n---\nnothing open\n", encoding="utf-8")
    assert morning_brief(str(tmp_path / "*.md"), today) is None


def test_quote_guard():
    notes = ["## TODO\n- [ ] record the demo video   (checklist unticked)\n"]
    assert quote_is_backed('Demo still pending: "record the demo video"', notes)
    assert quote_is_backed('Pending: \u201crecord the  demo video\u201d', notes)       # curly quotes, spacing
    assert not quote_is_backed('Due Friday: "demo due on friday"', notes)           # invented
    assert not quote_is_backed("The demo video is due soon.", notes)                # no quote at all
