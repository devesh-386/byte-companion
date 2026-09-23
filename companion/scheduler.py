"""Scheduled background tasks (tasks.yaml), run by a fresh agent with read-only tools.

Schedules:  at: "08:30"  (daily)   at: "Sun 11:00"  (weekly)   cron: "0 11 * * 0"  (5-field cron)
A daily/weekly `at` task that was missed because the laptop was off runs once when Byte starts later
that day. Cron tasks never catch up.

`require_quote: true` makes a task's notification prove itself: it must contain a "quoted fragment"
that appears word for word in a file the agent read during that run. A small model cannot then invent
a deadline, because the code refuses any notification it cannot back with a real quote.
"""
import datetime as dt
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable

import yaml

from .actionlog import ActionLog
from .agent import Agent
from .plugins import Deps, load_plugins
from .protocol import build_system_prompt
from .tools import LOOK, ToolError, ToolRegistry

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass
class Task:
    name: str
    prompt: str
    at: tuple[int | None, int, int] | None = None   # (weekday or None, hour, minute)
    cron: str | None = None
    enabled: bool = True
    require_quote: bool = False
    preload: str | None = None      # glob of files the CODE reads and hands to the model
    extract: str = "open-items"     # what to pull from them: "open-items" | "status"
    builtin: str | None = None      # run a code routine instead of the model, e.g. "morning-brief"


def parse_at(spec: str) -> tuple[int | None, int, int]:
    parts = spec.strip().split()
    day = None
    if len(parts) == 2:
        day = DAYS.index(parts[0].lower()[:3])
    hh, mm = parts[-1].split(":")
    h, m = int(hh), int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"bad time {spec!r}")
    return day, h, m


def _cron_field(spec: str, value: int, lo: int, hi: int) -> bool:
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-")
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def cron_matches(spec: str, now: dt.datetime) -> bool:
    fields = spec.split()
    if len(fields) != 5:
        raise ValueError(f"cron needs 5 fields: {spec!r}")
    minute, hour, dom, mon, dow = fields
    cron_dow = (now.weekday() + 1) % 7  # cron: 0 = Sunday; Python: 0 = Monday
    return (_cron_field(minute, now.minute, 0, 59) and _cron_field(hour, now.hour, 0, 23)
            and _cron_field(dom, now.day, 1, 31) and _cron_field(mon, now.month, 1, 12)
            and (_cron_field(dow, cron_dow, 0, 7) or (cron_dow == 0 and _cron_field(dow, 7, 0, 7))))


def load_tasks(path: Path) -> list[Task]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = []
    for name, spec in (data.get("tasks") or {}).items():
        if not spec.get("prompt") and not spec.get("builtin"):
            raise ValueError(f"task {name!r} needs a prompt or a builtin")
        tasks.append(Task(name=name, prompt=str(spec.get("prompt") or "").strip(),
                          at=parse_at(spec["at"]) if spec.get("at") else None, cron=spec.get("cron"),
                          enabled=spec.get("enabled", True), require_quote=spec.get("require_quote", False),
                          preload=spec.get("preload"), extract=spec.get("extract", "open-items"),
                          builtin=spec.get("builtin")))
    return tasks


MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH = re.compile(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r")[a-z]*\.?(?: (\d{4}))?\b", re.I)
_DEADLINE_WORDS = re.compile(r"- \[ \]|\b(due|deadline|exam|demo|submission|submit|review|viva|interview|"
                             r"presentation|by)\b", re.I)
_DONE_STATUS = re.compile(r"^status:\s*(complete|completed|done|retired|archived|shipped)\b", re.I | re.M)


def _dates_in(line: str, today: dt.date) -> list[dt.date]:
    found = []
    for y, m, d in _ISO_DATE.findall(line):
        try:
            found.append(dt.date(int(y), int(m), int(d)))
        except ValueError:
            pass
    for d, mon, y in _DAY_MONTH.findall(line):
        try:
            found.append(dt.date(int(y) if y else today.year, MONTHS.index(mon.lower()[:3]) + 1, int(d)))
        except ValueError:
            pass
    return found


def morning_brief(pattern: str, today: dt.date, horizon_days: int = 14) -> str | None:
    """Deterministic: a small model misread a metadata date as a deadline, so rules decide here.
    Soonest upcoming dated task in a note body wins; otherwise the newest note's first unticked box."""
    import glob
    dated: list[tuple[dt.date, str, str]] = []
    undated: list[tuple[float, str, str]] = []
    for path in glob.glob(pattern):
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        head = text[: text.find("\n---", 3) + 4] if text.startswith("---") else ""
        if _DONE_STATUS.search(head):
            continue
        body = text[len(head):]
        for raw in body.splitlines():
            line = raw.strip()
            if not line or "- [x]" in line.lower():
                continue
            if _DEADLINE_WORDS.search(line):
                upcoming = [d for d in _dates_in(line, today) if 0 <= (d - today).days <= horizon_days]
                if upcoming:
                    dated.append((min(upcoming), line, p.stem))
                    continue
            if line.startswith("- [ ]"):
                undated.append((-p.stat().st_mtime, line, p.stem))
    if dated:
        when, line, note = min(dated)
        days = (when - today).days
        lead = "Today" if days == 0 else "Tomorrow" if days == 1 else f"In {days} days ({when:%a %d %b})"
        return f"{lead}, from {note}: “{_tidy(line)}”"
    if undated:
        _, line, note = min(undated)
        return f"No dated deadlines. Outstanding in {note}: “{_tidy(line)}”"
    return None


def _tidy(line: str, limit: int = 140) -> str:
    """Readable quote for a notification: no checkbox or markdown marks, cut on a word boundary."""
    text = re.sub(r"^- \[ \]\s*", "", line)
    text = re.sub(r"[*`_]+", "", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-—") + "…"


_OPEN_ITEM = re.compile(
    r"- \[ \]|\b(pending|deferred|unverified|todo|to do|to record|outstanding|not yet|blocked|deadline|due)\b"
    r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)


def _frontmatter(text: str) -> list[str]:
    if not text.startswith("---"):
        return []
    end = text.find("\n---", 3)
    return [l.strip() for l in text[3:end].splitlines() if l.strip()][:6] if end > 0 else []


def gather(pattern: str, mode: str, now: dt.datetime, per_file: int = 10, budget: int = 7000) -> tuple[str, int]:
    """Read notes in code, so the model can't skip the reading, and keep only the lines that matter.
    Returns (text for the model, number of open items / status lines found)."""
    import glob
    blocks = []
    found = 0
    for path in sorted(glob.glob(pattern)):
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # File timestamps can land a hair after `now`, which would read as "-1 days ago".
        age = max(0, (now - dt.datetime.fromtimestamp(p.stat().st_mtime)).days)
        if mode == "status":
            m = re.search(r"^#+ *Resume.*?$(.*?)(?=^#+ |\Z)", text, re.S | re.M | re.I)
            picked = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()][:4] if m else []
        else:
            body = text[text.find("\n---", 3) + 4:] if text.startswith("---") else text
            picked = [l.strip() for l in body.splitlines() if _OPEN_ITEM.search(l)][:per_file]
        found += len(picked)
        lines = _frontmatter(text) + picked
        body = "\n".join(f"  {l[:220]}" for l in lines) or "  (nothing matched)"
        blocks.append(f"### {p.name} (last edited {age} days ago)\n{body}")
    return "\n\n".join(blocks)[:budget], found


class Scheduler:
    def __init__(self, tasks_path: Path, state_path: Path, run_task: Callable[[Task], None],
                 clock: Callable[[], dt.datetime] = dt.datetime.now, poll_s: float = 20):
        self.tasks_path = tasks_path
        self.state_path = state_path
        self.run_task = run_task
        self.clock = clock
        self.poll_s = poll_s
        self.tasks: list[Task] = []
        self._mtime = 0.0
        self._stop = threading.Event()
        try:
            self.last_run: dict[str, str] = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.last_run = {}

    def reload(self) -> None:
        try:
            mtime = self.tasks_path.stat().st_mtime
        except OSError:
            self.tasks = []
            return
        if mtime != self._mtime:
            self.tasks = load_tasks(self.tasks_path)
            self._mtime = mtime

    def due(self, now: dt.datetime) -> list[Task]:
        out = []
        for t in self.tasks:
            if not t.enabled:
                continue
            last = self.last_run.get(t.name)
            if t.at is not None:
                day, h, m = t.at
                if day is not None and now.weekday() != day:
                    continue
                if (now.hour, now.minute) >= (h, m) and last != now.date().isoformat():
                    out.append(t)
            elif t.cron and cron_matches(t.cron, now) and last != now.strftime("%Y-%m-%dT%H:%M"):
                out.append(t)
        return out

    def mark_ran(self, task: Task, now: dt.datetime) -> None:
        self.last_run[task.name] = now.date().isoformat() if task.at else now.strftime("%Y-%m-%dT%H:%M")
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.last_run, indent=1), encoding="utf-8")
        except OSError:
            pass

    def check(self) -> list[Task]:
        self.reload()
        now = self.clock()
        fired = self.due(now)
        for t in fired:
            self.mark_ran(t, now)  # before running, so a crash can't make it fire in a loop
            threading.Thread(target=self.run_task, args=(t,), daemon=True, name=f"task-{t.name}").start()
        return fired

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.check()
                except Exception as e:  # a broken tasks.yaml must not kill the scheduler
                    print(f"[scheduler] {type(e).__name__}: {e}")
                self._stop.wait(self.poll_s)

        threading.Thread(target=loop, daemon=True, name="scheduler").start()

    def stop(self) -> None:
        self._stop.set()


_QUOTE = re.compile(r"[\"“”']([^\"“”']{8,})[\"“”']")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_is_backed(message: str, sources: list[str]) -> bool:
    haystack = _norm("\n".join(sources))
    return any(_norm(q) in haystack for q in _QUOTE.findall(message))


TASK_PERSONA = (
    "You are Byte, running a scheduled background task for {user}. Nobody is watching the chat. "
    "Do the task using tools, following its rules exactly. To tell {user} something, call "
    "send_notification. Finish with a one-line summary of what you did."
)


def make_task_runner(llm, notify: Callable[[str], None], user: str, log_path: Path) -> Callable[[Task], None]:
    def run(task: Task) -> None:
        if task.builtin == "morning-brief" and task.preload:
            msg = morning_brief(task.preload, dt.date.today())
            if msg:
                notify(msg)
            _log(log_path, f"{dt.datetime.now():%Y-%m-%d %H:%M} {task.name} (builtin) notified={msg!r}\n")
            return
        reads: list[str] = []
        sent: list[str] = []
        blocked: list[str] = []
        registry = ToolRegistry(max_output_chars=6000)
        # Only the local families: a background task reads your files, it doesn't browse the web.
        load_plugins(registry, Deps(), include={"system", "files"})

        def send_notification(message: Annotated[str, "At most two short sentences"]) -> str:
            """Show the user a desktop notification."""
            if task.require_quote and not quote_is_backed(message, reads):
                blocked.append(message)
                raise ToolError('Blocked: the message must include, in "double quotes", a fragment copied '
                                'word for word from a file you read in this task. That quote was not found. '
                                'Quote the file exactly, or send nothing and reply "nothing pressing".')
            sent.append(message)
            notify(message)
            return "Notification shown."

        registry.register(send_notification)

        def on_event(e) -> None:
            if e.kind == "tool_result" and e.data["name"] == "read_text_file":
                reads.append(e.data["result"])

        # Nobody is watching a scheduled task, so it may only look (tier 0). Enforced by the agent, and
        # anything above that is also hidden from the prompt.
        agent = Agent(llm, registry,
                      build_system_prompt(TASK_PERSONA.format(user=user), registry.schemas(max_tier=LOOK)),
                      max_steps=10, history_budget_chars=60000, max_tier=LOOK,
                      action_log=ActionLog(log_path.parent / "actions.jsonl", f"scheduler:{task.name}"))
        started = time.time()
        prompt = task.prompt
        if task.preload:
            now = dt.datetime.now()
            context, found = gather(task.preload, task.extract, now)
            if found == 0:
                _log(log_path, f"{now:%Y-%m-%d %H:%M} {task.name} no open items found - quiet morning\n")
                return
            reads.append(context)  # quotes may come from anything the code handed over
            prompt += (f"\n\nToday is {now:%A %d %B %Y}. Here is what I read from the notes "
                       f"(line fragments, per file):\n\n{context}")
        try:
            answer = agent.run(prompt, on_event=on_event)
        except Exception as e:
            answer = f"FAILED {type(e).__name__}: {e}"
        _log(log_path, f"{dt.datetime.now():%Y-%m-%d %H:%M} {task.name} ({time.time() - started:.0f}s) "
                       f"sources={len(reads)} notified={sent!r} blocked={blocked!r} answer={answer[:300]!r}\n")

    return run


def _log(path: Path, line: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
