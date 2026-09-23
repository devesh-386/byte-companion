"""Structure for remembered facts (phase 5): a category and, for dated things, an expiry.

Both come from plain rules, not the model. Rule of thumb from the morning brief: don't use a small LLM
where a rule will do. The rules only need to be right about the easy, common cases; anything they can't
place is 'other' and never expires, which is the safe default.

    "The user has an exam tomorrow."          said Mon 22 Sep  -> schedule, expires end of Wed 24 Sep
    "The user's sister Priya visits next week" said Tue 23 Sep -> people,   expires end of Sun 5 Oct
"""
import datetime as dt
import re

CATEGORIES = ["identity", "people", "schedule", "work", "preferences", "setup", "other"]
LABELS = {"identity": "Who you are", "people": "People in your life", "schedule": "Coming up",
          "work": "Work, study & projects", "preferences": "Likes & dislikes", "setup": "Your computer",
          "other": "Other"}

_RULES = [
    ("identity", r"\b(name is|is called|years old|was born|lives in|is from|birthday)\b"),
    ("people", r"\b(sister|brother|mother|mom|mum|father|dad|parents?|friend|girlfriend|boyfriend|wife|husband|"
               r"cousin|uncle|aunt|grandma|grandpa|roommate|teacher|professor|boss|colleague|son|daughter)\b"),
    ("schedule", r"\b(exam|test|deadline|due|meeting|interview|appointment|trip|flight|visit(?:ing|s)?|"
                 r"tomorrow|tonight|next week|this weekend|on (?:mon|tues|wednes|thurs|fri|satur|sun)day)\b"),
    # work before setup, so "studies computer science" is about study, not a computer
    ("work", r"\b(works?|job|intern(?:ship)?|project|studying|studies|student|college|university|course|"
             r"semester|learning|building|coding|repo|startup|company|class)\b"),
    ("setup", r"\b(laptop|pc|computer|gpu|rtx|windows|monitor|keyboard|mouse|headphones|phone|linux|mac)\b"),
    ("preferences", r"\b(likes?|loves?|hates?|dislikes?|prefers?|favou?rite|enjoys?|can't stand|into)\b"),
]

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"], 1)}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


def categorize(text: str) -> str:
    """The first rule that matches wins; the order puts the most specific categories first."""
    low = text.lower()
    has_date = find_date(low, dt.date(2000, 1, 1)) is not None
    for cat, pattern in _RULES:
        if re.search(pattern, low):
            # A person *and* a date ("sister visits next week") is about the schedule once it has a date.
            if cat == "people" and has_date and re.search(_RULES[2][1], low):
                return "schedule"
            return cat
    return "schedule" if has_date else "other"


def find_date(text: str, today: dt.date) -> dt.date | None:
    """The last day a dated fact is about, relative to the day it was said. None = no date in it."""
    low = text.lower()
    if m := re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", low):
        return _safe(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:\s+(20\d\d))?", low):
        return _with_year(int(m[1]), MONTHS[m[2]], m[3], today)
    if m := re.search(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b", low):
        return _with_year(int(m[2]), MONTHS[m[1]], m[3], today)
    if re.search(r"\b(today|tonight|this evening)\b", low):
        return today
    if re.search(r"\bday after tomorrow\b", low):  # before "tomorrow", which it contains
        return today + dt.timedelta(days=2)
    if re.search(r"\btomorrow\b", low):
        return today + dt.timedelta(days=1)
    if re.search(r"\bthis weekend\b", low):
        return today + dt.timedelta(days=(6 - today.weekday()) % 7)
    if re.search(r"\bnext week\b", low):
        return today + dt.timedelta(days=7 - today.weekday() + 6)  # the Sunday that ends next week
    if re.search(r"\bnext month\b", low):
        first = (today.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        return (first + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    for i, day in enumerate(WEEKDAYS):
        if re.search(rf"\b(?:on|this|next|coming|by)\s+{day}\b", low):
            # The next time that weekday comes round ("on Friday" said on a Friday means today).
            ahead = (i - today.weekday()) % 7
            if ahead == 0 and re.search(rf"\bnext\s+{day}\b", low):
                ahead = 7
            return today + dt.timedelta(days=ahead)
    return None


def expires_at(text: str, said_at: float) -> float | None:
    """Unix time after which the fact is out of date: the end of the day after its date. None = lasting."""
    today = dt.datetime.fromtimestamp(said_at).date()
    day = find_date(text, today)
    if day is None or day < today - dt.timedelta(days=1):
        return None  # no date, or a date in the past ("born on 3 May 2004"): not an event to expire
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(23, 59, 59))
    return end.timestamp()


def _safe(y: int, m: int, d: int) -> dt.date | None:
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def _with_year(day: int, month: int, year: str | None, today: dt.date) -> dt.date | None:
    if year:
        return _safe(int(year), month, day)
    guess = _safe(today.year, month, day)
    if guess and guess < today - dt.timedelta(days=60):  # "5 January" said in December means next year
        guess = _safe(today.year + 1, month, day)
    return guess
