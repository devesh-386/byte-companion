import queue
import re
import threading
import time
from typing import Callable

from ..agent_events import Event
from .store import Memory, MemoryStore

EXTRACT_PROMPT = (
    "You pick out lasting facts about the user from one message they sent to their AI companion.\n"
    "A lasting fact stays true for weeks: their name, people in their life, preferences, projects, "
    "goals, skills, schedule, upcoming events with dates, their computer setup.\n"
    "Not facts: requests, questions, moods of the moment, anything about the assistant.\n"
    "Write each fact as one short third-person sentence starting with 'The user', one per line, "
    "each line starting with '- '. If there are none, reply exactly: NONE"
)
# Small models follow examples far better than instructions, so we show it the job first.
EXTRACT_EXAMPLES = [
    ("I work at Infosys as a data analyst and I'm learning Rust in my free time",
     "- The user works at Infosys as a data analyst.\n- The user is learning Rust."),
    ("can you open my downloads folder", "NONE"),
    ("ugh I'm so tired today", "NONE"),
    ("my sister Priya is visiting next week, she's a doctor. also I hate mornings",
     "- The user has a sister named Priya, who is a doctor.\n- The user's sister Priya is visiting next week.\n"
     "- The user hates mornings."),
    ("I'm Sam btw, my cat Mochi keeps sitting on my keyboard while I work on my game",
     "- The user's name is Sam.\n- The user has a cat named Mochi.\n- The user is making a game."),
]
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|my|me|mine|we|our)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+)$")


_EXAMPLE_FACTS = {line.lstrip("- ").strip().lower() for _, out in EXTRACT_EXAMPLES for line in out.splitlines()}
_GENERIC = {"user", "users", "name", "named", "called", "has", "likes", "loves", "hates", "wants", "working",
            "works", "learning", "making", "visiting", "week", "next", "sister", "brother", "friend"}


def grounded(fact: str, user_text: str) -> bool:
    """A fact must come from the user's own words, never from the prompt's examples.
    Qwen3.5-4B, asked about "what do you remember about me?", once returned the few-shot example answers
    ("The user's name is Sam", "works at Infosys") and they were saved. Rules, not the model, stop that:
      1. an exact copy of an example answer is rejected;
      2. at least one specific word of the fact (not 'user', 'name', 'likes'...) must appear in the message.
    """
    from .retriever import keywords
    if fact.strip().lower() in _EXAMPLE_FACTS:
        return False
    said = set(keywords(user_text))
    said |= {w.rstrip("s") for w in said}
    specific = [w for w in keywords(fact) if w not in _GENERIC and len(w) > 2]
    return any(w in said or w.rstrip("s") in said for w in specific)


def _age(seconds: float) -> str:
    days = seconds / 86400
    if days < 1 / 24:
        return "just now"
    if days < 1:
        return f"{int(days * 24)}h ago"
    if days < 2:
        return "yesterday"
    return f"{int(days)} days ago"


def format_memories(hits: list[Memory], now: float) -> str:
    lines = []
    for h in hits:
        tag = "fact" if h.kind == "fact" else f"conversation, {_age(now - h.created_at)}"
        lines.append(f"- [{tag}] {h.text}")
    return ("Things you remember from earlier (use them only if relevant; never claim to remember "
            "anything not listed):\n" + "\n".join(lines))


class FactExtractor:
    """Runs fact extraction on a background thread so the user never waits for it."""

    def __init__(self, llm, store: MemoryStore, on_saved: Callable[[list[str]], None] | None = None):
        self.llm = llm
        self.store = store
        self.on_saved = on_saved
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    @staticmethod
    def worth_checking(text: str) -> bool:
        return len(text) >= 12 and bool(_FIRST_PERSON.search(text))

    def submit(self, user_text: str) -> None:
        self._queue.put(user_text)

    def extract(self, user_text: str) -> list[str]:
        messages = [{"role": "system", "content": EXTRACT_PROMPT}]
        for example_in, example_out in EXTRACT_EXAMPLES:
            messages += [{"role": "user", "content": example_in}, {"role": "assistant", "content": example_out}]
        messages.append({"role": "user", "content": user_text})
        out = self.llm.chat(messages, temperature=0.0, max_tokens=150)
        facts = []
        for line in out.splitlines():
            m = _BULLET.match(line)
            if m and "user" in m.group(1).lower() and 10 < len(m.group(1)) < 200:
                fact = m.group(1).strip()
                if grounded(fact, user_text):
                    facts.append(fact)
        return facts[:5]

    def close(self, timeout: float = 5) -> None:
        self._queue.put(None)
        self._thread.join(timeout)

    def _work(self) -> None:
        while (text := self._queue.get()) is not None:
            try:
                facts = self.extract(text)
                saved = [f for f in facts if self.store.add(f, kind="fact")[1]]
                if saved and self.on_saved:
                    self.on_saved(saved)
            except Exception:
                # Memory is best-effort; a failed extraction must never break the conversation.
                pass


class MemoryHook:
    def __init__(self, store: MemoryStore, extractor: FactExtractor | None = None, k: int = 4,
                 clock: Callable[[], float] = time.time, min_score: float = 0.35):
        self.store = store
        self.extractor = extractor
        self.k = k
        self.clock = clock
        self.min_score = min_score

    def before_turn(self, user_text: str, emit) -> str | None:
        hits = self.store.search(user_text, k=self.k, min_score=self.min_score)
        emit(Event("memory", {"recalled": [{"text": h.text, "kind": h.kind, "score": round(h.score, 3)}
                                           for h in hits]}))
        return format_memories(hits, self.clock()) if hits else None

    def after_turn(self, user_text: str, answer: str) -> None:
        if not answer or answer.startswith("(Stopped"):
            return
        self.store.add(f"User: {user_text}\nYou: {answer[:400]}", kind="episode")
        if self.extractor and FactExtractor.worth_checking(user_text):
            self.extractor.submit(user_text)

