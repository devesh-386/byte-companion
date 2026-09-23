"""Long-term memory: remember, recall, forget. Only loads when the memory store is available."""
from typing import Annotated

from companion.tools import CHANGE, LOOK, OPEN, ToolError

ORDER = 50


def _day(ts: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ts).strftime("%a %d %b")


def register(registry, deps) -> None:
    store = deps.store
    if store is None:
        return

    def remember(fact: Annotated[str, "The fact to remember, as a short sentence about the user"]) -> str:
        """Save a lasting fact about the user to long-term memory. Use when they tell you something worth remembering or ask you to remember it."""
        _, is_new = store.add(fact, kind="fact")
        return "Saved." if is_new else "Updated an existing memory."

    def recall(query: Annotated[str, "What to look for"]) -> str:
        """Search long-term memory (facts and past conversations) for anything related to the query."""
        hits = store.search(query, k=6, min_score=0.3)
        if not hits:
            return "Nothing relevant in memory."
        return "\n".join(f"- ({h.kind}) {h.text}" for h in hits)

    def forget(description: Annotated[str, "Which remembered fact to delete"]) -> str:
        """Delete the remembered fact that best matches the description."""
        hits = store.search(description, k=1, kinds=["fact"], min_score=0.4)
        if not hits:
            raise ToolError("No remembered fact matches that closely enough.")
        store.delete(hits[0].id)
        return f"Forgot: {hits[0].text}"

    def memory_overview() -> str:
        """Everything you remember about the user, grouped (who they are, people, schedule, work, likes...).
        Use when they ask what you know or remember about them."""
        from companion.memory.structure import LABELS
        groups = store.overview()
        if not groups:
            return "You don't remember any facts about the user yet."
        lines = []
        for cat, mems in groups.items():
            lines.append(f"{LABELS.get(cat, cat)}:")
            lines += [f"- {m.text}" + (f" (until {_day(m.expires_at)})" if m.expires_at else "") for m in mems]
        return "\n".join(lines)

    def _dry_forget(description: str) -> str:
        hits = store.search(description, k=1, kinds=["fact"], min_score=0.4)
        if not hits:
            raise ToolError("No remembered fact matches that closely enough.")
        return f"Forgot: {hits[0].text}"

    registry.register(remember, tier=OPEN, dry=lambda fact: "Saved.", examples=(
        "remember that my exam is on Friday", "my favourite game is Valorant", "don't forget I like dark mode", "keep in mind that I dislike something", "note this fact about me"))
    registry.register(recall, tier=LOOK, examples=(
        "what did I tell you about my exam", "do you know my favourite game", "did I mention my sister"))
    registry.register(memory_overview, tier=LOOK, examples=(
        "what do you remember about me", "what do you know about me", "list everything you know about me"))
    registry.register(forget, tier=CHANGE, dry=_dry_forget, examples=(
        "forget what I said about my exam", "delete that memory", "stop remembering my address"))
