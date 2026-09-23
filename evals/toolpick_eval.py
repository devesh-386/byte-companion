"""How often does tool retrieval show the right tool? Phase 1 done-when: hit@6 >= 95% on 60 queries.

Run: .venv\\Scripts\\python.exe -m evals.toolpick_eval [--k 6]
"""
import argparse
import json
import tempfile
from pathlib import Path

from companion import config
from companion.memory.embedder import Embedder
from companion.memory.store import MemoryStore
from companion.plugins import Deps, load_plugins
from companion.toolpick import ToolPicker
from companion.tools import ToolRegistry

QUERIES = Path(__file__).resolve().parent / "tool_queries.jsonl"


def load_queries() -> list[dict]:
    return [json.loads(l) for l in QUERIES.read_text(encoding="utf-8").splitlines() if l.strip()]


def accepted(item: dict) -> list[str]:
    return item["tool"] if isinstance(item["tool"], list) else [item["tool"]]


def evaluate(picker: ToolPicker, queries: list[dict], k: int) -> tuple[float, float, list[str]]:
    hits = top1 = 0
    misses = []
    for item in queries:
        picked = picker.pick(item["q"], k=k)
        ok = accepted(item)  # some requests have more than one right first step
        hits += any(t in picked for t in ok)
        top1 += picked[0] in ok
        if not any(t in picked for t in ok):
            misses.append(f"{item['q']!r}: wanted {item['tool']}, got {picked}")
    return hits / len(queries), top1 / len(queries), misses


def build() -> ToolPicker:
    embedder = Embedder(config.EMBEDDER_DIR)
    store = MemoryStore(Path(tempfile.mkdtemp()) / "m.db", embedder)
    registry = ToolRegistry()
    load_plugins(registry, Deps(store=store, vision=True))  # every tool Byte can have
    return ToolPicker(embedder, registry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOOLS_PER_TURN or 6)
    args = ap.parse_args()
    picker = build()
    queries = load_queries()
    hit, top1, misses = evaluate(picker, queries, args.k)
    print(f"{len(picker.registry.names())} tools, {len(queries)} queries: "
          f"hit@{args.k} {hit:.1%}  top-1 {top1:.1%}")
    for m in misses:
        print("  miss", m)


if __name__ == "__main__":
    main()
