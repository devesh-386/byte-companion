"""Tool retrieval: show the model only the few tools that fit this message.

    at start-up:  every tool ─► "name: description" + its example requests ─► bge vectors
    per message:  user text ─► bge vector ─► best cosine per tool ─► top k tool names

A small model picks the right tool far more often from 6 choices than from 60, and the prompt stays short.
"""
import numpy as np

from .memory.retriever import Weights, bm25_scores, keywords
from .tools import ToolRegistry

KW_WEIGHT = 0.3   # tuned on evals/tool_queries.jsonl, checked on the held-out set
KW_HALF = 3.0


class ToolPicker:
    def __init__(self, embedder, registry: ToolRegistry, k: int = 6):
        self.embedder = embedder
        self.registry = registry
        self.k = k
        self._names: list[str] = []
        self._owner = np.zeros(0, dtype=np.int64)  # row -> tool index
        self._vecs = np.zeros((0, 1), dtype=np.float32)
        self.refresh()

    def refresh(self) -> None:
        """Re-embed after tools are added or removed."""
        texts, owner, self._words = [], [], []
        self._names = self.registry.names()
        for i, name in enumerate(self._names):
            tool = self.registry.get(name)
            texts.append(f"{name.replace('_', ' ')}: {tool.description}")
            owner.append(i)
            for ex in tool.examples:
                texts.append(ex)
                owner.append(i)
            params = " ".join(p.get("description", "") for p in tool.parameters["properties"].values())
            self._words.append(keywords(" ".join([name.replace("_", " "), tool.description, params, *tool.examples])))
        self._vecs = self.embedder.embed(texts)
        self._owner = np.array(owner, dtype=np.int64)

    def scores(self, text: str) -> dict[str, float]:
        """meaning (best bge cosine over description + examples, rescaled) + words (BM25, squashed).
        Meaning handles paraphrases; words catch literal hints like 'spotify', 'volume' or a URL."""
        q = self.embedder.embed([text], query=True)[0]
        sims = self._vecs @ q
        best = np.full(len(self._names), -1.0, dtype=np.float32)
        np.maximum.at(best, self._owner, sims)
        w = Weights()
        semantic = np.clip((best - w.cos_floor) / (w.cos_ceil - w.cos_floor), 0.0, 1.0)
        kw = bm25_scores(keywords(text), self._words)
        kw = kw / (kw + KW_HALF)
        return dict(zip(self._names, (semantic + KW_WEIGHT * kw).tolist()))

    def pick(self, text: str, k: int | None = None) -> list[str]:
        if self._names != self.registry.names():
            self.refresh()
        k = k or self.k
        if len(self._names) <= k:
            return list(self._names)
        ranked = sorted(self.scores(text).items(), key=lambda kv: -kv[1])
        return [name for name, _ in ranked[:k]]
