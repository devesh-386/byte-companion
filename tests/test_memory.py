import hashlib
from pathlib import Path

import numpy as np
import pytest

from companion.memory.hook import FactExtractor, MemoryHook, format_memories
from companion.plugins import Deps, load_plugins
from companion.memory.retriever import bm25_scores, keywords, mmr, recency
from companion.memory.store import MemoryStore
from companion.tools import ToolRegistry

BGE = Path(__file__).resolve().parents[1] / "models" / "bge-small-en-v1.5"


class HashEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words point the same way."""
    dim = 64

    def embed(self, texts, query=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in keywords(t):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            n = np.linalg.norm(out[i])
            out[i] = out[i] / n if n else out[i]
        return out


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def test_keywords_drop_stopwords():
    assert keywords("What is my dog's name?") == ["dogs", "name"]


def test_bm25_prefers_rare_matching_word():
    docs = [["python", "code"], ["python", "bruno", "dog"], ["python", "tests"]]
    scores = bm25_scores(["bruno", "python"], docs)
    assert int(np.argmax(scores)) == 1


def test_recency_halves_every_half_life():
    assert np.allclose(recency(np.array([0.0, 30.0, 60.0])), [1.0, 0.5, 0.25])


def test_mmr_skips_near_duplicates():
    vecs = np.array([[1, 0], [0.999, 0.045], [0, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    picked = mmr(np.array([0.9, 0.89, 0.6]), vecs, k=2, diversity=0.5)
    assert picked == [0, 2]


def make_store(clock=None):
    from companion.memory.retriever import Weights
    # Hash vectors have no bge-style similarity floor, so use the raw cosine.
    return MemoryStore(":memory:", HashEmbedder(), clock=clock or Clock(),
                       weights=Weights(cos_floor=0.0, cos_ceil=1.0))


def test_store_add_search_and_dedupe():
    store = make_store()
    store.add("The user's dog is named Bruno")
    store.add("The user studies computer science at SRM")
    _, is_new = store.add("The user's dog is named Bruno!")
    assert not is_new and store.count("fact") == 2
    hits = store.search("dog named Bruno", k=1, min_score=0.1)
    assert "Bruno" in hits[0].text


def test_store_kind_filter_and_delete():
    store = make_store()
    fid, _ = store.add("The user likes chess")
    store.add("User: chess tonight?\nYou: sure", kind="episode")
    assert all(h.kind == "episode" for h in store.search("chess", kinds=["episode"], min_score=0.0))
    store.delete(fid)
    assert store.count("fact") == 0


def test_min_score_filters_unrelated():
    store = make_store()
    store.add("The user plays guitar")
    assert store.search("quantum chromodynamics lecture", min_score=0.5) == []


def test_memory_hook_recall_and_episode():
    clock = Clock()
    store = make_store(clock)
    store.add("The user plays guitar")
    submitted = []

    class FakeExtractor:
        def submit(self, text):
            submitted.append(text)

    hook = MemoryHook(store, FakeExtractor(), clock=clock, min_score=0.2)
    events = []
    ctx = hook.before_turn("guitar practice tips", events.append)
    assert "The user plays guitar" in ctx and events[0].kind == "memory"
    hook.after_turn("I practiced guitar for two hours", "Nice work!")
    assert store.count("episode") == 1 and submitted == ["I practiced guitar for two hours"]
    hook.after_turn("what time is it", "3pm")
    assert submitted == ["I practiced guitar for two hours"]  # no first person, no extraction


def test_format_memories_labels_age():
    from companion.memory.store import Memory
    text = format_memories([Memory(1, "episode", "User: hi", 0.0)], now=86400 * 3)
    assert "3 days ago" in text


def test_extractor_parses_facts():
    class FakeLLM:
        def chat(self, messages, **kw):
            return "-The user's sister is called Priya.\n- random line\n* The user has an exam on Friday."

    ex = FactExtractor.__new__(FactExtractor)
    ex.llm = FakeLLM()
    assert ex.extract("my sister priya... and my exam is friday") == ["The user's sister is called Priya.",
                                                                     "The user has an exam on Friday."]
    assert ex.extract("my sister priya...") == ["The user's sister is called Priya."]  # exam never said


def test_extractor_worth_checking():
    assert FactExtractor.worth_checking("my exam is on friday")
    assert not FactExtractor.worth_checking("what time is it")


def test_memory_tools():
    store = make_store()
    reg = ToolRegistry()
    load_plugins(reg, Deps(store=store), include={"memory"})
    assert reg.call("remember", {"fact": "The user loves biryani"}) == "Saved."
    assert "biryani" in reg.call("recall", {"query": "biryani"})
    from companion.tools import CHANGE
    assert reg.tier("forget") == CHANGE
    assert "Forgot" in reg.call("forget", {"description": "loves biryani"})


@pytest.mark.skipif(not BGE.exists(), reason="bge model not downloaded")
def test_real_embeddings_calibration():
    from companion.memory.embedder import Embedder
    store = MemoryStore(":memory:", Embedder(BGE))
    for fact in ["The user's dog is named Bruno",
                 "The user is building a local AI companion called Companion",
                 "The user has a Computer Networks exam on 3 September",
                 "The user prefers dark mode"]:
        store.add(fact)
    assert "Bruno" in store.search("what's my pet called?", k=1)[0].text
    assert "exam" in store.search("when is my networks test", k=1)[0].text
    assert store.search("what is 2 + 2", k=3) == []
    assert store.search("hello", k=3) == []
    assert store.add("The user's dog's name is Bruno")[1] is False  # paraphrase deduped



def test_extracted_facts_must_come_from_the_message():
    from companion.memory.hook import grounded
    # the real failure: examples parroted back for a question that contains no facts
    assert not grounded("The user's name is Sam.", "what do you remember about me?")
    assert not grounded("The user works at Infosys as a data analyst.", "what do you remember about me?")
    # real facts pass, including light paraphrase
    assert grounded("The user's name is Devesh.", "hey I'm Devesh by the way")
    assert grounded("The user has an exam on Friday.", "ugh my CN exam is on friday")
    assert grounded("The user plays Valorant.", "I play valorant every night")
    # an example answer is refused even when the words happen to match
    assert not grounded("The user is learning Rust.", "I'm learning rust")


def test_extractor_drops_parroted_examples():
    from companion.memory.hook import FactExtractor

    class Parrot:
        def chat(self, messages, **kw):
            return "- The user's name is Sam.\n- The user works at Infosys as a data analyst.\n- The user loves chess."
    ex = FactExtractor.__new__(FactExtractor)
    ex.llm = Parrot()
    assert ex.extract("honestly I love chess more than anything") == ["The user loves chess."]
