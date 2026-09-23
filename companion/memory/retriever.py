"""Ranking memories. Pure functions over numpy arrays so every step can be tested and inspected.

final score = w_sem * cosine(meaning)  +  w_kw * BM25(keywords, normalised)  +  w_rec * recency
then MMR picks results that are relevant but not near-duplicates of each other.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the and or but if of to in on at for with from by is are was were be been am i me my you your "
    "we our it its this that these those do does did have has had what which who how when where why can "
    "could would should will just so not no yes about as up out into than then there here they them he "
    "she his her im ive dont".split()
)


def keywords(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower().replace("'", "")) if w not in STOPWORDS]


def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Classic BM25: rare words that appear in a doc count a lot; long docs are gently penalised."""
    n = len(docs)
    if n == 0 or not query:
        return np.zeros(n)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df = Counter(w for d in docs for w in set(d))
    scores = np.zeros(n)
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        for w in set(query):
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            scores[i] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(doc) / avg_len))
    return scores


def recency(age_days: np.ndarray, half_life_days: float = 30.0) -> np.ndarray:
    return np.power(0.5, np.maximum(age_days, 0) / half_life_days)


@dataclass
class Weights:
    semantic: float = 0.75
    keyword: float = 0.15
    recency: float = 0.10
    # bge rates even unrelated sentences ~0.35-0.52 and real matches ~0.6-0.8 (measured), so the
    # raw cosine is rescaled: <= cos_floor counts as no match, >= cos_ceil as a perfect one.
    cos_floor: float = 0.45
    cos_ceil: float = 0.75
    # BM25 -> 0..1 with s/(s+kw_half): one shared common word can't look like a perfect match.
    kw_half: float = 3.0


def hybrid_scores(query_vec: np.ndarray, doc_vecs: np.ndarray, query_words: list[str],
                  doc_words: list[list[str]], age_days: np.ndarray, w: Weights = Weights()) -> np.ndarray:
    if len(doc_vecs) == 0:
        return np.zeros(0)
    cosine = doc_vecs @ query_vec  # vectors are unit length, so a dot product is the cosine
    semantic = np.clip((cosine - w.cos_floor) / (w.cos_ceil - w.cos_floor), 0.0, 1.0)
    kw = bm25_scores(query_words, doc_words)
    kw = kw / (kw + w.kw_half)
    return w.semantic * semantic + w.keyword * kw + w.recency * recency(age_days)


def mmr(scores: np.ndarray, doc_vecs: np.ndarray, k: int, diversity: float = 0.3,
        min_score: float = -1.0) -> list[int]:
    """Maximal Marginal Relevance: greedily take the best doc, penalising similarity to ones already taken."""
    candidates = [i for i in np.argsort(-scores) if scores[i] >= min_score]
    chosen: list[int] = []
    while candidates and len(chosen) < k:
        if not chosen:
            best = candidates[0]
        else:
            sim_to_chosen = (doc_vecs[candidates] @ doc_vecs[chosen].T).max(axis=1)
            mmr_scores = (1 - diversity) * scores[candidates] - diversity * sim_to_chosen
            best = candidates[int(np.argmax(mmr_scores))]
        chosen.append(int(best))
        candidates.remove(best)
    return chosen
