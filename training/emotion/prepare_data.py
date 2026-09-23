"""Download GoEmotions + dair-ai/emotion and map both onto the companion's 6 reactions.

Output: data/emotion/{train,val,test}.jsonl with {"text": ..., "label": ...}
Run:    .venv/Scripts/python training/emotion/prepare_data.py
"""
import collections
import json
import random
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "emotion"

sys.path.insert(0, str(ROOT))
from companion.ml.labels import REACTIONS  # noqa: E402

HF = "https://huggingface.co/datasets"
SOURCES = {
    "goemotions": f"{HF}/google-research-datasets/go_emotions/resolve/main/simplified/{{split}}-00000-of-00001.parquet",
    "dair": f"{HF}/dair-ai/emotion/resolve/main/split/{{split}}-00000-of-00001.parquet",
}
SPLITS = {"train": "train", "val": "validation", "test": "test"}

GOEMOTIONS_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
# GoEmotions' own Ekman grouping, with disgust folded into anger (the character has no
# separate "disgusted" face).
GOEMOTIONS_TO_REACTION = {
    **dict.fromkeys(["admiration", "amusement", "approval", "caring", "desire", "excitement",
                     "gratitude", "joy", "love", "optimism", "pride", "relief"], "joy"),
    **dict.fromkeys(["disappointment", "embarrassment", "grief", "remorse", "sadness"], "sadness"),
    **dict.fromkeys(["anger", "annoyance", "disapproval", "disgust"], "anger"),
    **dict.fromkeys(["fear", "nervousness"], "fear"),
    **dict.fromkeys(["confusion", "realization", "surprise"], "surprise"),
    # Curiosity is mostly questions; for a companion a question is not a surprise.
    **dict.fromkeys(["curiosity", "neutral"], "neutral"),
}
SEED = Path(__file__).resolve().parent / "companion_seed.tsv"
SEED_REPEATS = 8
DAIR_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
DAIR_TO_REACTION = {"sadness": "sadness", "joy": "joy", "love": "joy", "anger": "anger",
                    "fear": "fear", "surprise": "surprise"}


def download(source: str, split: str) -> Path:
    dest = RAW / source / f"{split}.parquet"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = SOURCES[source].format(split=split)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_goemotions(split: str) -> list[dict]:
    rows, dropped = [], 0
    for text, labels in zip(*pq.read_table(download("goemotions", split), columns=["text", "labels"]).to_pydict().values()):
        reactions = {GOEMOTIONS_TO_REACTION[GOEMOTIONS_NAMES[i]] for i in labels}
        # Comments tagged with emotions from different groups are ambiguous; drop them.
        if len(reactions) != 1:
            dropped += 1
            continue
        rows.append({"text": text.strip(), "label": reactions.pop()})
    print(f"  goemotions/{split}: kept {len(rows)}, dropped {dropped} multi-reaction")
    return rows


def load_dair(split: str) -> list[dict]:
    table = pq.read_table(download("dair", split), columns=["text", "label"]).to_pydict()
    rows = [{"text": t.strip(), "label": DAIR_TO_REACTION[DAIR_NAMES[l]]}
            for t, l in zip(table["text"], table["label"])]
    print(f"  dair/{split}: kept {len(rows)}")
    return rows


EXTRA = Path("D:/byte-data")  # v2 sources, downloaded to D: (see the vault note)
DD_NAMES = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
DD_TO_REACTION = {"neutral": "neutral", "anger": "anger", "disgust": "anger", "fear": "fear",
                  "happiness": "joy", "sadness": "sadness", "surprise": "surprise"}
TWEET_NAMES = ["anger", "joy", "optimism", "sadness"]
TWEET_TO_REACTION = {"anger": "anger", "joy": "joy", "optimism": "joy", "sadness": "sadness"}


def load_dailydialog(split: str, rng: random.Random) -> list[dict]:
    """Everyday two-person conversations, one emotion per line: the closest thing to chatting with Byte.
    ~80% of lines are 'no emotion', so neutral is capped at the number of emotional lines."""
    path = EXTRA / "dailydialog" / f"{split}.parquet"
    if not path.exists():
        return []
    t = pq.read_table(path, columns=["emotions", "utterances"]).to_pydict()
    emotional, neutral = [], []
    for emos, utts in zip(t["emotions"], t["utterances"]):
        for e, u in zip(emos, utts):
            text = " ".join(u.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
                            .replace(" ' ", "'").replace(" ’ ", "'").split())
            if 3 <= len(text) <= 300:
                (neutral if e == 0 else emotional).append({"text": text, "label": DD_TO_REACTION[DD_NAMES[e]]})
    rng.shuffle(neutral)
    rows = emotional + neutral[:len(emotional)]
    print(f"  dailydialog/{split}: {len(emotional)} emotional + {min(len(neutral), len(emotional))} neutral")
    return rows


def load_tweet_eval(split: str) -> list[dict]:
    path = EXTRA / "tweet_eval" / f"{split}.parquet"
    if not path.exists():
        return []
    t = pq.read_table(path, columns=["text", "label"]).to_pydict()
    rows = [{"text": x.strip(), "label": TWEET_TO_REACTION[TWEET_NAMES[l]]} for x, l in zip(t["text"], t["label"])]
    print(f"  tweet_eval/{split}: kept {len(rows)}")
    return rows


def load_seed() -> list[dict]:
    """Hand-written companion-style messages. Tiny, so repeated to be heard over 55k Reddit rows."""
    rows = []
    for line in SEED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            label, text = line.split("\t", 1)
            rows.append({"text": text.strip(), "label": label})
    print(f"  companion seed: {len(rows)} rows x{SEED_REPEATS}")
    return rows * SEED_REPEATS


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1337)
    for ours, theirs in SPLITS.items():
        base = load_goemotions(theirs) + load_dair(theirs)
        if ours == "test":
            # The v1 test set, kept on its own so old and new models are compared on identical rows.
            with open(OUT / "test_v1.jsonl", "w", encoding="utf-8") as f:
                for r in base:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows = base + load_dailydialog(theirs, rng) + load_tweet_eval(theirs)
        if ours == "train":
            rows += load_seed()
        rows = [r for r in rows if r["text"]]
        assert all(r["label"] in REACTIONS for r in rows)
        rng.shuffle(rows)
        with open(OUT / f"{ours}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = collections.Counter(r["label"] for r in rows)
        print(f"{ours}: {len(rows)} rows  " + "  ".join(f"{k}={counts[k]}" for k in REACTIONS))


if __name__ == "__main__":
    main()
