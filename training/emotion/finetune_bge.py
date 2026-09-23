"""Emotion model, version C: fine-tune a *pretrained* encoder (bge-small, 33M params, already on disk for
memory) instead of training from scratch. Same data and same test sets as train.py, so the numbers compare.

    text ─► bge-small (pretrained on huge amounts of English) ─► [CLS] vector ─► linear layer ─► 6 reactions
                    └─ all of it is fine-tuned, with a smaller learning rate than the new head

Run:  .venv/Scripts/python.exe training/emotion/finetune_bge.py
Out:  artifacts/emotion_bge/{encoder/, head.pt, report.md}
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from companion import config  # noqa: E402
from companion.ml.labels import REACTIONS  # noqa: E402

DATA = ROOT / "data" / "emotion"
OUT = ROOT / "artifacts" / "emotion_bge"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import PROBES, load  # noqa: E402


class Classifier(torch.nn.Module):
    def __init__(self, encoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(encoder.config.hidden_size, n_classes)

    def forward(self, **batch):
        cls = self.encoder(**batch).last_hidden_state[:, 0]
        return self.head(self.drop(cls))


def run_eval(model, tok, rows, device, bs=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            enc = tok([r["text"] for r in rows[i:i + bs]], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device)
            preds += model(**enc).argmax(-1).tolist()
    gold = [REACTIONS.index(r["label"]) for r in rows]
    return sum(p == g for p, g in zip(preds, gold)) / len(gold), f1_score(gold, preds, average="macro"), preds, gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()
    torch.manual_seed(1337)
    rng = random.Random(1337)
    device = "cuda"
    train, val, test = load("train"), load("val"), load("test")
    test_v1 = load("test_v1")
    tok = AutoTokenizer.from_pretrained(str(config.EMBEDDER_DIR))
    model = Classifier(AutoModel.from_pretrained(str(config.EMBEDDER_DIR)), len(REACTIONS)).to(device)

    counts = collections.Counter(r["label"] for r in train)
    weights = torch.tensor([math.sqrt(len(train) / counts[c]) for c in REACTIONS])
    weights = (weights / weights.mean()).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.lr * 30}], weight_decay=0.01)
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 30], total_steps=steps,
                                                pct_start=0.06, anneal_strategy="cos")
    best, t0 = -1.0, time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            rows = train[i:i + args.batch_size]
            enc = tok([r["text"] for r in rows], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device)
            y = torch.tensor([REACTIONS.index(r["label"]) for r in rows], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(**enc).float(), y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        acc, f1, _, _ = run_eval(model, tok, val, device)
        flag = ""
        if f1 > best:
            best, flag = f1, "  *best*"
            model.encoder.save_pretrained(OUT / "encoder")
            tok.save_pretrained(OUT / "encoder")
            torch.save({"head": model.head.state_dict(), "labels": REACTIONS}, OUT / "head.pt")
        print(f"epoch {epoch}  val acc {acc:.3f}  val macro-F1 {f1:.3f}  ({time.time() - t0:.0f}s){flag}", flush=True)

    model.encoder = AutoModel.from_pretrained(str(OUT / "encoder")).to(device)
    model.head.load_state_dict(torch.load(OUT / "head.pt")["head"])
    acc, f1, preds, gold = run_eval(model, tok, test, device)
    acc1, f11, _, _ = run_eval(model, tok, test_v1, device)
    report = classification_report(gold, preds, target_names=REACTIONS, digits=3)
    probe_lines = []
    model.eval()
    with torch.no_grad():
        enc = tok(PROBES, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        probs = model(**enc).softmax(-1).cpu()
    for text, p in zip(PROBES, probs):
        i = int(p.argmax())
        probe_lines.append(f"| {text} | {REACTIONS[i]} | {p[i]:.2f} |")
    (OUT / "report.md").write_text("\n".join([
        "# Emotion classifier report (fine-tuned bge-small)", "",
        f"- Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (pretrained encoder + new head)",
        f"- Test accuracy: **{acc:.3f}**, test macro-F1: **{f1:.3f}**",
        f"- On the v1 test set (GoEmotions + dair only): accuracy **{acc1:.3f}**, macro-F1 **{f11:.3f}**",
        "", "## Per class (test)", "```", report, "```", "",
        "## Companion-style probes", "| message | prediction | confidence |", "|---|---|---|", *probe_lines, ""]),
        encoding="utf-8")
    print(f"\nTEST accuracy {acc:.3f}  macro-F1 {f1:.3f}   v1-test accuracy {acc1:.3f}  macro-F1 {f11:.3f}\n{report}")
    print("\n".join(probe_lines))


if __name__ == "__main__":
    main()
