"""Train the emotion classifier from scratch: our own BPE tokenizer + our own Transformer.

Run:  .venv/Scripts/python training/emotion/train.py
Out:  artifacts/emotion/{tokenizer.json, model.pt, report.md, history.csv}
"""
import argparse
import collections
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from companion.ml.emotion_model import (EmotionModelConfig, EmotionTransformer,  # noqa: E402
                                        encode_batch, predict_probs)
from companion.ml.labels import REACTIONS  # noqa: E402

DATA = ROOT / "data" / "emotion"
OUT = ROOT / "artifacts" / "emotion"

# Messages people actually type to a desktop companion — nothing like these is in the training data.
PROBES = [
    "what time is it?",
    "open my downloads folder",
    "I GOT THE INTERNSHIP!!!",
    "ugh my code broke again, I've been at this for 3 hours",
    "my grandma passed away last night",
    "wait, the exam is TOMORROW??",
    "I'm really nervous about my presentation",
    "thanks, that actually helped a lot",
    "this is so stupid, nothing works",
    "whoa, I didn't know you could do that",
    "I feel kind of lonely today",
    "can you calculate 15% of 2400",
]


def load(split: str) -> list[dict]:
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=["[PAD]", "[UNK]"])
    tok.train_from_iterator(texts, trainer)
    assert tok.token_to_id("[PAD]") == 0
    return tok


def batches(rows: list[dict], size: int, shuffle: bool, rng: random.Random):
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        chunk = [rows[j] for j in idx[i:i + size]]
        yield [r["text"] for r in chunk], torch.tensor([REACTIONS.index(r["label"]) for r in chunk])


@torch.no_grad()
def evaluate(model, tok, rows, device, batch_size=512) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    preds, gold = [], []
    for texts, labels in batches(rows, batch_size, False, random.Random(0)):
        probs = predict_probs(model, tok, texts)
        preds += probs.argmax(-1).tolist()
        gold += labels.tolist()
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    return acc, f1_score(gold, preds, average="macro"), preds, gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT), help="where the model and report go")
    args = ap.parse_args()
    out = Path(args.out)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out.mkdir(parents=True, exist_ok=True)

    train, val, test = load("train"), load("val"), load("test")
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  device {device}")

    tok = train_tokenizer([r["text"] for r in train], args.vocab_size)
    tok.save(str(out / "tokenizer.json"))
    print(f"tokenizer: {tok.get_vocab_size()} tokens, e.g. {tok.encode('ugh my code broke again').tokens}")

    cfg = EmotionModelConfig(vocab_size=tok.get_vocab_size(), n_classes=len(REACTIONS), d_model=args.d_model,
                             n_layers=args.layers, n_heads=args.heads, d_ff=4 * args.d_model)
    model = EmotionTransformer(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters")

    # Rare classes get a bigger say in the loss (sqrt-inverse frequency, so it isn't extreme).
    counts = collections.Counter(r["label"] for r in train)
    weights = torch.tensor([math.sqrt(len(train) / counts[c]) for c in REACTIONS])
    weights = (weights / weights.mean()).to(device)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total, warmup = steps_per_epoch * args.epochs, steps_per_epoch // 2

    def lr_at(step: int) -> float:
        # Linear warm-up, then cosine decay to 5% of the peak.
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))

    history, best_f1, bad_epochs, step = [], -1.0, 0, 0
    use_amp = device == "cuda"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n_seen = time.time(), 0.0, 0
        for texts, labels in batches(train, args.batch_size, True, rng):
            ids = encode_batch(tok, texts, cfg.max_len).to(device)
            labels = labels.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids)
            loss = F.cross_entropy(logits.float(), labels, weight=weights, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            loss_sum += loss.item() * len(texts)
            n_seen += len(texts)

        val_acc, val_f1, _, _ = evaluate(model, tok, val, device)
        train_loss = loss_sum / n_seen
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_acc": round(val_acc, 4), "val_macro_f1": round(val_f1, 4)})
        flag = ""
        if val_f1 > best_f1:
            best_f1, bad_epochs, flag = val_f1, 0, "  *best*"
            torch.save({"config": cfg.to_dict(), "labels": REACTIONS, "state_dict": model.state_dict(),
                        "val_macro_f1": val_f1, "epoch": epoch}, out / "model.pt")
        else:
            bad_epochs += 1
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  val acc {val_acc:.3f}  "
              f"val macro-F1 {val_f1:.3f}  ({time.time() - t0:.1f}s){flag}")
        if bad_epochs >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping early")
            break

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    ckpt = torch.load(out / "model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_acc, test_f1, preds, gold = evaluate(model, tok, test, device)
    report = classification_report(gold, preds, target_names=REACTIONS, digits=3)
    # The v1 test rows on their own, so every version is compared on exactly the same sentences.
    v1_acc, v1_f1 = evaluate(model, tok, load("test_v1"), device)[:2] if (DATA / "test_v1.jsonl").exists() else (0, 0)
    cm = confusion_matrix(gold, preds)

    probe_probs = predict_probs(model, tok, PROBES)
    probe_lines = []
    for text, p in zip(PROBES, probe_probs):
        i = int(p.argmax())
        probe_lines.append(f"| {text} | {REACTIONS[i]} | {p[i]:.2f} |")

    cm_lines = ["| true \\ pred | " + " | ".join(REACTIONS) + " |",
                "|---" * (len(REACTIONS) + 1) + "|"]
    cm_lines += [f"| {REACTIONS[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)]
    (out / "report.md").write_text("\n".join([
        "# Emotion classifier report", "",
        f"- Parameters: {model.num_params() / 1e6:.2f}M, vocab {cfg.vocab_size}, best epoch {ckpt['epoch']}",
        f"- Test accuracy: **{test_acc:.3f}**, test macro-F1: **{test_f1:.3f}**",
        f"- On the v1 test set (GoEmotions + dair only): accuracy **{v1_acc:.3f}**, macro-F1 **{v1_f1:.3f}**",
        f"- Majority-class baseline accuracy (always 'joy'): {collections.Counter(gold).most_common(1)[0][1] / len(gold):.3f}",
        "", "## Per class (test)", "```", report, "```", "", "## Confusion matrix (test)", *cm_lines,
        "", "## Companion-style probes (not in training data)", "| message | prediction | confidence |",
        "|---|---|---|", *probe_lines, "",
    ]), encoding="utf-8")

    print(f"\nTEST accuracy {test_acc:.3f}  macro-F1 {test_f1:.3f}   v1-test accuracy {v1_acc:.3f}  macro-F1 {v1_f1:.3f}\n{report}")
    print("\n".join(probe_lines))


if __name__ == "__main__":
    main()
