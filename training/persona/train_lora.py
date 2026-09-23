"""Train Byte's personality as a LoRA adapter on Qwen2.5-3B-Instruct (QLoRA: 4-bit base, bf16 adapter).

What gets trained: ~30M adapter weights (about 1% of the model) sitting beside the frozen attention and
MLP layers. What gets graded: only the tokens of Byte's reply, never the system prompt or the user.

Run:  .venv/Scripts/python.exe training/persona/train_lora.py
Out:  artifacts/persona/adapter/ (PEFT format), artifacts/persona/train_log.csv
Then: .venv/Scripts/python.exe training/persona/export_lora.py   (-> GGUF for llama-server)
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
ASSISTANT_START = "<|im_start|>assistant\n"
END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
# Which layers get an adapter: attention q/k/v/o and the MLP, never the vision tower. Qwen2.5 has attention in
# every layer; Qwen3.5 only in every 4th (the rest are "linear attention"), but all 32 share the MLP.
TARGETS = r"^(?!.*visual).*\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"


def encode(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int]] | None:
    """Token ids + the positions that predict a token of the final assistant reply (incl. its <|im_end|>)."""
    text = tok.apply_chat_template(messages, tokenize=False)
    start = text.rindex(ASSISTANT_START) + len(ASSISTANT_START)
    # Qwen3.5 puts an empty "<think>\n\n</think>\n\n" at the start of every reply when thinking is off.
    # llama-server sends that as part of the prompt, so it isn't Byte's to learn: grade what follows it.
    if text.startswith(EMPTY_THINK, start):
        start += len(EMPTY_THINK)
    end = text.index(END, start) + len(END)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    targets = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a >= start and b <= end and b > a]
    # Position p predicts token p+1, so grade the positions just before each target token.
    predict_at = [t - 1 for t in targets if t > 0]
    return (ids, predict_at) if predict_at else None


def load_model(base: Path, rank: int, alpha: int, dropout: float):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(base), quantization_config=bnb, device_map={"": 0},
                                                 dtype=torch.bfloat16)
    print(f"loaded {type(model).__name__}", flush=True)
    model.config.use_cache = False
    # Recompute activations during backward instead of storing all 36 layers of them.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
                      target_modules=TARGETS)
    model = get_peft_model(model, lora)
    model.train()  # gradient checkpointing silently does nothing in eval mode
    return model


def sample_loss(model, ids: list[int], predict_at: list[int]) -> torch.Tensor:
    x = torch.tensor([ids], device="cuda")
    keep = torch.tensor(predict_at, device="cuda")
    # Only compute the 151k-way vocabulary projection where we actually grade (saves ~1 GB).
    logits = model(input_ids=x, logits_to_keep=keep).logits[0]
    return F.cross_entropy(logits.float(), x[0, keep + 1])


@torch.no_grad()
def evaluate(model, data) -> float:
    model.eval()
    losses = [sample_loss(model, ids, p).item() for ids, p in data]
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "persona" / "train.jsonl"))
    ap.add_argument("--base", default=str(ROOT / "models" / "Qwen2.5-3B-Instruct"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "persona"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    encoded, skipped = [], 0
    for r in rows:
        e = encode(tok, r["messages"], args.max_len)
        if e is None:
            skipped += 1
        else:
            encoded.append(e)
    random.shuffle(encoded)
    n_val = max(8, int(len(encoded) * args.val_frac))
    val, train = encoded[:n_val], encoded[n_val:]
    lengths = sorted(len(ids) for ids, _ in train)
    print(f"samples: train {len(train)}, val {len(val)}, skipped {skipped} (too long/empty); "
          f"tokens median {lengths[len(lengths) // 2]}, max {lengths[-1]}; "
          f"graded tokens {sum(len(p) for _, p in train)}", flush=True)

    model = load_model(Path(args.base), args.rank, args.alpha, args.dropout)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    total_steps = math.ceil(len(train) * args.epochs / args.grad_accum)
    warmup = max(1, min(20, total_steps // 10))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    log_path = out / "train_log.csv"
    log = csv.writer(open(log_path, "w", newline=""))
    log.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    val_loss = evaluate(model, val)
    print(f"step 0  val loss {val_loss:.4f} (before training)", flush=True)
    log.writerow([0, 0, "", round(val_loss, 4), 0, 0])

    step, micro, running, t0 = 0, 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        for ids, predict_at in train:
            loss = sample_loss(model, ids, predict_at)
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            micro += 1
            if micro % args.grad_accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == total_steps:
                elapsed = time.time() - t0
                eta = elapsed / step * (total_steps - step)
                avg = sum(running) / len(running)
                running.clear()
                print(f"step {step}/{total_steps}  epoch {epoch}  loss {avg:.4f}  lr {lr_at(step):.2e}  "
                      f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)
                log.writerow([step, epoch, round(avg, 4), "", f"{lr_at(step):.2e}", round(elapsed)])

        val_loss = evaluate(model, val)
        print(f"== epoch {epoch} done  val loss {val_loss:.4f}", flush=True)
        log.writerow([step, epoch, "", round(val_loss, 4), "", round(time.time() - t0)])
        model.save_pretrained(str(out / "adapter"))
        print(f"saved adapter -> {out / 'adapter'}", flush=True)

    print(f"TRAINING DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
