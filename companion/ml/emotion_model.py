"""A small Transformer encoder for sentence classification, written from the ground up.

text -> token ids -> embeddings (+ position) -> N x [self-attention -> feed-forward] -> mean-pool -> class logits
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmotionModelConfig:
    vocab_size: int
    n_classes: int
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 768
    max_len: int = 64
    dropout: float = 0.1
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # One matmul makes queries, keys and values for every head: (3, B, heads, T, d_head).
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        # How much each token should look at every other token, scaled so softmax stays soft.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)          # (B, heads, T, T)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))  # never look at padding
        weights = self.attn_drop(scores.softmax(dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(B, T, D)                  # re-join the heads
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: EmotionModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual blocks: each sub-layer adds a correction to x rather than replacing it.
        x = x + self.drop(self.attn(self.ln1(x), pad_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class EmotionTransformer(nn.Module):
    def __init__(self, cfg: EmotionModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pad_mask = ids == self.cfg.pad_id
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos)[None])
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_f(x)
        # Average the real (non-padding) token vectors into one sentence vector.
        keep = (~pad_mask).unsqueeze(-1).float()
        sentence = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(sentence)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def encode_batch(tokenizer, texts: list[str], max_len: int, pad_id: int = 0) -> torch.Tensor:
    encoded = tokenizer.encode_batch(texts)
    length = max(1, min(max_len, max(len(e.ids) for e in encoded)))
    batch = torch.full((len(texts), length), pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        ids = e.ids[:length]
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


def predict_probs(model: EmotionTransformer, tokenizer, texts: list[str]) -> torch.Tensor:
    ids = encode_batch(tokenizer, texts, model.cfg.max_len, model.cfg.pad_id)
    ids = ids.to(next(model.parameters()).device)
    with torch.no_grad():
        return F.softmax(model(ids).float(), dim=-1).cpu()
