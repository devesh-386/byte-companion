from pathlib import Path

import numpy as np

# bge models were trained with this prefix on search queries (not on the stored passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Turns text into unit-length 384-number vectors using bge-small (runs on CPU)."""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir)).eval()
        self.dim = self.model.config.hidden_size

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with self._torch.no_grad():
            cls = self.model(**batch).last_hidden_state[:, 0]  # bge uses the [CLS] vector
        vecs = self._torch.nn.functional.normalize(cls, dim=-1)
        return vecs.numpy().astype(np.float32)
