from pathlib import Path

import torch
from tokenizers import Tokenizer

from .emotion_model import EmotionModelConfig, EmotionTransformer, predict_probs


class EmotionClassifier:
    """Loads the trained model once and runs it on the CPU (it's small; the GPU belongs to the LLM).

    Two kinds of artifact folder:
      head.pt + encoder/    fine-tuned bge-small (v2, the default: 77.4% on the v1 test set)
      model.pt + tokenizer  the from-scratch Transformer (v1: 73.7%), kept as the learning version
    """

    def __init__(self, artifact_dir: Path):
        if (artifact_dir / "head.pt").exists():
            from transformers import AutoModel, AutoTokenizer
            ckpt = torch.load(artifact_dir / "head.pt", map_location="cpu", weights_only=True)
            self.labels: list[str] = ckpt["labels"]
            self.encoder = AutoModel.from_pretrained(str(artifact_dir / "encoder")).eval()
            self.hf_tok = AutoTokenizer.from_pretrained(str(artifact_dir / "encoder"))
            self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(self.labels))
            self.head.load_state_dict(ckpt["head"])
            self.head.eval()
            self.kind = "bge"
            return
        ckpt = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        self.labels = ckpt["labels"]
        self.model = EmotionTransformer(EmotionModelConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
        self.kind = "scratch"

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        if self.kind == "bge":
            with torch.no_grad():
                enc = self.hf_tok([text], truncation=True, max_length=64, return_tensors="pt")
                probs = self.head(self.encoder(**enc).last_hidden_state[:, 0]).softmax(-1)[0]
        else:
            probs = predict_probs(self.model, self.tokenizer, [text])[0]
        i = int(probs.argmax())
        return self.labels[i], float(probs[i]), {l: round(float(p), 3) for l, p in zip(self.labels, probs)}
