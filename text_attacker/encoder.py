"""Frozen text encoder of the text attacker.

A sentence is represented by the mean of the last-layer token states of the
encoder (padding excluded). The encoder is never trained and always runs in
float32: half precision changes the EER of the released heads by several
points.
"""
import numpy as np
import torch

EMPTY_TEXT = "."  # stands in for an empty transcript, as when the released heads were trained


class PooledEncoder:
    def __init__(self, backbone="BAAI/bge-m3", revision=None, device="cpu", tokenizer=None, model=None):
        """
        Args:
            backbone: Hugging Face model id or local directory
            revision: commit of the model on the hub; ignored for a local directory
            device: device the encoder runs on
            tokenizer, model: ready objects to use instead of loading ``backbone`` (for tests)
        """
        self.backbone = str(backbone)
        self.revision = revision
        self.device = torch.device(device)
        if tokenizer is None or model is None:
            from transformers import AutoModel, AutoTokenizer  # imported late: only needed with a real backbone
            tokenizer = AutoTokenizer.from_pretrained(self.backbone, revision=revision)
            model = AutoModel.from_pretrained(self.backbone, revision=revision, torch_dtype=torch.float32)
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        dtypes = {p.dtype for p in self.model.parameters()}
        if dtypes - {torch.float32}:
            raise ValueError(f"The text encoder must run in float32, found {sorted(str(d) for d in dtypes)}")

    def fingerprint(self):
        """Identifies the features this encoder produces; part of the cache key."""
        return f"{self.backbone}@{self.revision or 'local'}|mean-pool|float32"

    def pool(self, texts, max_length=96, batch_size=32):
        """Mean-pooled last-layer states of each text.

        Texts are truncated to ``max_length`` tokens. Empty texts are replaced
        by ``EMPTY_TEXT``.

        Returns:
            float32 array [len(texts), hidden size]
        """
        texts = [t if t.strip() else EMPTY_TEXT for t in texts]
        features = None
        order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths together: little padding
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            batch = self.tokenizer([texts[i] for i in index], padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                states = self.model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"]).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            if features is None:
                features = np.zeros((len(texts), pooled.shape[-1]), dtype=np.float32)
            features[index] = pooled.float().cpu().numpy()
        if features is None:
            return np.zeros((0, 0), dtype=np.float32)
        return features
