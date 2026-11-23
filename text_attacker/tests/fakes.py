"""A tiny tokenizer and encoder, so that tests need neither a download nor a GPU."""
from types import SimpleNamespace

import torch


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    """One token per whitespace-separated word, id = 1 + (sum of code points mod 50); 0 is padding."""

    def __init__(self):
        self.seen = []

    def __call__(self, texts, padding=True, truncation=True, max_length=96, return_tensors="pt"):
        self.seen.extend(texts)
        rows = [[1 + sum(map(ord, word)) % 50 for word in text.split()][:max_length] for text in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        return FakeBatch(input_ids=ids, attention_mask=(ids > 0).long())


class FakeModel(torch.nn.Module):
    def __init__(self, hidden=16, activation="gelu"):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = torch.nn.Embedding(51, hidden)
        settings = {"hidden_size": hidden, "hidden_act": activation, "_name_or_path": "/somewhere", "transformers_version": "0"}
        self.config = SimpleNamespace(to_dict=lambda: dict(settings))

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
