"""Speaker head of the text attacker.

The head maps a pooled sentence feature of the frozen text encoder to a
256-dimensional speaker embedding. For training it carries a speaker
classifier, either a plain softmax layer or an AAM-softmax layer; the
classifier is not needed to embed text.

A released head is a directory with two files:
    head.safetensors   the projection (and the classifier, if it was saved)
    head.json          architecture, training objective and provenance
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OBJECTIVES = ("softmax", "aam-softmax")


class Head(nn.Module):
    def __init__(self, speakers, objective, hidden=1024, bottleneck=512, embedding=256, scale=30.0, margin=0.2):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}', expected one of {OBJECTIVES}")
        self.objective = objective
        self.scale = scale
        self.margin = margin
        # the parameters are created in the same order as in the original
        # training code, so that a seed gives the same initialisation
        self.project = nn.Sequential(nn.Linear(hidden, bottleneck), nn.ReLU(), nn.Linear(bottleneck, embedding))
        if objective == "aam-softmax":
            self.weight = nn.Parameter(torch.empty(speakers, embedding))
            nn.init.xavier_normal_(self.weight)
        else:
            self.head = nn.Linear(embedding, speakers, bias=False)

    def embed(self, pooled):
        """Pooled encoder features [N, hidden] -> unit-length speaker embeddings [N, embedding]."""
        return F.normalize(self.project(pooled), dim=-1)

    def forward(self, pooled, targets=None):
        """Speaker logits. With AAM-softmax the margin is applied to ``targets`` when they are given."""
        z = self.embed(pooled)
        if self.objective == "softmax":
            return self.head(z)
        cos = F.linear(z, F.normalize(self.weight, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        if targets is None:
            return self.scale * cos
        sine = torch.sqrt((1 - cos.pow(2)).clamp(min=1e-2))
        with_margin = cos * math.cos(self.margin) - sine * math.sin(self.margin)
        with_margin = torch.where(cos > math.cos(math.pi - self.margin), with_margin,
                                  cos - self.margin * math.sin(self.margin))
        return self.scale * torch.where(F.one_hot(targets, cos.size(1)).bool(), with_margin, cos)


def save_head(head_dir, head, meta, with_classifier=False):
    """Write ``head.safetensors`` and ``head.json`` to ``head_dir``.

    ``meta`` is stored in head.json next to the architecture; it describes the
    training data and the text convention (see text_attacker/README.md).
    """
    from safetensors.torch import save_file

    head_dir = Path(head_dir)
    head_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()
             if with_classifier or k.startswith("project.")}
    save_file(state, str(head_dir / "head.safetensors"))
    linear_in, linear_out = head.project[0], head.project[2]
    info = dict(meta)
    info.update({"hidden": linear_in.in_features, "bottleneck": linear_in.out_features,
                 "embedding": linear_out.out_features, "objective": head.objective,
                 "speakers": int((head.weight if head.objective == "aam-softmax" else head.head.weight).shape[0]),
                 "scale": head.scale, "margin": head.margin, "has_classifier": with_classifier})
    with open(head_dir / "head.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_head(head_dir, map_location="cpu"):
    """Load a head saved by ``save_head``, in evaluation mode.

    Returns:
        head: Head; its classifier holds trained weights only if head.json says ``has_classifier``
        meta: content of head.json
    """
    from safetensors.torch import load_file

    head_dir = Path(head_dir)
    if not (head_dir / "head.json").is_file():
        raise FileNotFoundError(f"No head.json in {head_dir}. Download the heads with 01_download_text_attacker.sh "
                                f"or point head_dir to a trained head.")
    with open(head_dir / "head.json", encoding="utf-8") as f:
        meta = json.load(f)
    head = Head(speakers=meta["speakers"], objective=meta["objective"], hidden=meta["hidden"],
                bottleneck=meta["bottleneck"], embedding=meta["embedding"],
                scale=meta.get("scale", 30.0), margin=meta.get("margin", 0.2))
    state = load_file(str(head_dir / "head.safetensors"), device=str(map_location))
    missing, unexpected = head.load_state_dict(state, strict=False)
    classifier_keys = {"weight", "head.weight"}
    if unexpected or (set(missing) - classifier_keys) or (missing and meta.get("has_classifier")):
        raise RuntimeError(f"head.safetensors in {head_dir} does not match head.json "
                           f"(missing: {missing}, unexpected: {unexpected})")
    head.eval()
    return head, meta
