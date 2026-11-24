"""Convert a training checkpoint of the text attacker into a released head.

The original checkpoints (best.pt) store the frozen encoder next to the
head, 2.27 GB each. This tool keeps the projection only (2.6 MB) and writes
head.safetensors and head.json. It is the only place that unpickles a
checkpoint, so run it on checkpoints you trust.

Example:
    python -m text_attacker.tools.export_head --checkpoint best.pt --out_dir exp/text_attacker/heads/libri \
        --name libri --hyp_norm upper_letters --train_corpus "LibriSpeech train-clean-360" --train_text reference
"""
import argparse
import hashlib
from pathlib import Path

import torch

from text_attacker.head import Head, save_head

BACKBONE = "BAAI/bge-m3"
BACKBONE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def head_from_checkpoint(path):
    """Build a Head from the 'model' state dict of a training checkpoint, ignoring the encoder."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = {k: v for k, v in saved["model"].items() if not k.startswith("encoder.")}
    objective = "aam-softmax" if "weight" in state else "softmax"
    classifier = state["weight"] if objective == "aam-softmax" else state["head.weight"]
    head = Head(speakers=classifier.shape[0], objective=objective)
    head.load_state_dict(state)
    head.eval()
    return head, saved.get("config", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--name", required=True, help="head name, e.g. libri, en, de")
    parser.add_argument("--hyp_norm", required=True, choices=["none", "upper_letters", "lower_letters"],
                        help="normalization that brings ASR hypotheses to the training text convention")
    parser.add_argument("--train_corpus", required=True)
    parser.add_argument("--train_text", required=True, help="reference or the ASR system of the pseudo-transcripts")
    parser.add_argument("--with_classifier", action="store_true")
    args = parser.parse_args()

    head, config = head_from_checkpoint(args.checkpoint)
    meta = {"name": args.name, "backbone_id": BACKBONE, "backbone_revision": BACKBONE_REVISION, "pooling": "mean",
            "max_length": int(config.get("max_length", 96)), "hyp_norm": args.hyp_norm,
            "train_corpus": args.train_corpus, "train_text": args.train_text,
            "source_checkpoint_sha256": sha256_of(args.checkpoint)}
    save_head(args.out_dir, head, meta, with_classifier=args.with_classifier)
    print(f"{args.name}: objective={head.objective} max_length={meta['max_length']} "
          f"head_sha256={sha256_of(args.out_dir / 'head.safetensors')}")


if __name__ == "__main__":
    main()
