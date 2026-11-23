"""Readers and writers for the Kaldi-style files the text attacker uses.

This module depends only on the standard library and numpy, so that scoring
and its regression checks run without torch.
"""
from pathlib import Path

import numpy as np


def read_pairs(path):
    """Read a two-column Kaldi file (``text``, ``utt2spk``) into a dict.

    Each line is split at the first space; the value keeps its inner spacing.
    """
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, _, value = line.partition(" ")
            out[key] = value.strip()
    return out


def read_trials(path):
    """Read an official trial list: ``<enrol_spk> <trial_utt> target|nontarget``.

    A pair that is listed more than once keeps the label of its last
    occurrence, which is what the official ASV scorer does (it stores the
    labels in a dict keyed by the pair). The pairs are returned in order of
    first occurrence.

    Returns:
        pairs: list of (enrol_spk, trial_utt)
        labels: int array, 1 for target and 0 for nontarget
    """
    labels = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            enrol, utt, label = line.split()
            labels[(enrol, utt)] = int(label == "target")
    pairs = list(labels)
    return pairs, np.array([labels[p] for p in pairs], dtype=int)


def read_table(path):
    """Read a two-column tab-separated file ``<utt>\\t<text>`` into a dict.

    A header line whose first field is ``uttid`` is skipped.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines and lines[0].split("\t")[0].strip() == "uttid":
        lines = lines[1:]
    out = {}
    for line in lines:
        key, _, value = line.partition("\t")
        if key.strip():
            out[key.strip()] = value.strip()
    return out


def write_scores(path, pairs, scores):
    """Write ``<enrol_spk> <trial_utt> <score>`` lines, as the ASV scorer does."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for (enrol, utt), score in zip(pairs, scores):
            f.write(f"{enrol} {utt} {float(score)}\n")
