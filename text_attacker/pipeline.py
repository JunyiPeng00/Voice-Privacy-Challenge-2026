"""From transcripts to speaker embeddings and an EER.

This module knows nothing about dataset names, directory layouts or config
files: it works on dicts of ``utterance id -> text`` and on trial pairs.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

from text_attacker import scoring


def _cache_key(encoder, ids, texts, max_length):
    digest = hashlib.sha256(f"{encoder.fingerprint()}|{max_length}".encode("utf-8"))
    for utt, text in zip(ids, texts):
        digest.update(f"\n{utt}\t{text}".encode("utf-8"))
    return digest.hexdigest()[:16]


def cached_pool(encoder, ids, texts, max_length=96, batch_size=32, cache_dir=None):
    """``encoder.pool`` with an optional cache on disk.

    The cache file is named after a hash of the encoder, ``max_length`` and
    the ordered (id, text) pairs, so a changed text never hits a stale entry.
    """
    if cache_dir is None:
        return encoder.pool(texts, max_length=max_length, batch_size=batch_size)
    cache_file = Path(cache_dir) / f"pooled_{_cache_key(encoder, ids, texts, max_length)}.npy"
    if cache_file.is_file():
        return np.load(cache_file)
    features = encoder.pool(texts, max_length=max_length, batch_size=batch_size)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, features)
    return features


def embed_texts(encoder, head, texts, max_length=96, batch_size=32, cache_dir=None):
    """Speaker embedding of every transcript.

    Args:
        texts: dict utterance id -> text

    Returns:
        dict utterance id -> unit-length float32 vector
    """
    ids = sorted(texts)
    if not ids:
        return {}
    pooled = cached_pool(encoder, ids, [texts[i] for i in ids], max_length, batch_size, cache_dir)
    device = next(head.parameters()).device
    with torch.inference_mode():
        emb = head.embed(torch.from_numpy(pooled).to(device)).float().cpu().numpy()
    return dict(zip(ids, emb))


def evaluate_pair(enrol_text, enrol_utt2spk, trial_text, pairs, labels, encoder, head, max_length=96,
                  batch_size=32, cache_dir=None, eer_fn=None):
    """Score one enrolment set against one trial set.

    Args:
        enrol_text: dict enrolment utterance id -> text
        enrol_utt2spk: dict enrolment utterance id -> speaker id
        trial_text: dict trial utterance id -> text; trials whose utterance is missing here are skipped
        pairs, labels: trial list as returned by ``kaldi_io.read_trials``
        eer_fn: optional ``f(target_scores, nontarget_scores) -> EER in percent`` for the reported value;
            the numpy EER of ``scoring`` is used when it is not given

    Returns:
        dict with eer, eer_numpy, n_trials, n_target, n_nontarget, n_skipped, n_ties,
        and pairs / labels / scores of the kept trials
    """
    enrol_ids = sorted(enrol_text)
    enrol_emb = embed_texts(encoder, head, enrol_text, max_length, batch_size, cache_dir)
    profiles = scoring.build_profiles([enrol_emb[u] for u in enrol_ids], [enrol_utt2spk[u] for u in enrol_ids])
    needed = {utt for _, utt in pairs}
    trial_emb = embed_texts(encoder, head, {u: t for u, t in trial_text.items() if u in needed},
                            max_length, batch_size, cache_dir)
    scores, kept, skipped = scoring.score_trials(profiles, trial_emb, pairs)
    labels = np.asarray(labels)[kept]
    eer_numpy, ties = scoring.eer_with_ties(scores, labels)
    eer = eer_fn(scores[labels == 1], scores[labels == 0]) if eer_fn is not None else eer_numpy
    return {"eer": float(eer), "eer_numpy": eer_numpy, "n_trials": int(kept.sum()), "n_target": int(labels.sum()),
            "n_nontarget": int((labels == 0).sum()), "n_skipped": skipped, "n_ties": ties,
            "pairs": [p for p, k in zip(pairs, kept) if k], "labels": labels, "scores": scores}


def save_embeddings(path, embeddings, speakers=None, role=""):
    """Save a dict of embeddings as npz with the arrays keys, emb, speaker, role."""
    keys = sorted(embeddings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array(keys), emb=np.stack([embeddings[k] for k in keys]),
             speaker=np.array([(speakers or {}).get(k, "") for k in keys]), role=np.array([role] * len(keys)))


def load_embeddings(path):
    """Inverse of ``save_embeddings``; returns (embeddings, speakers)."""
    data = np.load(path)
    return dict(zip(data["keys"], data["emb"])), dict(zip(data["keys"], data["speaker"]))
