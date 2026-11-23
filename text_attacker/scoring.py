"""Speaker profiles, trial scores and EER.

This module depends only on numpy. The scoring follows the official ASV
evaluation: one profile per enrolment speaker (the mean of the speaker's
utterance embeddings), cosine similarity between profile and trial
embedding, and trials without an embedding are skipped and counted. The EER
follows the SpeechBrain routine that the official evaluation calls.
"""
from collections import defaultdict

import numpy as np


def _unit(v):
    # a zero vector stays zero, so that its cosine with anything is 0, as in the official scorer
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def build_profiles(embeddings, speakers):
    """Average the unit-length embeddings of each speaker and renormalise.

    Args:
        embeddings: array [N, D], one row per enrolment utterance
        speakers: N speaker ids

    Returns:
        dict speaker id -> unit vector [D]
    """
    per_speaker = defaultdict(list)
    for emb, spk in zip(np.asarray(embeddings, dtype=np.float64), speakers):
        per_speaker[str(spk)].append(_unit(emb))
    return {spk: _unit(np.mean(embs, axis=0)) for spk, embs in per_speaker.items()}


def score_trials(profiles, trial_embeddings, pairs):
    """Cosine score of every (enrol_spk, trial_utt) pair.

    Args:
        profiles: dict speaker id -> unit vector, from ``build_profiles``
        trial_embeddings: dict utterance id -> vector
        pairs: list of (enrol_spk, trial_utt)

    Returns:
        scores: array with one score per kept pair
        kept: boolean array over ``pairs``, False where an embedding is missing
        skipped: number of pairs without a profile or a trial embedding
    """
    trial_unit = {}
    kept = np.zeros(len(pairs), dtype=bool)
    scores = []
    for i, (enrol, utt) in enumerate(pairs):
        if enrol not in profiles or utt not in trial_embeddings:
            continue
        if utt not in trial_unit:
            trial_unit[utt] = _unit(np.asarray(trial_embeddings[utt], dtype=np.float64))
        scores.append(float(profiles[enrol] @ trial_unit[utt]))
        kept[i] = True
    return np.array(scores), kept, int((~kept).sum())


def eer_with_ties(scores, labels):
    """Equal error rate in percent, and the number of trials whose score is not unique.

    Reproduces ``speechbrain.utils.metric_stats.EER``, which the official ASV
    evaluation calls, including its float32 arithmetic: scores are float32;
    candidate thresholds are the distinct scores and the midpoints between
    neighbours; a target is rejected when its score is <= the threshold and a
    nontarget accepted when its score is > the threshold; thresholds are
    visited in ascending order and one replaces the current choice only if
    its |FAR - FRR| is strictly smaller; EER = (FAR + FRR) / 2. The result
    does not depend on the order of the trials.
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        raise ValueError("EER needs at least one target and one nontarget trial")
    if np.isnan(scores).any():
        raise ValueError("Scores contain NaN")
    target, nontarget = np.sort(scores[labels]), np.sort(scores[~labels])
    distinct = np.unique(scores)
    midpoints = (distinct[:-1] + distinct[1:]) / np.float32(2)
    thresholds = np.sort(np.concatenate([distinct, midpoints]))
    rejected = np.searchsorted(target, thresholds, side="right")
    accepted = len(nontarget) - np.searchsorted(nontarget, thresholds, side="right")
    frr = rejected.astype(np.float32) / np.float32(len(target))
    far = accepted.astype(np.float32) / np.float32(len(nontarget))
    # SpeechBrain compares the float32 gap of a candidate with the gap of the
    # stored choice recomputed in double precision; keep exactly that.
    gap32 = np.abs(far - frr).astype(np.float64)
    gap64 = np.abs(far.astype(np.float64) - frr.astype(np.float64))
    best, best_gap = 0, gap64[0]
    for i, (candidate, stored) in enumerate(zip(gap32.tolist(), gap64.tolist())):
        if candidate < best_gap:
            best, best_gap = i, stored
    ties = int(len(scores) - len(distinct))
    return 100 * (float(far[best]) + float(frr[best])) / 2, ties


def eer(scores, labels):
    """Equal error rate in percent."""
    return eer_with_ties(scores, labels)[0]
