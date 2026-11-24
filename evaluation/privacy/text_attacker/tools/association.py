"""Acoustic-text association: do the ASV attacker and the text attacker confuse the same nontarget trials?

rho_dc is the Spearman correlation between the ASV scores and the text attacker scores of the nontarget trials,
after the mean of each enrollment speaker and of each trial utterance has been removed from both (double centering),
so that speakers or utterances that score high against everything do not count as agreement.

Example:
    python -m evaluation.privacy.text_attacker.tools.association --trials data/en_test_trials_mixed/trials \
        --asv_scores exp/asv_ssl/cosine_out/en_test_enrolls-en_test_trials_mixed/scores \
        --text_scores exp/text_attacker/results/track2/en_test_enrolls-en_test_trials_mixed/scores
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
from scipy.stats import spearmanr

from utils.data_io import read_table


def double_center(scores, enrol, utt):
    """Remove the enrollment-speaker and the trial-utterance mean from the scores."""
    scores = scores.copy()
    for _ in range(10):  # alternating means converge fast
        for group in (enrol, utt):
            _, index = np.unique(group, return_inverse=True)
            scores -= (np.bincount(index, scores) / np.bincount(index))[index]
    return scores


def score_association(asv_scores_file, text_scores_file, trials_file):
    """rho_dc of two score files of the ASV evaluation format on the nontarget trials of a trial list."""
    trials = read_table(trials_file, names=['enrol', 'utt', 'label'], dtype=str)
    trials = trials.drop_duplicates(['enrol', 'utt'], keep='last')
    for name, scores_file in (('asv', asv_scores_file), ('text', text_scores_file)):
        scores = read_table(scores_file, names=['enrol', 'utt', name], dtype=[str, str, float])
        trials = trials.merge(scores)
    nontarget = trials[trials['label'] == 'nontarget']
    asv = double_center(nontarget['asv'].values, nontarget['enrol'].values, nontarget['utt'].values)
    text = double_center(nontarget['text'].values, nontarget['enrol'].values, nontarget['utt'].values)
    return spearmanr(asv, text)[0]


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--trials', required=True, help='trial list')
    parser.add_argument('--asv_scores', required=True, help='scores file of the ASV evaluation')
    parser.add_argument('--text_scores', required=True, help='scores file of the text attacker')
    args = parser.parse_args()

    print(f'rho_dc={score_association(args.asv_scores, args.text_scores, args.trials):.3f}')
