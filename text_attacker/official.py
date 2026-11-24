"""Run the text attacker on the data layout and config format of this repository.

Enrollment always uses reference transcripts. The trial text is one of:
    ref        reference transcripts (the ``text`` file of the data directory)
    asr_orig   ASR transcripts of the original trial audio
    asr_anon   ASR transcripts of the anonymized trial audio The ASR transcripts are the
``text`` files that the ASR evaluation of run_evaluation.py writes.
"""
import itertools

import pandas as pd

from evaluation.privacy.asv.asv import ASV
from text_attacker.encoder import PooledEncoder
from text_attacker.extraction import TextExtraction
from text_attacker.head import load_head


class TextASV(ASV):
    """Trial scoring and EER of the ASV evaluation (cosine), on the embeddings of a TextExtraction."""

    def __init__(self, extractor, score_save_dir):
        self.extractor = extractor
        self.score_save_dir = score_save_dir
        self.distance = 'cosine'


def load_encoder(attacker_params, device):
    return PooledEncoder(backbone=attacker_params['backbone'], revision=attacker_params['backbone_revision'],
                         device=device)


def load_attacker(attacker_params, dataset_name, encoder, force_compute=False):
    """TextASV for a dataset; its head is chosen by the dataset prefix (libri, en, de, ...)."""
    eval_params = attacker_params['evaluation']
    head, meta = load_head(attacker_params['heads'][dataset_name.split('_')[0]]['head_dir'], device=encoder.device)
    score_save_dir = eval_params['results_dir'] / 'cosine_out'
    extractor = TextExtraction(encoder=encoder, head=head, max_length=meta['max_length'],
                               hyp_norm=meta['hyp_norm'] if eval_params['hyp_norm'] == 'head' else eval_params['hyp_norm'],
                               asr_text_dir=eval_params['asr_text_dir'], results_dir=score_save_dir / 'emb_text',
                               batch_size=eval_params['eval_batchsize'], force_compute=force_compute)
    score_save_dir.mkdir(parents=True, exist_ok=True)
    return TextASV(extractor=extractor, score_save_dir=score_save_dir)


def trial_text_dir(trial_source, data_dir, trial_name, dataset_name, asr_text_dir, anon_data_suffix):
    """Directory whose ``text`` file holds the trial transcripts of a condition."""
    if trial_source == 'ref':
        return data_dir / trial_name
    return asr_text_dir / f'{dataset_name}_asr{anon_data_suffix if trial_source == "asr_anon" else ""}'


def evaluate_text_attacker(params, device, force_compute=False):
    """Evaluate the text attacker as described by ``params['privacy']['text_attacker']``.

    Returns:
        DataFrame with one row per (enrollment set, trial set, trial text); also written to
        ``results_dir/results<anon_data_suffix>.csv``.
    """
    attacker_params = params['privacy']['text_attacker']
    eval_params = attacker_params['evaluation']
    data_dir = params['data_dir']
    anon_data_suffix = params['anon_data_suffix']
    encoder = load_encoder(attacker_params, device)
    results = []

    for name in attacker_params['dataset_name']:
        dataset = next(d for d in params['datasets'] if d['name'] == name)
        asv = load_attacker(attacker_params, name, encoder, force_compute=force_compute)

        for enroll, trial in itertools.product(dataset['enrolls'], dataset['trials']):
            enroll_name, trial_name = f'{dataset["data"]}{enroll}', f'{dataset["data"]}{trial}'
            for trial_source in eval_params['trial_text']:
                test_dir = trial_text_dir(trial_source, data_dir, trial_name, name, eval_params['asr_text_dir'],
                                          anon_data_suffix)
                # as in the ASV evaluation, the trial list is always the one of the original trial directory
                EER = asv.eer_compute(enrol_dir=data_dir / enroll_name, test_dir=test_dir,
                                      trial_runs_file=data_dir / trial_name / 'trials')
                EER = round(EER * 100, 3)

                print(f'{enroll_name}-{test_dir.name}: {trial_source}-EER={EER}')
                trials_info = trial_name.split('_')
                gender = trials_info[3]
                if 'common' in trial_name:
                    gender += '_common'
                results.append({'dataset': trials_info[0], 'split': trials_info[1], 'gender': gender,
                                'trial_text': trial_source, 'EER': EER})

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(eval_params['results_dir'] / f'results{anon_data_suffix}.csv', index=False)
    return results_df
