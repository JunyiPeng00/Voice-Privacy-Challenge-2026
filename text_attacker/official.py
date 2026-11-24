"""Run the text attacker on the data layout and config format of this repository.

This is the only module of the package that knows about config files,
dataset names, the ``data`` and ``exp`` directories and the anonymization
suffix. Everything below it works on plain dicts and arrays.

Attack conditions are pairs [enrollment text, trial text]:
    ref        reference transcripts (the ``text`` file of the data directory)
    asr_orig   ASR transcripts of the original trial audio
    asr_anon   ASR transcripts of the anonymized trial audio
Enrollment always uses reference transcripts. The ASR transcripts are the
``text`` files that the ASR evaluation of run_evaluation.py writes.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from text_attacker import kaldi_io, pipeline, textnorm
from text_attacker.encoder import PooledEncoder
from text_attacker.head import load_head
from utils import read_kaldi_format, setup_logger

logger = setup_logger(__name__)

REFERENCE_DIR = Path(__file__).parent / 'reference'
TRIAL_SOURCES = ('ref', 'asr_orig', 'asr_anon')


def _speechbrain_eer():
    """EER function of the official ASV evaluation, or None if SpeechBrain cannot be imported."""
    try:
        import torch
        from speechbrain.utils.metric_stats import EER
    except ImportError:
        logger.warning('SpeechBrain is not available, the EER is computed with the numpy implementation.')
        return None
    return lambda target, nontarget: 100 * EER(torch.tensor(target), torch.tensor(nontarget))[0]


def _reference_counts():
    """(track, set, split) -> (n_target, n_nontarget) of the official trial lists the reference EERs belong to."""
    counts = {}
    lines = [line for line in (REFERENCE_DIR / 'eer_reference.tsv').read_text().splitlines()
             if line and not line.startswith('#')]
    for row in (line.split('\t') for line in lines[1:]):
        if row[3] == 'trials':
            counts[(row[0], row[1], row[2])] = (int(row[6]), int(row[7]))
    return counts


def check_canary(encoder, head, meta, tolerance=1e-3):
    """Embed a few fixed sentences and compare with stored values.

    This catches a wrong backbone, a changed tokenizer, half precision or a
    different pooling before any result is written.
    """
    canary_file = REFERENCE_DIR / 'canary.json'
    name = meta.get('name')
    if not canary_file.is_file():
        return
    probes = json.loads(canary_file.read_text(encoding='utf-8'))['heads'].get(name)
    if probes is None:
        logger.info(f'No canary for head "{name}", skipping the check.')
        return
    emb = pipeline.embed_texts(encoder, head, {i: p['text'] for i, p in enumerate(probes)}, meta['max_length'])
    deviation = max(float(np.abs(emb[i][:len(p['embedding_head'])] - np.array(p['embedding_head'])).max())
                    for i, p in enumerate(probes))
    if deviation > tolerance:
        raise RuntimeError(f'Head "{name}" does not reproduce its reference embeddings (max deviation {deviation:.2e}). '
                           f'Check the backbone ({encoder.backbone}), its revision and the transformers version.')
    logger.info(f'Canary check of head "{name}" passed (max deviation {deviation:.1e}).')


def read_trial_text(source, trial_dir, dataset_name, asr_text_dir, anon_data_suffix, hyp_norm, needed, on_missing):
    """Texts of the trial utterances for one source.

    Returns (texts, stats), or (None, None) if the hypothesis file is missing and may be skipped.
    """
    if source == 'ref':
        text = read_kaldi_format(trial_dir / 'text', values_as_string=True)
        return {utt: text[utt] for utt in needed if utt in text}, {'n_empty_hyp': 0, 'n_digit_chars': 0}

    suffix = anon_data_suffix if source == 'asr_anon' else ''
    hyp_file = Path(asr_text_dir, f'{dataset_name}_asr{suffix}', 'text')
    if not hyp_file.is_file():
        message = (f'No ASR transcripts at {hyp_file}. They are written by the ASR evaluation of run_evaluation.py; '
                   f'run it first for anon_data_suffix "{anon_data_suffix}".')
        if on_missing == 'error':
            raise FileNotFoundError(message)
        logger.warning(f'{message} Skipping the condition "{source}".')
        return None, None
    hypotheses = read_kaldi_format(hyp_file, values_as_string=True)
    missing = [utt for utt in needed if utt not in hypotheses]
    if missing:
        # no fallback to the reference: the attacker must not see text it would not have
        raise ValueError(f'{hyp_file} lacks {len(missing)} of {len(needed)} trial utterances, e.g. {missing[:3]}.')
    raw = {utt: hypotheses[utt] if isinstance(hypotheses[utt], str) else '' for utt in needed}
    texts = {utt: textnorm.normalize(text, hyp_norm) for utt, text in raw.items()}
    stats = {'n_empty_hyp': sum(not text.strip() for text in texts.values()),
             'n_digit_chars': sum(textnorm.count_digit_chars(text) for text in raw.values())}
    return texts, stats


def evaluate_text_attacker(params, device, force_compute=False, encoder=None):
    """Evaluate the text attacker as described by ``params['privacy']['text_attacker']``.

    Args:
        params: config parsed by ``utils.parse_yaml``
        device: device for the text encoder and the heads
        force_compute: recompute the cached pooled features and replace them
        encoder: a ready PooledEncoder to use instead of loading the configured backbone (for tests)

    Returns:
        DataFrame with one row per (enrollment set, trial set, condition); also written to
        ``results_dir/results<anon_data_suffix>.csv``.
    """
    attacker_params = params['privacy']['text_attacker']
    eval_params = attacker_params['evaluation']
    data_dir = params['data_dir']
    anon_data_suffix = params.get('anon_data_suffix') or ''
    track = params.get('track', '')
    results_dir = eval_params['results_dir']
    cache_dir = eval_params.get('cache_dir')
    on_missing = eval_params.get('on_missing_hypothesis', 'skip')
    if on_missing not in ('skip', 'error'):
        raise ValueError(f'on_missing_hypothesis must be "skip" or "error", got "{on_missing}"')

    scenarios = [tuple(s) for s in eval_params.get('scenarios', [['ref', 'ref']])]
    for enrol_source, trial_source in scenarios:
        if enrol_source != 'ref' or trial_source not in TRIAL_SOURCES:
            raise ValueError(f'Unsupported condition [{enrol_source}, {trial_source}]: enrollment must be "ref" and '
                             f'the trial text one of {TRIAL_SOURCES}.')
    if not anon_data_suffix:
        scenarios = [s for s in scenarios if s[1] != 'asr_anon']

    if encoder is None:
        backbone = str(attacker_params['backbone'])
        revision = None if Path(backbone).is_dir() else attacker_params.get('backbone_revision')
        encoder = PooledEncoder(backbone=backbone, revision=revision, device=device)
    eer_fn = _speechbrain_eer()
    reference_counts = _reference_counts()
    heads = {}
    results = []

    for name in attacker_params['dataset_name']:
        dataset = next((d for d in params['datasets'] if d['name'] == name), None)
        if dataset is None or 'enrolls' not in dataset or 'trials' not in dataset:
            raise ValueError(f'{name} is missing an enrolls/trials split')
        head_name = name.split('_')[0]
        if head_name not in heads:
            if head_name not in attacker_params['heads']:
                raise KeyError(f'No head configured for "{head_name}" (dataset {name}); see privacy.text_attacker.heads')
            head, meta = load_head(attacker_params['heads'][head_name]['head_dir'], map_location=device)
            if eval_params.get('canary_check', True):
                check_canary(encoder, head, meta)
            heads[head_name] = (head, meta)
        head, meta = heads[head_name]
        hyp_norm = eval_params.get('hyp_norm', 'head')
        hyp_norm = meta.get('hyp_norm', 'none') if hyp_norm == 'head' else hyp_norm

        for enroll, trial in itertools.product(dataset['enrolls'], dataset['trials']):
            enrol_name, trial_name = f'{dataset["data"]}{enroll}', f'{dataset["data"]}{trial}'
            enrol_dir, trial_dir = data_dir / enrol_name, data_dir / trial_name
            enrol_text = read_kaldi_format(enrol_dir / 'text', values_as_string=True)
            enrol_text = {utt: text if isinstance(text, str) else '' for utt, text in enrol_text.items()}
            enrol_utt2spk = read_kaldi_format(enrol_dir / 'utt2spk')
            malformed = [utt for utt in enrol_text
                         if not isinstance(enrol_utt2spk.get(utt), str) or not enrol_utt2spk[utt].strip()]
            if malformed:
                raise ValueError(f'{enrol_dir / "utt2spk"} has no single speaker for {len(malformed)} utterances of '
                                 f'{enrol_dir / "text"}, e.g. {malformed[:3]}.')
            # as in the ASV evaluation, the trial list is always the one of the original trial directory
            pairs, labels = kaldi_io.read_trials(trial_dir / 'trials')
            needed = sorted({utt for _, utt in pairs})
            trials_info = trial_name.split('_')
            expected = reference_counts.get((track, trials_info[0], trials_info[1]))
            found = (int(labels.sum()), int((labels == 0).sum()))
            if expected is not None and expected != found:
                logger.warning(f'{trial_dir / "trials"} has {found[0]} target and {found[1]} nontarget trials, the '
                               f'reference EERs were computed on {expected[0]} and {expected[1]}: the trial list differs.')

            for enrol_source, trial_source in scenarios:
                trial_text, stats = read_trial_text(trial_source, trial_dir, name, eval_params.get('asr_text_dir'),
                                                anon_data_suffix, hyp_norm, needed, on_missing)
                if trial_text is None:
                    continue
                result = pipeline.evaluate_pair(enrol_text, enrol_utt2spk, trial_text, pairs, labels, encoder, head,
                                                max_length=meta['max_length'],
                                                batch_size=eval_params.get('eval_batchsize', 32),
                                                cache_dir=cache_dir, refresh=force_compute, eer_fn=eer_fn)
                if abs(result['eer'] - result['eer_numpy']) > 1e-3:
                    logger.warning(f'SpeechBrain EER {result["eer"]:.4f} and numpy EER {result["eer_numpy"]:.4f} differ '
                                   f'({result["n_ties"]} tied scores).')
                eer = round(result['eer'], 3)
                suffix = anon_data_suffix if trial_source == 'asr_anon' else ''
                save_dir = results_dir / f'{enrol_name}-{trial_name}__{trial_source}{suffix}'
                kaldi_io.write_scores(save_dir / 'scores', result['pairs'], result['scores'])
                with open(save_dir / 'EER', 'w') as f:
                    f.write(str(result['eer'] / 100))

                print(f'{enrol_name}-{trial_name}: [{enrol_source}, {trial_source}{suffix}]-EER={eer}')
                gender = trials_info[3]
                if 'common' in trial_name:
                    gender += '_common'
                results.append({'dataset': trials_info[0], 'split': trials_info[1], 'gender': gender,
                                'enrollment_text': enrol_source, 'trial_text': trial_source, 'EER': eer,
                                'n_target': result['n_target'], 'n_nontarget': result['n_nontarget'],
                                'n_skipped': result['n_skipped'], 'n_empty_hyp': stats['n_empty_hyp'],
                                'n_digit_chars': stats['n_digit_chars'], 'head': meta.get('name', head_name),
                                'metric_backend': 'speechbrain' if eer_fn is not None else 'numpy'})

    results_df = pd.DataFrame(results)
    print(results_df)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / f'results{anon_data_suffix}.csv')
    return results_df
