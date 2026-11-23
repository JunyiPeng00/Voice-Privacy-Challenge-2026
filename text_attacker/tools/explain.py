"""Word importance for trials of the text attacker (see text_attacker/attribution.py).

The dataset is scored as in the evaluation, then the selected trials are
explained: every word of the trial transcript is deleted in turn and the trial
is scored again. Writes an HTML page and a table of the word values.

Example:
    python -m text_attacker.tools.explain --config configs/track1/eval_text.yaml --dataset libri_dev \
        --speaker 1673 --label target --num 5
"""
import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np

from text_attacker import attribution, kaldi_io, pipeline, scoring
from text_attacker.encoder import PooledEncoder
from text_attacker.head import load_head
from text_attacker.official import _trial_text
from utils import parse_yaml, read_kaldi_format, setup_logger

logger = setup_logger(__name__)


def explain_trials(params, dataset_name, device='cpu', speaker=None, label='all', num=5, order='list',
                   trial_source='ref', enrolls=None, trials=None, encoder=None):
    """Explain trials of one dataset.

    Args:
        params: config parsed by ``utils.parse_yaml`` (the config of run_text_attacker.py)
        dataset_name: one of ``privacy.text_attacker.dataset_name``
        speaker: enrollment speaker to restrict the trials to, or None
        label: 'target', 'nontarget' or 'all'
        num: number of trials to explain
        order: 'list' (order of the trial list), 'score' (highest scores first) or 'random'
        trial_source: 'ref', 'asr_orig' or 'asr_anon'
        enrolls, trials: enrollment and trial split of the dataset (e.g. '_enrolls', '_trials_mixed'); needed only if
            the dataset has more than one
        encoder: a ready PooledEncoder to use instead of loading the configured backbone (for tests)

    Returns:
        list of dicts with keys enrol, utt, label, score, threshold, words, importance; score is the score of the
        evaluation, the decision is score > threshold
    """
    attacker_params = params['privacy']['text_attacker']
    eval_params = attacker_params['evaluation']
    dataset = next((d for d in params['datasets'] if d['name'] == dataset_name), None)
    if dataset is None or 'enrolls' not in dataset or 'trials' not in dataset:
        raise ValueError(f'{dataset_name} is missing an enrolls/trials split')
    splits = {}
    for kind, chosen in (('enrolls', enrolls), ('trials', trials)):
        if chosen is None and len(dataset[kind]) > 1:
            raise ValueError(f'{dataset_name} has several {kind} splits {list(dataset[kind])}, choose one')
        if chosen is not None and chosen not in dataset[kind]:
            raise ValueError(f'{dataset_name} has no {kind} split "{chosen}", only {list(dataset[kind])}')
        splits[kind] = dataset[kind][0] if chosen is None else chosen
    enrol_dir = params['data_dir'] / f'{dataset["data"]}{splits["enrolls"]}'
    trial_dir = params['data_dir'] / f'{dataset["data"]}{splits["trials"]}'
    anon_data_suffix = params.get('anon_data_suffix') or ''
    if trial_source == 'asr_anon' and not anon_data_suffix:
        raise ValueError('The trial text "asr_anon" needs an anon_data_suffix (e.g. --overwrite \'{"anon_data_suffix": "_mcadams"}\')')

    if encoder is None:
        backbone = str(attacker_params['backbone'])
        revision = None if Path(backbone).is_dir() else attacker_params.get('backbone_revision')
        encoder = PooledEncoder(backbone=backbone, revision=revision, device=device)
    head, meta = load_head(attacker_params['heads'][dataset_name.split('_')[0]]['head_dir'], map_location=device)
    hyp_norm = eval_params.get('hyp_norm', 'head')
    hyp_norm = meta.get('hyp_norm', 'none') if hyp_norm == 'head' else hyp_norm
    batch_size = eval_params.get('eval_batchsize', 32)
    cache_dir = eval_params.get('cache_dir')

    enrol_text = read_kaldi_format(enrol_dir / 'text', values_as_string=True)
    enrol_text = {utt: text if isinstance(text, str) else '' for utt, text in enrol_text.items()}
    enrol_utt2spk = read_kaldi_format(enrol_dir / 'utt2spk')
    pairs, labels = kaldi_io.read_trials(trial_dir / 'trials')
    needed = sorted({utt for _, utt in pairs})
    trial_text, _ = _trial_text(trial_source, trial_dir, dataset_name, eval_params.get('asr_text_dir'),
                                anon_data_suffix, hyp_norm, needed, on_missing='error')

    # score the whole dataset to get the EER threshold the decisions refer to
    result = pipeline.evaluate_pair(enrol_text, enrol_utt2spk, trial_text, pairs, labels, encoder, head,
                                    max_length=meta['max_length'], batch_size=batch_size, cache_dir=cache_dir)
    threshold = scoring.eer_threshold(result['scores'], result['labels'])
    logger.info(f'{dataset_name}: EER {result["eer"]:.3f} %, threshold {threshold:.4f}')

    selected = [i for i, (enrol, _) in enumerate(result['pairs'])
                if (speaker is None or enrol == str(speaker))
                and (label == 'all' or result['labels'][i] == int(label == 'target'))]
    if not selected:
        raise ValueError(f'No {label} trial of enrollment speaker {speaker} in {trial_dir / "trials"}')
    if order == 'score':
        selected.sort(key=lambda i: -result['scores'][i])
    elif order == 'random':
        selected = list(np.random.default_rng(0).permutation(selected))
    elif order != 'list':
        raise ValueError(f'order must be "list", "score" or "random", got "{order}"')
    selected = selected[:num]

    enrol_ids = sorted(enrol_text)
    enrol_emb = pipeline.embed_texts(encoder, head, enrol_text, meta['max_length'], batch_size, cache_dir)
    profiles = scoring.build_profiles([enrol_emb[u] for u in enrol_ids], [enrol_utt2spk[u] for u in enrol_ids])
    rows = []
    for i in selected:
        enrol, utt = result['pairs'][i]
        words, importance, _ = attribution.word_occlusion(encoder, head, profiles[enrol], trial_text[utt],
                                                          max_length=meta['max_length'], batch_size=batch_size)
        # the score and the decision are those of the evaluation, not of the recomputation above
        rows.append({'enrol': enrol, 'utt': utt, 'label': int(result['labels'][i]), 'score': float(result['scores'][i]),
                     'threshold': threshold, 'words': words, 'importance': importance})
    return rows


def write_table(path, rows):
    """One line per word: enrollment speaker, trial utterance, label, score, threshold, position, word, importance."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('enrollment_speaker\ttrial_utterance\tlabel\tscore\tthreshold\tposition\tword\timportance\n')
        for row in rows:
            for position, (word, value) in enumerate(zip(row['words'], row['importance'])):
                f.write(f'{row["enrol"]}\t{row["utt"]}\t{"target" if row["label"] == 1 else "nontarget"}\t'
                        f'{row["score"]:.6f}\t{row["threshold"]:.6f}\t{position}\t{word}\t{value:.6f}\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/track1/eval_text.yaml')
    parser.add_argument('--overwrite', type=str, default='{}')
    parser.add_argument('--dataset', required=True, help='e.g. libri_dev, en_test')
    parser.add_argument('--speaker', default=None, help='enrollment speaker; all speakers if not given')
    parser.add_argument('--label', default='all', choices=['target', 'nontarget', 'all'])
    parser.add_argument('--num', type=int, default=5, help='number of trials to explain')
    parser.add_argument('--order', default='list', choices=['list', 'score', 'random'])
    parser.add_argument('--trial_text', default='ref', choices=['ref', 'asr_orig', 'asr_anon'])
    parser.add_argument('--enrolls', default=None, help='enrollment split, if the dataset has several')
    parser.add_argument('--trials', default=None, help='trial split, if the dataset has several')
    parser.add_argument('--out_dir', default=None, help='default: <exp_dir>/text_attacker/explain')
    args = parser.parse_args()

    import torch
    params = parse_yaml(Path(args.config), overrides=json.loads(args.overwrite))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rows = explain_trials(params, args.dataset, device=device, speaker=args.speaker, label=args.label, num=args.num,
                          order=args.order, trial_source=args.trial_text, enrolls=args.enrolls, trials=args.trials)

    out_dir = Path(args.out_dir) if args.out_dir else params['exp_dir'] / 'text_attacker' / 'explain'
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (params.get('anon_data_suffix') or '') if args.trial_text == 'asr_anon' else ''
    name = f'{args.dataset}{args.enrolls or ""}{args.trials or ""}_{args.speaker or "all"}_{args.label}_{args.trial_text}{suffix}'
    title = f'Text attacker: {args.dataset}, enrollment speaker {args.speaker or "all"}, {args.label} trials ({args.trial_text})'
    (out_dir / f'{name}.html').write_text(attribution.render_html(rows, title=title), encoding='utf-8')
    write_table(out_dir / f'{name}.tsv', rows)
    for row in rows:
        top = sorted(zip(row['words'], row['importance']), key=lambda pair: -pair[1])[:3]
        print(f'{row["enrol"]} {row["utt"]} {"target" if row["label"] == 1 else "nontarget"} score={row["score"]:.3f} '
              f'top words: ' + ', '.join(f'{word} ({value:+.3f})' for word, value in top))
    print(f'Wrote {out_dir / name}.html and .tsv')
