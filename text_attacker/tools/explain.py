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

import torch
from speechbrain.utils.metric_stats import EER

from text_attacker import attribution
from text_attacker.official import load_attacker, load_encoder, trial_text_dir
from utils import parse_yaml
from utils.data_io import read_table


def explain_trials(params, dataset_name, asv, speaker=None, label='all', num=5, order='list', trial_source='ref',
                   enrolls=None, trials=None):
    """Explain trials of one dataset.

    Args:
        params: config parsed by ``utils.parse_yaml`` (the config of run_text_attacker.py)
        dataset_name: one of ``privacy.text_attacker.dataset_name``
        asv: TextASV of the dataset (``official.load_attacker``)
        speaker: enrollment speaker to restrict the trials to, or None
        label: 'target', 'nontarget' or 'all'
        num: number of trials to explain
        order: 'list' (order of the trial list), 'score' (highest scores first) or 'random'
        trial_source: 'ref', 'asr_orig' or 'asr_anon'
        enrolls, trials: enrollment and trial split of the dataset; the first one if not given

    Returns:
        DataFrame with one row per trial and the columns enrol_set, trial_set, enrol, utt, label, score, threshold,
        words, importance; the decision is score > threshold
    """
    dataset = next(d for d in params['datasets'] if d['name'] == dataset_name)
    enrol_dir = params['data_dir'] / f'{dataset["data"]}{enrolls or dataset["enrolls"][0]}'
    trial_name = f'{dataset["data"]}{trials or dataset["trials"][0]}'
    test_dir = trial_text_dir(trial_source, params['data_dir'], trial_name, dataset_name,
                              params['privacy']['text_attacker']['evaluation']['asr_text_dir'], params['anon_data_suffix'])
    trials_file = params['data_dir'] / trial_name / 'trials'

    # score the whole dataset to get the EER threshold the decisions refer to
    asv.eer_compute(enrol_dir=enrol_dir, test_dir=test_dir, trial_runs_file=trials_file)
    scores = read_table(asv.score_save_dir.parent / f'{enrol_dir.name}-{test_dir.name}' / 'scores',
                        names=['enrol', 'utt', 'score'], dtype=[str, str, float])
    labels = read_table(trials_file, names=['enrol', 'utt', 'label'], dtype=str)
    rows = scores.merge(labels.drop_duplicates(['enrol', 'utt'], keep='last'))
    target = rows['label'] == 'target'
    _, threshold = EER(torch.tensor(rows['score'][target].values), torch.tensor(rows['score'][~target].values))

    if speaker is not None:
        rows = rows[rows['enrol'] == speaker]
    if label != 'all':
        rows = rows[rows['label'] == label]
    if order == 'score':
        rows = rows.sort_values('score', ascending=False)
    elif order == 'random':
        rows = rows.sample(frac=1, random_state=0)
    rows = rows.head(num).assign(enrol_set=enrol_dir.name, trial_set=test_dir.name, threshold=threshold)

    extractor = asv.extractor
    profiles = extractor.extract_speakers(enrol_dir, emb_level='spk')
    text = extractor.read_text(test_dir)
    explained = [attribution.word_occlusion(extractor.encoder, extractor.head,
                                            profiles.get_embedding_for_identifier(enrol), text[utt], extractor.max_length)
                 for enrol, utt in zip(rows['enrol'], rows['utt'])]
    return rows.assign(words=[words for words, _ in explained], importance=[values for _, values in explained])


def write_table(path, rows):
    """One line per word: enrollment speaker, trial utterance, label, score, threshold, position, word, importance."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('enrollment_speaker\ttrial_utterance\tlabel\tscore\tthreshold\tposition\tword\timportance\n')
        for row in rows.itertuples():
            for position, (word, value) in enumerate(zip(row.words, row.importance)):
                f.write(f'{row.enrol}\t{row.utt}\t{row.label}\t{row.score:.6f}\t{row.threshold:.6f}\t{position}\t'
                        f'{word}\t{value:.6f}\n')


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
    parser.add_argument('--enrolls', default=None, help='enrollment split; the first one of the dataset if not given')
    parser.add_argument('--trials', default=None, help='trial split; the first one of the dataset if not given')
    args = parser.parse_args()

    params = parse_yaml(Path(args.config), overrides=json.loads(args.overwrite))
    attacker_params = params['privacy']['text_attacker']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    asv = load_attacker(attacker_params, args.dataset, load_encoder(attacker_params, device))
    rows = explain_trials(params, args.dataset, asv, speaker=args.speaker, label=args.label, num=args.num,
                          order=args.order, trial_source=args.trial_text, enrolls=args.enrolls, trials=args.trials)

    out_dir = params['exp_dir'] / 'text_attacker' / 'explain'
    out_dir.mkdir(parents=True, exist_ok=True)
    # named after the enrollment and trial sets that were used, as the result directories of the evaluation
    sets = f'{rows["enrol_set"].iloc[0]}-{rows["trial_set"].iloc[0]}'
    name = f'{sets}_{args.speaker or "all"}_{args.label}'
    title = f'Text attacker: {sets}, enrollment speaker {args.speaker or "all"}, {args.label} trials'
    (out_dir / f'{name}.html').write_text(attribution.render_html(rows.to_dict('records'), title=title), encoding='utf-8')
    write_table(out_dir / f'{name}.tsv', rows)
    for row in rows.itertuples():
        top = sorted(zip(row.words, row.importance), key=lambda pair: -pair[1])[:3]
        print(f'{row.enrol} {row.utt} {row.label} score={row.score:.3f} top words: '
              + ', '.join(f'{word} ({value:+.3f})' for word, value in top))
    print(f'Wrote {out_dir / name}.html and .tsv')
