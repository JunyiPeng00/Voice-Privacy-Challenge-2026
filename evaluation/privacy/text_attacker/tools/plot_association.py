"""Plot the acoustic-text association (rho_dc, see association.py) of several anonymization systems.

One panel per language, one row per system, with the value for the pretrained and for the fine-tuned
(semi-informed) ASV model. The values are read from a tab-separated file with the header
    system  language  pretrained  fine_tuned
The row with the system name ``original`` gives the dashed line (pretrained ASV on original speech).
Systems are drawn in the order of the file; ``--baselines`` of them are set apart at the top.

Example:
    python -m evaluation.privacy.text_attacker.tools.plot_association --values rho_dc.tsv --out rho_dc.png
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LANGUAGES = {'libri': 'English (LibriSpeech)', 'en': 'English', 'de': 'German', 'es': 'Spanish', 'fr': 'French'}
PRETRAINED, FINE_TUNED = '#2a6fb0', '#d9661f'


def plot_association(values, out_file, baselines=0, label=r'Acoustic–text association ($\rho_{\mathrm{dc}}$)'):
    original = values[values['system'] == 'original'].set_index('language')['pretrained']
    values = values[values['system'] != 'original']
    systems = list(dict.fromkeys(values['system']))
    languages = list(dict.fromkeys(values['language']))

    fig, axes = plt.subplots(1, len(languages), figsize=(3.25 * len(languages), 1.2 + 0.39 * len(systems)),
                             sharey=True, squeeze=False)
    for ax, language in zip(axes[0], languages):
        rows = values[values['language'] == language].set_index('system')
        for i, system in enumerate(systems):
            pretrained, fine_tuned = rows.loc[system, 'pretrained'], rows.loc[system, 'fine_tuned']
            ax.plot([pretrained, fine_tuned], [i, i], color='#b8bec7', lw=2.2, zorder=1)
            ax.scatter(pretrained, i, s=62, color=PRETRAINED, marker='o', edgecolor='white', linewidth=0.8, zorder=2,
                       label='Pretrained' if i == 0 else None)
            ax.scatter(fine_tuned, i, s=58, color=FINE_TUNED, marker='s', edgecolor='white', linewidth=0.8, zorder=3,
                       label='Fine-tuned' if i == 0 else None)
        if language in original:
            ax.axvline(original[language], color='#5d6675', lw=1, ls='--', zorder=0)
        if baselines:
            ax.axhline(baselines - 0.5, color='#d0d4da', lw=1)
        ax.set_title(LANGUAGES.get(language, language), fontsize=13)
        ax.set_xlim(-0.06, 0.40)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.grid(axis='x', color='#e6e8ec')
        ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0][0].set_yticks(range(len(systems)), systems, fontsize=12)
    axes[0][0].invert_yaxis()
    fig.supxlabel(label, fontsize=12)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', frameon=False, fontsize=12, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_file, dpi=200)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--values', required=True, help='tab-separated file: system, language, pretrained, fine_tuned')
    parser.add_argument('--out', required=True, help='figure file, e.g. rho_dc.png or rho_dc.pdf')
    parser.add_argument('--baselines', type=int, default=0, help='number of leading systems to set apart as baselines')
    parser.add_argument('--label', default=None, help='label of the x axis')
    args = parser.parse_args()

    kwargs = {'label': args.label} if args.label else {}
    plot_association(pd.read_csv(args.values, sep='\t'), args.out, baselines=args.baselines, **kwargs)
    print(f'Wrote {args.out}')
