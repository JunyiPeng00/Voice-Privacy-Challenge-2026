"""Which words of a trial transcript drive the decision of the text attacker?

Word importance by occlusion: a word's importance for a trial is the drop of
the trial score when the word is deleted from the trial transcript and the
transcript is encoded again,

    importance(word) = score(transcript) - score(transcript without the word).

A positive value means the word pushes the trial towards "same speaker", a
negative value towards "different speaker". The values are measured, not
approximated from gradients, and they refer to the trial side only: the
enrollment profile is kept fixed.
"""
import html

import numpy as np
import torch

from text_attacker import scoring


def _scores(encoder, head, profile, texts, max_length, batch_size):
    device = next(head.parameters()).device
    with torch.inference_mode():
        pooled = torch.from_numpy(encoder.pool(texts, max_length=max_length, batch_size=batch_size)).to(device)
        emb = head.embed(pooled).float().cpu().numpy()
    pairs = [("profile", i) for i in range(len(texts))]
    return scoring.score_trials({"profile": profile}, dict(enumerate(emb)), pairs)[0]


def word_occlusion(encoder, head, profile, text, max_length=96, batch_size=32):
    """Importance of every word of ``text`` for its score against ``profile``.

    Args:
        profile: speaker profile from ``scoring.build_profiles``
        text: trial transcript; words are separated by whitespace

    Returns:
        words: list of the words
        importance: float array, one value per word (see the module docstring)
        score: score of the full transcript
    """
    words = text.split()
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    scores = _scores(encoder, head, profile, variants, max_length, batch_size)
    return words, (scores[0] - scores[1:]).astype(np.float64), float(scores[0])


def _tint(value, scale):
    if scale <= 0 or value == 0:
        return "transparent"
    strength = min(1.0, abs(value) / scale) ** 0.8 * 0.85
    color = "214, 88, 30" if value > 0 else "37, 110, 196"      # orange: same speaker, blue: different speaker
    return f"rgba({color}, {strength:.3f})"


def render_html(rows, title="Text attacker: word importance"):
    """A self-contained HTML page with one table row per explained trial.

    Args:
        rows: list of dicts with keys enrol, utt, label (1 target, 0 nontarget), score, threshold, words, importance
    """
    body = []
    for row in rows:
        scale = float(np.max(np.abs(row["importance"]))) if len(row["importance"]) else 0.0
        words = " ".join(
            f'<span style="background:{_tint(value, scale)}" title="{value:+.4f}">{html.escape(word)}</span>'
            for word, value in zip(row["words"], row["importance"]))
        accepted = row["score"] > row["threshold"]
        body.append(
            f'<tr><td>{html.escape(str(row["enrol"]))}</td><td>{html.escape(str(row["utt"]))}</td>'
            f'<td>{"target" if row["label"] == 1 else "nontarget"}</td>'
            f'<td class="{"acc" if accepted else "rej"}">{"accept" if accepted else "reject"} ({row["score"]:.2f})</td>'
            f'<td class="num">{float(np.sum(row["importance"])):+.2f}</td><td class="text">{words}</td></tr>')
    threshold = f'{rows[0]["threshold"]:.3f}' if rows else "n/a"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1c2330; background: #ffffff; }}
h1 {{ font-size: 1.2rem; }}
p {{ max-width: 80ch; color: #4a5361; font-size: 14px; }}
table {{ border-collapse: collapse; font-size: 14px; }}
th, td {{ text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid #dde1e7; white-space: nowrap; }}
th {{ color: #4a5361; font-weight: 500; }}
td.text {{ white-space: normal; font-family: Georgia, serif; font-size: 16px; line-height: 1.9; min-width: 28ch; }}
td.text span {{ padding: 1px 2px; border-radius: 3px; }}
td.num {{ font-variant-numeric: tabular-nums; }}
td.acc {{ color: #8a2e06; }} td.rej {{ color: #1d4f8a; }}
.key span {{ padding: 1px 6px; border-radius: 3px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Each word is tinted by the drop of the trial score when the word is deleted from the trial transcript:
<span class="key"><span style="background:rgba(214, 88, 30, 0.6)">pushes towards &ldquo;same speaker&rdquo;</span>
<span style="background:rgba(37, 110, 196, 0.6)">pushes towards &ldquo;different speaker&rdquo;</span></span>.
The tint is relative to the strongest word of the row; hover over a word for its value. A trial is accepted when
its score is above the EER threshold of the dataset ({threshold}). &ldquo;Sum&rdquo; is the sum of the word values.</p>
<table>
<tr><th>Enrollment speaker</th><th>Trial utterance</th><th>Label</th><th>Decision (score)</th><th>Sum</th><th>Word importance</th></tr>
{chr(10).join(body)}
</table>
</body></html>
"""
