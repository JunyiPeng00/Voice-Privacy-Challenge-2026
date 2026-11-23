import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from text_attacker import attribution, pipeline, scoring
    from text_attacker.encoder import PooledEncoder
    from text_attacker.head import Head, save_head
    from text_attacker.tests.fakes import FakeModel, FakeTokenizer
    from text_attacker.tests.test_official import CONFIG
    from text_attacker.tools.explain import explain_trials, write_table
    from utils import parse_yaml
except ImportError:  # needs torch and the dependencies of this repository
    torch = None


@unittest.skipIf(torch is None, "torch or the repository dependencies are not installed")
class AttributionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.encoder = PooledEncoder(backbone="fake", tokenizer=FakeTokenizer(), model=FakeModel())
        self.head = Head(3, "softmax", hidden=16, bottleneck=8, embedding=4).eval()
        enrol = pipeline.embed_texts(self.encoder, self.head, {"a": "THE SHIP SAILED NORTH", "b": "THE SHIP SANK"})
        self.profile = scoring.build_profiles([enrol["a"], enrol["b"]], ["s", "s"])["s"]

    def score(self, text):
        emb = pipeline.embed_texts(self.encoder, self.head, {"t": text})
        return float(scoring.score_trials({"s": self.profile}, emb, [("s", "t")])[0][0])

    def test_importance_is_the_score_drop_without_the_word(self):
        text = "THE  SHIP SAILED AT DAWN"
        words, importance, score = attribution.word_occlusion(self.encoder, self.head, self.profile, text, batch_size=2)
        self.assertEqual(words, ["THE", "SHIP", "SAILED", "AT", "DAWN"])
        self.assertAlmostEqual(score, self.score(text), places=6)
        for i in range(len(words)):
            without = " ".join(words[:i] + words[i + 1:])
            self.assertAlmostEqual(importance[i], score - self.score(without), places=6)

    def test_single_word_and_empty_transcript(self):
        words, importance, _ = attribution.word_occlusion(self.encoder, self.head, self.profile, "SHIP")
        self.assertEqual((words, len(importance)), (["SHIP"], 1))
        words, importance, _ = attribution.word_occlusion(self.encoder, self.head, self.profile, "")
        self.assertEqual((words, len(importance)), ([], 0))

    def test_html_escapes_words_and_states_the_decision(self):
        rows = [{"enrol": "s<1>", "utt": "u1", "label": 1, "score": 0.6, "threshold": 0.5,
                 "words": ["a<b", "ok"], "importance": np.array([0.2, -0.1])},
                {"enrol": "s2", "utt": "u2", "label": 0, "score": 0.4, "threshold": 0.5, "words": [], "importance": np.array([])}]
        page = attribution.render_html(rows)
        self.assertIn("a&lt;b", page)
        self.assertNotIn("a<b", page)
        self.assertIn("s&lt;1&gt;", page)
        self.assertIn("accept (0.60)", page)
        self.assertIn("reject (0.40)", page)
        self.assertIn("rgba(214, 88, 30", page)      # positive word
        self.assertIn("rgba(37, 110, 196", page)     # negative word


@unittest.skipIf(torch is None, "torch or the repository dependencies are not installed")
class ExplainToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = {"data/toy_dev_enrolls/text": "s1-a THE SHIP SAILED NORTH\ns1-b THE SHIP SANK\ns2-a BREAD AND BUTTER\n",
                 "data/toy_dev_enrolls/utt2spk": "s1-a s1\ns1-b s1\ns2-a s2\n",
                 "data/toy_dev_trials_mixed/text": "t1 THE SHIP SAILED\nt2 BREAD AND CAKE\n",
                 "data/toy_dev_trials_mixed/trials": "s1 t1 target\ns2 t1 nontarget\ns1 t2 nontarget\ns2 t2 target\n",
                 "exp/asr/toy_dev_asr/text": "t1 the ship sailed away\nt2 bread and cake\n",
                 "config.yaml": CONFIG.format(root=self.root, suffix="", on_missing="skip")}
        for relative, content in files.items():
            (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.root / relative).write_text(content, encoding="utf-8")
        torch.manual_seed(0)
        save_head(self.root / "exp/heads/toy", Head(3, "softmax", hidden=16, bottleneck=8, embedding=4),
                  {"name": "toy", "max_length": 96, "hyp_norm": "upper_letters"})
        self.encoder = PooledEncoder(backbone="fake", tokenizer=FakeTokenizer(), model=FakeModel())
        self.params = parse_yaml(self.root / "config.yaml")

    def tearDown(self):
        self.tmp.cleanup()

    def test_selection_and_scores_match_the_evaluation(self):
        rows = explain_trials(self.params, "toy_dev", speaker="s1", label="all", num=5, encoder=self.encoder)
        self.assertEqual([(r["enrol"], r["utt"], r["label"]) for r in rows], [("s1", "t1", 1), ("s1", "t2", 0)])
        self.assertEqual(rows[0]["words"], ["THE", "SHIP", "SAILED"])
        from text_attacker import official
        official.evaluate_text_attacker(self.params, device="cpu", encoder=self.encoder)
        scores = (self.root / "exp/text_attacker/results/toy_dev_enrolls-toy_dev_trials_mixed__ref/scores").read_text()
        evaluated = {tuple(l.split()[:2]): float(l.split()[2]) for l in scores.splitlines()}
        for row in rows:
            self.assertAlmostEqual(row["score"], evaluated[(row["enrol"], row["utt"])], places=6)

    def test_label_order_and_number(self):
        rows = explain_trials(self.params, "toy_dev", label="target", num=1, order="score", encoder=self.encoder)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 1)
        with self.assertRaises(ValueError):
            explain_trials(self.params, "toy_dev", speaker="nobody", encoder=self.encoder)

    def test_asr_transcripts_are_normalized(self):
        rows = explain_trials(self.params, "toy_dev", speaker="s1", label="target", trial_source="asr_orig",
                              encoder=self.encoder)
        self.assertEqual(rows[0]["words"], ["THE", "SHIP", "SAILED", "AWAY"])

    def test_anonymized_condition_needs_a_suffix(self):
        with self.assertRaises(ValueError):
            explain_trials(self.params, "toy_dev", trial_source="asr_anon", encoder=self.encoder)

    def test_split_selection(self):
        rows = explain_trials(self.params, "toy_dev", enrolls="_enrolls", trials="_trials_mixed", num=1, encoder=self.encoder)
        self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError):
            explain_trials(self.params, "toy_dev", trials="_trials_f", encoder=self.encoder)
        self.params["datasets"][0]["trials"] = ["_trials_mixed", "_trials_f"]
        with self.assertRaises(ValueError):                                    # ambiguous without a choice
            explain_trials(self.params, "toy_dev", encoder=self.encoder)

    def test_reported_score_is_the_evaluation_score(self):
        import unittest.mock as mock
        shifted = lambda *a, **k: (["X"], np.array([0.0]), 123.0)
        with mock.patch("text_attacker.tools.explain.attribution.word_occlusion", shifted):
            rows = explain_trials(self.params, "toy_dev", num=2, encoder=self.encoder)
        self.assertTrue(all(row["score"] != 123.0 for row in rows))

    def test_table(self):
        rows = explain_trials(self.params, "toy_dev", speaker="s2", label="target", encoder=self.encoder)
        write_table(self.root / "words.tsv", rows)
        lines = (self.root / "words.tsv").read_text().splitlines()
        self.assertEqual(len(lines), 1 + 3)
        self.assertEqual(lines[1].split("\t")[:3] + lines[1].split("\t")[5:7], ["s2", "t2", "target", "0", "BREAD"])


if __name__ == "__main__":
    unittest.main()
