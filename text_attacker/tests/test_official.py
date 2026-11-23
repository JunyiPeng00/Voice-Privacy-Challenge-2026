import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from text_attacker import official
    from text_attacker.encoder import PooledEncoder
    from text_attacker.head import Head, save_head
    from text_attacker.tests.fakes import FakeModel, FakeTokenizer
    from utils import parse_yaml
except ImportError:  # needs torch and the dependencies of this repository
    torch = None

CONFIG = """
data_dir: {root}/data
exp_dir: {root}/exp
track: toy
anon_data_suffix: '{suffix}'
datasets:
  - name: toy_dev
    data: toy_dev
    enrolls: [_enrolls]
    trials: [_trials_mixed]
privacy:
  text_attacker:
    dataset_name: [toy_dev]
    backbone: fake
    heads:
      toy:
        head_dir: !ref <exp_dir>/heads/toy
    evaluation:
      results_dir: !ref <exp_dir>/text_attacker/results
      cache_dir: !ref <exp_dir>/text_attacker/feat_cache
      asr_text_dir: !ref <exp_dir>/asr
      scenarios:
        - [ref, ref]
        - [ref, asr_orig]
        - [ref, asr_anon]
      hyp_norm: head
      on_missing_hypothesis: {on_missing}
      canary_check: true
      eval_batchsize: 2
"""


@unittest.skipIf(torch is None, "torch or the repository dependencies are not installed")
class OfficialAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.write("data/toy_dev_enrolls/text", "s1-a THE SHIP SAILED NORTH\ns1-b THE SHIP SANK\ns2-a BREAD AND BUTTER\ns2-b\n")
        self.write("data/toy_dev_enrolls/utt2spk", "s1-a s1\ns1-b s1\ns2-a s2\ns2-b s2\n")
        self.write("data/toy_dev_trials_mixed/text", "t1 THE SHIP SAILED\nt2 BREAD AND CAKE\nt3 NOT IN ANY TRIAL\n")
        # the pair (s2, t2) is listed twice: the last label counts
        self.write("data/toy_dev_trials_mixed/trials",
                   "s1 t1 target\ns2 t1 nontarget\ns1 t2 nontarget\ns2 t2 nontarget\ns2 t2 target\n")
        torch.manual_seed(0)
        save_head(self.root / "exp/heads/toy", Head(3, "softmax", hidden=16, bottleneck=8, embedding=4),
                  {"name": "toy", "max_length": 96, "hyp_norm": "upper_letters"})
        self.encoder = PooledEncoder(backbone="fake", tokenizer=FakeTokenizer(), model=FakeModel())

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def run_attacker(self, suffix="", on_missing="skip"):
        self.write("config.yaml", CONFIG.format(root=self.root, suffix=suffix, on_missing=on_missing))
        params = parse_yaml(self.root / "config.yaml")
        return official.evaluate_text_attacker(params, device="cpu", encoder=self.encoder)

    def test_reference_only_when_no_asr_output_exists(self):
        results = self.run_attacker()
        self.assertEqual(results["trial_text"].tolist(), ["ref"])
        row = results.iloc[0]
        self.assertEqual((row["dataset"], row["split"], row["gender"], row["head"]), ("toy", "dev", "mixed", "toy"))
        self.assertEqual((row["n_target"], row["n_nontarget"], row["n_skipped"]), (2, 2, 0))
        out = self.root / "exp/text_attacker/results"
        scores = (out / "toy_dev_enrolls-toy_dev_trials_mixed__ref/scores").read_text().splitlines()
        self.assertEqual([line.split()[:2] for line in scores], [["s1", "t1"], ["s2", "t1"], ["s1", "t2"], ["s2", "t2"]])
        self.assertAlmostEqual(float((out / "toy_dev_enrolls-toy_dev_trials_mixed__ref/EER").read_text()) * 100,
                               row["EER"], places=2)
        self.assertTrue((out / "results.csv").is_file())

    def test_missing_asr_output_can_be_an_error(self):
        with self.assertRaises(FileNotFoundError):
            self.run_attacker(on_missing="error")

    def test_asr_conditions(self):
        # the hypotheses differ from the reference in case, punctuation and a digit; t2 is empty for the anonymized audio
        self.write("exp/asr/toy_dev_asr/text", "t1 the ship, sailed!\nt2 bread and 2 cakes\nt3 x\n")
        self.write("exp/asr/toy_dev_asr_anon/text", "t1 the ship sailed\nt2\nt3 x\n")
        results = self.run_attacker(suffix="_anon").set_index("trial_text")
        self.assertEqual(sorted(results.index), ["asr_anon", "asr_orig", "ref"])
        self.assertEqual((results.loc["asr_orig", "n_digit_chars"], results.loc["asr_orig", "n_empty_hyp"]), (1, 0))
        self.assertEqual((results.loc["asr_anon", "n_digit_chars"], results.loc["asr_anon", "n_empty_hyp"]), (0, 1))
        out = self.root / "exp/text_attacker/results"
        self.assertTrue((out / "toy_dev_enrolls-toy_dev_trials_mixed__asr_anon_anon/scores").is_file())
        self.assertTrue((out / "results_anon.csv").is_file())
        # upper_letters makes "the ship, sailed!" equal to the reference of t1: same scores for t1
        def t1_scores(source):
            lines = (out / f"toy_dev_enrolls-toy_dev_trials_mixed__{source}/scores").read_text().splitlines()
            return [float(line.split()[2]) for line in lines if line.split()[1] == "t1"]
        self.assertEqual(t1_scores("asr_orig"), t1_scores("ref"))

    def test_anonymized_condition_needs_a_suffix(self):
        self.write("exp/asr/toy_dev_asr/text", "t1 a\nt2 b\n")
        results = self.run_attacker(suffix="")
        self.assertEqual(sorted(results["trial_text"]), ["asr_orig", "ref"])

    def test_incomplete_hypothesis_file_is_rejected(self):
        self.write("exp/asr/toy_dev_asr/text", "t1 the ship sailed\n")
        with self.assertRaises(ValueError):
            self.run_attacker()

    def test_force_compute_replaces_stale_features(self):
        self.write("config.yaml", CONFIG.format(root=self.root, suffix="", on_missing="skip"))
        params = parse_yaml(self.root / "config.yaml")
        first = official.evaluate_text_attacker(params, device="cpu", encoder=self.encoder)
        scores_file = self.root / "exp/text_attacker/results/toy_dev_enrolls-toy_dev_trials_mixed__ref/scores"
        scores = scores_file.read_text()
        for cache_file in (self.root / "exp/text_attacker/feat_cache").glob("pooled_*.npy"):
            stale = np.load(cache_file)
            np.save(cache_file, stale[::-1].copy())
        official.evaluate_text_attacker(parse_yaml(self.root / "config.yaml"), device="cpu", encoder=self.encoder)
        self.assertNotEqual(scores_file.read_text(), scores)                    # the stale features were used
        official.evaluate_text_attacker(parse_yaml(self.root / "config.yaml"), device="cpu", encoder=self.encoder,
                                        force_compute=True)
        self.assertEqual(scores_file.read_text(), scores)
        official.evaluate_text_attacker(parse_yaml(self.root / "config.yaml"), device="cpu", encoder=self.encoder)
        self.assertEqual(scores_file.read_text(), scores)                       # and the cache was repaired
        self.assertEqual(len(first), 1)

    def test_malformed_utt2spk_is_rejected(self):
        self.write("data/toy_dev_enrolls/utt2spk", "s1-a s1 extra\ns1-b s1\ns2-a s2\ns2-b s2\n")
        with self.assertRaises(ValueError):
            self.run_attacker()

    def test_only_reference_enrolment_is_supported(self):
        self.write("config.yaml", CONFIG.format(root=self.root, suffix="", on_missing="skip").replace("[ref, asr_orig]", "[asr_orig, ref]"))
        with self.assertRaises(ValueError):
            official.evaluate_text_attacker(parse_yaml(self.root / "config.yaml"), device="cpu", encoder=self.encoder)

    def test_canary_detects_a_different_encoder(self):
        probes = {"toy": [{"text": "THE SHIP", "embedding_head": [0.0, 0.0, 0.0, 9.0]}]}
        original = official.REFERENCE_DIR
        try:
            official.REFERENCE_DIR = self.root / "reference"
            self.write("reference/canary.json", __import__("json").dumps({"heads": probes}))
            self.write("reference/eer_reference.tsv", "track\tset\tsplit\ttrials_file\teer\tn_trials\tn_target\tn_nontarget\n")
            with self.assertRaises(RuntimeError):
                self.run_attacker()
        finally:
            official.REFERENCE_DIR = original


if __name__ == "__main__":
    unittest.main()
