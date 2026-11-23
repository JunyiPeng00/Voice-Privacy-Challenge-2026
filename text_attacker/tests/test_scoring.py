import unittest

import numpy as np

from text_attacker import scoring


class ScoringTest(unittest.TestCase):
    def test_profile_is_normalised_mean_of_unit_embeddings(self):
        emb = np.array([[2.0, 0.0], [0.0, 5.0], [0.0, 3.0]])
        profiles = scoring.build_profiles(emb, ["a", "a", "b"])
        np.testing.assert_allclose(profiles["a"], np.array([1.0, 1.0]) / np.sqrt(2))
        np.testing.assert_allclose(profiles["b"], [0.0, 1.0])

    def test_score_trials_cosine_and_skips(self):
        profiles = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
        trials = {"u1": np.array([3.0, 0.0]), "u2": np.array([1.0, 1.0])}
        pairs = [("a", "u1"), ("b", "u1"), ("a", "u2"), ("a", "missing"), ("c", "u1")]
        scores, kept, skipped = scoring.score_trials(profiles, trials, pairs)
        np.testing.assert_allclose(scores, [1.0, 0.0, 1 / np.sqrt(2)])
        np.testing.assert_array_equal(kept, [True, True, True, False, False])
        self.assertEqual(skipped, 2)

    def test_zero_vectors_score_zero(self):
        profiles = scoring.build_profiles(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]), ["cancel", "cancel", "zero", "ok"])
        np.testing.assert_array_equal(profiles["cancel"], [0.0, 0.0])
        np.testing.assert_array_equal(profiles["zero"], [0.0, 0.0])
        trials = {"u": np.array([0.0, 1.0]), "empty": np.array([0.0, 0.0])}
        pairs = [("cancel", "u"), ("zero", "u"), ("ok", "empty"), ("ok", "u")]
        scores, kept, skipped = scoring.score_trials(profiles, trials, pairs)
        np.testing.assert_array_equal(scores, [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(kept.all())
        self.assertEqual(skipped, 0)

    def test_eer_separable_and_reversed(self):
        labels = np.array([1, 1, 0, 0])
        self.assertEqual(scoring.eer(np.array([0.9, 0.8, 0.2, 0.1]), labels), 0.0)
        self.assertEqual(scoring.eer(np.array([0.1, 0.2, 0.8, 0.9]), labels), 100.0)

    def test_eer_known_value(self):
        # one of four nontargets above one of four targets: FAR = FRR = 25 %
        scores = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(scoring.eer(scores, labels), 25.0)

    def test_eer_tied_scores_cannot_be_separated(self):
        # values of speechbrain.utils.metric_stats.EER on the same inputs
        self.assertEqual(scoring.eer_with_ties(np.array([0.5, 0.5]), np.array([1, 0])), (50.0, 1))
        self.assertEqual(scoring.eer(np.array([0.5, 0.5]), np.array([0, 1])), 50.0)

    def test_eer_takes_first_of_equally_good_thresholds(self):
        # two thresholds have the same |FAR - FRR|; SpeechBrain keeps the lower one and returns 75 %
        self.assertAlmostEqual(scoring.eer(np.array([3.0, 2.0, 1.0]), np.array([1, 0, 1])), 75.0)

    def test_eer_does_not_depend_on_trial_order(self):
        rng = np.random.default_rng(0)
        scores = np.round(rng.normal(size=400), 1)      # many ties
        labels = (rng.random(400) < 0.3).astype(int)
        reference = scoring.eer(scores, labels)
        for _ in range(5):
            order = rng.permutation(400)
            self.assertEqual(scoring.eer(scores[order], labels[order]), reference)

    def test_eer_matches_threshold_sweep(self):
        # direct transcription of the SpeechBrain loop
        rng = np.random.default_rng(1)
        scores = np.round(rng.normal(size=200) + (np.arange(200) < 60), 2)
        labels = (np.arange(200) < 60).astype(int)
        pos, neg = scores[labels == 1], scores[labels == 0]
        distinct = np.unique(scores)
        thresholds = np.sort(np.concatenate([distinct, (distinct[:-1] + distinct[1:]) / 2]))
        best = None
        for t in thresholds:
            frr, far = np.mean(pos <= t), np.mean(neg > t)
            if best is None or abs(far - frr) < abs(best[0] - best[1]):
                best = (far, frr)
        self.assertAlmostEqual(scoring.eer(scores, labels), 100 * (best[0] + best[1]) / 2, places=10)

    def test_eer_rejects_nan_and_single_class(self):
        with self.assertRaises(ValueError):
            scoring.eer(np.array([0.1, 0.2]), np.array([1, 1]))
        with self.assertRaises(ValueError):
            scoring.eer(np.array([0.1, np.nan]), np.array([1, 0]))


class LayeringTest(unittest.TestCase):
    def test_numpy_layer_does_not_import_torch(self):
        import subprocess
        import sys
        code = ("import sys; from text_attacker import kaldi_io, textnorm, scoring; "
                "bad = [m for m in ('torch', 'transformers', 'utils', 'evaluation') if m in sys.modules]; "
                "sys.exit(1 if bad else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-c", code]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
