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
        np.testing.assert_allclose(scores, [1.0, 0.0, 1 / np.sqrt(2)], atol=1e-6)
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

    def test_scores_follow_the_official_float32_cosine(self):
        # 1 - clip(1 - cos, 0, 2) in float32 rounds +-1e-8 to 0, as the official scorer does
        profiles = {"s": np.array([1.0, 0.0])}
        trials = {"target": np.array([1e-8, 1.0]), "nontarget": np.array([-1e-8, 1.0])}
        scores, _, _ = scoring.score_trials(profiles, trials, [("s", "target"), ("s", "nontarget")])
        self.assertEqual(scores.dtype, np.float32)
        np.testing.assert_array_equal(scores, [0.0, 0.0])
        self.assertEqual(scoring.eer(scores, np.array([1, 0])), 50.0)

    def test_no_scorable_trial(self):
        scores, kept, skipped = scoring.score_trials({}, {}, [("a", "u")])
        self.assertEqual((len(scores), kept.tolist(), skipped), (0, [False], 1))

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

    @staticmethod
    def speechbrain_loop(scores, labels):
        """speechbrain.utils.metric_stats.EER written out step by step with float32 numpy scalars."""
        scores = np.asarray(scores, dtype=np.float32)
        pos, neg = scores[np.asarray(labels) == 1], scores[np.asarray(labels) == 0]
        distinct = np.unique(scores)
        thresholds = np.sort(np.concatenate([distinct, (distinct[:-1] + distinct[1:]) / np.float32(2)]))
        final_far = final_frr = 0.0
        for i, t in enumerate(thresholds):
            frr = np.float32((pos <= t).sum()) / np.float32(len(pos))
            far = np.float32((neg > t).sum()) / np.float32(len(neg))
            if float(np.abs(far - frr)) < abs(final_far - final_frr) or i == 0:
                final_far, final_frr = float(far), float(frr)
        return 100 * (final_far + final_frr) / 2

    def test_eer_matches_speechbrain_loop(self):
        rng = np.random.default_rng(1)
        for n, n_target, decimals in ((200, 60, 2), (37, 5, 1), (500, 250, 3), (64, 7, 8)):
            labels = (np.arange(n) < n_target).astype(int)
            scores = np.round(rng.normal(size=n) + labels, decimals)
            self.assertEqual(scoring.eer(scores, labels), self.speechbrain_loop(scores, labels))

    def test_eer_float32_rates_decide_between_thresholds(self):
        # in float32 the gaps at thresholds -0.5 and 0 differ in the last bit; SpeechBrain returns 41.67 %
        scores, labels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([0, 1, 1, 0, 1])
        self.assertAlmostEqual(scoring.eer(scores, labels), 100 * 5 / 12, places=4)
        self.assertEqual(scoring.eer(scores, labels), self.speechbrain_loop(scores, labels))

    def test_eer_scores_equal_in_float32_are_tied(self):
        self.assertEqual(scoring.eer_with_ties(np.array([0.5, 0.50000001]), np.array([0, 1])), (50.0, 1))

    def test_eer_adjacent_float32_scores(self):
        low = np.float32(0.5)
        high = np.nextafter(low, np.float32(1))      # the midpoint rounds onto one of the two
        self.assertEqual(scoring.eer(np.array([low, high]), np.array([0, 1])), 0.0)
        self.assertEqual(scoring.eer(np.array([low, high]), np.array([1, 0])), 100.0)

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
