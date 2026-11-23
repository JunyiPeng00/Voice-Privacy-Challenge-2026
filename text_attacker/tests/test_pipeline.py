import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from text_attacker import pipeline, scoring
    from text_attacker.encoder import PooledEncoder
    from text_attacker.head import Head
    from text_attacker.tests.fakes import FakeModel, FakeTokenizer
except ImportError:
    torch = None

ENROL_TEXT = {"s1-a": "the ship sailed north", "s1-b": "the ship sank", "s2-a": "bread and butter", "s2-b": "butter cake"}
ENROL_UTT2SPK = {"s1-a": "s1", "s1-b": "s1", "s2-a": "s2", "s2-b": "s2"}
TRIAL_TEXT = {"t1": "the ship sailed", "t2": "bread and cake", "unused": "something else"}
PAIRS = [("s1", "t1"), ("s2", "t1"), ("s1", "t2"), ("s2", "t2"), ("s1", "absent")]
LABELS = np.array([1, 0, 0, 1, 0])


@unittest.skipIf(torch is None, "torch is not installed")
class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokenizer = FakeTokenizer()
        self.encoder = PooledEncoder(backbone="fake", tokenizer=self.tokenizer, model=FakeModel())
        self.head = Head(speakers=3, objective="softmax", hidden=16, bottleneck=8, embedding=4).eval()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_embed_texts_matches_manual_path(self):
        emb = pipeline.embed_texts(self.encoder, self.head, ENROL_TEXT)
        self.assertEqual(sorted(emb), sorted(ENROL_TEXT))
        manual = self.head.embed(torch.from_numpy(self.encoder.pool([ENROL_TEXT["s2-a"]]))).detach().numpy()[0]
        np.testing.assert_allclose(emb["s2-a"], manual, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(emb["s2-a"]), 1.0, atol=1e-6)
        self.assertEqual(pipeline.embed_texts(self.encoder, self.head, {}), {})

    def test_cache_is_reused_and_keyed_by_text(self):
        first = pipeline.embed_texts(self.encoder, self.head, ENROL_TEXT, cache_dir=self.dir)
        calls = len(self.tokenizer.seen)
        second = pipeline.embed_texts(self.encoder, self.head, ENROL_TEXT, cache_dir=self.dir)
        self.assertEqual(len(self.tokenizer.seen), calls)                # served from the cache
        np.testing.assert_array_equal(first["s1-a"], second["s1-a"])
        changed = dict(ENROL_TEXT, **{"s1-a": "a different sentence"})
        pipeline.embed_texts(self.encoder, self.head, changed, cache_dir=self.dir)
        self.assertGreater(len(self.tokenizer.seen), calls)              # new text, new key
        self.assertEqual(len(list(self.dir.glob("pooled_*.npy"))), 2)

    def test_cache_key_separates_ambiguous_inputs(self):
        one = pipeline._cache_key(self.encoder, ["a"], ["x\nb\ty"], 96)
        two = pipeline._cache_key(self.encoder, ["a", "b"], ["x", "y"], 96)
        self.assertNotEqual(one, two)
        self.assertNotEqual(two, pipeline._cache_key(self.encoder, ["a", "b"], ["x", "y"], 32))

    def test_cache_key_follows_the_encoder_weights(self):
        other = PooledEncoder(backbone="fake", tokenizer=FakeTokenizer(), model=FakeModel())
        self.assertEqual(self.encoder.fingerprint(), other.fingerprint())     # same weights, same features
        with torch.no_grad():
            other.model.embedding.weight[3, 0] += 1.0
        other._fingerprint = None
        self.assertNotEqual(self.encoder.fingerprint(), other.fingerprint())

    def test_broken_cache_entry_is_recomputed_and_replaced(self):
        ids, texts = sorted(ENROL_TEXT), [ENROL_TEXT[i] for i in sorted(ENROL_TEXT)]
        good = pipeline.cached_pool(self.encoder, ids, texts, cache_dir=self.dir)
        cache_file = next(self.dir.glob("pooled_*.npy"))
        for broken in (b"", cache_file.read_bytes()[:100]):                    # empty and truncated files
            cache_file.write_bytes(broken)
            np.testing.assert_array_equal(pipeline.cached_pool(self.encoder, ids, texts, cache_dir=self.dir), good)
            np.testing.assert_array_equal(np.load(cache_file), good)
        np.save(cache_file, good[:2])                                          # wrong number of rows
        np.testing.assert_array_equal(pipeline.cached_pool(self.encoder, ids, texts, cache_dir=self.dir), good)
        self.assertEqual([p.name for p in self.dir.iterdir()], [cache_file.name])   # no temporary file left

    def test_refresh_replaces_the_cache_entry(self):
        ids, texts = sorted(ENROL_TEXT), [ENROL_TEXT[i] for i in sorted(ENROL_TEXT)]
        good = pipeline.cached_pool(self.encoder, ids, texts, cache_dir=self.dir)
        cache_file = next(self.dir.glob("pooled_*.npy"))
        np.save(cache_file, np.zeros_like(good))                               # a stale but well-formed entry
        np.testing.assert_array_equal(pipeline.cached_pool(self.encoder, ids, texts, cache_dir=self.dir), 0 * good)
        np.testing.assert_array_equal(
            pipeline.cached_pool(self.encoder, ids, texts, cache_dir=self.dir, refresh=True), good)
        np.testing.assert_array_equal(np.load(cache_file), good)

    def test_evaluate_pair(self):
        result = pipeline.evaluate_pair(ENROL_TEXT, ENROL_UTT2SPK, TRIAL_TEXT, PAIRS, LABELS, self.encoder, self.head)
        self.assertEqual((result["n_trials"], result["n_target"], result["n_nontarget"], result["n_skipped"]), (4, 2, 2, 1))
        self.assertEqual(result["pairs"], PAIRS[:4])
        self.assertNotIn(TRIAL_TEXT["unused"], self.tokenizer.seen)
        # same numbers from the building blocks
        enrol = pipeline.embed_texts(self.encoder, self.head, ENROL_TEXT)
        trial = pipeline.embed_texts(self.encoder, self.head, TRIAL_TEXT)
        ids = sorted(ENROL_TEXT)
        profiles = scoring.build_profiles([enrol[u] for u in ids], [ENROL_UTT2SPK[u] for u in ids])
        scores, _, _ = scoring.score_trials(profiles, trial, PAIRS)
        np.testing.assert_allclose(result["scores"], scores, atol=1e-6)
        self.assertEqual(result["eer"], result["eer_numpy"])

    def test_evaluate_pair_uses_given_eer_function(self):
        seen = {}

        def eer_fn(target, nontarget):
            seen["sizes"] = (len(target), len(nontarget))
            return 12.5

        result = pipeline.evaluate_pair(ENROL_TEXT, ENROL_UTT2SPK, TRIAL_TEXT, PAIRS, LABELS, self.encoder, self.head,
                                        eer_fn=eer_fn)
        self.assertEqual(result["eer"], 12.5)
        self.assertEqual(seen["sizes"], (2, 2))

    def test_embeddings_round_trip(self):
        emb = pipeline.embed_texts(self.encoder, self.head, ENROL_TEXT)
        pipeline.save_embeddings(self.dir / "sub" / "enrol.npz", emb, speakers=ENROL_UTT2SPK, role="enroll")
        loaded, speakers = pipeline.load_embeddings(self.dir / "sub" / "enrol.npz")
        np.testing.assert_array_equal(loaded["s2-b"], emb["s2-b"])
        self.assertEqual(speakers["s2-b"], "s2")


if __name__ == "__main__":
    unittest.main()
