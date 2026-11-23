import unittest

import numpy as np

try:
    import torch
    from text_attacker.encoder import EMPTY_TEXT, PooledEncoder
    from text_attacker.tests.fakes import FakeModel, FakeTokenizer
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class PooledEncoderTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.encoder = PooledEncoder(backbone="fake", tokenizer=self.tokenizer, model=FakeModel())

    def test_mean_pool_ignores_padding_and_batching(self):
        texts = ["a b c d e", "a", "c d", "b b b"]
        together = self.encoder.pool(texts, batch_size=4)
        alone = np.concatenate([self.encoder.pool([t], batch_size=1) for t in texts])
        self.assertEqual(together.shape, (4, 16))
        self.assertEqual(together.dtype, np.float32)
        np.testing.assert_allclose(together, alone, atol=1e-6)

    def test_output_follows_input_order(self):
        texts = ["one two three four", "one", "one two"]
        pooled = self.encoder.pool(texts, batch_size=2)
        np.testing.assert_allclose(pooled[1], self.encoder.pool(["one"])[0], atol=1e-6)

    def test_empty_text_is_replaced(self):
        pooled = self.encoder.pool(["", "   ", EMPTY_TEXT])
        self.assertTrue(all(t == EMPTY_TEXT for t in self.tokenizer.seen))
        np.testing.assert_allclose(pooled[0], pooled[2], atol=1e-6)
        np.testing.assert_allclose(pooled[1], pooled[2], atol=1e-6)

    def test_truncation(self):
        long, short = "a b c d e f", "a b"
        np.testing.assert_allclose(self.encoder.pool([long], max_length=2), self.encoder.pool([short]), atol=1e-6)

    def test_runs_in_eval_mode_and_rejects_half_precision(self):
        self.assertFalse(self.encoder.model.training)
        with self.assertRaises(ValueError):
            PooledEncoder(backbone="fake", tokenizer=FakeTokenizer(), model=FakeModel().half())

    def test_fingerprint_names_backbone_and_revision(self):
        other = PooledEncoder(backbone="fake", revision="abc", tokenizer=FakeTokenizer(), model=FakeModel())
        self.assertNotEqual(self.encoder.fingerprint(), other.fingerprint())


if __name__ == "__main__":
    unittest.main()
