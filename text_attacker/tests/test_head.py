import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from text_attacker.head import Head, load_head, save_head
except ImportError:  # the numpy-only tests still run without torch
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class HeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.pooled = torch.randn(5, 16)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, objective):
        return Head(speakers=7, objective=objective, hidden=16, bottleneck=8, embedding=4)

    def test_embeddings_have_unit_length(self):
        for objective in ("softmax", "aam-softmax"):
            emb = self.make(objective).embed(self.pooled)
            self.assertEqual(tuple(emb.shape), (5, 4))
            torch.testing.assert_close(emb.norm(dim=-1), torch.ones(5))

    def test_unknown_objective(self):
        with self.assertRaises(ValueError):
            self.make("aam")

    def test_aam_margin_lowers_only_the_target_logit(self):
        head = self.make("aam-softmax")
        targets = torch.tensor([0, 1, 2, 3, 4])
        plain, margin = head(self.pooled), head(self.pooled, targets)
        is_target = torch.nn.functional.one_hot(targets, 7).bool()
        self.assertTrue(torch.all(margin[is_target] < plain[is_target]))
        torch.testing.assert_close(margin[~is_target], plain[~is_target])

    def test_round_trip_projection_only(self):
        for objective in ("softmax", "aam-softmax"):
            head = self.make(objective).eval()
            save_head(self.dir / objective, head, {"name": "toy", "max_length": 96, "hyp_norm": "none"})
            loaded, meta = load_head(self.dir / objective)
            self.assertFalse(loaded.training)
            self.assertEqual((meta["objective"], meta["speakers"], meta["has_classifier"]), (objective, 7, False))
            self.assertTrue(torch.equal(loaded.embed(self.pooled), head.embed(self.pooled)))

    def test_round_trip_with_classifier(self):
        head = self.make("aam-softmax").eval()
        save_head(self.dir / "full", head, {"name": "toy"}, with_classifier=True)
        loaded, _ = load_head(self.dir / "full")
        self.assertTrue(torch.equal(loaded(self.pooled), head(self.pooled)))

    def test_mismatching_files_are_rejected(self):
        save_head(self.dir / "bad", self.make("softmax"), {"name": "toy"})
        info = json.loads((self.dir / "bad" / "head.json").read_text())
        info["has_classifier"] = True
        (self.dir / "bad" / "head.json").write_text(json.dumps(info))
        with self.assertRaises(RuntimeError):
            load_head(self.dir / "bad")

    def test_missing_head_gives_a_hint(self):
        with self.assertRaises(FileNotFoundError):
            load_head(self.dir / "nowhere")


if __name__ == "__main__":
    unittest.main()
