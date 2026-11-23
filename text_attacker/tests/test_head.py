import json
import tempfile
import unittest
from pathlib import Path

try:
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from text_attacker.head import Head, load_head, save_head
    from text_attacker.tools.export_head import head_from_checkpoint
except ImportError:  # the numpy-only tests still run without torch
    torch = None


if torch is not None:
    class OriginalHead(nn.Module):
        """The head as written in the code the released heads were trained with; the reference for Head."""

        def __init__(self, speakers, aam):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))
            self.aam = aam
            if aam:
                self.weight = nn.Parameter(torch.empty(speakers, 256)); nn.init.xavier_normal_(self.weight)
            else:
                self.head = nn.Linear(256, speakers, bias=False)

        def embed(self, x):
            return F.normalize(self.project(x), dim=-1)

        def forward(self, x, y=None, s=30.0, m=0.2):
            z = self.embed(x)
            if not self.aam:
                return self.head(z)
            cos = F.linear(z, F.normalize(self.weight, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
            if y is None:
                return s * cos
            sine = torch.sqrt((1 - cos.pow(2)).clamp(min=1e-2))
            sh = cos * math.cos(m) - sine * math.sin(m)
            sh = torch.where(cos > math.cos(math.pi - m), sh, cos - m * math.sin(m))
            return s * torch.where(F.one_hot(y, cos.size(1)).bool(), sh, cos)


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

    def test_same_initialisation_and_outputs_as_the_original(self):
        features = torch.randn(6, 1024)
        targets = torch.tensor([0, 1, 2, 3, 4, 5])
        for objective, aam in (("softmax", False), ("aam-softmax", True)):
            torch.manual_seed(1701)
            original = OriginalHead(9, aam)
            torch.manual_seed(1701)
            head = Head(speakers=9, objective=objective)
            self.assertEqual(list(head.state_dict()), list(original.state_dict()))
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(head.state_dict()[key], value), key)
            self.assertTrue(torch.equal(head.embed(features), original.embed(features)))
            self.assertTrue(torch.equal(head(features), original(features)))
            self.assertTrue(torch.equal(head(features, targets), original(features, targets)))

    def test_aam_margin_branches_match_the_original(self):
        # embeddings placed at chosen angles to the class weights: both margin branches and the sine floor
        torch.manual_seed(3)
        original = OriginalHead(4, True)
        head = Head(speakers=4, objective="aam-softmax")
        head.load_state_dict(original.state_dict())
        for module in (original, head):
            module.project = nn.Identity()
        directions = F.normalize(original.weight.detach(), dim=-1)
        features = torch.cat([directions, -directions, directions + 1e-4 * torch.randn(4, 256)])
        targets = torch.arange(4).repeat(3)
        self.assertTrue(torch.equal(head(features, targets), original(features, targets)))

    def test_export_from_training_checkpoint(self):
        for objective, aam in (("softmax", False), ("aam-softmax", True)):
            original = OriginalHead(5, aam)
            state = dict(original.state_dict())
            state["encoder.embeddings.weight"] = torch.randn(3, 2)
            torch.save({"model": state, "config": {"max_length": 96, "backbone": "/gone"}}, self.dir / "best.pt")
            head, config = head_from_checkpoint(self.dir / "best.pt")
            self.assertEqual((head.objective, config["max_length"]), (objective, 96))
            save_head(self.dir / objective, head, {"name": "toy"})
            loaded, _ = load_head(self.dir / objective)
            for key, value in original.state_dict().items():
                if key.startswith("project."):
                    self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)
            features = torch.randn(3, 1024)
            self.assertTrue(torch.equal(loaded.embed(features), original.embed(features)))

    def test_projection_only_head_has_no_logits(self):
        for objective in ("softmax", "aam-softmax"):
            save_head(self.dir / objective, self.make(objective), {"name": "toy"})
            loaded, _ = load_head(self.dir / objective)
            self.assertFalse(loaded.has_classifier)
            with self.assertRaises(RuntimeError):
                loaded(self.pooled)
            with self.assertRaises(ValueError):
                save_head(self.dir / "again", loaded, {"name": "toy"}, with_classifier=True)
            save_head(self.dir / "again", loaded, {"name": "toy"})           # projection only is fine

    def test_round_trip_projection_only(self):
        for objective in ("softmax", "aam-softmax"):
            head = self.make(objective).eval()
            save_head(self.dir / objective, head, {"name": "toy", "max_length": 96, "hyp_norm": "none"})
            loaded, meta = load_head(self.dir / objective)
            self.assertFalse(loaded.training)
            self.assertEqual((meta["objective"], meta["speakers"], meta["has_classifier"]), (objective, 7, False))
            self.assertTrue(torch.equal(loaded.embed(self.pooled), head.embed(self.pooled)))

    def test_round_trip_with_classifier(self):
        targets = torch.tensor([0, 1, 2, 3, 4])
        for objective in ("softmax", "aam-softmax"):
            head = self.make(objective).eval()
            save_head(self.dir / objective, head, {"name": "toy"}, with_classifier=True)
            loaded, _ = load_head(self.dir / objective)
            self.assertTrue(loaded.has_classifier)
            self.assertTrue(torch.equal(loaded(self.pooled), head(self.pooled)))
            self.assertTrue(torch.equal(loaded(self.pooled, targets), head(self.pooled, targets)))

    def test_mismatching_files_are_rejected(self):
        # head.json and head.safetensors must agree on the classifier, in both directions
        for objective in ("softmax", "aam-softmax"):
            for saved_with, claimed in ((False, True), (True, False)):
                path = self.dir / f"{objective}-{saved_with}"
                save_head(path, self.make(objective), {"name": "toy"}, with_classifier=saved_with)
                info = json.loads((path / "head.json").read_text())
                info["has_classifier"] = claimed
                (path / "head.json").write_text(json.dumps(info))
                with self.assertRaises(RuntimeError):
                    load_head(path)

    def test_missing_head_gives_a_hint(self):
        with self.assertRaises(FileNotFoundError):
            load_head(self.dir / "nowhere")


if __name__ == "__main__":
    unittest.main()
