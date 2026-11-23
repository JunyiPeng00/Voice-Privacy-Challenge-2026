import tempfile
import unittest
from pathlib import Path

import numpy as np

from text_attacker import kaldi_io


class KaldiIOTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, content):
        path = self.dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_read_pairs_splits_at_first_space(self):
        path = self.write("text", "utt1 HELLO  WORLD \n\nutt2 señor día\nutt3\n")
        self.assertEqual(kaldi_io.read_pairs(path), {"utt1": "HELLO  WORLD", "utt2": "señor día", "utt3": ""})

    def test_read_trials_last_label_wins(self):
        path = self.write("trials", "s1 u1 target\ns1 u2 nontarget\ns2 u1 nontarget\ns1 u2 target")
        pairs, labels = kaldi_io.read_trials(path)
        self.assertEqual(pairs, [("s1", "u1"), ("s1", "u2"), ("s2", "u1")])
        np.testing.assert_array_equal(labels, [1, 1, 0])

    def test_read_table_skips_header(self):
        with_header = self.write("a.tsv", "uttid\ttext\nu1\thello there\nu2\t\n")
        without = self.write("b.tsv", "u1\thello there\nu2\t\n")
        expected = {"u1": "hello there", "u2": ""}
        self.assertEqual(kaldi_io.read_table(with_header), expected)
        self.assertEqual(kaldi_io.read_table(without), expected)

    def test_write_scores_round_trip(self):
        pairs = [("s1", "u1"), ("s2", "u1")]
        path = self.dir / "new_dir" / "scores"
        kaldi_io.write_scores(path, pairs, np.array([0.25, -0.5]))
        rows = [line.split() for line in path.read_text().splitlines()]
        self.assertEqual([tuple(r[:2]) for r in rows], pairs)
        self.assertEqual([float(r[2]) for r in rows], [0.25, -0.5])


if __name__ == "__main__":
    unittest.main()
