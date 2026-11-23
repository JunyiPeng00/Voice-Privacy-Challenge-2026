import unittest

from text_attacker import textnorm


class TextNormTest(unittest.TestCase):
    def test_none_is_identity(self):
        text = "  Mr. Smith's 2 dogs — well-known!  "
        self.assertEqual(textnorm.normalise(text), text)

    def test_upper_letters(self):
        self.assertEqual(textnorm.normalise("Mr. Smith’s 2 dogs, well-known!", "upper_letters"),
                         "MR SMITH'S DOGS WELL KNOWN")

    def test_upper_letters_is_identity_on_librispeech_style_text(self):
        text = "HE DIDN'T KNOW WHAT TO SAY"
        self.assertEqual(textnorm.normalise(text, "upper_letters"), text)

    def test_lower_letters_keeps_hyphens_and_accents(self):
        self.assertEqual(textnorm.normalise("«Qu’est-ce que c’est?» dit-il, à 3 heures.", "lower_letters"),
                         "qu'est-ce que c'est dit-il à heures")

    def test_lower_letters_is_identity_on_mls_style_text(self):
        text = "weiß nicht wo die straße über den fluss führt"
        self.assertEqual(textnorm.normalise(text, "lower_letters"), text)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            textnorm.normalise("a", "lower")

    def test_count_digit_chars(self):
        self.assertEqual(textnorm.count_digit_chars("in 1984, 3 men"), 5)

    def test_training_filter(self):
        self.assertIsNone(textnorm.training_filter("in 1984"))
        self.assertEqual(textnorm.training_filter("Hello, World - it's fine."), "hello world - it's fine")


if __name__ == "__main__":
    unittest.main()
