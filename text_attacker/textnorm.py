"""Text normalisation for ASR hypotheses.

Reference transcripts are fed to the attacker as they are. Normalisation is
applied to ASR hypotheses only, to bring them to the convention of the text
each head was trained on:

    none            identity
    upper_letters   upper case; letters, spaces and apostrophes only
                    (LibriSpeech convention, Track 1 head)
    lower_letters   lower case; letters, spaces, apostrophes and hyphens only
                    (MLS convention, Track 2 heads)

Digits are removed in both letter modes. ``count_digit_chars`` reports how
many, since a digit stands for a spoken word that the attacker then misses.
"""
import re
import unicodedata

MODES = ("none", "upper_letters", "lower_letters")

# a Unicode letter, i.e. a word character that is neither a digit nor "_"
_LETTER = r"[^\W\d_]"
_KEEP_UPPER = re.compile(rf"{_LETTER}|[ ']", re.UNICODE)
_KEEP_LOWER = re.compile(rf"{_LETTER}|[ '\-]", re.UNICODE)


def _canonical(text):
    return unicodedata.normalize("NFC", text).replace("’", "'")


def normalise(text, mode="none"):
    """Normalise one hypothesis. See the module docstring for the modes."""
    if mode == "none":
        return text
    if mode == "upper_letters":
        # other characters become spaces, so that "well-known" stays two words
        text = _canonical(text).upper()
        kept = "".join(ch if _KEEP_UPPER.fullmatch(ch) else " " for ch in text)
        return " ".join(kept.split())
    if mode == "lower_letters":
        # other characters are deleted, as when the Track 2 training text was built
        text = _canonical(text).lower()
        kept = "".join(ch for ch in text if _KEEP_LOWER.fullmatch(ch))
        return " ".join(kept.split())
    raise ValueError(f"Unknown normalisation mode '{mode}', expected one of {MODES}")


def count_digit_chars(text):
    """Number of digit characters in a text."""
    return sum(ch.isdigit() for ch in text)


def training_filter(text):
    """Normalise a training transcript of a Track 2 head.

    Utterances that contain digits are dropped (None is returned). The others
    are lower-cased and reduced to letters, spaces, apostrophes and hyphens.
    Unlike ``lower_letters`` this applies no Unicode canonicalisation: it is
    the exact function the released heads were trained with.
    """
    if re.search(r"\d", text):
        return None
    kept = "".join(ch for ch in text.lower() if _KEEP_LOWER.fullmatch(ch))
    return " ".join(kept.split())
