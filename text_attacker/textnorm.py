"""Text normalization for ASR hypotheses.

Reference transcripts are fed to the attacker as they are. Normalization is
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

# the character filter of the Track 2 training text; it lets numeric characters such as "²" through,
# which normalize() does not (it keeps letters only)
_KEEP_LOWER = re.compile(r"[^\W\d_]|[ '\-]", re.UNICODE)


def _canonical(text):
    return unicodedata.normalize("NFC", text).replace("’", "'")


def normalize(text, mode="none"):
    """Normalize one hypothesis. See the module docstring for the modes."""
    if mode == "none":
        return text
    if mode == "upper_letters":
        # other characters become spaces, so that "well-known" stays two words
        text = _canonical(text).upper()
        kept = "".join(ch if ch.isalpha() or ch in " '" else " " for ch in text)
        return " ".join(kept.split())
    if mode == "lower_letters":
        # other characters are deleted, as when the Track 2 training text was built
        text = _canonical(text).lower()
        kept = "".join(ch for ch in text if ch.isalpha() or ch in " '-")
        return " ".join(kept.split())
    raise ValueError(f"Unknown normalization mode '{mode}', expected one of {MODES}")


def count_digit_chars(text):
    """Number of digit characters in a text."""
    return sum(ch.isdigit() for ch in text)


def training_filter(text):
    """Normalize a training transcript of a Track 2 head.

    Utterances that contain digits are dropped (None is returned). The others
    are lower-cased and reduced to letters, spaces, apostrophes and hyphens.
    Unlike ``lower_letters`` this applies no Unicode canonicalization: it is
    the exact function the released heads were trained with.
    """
    if re.search(r"\d", text):
        return None
    kept = "".join(ch for ch in text.lower() if _KEEP_LOWER.fullmatch(ch))
    return " ".join(kept.split())
