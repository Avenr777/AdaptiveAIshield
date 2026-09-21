"""
Normalization / de-obfuscation.

Everything here tries to undo the mutator-style tricks the Red Team
uses (homoglyphs, zero-width spaces, leetspeak, spacing, base64) so the
lexical and semantic layers see something close to the attacker's
actual intent rather than its disguise. This is the single most
important defense against the "obfuscation_encoding" attack category —
without it, a purely keyword-based filter is trivially evaded (as the
dummy stand-in Blue Team demonstrated).
"""

from __future__ import annotations

import base64
import re
import unicodedata

# Reverse mapping of mutator.py's homoglyph substitutions, plus a few
# other common look-alikes attackers use. Built with both lower- and
# upper-case forms, since the mutator uppercases the substitute
# whenever the original letter was uppercase (e.g. 'I' -> Cyrillic 'І',
# not just lowercase 'i' -> 'і') — a map with only lowercase entries
# would miss every homoglyph at the start of a sentence.
_HOMOGLYPH_PAIRS = [
    ("а", "a"), ("е", "e"), ("о", "o"), ("і", "i"), ("с", "c"),  # Cyrillic
]
_HOMOGLYPH_MAP = {}
for _glyph, _latin in _HOMOGLYPH_PAIRS:
    _HOMOGLYPH_MAP[_glyph] = _latin
    _HOMOGLYPH_MAP[_glyph.upper()] = _latin.upper()
# A few extra look-alikes not produced by mutator.py but seen in the wild.
_HOMOGLYPH_MAP.update({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",  # Greek
    "０": "0", "１": "1", "２": "2",                       # fullwidth digits
})

# Matches mutator.py's forward map {"a":"4","e":"3","i":"1","o":"0","s":"5","t":"7"}
_LEET_MAP = str.maketrans({
    "4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "$": "s",
})
_LEET_CHARS = set("43105$7")

# Tokens made of letters/digits/$ — this is what unleet() operates on,
# one token at a time, rather than translating every digit in the
# entire text. That distinction matters: "I have 5 apples" should stay
# "I have 5 apples", not become "I have s apples". Leetspeak digits
# only make sense to reverse when they're mixed in with letters inside
# the same word (e.g. "in5truction5"); a standalone numeric token like
# "5" or "2024" is almost certainly just a number.
_TOKEN_RE = re.compile(r"[A-Za-z0-9$]+")

_ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff"]

_BASE64_LIKE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


def strip_zero_width(text: str) -> str:
    for ch in _ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return text


def unhomoglyph(text: str) -> str:
    return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def collapse_spacing(text: str) -> str:
    """Collapse 'i g n o r e' -> 'ignore' style spacing evasion without
    destroying normal multi-word text: only collapses runs of 4+
    single-letter tokens separated by single spaces."""
    def _collapse(match: re.Match) -> str:
        return match.group(0).replace(" ", "")

    pattern = re.compile(r"\b(?:[A-Za-z]\s){3,}[A-Za-z]\b")
    return pattern.sub(_collapse, text)


def unleet(text: str) -> str:
    def _maybe_unleet_token(match: re.Match) -> str:
        token = match.group(0)
        has_letter = any(ch.isalpha() for ch in token)
        has_leet_char = any(ch in _LEET_CHARS for ch in token)
        if has_letter and has_leet_char:
            return token.translate(_LEET_MAP)
        return token  # pure-numeric token (quantity, year, etc.) — leave it alone

    return _TOKEN_RE.sub(_maybe_unleet_token, text)


def _looks_like_text(s: str) -> bool:
    """
    Looser check than str.isprintable(): treats zero-width/format
    characters (category 'Cf', e.g. zero-width space) as acceptable
    rather than disqualifying, since that's exactly the kind of
    obfuscation a decoded payload might legitimately contain — a
    strict isprintable() check would silently reject a correctly
    decoded attack just because it still has zero-width spaces in it
    (those get cleaned up by strip_zero_width() on the next pass).
    Only rejects strings with a meaningful fraction of real control
    characters / binary noise, which indicates the base64 blob wasn't
    actually encoded text at all.
    """
    if not s:
        return False
    bad = sum(
        1 for ch in s
        if not ch.isprintable() and unicodedata.category(ch) != "Cf"
    )
    return (bad / len(s)) < 0.05


def try_decode_base64(text: str) -> str:
    """
    Find base64-looking substrings and replace them in-place with their
    decoded plaintext (if they decode to something that looks like text).

    Replacing in-place — rather than leaving the raw encoded blob in
    the text alongside the decoded content — matters for correctness:
    other normalization steps (unleet in particular) do digit/character
    substitution, and base64's alphabet is full of exactly those
    characters. If the raw blob were left behind, a later pass could
    corrupt it (e.g. '0' -> 'o') and make it undecodable on the next
    iteration. Replacing it removes that risk and keeps the pipeline
    idempotent: once decoded, there's no leftover encoded text for a
    later pass to re-match.
    """
    def _replace(match: re.Match) -> str:
        candidate = match.group(0)
        try:
            decoded = base64.b64decode(candidate, validate=True).decode("utf-8")
            if _looks_like_text(decoded):
                return decoded
        except Exception:
            pass
        return candidate  # not valid base64 / not text-like — leave untouched

    return _BASE64_LIKE.sub(_replace, text)


def unicode_normalize(text: str) -> str:
    """NFKC normalization also folds many fullwidth / compatibility
    characters into their ASCII equivalents."""
    return unicodedata.normalize("NFKC", text)


def _one_pass(text: str) -> str:
    text = unicode_normalize(text)
    text = strip_zero_width(text)
    text = unhomoglyph(text)
    text = collapse_spacing(text)
    text = try_decode_base64(text)
    text = unleet(text)
    return text


def normalize(text: str, max_iterations: int = 5) -> str:
    """
    Full normalization pipeline, applied before lexical/semantic scoring.

    Runs iteratively to a fixed point (or `max_iterations`, whichever
    comes first) rather than a single straight-line pass. This matters
    for COMPOUNDED obfuscation — e.g. an attack that's been through
    several rounds of the Red Team's mutator and now has homoglyphs
    layered on top of zero-width spaces layered on top of a base64
    wrapper. A single pass only undoes one layer per technique; looping
    until nothing changes anymore peels back however many layers are
    actually stacked, up to the iteration cap (which also prevents any
    pathological input from looping forever).
    """
    previous = None
    current = text
    for _ in range(max_iterations):
        if current == previous:
            break
        previous = current
        current = _one_pass(current)
    return current
