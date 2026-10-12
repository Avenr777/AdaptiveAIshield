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
# Reverse mapping of mutator.py's homoglyph substitutions, plus other
# common look-alikes attackers use (Cyrillic, Greek, fullwidth).
_HOMOGLYPH_PAIRS = [
    ("а", "a"), ("е", "e"), ("о", "o"), ("і", "i"), ("с", "c"),
    ("р", "p"), ("у", "y"), ("х", "x"), ("ѕ", "s"), ("ј", "j"),
    ("ԛ", "q"), ("ԁ", "d"), ("ԛ", "q"),
]
_HOMOGLYPH_MAP = {}
for _glyph, _latin in _HOMOGLYPH_PAIRS:
    _HOMOGLYPH_MAP[_glyph] = _latin
    _HOMOGLYPH_MAP[_glyph.upper()] = _latin.upper()
# Extra uppercase Cyrillic & Greek look-alikes
_HOMOGLYPH_MAP.update({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})

# Matches mutator.py's forward map {"a":"4","e":"3","i":"1","o":"0","s":"5","t":"7"}
_LEET_MAP = str.maketrans({
    "4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "$": "s",
})
_LEET_CHARS = set("43105$7")

_TOKEN_RE = re.compile(r"[A-Za-z0-9$]+")
_ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff"]

# Base64 tokens: length 4+ with padding, or 8+ unpadded
_BASE64_LIKE = re.compile(r"(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{4,}={1,2}|[A-Za-z0-9+/]{8,})(?![A-Za-z0-9+/=])")
# Escaped hex sequences: \x72\x6d...
_HEX_ESCAPE = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
# Raw continuous hex tokens: 726d202d7266202f
_HEX_TOKEN = re.compile(r"\b(?:[0-9a-fA-F]{2}){4,}\b")


def strip_zero_width(text: str) -> str:
    for ch in _ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return text


def unhomoglyph(text: str) -> str:
    return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def collapse_whitespace(text: str) -> str:
    """Normalize and collapse consecutive whitespace characters."""
    return re.sub(r"[ \t\r\n\f\v]+", " ", text).strip()


def collapse_spacing(text: str) -> str:
    """Collapse 'i g n o r e' -> 'ignore' style spacing evasion."""
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
        return token

    return _TOKEN_RE.sub(_maybe_unleet_token, text)


def _looks_like_text(s: str) -> bool:
    if not s or len(s.strip()) == 0:
        return False
    # Check that it contains mostly printable text / spaces
    bad = sum(
        1 for ch in s
        if not ch.isprintable() and unicodedata.category(ch) != "Cf" and ch not in "\r\n\t"
    )
    return (bad / len(s)) < 0.05


def try_decode_hex(text: str) -> str:
    """Decode escaped hex (\\x63\\x61\\x74) and raw hex strings (726d202d7266202f)."""
    def _replace_escape(match: re.Match) -> str:
        chunk = match.group(0).replace("\\x", "")
        try:
            decoded = bytes.fromhex(chunk).decode("utf-8", errors="replace")
            if _looks_like_text(decoded):
                return decoded
        except Exception:
            pass
        return match.group(0)

    text = _HEX_ESCAPE.sub(_replace_escape, text)

    def _replace_raw_hex(match: re.Match) -> str:
        token = match.group(0)
        if len(token) % 2 != 0:
            return token
        try:
            decoded = bytes.fromhex(token).decode("utf-8", errors="replace")
            if _looks_like_text(decoded) and any(w in decoded.lower() for w in ["rm", "cat", "sh", "exec", "select", "drop", "sudo", "eval", "id"]):
                return decoded
        except Exception:
            pass
        return token

    return _HEX_TOKEN.sub(_replace_raw_hex, text)


def try_decode_base64(text: str) -> str:
    """Find base64 substrings (bounded depth) and replace with plaintext."""
    def _replace(match: re.Match) -> str:
        candidate = match.group(0)
        try:
            # Pad if needed
            pad_len = (4 - (len(candidate) % 4)) % 4
            padded = candidate + ("=" * pad_len)
            decoded = base64.b64decode(padded, validate=False).decode("utf-8", errors="replace")
            if _looks_like_text(decoded):
                return decoded
        except Exception:
            pass
        return candidate

    return _BASE64_LIKE.sub(_replace, text)


def unicode_normalize(text: str) -> str:
    """Unicode NFKC normalization folds compatibility and fullwidth glyphs."""
    return unicodedata.normalize("NFKC", text)


def _one_pass(text: str) -> str:
    text = unicode_normalize(text)
    text = strip_zero_width(text)
    text = collapse_whitespace(text)
    text = unhomoglyph(text)
    text = collapse_spacing(text)
    text = try_decode_hex(text)
    text = try_decode_base64(text)
    text = unleet(text)
    text = collapse_whitespace(text)
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
