"""
Text cleaning for RM visit remarks.

Deliberately minimal — lowercase, whitespace normalisation, symbol/bullet
stripping, mojibake cleanup, repeated-punctuation collapsing.

NO stemming, NO lemmatization, NO keyword removal.  Business vocabulary like
"escalated", "SEC", "root cause" carries the signal the model depends on.

This must exactly replicate the training notebook's Section 3 logic.
"""

from __future__ import annotations

import re
import unicodedata


def clean_remark_text(text: str) -> str:
    """
    Clean a raw RM remark for downstream feature extraction and embedding.

    Parameters
    ----------
    text : str
        Raw remark string (may contain bullets, symbols, mojibake, etc.).

    Returns
    -------
    str
        Cleaned, lowercased remark text.
    """
    if not text or not isinstance(text, str):
        return ""

    # ── Lowercase ────────────────────────────────────────────────────────
    text = text.lower()

    # ── Unicode normalisation (NFC) — fix mojibake / combining chars ─────
    text = unicodedata.normalize("NFC", text)

    # ── Strip common mojibake artefacts ──────────────────────────────────
    # Characters like â€™ (UTF-8 decoded as latin-1), Â, etc.
    mojibake_patterns = [
        ("\u00e2\u0080\u0099", "'"),    # right single quote mojibake
        ("\u00e2\u0080\u009c", '"'),    # left double quote mojibake
        ("\u00e2\u0080\u009d", '"'),    # right double quote mojibake
        ("\u00e2\u0080\u0093", "-"),    # en-dash mojibake
        ("\u00e2\u0080\u0094", "-"),    # em-dash mojibake
        ("\u00c2", ""),                 # stray A-circumflex from UTF-8/latin-1
        ("\u00c3\u00a9", "e"),          # accented e mojibake
        ("\u00c3\u00a2", "a"),          # accented a mojibake
    ]
    for pattern, replacement in mojibake_patterns:
        text = text.replace(pattern, replacement)

    # ── Remove bullet / list marker noise ────────────────────────────────
    # Common patterns: •, ●, ◦, ▪, -, *, numbered bullets (1., 2., etc.)
    text = re.sub(r"^[\s]*[•●◦▪▸▹►▻\-\*]+\s*", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+[.)]\s*", " ", text, flags=re.MULTILINE)

    # ── Collapse multiple punctuation ────────────────────────────────────
    text = re.sub(r"([.!?,;:])\1+", r"\1", text)

    # ── Strip non-alphanumeric symbols (keep hyphens, apostrophes) ───────
    # Keep: letters, digits, spaces, basic punctuation (. , ! ? ; : - ')
    text = re.sub(r"[^\w\s.,!?;:\-']", " ", text)

    # ── Whitespace normalisation ─────────────────────────────────────────
    text = re.sub(r"\s+", " ", text).strip()

    return text
