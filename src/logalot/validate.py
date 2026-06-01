"""Validation and normalisation for parsed QSO fields.

Implemented and stdlib-only. The callsign regex is pragmatic, not a formal ITU
grammar: it accepts the vast majority of real calls and rejects obvious noise.
Portable prefixes/suffixes (DL/HB9IKS, HB9IKS/P) are split off before matching.
"""
from __future__ import annotations

import re

# Pragmatic callsign pattern: prefix (1-3 alnum incl. a digit) + region digit +
# 1-4 letter suffix. Matches HB9IKS, W1AW, 4X4AA, 2E0ABC; rejects bare words.
_CALL_RE = re.compile(r"^[A-Z0-9]{1,3}[0-9][A-Z]{1,4}$")

# Affixes carrying no call identity; stripped before validation, kept as context.
_KNOWN_SUFFIXES = {"P", "M", "MM", "AM", "QRP", "A"}

# Calling indicators and Q-codes the parser sometimes mistakes for a worked
# callsign. None of these is ever a valid amateur call, so the candidate builder
# drops a "call" that matches one of them.
NON_CALLSIGNS = {
    "CQ", "CQCQ", "CQDX", "QRZ", "DX", "DE", "SOS", "TEST",
    "QSL", "QTH", "QRM", "QRN", "QSB", "QSY", "QSO", "QRP", "QRL", "73", "88",
}


def looks_like_callsign(call: str) -> bool:
    """A worked-station call worth showing: not a CQ/Q-code, and call-shaped."""
    c = call.upper().strip()
    return c not in NON_CALLSIGNS and is_valid_call(c)

NATO = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D",
    "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I",
    "juliett": "J", "juliet": "J", "kilo": "K", "lima": "L", "mike": "M",
    "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R",
    "sierra": "S", "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W",
    "whisky": "W", "xray": "X", "x-ray": "X", "yankee": "Y", "zulu": "Z",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "niner": "9", "nine": "9", "six": "6", "seven": "7", "eight": "8",
}


def expand_phonetics(text: str) -> str:
    """Turn 'hotel bravo nine india kilo sierra' into 'HB9IKS'.

    Tokens that are NATO words/digits map to their letter/digit; anything else is
    dropped. Order is preserved. Whisper output is lower-cased and punctuated, so
    we normalise aggressively.
    """
    out: list[str] = []
    for tok in re.split(r"[\s.,/-]+", text.strip().lower()):
        if not tok:
            continue
        if tok in NATO:
            out.append(NATO[tok])
    return "".join(out)


def normalise_call(raw: str) -> tuple[str, list[str]]:
    """Upper-case, strip whitespace, split off portable affixes.

    Returns (core_call, affixes). Affixes are the non-core parts (e.g. ['P'] or a
    DXCC prefix) so the caller can record them in the comment field.
    """
    parts = [p for p in raw.upper().strip().split("/") if p]
    if not parts:
        return "", []
    # The core is the longest part that looks most call-like; affixes are the rest.
    core = max(parts, key=lambda p: (_CALL_RE.match(p) is not None, len(p)))
    affixes = [p for p in parts if p != core]
    return core, affixes


def is_valid_call(call: str) -> bool:
    return bool(_CALL_RE.match(call.upper()))


def confidence_flag(call: str) -> str:
    """Coarse confidence label for the review queue. M3 may replace this with
    model logprobs."""
    return "ok" if is_valid_call(normalise_call(call)[0]) else "review"
