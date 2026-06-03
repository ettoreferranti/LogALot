"""Validation and normalisation for parsed QSO fields.

Implemented and stdlib-only. The callsign regex is pragmatic, not a formal ITU
grammar: it accepts the vast majority of real calls and rejects obvious noise.
Portable prefixes/suffixes (DL/HB9IKS, HB9IKS/P) are split off before matching.
"""
from __future__ import annotations

import re

# Pragmatic callsign pattern: prefix (1-3 alnum incl. a digit) + a digit group +
# 1-4 letter suffix. The digit group is 1-4 digits so special-event/year calls
# (DL2026R, EG20XXX) validate alongside ordinary ones (HB9IKS, W1AW, 4X4AA,
# 2E0ABC); still rejects bare words. The wider group lets more phonetic garble
# through too, but candidates are advisory (the human commits), and missing the
# band's dominant station is the worse failure.
_CALL_RE = re.compile(r"^[A-Z0-9]{1,3}[0-9]{1,4}[A-Z]{1,4}$")

# Every Q-code is the letter Q followed by two letters (QTH, QSL, QRZ, …). One
# rule covers all of them — no list to maintain — so the parser can never log a
# Q-code as a callsign.
_Q_CODE_RE = re.compile(r"^Q[A-Z]{2}$")

# Affixes carrying no call identity; stripped before validation, kept as context.
_KNOWN_SUFFIXES = {"P", "M", "MM", "AM", "QRP", "A"}

# Calling indicators and Q-codes the parser sometimes mistakes for a worked
# callsign. None of these is ever a valid amateur call, so the candidate builder
# drops a "call" that matches one of them.
NON_CALLSIGNS = {
    "CQ", "CQCQ", "CQDX", "QRZ", "DX", "DE", "SOS", "TEST",
    "QSL", "QTH", "QRM", "QRN", "QSB", "QSY", "QSO", "QRP", "QRL", "73", "88",
}


def is_q_code(token: str) -> bool:
    return bool(_Q_CODE_RE.match(token.upper().strip()))


def looks_like_callsign(call: str) -> bool:
    """A complete, worked-station call worth putting in the candidate's call field:
    a full callsign (prefix+digit+suffix), not a Q-code, CQ, or other indicator.
    Partials like 'G4' (no suffix) fail this."""
    c = call.upper().strip()
    return c not in NON_CALLSIGNS and not is_q_code(c) and is_valid_call(c)

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

# German (DIN 5009) spelling alphabet + numbers — HB9 works a lot of DL/OE/HB.
GERMAN = {
    "anton": "A", "berta": "B", "cäsar": "C", "caesar": "C", "casar": "C",
    "dora": "D", "emil": "E", "friedrich": "F", "gustav": "G", "heinrich": "H",
    "ida": "I", "julius": "J", "kaufmann": "K", "konrad": "K", "ludwig": "L",
    "martha": "M", "nordpol": "N", "otto": "O", "paula": "P", "quelle": "Q",
    "richard": "R", "samuel": "S", "siegfried": "S", "theodor": "T",
    "ulrich": "U", "viktor": "V", "wilhelm": "W", "xaver": "X", "xanthippe": "X",
    "ypsilon": "Y", "zacharias": "Z", "zeppelin": "Z",
    "null": "0", "eins": "1", "zwei": "2", "zwo": "2", "drei": "3", "vier": "4",
    "fünf": "5", "funf": "5", "sechs": "6", "sieben": "7", "acht": "8", "neun": "9",
}

# Russian spelling alphabet (best-effort: words whose Latin transliteration starts
# with the letter they stand for) + numbers. Russian ops often use NATO too.
RUSSIAN = {
    "anna": "A", "boris": "B", "vasiliy": "V", "vasili": "V", "galina": "G",
    "dmitriy": "D", "dmitri": "D", "yelena": "E", "elena": "E", "zinaida": "Z",
    "ivan": "I", "konstantin": "K", "leonid": "L", "mikhail": "M",
    "nikolai": "N", "nikolay": "N", "olga": "O", "pavel": "P", "roman": "R",
    "sergei": "S", "sergey": "S", "semyon": "S", "tatiana": "T", "tatyana": "T",
    "ulyana": "U", "fyodor": "F", "fedor": "F",
    "nol": "0", "nul": "0", "odin": "1", "dva": "2", "tri": "3", "chetyre": "4",
    "pyat": "5", "shest": "6", "sem": "7", "vosem": "8", "devyat": "9",
}

# Improvised / old-style phonetics: operators substitute place/word names,
# especially for R, A, O. Each maps to its initial letter.
INFORMAL = {
    "radio": "R", "america": "A", "ocean": "O", "germany": "G", "england": "E",
    "italy": "I", "japan": "J", "norway": "N", "mexico": "M", "canada": "C",
    "denmark": "D", "florida": "F", "portugal": "P", "london": "L",
    "boston": "B", "santiago": "S", "kilowatt": "K", "washington": "W",
    "united": "U", "victoria": "V", "queen": "Q", "young": "Y", "zanzibar": "Z",
    "ontario": "O", "honolulu": "H",
}

# Merged lookup. Keys are distinct across alphabets, so a plain merge is safe.
PHONETICS = {**NATO, **GERMAN, **RUSSIAN, **INFORMAL}


def expand_phonetics(text: str) -> str:
    """Turn spoken phonetics into callsign characters: 'hotel bravo nine india
    kilo sierra' -> 'HB9IKS', 'Echo Golf 20 Radio Charlie Hotel' -> 'EG20RCH'.

    Per token, in order: a known phonetic word (NATO/German/Russian) maps to its
    letter/digit; a bare number is kept; an alphanumeric chunk that already
    contains a digit (e.g. 'HB9', 'EG20') is passed through as a call fragment;
    anything else is dropped. Whisper output is lower-cased and punctuated, so we
    split aggressively.
    """
    out: list[str] = []
    for tok in re.split(r"[\s.,/-]+", text.strip().lower()):
        if not tok:
            continue
        if tok in PHONETICS:
            out.append(PHONETICS[tok])
        elif tok.isdigit():
            out.append(tok)
        elif tok.isalnum() and any(ch.isdigit() for ch in tok):
            out.append(tok.upper())
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
