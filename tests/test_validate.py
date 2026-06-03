from logalot.validate import (
    confidence_flag,
    expand_phonetics,
    is_q_code,
    is_valid_call,
    looks_like_callsign,
    normalise_call,
)


def test_expand_phonetics_callsign():
    assert expand_phonetics("hotel bravo nine india kilo sierra") == "HB9IKS"


def test_expand_phonetics_mixed_punctuation():
    assert expand_phonetics("Whiskey-One, Alpha Whiskey.") == "W1AW"


def test_expand_phonetics_keeps_digits():
    assert expand_phonetics("Echo Golf 20") == "EG20"
    assert expand_phonetics("Echo Golf 2-0 Radio Charlie Hotel") == "EG20RCH"


def test_expand_phonetics_passes_through_call_fragments():
    # The LLM sometimes half-expands, leaving "HB9" then spelling the suffix.
    assert expand_phonetics("HB9 India Kilo Sierra") == "HB9IKS"


def test_expand_phonetics_german_alphabet():
    assert expand_phonetics("Delta Lima Eins Anton Berta Caesar") == "DL1ABC"
    assert expand_phonetics("Otto Emil neun Anton Berta") == "OE9AB"


def test_expand_phonetics_drops_non_phonetic_words():
    # 'fox' is not a phonetic word (NATO is 'foxtrot'); plain words are dropped.
    assert expand_phonetics("the quick brown fox") == ""
    assert expand_phonetics("hello there") == ""


def test_valid_calls():
    for c in ["HB9IKS", "W1AW", "4X4AA", "2E0ABC", "DL1ABC"]:
        assert is_valid_call(c), c


def test_valid_special_event_calls():
    # Multi-digit region groups: year-themed / anniversary special-event calls.
    for c in ["DL2026R", "EG20RCH", "OE100M"]:
        assert is_valid_call(c), c


def test_invalid_calls():
    for c in ["HELLO", "CQ", "123", ""]:
        assert not is_valid_call(c), c


def test_normalise_strips_portable_affix():
    core, affixes = normalise_call("HB9IKS/P")
    assert core == "HB9IKS"
    assert affixes == ["P"]


def test_normalise_prefix_form():
    core, affixes = normalise_call("DL/HB9IKS")
    assert core == "HB9IKS"
    assert "DL" in affixes


def test_confidence_flag():
    assert confidence_flag("HB9IKS") == "ok"
    assert confidence_flag("not-a-call") == "review"


def test_is_q_code_rule_covers_all():
    for q in ["QTH", "QSL", "QRZ", "QRM", "QSY", "QSB", "QRP"]:
        assert is_q_code(q), q
    assert not is_q_code("HB9IKS")
    assert not is_q_code("QQ")          # too short


def test_looks_like_callsign_rejects_cq_and_qcodes():
    assert looks_like_callsign("HB9IKS")
    assert looks_like_callsign("w1aw")
    for junk in ["CQ", "CQDX", "QRZ", "DX", "QTH", "73"]:
        assert not looks_like_callsign(junk), junk


def test_looks_like_callsign_rejects_partials():
    # A bare prefix+digit with no suffix is not a complete callsign.
    assert not looks_like_callsign("G4")
    assert not looks_like_callsign("EA")
    assert looks_like_callsign("G4WGU")
