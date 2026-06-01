from logalot.dxcc import country_for_call, has_known_prefix


def test_common_countries():
    cases = {
        "HB9IKS": "Switzerland",
        "DL1ABC": "Germany",
        "W1AW": "United States",
        "G4WGU": "England",
        "F5XYZ": "France",
        "I2ABC": "Italy",
        "EA4ABC": "Spain",
        "PA0ABC": "Netherlands",
        "JA1ABC": "Japan",
        "VK2ABC": "Australia",
        "4X4AA": "Israel",
        "9A1A": "Croatia",
    }
    for call, country in cases.items():
        assert country_for_call(call) == country, call


def test_longest_prefix_wins():
    # 3-char / specific blocks beat the broader 2-char allocation.
    assert country_for_call("EA8ABC") == "Canary Is."
    assert country_for_call("EA4ABC") == "Spain"
    assert country_for_call("HB0ABC") == "Liechtenstein"
    assert country_for_call("HB9IKS") == "Switzerland"


def test_uk_nations_disambiguated():
    assert country_for_call("GW4ABC") == "Wales"
    assert country_for_call("GM4ABC") == "Scotland"
    assert country_for_call("G4ABC") == "England"


def test_portable_prefix_uses_core():
    assert country_for_call("HB9IKS/P") == "Switzerland"


def test_unknown_prefix_is_none():
    assert country_for_call("QQ1ZZ") is None
    assert not has_known_prefix("QQ1ZZ")
    assert has_known_prefix("HB9IKS")
