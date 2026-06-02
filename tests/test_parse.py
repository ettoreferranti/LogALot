"""Tests for the parse layer's MLX-independent logic: JSON extraction and schema
validation. The mlx-lm generate path itself needs Apple-Silicon hardware + a
downloaded model, so it is exercised manually, not in CI."""
from logalot.parse import MLXParser, _JSON_PREFIX, extract_json, validate_payload


class FakeRuntime:
    """Stands in for an LLMRuntime: returns canned continuations, mimicking the
    real generate() which prepends the priming prefix to its output."""

    def __init__(self, continuations):
        self.continuations = list(continuations)
        self.prefixes = []

    def generate(self, messages, prefix=""):
        self.prefixes.append(prefix)
        return prefix + self.continuations.pop(0)


def test_extract_plain_json():
    assert extract_json('{"call": "HB9XYZ", "mode": "SSB"}') == {"call": "HB9XYZ", "mode": "SSB"}


def test_extract_json_with_surrounding_prose():
    # Small local models sometimes wrap JSON in prose / a code fence.
    text = 'Sure! Here you go:\n```json\n{"call": "W1AW", "mode": "CW"}\n```\nHope that helps.'
    assert extract_json(text) == {"call": "W1AW", "mode": "CW"}


def test_extract_first_object_only():
    assert extract_json('{"call": "A"} {"call": "B"}') == {"call": "A"}


def test_extract_skips_unparseable_then_finds_valid():
    # A broken brace pair followed by a valid object.
    assert extract_json("{not json} {\"call\": \"DL1ABC\", \"mode\": \"FM\"}") == {
        "call": "DL1ABC", "mode": "FM"}


def test_extract_none_when_no_json():
    assert extract_json("no json here") is None


def test_validate_payload_ok():
    assert validate_payload({"call": "HB9IKS"})
    assert validate_payload({"call": "HB9IKS", "name": "Tom", "rst_sent": "59"})


def test_validate_payload_callsign_is_the_only_requirement():
    # mode/freq/band/time come from CAT, not the LLM — a bare call is valid,
    # but a payload without a call is not.
    assert validate_payload({"call": "HB9IKS"})
    assert not validate_payload({"mode": "SSB"})           # no call


def test_validate_payload_call_must_be_str():
    assert not validate_payload({"call": None})


def test_validate_payload_rejects_unknown_keys():
    assert not validate_payload({"call": "HB9IKS", "freq_mhz": "14"})


def test_loads_lenient_via_extract_json_trailing_comma():
    # Small models often emit a trailing comma; we should still parse.
    assert extract_json('{"call": "DL1ABC", "name": "Tom",}') == {"call": "DL1ABC", "name": "Tom"}


def test_parser_uses_json_priming_prefix():
    rt = FakeRuntime([' "HB9IKS", "name": "Tom"}'])
    out = MLXParser(runtime=rt).parse("hotel bravo nine india kilo sierra, name tom")
    assert out == {"call": "HB9IKS", "name": "Tom"}
    assert rt.prefixes == [_JSON_PREFIX]          # the assistant reply was primed


def test_parser_retries_once_on_unparseable_output():
    # First reply is prose-ish junk; the retry yields valid JSON.
    rt = FakeRuntime([" sorry, I cannot help with that", ' "W1AW"}'])
    out = MLXParser(runtime=rt).parse("whiskey one alpha whiskey")
    assert out == {"call": "W1AW"}
    assert len(rt.prefixes) == 2                   # primed both attempts


def test_parser_returns_none_when_call_is_null():
    rt = FakeRuntime([' null, "name": "Tom"}'])
    assert MLXParser(runtime=rt).parse("just chatting, no call") is None


def test_parse_over_extracts_from_and_to():
    rt = FakeRuntime([' "EA1ABC", "to_call": "CQ", "name": "Jose"}'])
    out = MLXParser(runtime=rt).parse_over("CQ CQ this is EA1ABC, name Jose")
    assert out == {"from_call": "EA1ABC", "to_call": "CQ", "name": "Jose"}
    assert rt.prefixes == ['{"from_call":']       # over-priming used


def test_parse_over_drops_nulls_and_unknown_keys():
    rt = FakeRuntime([' "G4XYZ", "to_call": null, "report": "59", "junk": "x"}'])
    out = MLXParser(runtime=rt).parse_over("this is G4XYZ, you are 59")
    assert out == {"from_call": "G4XYZ", "report": "59"}


def test_parse_over_empty_when_unparseable():
    rt = FakeRuntime([" sorry no json", " still no json"])
    assert MLXParser(runtime=rt).parse_over("noise") == {}
