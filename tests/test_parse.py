"""Tests for the parse layer's MLX-independent logic: JSON extraction and schema
validation. The mlx-lm generate path itself needs Apple-Silicon hardware + a
downloaded model, so it is exercised manually, not in CI."""
from logalot.parse import extract_json, validate_payload


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
    assert validate_payload({"call": "HB9IKS", "mode": "SSB"})
    assert validate_payload({"call": "HB9IKS", "mode": "SSB", "name": "Tom", "rst_sent": "59"})


def test_validate_payload_missing_required():
    assert not validate_payload({"call": "HB9IKS"})        # no mode
    assert not validate_payload({"mode": "SSB"})           # no call


def test_validate_payload_required_must_be_str():
    assert not validate_payload({"call": None, "mode": "SSB"})


def test_validate_payload_rejects_unknown_keys():
    assert not validate_payload({"call": "HB9IKS", "mode": "SSB", "freq_mhz": "14"})
