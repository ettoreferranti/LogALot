"""Canonical data model for a QSO and the band plan helper.

Stdlib-only by design. UTC and SI throughout. Times/dates are stored as ADIF
strings (YYYYMMDD / HHMMSS) because that is the lingua franca for interchange and
avoids timezone ambiguity creeping into the record.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


# IARU Region 1 band edges in MHz (HF + 6 m / 2 m / 70 cm). Coarse, for labelling.
_BANDS_MHZ: list[tuple[str, float, float]] = [
    ("160m", 1.810, 2.000),
    ("80m", 3.500, 3.800),
    ("60m", 5.351, 5.367),
    ("40m", 7.000, 7.200),
    ("30m", 10.100, 10.150),
    ("20m", 14.000, 14.350),
    ("17m", 18.068, 18.168),
    ("15m", 21.000, 21.450),
    ("12m", 24.890, 24.990),
    ("10m", 28.000, 29.700),
    ("6m", 50.000, 54.000),
    ("2m", 144.000, 146.000),
    ("70cm", 430.000, 440.000),
]


def band_for_freq(freq_mhz: float | None) -> str | None:
    """Map a frequency in MHz to its IARU R1 band label, or None if out of plan."""
    if freq_mhz is None:
        return None
    for label, lo, hi in _BANDS_MHZ:
        if lo <= freq_mhz <= hi:
            return label
    return None


@dataclass(slots=True)
class QSO:
    """A single logged contact. Optional fields default to None and are simply
    omitted from ADIF output."""

    call: str
    qso_date: str          # UTC, YYYYMMDD
    time_on: str           # UTC, HHMMSS
    mode: str
    band: str | None = None
    freq_mhz: float | None = None
    rst_sent: str | None = None
    rst_rcvd: str | None = None
    name: str | None = None
    qth: str | None = None
    gridsquare: str | None = None
    comment: str | None = None
    my_call: str = "HB9IKS"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# JSON schema the local LLM (parse module) must emit. Kept here so model and
# parser cannot drift. Numbers the rig can supply (freq/band/time) are NOT asked
# of the LLM — they come from CAT.
QSO_LLM_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["call", "mode"],
    "properties": {
        "call": {"type": "string", "description": "Worked station callsign, normalised, no phonetics"},
        "mode": {"type": "string", "description": "e.g. SSB, CW, FT8, FM"},
        "rst_sent": {"type": ["string", "null"]},
        "rst_rcvd": {"type": ["string", "null"]},
        "name": {"type": ["string", "null"]},
        "qth": {"type": ["string", "null"]},
        "gridsquare": {"type": ["string", "null"]},
        "comment": {"type": ["string", "null"]},
    },
}


def schema_json() -> str:
    return json.dumps(QSO_LLM_SCHEMA, indent=2)
