"""ADIF 3.1.4 export. Stdlib-only.

ADIF encodes each field as ``<NAME:length>value`` where length is the byte
length of the value; a record ends with ``<EOR>`` and the header with ``<EOH>``.
Empty/None fields are simply omitted. This is the interchange boundary into
MacLoggerDX and essentially every other logger.

Spec: https://adif.org/314/ADIF_314.htm
"""
from __future__ import annotations

import time
from collections.abc import Iterable

from .models import QSO

ADIF_VERSION = "3.1.4"
PROGRAM_ID = "LogALot"

# QSO attribute -> ADIF field name.
_FIELD_MAP: list[tuple[str, str]] = [
    ("call", "CALL"),
    ("qso_date", "QSO_DATE"),
    ("time_on", "TIME_ON"),
    ("band", "BAND"),
    ("freq_mhz", "FREQ"),
    ("mode", "MODE"),
    ("rst_sent", "RST_SENT"),
    ("rst_rcvd", "RST_RCVD"),
    ("name", "NAME"),
    ("qth", "QTH"),
    ("gridsquare", "GRIDSQUARE"),
    ("comment", "COMMENT"),
    ("my_call", "STATION_CALLSIGN"),
]


def _tag(name: str, value: object) -> str:
    s = str(value)
    return f"<{name}:{len(s.encode('utf-8'))}>{s}"


def record(q: QSO) -> str:
    parts = []
    for attr, adif_name in _FIELD_MAP:
        val = getattr(q, attr)
        if val is None or val == "":
            continue
        if attr == "freq_mhz":          # ADIF FREQ is MHz, decimal
            val = f"{float(val):.6f}".rstrip("0").rstrip(".")
        parts.append(_tag(adif_name, val))
    parts.append("<EOR>")
    return " ".join(parts)


def header() -> str:
    stamp = time.strftime("%Y%m%d %H%M%S", time.gmtime())
    return (
        f"LogALot ADIF export\n"
        f"{_tag('ADIF_VER', ADIF_VERSION)} "
        f"{_tag('PROGRAMID', PROGRAM_ID)} "
        f"{_tag('CREATED_TIMESTAMP', stamp)} "
        f"<EOH>"
    )


def to_adif(qsos: Iterable[QSO]) -> str:
    lines = [header()]
    lines.extend(record(q) for q in qsos)
    return "\n".join(lines) + "\n"


def write_adif(qsos: Iterable[QSO], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_adif(qsos))
