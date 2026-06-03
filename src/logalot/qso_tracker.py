"""Stateful QSO tracker — turns a stream of per-over records into a persistent
list of QSO rows (the review queue), instead of one constantly-overwritten
candidate.

A QSO is a contact between two stations on a frequency. We hear one station at a
time (the one transmitting), so each *over* (one transmission) is attributed to a
speaker and, where stated, the station it addresses. Over time these overs are
assembled into QSO rows: a side we actually copy is ``heard``; a side we only
learn about because the heard station named it is present but ``heard=False``.

Coreference is heuristic (same band + overlapping callsigns within a time window;
a fresh CQ/new pair starts a new QSO). Rows are therefore often partial — correct
for a review queue, where the human is the final arbiter. Pure/stdlib: no MLX,
no audio; fully unit-testable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .dxcc import country_for_call

# Tokens that mean "no specific addressee" rather than a callsign.
_WILDCARD = {"CQ", "CQDX", "CQCQ", "DX", None, ""}


@dataclass
class Side:
    """One station in a QSO."""
    call: str | None = None
    name: str | None = None
    qth: str | None = None
    heard: bool = False          # did we actually copy this station transmitting?

    def to_dict(self) -> dict:
        return {
            "call": self.call, "name": self.name, "qth": self.qth,
            "heard": self.heard,
            "country": country_for_call(self.call) if self.call else None,
        }


@dataclass
class OverRecord:
    """What one over (transmission) told us. Built from the per-over parser plus
    the CAT snapshot taken while it was heard."""
    speaker: str | None                 # the transmitting station's own call
    addressed: str | None = None        # who they addressed (None / CQ = nobody specific)
    report: str | None = None           # RST the speaker gave the addressed station
    name: str | None = None             # speaker's name
    qth: str | None = None              # speaker's location
    text: str = ""
    utc: str = ""
    freq_mhz: float | None = None
    band: str | None = None
    mode: str | None = None


@dataclass
class Qso:
    id: int
    a: Side = field(default_factory=Side)
    b: Side = field(default_factory=Side)
    report_a_to_b: str | None = None
    report_b_to_a: str | None = None
    freq_mhz: float | None = None
    band: str | None = None
    mode: str | None = None
    started_utc: str = ""
    updated_utc: str = ""
    last_text: str = ""
    _t: float = field(default=0.0, repr=False)   # monotonic, for the merge window

    def calls(self) -> set[str]:
        return {s.call for s in (self.a, self.b) if s.call}

    def side_for(self, call: str) -> Side:
        """The Side that is (or should become) ``call``, filling an empty slot."""
        if self.a.call == call:
            return self.a
        if self.b.call == call:
            return self.b
        if self.a.call is None:
            self.a.call = call
            return self.a
        self.b.call = call
        return self.b

    def other_side(self, call: str) -> Side:
        return self.b if self.a.call == call else self.a

    def to_dict(self) -> dict:
        return {
            "kind": "qso", "id": self.id,
            "freq_mhz": self.freq_mhz, "band": self.band, "mode": self.mode,
            "started": self.started_utc, "updated": self.updated_utc,
            "a": self.a.to_dict(), "b": self.b.to_dict(),
            "report_a_to_b": self.report_a_to_b, "report_b_to_a": self.report_b_to_a,
            "last_text": self.last_text,
        }


class QsoTracker:
    def __init__(self, merge_window_s: float = 600.0) -> None:
        self.merge_window_s = merge_window_s
        self.qsos: list[Qso] = []
        self._next_id = 1

    def all(self) -> list[Qso]:
        return list(self.qsos)

    def counterpart(self, call: str) -> str | None:
        """The other station's call in the most recent QSO involving ``call`` —
        used to attribute an over that addresses ``call`` without self-ID (in a
        2-station QSO, that over is from ``call``'s partner)."""
        for qso in reversed(self.qsos):
            if call in qso.calls():
                other = qso.other_side(call)
                if other.call and other.call != call:
                    return other.call
        return None

    def ingest(self, over: OverRecord) -> Qso | None:
        """Fold one over into the QSO list; return the affected QSO (for the UI),
        or None if the over couldn't be attributed to a speaker."""
        speaker = over.speaker
        if not speaker:
            return None
        addressed = over.addressed if over.addressed not in _WILDCARD else None
        # A station can't work itself: when the parser splits one spoken callsign
        # into two near-identical reads (e.g. "India Radio Zero / Radio Charlie
        # Alpha Golf" both -> IR0CAG), from_call == to_call. Drop the self-address
        # so the counterpart slot stays empty instead of duplicating the speaker.
        if addressed == speaker:
            addressed = None
        now = time.monotonic()

        qso = self._match(speaker, addressed, over.band, now)
        if qso is None:
            qso = Qso(id=self._next_id, freq_mhz=over.freq_mhz, band=over.band,
                      mode=over.mode, started_utc=over.utc)
            self._next_id += 1
            self.qsos.append(qso)

        # The speaker is, by definition, heard.
        side = qso.side_for(speaker)
        side.heard = True
        if over.name:
            side.name = over.name
        if over.qth:
            side.qth = over.qth
        # Represent the addressed station on the other side (named-only for now).
        if addressed:
            other = qso.other_side(speaker)
            if other.call is None:
                other.call = addressed
        # Record the report the speaker gave.
        if over.report and addressed:
            self._set_report(qso, speaker, addressed, over.report)

        # CAT fields: fill if missing (first over wins for freq/band/mode).
        qso.freq_mhz = qso.freq_mhz if qso.freq_mhz is not None else over.freq_mhz
        qso.band = qso.band or over.band
        qso.mode = qso.mode or over.mode
        qso.updated_utc = over.utc or qso.updated_utc
        qso.last_text = over.text
        qso._t = now
        return qso

    # --- helpers --------------------------------------------------------------

    def _match(self, speaker: str, addressed: str | None, band: str | None,
               now: float) -> Qso | None:
        """Find the QSO this over belongs to, newest first, within the window."""
        for qso in reversed(self.qsos):
            if now - qso._t > self.merge_window_s:
                continue
            if band and qso.band and band != qso.band:
                continue
            calls = qso.calls()
            has_empty = self.a_or_b_empty(qso)
            if speaker in calls:
                # Same station again, or it's now naming/answering the other side.
                if addressed is None or addressed in calls or has_empty:
                    return qso
            elif addressed and addressed in calls and has_empty:
                # The speaker is the as-yet-unknown other party answering 'addressed'.
                return qso
        return None

    @staticmethod
    def a_or_b_empty(qso: Qso) -> bool:
        return qso.a.call is None or qso.b.call is None

    @staticmethod
    def _set_report(qso: Qso, giver: str, receiver: str, report: str) -> None:
        if qso.a.call == giver:
            qso.report_a_to_b = report
        elif qso.b.call == giver:
            qso.report_b_to_a = report
