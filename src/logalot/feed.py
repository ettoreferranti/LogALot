"""Step C/D — live transcript + advisory candidate feed.

A background thread taps the RX audio stream, segments it with
:class:`logalot.asr.EnergyVAD`, transcribes each utterance, and publishes the
text to SSE subscribers (Step C). A second debounced thread re-parses the
accumulated transcript of the current QSO into proposed fields and merges them
with the CAT-authoritative freq/band/mode/time to produce an *advisory*
candidate (Step D).

Everything here is read-only and advisory — it never writes the canonical log.
The human commits later (M4). Both the transcriber and the parser are injected,
so the whole pipeline is testable with stubs (no MLX needed).
"""
from __future__ import annotations

import collections
import datetime as dt
import queue
import threading
import time
from dataclasses import dataclass

from .asr import EnergyVAD, resample_to_16k
from .capture import utc_now_adif
from .validate import confidence_flag, normalise_call

# Optional LLM-extracted fields we carry from the parser onto the candidate.
_LLM_FIELDS = ("name", "qth", "rst_sent", "rst_rcvd", "gridsquare", "comment")


@dataclass(slots=True)
class TranscriptEntry:
    utc: str
    text: str
    speaker: str               # "RX" = remote operator; TX segments are skipped
    dur_s: float
    avg_logprob: float | None = None


class TranscriptFeed:
    def __init__(self, audio, transcriber, cat=None, parser=None, history: int = 200,
                 parse_debounce_s: float = 1.2, qso_idle_reset_s: float = 45.0,
                 vad_threshold_dbfs: float = -45.0) -> None:
        self.audio = audio
        self.transcriber = transcriber
        self.cat = cat                 # PTT (skip TX) + CAT snapshot for candidates
        self.parser = parser           # None -> no candidate panel
        self.parse_debounce_s = parse_debounce_s
        self.qso_idle_reset_s = qso_idle_reset_s
        # Speech gate: frames above this RMS count as voice. Raise it toward the
        # band noise floor (watch the dashboard's Audio RX dBFS) so SSB hiss
        # doesn't read as one endless utterance.
        self.vad_threshold_dbfs = vad_threshold_dbfs

        self.entries: collections.deque[TranscriptEntry] = collections.deque(maxlen=history)
        self.last_candidate: dict | None = None
        self._window: collections.deque[TranscriptEntry] = collections.deque(maxlen=16)
        self._win_version = 0
        self._last_utterance_at = 0.0

        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._threads = [threading.Thread(target=self._run, name="transcript-feed", daemon=True)]
        if self.parser is not None:
            self._threads.append(
                threading.Thread(target=self._parse_loop, name="candidate-parse", daemon=True))
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()

    # --- SSE pub/sub ----------------------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _publish(self, msg) -> None:
        with self._lock:
            for q in self._subs:
                q.put(msg)

    # --- transcript worker ----------------------------------------------------

    def _run(self) -> None:
        tap = self.audio.tap()
        vad = EnergyVAD(self.audio.samplerate, threshold_dbfs=self.vad_threshold_dbfs)
        while not self._stop.is_set():
            try:
                block = tap.get(timeout=0.2)
            except queue.Empty:
                continue
            for seg in vad.push(block):
                self._handle_segment(seg)

    def _handle_segment(self, seg) -> None:
        # While transmitting, the RX stream is our own sidetone/silence, not the
        # remote operator — drop it. (CAT PTT, not the audio, is the source.)
        if self._transmitting():
            return
        audio16k = resample_to_16k(seg, self.audio.samplerate)
        segments = self.transcriber.transcribe(audio16k)
        text = " ".join(s.text for s in segments).strip()
        if not text:
            return
        logps = [s.avg_logprob for s in segments if s.avg_logprob is not None]
        entry = TranscriptEntry(
            utc=dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"),
            text=text,
            speaker="RX",
            dur_s=round(len(seg) / self.audio.samplerate, 1),
            avg_logprob=(sum(logps) / len(logps) if logps else None),
        )
        with self._lock:
            self.entries.append(entry)
            # A long gap means a new QSO — start the parse window fresh.
            now = time.monotonic()
            if self._window and now - self._last_utterance_at > self.qso_idle_reset_s:
                self._window.clear()
            self._window.append(entry)
            self._last_utterance_at = now
            self._win_version += 1
        self._publish(entry)

    # --- candidate (parse) worker --------------------------------------------

    def _parse_loop(self) -> None:
        last_version = -1
        while not self._stop.wait(0.4):
            with self._lock:
                version = self._win_version
                idle = time.monotonic() - self._last_utterance_at
                text = " ".join(e.text for e in self._window)
            if not text or version == last_version:
                continue
            if idle < self.parse_debounce_s:   # let the operator finish speaking
                continue
            cand = self._build_candidate(text)
            last_version = version
            self.last_candidate = cand
            self._publish(cand)

    def _build_candidate(self, text: str) -> dict:
        try:
            fields = self.parser.parse(text) or {}
        except Exception:
            fields = {}
        cand: dict = {"kind": "candidate", "source_text": text}
        raw_call = fields.get("call")
        if raw_call:
            core, affixes = normalise_call(raw_call)
            call = core or raw_call
            cand["call"] = call
            cand["call_confidence"] = confidence_flag(call)
            if affixes:
                cand["affixes"] = affixes
        for k in _LLM_FIELDS:
            if fields.get(k):
                cand[k] = fields[k]
        cand.update(self._cat_fields())   # CAT is authoritative for freq/band/mode/time
        return cand

    def _cat_fields(self) -> dict:
        date, t = utc_now_adif()
        out: dict = {"qso_date": date, "time_on": t}
        if self.cat is not None and hasattr(self.cat, "state"):
            try:
                st = self.cat.state()
                out["freq_mhz"] = st.freq_mhz
                out["band"] = st.band
                out["mode"] = st.mode
            except Exception:
                pass
        return out

    def _transmitting(self) -> bool:
        if self.cat is None:
            return False
        try:
            return self.cat.ptt()
        except Exception:
            return False        # CAT hiccup: assume RX, don't drop audio
