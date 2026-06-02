"""Live transcript + QSO tracking feed.

A background thread taps the RX audio stream, segments it with
:class:`logalot.asr.EnergyVAD`, transcribes each utterance, and publishes the
text to SSE subscribers (the transcript panel).

A second thread assembles consecutive transcript into *overs* (one station's
transmission, delimited by a silence gap), runs a per-over LLM extraction
(speaker / addressed / report / name / qth), and folds each over into the
stateful :class:`logalot.qso_tracker.QsoTracker` — a persistent list of QSO rows
rather than one overwritten candidate. QSO updates are published as ``kind:qso``.

Everything is read-only/advisory — it never writes the canonical log; the human
commits later (M4). The transcriber and parser are injected, so the whole
pipeline is testable with stubs (no MLX needed).
"""
from __future__ import annotations

import collections
import datetime as dt
import queue
import re
import threading
import time
from dataclasses import dataclass

from .asr import EnergyVAD, resample_to_16k
from .qso_tracker import OverRecord, QsoTracker
from .validate import expand_phonetics, looks_like_callsign, normalise_call

_ADDRESSED_WILDCARDS = {"CQ", "CQCQ", "CQDX", "DX"}

# "<to> this is <from>" — small LLMs keep inverting this, but it's a fixed
# positional rule (the call after 'this is' is the transmitter), so we resolve it
# deterministically from the text and only fall back to the LLM when absent.
_THIS_IS_RE = re.compile(r"\b(?:this is|hier ist|here is)\b", re.I)
_CQ_RE = re.compile(r"\bcq\b", re.I)


def positional_from_to(text: str) -> tuple[str | None, str | None]:
    """Resolve (from_call, to_call) from the '<to> this is <from>' structure.
    Either may be None; to_call may be 'CQ'. Phonetic words around 'this is' are
    expanded and validated; over-grabbing is bounded to ~6 words each side."""
    m = _THIS_IS_RE.search(text)
    if not m:
        return None, None
    before = " ".join(text[:m.start()].split()[-6:])
    after = " ".join(text[m.end():].split()[:6])
    frm = expand_phonetics(after)
    frm = frm if looks_like_callsign(frm) else None
    if _CQ_RE.search(before):
        to = "CQ"
    else:
        to = expand_phonetics(before)
        to = to if looks_like_callsign(to) else None
    return frm, to


def is_repetitive(text: str) -> bool:
    """True if one token dominates the text — Whisper's repetition hallucination
    on noise/carriers ('pink pink pink…', '. . . .'). Short segments are exempt
    (a genuine 'pink pink' could be real)."""
    words = text.lower().split()
    if len(words) < 6:
        return False
    most = max(collections.Counter(words).values())
    return most / len(words) > 0.5


@dataclass(slots=True)
class TranscriptEntry:
    utc: str
    text: str
    speaker: str               # "RX" = remote operator; TX segments are skipped
    dur_s: float
    avg_logprob: float | None = None


class TranscriptFeed:
    def __init__(self, audio, transcriber, cat=None, parser=None, history: int = 200,
                 vad_threshold_dbfs: float = -45.0, min_logprob: float = -1.0,
                 over_gap_s: float = 2.5) -> None:
        self.audio = audio
        self.transcriber = transcriber
        self.cat = cat                 # PTT (skip TX) + CAT snapshot for QSO rows
        self.parser = parser           # None -> transcript only, no QSO table
        # A silence gap longer than this closes the current over (one transmission).
        self.over_gap_s = over_gap_s
        # Speech gate: frames above this RMS count as voice. Raise it toward the
        # band noise floor (watch the dashboard's Audio RX dBFS) so SSB hiss
        # doesn't read as one endless utterance.
        self.vad_threshold_dbfs = vad_threshold_dbfs
        # Drop transcripts below this mean token logprob — Whisper's low-confidence
        # noise garbage (the very negative-logprob lines).
        self.min_logprob = min_logprob

        self.tracker = QsoTracker()
        self.entries: collections.deque[TranscriptEntry] = collections.deque(maxlen=history)
        self._over: list[TranscriptEntry] = []   # entries of the current open over
        self._over_last_at = 0.0

        self._vad = None               # set in _run; kept so settings can retune it live
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- live settings (mutated from the dashboard, no restart) ---------------

    def settings(self) -> dict:
        lang = getattr(self.transcriber, "language", None)
        return {
            "vad_threshold": self.vad_threshold_dbfs,
            "min_logprob": self.min_logprob,
            "language": lang if lang else "auto",
            "translate": bool(getattr(self.transcriber, "translate", False)),
            "asr_model": getattr(self.transcriber, "model_path", None),
            "parse_model": getattr(self.parser, "model_path", None),
        }

    def set_vad_threshold(self, dbfs: float) -> None:
        self.vad_threshold_dbfs = dbfs
        if self._vad is not None:
            self._vad.threshold_dbfs = dbfs

    def set_min_logprob(self, value: float) -> None:
        self.min_logprob = value

    def set_language(self, lang: str | None) -> None:
        self.transcriber.language = None if (not lang or lang == "auto") else lang

    def set_translate(self, on: bool) -> None:
        self.transcriber.translate = bool(on)

    def set_asr_model(self, model_path: str) -> None:
        if model_path and model_path != getattr(self.transcriber, "model_path", None):
            self.transcriber.model_path = model_path   # loads on next utterance
            print(f"asr: model set to {model_path} (loads on next utterance)", flush=True)

    def set_parse_model(self, model_path: str) -> None:
        if self.parser is not None and model_path and \
                model_path != getattr(self.parser, "model_path", None):
            self.parser.model_path = model_path   # loads lazily on next parse
            print(f"parse: model set to {model_path} (loads on next parse)", flush=True)

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._threads = [threading.Thread(target=self._run, name="transcript-feed", daemon=True)]
        if self.parser is not None:
            self._threads.append(
                threading.Thread(target=self._over_loop, name="qso-tracker", daemon=True))
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
        vad = self._vad = EnergyVAD(self.audio.samplerate, threshold_dbfs=self.vad_threshold_dbfs)
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
        if not text or is_repetitive(text):
            return
        logps = [s.avg_logprob for s in segments if s.avg_logprob is not None]
        avg_logprob = sum(logps) / len(logps) if logps else None
        if avg_logprob is not None and avg_logprob < self.min_logprob:
            return        # low-confidence noise garbage
        entry = TranscriptEntry(
            utc=dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"),
            text=text,
            speaker="RX",
            dur_s=round(len(seg) / self.audio.samplerate, 1),
            avg_logprob=avg_logprob,
        )
        with self._lock:
            self.entries.append(entry)
            self._over.append(entry)          # accumulate into the open over
            self._over_last_at = time.monotonic()
        self._publish(entry)

    # --- over assembly / QSO tracking worker ----------------------------------

    def _over_loop(self) -> None:
        while not self._stop.wait(0.4):
            with self._lock:
                if not self._over or time.monotonic() - self._over_last_at < self.over_gap_s:
                    continue
                over_entries = self._over
                self._over = []
            self._process_over(over_entries)

    def _process_over(self, entries: list[TranscriptEntry]) -> None:
        text = " ".join(e.text for e in entries).strip()
        if not text:
            return
        try:
            fields = self.parser.parse_over(text)
        except Exception:
            fields = {}

        # Structure (who is from/to) is resolved from the text first — it's a
        # fixed 'this is' rule the LLM keeps inverting; the LLM fills the gaps.
        pos_from, pos_to = positional_from_to(text)
        speaker = pos_from or self._clean_call(fields.get("from_call"))
        addressed = pos_to or self._clean_addressed(fields.get("to_call"))
        # Turn-based attribution: an over that addresses X without self-ID is, in a
        # 2-station QSO, from X's partner. (We don't guess from a bare continuation
        # over — turns alternate, so the "last speaker" is unreliable.)
        if speaker is None and addressed and addressed != "CQ":
            speaker = self.tracker.counterpart(addressed)
        if speaker is None:
            return                              # nothing we can attribute

        cat = self._cat_snapshot()
        over = OverRecord(
            speaker=speaker,
            addressed=addressed,
            report=fields.get("report"),
            name=fields.get("name"),
            qth=fields.get("qth"),
            text=text, utc=cat["utc"],
            freq_mhz=cat["freq_mhz"], band=cat["band"], mode=cat["mode"],
        )
        qso = self.tracker.ingest(over)
        if qso is not None:
            self._publish(qso.to_dict())

    # --- helpers --------------------------------------------------------------

    def _clean_call(self, raw) -> str | None:
        """A raw call (possibly phonetic words) -> a complete valid callsign, or
        None. Reuses the validate/phonetics backstop."""
        if not raw:
            return None
        core, _ = normalise_call(raw)
        if looks_like_callsign(core):
            return core.upper()
        expanded = expand_phonetics(raw)
        if looks_like_callsign(expanded):
            return expanded
        return None

    def _clean_addressed(self, raw) -> str | None:
        if not raw:
            return None
        if raw.strip().upper() in _ADDRESSED_WILDCARDS:
            return "CQ"
        return self._clean_call(raw)

    def _cat_snapshot(self) -> dict:
        utc = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
        out = {"utc": utc, "freq_mhz": None, "band": None, "mode": None}
        if self.cat is not None and hasattr(self.cat, "state"):
            try:
                st = self.cat.state()
                out["freq_mhz"], out["band"], out["mode"] = st.freq_mhz, st.band, st.mode
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
