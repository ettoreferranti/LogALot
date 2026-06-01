"""Step C — live transcript feed: audio → VAD → ASR, streamed to the dashboard.

A background thread taps the RX audio stream, segments it with
:class:`logalot.asr.EnergyVAD`, transcribes each utterance, and publishes the
text to SSE subscribers plus a bounded history. Read-only and advisory — it
never writes the canonical log.

The transcriber is injected, so this whole pipeline is testable with a stub (no
MLX/whisper needed); in production it's :class:`logalot.asr.WhisperTranscriber`,
whose ``transcribe`` already funnels through the shared Metal thread.
"""
from __future__ import annotations

import collections
import datetime as dt
import queue
import threading
from dataclasses import dataclass

from .asr import EnergyVAD, resample_to_16k


@dataclass(slots=True)
class TranscriptEntry:
    utc: str
    text: str
    speaker: str               # "RX" = remote operator; TX segments are skipped
    dur_s: float
    avg_logprob: float | None = None


class TranscriptFeed:
    def __init__(self, audio, transcriber, cat=None, history: int = 200) -> None:
        self.audio = audio
        self.transcriber = transcriber
        self.cat = cat                 # for PTT: skip RX audio while we transmit
        self.entries: collections.deque[TranscriptEntry] = collections.deque(maxlen=history)
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="transcript-feed", daemon=True)
        self._thread.start()

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

    def _publish(self, entry: TranscriptEntry) -> None:
        with self._lock:
            self.entries.append(entry)
            for q in self._subs:
                q.put(entry)

    # --- worker ---------------------------------------------------------------

    def _run(self) -> None:
        tap = self.audio.tap()
        vad = EnergyVAD(self.audio.samplerate)
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
        self._publish(TranscriptEntry(
            utc=dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"),
            text=text,
            speaker="RX",
            dur_s=round(len(seg) / self.audio.samplerate, 1),
            avg_logprob=(sum(logps) / len(logps) if logps else None),
        ))

    def _transmitting(self) -> bool:
        if self.cat is None:
            return False
        try:
            return self.cat.ptt()
        except Exception:
            return False        # CAT hiccup: assume RX, don't drop audio
