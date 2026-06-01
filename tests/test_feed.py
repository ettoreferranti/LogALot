"""End-to-end test of the transcript feed with a stub transcriber and fake audio
source — exercises the audio→VAD→ASR→publish path with no MLX/whisper."""
import queue
import time

import pytest

np = pytest.importorskip("numpy")

from logalot.asr import Segment  # noqa: E402
from logalot.capture import RigState  # noqa: E402
from logalot.feed import TranscriptFeed  # noqa: E402


class FakeAudio:
    """Minimal AudioCapture stand-in: a tap queue the test pushes blocks into."""

    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self._q: queue.Queue = queue.Queue()

    def tap(self):
        return self._q

    def feed_blocks(self, audio, block=480):
        for i in range(0, len(audio), block):
            self._q.put(audio[i:i + block])


class StubTranscriber:
    def __init__(self, text="hotel bravo nine"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio16k):
        self.calls += 1
        return [Segment(text=self.text, start_s=0.0, end_s=1.0, avg_logprob=-0.3)]


class StubCat:
    def __init__(self, ptt):
        self._ptt = ptt

    def ptt(self):
        return self._ptt


class StubParser:
    def __init__(self, fields):
        self.fields = fields
        self.calls = 0

    def parse(self, text):
        self.calls += 1
        return dict(self.fields)


class StubCatFull(StubCat):
    """CAT stub that also answers state() for candidate building."""

    def state(self):
        return RigState(freq_mhz=14.074, mode="SSB", band="20m", raw_mode="USB")


def _utterance(sr=16000):
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    t = np.arange(int(1.0 * sr), dtype=np.float32) / sr
    speech = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return np.concatenate([silence, speech, silence, silence])


def test_feed_publishes_transcript_entry():
    audio = FakeAudio()
    tr = StubTranscriber("hotel bravo nine india kilo sierra")
    feed = TranscriptFeed(audio, tr)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        entry = sub.get(timeout=3.0)
    finally:
        feed.stop()
    assert entry.text == "hotel bravo nine india kilo sierra"
    assert entry.speaker == "RX"
    assert entry.dur_s > 0
    assert entry.avg_logprob == pytest.approx(-0.3)
    assert tr.calls == 1
    assert list(feed.entries)[-1] is entry


def test_feed_skips_segments_while_transmitting():
    audio = FakeAudio()
    tr = StubTranscriber()
    feed = TranscriptFeed(audio, tr, cat=StubCat(ptt=True))
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        with pytest.raises(queue.Empty):
            sub.get(timeout=1.5)       # PTT on -> nothing transcribed/published
    finally:
        feed.stop()
    assert tr.calls == 0


def test_feed_builds_advisory_candidate():
    audio = FakeAudio()
    tr = StubTranscriber("hotel bravo nine india kilo sierra")
    parser = StubParser({"call": "HB9IKS", "name": "Tom", "rst_rcvd": "59"})
    feed = TranscriptFeed(audio, tr, cat=StubCatFull(ptt=False), parser=parser,
                          parse_debounce_s=0.0)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        cand = None
        deadline = time.time() + 5
        while time.time() < deadline:
            msg = sub.get(timeout=5)
            if isinstance(msg, dict) and msg.get("kind") == "candidate":
                cand = msg
                break
    finally:
        feed.stop()
    assert cand is not None
    assert cand["call"] == "HB9IKS"
    assert cand["call_confidence"] == "ok"
    assert cand["name"] == "Tom"
    # CAT is authoritative for these — they come from state(), not the LLM.
    assert cand["mode"] == "SSB"
    assert cand["band"] == "20m"
    assert cand["freq_mhz"] == 14.074
    assert "qso_date" in cand and "time_on" in cand
    assert parser.calls >= 1


def test_feed_no_parser_means_no_candidate():
    audio = FakeAudio()
    feed = TranscriptFeed(audio, StubTranscriber(), parser=None)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        # Only transcript entries should ever arrive, never a candidate dict.
        for _ in range(2):
            try:
                msg = sub.get(timeout=1.5)
            except queue.Empty:
                break
            assert not (isinstance(msg, dict) and msg.get("kind") == "candidate")
    finally:
        feed.stop()
    assert feed.last_candidate is None
