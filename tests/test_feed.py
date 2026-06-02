"""End-to-end tests of the transcript + QSO-tracking feed, with a stub
transcriber and a stub per-over parser (no MLX/whisper/audio hardware)."""
import queue
import time

import pytest

np = pytest.importorskip("numpy")

from logalot.asr import Segment  # noqa: E402
from logalot.capture import RigState  # noqa: E402
from logalot.feed import TranscriptFeed, is_repetitive  # noqa: E402


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
    def __init__(self, text="hotel bravo nine", avg_logprob=-0.3):
        self.text = text
        self.avg_logprob = avg_logprob
        self.calls = 0

    def transcribe(self, audio16k):
        self.calls += 1
        return [Segment(text=self.text, start_s=0.0, end_s=1.0, avg_logprob=self.avg_logprob)]


class StubOverParser:
    """Returns canned per-over field dicts in sequence (one per closed over)."""

    def __init__(self, *over_fields, model_path="stub"):
        self.queue = list(over_fields)
        self.model_path = model_path
        self.calls = 0

    def parse_over(self, text):
        self.calls += 1
        return self.queue.pop(0) if self.queue else {}


class StubCat:
    def __init__(self, ptt=False):
        self._ptt = ptt

    def ptt(self):
        return self._ptt

    def state(self):
        return RigState(freq_mhz=14.2, mode="SSB", band="20m", raw_mode="USB")


def _utterance(sr=16000):
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    t = np.arange(int(1.0 * sr), dtype=np.float32) / sr
    speech = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return np.concatenate([silence, speech, silence, silence])


def _next_qso(sub, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = sub.get(timeout=timeout)
        except queue.Empty:
            return None
        if isinstance(msg, dict) and msg.get("kind") == "qso":
            return msg
    return None


# --- transcript path (unchanged behaviour) -----------------------------------

def test_feed_publishes_transcript_entry():
    audio = FakeAudio()
    tr = StubTranscriber("hotel bravo nine india kilo sierra")
    feed = TranscriptFeed(audio, tr)            # no parser -> transcript only
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        entry = sub.get(timeout=3.0)
    finally:
        feed.stop()
    assert entry.text == "hotel bravo nine india kilo sierra"
    assert entry.speaker == "RX"


def test_feed_skips_segments_while_transmitting():
    audio = FakeAudio()
    tr = StubTranscriber()
    feed = TranscriptFeed(audio, tr, cat=StubCat(ptt=True))
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        with pytest.raises(queue.Empty):
            sub.get(timeout=1.0)
    finally:
        feed.stop()
    assert tr.calls == 0


# --- QSO tracking path -------------------------------------------------------

def test_feed_builds_qso_from_over():
    audio = FakeAudio()
    parser = StubOverParser({"from_call": "EA1ABC", "to_call": "CQ", "name": "Jose"})
    feed = TranscriptFeed(audio, StubTranscriber(), cat=StubCat(), parser=parser,
                          over_gap_s=0.0)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        qso = _next_qso(sub)
    finally:
        feed.stop()
    assert qso is not None
    assert qso["a"]["call"] == "EA1ABC" and qso["a"]["heard"]
    assert qso["a"]["name"] == "Jose"
    assert qso["a"]["country"] == "Spain"
    assert qso["band"] == "20m" and qso["freq_mhz"] == 14.2


def test_feed_cleans_phonetic_speaker():
    audio = FakeAudio()
    parser = StubOverParser({"from_call": "Hotel Bravo Nine India Kilo Sierra", "to_call": "CQ"})
    feed = TranscriptFeed(audio, StubTranscriber(), cat=StubCat(), parser=parser,
                          over_gap_s=0.0)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        qso = _next_qso(sub)
    finally:
        feed.stop()
    assert qso is not None
    assert qso["a"]["call"] == "HB9IKS"          # phonetics expanded


def test_feed_drops_over_without_identifiable_speaker():
    audio = FakeAudio()
    parser = StubOverParser({"from_call": None, "to_call": "CQ"})  # no call heard
    feed = TranscriptFeed(audio, StubTranscriber(), cat=StubCat(), parser=parser,
                          over_gap_s=0.0)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        assert _next_qso(sub, timeout=1.5) is None
    finally:
        feed.stop()
    assert feed.tracker.all() == []


def test_feed_live_model_and_translate_setters():
    audio = FakeAudio()
    tr = StubTranscriber()
    tr.model_path = "mlx-community/whisper-large-v3-turbo"
    feed = TranscriptFeed(audio, tr, parser=StubOverParser())
    feed.set_translate(True)
    feed.set_asr_model("mlx-community/whisper-large-v3-mlx")
    s = feed.settings()
    assert s["translate"] is True
    assert s["asr_model"].endswith("whisper-large-v3-mlx")
    assert tr.translate is True


class FakeEnhancer:
    def __init__(self):
        self.calls = 0

    def enhance(self, audio):
        self.calls += 1
        return audio


def test_feed_applies_enhancer_when_on():
    audio = FakeAudio()
    enh = FakeEnhancer()
    feed = TranscriptFeed(audio, StubTranscriber(), enhancer=enh, enhance=True)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        sub.get(timeout=3.0)
    finally:
        feed.stop()
    assert enh.calls >= 1
    s = feed.settings()
    assert s["enhance"] is True and s["enhance_available"] is True


def test_feed_skips_enhancer_when_off_and_toggles_live():
    audio = FakeAudio()
    enh = FakeEnhancer()
    feed = TranscriptFeed(audio, StubTranscriber(), enhancer=enh, enhance=False)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        sub.get(timeout=3.0)
    finally:
        feed.stop()
    assert enh.calls == 0
    feed.set_enhance(True)
    assert feed.settings()["enhance"] is True


def test_feed_enhance_unavailable_without_enhancer():
    feed = TranscriptFeed(FakeAudio(), StubTranscriber(), enhancer=None, enhance=True)
    s = feed.settings()
    assert s["enhance_available"] is False
    assert s["enhance"] is False        # can't enable without an enhancer


def test_is_repetitive_on_real_whisper_garbage():
    assert is_repetitive("pink " * 100)
    assert is_repetitive(". . . . . .")
    assert not is_repetitive("made an NVIS aerial for 40 metres and it was enormous")
    assert not is_repetitive("CQ CQ this is Hotel Bravo Nine India Kilo Sierra")
