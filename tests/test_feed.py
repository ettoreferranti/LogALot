"""End-to-end test of the transcript feed with a stub transcriber and fake audio
source — exercises the audio→VAD→ASR→publish path with no MLX/whisper."""
import queue

import pytest

np = pytest.importorskip("numpy")

from logalot.asr import Segment  # noqa: E402
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
