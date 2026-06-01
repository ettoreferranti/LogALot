"""End-to-end test of the transcript feed with a stub transcriber and fake audio
source — exercises the audio→VAD→ASR→publish path with no MLX/whisper."""
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


def test_is_repetitive_on_real_whisper_garbage():
    # Lines actually captured off-air during the first live test.
    assert is_repetitive("pink " * 100)
    assert is_repetitive("Delta " + "pink " * 80)
    assert is_repetitive(". . . . . .")
    assert is_repetitive("Brains pink pink pink Brains pink Brains pink Brains pink")


def test_is_repetitive_passes_normal_speech():
    assert not is_repetitive("made an NVIS aerial for 40 metres and it was enormous")
    assert not is_repetitive("pink pink")           # too short to judge
    assert not is_repetitive("CQ CQ this is Hotel Bravo Nine India Kilo Sierra")


def _drain_for_transcript(sub, timeout=1.5):
    """Return the first TranscriptEntry seen, or None if only non-entries arrive."""
    import time as _t
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        try:
            msg = sub.get(timeout=timeout)
        except queue.Empty:
            return None
        if not isinstance(msg, dict):   # TranscriptEntry, not a candidate
            return msg
    return None


def test_feed_drops_repetitive_transcript():
    audio = FakeAudio()
    tr = StubTranscriber("pink " * 100)
    feed = TranscriptFeed(audio, tr)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        assert _drain_for_transcript(sub) is None
    finally:
        feed.stop()


def test_feed_drops_low_logprob_transcript():
    audio = FakeAudio()
    tr = StubTranscriber("pink", avg_logprob=-3.8)
    feed = TranscriptFeed(audio, tr, min_logprob=-1.0)
    sub = feed.subscribe()
    feed.start()
    try:
        audio.feed_blocks(_utterance())
        assert _drain_for_transcript(sub) is None
    finally:
        feed.stop()


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


def test_feed_live_settings_are_mutable():
    audio = FakeAudio()
    tr = StubTranscriber()
    parser = StubParser({"call": "HB9IKS"})
    feed = TranscriptFeed(audio, tr, parser=parser)

    feed.set_vad_threshold(-28.0)
    feed.set_min_logprob(-0.7)
    feed.set_language("de")
    feed.set_parse_model("mlx-community/Qwen2.5-7B-Instruct-4bit")
    s = feed.settings()
    assert s["vad_threshold"] == -28.0
    assert s["min_logprob"] == -0.7
    assert s["language"] == "de"
    assert s["parse_model"].endswith("7B-Instruct-4bit")

    # "auto" maps to None on the transcriber but surfaces as "auto" in settings.
    feed.set_language("auto")
    assert feed.settings()["language"] == "auto"
    assert tr.language is None

    # translate toggle (default off) flips the transcriber + settings.
    assert feed.settings()["translate"] is False
    feed.set_translate(True)
    assert tr.translate is True
    assert feed.settings()["translate"] is True


def test_feed_vad_threshold_retunes_running_vad():
    audio = FakeAudio()
    feed = TranscriptFeed(audio, StubTranscriber())
    feed.start()
    try:
        for _ in range(40):                 # wait for _run to build the VAD
            if feed._vad is not None:
                break
            time.sleep(0.02)
        assert feed._vad is not None
        feed.set_vad_threshold(-25.0)
        assert feed._vad.threshold_dbfs == -25.0
    finally:
        feed.stop()


def test_recent_text_caps_to_recent_window():
    # The candidate must parse the recent exchange, not the whole ragchew.
    feed = TranscriptFeed(FakeAudio(), StubTranscriber(), parse_window_chars=40)
    for word in ["oldest filler text here", "middle filler text", "the newest over wins"]:
        feed._window.append(
            type("E", (), {"text": word})()  # lightweight entry-with-.text
        )
    text = feed._recent_text()
    assert "newest over wins" in text
    assert "oldest filler" not in text       # trimmed by the char budget
    assert len(text) <= 60


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
