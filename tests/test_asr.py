"""Tests for the ASR segmenter/resampler — pure numpy, no MLX/whisper needed."""
import pytest

np = pytest.importorskip("numpy")

from logalot.asr import EnergyVAD, resample_to_16k  # noqa: E402


def test_resample_48k_to_16k_length_and_dtype():
    audio = np.zeros(4800, dtype=np.float32)        # 0.1 s @ 48 kHz
    out = resample_to_16k(audio, 48000)
    assert out.dtype == np.float32
    assert len(out) == 1600                          # exact 3:1 decimation


def test_resample_noop_when_already_16k():
    audio = np.ones(160, dtype=np.float32)
    out = resample_to_16k(audio, 16000)
    assert np.array_equal(out, audio)


def test_resample_preserves_a_tone_roughly():
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = resample_to_16k(tone, sr)
    # Amplitude/energy should survive downsampling (440 Hz is well below 8 kHz).
    assert out.max() > 0.3
    assert abs(float(np.sqrt(np.mean(out**2))) - 0.354) < 0.05  # RMS of a 0.5 sine


def _frames(vad, audio, block=480):
    """Push audio through the VAD in fixed blocks; return completed segments."""
    out = []
    for i in range(0, len(audio), block):
        out += vad.push(audio[i:i + block])
    return out


def test_vad_emits_one_segment_for_one_utterance():
    sr = 16000
    vad = EnergyVAD(sr, threshold_dbfs=-45)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    t = np.arange(int(1.0 * sr), dtype=np.float32) / sr
    speech = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # ~ -10 dBFS
    audio = np.concatenate([silence, speech, silence, silence])      # trailing hang
    segs = _frames(vad, audio)
    assert len(segs) == 1
    # Segment ~ speech length plus pre-roll, well under speech+hang.
    assert 0.9 * sr < len(segs[0]) < 2.0 * sr


def test_vad_ignores_pure_silence():
    sr = 16000
    vad = EnergyVAD(sr)
    assert _frames(vad, np.zeros(2 * sr, dtype=np.float32)) == []


def test_vad_two_utterances_split_by_silence():
    sr = 16000
    vad = EnergyVAD(sr, threshold_dbfs=-45)
    t = np.arange(int(0.8 * sr), dtype=np.float32) / sr
    tone = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    gap = np.zeros(int(1.0 * sr), dtype=np.float32)
    audio = np.concatenate([gap, tone, gap, tone, gap])
    segs = _frames(vad, audio)
    assert len(segs) == 2
