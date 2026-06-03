"""M2 — automatic speech recognition, in-process with MLX.

``mlx-whisper`` on the M4 (Metal), weights pulled from the HuggingFace
``mlx-community`` hub on first use — no server. Voice-activity segmentation chops
the RX stream into utterances; each segment is transcribed and handed to
``parse``. Expect poor copy on weak/fading signals — this is the SNR-limited
stage and no amount of cleverness downstream fully recovers it.

Transcription runs on the shared Metal worker thread (see
:mod:`logalot.mlx_runtime`) so it never collides with the parse LLM. ``default
large-v3-turbo`` trades a little accuracy for the throughput we need near real
time; set ``LOGALOT_ASR_MODEL`` to ``mlx-community/whisper-large-v3-mlx`` for
maximum accuracy at lower speed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from .audio import rms_to_dbfs
from .mlx_runtime import DEFAULT_ASR_MODEL, run_blocking

# Seeds Whisper's decoder with ham-radio idiom so it stops mis-hearing the genre's
# core words — without it, "CQ" reliably becomes "secure"/"sicchio", phonetics
# drift, and signal reports garble. Whisper uses the prompt as prior-context for
# the first window (it applies even with condition_on_previous_text=False). Kept
# short and front-loaded with "CQ"; an over-long prompt over-biases and can
# induce hallucinations of these words on pure noise.
HAM_PROMPT = (
    "Amateur radio voice contact. Calling CQ: \"CQ CQ CQ\". NATO phonetics: "
    "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima "
    "Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey "
    "X-ray Yankee Zulu. Signal report five nine. QRZ QSL QTH RST 73 over."
)


@dataclass(slots=True)
class Segment:
    text: str
    start_s: float
    end_s: float
    avg_logprob: float | None = None   # may feed confidence in M3


class Transcriber(Protocol):
    def transcribe(self, audio: "object") -> list[Segment]: ...


def resample_to_16k(audio, src_sr: int, target_sr: int = 16000):
    """Resample mono float32 to Whisper's 16 kHz. The capture path is 48 kHz, an
    exact 3:1 ratio — averaging groups of 3 both low-passes and decimates in one
    step. Non-integer ratios fall back to linear interpolation."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if src_sr == target_sr or audio.size == 0:
        return audio
    ratio = src_sr / target_sr
    if ratio == int(ratio):
        f = int(ratio)
        trimmed = audio[: (audio.size // f) * f]
        if trimmed.size == 0:
            return np.zeros(0, dtype=np.float32)
        return trimmed.reshape(-1, f).mean(axis=1).astype(np.float32)
    n = int(round(audio.size * target_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class EnergyVAD:
    """Energy-gated streaming voice-activity segmenter.

    Feed audio blocks of any length via :meth:`push`; it returns completed
    utterance segments (mono float32 at the source samplerate) as speech ends.
    Crude by design — on weak SSB the "speech" is noisy — but enough to chop the
    RX stream into utterances for per-segment transcription (CLAUDE.md open Q3).
    A short pre-roll is prepended so word onsets aren't clipped, and a trailing
    silence "hang" closes the segment.
    """

    def __init__(self, samplerate: int, frame_ms: int = 30, threshold_dbfs: float = -45.0,
                 start_frames: int = 3, hang_ms: int = 600, min_speech_ms: int = 300,
                 max_segment_s: float = 20.0, preroll_ms: int = 200) -> None:
        import numpy as np

        self._np = np
        self.samplerate = samplerate
        self.frame = max(1, int(samplerate * frame_ms / 1000))
        self.threshold_dbfs = threshold_dbfs
        self.start_frames = start_frames
        self.hang_frames = max(1, int(hang_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.max_frames = max(1, int(max_segment_s * 1000 / frame_ms))
        self._acc = np.zeros(0, dtype=np.float32)
        self._pre: deque = deque(maxlen=max(1, int(preroll_ms / frame_ms)))
        self._seg: list = []
        self._active = False
        self._voiced_run = 0
        self._silence_run = 0
        self._speech_frames = 0

    def push(self, block) -> list:
        """Add audio; return any utterance segments completed by it."""
        np = self._np
        self._acc = np.concatenate([self._acc, np.asarray(block, dtype=np.float32)])
        out = []
        while self._acc.size >= self.frame:
            frame = self._acc[: self.frame]
            self._acc = self._acc[self.frame:]
            seg = self._frame(frame)
            if seg is not None:
                out.append(seg)
        return out

    def flush(self):
        """Close and return any in-progress utterance (call on stop). None if
        nothing worth keeping."""
        if self._active and self._speech_frames >= self.min_speech_frames:
            return self._close()
        self._reset()
        return None

    def _frame(self, frame):
        np = self._np
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        voiced = rms_to_dbfs(rms) > self.threshold_dbfs
        if not self._active:
            self._pre.append(frame)
            if voiced:
                self._voiced_run += 1
                if self._voiced_run >= self.start_frames:
                    self._active = True
                    self._seg = list(self._pre)
                    self._pre.clear()
                    self._speech_frames = self._voiced_run
                    self._silence_run = 0
            else:
                self._voiced_run = 0
            return None
        self._seg.append(frame)
        if voiced:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
        if self._silence_run >= self.hang_frames or len(self._seg) >= self.max_frames:
            return self._close()
        return None

    def _close(self):
        seg = self._np.concatenate(self._seg) if self._seg else None
        keep = seg is not None and self._speech_frames >= self.min_speech_frames
        self._reset()
        return seg if keep else None

    def _reset(self) -> None:
        self._seg = []
        self._active = False
        self._voiced_run = 0
        self._silence_run = 0
        self._speech_frames = 0


class WhisperTranscriber:
    """Wraps mlx-whisper. ``audio`` is a float32 mono ndarray at 16 kHz (what the
    capture ring buffer hands us after resampling the 48 kHz USB CODEC stream)."""

    def __init__(self, model_path: str = DEFAULT_ASR_MODEL, language: str | None = "en",
                 translate: bool = False, initial_prompt: str | None = HAM_PROMPT) -> None:
        self.model_path = model_path
        self.language = language
        # translate=True -> Whisper renders any language into English (task
        # "translate"); False -> transcribe in the spoken language.
        self.translate = translate
        # Domain vocabulary prior (see HAM_PROMPT); None disables it.
        self.initial_prompt = initial_prompt

    def transcribe(self, audio) -> list[Segment]:
        def _run() -> dict:
            import mlx_whisper  # lazy: needs the [asr] extra (Apple Silicon)

            # condition_on_previous_text=False stops Whisper from spiralling into
            # repetition loops ("pink pink pink…") when it hits noise/carriers.
            return mlx_whisper.transcribe(
                audio, path_or_hf_repo=self.model_path, language=self.language,
                task="translate" if self.translate else "transcribe",
                condition_on_previous_text=False,
                initial_prompt=self.initial_prompt,
            )

        result = run_blocking(_run)
        return [
            Segment(
                text=s["text"].strip(),
                start_s=s["start"],
                end_s=s["end"],
                avg_logprob=s.get("avg_logprob"),
            )
            for s in result.get("segments", [])
        ]
