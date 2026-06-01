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

from dataclasses import dataclass
from typing import Protocol

from .mlx_runtime import DEFAULT_ASR_MODEL, run_blocking


@dataclass(slots=True)
class Segment:
    text: str
    start_s: float
    end_s: float
    avg_logprob: float | None = None   # may feed confidence in M3


class Transcriber(Protocol):
    def transcribe(self, audio: "object") -> list[Segment]: ...


class WhisperTranscriber:
    """Wraps mlx-whisper. ``audio`` is a float32 mono ndarray at 16 kHz (what the
    capture ring buffer hands us after resampling the 48 kHz USB CODEC stream)."""

    def __init__(self, model_path: str = DEFAULT_ASR_MODEL, language: str = "en") -> None:
        self.model_path = model_path
        self.language = language

    def transcribe(self, audio) -> list[Segment]:
        def _run() -> dict:
            import mlx_whisper  # lazy: needs the [asr] extra (Apple Silicon)

            return mlx_whisper.transcribe(
                audio, path_or_hf_repo=self.model_path, language=self.language
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
