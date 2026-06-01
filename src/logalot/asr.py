"""M2 — automatic speech recognition. STUB.

faster-whisper (CTranslate2) with ``large-v3`` on the M4. Voice-activity
segmentation to chop the RX stream into utterances; each segment is transcribed
and handed to ``parse``. Expect poor copy on weak/fading signals — this is the
SNR-limited stage and no amount of cleverness downstream fully recovers it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class Segment:
    text: str
    start_s: float
    end_s: float
    avg_logprob: float | None = None   # may feed confidence in M3


class Transcriber(Protocol):
    def transcribe(self, audio: "object") -> list[Segment]: ...


class WhisperTranscriber:
    """Wraps faster-whisper. TODO(M2)."""

    def __init__(self, model: str = "large-v3", device: str = "auto",
                 language: str = "en") -> None:
        self.model = model
        self.device = device
        self.language = language

    def transcribe(self, audio) -> list[Segment]:
        raise NotImplementedError("M2: run faster-whisper with VAD segmentation")
