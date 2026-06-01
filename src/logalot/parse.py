"""M3 — transcript → structured QSO via a *local* LLM. STUB.

Talks to LM Studio's OpenAI-compatible HTTP endpoint on localhost. The model
emits JSON only, conforming to ``models.QSO_LLM_SCHEMA``; we validate and reject
malformed output rather than trusting it. Candidate models: Apertus (ETH/EPFL),
Qwen2.5-Instruct. Everything stays on the machine.

Pipeline contract: phonetics are expanded and the callsign normalised in
``validate`` *after* this returns, before the candidate hits the queue. The LLM
is told to do its best but the regex is the backstop.
"""
from __future__ import annotations

import json
from typing import Protocol

from .models import QSO_LLM_SCHEMA, schema_json

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

SYSTEM_PROMPT = (
    "You extract amateur-radio QSO details from a noisy speech transcript. "
    "Respond with a single JSON object and nothing else — no prose, no Markdown "
    "fences. Expand NATO phonetics into letters for the callsign. If a field is "
    "absent, use null. Do NOT invent a callsign you cannot hear clearly; emit "
    "null and let a human fill it. The JSON must conform to this schema:\n\n"
    + schema_json()
)


class Parser(Protocol):
    def parse(self, transcript: str) -> dict | None: ...


def validate_payload(payload: dict) -> bool:
    """Minimal structural check against the schema's required keys and types.

    Intentionally dependency-free. M3 may swap in pydantic/jsonschema for full
    validation (it's in the ``parse`` extra)."""
    required = QSO_LLM_SCHEMA["required"]
    if not all(k in payload and isinstance(payload[k], str) for k in required):
        return False
    allowed = set(QSO_LLM_SCHEMA["properties"])
    return set(payload).issubset(allowed)


class LMStudioParser:
    """Calls a local LM Studio model. TODO(M3)."""

    def __init__(self, url: str = LM_STUDIO_URL, model: str = "local-model") -> None:
        self.url = url
        self.model = model

    def parse(self, transcript: str) -> dict | None:
        raise NotImplementedError(
            "M3: POST to LM Studio, parse JSON, run validate_payload(), "
            "return dict or None"
        )
