"""M3 — transcript → structured QSO via a *local* LLM, in-process with MLX.

No LM Studio, no Ollama, no server: the model is loaded inside this process via
``mlx-lm`` (see :mod:`logalot.mlx_runtime`), weights pulled from the HuggingFace
``mlx-community`` hub on first use. Everything stays on the machine.

The model emits JSON only, conforming to ``models.QSO_LLM_SCHEMA``; we extract,
validate, and *reject* malformed output rather than trusting it — a hallucinated
field must never reach the record. Phonetics expansion and the callsign regex run
in ``validate`` *after* this returns; the LLM is told to do its best and the regex
is the backstop.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .models import QSO_LLM_SCHEMA, schema_json
from .mlx_runtime import DEFAULT_PARSE_MODEL, get_llm

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

    Intentionally dependency-free so it can guard output regardless of backend.
    """
    required = QSO_LLM_SCHEMA["required"]
    if not all(k in payload and isinstance(payload[k], str) for k in required):
        return False
    allowed = set(QSO_LLM_SCHEMA["properties"])
    return set(payload).issubset(allowed)


def extract_json(text: str) -> dict | None:
    """Pull the first balanced ``{...}`` object out of model output and parse it.

    Tolerates a stray code fence or surrounding prose even though we ask for raw
    JSON — small local models slip occasionally. Returns None if nothing parses.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                return obj if isinstance(obj, dict) else None
    return None


class MLXParser:
    """In-process mlx-lm parser. The model loads lazily on the first
    :meth:`parse` call (multi-GB, multi-second the first time), then is cached
    for the process lifetime by :mod:`logalot.mlx_runtime`."""

    def __init__(self, model_path: str = DEFAULT_PARSE_MODEL, max_tokens: int = 256) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens

    def parse(self, transcript: str) -> dict | None:
        """Transcript → validated QSO-field dict, or None if the model produced
        nothing schema-valid (caller leaves it for the human)."""
        runtime = get_llm(self.model_path, max_tokens=self.max_tokens)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        text = runtime.generate(messages)
        payload = extract_json(text)
        if payload is None or not validate_payload(payload):
            return None
        # Drop explicit nulls so downstream sees only present fields.
        return {k: v for k, v in payload.items() if v is not None}
