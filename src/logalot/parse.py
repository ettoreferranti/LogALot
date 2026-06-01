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
    "You extract structured QSO (contact) details from a transcript of received "
    "amateur-radio speech. The transcript is often noisy, may mix languages, and "
    "may contain more than one station. Reply with a SINGLE JSON object and "
    "nothing else — no prose, no Markdown fences.\n"
    "\n"
    "HOW HAM CONTACTS SOUND\n"
    "- A calling station says 'CQ CQ CQ this is <callsign>' (CQ = a general call "
    "to anyone; 'CQ DX' = seeking distant stations). 'QRZ?' = 'who is calling "
    "me?'. CQ / CQDX / QRZ are NOT callsigns.\n"
    "- A reply addresses the other station, then gives its own: '<call> this is "
    "<call>'.\n"
    "- Callsigns are spelled in phonetics — standard NATO (Alpha, Bravo, Charlie, "
    "...) but operators often improvise words (e.g. 'Germany', 'Radio', 'Ocean'). "
    "Convert ALL phonetics to the letter/digit they stand for. 'niner'=9, "
    "'zero'/'null'=0.\n"
    "- RST signal report = two or three digits, spoken like 'five nine' (59) or "
    "'five nine nine' (599).\n"
    "- Name: 'my name is ...'. QTH = location: 'QTH is ...', 'I'm in ...'.\n"
    "\n"
    "CALLSIGN RULES (important)\n"
    "- A real amateur callsign is a country PREFIX + a digit + a 1-4 letter "
    "suffix, e.g. HB9IKS (Switzerland), DL1ABC (Germany), W1AW (USA), F5XYZ "
    "(France), I2ABC (Italy), G3ABC (England), EA4ABC (Spain), PA0ABC "
    "(Netherlands), RA3ABC (Russia), JA1ABC (Japan), VK2ABC (Australia). The "
    "prefix encodes the country — there are hundreds; these are only examples.\n"
    "- Return the callsign of the station whose transmission this is — the call "
    "it gives as its OWN ('this is X', 'X calling CQ', '73 from X'), NOT a call it "
    "is trying to reach or working.\n"
    "- Only output a COMPLETE call (prefix + digit + full suffix). A bare prefix "
    "like 'G4' or 'EA' with no suffix is NOT a callsign — set 'call' to null and "
    "wait until the whole call is heard. Do not pad or invent the suffix.\n"
    "- It MUST contain at least one digit and follow that shape. A '/' marks a "
    "portable add-on (HB9IKS/P, DL/HB9IKS).\n"
    "- NEVER return CQ, CQDX, QRZ, DX, a Q-code (QTH, QSL, QRM, QSY, ...), '73', a "
    "signal report, a name, or a plain word as the callsign. If you cannot hear a "
    "clear, well-formed worked-station callsign, set 'call' to null. Do NOT guess.\n"
    "\n"
    "COMMON Q-CODES (interpret them, never put them in 'call')\n"
    "QTH=location, QSL=confirm, QRZ=who's calling, QRM=interference, QSB=fading, "
    "QRP=low power, QSY=change frequency, QSO=a contact, 73=best regards.\n"
    "\n"
    "OUTPUT\n"
    "Only the fields in the schema below; use null for anything you did not "
    "clearly hear. Mode, frequency, band and time come from the radio, not from "
    "you. The JSON must conform to this schema:\n\n"
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
