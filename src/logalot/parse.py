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

# Shared callsign/phonetics knowledge, reused by the single-shot and per-over
# prompts so they cannot drift.
_HAM_KNOWLEDGE = (
    "HOW HAM CONTACTS SOUND\n"
    "- A calling station says 'CQ CQ CQ this is <callsign>' (CQ = a general call "
    "to anyone). 'QRZ?' = 'who is calling me?'. CQ / CQDX / QRZ are NOT callsigns.\n"
    "- A reply addresses the other station, then gives its own: '<call> this is "
    "<call>'.\n"
    "- Callsigns are spelled in phonetics — NATO (Alpha, Bravo, Charlie, ...) but "
    "operators improvise (e.g. 'Germany', 'Radio', 'Ocean'). Convert ALL phonetics "
    "to the letter/digit they stand for. 'niner'=9, 'zero'/'null'=0.\n"
    "- RST report = two/three digits, spoken 'five nine' (59) or 'five nine nine' "
    "(599). Name: 'my name is ...'. QTH = location: 'QTH is ...', 'I'm in ...'.\n"
    "\n"
    "CALLSIGN RULES\n"
    "- A real callsign is a country PREFIX + a digit + a 1-4 letter suffix (HB9IKS, "
    "DL1ABC, W1AW, F5XYZ, I2ABC, G3ABC, EA4ABC, RA3ABC, JA1ABC). It MUST contain a "
    "digit. A bare prefix like 'G4' (no suffix) is NOT a complete callsign.\n"
    "- NEVER return CQ, CQDX, QRZ, DX, a Q-code (QTH, QSL, QRM, ...), '73', a "
    "report, or a name as a callsign. If you cannot hear a clear, complete call, "
    "use null. Do NOT guess.\n"
)

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


# --- per-over extraction (the QSO tracker's input) ---------------------------

# One "over" = one station's transmission. We extract who is speaking (their own
# call), who they address, and any facts they volunteer. Only "speaker" matters
# for attribution; everything else is best-effort.
QSO_OVER_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["from_call"],
    "properties": {
        "from_call": {"type": ["string", "null"],
                      "description": "the TRANSMITTING station's own callsign (the call AFTER 'this is'/'DE'), or null if not stated in this over"},
        "to_call": {"type": ["string", "null"],
                    "description": "the callsign being called/worked (the call BEFORE 'this is'), or 'CQ', or null"},
        "report": {"type": ["string", "null"],
                   "description": "RST the transmitting station gave the other, e.g. 59 / 599"},
        "name": {"type": ["string", "null"], "description": "the transmitting station's name"},
        "qth": {"type": ["string", "null"], "description": "the transmitting station's location"},
    },
}


def over_schema_json() -> str:
    return json.dumps(QSO_OVER_SCHEMA, indent=2)


OVER_PROMPT = (
    "You are listening to ONE over (a single transmission by ONE amateur-radio "
    "station). The text is a possibly-noisy, possibly-non-English transcript and "
    "may include off-topic chat (weather, equipment). Reply with a SINGLE JSON "
    "object and nothing else.\n"
    "\n"
    + _HAM_KNOWLEDGE +
    "\n"
    "FROM vs TO (critical — do not get this backwards):\n"
    "- The phrase order is '<TO-call> this is <FROM-call>' (also '<TO> DE "
    "<FROM>'). The FROM-call is the station TRANSMITTING; the TO-call is who it is "
    "calling. In 'HB9IKS this is DL1ABC', from_call=DL1ABC and to_call=HB9IKS.\n"
    "- A CQ call ('CQ CQ CQ this is X', even with several CQs) is still the FROM "
    "station X calling: from_call=X, to_call='CQ'. The call after 'this is' is "
    "always the FROM station. '73 from X' / 'X here' -> from_call X.\n"
    "- If the over addresses a station but never gives the TRANSMITTER'S OWN call "
    "(e.g. 'G3XYZ you are 59'), set from_call to null — do NOT reuse the addressed "
    "call. report, name and qth describe the FROM (transmitting) station.\n"
    "\n"
    "EXAMPLES (input -> output):\n"
    "'CQ CQ CQ this is Delta Lima One Alpha Bravo Charlie standing by' -> "
    "{\"from_call\":\"DL1ABC\",\"to_call\":\"CQ\"}\n"
    "'Golf Three Xray Yankee Zulu this is Whiskey One Alpha Whiskey, good evening' "
    "-> {\"from_call\":\"W1AW\",\"to_call\":\"G3XYZ\"}\n"
    "'Whiskey One Alpha Whiskey you are five nine, my name is Tom, QTH Bern' -> "
    "{\"from_call\":null,\"to_call\":\"W1AW\",\"report\":\"59\",\"name\":\"Tom\",\"qth\":\"Bern\"}\n"
    "\n"
    "Now extract for THIS over. Ignore weather/equipment chatter — it is not a "
    "field. The JSON must conform to this schema:\n\n"
    + over_schema_json()
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


def _loads_lenient(s: str) -> dict | None:
    """json.loads, but tolerant of the trailing commas small models love to emit
    (``{"call":"X",}``)."""
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",(\s*[}\]])", r"\1", s))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


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
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                obj = _loads_lenient(text[start : i + 1])
                if obj is not None:
                    return obj
                start = -1
    return None


# Prime the assistant reply with the start of the JSON object. The model can
# only continue it, so it cannot wander into prose first — the single biggest
# reliability win without a grammar engine (outlines_core has no py3.14 wheel).
_JSON_PREFIX = '{"call":'


class MLXParser:
    """In-process mlx-lm parser. The model loads lazily on the first
    :meth:`parse` call (multi-GB, multi-second the first time), then is cached
    for the process lifetime by :mod:`logalot.mlx_runtime`.

    ``runtime`` may be injected for testing; otherwise the cached MLX runtime for
    ``model_path`` is used (so a live model swap is picked up)."""

    def __init__(self, model_path: str = DEFAULT_PARSE_MODEL, max_tokens: int = 256,
                 runtime=None) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens
        self._runtime = runtime

    def _runtime_for(self):
        return self._runtime or get_llm(self.model_path, max_tokens=self.max_tokens)

    def parse(self, transcript: str) -> dict | None:
        """Transcript → validated QSO-field dict, or None if the model produced
        nothing schema-valid (caller leaves it for the human)."""
        runtime = self._runtime_for()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        payload = extract_json(runtime.generate(messages, prefix=_JSON_PREFIX))
        if payload is None:
            # Greedy decoding is deterministic, so a retry must change the input:
            # add a blunt JSON-only reminder.
            messages = messages + [{"role": "user", "content":
                "Reply with ONLY one JSON object for the schema, nothing else."}]
            payload = extract_json(runtime.generate(messages, prefix=_JSON_PREFIX))
        if payload is None or not validate_payload(payload):
            return None
        # Drop explicit nulls so downstream sees only present fields.
        return {k: v for k, v in payload.items() if v is not None}

    def parse_over(self, text: str) -> dict:
        """Extract one over's fields (speaker/addressed/report/name/qth). Returns
        a dict with only the present, schema-known fields (callsigns are cleaned
        downstream); ``{}`` if nothing parsed."""
        runtime = self._runtime_for()
        messages = [
            {"role": "system", "content": OVER_PROMPT},
            {"role": "user", "content": text},
        ]
        payload = extract_json(runtime.generate(messages, prefix='{"from_call":'))
        if payload is None:
            messages = messages + [{"role": "user", "content":
                "Reply with ONLY one JSON object for the schema, nothing else."}]
            payload = extract_json(runtime.generate(messages, prefix='{"from_call":'))
        if not isinstance(payload, dict):
            return {}
        allowed = set(QSO_OVER_SCHEMA["properties"])
        return {k: v for k, v in payload.items() if k in allowed and v is not None}
