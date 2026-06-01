# CLAUDE.md — LogALot

Context for Claude Code working in this repo. Read this first.

## What this is

LogALot is a local-first amateur-radio logger with an **AI capture front-end**:
it listens to received audio, transcribes it, parses QSO details with a *local*
LLM, and writes them to a logbook. Nothing leaves the machine. The novel part is
the local-LLM QSO capture; the logging core is deliberately boring and reliable.

Operator: HB9IKS (Switzerland). Rig: Yaesu FTDX10. Host: MacBook Pro M4, 48 GB.
AI runs **in-process via MLX** — no LM Studio/Ollama/server on the side. Models
are pulled from the HuggingFace `mlx-community` hub and loaded inside the app
(`mlx-lm` for parse, `mlx-whisper` for ASR); all MLX work funnels through one
shared Metal thread (`logalot.mlx_runtime`). Candidate parse models: Qwen2.5-
Instruct (default), Apertus (ETH/EPFL, the Swiss option).

## The one invariant that must never break

**The AI is advisory. The human commits.**

- The transcription → parse pipeline writes *candidate* QSOs into a mutable
  review queue (`candidates` table).
- The canonical logbook (`qso` table) is **append-only** and is written *only*
  on explicit human confirmation.
- ADIF export reads from the canonical store only.
- If the entire AI layer falls over, the manual logging path still works.
  Fail-safe, not fail-open.

Rationale is data quality, not law (see next section). But the invariant holds
regardless — a hallucinated callsign must never silently enter the record.

## Legal framing (checked, June 2026)

Logbook keeping is **not mandatory** in Switzerland. BAKOM *Vorschriften für den
Amateurfunk*, citing Art. 36 FKV: the authority *kann* (may) oblige a licensee to
keep records of their traffic — a latent power, not a standing duty. The
BAKOM/USKA exam catalogue confirms it. (Germany: BNetzA dropped the requirement
entirely as of the 24 June 2024 AFuV revision.)

Consequence: append-only is an engineering choice for a trustworthy record, not a
compliance constraint. We keep it anyway. The only scenario that flips this is a
specific BAKOM order under Art. 36 (e.g. after an EMC complaint).

Note: storing QSO *content* (not just metadata) touches the other station's
personal data — keep DSG/GDPR in mind before any publish/sync feature.

## Module decomposition

Each module is independently replaceable behind a narrow interface.

| Module            | Responsibility                                   | State at scaffold |
|-------------------|--------------------------------------------------|-------------------|
| `capture`         | RX audio + CAT (rigctld/Hamlib → freq/mode/time) | stub + interface  |
| `asr`             | faster-whisper → raw transcript                  | stub + interface  |
| `parse`           | local LLM → strict JSON, schema-validated        | stub + prompt/schema |
| `validate`        | callsign regex, NATO phonetic expansion, flags   | **implemented**   |
| `store`           | candidates (mutable) + qso (append-only, hashed) | **implemented**   |
| `export_adif`     | ADIF 3.1.4 writer                                | **implemented**   |

Frequency, mode and time come from **CAT, not audio** — never trust Whisper for
numbers it can get from the rig. System clock in UTC for timestamps.

## Milestones

- **M0 — ADIF-first MVP (scaffolded here).** Manual entry → append-only store →
  ADIF export → import into MacLoggerDX. Zero external dependencies; stdlib only.
  Verify this round-trips into MacLoggerDX before building anything clever.
- **M1 — capture.** rigctld integration (Hamlib `rigctld -m <id> -r <device>`),
  audio device selection (FTDX10 USB CODEC), ring buffer.
- **M2 — asr.** in-process `mlx-whisper` (`large-v3-turbo` default), VAD
  segmentation. Expect poor copy on weak signals — this is the SNR-limited part.
- **M3 — parse.** in-process `mlx-lm` generate, JSON-only prompt, schema
  validation, rejection of malformed output. NATO-phonetic expansion before
  callsign regex.
- **M4 — review queue UI.** Thin local web UI (or Tauri) to confirm/edit/reject
  candidates. This is where the human commits.
- **M5 — logger adapters.** Beyond ADIF: Wavelog/Cloudlog API, later LoTW/TQSL.

Do not reimplement band plans, contest scoring, award tracking — grow into a full
logger only if the capture daemon earns it.

## Stack & conventions

- Python ≥ 3.11. Core (`models`, `store`, `validate`, `export_adif`) is
  **stdlib-only** and must stay that way — it's the part that must never fail.
- Runtime deps live in optional extras: `[capture]`, `[asr]`, `[parse]`.
- Type hints everywhere. UTC everywhere. Metric/SI everywhere.
- Tests with `pytest`; the stdlib core must stay fully tested.
- Licence: GPL-3.0 (matches the ham OSS ecosystem — CQRLOG, Wavelog precedent).
  Add the LICENSE file via GitHub's picker; it's not in this bundle.

## Open questions to settle in-session

1. Review UI: thin web (FastAPI + htmx) vs Tauri? Leaning web for speed.
2. Candidate confidence: simple regex-pass flag (current) vs model logprobs?
3. Multi-QSO segmentation: how to split overlapping/rapid contacts from one
   audio stream? Probably VAD + per-segment parse, but contests will stress it.
4. CAT polling cadence vs logging the freq/mode at QSO *start* vs *end*.
