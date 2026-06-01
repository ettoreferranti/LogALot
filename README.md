# LogALot

Local-first amateur-radio logger with an AI capture front-end. It listens to
received audio, transcribes it, parses QSO details with a **local** LLM, and
writes them to your logbook. Nothing leaves the machine.

> The Knight of the Round Logbook. Logs a lot.

## Status

Early scaffold. The logging core (append-only store + ADIF export + validation)
is implemented and stdlib-only. The AI pipeline (audio capture, ASR, LLM parse)
is stubbed behind clean interfaces — see `CLAUDE.md` for the build plan.

## Design in one line

**The AI is advisory; the human commits.** Candidate QSOs from the pipeline land
in a mutable review queue. The canonical logbook is append-only and written only
on explicit confirmation. ADIF export feeds MacLoggerDX (or any ADIF importer).

## Quick start (M0 — ADIF-first MVP, no dependencies)

```bash
python -m pip install -e .
python - <<'PY'
from logalot.store import Store
from logalot.models import QSO, band_for_freq
from logalot.export_adif import write_adif

s = Store("logalot.db")
q = QSO(call="HB9XX", qso_date="20260601", time_on="123000",
        freq_mhz=14.074, band=band_for_freq(14.074), mode="FT8",
        rst_sent="-07", rst_rcvd="-12", my_call="HB9IKS")
s.add_qso(q)                       # straight to canonical (manual path)
assert s.verify_chain()            # tamper-evident hash chain intact
write_adif(s.all_qso(), "out.adi") # import out.adi into MacLoggerDX
PY
```

## Architecture

`capture` → `asr` → `parse` → `validate` → `candidates` queue → **human** →
append-only `qso` store → `export_adif`.

CAT (rigctld/Hamlib) supplies frequency, mode and time — never the transcript.

## Hardware target

FTDX10 over CAT (rigctld); MacBook Pro M4 (48 GB) running faster-whisper and a
local LLM via LM Studio (MLX). All local, all offline-capable.

## Licence

GPL-3.0.
