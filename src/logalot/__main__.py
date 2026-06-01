"""Minimal CLI for the M0 path: inspect the log, verify the chain, export ADIF.

The capture daemon (M1+) will grow its own subcommands. For now this is enough to
prove the ADIF-first round-trip into MacLoggerDX.
"""
from __future__ import annotations

import argparse

from .export_adif import write_adif
from .store import Store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="logalot")
    p.add_argument("--db", default="logalot.db", help="SQLite path")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="verify the canonical hash chain")
    e = sub.add_parser("export", help="export canonical log to ADIF")
    e.add_argument("out", help="output .adi path")
    sub.add_parser("count", help="print number of canonical QSOs")

    m = sub.add_parser("monitor", help="live rig dashboard (needs [ui] extra + rigctld)")
    m.add_argument("--rig-host", default="127.0.0.1", help="rigctld host")
    m.add_argument("--rig-port", type=int, default=4532, help="rigctld port")
    m.add_argument("--host", default="127.0.0.1", help="web bind host")
    m.add_argument("--port", type=int, default=8765, help="web bind port")
    m.add_argument("--audio-device", default=None,
                   help="input device name substring (default: USB Audio CODEC)")
    m.add_argument("--no-audio", action="store_true", help="disable audio capture")
    m.add_argument("--no-asr", action="store_true",
                   help="disable the transcript feed (audio level only)")
    m.add_argument("--no-parse", action="store_true",
                   help="disable the LLM candidate panel (transcript only)")
    m.add_argument("--vad-threshold", type=float, default=-45.0,
                   help="speech gate in dBFS; raise toward the band noise floor (default -45)")
    m.add_argument("--list-audio", action="store_true",
                   help="list input devices and exit")

    args = p.parse_args(argv)

    # The live monitor needs the [ui] extra and never touches the store, so it is
    # handled before opening the DB and imported lazily (keeps the core CLI
    # dependency-free).
    if args.cmd == "monitor":
        return _run_monitor(args)

    store = Store(args.db)

    if args.cmd == "verify":
        ok = store.verify_chain()
        print("chain OK" if ok else "CHAIN BROKEN")
        return 0 if ok else 1
    if args.cmd == "export":
        qsos = store.all_qso()
        write_adif(qsos, args.out)
        print(f"wrote {len(qsos)} QSO(s) to {args.out}")
        return 0
    if args.cmd == "count":
        print(len(store.all_qso()))
        return 0
    return 2


def _run_monitor(args) -> int:
    if args.list_audio:
        try:
            from .audio import list_input_devices
        except ModuleNotFoundError:
            print("audio needs the [capture] extra: pip install -e '.[capture]'")
            return 1
        for d in list_input_devices():
            print(f"  [{d.index}] {d.name}  ({d.channels} ch, {int(d.samplerate)} Hz)")
        return 0

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("monitor needs the [ui] extra: pip install -e '.[ui]'")
        return 1
    from .monitor import create_app

    app = create_app(args.rig_host, args.rig_port,
                     audio_device=args.audio_device, enable_audio=not args.no_audio,
                     enable_asr=not args.no_asr, enable_parse=not args.no_parse,
                     vad_threshold=args.vad_threshold)
    print(f"rig monitor on http://{args.host}:{args.port}  (rigctld {args.rig_host}:{args.rig_port})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
