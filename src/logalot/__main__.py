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

    cp = sub.add_parser("cat-probe",
                        help="auto-detect the rig CAT serial port and launch rigctld")
    cp.add_argument("--model", default="1042", help="Hamlib rig model id (default 1042 = FTDX10)")
    cp.add_argument("--baud", type=int, default=38400, help="CAT baud (default 38400)")
    cp.add_argument("--rig-port", type=int, default=4532, help="rigctld TCP port (default 4532)")
    cp.add_argument("--no-launch", action="store_true",
                    help="just find the device and print the rigctld command, don't run it")
    cp.add_argument("--device", action="append",
                    help="probe only this device path (repeatable); default: scan USB-serial")

    m = sub.add_parser("monitor", help="live rig dashboard (needs [ui] extra + rigctld)")
    m.add_argument("--rig-host", default="127.0.0.1", help="rigctld host")
    m.add_argument("--rig-port", type=int, default=4532, help="rigctld port")
    m.add_argument("--auto-cat", action="store_true",
                   help="auto-detect the CAT port and launch rigctld in-process (one terminal)")
    m.add_argument("--cat-model", default="1042", help="rig model for --auto-cat (default 1042=FTDX10)")
    m.add_argument("--cat-baud", type=int, default=38400, help="CAT baud for --auto-cat (default 38400)")
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
    m.add_argument("--min-logprob", type=float, default=-1.0,
                   help="drop transcripts below this mean token logprob (default -1.0)")
    m.add_argument("--language", default="auto",
                   help="ASR language code, or 'auto' to detect (default auto); tunable live")
    m.add_argument("--offline", action="store_true",
                   help="use only cached models; skip all HuggingFace network checks")
    m.add_argument("--translate", action="store_true",
                   help="translate non-English speech to English (default off); tunable live")
    m.add_argument("--enhance", action="store_true",
                   help="denoise RX audio before ASR (needs [enhance] extra); tunable live")
    m.add_argument("--list-audio", action="store_true",
                   help="list input devices and exit")

    args = p.parse_args(argv)

    # The live monitor needs the [ui] extra and never touches the store, so it is
    # handled before opening the DB and imported lazily (keeps the core CLI
    # dependency-free).
    if args.cmd == "monitor":
        return _run_monitor(args)
    if args.cmd == "cat-probe":
        return _run_cat_probe(args)

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


def _raise_keyboard_interrupt(*_args) -> None:
    raise KeyboardInterrupt


def _run_cat_probe(args) -> int:
    import signal

    from .cat import (
        candidate_devices,
        port_in_use,
        probe_device,
        rigctld_command,
        rigctld_path,
        stop_rigctld,
    )

    if rigctld_path() is None:
        print("rigctld not found — install Hamlib (brew install hamlib)")
        return 1
    if port_in_use(args.rig_port):
        print(f"something is already listening on :{args.rig_port} "
              f"(rigctld already running?). Stop it, or pass --rig-port.")
        return 1

    devices = args.device or candidate_devices()
    if not devices:
        print("no USB-serial devices found — is the rig plugged in and powered on?")
        return 1

    # Install interrupt handling BEFORE we spawn anything, and track the live
    # rigctld, so a Ctrl+C/SIGTERM at any point reaps it (no orphan on the port).
    # default_int_handler re-honours SIGINT even if it was inherited as ignored.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    live: list[object] = []   # at most one: the rigctld currently spawned

    print(f"probing {len(devices)} device(s) for a model-{args.model} rig @ {args.baud} baud…")
    try:
        for dev in devices:
            print(f"  {dev} … ", end="", flush=True)
            proc, freq = probe_device(dev, args.model, args.baud, args.rig_port,
                                      on_spawn=lambda p: live.append(p))
            if freq is None:
                live.clear()        # probe_device already reaped a failed proc
                print("no CAT response")
                continue
            print(f"CAT OK — {freq:.6f} MHz")
            print(f"\n{rigctld_command(dev, args.model, args.baud, args.rig_port)}")
            if args.no_launch:
                stop_rigctld(proc)
                return 0
            print(f"rigctld running on :{args.rig_port}. Leave this open; "
                  f"run 'logalot monitor' elsewhere. Ctrl+C to stop.")
            proc.wait()
            return 0
    except KeyboardInterrupt:
        print("\nstopping rigctld…")
    finally:
        for p in live:
            stop_rigctld(p)

    if not live:
        print("\nno CAT device answered. Check: rig on, CAT enabled in the menu, "
              "and the baud matches the rig's CAT RATE (try --baud 4800/9600/19200).")
    return 1


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

    if args.offline:
        # Cache-only: huggingface_hub won't hit the network at all (no per-launch
        # metadata checks, fully local). An uncached model then errors clearly.
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("offline: using cached models only (no HuggingFace network access)")

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("monitor needs the [ui] extra: pip install -e '.[ui]'")
        return 1
    from .monitor import create_app

    rigctld_procs: list = []
    if args.auto_cat:
        import signal

        # Handle interrupts before we spawn rigctld, and reap it in the finally
        # below, so --auto-cat never orphans rigctld on the serial port.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    try:
        if args.auto_cat:
            _autostart_rigctld(args, rigctld_procs)
        app = create_app(args.rig_host, args.rig_port,
                         audio_device=args.audio_device, enable_audio=not args.no_audio,
                         enable_asr=not args.no_asr, enable_parse=not args.no_parse,
                         vad_threshold=args.vad_threshold, min_logprob=args.min_logprob,
                         language=args.language, translate=args.translate,
                     enable_enhance=args.enhance)
        print(f"rig monitor on http://{args.host}:{args.port}  (rigctld {args.rig_host}:{args.rig_port})")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        if rigctld_procs:
            from .cat import stop_rigctld
            for p in rigctld_procs:
                stop_rigctld(p)
    return 0


def _autostart_rigctld(args, holder: list) -> None:
    """Probe for the CAT port and launch rigctld in-process (--auto-cat). On
    success ``holder`` ends with the running proc; if nothing answers we warn and
    continue — the dashboard still runs (audio/transcript), CAT just shows offline."""
    from .cat import (
        candidate_devices,
        port_in_use,
        probe_device,
        rigctld_path,
    )

    if rigctld_path() is None:
        print("auto-cat: rigctld not found (brew install hamlib) — continuing without CAT")
        return
    if port_in_use(args.rig_port):
        print(f"auto-cat: something already on :{args.rig_port} — using it")
        return
    devices = candidate_devices()
    if not devices:
        print("auto-cat: no USB-serial devices found — continuing without CAT")
        return
    print(f"auto-cat: probing {len(devices)} device(s) for a model-{args.cat_model} rig…")
    for dev in devices:
        proc, freq = probe_device(dev, args.cat_model, args.cat_baud, args.rig_port,
                                  on_spawn=holder.append)
        if freq is not None:
            print(f"auto-cat: CAT on {dev} ({freq:.6f} MHz); rigctld on :{args.rig_port}")
            return
        holder.clear()        # failed proc already reaped by probe_device
    print("auto-cat: no CAT device answered — continuing without CAT")


if __name__ == "__main__":
    raise SystemExit(main())
