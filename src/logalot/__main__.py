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

    args = p.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
