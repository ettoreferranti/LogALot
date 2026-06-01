"""Persistence: a mutable candidate queue and an append-only canonical logbook.

Stdlib-only (sqlite3). Two tables enforce the core invariant in storage:

* ``candidates`` — mutable scratch. The AI pipeline writes here; humans edit and
  reject here. Nothing in here is part of the record of truth.
* ``qso`` — append-only canonical log. INSERT only (no UPDATE/DELETE in this
  API). Each row carries a SHA-256 hash chained to its predecessor, so any later
  tampering is detectable via :meth:`Store.verify_chain`.

Append-only is a discipline here, not a database feature: we simply never expose
mutation of ``qso``. The hash chain makes that discipline auditable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable

from .models import QSO

_CANONICAL_FIELDS = (
    "call", "qso_date", "time_on", "mode", "band", "freq_mhz",
    "rst_sent", "rst_rcvd", "name", "qth", "gridsquare", "comment", "my_call",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    transcript  TEXT,
    confidence  TEXT,
    payload     TEXT NOT NULL          -- JSON of proposed QSO fields
);
CREATE TABLE IF NOT EXISTS qso (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    payload     TEXT NOT NULL,         -- JSON of canonical QSO fields
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL
);
"""

_GENESIS = "0" * 64


def _row_hash(prev_hash: str, payload_json: str) -> str:
    return hashlib.sha256((prev_hash + payload_json).encode("utf-8")).hexdigest()


def _canonical_json(q: QSO) -> str:
    # Deterministic serialisation so the hash is stable across runs.
    d = {k: getattr(q, k) for k in _CANONICAL_FIELDS}
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Store:
    def __init__(self, path: str = "logalot.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- candidate queue (mutable) -------------------------------------------

    def add_candidate(self, payload: dict, transcript: str = "",
                      confidence: str = "review") -> int:
        cur = self.conn.execute(
            "INSERT INTO candidates (created_at, transcript, confidence, payload)"
            " VALUES (?, ?, ?, ?)",
            (time.time(), transcript, confidence, json.dumps(payload)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_candidates(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, created_at, transcript, confidence, payload"
            " FROM candidates ORDER BY id"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def drop_candidate(self, candidate_id: int) -> None:
        self.conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        self.conn.commit()

    # --- canonical log (append-only) -----------------------------------------

    def _last_hash(self) -> str:
        row = self.conn.execute(
            "SELECT hash FROM qso ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["hash"] if row else _GENESIS

    def add_qso(self, q: QSO) -> int:
        """Append one confirmed QSO to the canonical log. The only write path."""
        payload = _canonical_json(q)
        prev = self._last_hash()
        h = _row_hash(prev, payload)
        cur = self.conn.execute(
            "INSERT INTO qso (created_at, payload, prev_hash, hash)"
            " VALUES (?, ?, ?, ?)",
            (time.time(), payload, prev, h),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def confirm_candidate(self, candidate_id: int, q: QSO) -> int:
        """Human action: promote a (possibly edited) candidate into the record,
        then remove it from the queue."""
        qso_id = self.add_qso(q)
        self.drop_candidate(candidate_id)
        return qso_id

    def all_qso(self) -> list[QSO]:
        rows = self.conn.execute(
            "SELECT payload FROM qso ORDER BY id"
        ).fetchall()
        return [QSO(**json.loads(r["payload"])) for r in rows]

    def verify_chain(self) -> bool:
        """Recompute the hash chain; False if any row was altered or reordered."""
        prev = _GENESIS
        for r in self.conn.execute(
            "SELECT payload, prev_hash, hash FROM qso ORDER BY id"
        ):
            if r["prev_hash"] != prev:
                return False
            if _row_hash(prev, r["payload"]) != r["hash"]:
                return False
            prev = r["hash"]
        return True
