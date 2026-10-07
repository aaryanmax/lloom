"""Request ledger: append-only log of every LLM call (cached or not) with
daily totals, so free-tier quotas are never blown by accident."""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path


class RequestLedger:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=30.0)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS requests ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts REAL, day TEXT, model TEXT, kind TEXT,"
            " cached INTEGER, tokens_in INTEGER, tokens_out INTEGER)"
        )
        self._conn.commit()

    def log(self, model: str, kind: str, cached: bool,
            tokens_in: int = 0, tokens_out: int = 0) -> None:
        now = _dt.datetime.now()
        self._conn.execute(
            "INSERT INTO requests (ts, day, model, kind, cached, tokens_in, tokens_out)"
            " VALUES (?,?,?,?,?,?,?)",
            (now.timestamp(), now.date().isoformat(), model, kind,
             int(cached), tokens_in, tokens_out),
        )
        self._conn.commit()

    def today_count(self, model: str, cached: bool = False) -> int:
        day = _dt.date.today().isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM requests WHERE day=? AND model=? AND cached=?",
            (day, model, int(cached)),
        ).fetchone()
        return row[0]

    def daily_report(self) -> list[tuple]:
        day = _dt.date.today().isoformat()
        return self._conn.execute(
            "SELECT model, kind, COUNT(*), SUM(tokens_in), SUM(tokens_out)"
            " FROM requests WHERE day=? GROUP BY model, kind", (day,)
        ).fetchall()
