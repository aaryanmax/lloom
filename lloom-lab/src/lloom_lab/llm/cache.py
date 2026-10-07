"""SQLite-backed LLM response cache.

Every prompt is stored once; re-runs, retries, and demo rehearsals cost zero
API calls. Keyed by (model, system, prompt, temperature, json_mode).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


def _key_digest(parts: tuple) -> str:
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class ResponseCache:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # timeout: concurrent grid workers write to the same DB — wait for a
        # lock instead of raising "database is locked"
        self._conn = sqlite3.connect(str(db_path), timeout=30.0)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY, model TEXT, created REAL, response TEXT)"
        )
        self._conn.commit()

    def get(self, key_parts: tuple) -> str | None:
        k = _key_digest(key_parts)
        row = self._conn.execute("SELECT response FROM cache WHERE key=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, key_parts: tuple, model: str, response: str) -> None:
        k = _key_digest(key_parts)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, model, created, response) VALUES (?,?,?,?)",
            (k, model, time.time(), response),
        )
        self._conn.commit()

    def stats(self) -> dict:
        n, models = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT model) FROM cache").fetchone()
        return {"entries": n, "distinct_models": models}
