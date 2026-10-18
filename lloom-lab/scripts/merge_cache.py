"""Union a cache DB produced on Kaggle into the local one (idempotent).

    python scripts/merge_cache.py path/to/kaggle/llm_cache.sqlite3
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import RESULTS_DIR


def merge(incoming_db: Path, local_db: Path | None = None) -> int:
    if local_db is None:
        local_db = RESULTS_DIR / "llm_cache.sqlite3"

    print(f"Merging: {incoming_db} -> {local_db}")
    conn = sqlite3.connect(local_db)
    cur = conn.cursor()

    cur.execute("ATTACH DATABASE ? AS incoming", (str(incoming_db),))
    cur.execute("""
        INSERT OR IGNORE INTO cache(key, model, created, response)
        SELECT key, model, created, response FROM incoming.cache
    """)
    n = cur.rowcount
    conn.commit()
    conn.close()
    print(f"Merged {n} new cache entries into {local_db}.")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("incoming", type=Path, help="Path to Kaggle's llm_cache.sqlite3")
    ap.add_argument("--local", type=Path, default=None, help="Path to local llm_cache.sqlite3")
    args = ap.parse_args()
    merge(args.incoming, args.local)


if __name__ == "__main__":
    main()
