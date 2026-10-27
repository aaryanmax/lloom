"""Score the 7 incomplete lloom_gemma cells (n=80/160) that have concepts but
no coverage_split. Uses only the matcher (Gemini), NOT the synthesize step —
the concepts are already on disk.

Estimated calls: ~7 cells × ~9 ground-truth batches each ≈ 63 Gemini calls.
All calls go through cache-first, so re-running this script after a partial run
costs 0 extra calls.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lloom-lab" / "src"))

from lloom_lab.config import make_client
from lloom_lab.experiments.e1 import METHODS, N_TRIALS, NS, E1_DIR, write_summary, collect
from lloom_lab.metrics import coverage as cov

FAILED_CELLS = [
    ("lloom_gemma", 80, 3),
    ("lloom_gemma", 80, 4),
    ("lloom_gemma", 160, 0),
    ("lloom_gemma", 160, 1),
    ("lloom_gemma", 160, 2),
    ("lloom_gemma", 160, 3),
    ("lloom_gemma", 160, 4),
]


def score_cell(client, path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    
    if record.get("coverage_split") is not None:
        cs = record["coverage_split"]
        print(f"  already scored: generic={cs.get('generic',0):.2f} "
              f"specific={cs.get('specific',0):.2f} — skip")
        return record

    gt = record.get("ground_truth", {})
    concepts = record.get("concepts", [])
    generated = cov.parse_generated(concepts)

    if not generated:
        print(f"  no concepts found — marking 0.00 / 0.00")
        record["coverage_split"] = {"generic": 0.0, "specific": 0.0}
        record["coverage"] = 0.0
        record["n_generated"] = 0
        record["match_rows"] = []
    else:
        all_gt = gt.get("generic", []) + gt.get("specific", [])
        types = {g: "generic" for g in gt.get("generic", [])}
        types.update({s: "specific" for s in gt.get("specific", [])})

        print(f"  running matcher: {len(generated)} concepts vs {len(all_gt)} ground truths …")
        t0 = time.time()
        rows = cov.match_concepts(client, all_gt, generated, model="gemini")
        split = cov.coverage_split(rows, types)
        elapsed = time.time() - t0

        record["coverage_split"] = split
        record["coverage"] = cov.coverage(rows)
        record["match_rows"] = rows
        record["n_generated"] = len(generated)
        print(f"  done in {elapsed:.1f}s — "
              f"generic={split.get('generic',0):.2f} specific={split.get('specific',0):.2f}")

    # Clear error status now that we have a score
    record["status"] = "complete"
    record["score_patched"] = True

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return record


def main():
    print("=" * 60)
    print(" Scoring 7 incomplete lloom_gemma cells (matcher only)")
    print("=" * 60)

    client = make_client()
    total_scored = 0

    for method, n, trial in FAILED_CELLS:
        path = E1_DIR / method / f"n{n:03d}_t{trial}.json"
        print(f"\n[{method}] n={n} t={trial} — {path.name}")
        if not path.exists():
            print("  FILE MISSING — skip")
            continue
        score_cell(client, path)
        total_scored += 1

    print(f"\n{'='*60}")
    print(f" Scored {total_scored}/{len(FAILED_CELLS)} cells.")
    print(" Regenerating E1 summary …")

    result = collect()
    jpath, mpath = write_summary(result)
    print(f" Updated: {jpath.name}, {mpath.name}")

    print("\n--- lloom_gemma final table ---")
    for e in result["per_method"].get("lloom_gemma", {}).get("per_n", []):
        g, s = e["generic"], e["specific"]
        if e["n_completed"] == 0:
            print(f"  n={e['n']:3d} | 0 runs | —")
        else:
            cv = f"{s['cv']:.2f}" if s.get("cv") is not None else "0.00"
            print(f"  n={e['n']:3d} | {e['n_completed']} runs | "
                  f"generic={g['mean']:.2f}±{g['std']:.2f} "
                  f"specific={s['mean']:.2f}±{s['std']:.2f} CV={cv}")

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
