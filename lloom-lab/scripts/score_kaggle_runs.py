"""Sync Kaggle lloom_qwen runs, score them with the Gemini matcher, and update E1 summary."""

import json
import shutil
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import RUNS_DIR, make_client
from lloom_lab.experiments import e1
from lloom_lab.metrics import coverage as cov


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    kaggle_qwen_dir = repo_root / "kaggle_results" / "runs" / "e1" / "lloom_qwen"
    local_qwen_dir = RUNS_DIR / "e1" / "lloom_qwen"
    backup_dir = RUNS_DIR / "e1" / "lloom_qwen_local_backup"

    if not kaggle_qwen_dir.exists():
        sys.exit(f"Kaggle results directory not found: {kaggle_qwen_dir}")

    # 1. Backup existing local lloom_qwen directory if not already backed up
    if local_qwen_dir.exists() and not backup_dir.exists():
        print(f"Backing up {local_qwen_dir} -> {backup_dir}...")
        shutil.copytree(local_qwen_dir, backup_dir)
        print("Backup complete.")

    local_qwen_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy Kaggle files to local if not present or missing coverage_split
    print("Syncing run files from Kaggle...")
    kaggle_files = sorted(kaggle_qwen_dir.glob("*.json"))
    for kf in kaggle_files:
        dest = local_qwen_dir / kf.name
        # If dest doesn't exist, copy directly
        if not dest.exists():
            shutil.copy(kf, dest)
        else:
            # If dest exists, check if it already has coverage_split
            local_data = json.loads(dest.read_text(encoding="utf-8"))
            if local_data.get("coverage_split") is None:
                shutil.copy(kf, dest)

    client = make_client()
    cells = sorted(local_qwen_dir.glob("*.json"))
    print(f"\nEvaluating Gemini matcher coverage on {len(cells)} cells...")

    for i, cell_path in enumerate(cells, 1):
        record = json.loads(cell_path.read_text(encoding="utf-8"))
        if record.get("coverage_split") is not None:
            split = record["coverage_split"]
            gen_val = split.get("generic", 0.0)
            spec_val = split.get("specific", 0.0)
            print(f"[{i:02d}/{len(cells)}] {cell_path.stem}: already scored "
                  f"(generic={gen_val:.2f}, specific={spec_val:.2f})")
            continue

        gt = record.get("ground_truth", {})
        concepts = record.get("concepts", [])
        generated = cov.parse_generated(concepts)

        if not generated:
            record["coverage_split"] = {"generic": 0.0, "specific": 0.0}
            record["coverage"] = 0.0
            record["n_generated"] = 0
            record["match_rows"] = []
        else:
            types = {g: "generic" for g in gt.get("generic", [])}
            types.update({s: "specific" for s in gt.get("specific", [])})
            all_gt = gt.get("generic", []) + gt.get("specific", [])

            t0 = time.time()
            try:
                rows = cov.match_concepts(client, all_gt, generated, model="gemini")
            except Exception as ex:
                print(f"[{i:02d}/{len(cells)}] {cell_path.stem} Matcher FAILED: {ex}")
                raise

            split = cov.coverage_split(rows, types)
            record["coverage_split"] = split
            record["coverage"] = cov.coverage(rows)
            record["match_rows"] = rows
            record["n_generated"] = len(generated)
            elapsed = time.time() - t0
            g_sc = split.get('generic', 0.0)
            s_sc = split.get('specific', 0.0)
            print(f"[{i:02d}/{len(cells)}] {cell_path.stem} scored in {elapsed:.1f}s -> "
                  f"generic={g_sc:.2f}, specific={s_sc:.2f}")

        # Save atomically
        tmp = cell_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(cell_path)

    # 3. Update E1 summary tables
    print("\nRegenerating E1 summary...")
    result = e1.collect()
    jpath, mpath = e1.write_summary(result)
    print(f"Updated summary files:\n  {jpath}\n  {mpath}")

    # Print summary table for lloom_qwen
    print("\n--- New lloom_qwen Results ---")
    summary = result["per_method"].get("lloom_qwen", {})
    for e in summary.get("per_n", []):
        g, s = e["generic"], e["specific"]
        cv = f"{s['cv']:.2f}" if s.get("cv") is not None else "—"
        print(f"n={e['n']:3d} ({e['n_completed']} runs): "
              f"generic={g['mean']:.2f} ± {g['std']:.2f} | "
              f"specific={s['mean']:.2f} ± {s['std']:.2f} | CV={cv}")


if __name__ == "__main__":
    main()
