"""E5: No-Free-Lunch (NFL) Theorem Empirical Demonstration.

Evaluates concept labeling performance under true ground truth vs 10 label derangements
(NFL variants). Demonstrates that without true semantic alignment, learning algorithms
collapse to random chance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lloom_lab.config import DATA_STORE, RESULTS_DIR


def compute_metrics(
    preds_map: dict[str, set[str]],
    gt_map: dict[str, set[str]],
) -> dict[str, float]:
    """Compute precision, recall, and F1 over multi-label doc-to-concept predictions."""
    tp = sum(len(preds_map[d] & gt_map[d]) for d in preds_map if d in gt_map)
    fp = sum(len(preds_map[d] - gt_map[d]) for d in preds_map if d in gt_map)
    fn = sum(len(gt_map[d] - preds_map[d]) for d in preds_map if d in gt_map)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def load_true_ground_truth(docs_path: Path | None = None) -> dict[str, set[str]]:
    """Load doc_id -> set of true specific concepts from docs.jsonl."""
    if docs_path is None:
        docs_path = DATA_STORE / "synthetic_v1" / "docs.jsonl"
    true_docs = [
        json.loads(line)
        for line in docs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {d["doc_id"]: set(d["concepts"]["specific"]) for d in true_docs}


def load_nfl_variants(nfl_dir: Path | None = None) -> dict[str, dict[str, set[str]]]:
    """Load 10 NFL derangement variants from JSON files."""
    if nfl_dir is None:
        nfl_dir = DATA_STORE / "synthetic_v1" / "nfl"
    nfl_files = sorted(nfl_dir.glob("*.json"))
    nfl_variants: dict[str, dict[str, set[str]]] = {}
    for nf in nfl_files:
        data = json.loads(nf.read_text(encoding="utf-8"))
        nfl_variants[nf.stem] = {d: set(data[d]["specific"]) for d in data}
    return nfl_variants


def evaluate_nfl_collapse(
    methods: list[str] | None = None,
    ns: list[int] | None = None,
    n_trials: int = 5,
    runs_dir: Path | None = None,
    docs_path: Path | None = None,
    nfl_dir: Path | None = None,
) -> dict:
    """Run E5 evaluation across methods and sample sizes."""
    if methods is None:
        methods = ["zeroshot_gemini", "zeroshot_qwen"]
    if ns is None:
        ns = [10, 20, 40, 80, 160]
    if runs_dir is None:
        runs_dir = RESULTS_DIR / "runs" / "e1"

    true_gt = load_true_ground_truth(docs_path)
    nfl_variants = load_nfl_variants(nfl_dir)

    results: dict = {}
    for method in methods:
        method_res: dict = {}
        for n in ns:
            true_f1s: list[float] = []
            nfl_f1s: list[float] = []

            for t in range(n_trials):
                run_file = runs_dir / method / f"n{n:03d}_t{t}.json"
                if not run_file.exists():
                    continue
                rec = json.loads(run_file.read_text(encoding="utf-8"))
                assignments = rec.get("assignments", {})
                if not assignments:
                    continue

                preds = {d: set(c) for d, c in assignments.items()}

                # Eval against True Ground Truth
                m_true = compute_metrics(preds, true_gt)
                true_f1s.append(m_true["f1"])

                # Eval against 10 NFL Variants
                trial_nfl_f1s = [
                    compute_metrics(preds, var_gt)["f1"]
                    for var_gt in nfl_variants.values()
                ]
                nfl_f1s.append(float(np.mean(trial_nfl_f1s)))

            if true_f1s:
                mean_true = float(np.mean(true_f1s))
                mean_nfl = float(np.mean(nfl_f1s))
                collapse = float((1 - mean_nfl / max(1e-9, mean_true)) * 100)
                method_res[str(n)] = {
                    "true_f1_mean": mean_true,
                    "true_f1_std": float(np.std(true_f1s)),
                    "nfl_f1_mean": mean_nfl,
                    "nfl_f1_std": float(np.std(nfl_f1s)),
                    "collapse_pct": collapse,
                }
        results[method] = method_res
    return results


def format_e5_markdown(results: dict) -> str:
    """Format E5 evaluation results into Markdown."""
    lines = [
        "# E5: No-Free-Lunch (NFL) Theorem Empirical Evaluation",
        "",
        "Demonstration of performance collapse when ground truth labels are permuted "
        "via derangements (zero fixed points).",
        "",
    ]
    for method, summary in results.items():
        lines.append(f"## {method}")
        lines.append(
            "| $n$ | True F1 (mean ± std) | NFL Permuted F1 (mean ± std) | "
            "Performance Collapse |"
        )
        lines.append("|---|---|---|---|")
        for n_str, s in summary.items():
            lines.append(
                f"| {n_str} | {s['true_f1_mean']:.3f} ± {s['true_f1_std']:.3f} | "
                f"{s['nfl_f1_mean']:.3f} ± {s['nfl_f1_std']:.3f} | "
                f"**{s['collapse_pct']:.1f}%** |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    print("Running E5: No-Free-Lunch Theorem Empirical Evaluation...")
    e5_dir = RESULTS_DIR / "e5"
    e5_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_nfl_collapse()

    for method, summary in results.items():
        for n_str, s in summary.items():
            print(
                f"[{method}] n={int(n_str):3d}: True F1={s['true_f1_mean']:.3f} vs "
                f"NFL F1={s['nfl_f1_mean']:.3f} (Collapse: {s['collapse_pct']:.1f}%)"
            )

    (e5_dir / "e5_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (e5_dir / "e5_summary.md").write_text(
        format_e5_markdown(results), encoding="utf-8"
    )
    print(
        f"\n[E5 COMPLETE] Results saved to:\n"
        f"  {e5_dir / 'e5_results.json'}\n"
        f"  {e5_dir / 'e5_summary.md'}"
    )


if __name__ == "__main__":
    main()
