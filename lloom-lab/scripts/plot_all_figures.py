"""Generate publication-ready figures for E1, E2, E3, and E4 experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_e1_coverage():
    summary_path = RESULTS_DIR / "e1" / "e1_summary.json"
    if not summary_path.exists():
        print(f"Skipping E1 plot: {summary_path} not found.")
        return

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per_method = data.get("per_method", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    colors = {
        "lloom_qwen": "#1f77b4",       # Blue
        "lloom_gemma": "#ff7f0e",      # Orange
        "zeroshot_gemini": "#2ca02c",  # Green
        "zeroshot_qwen": "#d62728",    # Red
        "bertopic": "#9467bd",         # Purple
    }
    markers = {
        "lloom_qwen": "o",
        "lloom_gemma": "s",
        "zeroshot_gemini": "^",
        "zeroshot_qwen": "v",
        "bertopic": "D",
    }
    labels = {
        "lloom_qwen": "LLooM (Qwen3-8B local)",
        "lloom_gemma": "LLooM (Gemini Cloud/Gemma)",
        "zeroshot_gemini": "Zero-shot (Gemini Flash-Lite)",
        "zeroshot_qwen": "Zero-shot (Qwen3-8B)",
        "bertopic": "BERTopic (Embedding Baseline)",
    }

    for method, summary in per_method.items():
        if method not in colors:
            continue
        c = colors[method]
        m = markers.get(method, "o")
        lbl = labels.get(method, method)

        ns = [e["n"] for e in summary["per_n"] if e["n_completed"] > 0]
        spec_means = [e["specific"]["mean"] for e in summary["per_n"] if e["n_completed"] > 0]
        spec_stds = [e["specific"]["std"] for e in summary["per_n"] if e["n_completed"] > 0]

        gen_means = [e["generic"]["mean"] for e in summary["per_n"] if e["n_completed"] > 0]
        gen_stds = [e["generic"]["std"] for e in summary["per_n"] if e["n_completed"] > 0]

        ls = "-" if "lloom" in method else "--"
        # Left plot: Specific concepts
        ax1.plot(ns, spec_means, label=lbl, color=c, marker=m, linestyle=ls)
        ax1.fill_between(ns, np.subtract(spec_means, spec_stds), np.add(spec_means, spec_stds),
                         color=c, alpha=0.15)

        # Right plot: Generic concepts
        ax2.plot(ns, gen_means, label=lbl, color=c, marker=m, linestyle=ls)
        ax2.fill_between(ns, np.subtract(gen_means, gen_stds), np.add(gen_means, gen_stds),
                         color=c, alpha=0.15)

    ax1.set_title("(A) Specific Concept Coverage (Primary Metric)")
    ax1.set_xlabel("Sample Size $n$ (documents)")
    ax1.set_ylabel("Ground-Truth Coverage")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.set_title("(B) Generic Concept Coverage (Umbrella Categories)")
    ax2.set_xlabel("Sample Size $n$ (documents)")
    ax2.set_ylabel("Ground-Truth Coverage")
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("E1: Empirical Sample Complexity Across Induction Methods", y=1.02)
    plt.tight_layout()
    out_file = CHARTS_DIR / "figure1_e1_coverage_vs_n.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_e1_cv():
    summary_path = RESULTS_DIR / "e1" / "e1_summary.json"
    if not summary_path.exists():
        return

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per_method = data.get("per_method", {})

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"lloom_qwen": "#1f77b4", "lloom_gemma": "#ff7f0e", "zeroshot_qwen": "#d62728"}
    labels = {
        "lloom_qwen": "LLooM (Qwen3-8B)",
        "lloom_gemma": "LLooM (Gemini Cloud)",
        "zeroshot_qwen": "Zero-shot (Qwen3-8B)",
    }

    for method in ["lloom_qwen", "lloom_gemma", "zeroshot_qwen"]:
        if method not in per_method:
            continue
        summary = per_method[method]
        ns = [e["n"] for e in summary["per_n"] if e["n_completed"] > 0 and e["specific"]["cv"] is not None]
        cvs = [e["specific"]["cv"] for e in summary["per_n"] if e["n_completed"] > 0 and e["specific"]["cv"] is not None]
        ax.plot(ns, cvs, marker="o", label=labels[method], color=colors[method], linewidth=2.2)

    ax.set_title("Variance Collapse: Coefficient of Variation ($CV = \\sigma / \\mu$) vs $n$")
    ax.set_xlabel("Sample Size $n$ (documents)")
    ax.set_ylabel("Coefficient of Variation $CV$")
    ax.set_ylim(-0.1, 2.5)
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    out_file = CHARTS_DIR / "figure2_e1_variance_collapse.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_e2_pac():
    e2_path = RESULTS_DIR / "e2" / "e2_rows.json"
    if not e2_path.exists():
        print(f"Skipping E2 plot: {e2_path} not found.")
        return

    rows = json.loads(e2_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    pairs = [(0.10, 0.05), (0.20, 0.05)]
    color_map = {(0.10, 0.05): "#1f77b4", (0.20, 0.05): "#2ca02c"}

    for eps, delta in pairs:
        sel = sorted([r for r in rows if abs(r["eps"] - eps) < 1e-4 and abs(r["delta"] - delta) < 1e-4], key=lambda x: x["m"])
        if not sel:
            continue
        ms = [r["m"] for r in sel]
        fails = [r["failure_rate"] for r in sel]
        bound_m = sel[0]["bound_m"]
        c = color_map[(eps, delta)]

        ax.plot(ms, fails, label=fr"Empirical Failure ($\epsilon={eps}, \delta={delta}$)", color=c, marker=".")
        ax.axvline(x=bound_m, color=c, linestyle=":", alpha=0.8, label=fr"PAC Bound $m \geq {bound_m}$")
        ax.axhline(y=delta, color="gray", linestyle="--", alpha=0.4)

    ax.set_title("E2: Axis-Parallel Rectangles vs Theoretical PAC Bound")
    ax.set_xlabel("Sample Size $m$")
    ax.set_ylabel(r"Empirical Failure Rate $P(\mathrm{error} > \epsilon)$")
    ax.set_ylim(-0.02, 0.35)
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    out_file = CHARTS_DIR / "figure3_e2_pac_bound_rectangles.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_e3_vc():
    e3_path = RESULTS_DIR / "e3" / "e3_results.json"
    if not e3_path.exists():
        print(f"Skipping E3 plot: {e3_path} not found.")
        return

    data = json.loads(e3_path.read_text(encoding="utf-8"))
    growth = data.get("growth", {})
    ms = growth.get("m", [])
    pi_h = growth.get("Pi_H", [])
    sauer = growth.get("sauer", [])

    fig, ax = plt.subplots(figsize=(8, 5))
    shatter_curve = [2**m for m in ms[:6]]

    ax.plot(ms[:6], shatter_curve, "k--", label=r"Full Shattering ($2^m$)", alpha=0.7)
    ax.plot(ms, pi_h, "o-", color="#1f77b4", label=r"Empirical Growth $\Pi_H(m)$", linewidth=2.2)
    ax.plot(ms, sauer, "s:", color="#d62728", label=r"Sauer Bound $\sum_{i=0}^d C(m, i)$", linewidth=2.0)

    ax.axvline(x=2, color="orange", linestyle="--", alpha=0.8, label=r"VC Dimension $d=2$")
    ax.set_yscale("log")
    ax.set_title(r"E3: Growth Function $\Pi_H(m)$ and Sauer's Lemma Bound (Intervals $d=2$)")
    ax.set_xlabel("Sample Size $m$")
    ax.set_ylabel("Dichotomies Realized (Log Scale)")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    out_file = CHARTS_DIR / "figure4_e3_growth_function_sauer.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_e4_generalization_gap():
    e4_path = RESULTS_DIR / "e4" / "e4_rows.json"
    if not e4_path.exists():
        print(f"Skipping E4 plot: {e4_path} not found.")
        return

    rows = json.loads(e4_path.read_text(encoding="utf-8"))
    if not rows:
        return

    ns = sorted(list({r["meta"]["n"] for r in rows}))
    mean_gaps = []
    std_gaps = []

    for n in ns:
        sel = [r["mean_gap"] for r in rows if r["meta"]["n"] == n]
        mean_gaps.append(float(np.mean(sel)))
        std_gaps.append(float(np.std(sel)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, mean_gaps, "o-", color="#2ca02c", linewidth=2.4, label="Mean Generalization Gap $|\\hat{R}_S - R_{held}|$")
    ax.fill_between(ns, np.subtract(mean_gaps, std_gaps), np.add(mean_gaps, std_gaps),
                    color="#2ca02c", alpha=0.2, label="$\\pm 1$ std across trials")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("E4: Uniform Convergence — Generalization Gap vs Sample Size $n$")
    ax.set_xlabel("Sample Size $n$ (documents)")
    ax.set_ylabel("Generalization Gap ($Score_{train} - Score_{held}$)")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    out_file = CHARTS_DIR / "figure5_e4_generalization_gap.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_e5_nfl():
    e5_path = RESULTS_DIR / "e5" / "e5_results.json"
    if not e5_path.exists():
        print(f"Skipping E5 plot: {e5_path} not found.")
        return

    data = json.loads(e5_path.read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    configs = [
        ("zeroshot_gemini", ax1, "Zero-shot (Gemini Flash-Lite)", "#2ca02c"),
        ("zeroshot_qwen", ax2, "Zero-shot (Qwen3-8B)", "#d62728")
    ]

    for key, ax, title, color in configs:
        summary = data.get(key, {})
        ns = [int(k) for k in summary]
        true_means = [summary[k]["true_f1_mean"] for k in summary]
        true_stds = [summary[k]["true_f1_std"] for k in summary]
        nfl_means = [summary[k]["nfl_f1_mean"] for k in summary]
        nfl_stds = [summary[k]["nfl_f1_std"] for k in summary]

        ax.plot(ns, true_means, "o-", color=color, label="True Ground Truth $F_1$", linewidth=2.2)
        ax.fill_between(ns, np.subtract(true_means, true_stds), np.add(true_means, true_stds),
                        color=color, alpha=0.15)

        ax.plot(ns, nfl_means, "s--", color="#7f7f7f", label="NFL Permuted $F_1$ (Deranged)", linewidth=2.0)
        ax.fill_between(ns, np.subtract(nfl_means, nfl_stds), np.add(nfl_means, nfl_stds),
                        color="#7f7f7f", alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Sample Size $n$ (documents)")
        ax.set_ylim(-0.02, 0.75)
        ax.legend(framealpha=0.9, loc="upper left")

    ax1.set_ylabel("F1 Score")
    fig.suptitle("E5: No-Free-Lunch Theorem — Performance Collapse Under Label Derangements", y=1.02)
    plt.tight_layout()
    out_file = CHARTS_DIR / "figure6_e5_nfl_collapse.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("Generating publication charts...")
    plot_e1_coverage()
    plot_e1_cv()
    plot_e2_pac()
    plot_e3_vc()
    plot_e4_generalization_gap()
    plot_e5_nfl()
    print("All available figures generated successfully!")


if __name__ == "__main__":
    main()
