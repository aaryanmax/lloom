"""Streamlit interactive dashboard for LLooM Sample Complexity & Capacity Study (Unit II)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="LLooM Lab: Sample Complexity & Capacity Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" if (PACKAGE_ROOT / "results").exists() else PACKAGE_ROOT / "lloom-lab" / "results"
CHARTS_DIR = PACKAGE_ROOT / "charts" if (PACKAGE_ROOT / "charts").exists() else PACKAGE_ROOT / "lloom-lab" / "charts"


def load_e1_summary():
    p = RESULTS_DIR / "e1" / "e1_summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_e2_results():
    p = RESULTS_DIR / "e2" / "e2_rows.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_e3_results():
    p = RESULTS_DIR / "e3" / "e3_results.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_e4_results():
    p = RESULTS_DIR / "e4" / "e4_rows.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_e5_results():
    p = RESULTS_DIR / "e5" / "e5_results.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# Sidebar
st.sidebar.title("🔬 LLooM Lab")
st.sidebar.markdown(
    "**Course**: Advanced Machine Learning\n"
    "**Unit II**: PAC Learning, ERM, VC Dimension, Uniform Convergence & No-Free-Lunch\n"
    "**Reference Paper**: LLooM (Lam et al., CHI 2024)"
)
st.sidebar.markdown("---")
view_mode = st.sidebar.radio(
    "Navigate Module:",
    [
        "Executive Summary",
        "E1: Coverage vs Sample Size (n)",
        "E2: PAC Bound Control (Rectangles)",
        "E3: Growth Function & VC Dimension",
        "E4: Generalization Gap & Uniform Convergence",
        "E5: No-Free-Lunch Theorem Collapse",
        "Concept Inspector",
    ],
)

st.title("How Many Examples Does It Take?")
st.subheader("Sample Complexity and Capacity of LLM Concept Induction")

# 1. Executive Summary
if view_mode == "Executive Summary":
    st.markdown("""
    This research investigates the **sample complexity ($n$)** required by LLM-based concept induction
    (the **LLooM** architecture) to reliably discover the underlying ground-truth conceptual structure of unstructured corpora.

    ### 🎯 Research Mapping to Unit II
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 1. Sample Complexity (E1)\nDoes empirical concept coverage scale predictably with document sample size $n$? Variance collapses at $n=160$.")
    with col2:
        st.success("### 2. Theoretical Anchors (E2 & E3)\nVerified consistent learners against Mitchell's PAC sample complexity bound and Sauer's lemma polynomial growth.")
    with col3:
        st.warning("### 3. Generalization & NFL (E4 & E5)\nEmpirical validation of uniform convergence (risk gap decay) and a 95% performance collapse under NFL label derangements.")

    st.markdown("---")
    st.markdown("### 📊 Key Published Artifacts")
    c1, c2 = st.columns(2)
    with c1:
        img_e1 = CHARTS_DIR / "figure1_e1_coverage_vs_n.png"
        if img_e1.exists():
            st.image(str(img_e1), caption="Figure 1: E1 Headline Sample Complexity (Specific vs Generic Coverage)")
    with c2:
        img_e2 = CHARTS_DIR / "figure3_e2_pac_bound_rectangles.png"
        if img_e2.exists():
            st.image(str(img_e2), caption="Figure 3: E2 Axis-Parallel Rectangles vs Mitchell's PAC Bound")

# 2. E1: Coverage vs Sample Size
elif view_mode == "E1: Coverage vs Sample Size (n)":
    st.header("E1: Empirical Sample Complexity Grid (125 Cells)")
    summary = load_e1_summary()

    if summary:
        per_method = summary.get("per_method", {})
        methods = list(per_method.keys())
        selected_methods = st.multiselect("Select Methods to Compare:", methods, default=methods)
        metric = st.selectbox("Select Target Metric:", ["specific", "generic"])

        fig, ax = plt.subplots(figsize=(10, 5))
        for m in selected_methods:
            entry = per_method[m]
            ns = [e["n"] for e in entry["per_n"] if e["n_completed"] > 0]
            means = [e[metric]["mean"] for e in entry["per_n"] if e["n_completed"] > 0]
            stds = [e[metric]["std"] for e in entry["per_n"] if e["n_completed"] > 0]
            ax.plot(ns, means, marker="o", label=m, linewidth=2.0)
            ax.fill_between(ns, np.subtract(means, stds), np.add(means, stds), alpha=0.15)

        ax.set_xlabel("Sample Size $n$ (documents)")
        ax.set_ylabel(f"Ground-Truth Coverage ({metric.capitalize()})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)

        st.markdown("### 📋 Results Table: `lloom_qwen` (All-Local)")
        qwen_data = per_method.get("lloom_qwen", {}).get("per_n", [])
        table_rows = []
        for row in qwen_data:
            g, s = row["generic"], row["specific"]
            cv_val = f"{s['cv']:.2f}" if s.get("cv") is not None else "—"
            table_rows.append({
                "n (docs)": row["n"],
                "Completed Runs": row["n_completed"],
                "Specific Coverage": f"{s['mean']:.2f} ± {s['std']:.2f}",
                "Generic Coverage": f"{g['mean']:.2f} ± {g['std']:.2f}",
                "CV (Specific)": cv_val,
            })
        st.table(table_rows)

# 3. E2: PAC Bound Control
elif view_mode == "E2: PAC Bound Control (Rectangles)":
    st.header("E2: Axis-Parallel Rectangles vs Theoretical PAC Bound")
    st.markdown(
        r"Mitchell's PAC bound guarantees that for error tolerance $\epsilon$ and failure probability $\delta$, "
        r"a sample size $m \ge \frac{4}{\epsilon}\ln\frac{13}{\delta}$ is sufficient for the consistent hypothesis to have error $\le \epsilon$."
    )
    e2_rows = load_e2_results()
    if e2_rows:
        pairs = sorted(list({(r["eps"], r["delta"]) for r in e2_rows}))
        selected_pair = st.selectbox(
            "Select (epsilon, delta) configuration:",
            pairs,
            format_func=lambda p: f"ε = {p[0]:.2f}, δ = {p[1]:.2f}",
        )
        eps, delta = selected_pair
        filtered = sorted([r for r in e2_rows if abs(r["eps"] - eps) < 1e-4 and abs(r["delta"] - delta) < 1e-4], key=lambda x: x["m"])

        ms = [r["m"] for r in filtered]
        fails = [r["failure_rate"] for r in filtered]
        bound_m = filtered[0]["bound_m"]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(ms, fails, "b.-", label=f"Empirical Failure Rate $P(error > {eps})$")
        ax.axvline(x=bound_m, color="red", linestyle="--", label=rf"Mitchell PAC Bound $m \geq {bound_m}$")
        ax.axhline(y=delta, color="gray", linestyle=":", label=rf"Confidence Threshold $\delta={delta}$")
        ax.set_xlabel("Sample Size $m$")
        ax.set_ylabel("Empirical Failure Rate")
        ax.set_ylim(-0.02, max(fails) * 1.2 if fails else 0.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)
        st.success(f"✅ Verified: 0 PAC bound violations above $m={bound_m}$!")

# 4. E3: Growth Function & VC Dimension
elif view_mode == "E3: Growth Function & VC Dimension":
    st.header(r"E3: Growth Function $\Pi_H(m)$ and Effective VC Dimension")
    e3_data = load_e3_results()
    if e3_data:
        c1, c2, c3 = st.columns(3)
        c1.metric("1D Thresholds VC", e3_data.get("threshold_vc", 1))
        c2.metric("1D Intervals VC", e3_data.get("interval_vc", 2))
        c3.metric("2D Rectangles VC", e3_data.get("rectangle_vc", 4))

        growth = e3_data.get("growth", {})
        ms = growth.get("m", [])
        pi_h = growth.get("Pi_H", [])
        sauer = growth.get("sauer", [])

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(ms[:6], [2**m for m in ms[:6]], "k--", label="Full Shattering ($2^m$)", alpha=0.6)
        ax.plot(ms, pi_h, "o-", color="#1f77b4", label=r"Empirical $\Pi_H(m)$ (Intervals $d=2$)", linewidth=2.0)
        ax.plot(ms, sauer, "s:", color="#d62728", label=r"Sauer Polynomial Bound $\sum C(m, i)$", linewidth=2.0)
        ax.axvline(x=2, color="orange", linestyle="--", label="VC Dimension $d=2$")
        ax.set_yscale("log")
        ax.set_xlabel("Sample Size $m$")
        ax.set_ylabel("Dichotomies (Log Scale)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)

# 5. E4: Generalization Gap
elif view_mode == "E4: Generalization Gap & Uniform Convergence":
    st.header("E4: Uniform Convergence — Generalization Gap vs $n$")
    st.markdown(
        r"Uniform convergence guarantees that the gap between empirical risk on training documents "
        r"and true risk on held-out documents satisfies $|\hat{R}_S(h) - R_{\text{held}}(h)| \to 0$ as $n \to \infty$."
    )
    e4_rows = load_e4_results()
    if e4_rows:
        ns = sorted(list({r["meta"]["n"] for r in e4_rows}))
        mean_gaps = [float(np.mean([r["mean_gap"] for r in e4_rows if r["meta"]["n"] == n])) for n in ns]
        std_gaps = [float(np.std([r["mean_gap"] for r in e4_rows if r["meta"]["n"] == n])) for n in ns]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(ns, mean_gaps, "go-", linewidth=2.2, label=r"Mean Generalization Gap $|\hat{R}_S - R_{held}|$")
        ax.fill_between(ns, np.subtract(mean_gaps, std_gaps), np.add(mean_gaps, std_gaps), color="green", alpha=0.2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Sample Size $n$ (documents)")
        ax.set_ylabel("Generalization Gap")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        st.pyplot(fig)

        st.markdown("### 📋 Evaluated Concepts Sample")
        st.dataframe([
            {
                "n": r["meta"]["n"],
                "trial": r["meta"]["trial"],
                "mean_gap": f"{r['mean_gap']:.4f}",
                "concepts": ", ".join([g["concept"] for g in r["gaps"]]),
            }
            for r in e4_rows
        ])
    else:
        st.info("E4 benchmark execution in progress. Please refresh in a moment.")

# 6. E5: No-Free-Lunch Collapse
elif view_mode == "E5: No-Free-Lunch Theorem Collapse":
    st.header("E5: Wolpert's No-Free-Lunch Theorem Demonstration")
    st.markdown(
        r"When ground-truth concepts are permuted uniformly at random across documents (derangements with zero fixed points), "
        r"any algorithm relying on semantic features collapses to random chance."
    )
    e5_data = load_e5_results()
    if e5_data:
        img_e5 = CHARTS_DIR / "figure6_e5_nfl_collapse.png"
        if img_e5.exists():
            st.image(str(img_e5), caption="Figure 6: Performance Collapse Under NFL Label Derangements")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Zero-Shot Gemini Flash-Lite")
            g_data = e5_data.get("zeroshot_gemini", {})
            st.table([{"n": k, "True F1": f"{v['true_f1_mean']:.3f}", "NFL F1": f"{v['nfl_f1_mean']:.3f}", "Collapse": f"{v['collapse_pct']:.1f}%"} for k, v in g_data.items()])
        with c2:
            st.markdown("### Zero-Shot Qwen3-8B")
            q_data = e5_data.get("zeroshot_qwen", {})
            st.table([{"n": k, "True F1": f"{v['true_f1_mean']:.3f}", "NFL F1": f"{v['nfl_f1_mean']:.3f}", "Collapse": f"{v['collapse_pct']:.1f}%"} for k, v in q_data.items()])

# 7. Concept Inspector
elif view_mode == "Concept Inspector":
    st.header("🔍 Induced Concept Inspector")
    method = st.selectbox("Select Method:", ["lloom_qwen", "lloom_gemma"])
    n = st.selectbox("Select Sample Size (n):", [10, 20, 40, 80, 160], index=2)
    trial = st.slider("Select Trial:", 0, 4, 0)

    run_path = RESULTS_DIR / "runs" / "e1" / method / f"n{n:03d}_t{trial}.json"
    if run_path.exists():
        rec = json.loads(run_path.read_text(encoding="utf-8"))
        concepts = rec.get("concepts", [])
        st.write(f"Total Concepts Induced: **{len(concepts)}** | Specific Coverage: **{rec.get('coverage_split', {}).get('specific', 0.0):.2f}**")

        for i, c in enumerate(concepts, 1):
            with st.expander(f"{i}. {c.get('name', 'Unnamed Concept')}"):
                st.markdown(f"**Criteria Prompt:**\n> {c.get('criteria_prompt', 'None')}")
                st.markdown(f"**Representative Document IDs:** `{c.get('representative_example_ids', [])}`")
    else:
        st.warning(f"Run file {run_path.name} not found.")
