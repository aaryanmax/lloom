# self.md — Complete Project Reference
> **"How Many Examples Does It Take?"**  
> Sample Complexity & Capacity Analysis of LLM Concept Induction (LLooM)  
> Advanced Machine Learning — Unit II Project · Aaryan · 2026-10-19

---

## 1. The Big Picture — What This Project Is

### The Paper We Build On
We implement and empirically study **LLooM** (Lam et al., CHI 2024, arXiv:2404.12259), a system that extracts *human-readable concepts* from unstructured text using an LLM pipeline.

Classic topic models (LDA, BERTopic) output **keyword blobs** like *"women, power, female"*. LLooM instead outputs **concepts** — each defined by:
1. A **name** (e.g., "Criticism of traditional gender roles")
2. An **inclusion criterion** — a natural-language question ("Does this example critique traditional gender roles?")
3. **Representative examples**
4. A **score 0–1** for every document in the dataset

The paper showed LLooM beats BERTopic by **+17.9% coverage** on Congressional bills — but it never asked the PAC-learning question:

> **How many examples (n) does concept induction need before the induced concepts cover ground truth?**

**That is this project.** We take LLooM's architecture, re-implement it under free-tier resource constraints, and study it through five classical learning-theory experiments (E1–E5) mapped directly to Unit II of the course.

---

## 2. What We Actually Did — Step by Step

### Step 1: Build a Synthetic Corpus (Phase 1)
- Created a **160-document political corpus** with exact ground-truth concept labels *by construction* (no labeling, no API calls).
- Based on the paper's Appendix C.4 taxonomy: **10 Generic + 40 Specific politics concepts** (e.g., Generic: "Healthcare"; Specific: "Mental health", "Drug policy", "Health insurance", "Universal healthcare").
- **382 hand-written seed sentences** (8 per Specific concept) ensure zero pre-training memorization — unlike the paper's GPT-4-generated sentences.
- Documents come in **4 cells**: length (5 or 10 sentences) × prevalence (20% or 40% seed sentences) × 40 docs per cell.
- Also built **10 No-Free-Lunch (NFL) label-derangement variants** — same texts, shuffled labels with zero fixed points.

### Step 2: Re-implement Mini-LLooM (Pipeline)
Implemented the paper's 4-operator pipeline under free-tier budget:

| Operator | Paper's Model | Our Model | Reason |
|---|---|---|---|
| **Distill** | GPT-3.5-Turbo | Qwen3-8B (local, llama.cpp) | Highest volume — keep off cloud |
| **Cluster** | ada-002 + HDBSCAN | bge-small-en-v1.5 + UMAP → HDBSCAN | Local, free; UMAP-first recipe won in offline harness |
| **Synthesize** | GPT-4 | Gemini Flash-Lite (cloud) | The "smart" call; low volume |
| **Score** | GPT-3.5-Turbo | Qwen3-8B (local) | Paper's costliest step (~80% of cost) |

**Key infrastructure:** Every LLM call goes through a SQLite cache (prompt-hash keyed), so re-runs cost nothing — 2,763 entries cached. Gemini is rate-limited at 500 requests/day with RPM throttling and a failover chain.

### Step 3: Validate the Coverage Matcher
We need a way to check if an *induced* concept matches a *ground-truth* concept. We built a **batched few-shot Gemini matcher** (10 ground truths per request) that scores semantic match. We validated it on 38 hand-labeled pairs.

**Key finding:** The matcher reliably evaluates *specific* concepts (precision reasonable), but **over-matches broad generic concepts** (e.g., "Voter and Campaign Activity" incorrectly matched to "Immigration"). So **specific concept coverage is the primary signal** throughout all experiments.

### Step 4: Run Experiments E1–E5
Five experiments mapping directly to Unit II theory (see Section 5).

### Step 5: Ship Deliverables
- 6 publication-ready charts (300 DPI PNG in `lloom-lab/charts/`)
- Streamlit interactive dashboard (`lloom-lab/app.py`)
- Written report (`REPORT.md`)
- Slide deck (`SLIDES.md`, `How_Many_Examples_Deck.pptx`)
- 72 offline unit tests (CI-gated via GitHub Actions)

---

## 3. Glossary — Every Term Explained

### From the LLooM Paper

| Term | What It Means |
|---|---|
| **LLooM** | The paper's system — "LLM-powered Looping concept induction". Both the algorithm and the interactive Workbench tool. |
| **Concept induction** | The task: automatically extract *high-level, human-readable concepts* from raw text. The output of LLooM. |
| **High-level concept** | A named idea + explicit inclusion criterion ("Does this example…?") + examples + 0–1 scores. Think of it as a reusable classification rule. |
| **Operator** | A named pipeline step: Distill, Cluster, Synthesize, Seed, Score, Loop. |
| **Distill** | Compresses each document into 2–4 bullet summaries (first extracts exact quotes via Filter, then abstracts via Summarize). Reduces noise before clustering. |
| **Cluster** | Groups distilled bullets by embedding similarity (HDBSCAN). Bullets are *detached from source documents* — this enables cross-cutting concepts that span many unrelated documents. |
| **Synthesize** | GPT-4 (or in our case Gemini/Qwen) proposes a unifying concept name + criteria prompt for each cluster. The core novelty of the paper. |
| **Seed** | Optional analyst-provided keyword (e.g., "social issues") that steers the Distill/Synthesize attention. |
| **Score** | Batches every document against every concept criteria using multiple-choice (A=Strongly agree → E=Strongly disagree), mapped to 0.25 steps (A=1.0, B=0.75, C=0.5, D=0.25, E=0.0). |
| **Loop** | Re-runs Distill→Cluster→Synthesize on examples that weren't covered, to find finer or higher-level concepts. We do NOT implement this. |
| **Concept prevalence** | Fraction of documents (in a slice) that match a concept. |
| **Slice** | A user-defined subset of data (e.g., `toxicity < 0.25` or documents matching concept X). |
| **Coverage** | The fraction of ground-truth concepts recovered by the method. Our primary metric. |
| **Generic concept** | A broad umbrella theme — e.g., "Healthcare", "Immigration", "Economy". 10 in total in the taxonomy. |
| **Specific concept** | A narrow, nuanced sub-theme — e.g., "Drug policy", "Skilled Worker Priorities", "Assault Weapons Ban". 40 in total. **LLooM's edge over baselines lives here.** |
| **Mixed-initiative** | Workflow where both human and AI drive: AI proposes concepts, human edits/merges/splits/seeds. |
| **IRR / Cohen's κ** | Inter-Rater Reliability. κ=0.64 means "about as agreeable as two humans". LLooM achieves κ=0.63 — human-comparable. |
| **HDBSCAN** | Hierarchical Density-Based Spatial Clustering of Applications with Noise. Chosen because it handles irregular cluster shapes and can leave noisy points unclustered (instead of forcing every point into a cluster). |
| **BERTopic** | State-of-the-art topic modeling baseline: embed documents → UMAP → HDBSCAN → label clusters with c-TF-IDF keyword extraction. |
| **n × c matrix** | The score matrix: n documents × c concepts, with each cell holding the Score operator's 0–1 value. The LLooM Workbench visualizes this. |
| **Inductive coding** | Building codes *from* the data (qualitative analysis). LLooM automates this. **Deductive coding** = applying existing codes back to data. LLooM's Score operator automates this. |

### From Our Project

| Term | What It Means |
|---|---|
| **E1** | **Sample Complexity Grid.** 5 methods × n ∈ {10,20,40,80,160} × 5 trials = 125 cells. Primary experiment. Asks: does coverage grow with n like a PAC sample-complexity curve? |
| **E2** | **PAC Bound Anchor.** Implements a *known-answer control*: axis-parallel rectangle consistent learner vs. Mitchell's theoretical bound m ≥ (4/ε)ln(13/δ). Verifies 0 violations above the bound. |
| **E3** | **Growth Function & VC Dimension.** Simulates empirical growth function Π_H(m) for thresholds (VC=1), intervals (VC=2), rectangles (VC=4). Visualizes phase transition from 2^m exponential shattering to Sauer's polynomial envelope. |
| **E4** | **Uniform Convergence / Generalization Gap.** Measures |empirical risk on training docs − risk on held-out docs|. Shows gap shrinks to ~0 as n grows. |
| **E5** | **No-Free-Lunch Collapse.** Tests all methods against 10 label-derangement variants (zero fixed points). F1 collapses from 0.613 → 0.032 (94.8% drop). |
| **Grid** | The 125-cell E1 experiment (5 n-values × 5 methods × 5 trials). Each "cell" is one (method, n, trial) run. |
| **lloom_qwen** | Mini-LLooM using Qwen3-8B for *both* Distill and Synthesize. Fully local. Run on Kaggle T4 for the full 25-cell block. |
| **lloom_gemma** | Mini-LLooM using Qwen Distill + Gemini/Gemma Synthesize. Hybrid. Had cloud timeout issues at n=160. |
| **zeroshot_gemini** | Direct: give Gemini the full 50-concept taxonomy upfront and ask it to label each document. No induction pipeline — a fundamentally different task. |
| **zeroshot_qwen** | Same as zeroshot_gemini but with local Qwen3-8B. |
| **bertopic** | Classical BERTopic pipeline on the same documents. Top-keyword concept names are passed through the same coverage matcher. Achieves ~0% specific coverage. |
| **n** | Sample size — number of documents given to the concept induction method per trial. Values: {10, 20, 40, 80, 160}. |
| **trial** | One independent run of a method at a given n. 5 trials per (method, n) cell gives variance estimates. |
| **CV** | Coefficient of Variation = σ/μ. Measures trial-to-trial variance. CV → 0 as coverage stabilizes (the "variance collapse" hallmark of uniform convergence). GO condition: CV ≤ 0.25 at n_max. |
| **Specific coverage** | Fraction of the 40 Specific ground-truth concepts that are recovered by the induced concepts. The trustworthy primary signal. |
| **Generic coverage** | Fraction of the 10 Generic ground-truth concepts recovered. Inflated by umbrella over-matching — reported but secondary. |
| **Coverage matcher** | Few-shot Gemini prompt that decides if an induced concept semantically matches a ground-truth concept. Exact-meaning standard: paraphrase=1, superset/subset=0. Batched 10 GTs per request. |
| **Label derangement** | A permutation of concept labels where ZERO documents retain their original label (σ(y) ≠ y for all documents). Used for E5 NFL test. Fixed seeds for reproducibility. |
| **PAC bound** | Probably Approximately Correct bound. For consistent learners on axis-parallel rectangles: m ≥ (4/ε)ln(13/δ) guarantees error ≤ ε with probability ≥ 1−δ. |
| **Sauer's Lemma** | Π_H(m) ≤ Σ_{i=0}^{d} C(m,i) ≤ (em/d)^d for m > d. Converts the potentially exponential growth function into a polynomial bound — enabling generalization. |
| **Growth function Π_H(m)** | Number of distinct labelings a hypothesis class H can produce on any m points. = 2^m before the VC dimension breakpoint (shattering phase); polynomial thereafter. |
| **VC dimension (d)** | The size of the largest set of points that H can shatter (label in all 2^m ways). Thresholds: d=1. Intervals: d=2. Rectangles: d=4 (via the diamond configuration). |
| **ERM** | Empirical Risk Minimization — find the hypothesis that minimizes error on the training sample. |
| **Generalization gap** | |R̂_S(h) − R_held(h)| — difference between empirical error on training set and true error on held-out set. Should shrink to 0 as n grows (uniform convergence). |
| **NFL / No Free Lunch** | Wolpert's theorem: averaged over all possible target distributions, no learning algorithm does better than random. Verified empirically via label derangements. |
| **Consistent learner** | A learner that outputs a hypothesis with zero error on the training sample. The rectangle learner in E2 is consistent by construction (tightest bounding box). |
| **Axis-parallel rectangle** | Hypothesis class h(x) = 1 iff point x falls within a rectangle aligned to the axes. VC dimension = 4. Used as the known-answer PAC control in E2. |
| **Diamond configuration** | The specific 4-point arrangement (north, south, east, west) that proves rectangles can shatter 4 points. No random 4-point set guarantees this; the diamond does. |
| **Kaggle T4** | NVIDIA Tesla T4 GPU (16 GB VRAM) on Kaggle's free-tier notebooks. Used to run lloom_qwen at full scale (25 grid cells in 45.6 minutes). |
| **Pilot** | A GO/NO-GO pre-experiment: n ∈ {10,25,50,100} × 2 trials to check curve stability (CV ≤ 0.25) before committing the full 125-cell E1 grid. First pilot: NO-GO (CV=0.75). Fixed recipe → GO (CV=0.24). |
| **UMAP** | Uniform Manifold Approximation and Projection. Dimensionality reduction before HDBSCAN. Added after the pilot NO-GO — reduced noise from 59% unclustered → 14%. |
| **SQLite cache** | Persistent key-value store keyed by hash(model + prompt). All LLM responses cached, so re-runs and dashboard rehearsals cost nothing. 2,763 entries at completion. |
| **Quota guard / ledger** | Per-model daily request counter. Hard-stops Gemini at 500 req/day. Pauses the grid cleanly; resumes next day from cache. |
| **RPM throttle** | Rate-per-minute throttle. Gemini Flash-Lite free tier: 15 RPM. Client spaces requests proactively (60/RPM × 1.15s apart) to avoid 429 errors. |
| **min_cluster_size** | HDBSCAN parameter. Set to 3 bullets. Too large → everything becomes noise; too small → hundreds of micro-clusters. |
| **n_concepts_per_cluster** | Initially 3 (produced ~95 micro-concepts per run, confusing the matcher). Changed to 1 — one unifying pattern per cluster, matching the paper's spirit. |
| **doc_id** | Document identifier format: `{length}sent_{prevalence}prev_{index}` e.g., `5sent_20prev_000`. Encodes the cell of origin in the 4-cell design. |
| **Failover chain** | Gemini model fallback: `gemini-flash-lite-latest` (500/day) → `gemini-3.1-flash-lite` (500/day) → Gemma. Ensures the grid continues across pool exhaustion. |
| **bge-small-en-v1.5** | Local embedding model from BAAI via fastembed. Replaces the paper's OpenAI ada-002. Free, runs on CPU, 384 dimensions. |
| **Qwen3-8B Q4_K_M** | 4-bit quantized version of Qwen3-8B. GGUF format, runs via llama.cpp. 4.87 GB on disk; partially offloaded to GTX 1650 with -ngl 25. |
| **llama.cpp / llama-server** | Local LLM inference engine. Exposes an OpenAI-compatible REST API on port 8123. Used for all high-volume Distill and Score calls. |
| **supervenv** | Shared Python virtual environment at `C:/Users/Admin/Projects/.supervenv/`. All dependencies installed here. |

---

## 4. The Dataset — The 4-Cell Design

```
                    Prevalence 20%             Prevalence 40%
Length 5 sents     40 docs → 1 seed sent      40 docs → 2 seed sents
Length 10 sents    40 docs → 2 seed sents     40 docs → 4 seed sents
```

- **Total: 160 documents**, 40 per cell.
- **"Prevalence"** = fraction of sentences that are concept seed sentences. 20% prevalence in a 5-sentence doc = 1 seed sentence; the rest are neutral fillers.
- **Ground truth**: each document has an exact set of `concepts.specific` and `concepts.generic` labels, determined at generation time.
- **Round-robin assignment**: Each of the 40 Specific concepts appears in exactly 6–7 documents across the corpus. Generic concepts always span multiple Specific sub-concepts.
- **Why hand-written sentences?** The paper used GPT-4 to generate sentences — risks pre-training memorization. Our hand-written sentences guarantee the model has never seen them, making ours a stricter test.
- **NFL variants**: 10 deranged label files in `data_store/synthetic_v1/nfl/`, each with fixed seeds, zero fixed points in the permutation.

---

## 5. The Five Experiments — Theory ↔ Empirical Mapping

### E1 — Sample Complexity Grid (The Headline Experiment)

**Theory:** PAC sample complexity — more examples → better coverage. The curve rises monotonically toward a plateau.

**What we measure:** 5 methods × n ∈ {10,20,40,80,160} × 5 trials = **125 runs**. Primary metric: **Specific Concept Coverage** (fraction of 40 ground-truth specific concepts recovered).

**Key results:**

| Method | Specific @ n=10 | Specific @ n=160 | Note |
|---|---|---|---|
| `lloom_qwen` | 0.01 ± 0.03 | **0.80 ± 0.00** | Classical PAC curve shape |
| `lloom_gemma` | 0.01 ± 0.03 | 0.00 ± 0.00* | Cloud timeout at n=160 |
| `zeroshot_gemini` | 0.73 ± 0.41 | 1.00 ± 0.00 | Starts high (sees full taxonomy) |
| `zeroshot_qwen` | 0.36 ± 0.13 | 0.89 ± 0.05 | Strong local model |
| `bertopic` | 0.00 ± 0.00 | 0.00 ± 0.00 | Keyword model can't match semantic criteria |

**Why zeroshot starts so high at n=10:** Zero-shot methods are given the *entire 50-concept taxonomy* upfront and just classify documents against it. They don't need to *discover* concepts from data. That's conceptually different from concept *induction*, which is what lloom_qwen does.

**The CV collapse:** `lloom_qwen` CV goes from 2.24 (n=10, nearly random) to 0.00 (n=160, perfectly stable). This is the uniform convergence signature in action.

**Why BERTopic gets 0%:** BERTopic produces keyword cluster names (e.g., "immigration, border, visa"). The semantic matcher requires the induced concept to be a *paraphrase* of the ground-truth concept. Keyword names never satisfy the exact-meaning standard.

---

### E2 — PAC Bound Control (Known-Answer Anchor)

**Theory:** Mitchell's PAC bound for consistent learners on axis-parallel rectangles:
```
m ≥ (4/ε) · ln(13/δ)
```
**What we do:** Run a consistent rectangle learner (tightest bounding box around positive examples in 2D) at m ∈ [10, 600] over 400 Monte Carlo trials per (ε, δ) setting. Measure empirical failure rate = fraction of runs where error > ε.

**Result:** **0 violations** across all (ε, δ) grids. The theoretical bound holds perfectly.

**Why this experiment?** This is a *sanity-check anchor*. Before claiming LLooM exhibits sample-complexity behavior, we verify our experimental infrastructure correctly connects to classical learning theory using a problem with a known analytical answer. If rectangles pass, the framework is trustworthy.

---

### E3 — Growth Function & VC Dimension

**Theory:**
- **VC Dimension (d):** Size of the largest set that H can shatter.
- **Growth function Π_H(m):** = 2^m for m ≤ d; breaks into polynomial for m > d.
- **Sauer's Lemma:** Π_H(m) ≤ (em/d)^d for m > d.

**What we do:** Simulate exact realizability tests for:
- **1D Thresholds** (h(x) = x > θ): VC = 1
- **1D Intervals** (h(x) = a ≤ x ≤ b): VC = 2. Can't shatter 3 points (the pattern {1,0,1} is not achievable)
- **2D Axis-Parallel Rectangles**: VC = 4. The *diamond configuration* (N, S, E, W points) is shattered; no 5-point set is.

**Result:** All textbook VC dimensions confirmed exactly (d=1, 2, 4). Growth function transition from 2^m to polynomial envelope visualized. For intervals at m=3: Π_H(3) = 7 < 2^3 = 8 — shattering breaks exactly where theory predicts.

---

### E4 — Uniform Convergence / Generalization Gap

**Theory:** With enough samples, empirical error on training data converges to true error on unseen data:
```
|R̂_S(h) − R_held(h)| → 0  as  n → ∞
```

**What we do:** Score induced concept criteria on *training* documents (empirical risk) and on a *disjoint held-out* pool. Measure the gap at n ∈ {40, 80, 160}.

**Results:**
- n=40: gap = +0.065 ± 0.082 (concepts slightly overfit training docs)
- n=80: gap = −0.0004 ± 0.032 (essentially zero — convergence achieved)
- n=160: gap near-zero, variance collapses

---

### E5 — No Free Lunch Theorem Collapse

**Theory:** Wolpert's NFL — averaged over all distributions, no learner outperforms random guessing. Remove the alignment between prior and distribution → performance collapses.

**What we do:** Generate 10 label derangements (texts unchanged, labels permuted with zero fixed points). Run zero-shot Gemini and Qwen on the deranged labels.

**Results:**

| Model | True GT F1 | Deranged F1 | Collapse |
|---|---|---|---|
| Zero-Shot Gemini (n=160) | 0.613 | 0.032 | **94.8%** |
| Zero-Shot Qwen (n=160) | 0.240 | 0.029 | **87.9%** |

**Interpretation:** LLM concept induction relies on **real-world semantic priors** aligned with true concept structure. Break that alignment → performance drops to random chance. This is exactly NFL.

---

## 6. Key Numbers to Know Cold

| Number | What it means |
|---|---|
| **160** | Total documents in synthetic corpus |
| **10** | Generic ground-truth concepts in taxonomy |
| **40** | Specific ground-truth concepts in taxonomy |
| **50** | Total concepts (10 Generic + 40 Specific) |
| **4** | Cells in the design (5/10 sentences × 20/40% prevalence) |
| **125** | Total runs in E1 grid (5 n × 5 methods × 5 trials) |
| **5** | n-values in E1: {10, 20, 40, 80, 160} |
| **0.80** | lloom_qwen specific coverage at n=160 |
| **1.00** | zeroshot_gemini specific coverage at n=160 |
| **0.00** | BERTopic specific coverage (all n) |
| **2.24 → 0.00** | CV collapse in lloom_qwen (variance result) |
| **0** | PAC bound violations in E2 |
| **1, 2, 4** | VC dimensions: thresholds, intervals, rectangles (E3) |
| **+0.065 → ~0** | Generalization gap at n=40 → n=80 (E4) |
| **94.8%** | Performance collapse in E5 NFL test (Gemini, n=160) |
| **72** | Offline unit tests passing in CI |
| **2,763** | LLM responses cached in SQLite |
| **45.6 min** | Kaggle T4 execution time for 25 lloom_qwen grid cells |
| **500/day** | Gemini Flash-Lite free tier daily quota |
| **15 RPM** | Gemini Flash-Lite rate limit |
| **$1.44** | Paper's full LLooM run cost (GPT-4 pricing, 2024) |
| **~$0** | Our project cost (free tier + local models + cache) |
| **382** | Hand-written sentences in corpus (320 seed + 62 filler) |
| **8** | Seed sentences per Specific concept |
| **6–7** | Docs each Specific concept appears in (round-robin) |
| **+17.9%** | LLooM's coverage advantage over BERTopic on Bills (from paper) |
| **κ ≈ 0.63** | LLooM's inter-rater reliability vs humans (paper result; human-human = 0.64) |

---

## 7. Architecture Decisions & Why

### Why UMAP before HDBSCAN?
Raw embeddings live in ~384 dimensions (bge-small). In high-dimensional space, distances concentrate — all points look almost equally far apart, so HDBSCAN can't find density clusters. UMAP (cosine metric, 5 dimensions) creates meaningful local structure. Validated in an **offline clustering harness** with zero API calls: UMAP-first recovered 66% of topics vs 40% raw, and reduced noise from 59% → 14%.

### Why local Qwen for Distill and Score?
These operators run on every document at every n — highest volume. At 160 docs × 5 trials, running on cloud would exhaust the 500/day Gemini quota instantly. Qwen3-8B runs locally (slow on GTX 1650 with -ngl 25 partial offload, but free).

### Why Gemini Flash-Lite for Synthesize?
Synthesize is the "smart call" — needs to propose coherent, named concepts. Runs once per cluster per run (not per document), so volume is low (~20–50 calls per n=160 run). This is where cloud model quality justifies the quota spend.

### Why n_concepts_per_cluster = 1?
Starting at 3 produced ~95 micro-concepts per n=100 run. The coverage matcher then compared 95 induced concepts against 50 ground truths, and the sheer volume caused systematic over-matching. Setting to 1 (one unifying pattern per cluster) matched the paper's spirit and dropped CV from 0.75 to 0.24 — the difference between NO-GO and GO.

### Why hand-written sentences?
1. **No memorization risk** — GPT-4's own outputs may be in its pre-training data.
2. **Label fidelity** — generator assigns labels at build time, exact by construction.
3. **Zero API cost** at dataset build time.

### Why the pilot GO/NO-GO gate?
The research question only holds if coverage-vs-n curves are *stable* across trials. Pilot checks CV ≤ 0.25 before committing 125 expensive runs. First pilot: NO-GO (CV=0.75) — root cause was clustering recipe, not the research question. Fix → GO (CV=0.24). This gate saved the project from generating 125 cells of noisy data.

### Why Kaggle for lloom_qwen?
Qwen3-8B = 4.87 GB; GTX 1650 = 4 GB VRAM → slow partial offload. Kaggle T4 = 16 GB VRAM → full offload, ~10× faster. Built `lloom_lab_kaggle.ipynb`, uploaded dataset + cache, ran 25 cells in 45.6 min, merged new cache entries locally.

---

## 8. Paper vs. Project Comparison

| Dimension | Paper (Lam et al., CHI 2024) | Our Project |
|---|---|---|
| **Research question** | Can LLMs extract human-readable concepts better than BERTopic? | How many examples does concept induction need? |
| **Evaluation lens** | HCI: helpfulness, interpretability, expert approval | CLT: PAC bounds, VC dim, UC, NFL |
| **Models** | GPT-4 (Synthesize), GPT-3.5 (Distill/Score), ada-002 | Qwen3-8B (local), Gemini Flash-Lite, bge-small |
| **Cost** | ~$1.44 per run | ~$0 (free tier + local) |
| **Dataset** | GPT-4-generated synthetic + real-world corpora | Hand-written synthetic corpus (no memorization) |
| **Coverage result (specific)** | 0.71 (LLooM) vs 0.55 (GPT-4-Turbo) | 0.80 (lloom_qwen) at n=160 |
| **Main deliverable** | The algorithm + interactive Workbench | Sample-complexity curves + theory-experiment mapping |
| **Loop operator** | Implemented | Not implemented |

---

## 9. Anticipated Demo & Viva Questions

**Q: What is LLooM and why does it matter?**  
LLooM (CHI 2024) extracts *high-level, human-readable concepts* from unstructured text using an LLM pipeline — not just keyword clusters. A concept has a name, a criteria prompt, examples, and scores for every document. LDA/BERTopic give you *"women, power, female"* (uninformative). LLooM gives you *"Dismissal of women's concerns"* — actually usable for research or policy analysis. It beat BERTopic by +17.9% coverage on Congressional bills.

---

**Q: What exactly did YOU contribute vs. just running the paper's code?**  
We did not use the paper's code. We: (1) re-implemented the pipeline from scratch under free-tier constraints, (2) designed and built a controlled synthetic corpus with exact labels, (3) designed and ran 5 experiments connecting LLooM to classical learning theory (PAC, VC, UC, NFL) — the paper has *zero* of this analysis, (4) built the UMAP-first clustering recipe via a zero-API offline harness (the paper uses raw HDBSCAN), (5) built cache-first infrastructure that makes the entire study reproducible at zero cost.

---

**Q: Why are generic and specific concepts different? Which matters more?**  
Generic = broad umbrella (10 total): "Healthcare", "Immigration", etc. Specific = nuanced sub-theme (40 total): "Drug policy", "Skilled Worker Priorities", etc. **Specific coverage is the primary signal** because: the coverage matcher over-matches generic concepts (false positives), and specific concepts are what analysts actually theorize with — where LLooM's paper result (+16% over GPT-4-Turbo) is strongest.

---

**Q: What is the grid?**  
The 125-cell E1 experiment: 5 sample sizes × 5 methods × 5 trials. Each cell is one independent run, producing a set of induced concepts that are scored against ground truth. Results form coverage-vs-n curves.

---

**Q: Why does lloom_qwen start at 0.01 while zeroshot_gemini starts at 0.73?**  
Different tasks. lloom_qwen must **discover** concepts from scratch — at n=10, there aren't enough data for HDBSCAN to form meaningful clusters. Zeroshot_gemini already has the full 50-concept taxonomy and just *classifies* documents against it. It's fundamentally easier: recognition, not induction.

---

**Q: What was the NO-GO pilot and why did it matter?**  
First pilot: 8 runs, CV=0.75 (too noisy). Root cause: 3 concepts per cluster → ~95 micro-concepts per run → the matcher over-matched everything → noise, not signal. Fix: UMAP-first clustering + 1 concept per cluster → CV=0.24 → GO. Without this gate, we'd have run 125 cells of noise.

---

**Q: What does "cache-first" mean and why is it important?**  
Every LLM call is stored in SQLite keyed by hash(model + prompt). Same call again = instant cache hit, zero API calls, zero cost. This means: (a) crashed experiments resume for free, (b) dashboard demos during presentations are free, (c) the entire study is reproducible without quota. 2,763 responses cached.

---

**Q: Why UMAP before HDBSCAN?**  
In 384 dimensions, all pairwise distances are nearly equal (curse of dimensionality) — HDBSCAN can't find meaningful density. UMAP to 5 dimensions (cosine metric) creates local structure. Validated in an offline harness: UMAP-first → noise 59%→14%, topic recovery 0.40→0.66.

---

**Q: How does the coverage matcher work? How trustworthy is it?**  
Few-shot Gemini prompt receives batches of (induced concept, ground-truth concept) pairs. Rule: paraphrase=1, superset/subset=0 (exact-meaning standard). Validated on 38 hand-labeled pairs. Reliable for specific concepts; over-matches generic ones. That's why specific coverage is the primary signal.

---

**Q: What does the NFL experiment prove?**  
That LLM concept induction works because it has **semantic priors aligned with true concept structure** — not because it found a universally good algorithm. Permute the labels → F1 drops 94.8% to random chance → exactly Wolpert's NFL theorem. Also a sanity check: if performance didn't collapse, the method wasn't using the labels at all.

---

**Q: Why rectangles in E2 if you're studying LLMs?**  
E2 is a *known-answer control*. Rectangles have an analytical PAC bound. If our framework sees 0 violations, our experiment infrastructure is correctly connected to learning theory. Then when we show similar monotonic curves for LLooM in E1, we can confidently call it sample-complexity behavior.

---

**Q: What is VC dimension and why does it matter here?**  
VC dimension (d) = the largest set a hypothesis class can shatter (label in all 2^d ways). It controls generalization: for m ≤ d, Π_H(m) = 2^m (unlimited capacity, can't generalize); for m > d, Sauer's lemma gives polynomial Π_H(m) → finite capacity → generalization possible. E3 verifies this phase transition on textbook hypothesis classes, grounding the theory before applying it to the LLM system.

---

**Q: What are the limitations of your study?**  
1. **Loop operator not implemented** — we stop after one Distill→Cluster→Synthesize pass; Loop would improve coverage further.  
2. **Small corpus** (160 docs) — necessary for free-tier, limits generalizability.  
3. **Weaker synthesizer** — Qwen3-8B < GPT-4 for concept synthesis; lloom_gemma timed out at n=160.  
4. **Generic coverage inflated** — matcher over-matches generic GTs; we report but don't trust it.  
5. **VC dimension approximation** — we estimate "effective VC dim" (shatter rate ≥ 0.5), not the exact mathematical maximum.

---

**Q: What would you do with more compute?**  
Run the Loop operator, use a stronger synthesizer (GPT-4 or Gemini Pro), extend corpus to 1,000+ documents for clearer plateau analysis, and measure the actual VC dimension of the promptable concept class.

---

## 10. File & Folder Map

```
AML-Project/
├── self.md                         ← THIS FILE — complete project reference
├── Paper.md                        ← Full breakdown of LLooM paper (all numbers, all figures)
├── REPORT.md                       ← Final written report (Unit II mapped, all 5 experiments)
├── SLIDES.md                       ← Presentation slide content & speaker notes
├── GUIDE.md                        ← Developer guide & theory-experiment mapping
├── PROJECT_STATE.md                ← Source of truth: status, journal, TODOs
├── CONTRIBUTING.md                 ← Dev workflow rules
├── How_Many_Examples_Deck.pptx     ← PowerPoint slide deck
├── images/                         ← 16 figures extracted from LLooM PDF (fig1–fig16)
└── lloom-lab/
    ├── app.py                      ← Streamlit dashboard entry point
    ├── charts/                     ← 6 publication-ready figures (PNG, 300 DPI)
    │   ├── figure1_e1_coverage_vs_n.png      ← E1 main result (coverage curves)
    │   ├── figure2_e1_variance_collapse.png  ← E1 CV collapse
    │   ├── figure3_e2_pac_bound_rectangles.png ← E2 PAC control
    │   ├── figure4_e3_growth_function_sauer.png ← E3 VC + Sauer
    │   ├── figure5_e4_generalization_gap.png ← E4 generalization
    │   └── figure6_e5_nfl_collapse.png       ← E5 NFL collapse
    ├── results/
    │   ├── e1/e1_summary.{json,md}           ← E1 aggregated results (125-cell grid)
    │   ├── e2/e2_rows.json                   ← E2 PAC bound simulation data
    │   ├── e3/e3_results.json                ← E3 VC + growth function data
    │   ├── e4/e4_rows.json                   ← E4 generalization gap data
    │   ├── e5/e5_{results,summary}.{json,md} ← E5 NFL data
    │   └── llm_cache.sqlite3                 ← 2,763 cached LLM responses
    └── src/lloom_lab/
        ├── config.py               ← All paths/models/quotas (central control)
        ├── llm/                    ← client.py, cache.py, accounting.py
        ├── pipeline/               ← distill.py, cluster.py, synthesize.py, score.py, lloom.py
        ├── data/                   ← concepts.py (taxonomy), sentences.py, synthetic.py, nfl.py
        ├── metrics/                ← coverage.py (few-shot matcher)
        ├── theory/                 ← rectangles.py (E2), vc.py (E3)
        ├── experiments/            ← sampling.py, run_lloom.py, e1.py … e5.py
        └── dashboard/              ← Streamlit multi-tab app
```

---

## 11. How to Run the Demo

```powershell
# Launch the interactive Streamlit dashboard (loads from cache — 0 API calls during demo)
streamlit run lloom-lab/app.py

# Run offline test suite (no API, no Qwen server needed)
pytest -m "not live"   # → 72 passed

# Rebuild the synthetic dataset from seeds (if data_store/ is missing)
build-dataset

# Start the local Qwen3-8B server (only needed for live inference, not for demo)
.\lloom-lab\scripts\start_qwen_server.bat
```

> [!IMPORTANT]
> The dashboard loads entirely from the SQLite cache. **No Qwen server or API key is needed** for the demo. Everything runs from 2,763 cached responses.

---

## 12. The One-Paragraph Narrative

> Topic models like LDA and BERTopic tell you which words co-occur. But analysts need to know *what idea the posts are expressing* — that's what LLooM does: it uses an LLM pipeline (Distill → Cluster → Synthesize → Score) to extract named, criteria-defined concepts from raw text. The CHI 2024 paper showed this works impressively, beating BERTopic by +17.9% coverage. We asked the question classical learning theory demands: *how many examples does it actually need?* We built a 160-document controlled corpus with exact ground-truth labels across 50 politics concepts, re-implemented LLooM on a zero-dollar budget (local Qwen3-8B + free Gemini), and ran five experiments mapping PAC learning, VC dimension, uniform convergence, and No Free Lunch directly onto the LLooM system. The headline result: LLooM's specific coverage rises monotonically from 1% at n=10 to 80% at n=160, with trial variance collapsing from CV=2.24 to 0.00 — a textbook PAC sample-complexity curve. The answer to our question is not a single number — it's a curve, and classical learning theory draws it.
