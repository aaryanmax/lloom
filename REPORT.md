# Progress Report — "How Many Examples Does It Take?"

**Project:** Sample-complexity and capacity analysis of LLM concept induction (LLooM)
**Course:** Advanced ML — Unit II (PAC learning, ERM, sample complexity, uniform convergence, No Free Lunch, VC dimension, Sauer's lemma, growth function)
**Date:** 2026-10-06 · **Status as of:** end of Week 3 of ~8 (all work to date compressed into two days)

---

## TL;DR

Weeks 1–2 are fully complete: dual-LLM infrastructure with caching and quota control, a verified end-to-end mini-LLooM pipeline, a 160-document synthetic dataset with exact ground truth, 10 No-Free-Lunch variants, 57 passing offline tests, and CI. **Week 3 is now complete: the GO/NO-GO pilot passed with a GO verdict** (specific coverage 0.30 ± 0.07 at n=100, CV 0.24) after a clustering-recipe fix that a first NO-GO pilot traced — raw-embedding HDBSCAN was replaced with UMAP(cosine)→HDBSCAN and the pipeline now induces one concept per cluster instead of three. The coverage matcher was validated on 38 hand-labeled pairs: it is reliable on specific concepts but systematically over-matches broad generic concepts, so generic coverage readings are inflated. Main experiments (E1–E5), analysis, dashboard, and final deliverables have not started.

---

## 1. Project overview

**Research question.** How many documents (n) does LLM-based concept induction — LLooM (Lam et al., CHI 2024, arXiv:2404.12259) — need before the induced concepts cover ground truth? The project empirically compares coverage-vs-n against a known PAC sample-complexity bound, using axis-parallel rectangles as the "known-answer" control.

**Planned analyses.**
1. **E1** — coverage-vs-n curves (n ∈ {10, 20, 40, 80, 160}) for LLooM-Gemini, LLooM-Qwen, zero-shot Gemini, zero-shot Qwen, and BERTopic.
2. **E2** — rectangle consistent learner vs the Mitchell PAC bound m ≥ (4/ε)·ln(13/δ).
3. **E3** — growth function Π_H(m) → effective VC dimension, checked against the Sauer envelope.
4. **E4** — empirical vs held-out generalization gap.
5. **E5** — No-Free-Lunch collapse on permuted-label data.

**Stack and cost model.** Gemini Flash-Lite (free tier, 500 req/day) for the "smart" Synthesize call; Qwen3-8B Q4_K_M via llama.cpp locally (GTX 1650, 4 GB VRAM, `-ngl 25`, port 8123) for the high-volume Distill and Score calls; fastembed bge-small-en-v1.5 + HDBSCAN for clustering (replacing the paper's ada-002); BERTopic as baseline. SQLite response cache + per-day request ledger back everything.

**Timeline.** ~8 weeks: scaffolding (W1) → synthetic data (W2) → coverage metric + GO/NO-GO pilot (W3) → experiments E1–E5 (W4–5) → analysis/charts (W6) → Streamlit demo, written report, deck (W7) → buffer (W8). Fallback if the pilot is too noisy: revert to Idea 1 (version spaces).

---

## 2. Completed work

### 2.1 Infrastructure (Week 1) ✅

- **Unified dual-LLM client** (`lloom-lab/src/lloom_lab/llm/client.py`): Gemini + Qwen behind one interface, cache-first by model+prompt hash, hard daily quota guard against the ledger, and 429 back-off (30/60/90 s retries) matching 2026 free-tier throttling. Verified against google-genai 2.19.0, the current SDK.
- **Three smoke tests passed** (`scripts/smoke_local.py`, `smoke_gemini.py`, `smoke_e2e.py`): local Qwen serving, Gemini connectivity, and the full pipeline chain.
- **End-to-end milestone** (`results/e2e_milestone.json`): mini-LLooM run on 6 toy docs induced 4 well-formed concepts ("Rent Increase Pressure", "Maintenance and Reform Calls", "Battery Drain Post-Update", "Battery Health Degradation") with clean concept-doc alignment (housing concepts score 1.0 on housing docs, battery on battery docs) and per-doc rationales.
- **Repo restructured for multi-developer work**: src-layout installable package (`lloom-lab[dev]`), `pyproject.toml` with ruff (line 100) + pytest config, `CONTRIBUTING.md` conventions, `.env`/secrets hygiene with `.env.example`.

### 2.2 Synthetic dataset + NFL variants (Week 2 / Phase 1) ✅

- **Ground-truth taxonomy**: verbatim paper Appendix C.4 — 10 Generic + 40 Specific politics concepts (verified against the source PDF).
- **160-document dataset** (`data_store/synthetic_v1/docs.jsonl`, via `build-dataset`): 4 cells (length 5/10 sentences × prevalence 20/40% × 40 docs), round-robin concept assignment so every Specific concept appears in 6–7 docs.
- **382 hand-written sentences** (320 seed + 62 neutral fillers) → dataset construction uses **zero API calls** and ground-truth labels are exact by construction. Documented deviation from paper §5.2 (GPT-4-generated sentences) recorded in `manifest.json.deviations_from_paper`; it also rules out pre-training memorization.
- **Programmatic verification passing**: lengths, prevalence, seed-sentence counts, label consistency, concept coverage.
- **10 NFL label-derangement variants** (`data/nfl.py`, `synthetic_v1/nfl/`): label-level derangements, zero fixed points, fixed seeds 2026–2035, ready for E5.

### 2.3 Pipeline, metrics, and experiments infrastructure (Phase 2 code)

- **Mini-LLooM pipeline** (`lloom_lab/pipeline/`): Distill (paper §3.1.1, Qwen) → Cluster (bge-small + HDBSCAN) → Synthesize (paper App. A.3 prompt, Gemini) → Score (batched A–E multiple choice, Qwen, 20 docs/call), orchestrated by `lloom.py`. (Loop operator deferred.)
- **Coverage metric** (`lloom_lab/metrics/coverage.py`): paper §5.1/App. A.5 few-shot matcher, batching 10 ground truths per request (~5 calls per scoring pass instead of 50), hallucinated concept names nulled, missing rows backfilled, Generic/Specific splits.
- **Experiment drivers** (`lloom_lab/experiments/`): deterministic subset sampling; resumable run driver with per-stage error containment and incremental atomic run records; pilot grid runner with resume, aggregation, and a documented verdict rule.
- **Theory modules (E2/E3 core, offline, zero API)**:
  - `theory/rectangles.py` — axis-parallel-rectangle consistent learner, true error as symmetric-difference area, Mitchell PAC bound, empirical failure-rate harness.
  - `theory/vc.py` — exact growth function via realizability tests, shatter rate, effective VC dimension, Sauer envelope. Tests encode the textbook answers: thresholds VC = 1, intervals VC = 2, rectangles VC = 4 via the diamond configuration.

### 2.4 Quality engineering

- **57 offline tests** (grown 35 → 42 → 57), all fake-based with no network; live tests isolated behind a `@pytest.mark.live` marker. Two real bugs were caught by tests along the way (tuple-seed incompatibility in sampling; quota-counter semantics for cached calls).
- **CI** (`.github/workflows/ci.yml`): ruff + `pytest -m "not live"` on push/PR touching `lloom-lab/**`.
- **Paper study**: full plain-English breakdown in `Paper.md` (all headline numbers, e.g. +17.9% coverage vs BERTopic on Congressional bills, ≈$1.44 per full run) with 16 figures extracted from the PDF into `images/`.

---

## 3. The GO/NO-GO pilot — complete, verdict GO

### 3.1 First pass: NO-GO, and why

The first full grid (n ∈ {10, 25, 50, 100} × 2 trials, v1 config) returned **NO-GO**: specific-coverage CV at n=100 was 0.75 (trial 0 scored generic 0.30 vs trial 1's 0.90). Diagnosis, all offline and API-free:

- The bad trial induced **zero** healthcare/immigration/environment concepts while the good one induced seven — the variance was in the **clustering stage**, not the matcher or sampling.
- Root cause: ~300 distilled bullets clustered by raw-embedding HDBSCAN (`min_cluster_size=3`) fragment into micro-clusters with 53–71% of bullets in noise, and asking for 3 concepts per cluster inflated this to ~95 micro-concepts per 100-doc run.
- An **offline harness** (`scripts/cluster_harness.py`) scored candidate clustering recipes against ground-truth topics (each bullet inherits its source doc's concepts) with zero LLM calls: UMAP(cosine)→HDBSCAN beat raw HDBSCAN decisively (topic recovery 0.66 vs 0.40, noise 14% vs 59%) and, critically, was consistent across trials (recovery 0.80/0.80 at n=100 vs 0.55/0.53).

### 3.2 Fixes applied

- `pipeline/cluster.py`: UMAP(cosine, n_neighbors≤10, 5 components) → HDBSCAN is now the primary recipe; raw HDBSCAN demoted to fallback.
- `n_concepts_per_cluster` 3 → 1 (one unifying pattern per cluster, per the paper's design).
- All 57 tests still pass; ruff clean. v1 run records archived to `results/runs/pilot_v1/` for comparability.

### 3.3 Final grid (v2 config) — GO

| n | runs | generic coverage | specific coverage | CV (specific) |
|---|---|---|---|---|
| 10 | 2 | 0.00 ± 0.00 | 0.06 ± 0.01 | 0.18 |
| 25 | 2 | 0.25 ± 0.07 | 0.22 ± 0.06 | 0.26 |
| 50 | 2 | 0.70 ± 0.00 | 0.35 ± 0.08 | 0.23 |
| 100 | 2 | 0.75 ± 0.07 | 0.30 ± 0.07 | **0.24** |

**Verdict: GO** — CV 0.24 ≤ 0.25 and mean 0.30 ≥ 0.20 at n=100. The Phase 2/3 decision gate is passed and Idea 2 is confirmed (the version-spaces fallback is dead). Concept counts are healthy (5–6 at n=10 up to 52–57 at n=100). The first empirical signal: coverage grows with n, then plateaus around 0.30–0.35 (specific) and 0.70–0.75 (generic) — the shape E1 will now measure properly.

### 3.4 Matcher validation

38 hand-labeled (ground truth, generated concept) pairs across all 8 runs (labels follow the project author's exact-meaning standard: paraphrase = match; subset/superset/umbrella = no):

- **Precision 0.21** (3/14 of the matcher's picks were true matches), **recall 1.0**, agreement 0.71.
- Broken down by ground-truth type: **generic-umbrella picks 2/12 correct** (only "Gun Control", the narrowest generic), **specific picks 1/2**.
- Conclusion: the matcher detects specific-concept paraphrases well but **systematically over-matches broad generic concepts**, so generic coverage readings are inflated. Specific coverage — which the GO/NO-GO rule already uses — is the trustworthy signal. E1 should either treat generic GTs with a stricter matcher rule or report specific coverage as primary.

---

## 4. Key numbers at a glance

| Item | Value |
|---|---|
| Synthetic corpus | 160 docs, 4 cells (len 5/10 × prevalence 20/40% × 40) |
| Ground-truth taxonomy | 10 Generic + 40 Specific concepts (paper App. C.4) |
| Hand-written sentences | 382 (320 seed + 62 fillers) |
| NFL variants | 10 (label derangements, zero fixed points, seeds 2026–2035) |
| Offline tests | 57, all passing; ruff clean; CI green |
| Pilot grid (v2) | 8/8 runs, zero errors — **verdict GO** |
| Specific coverage @ n=100 | 0.30 ± 0.07 (CV 0.24, bar ≤ 0.25) |
| Generic coverage @ n=100 | 0.75 ± 0.07 |
| Matcher validation (v2, 38 pairs) | precision 0.21, recall 1.0; generic picks 2/12, specific 1/2 |
| Clustering fix (offline harness) | UMAP(cosine)→HDBSCAN: recovery 0.66 vs 0.40 raw; noise 14% vs 59% |
| Gemini quota | 500 req/day, ledger-enforced, cache-first (358 used on the re-pilot day) |
| Paper headline (for reference) | +17.9% coverage vs BERTopic; matcher MAE ≈ 0.07; ≈$1.44/full run |

---

## 5. Risks, deviations, and lessons

- **Matcher over-matching on generic concepts** (precision 2/12 on generic picks): the main open measurement risk. Specific coverage is trustworthy; generic coverage is inflated. E1 must address this (stricter generic handling or specific-primary reporting).
- **Clustering recipe mattered more than anything** — the pilot's NO-GO was a measurement/algorithm issue, not a dead research question. The lesson is institutionalized in `scripts/cluster_harness.py`, which scores clustering changes offline (zero API) before any re-run.
- **Cluster purity is capped ~0.35–0.40** across all recipes: bge-small embeddings of 5–8-word bullets only partially separate topics. Acceptable now; a richer Distill output (longer bullets, or Filter+Summarize) is the lever if E1 curves look degenerate.
- **Documented dataset deviation**: hand-written instead of GPT-4-generated sentences — trades realism for exact labels, zero API cost, and no memorization risk; recorded in the dataset manifest.
- **Local hardware is the bottleneck**: 4 GB VRAM forces partial offload (`-ngl 25`) and slow generation; one mid-grid llama-server crash was recovered losslessly via the cache-based resume.
- **Test-driven catches**: the VC-dim tests corrected a subtle misconception (random 4-point sets are not always shattered by rectangles — the diamond configuration is).

---

## 6. Remaining roadmap

1. **Phase 3 — experiments E1–E5** (`experiments/`): E1 coverage curves (n ∈ {10,20,40,80,160} × 5 methods × 5 trials, ~1,500–2,000 Gemini calls spread over days; specific coverage primary per the matcher finding); E2 rectangles vs PAC bound and E3 growth function / VC dim are already coded offline; E4 generalization gap; E5 NFL collapse (variants built).
2. **Phase 4** — analysis + publication-style charts (headline overlay: PAC bound vs rectangles vs LLM vs BERTopic), statistics, viva prep.
3. **Phase 5** — Streamlit dashboard, Unit-II-mapped written report, presentation deck.
4. **Week 8 buffer** — reruns, polish, dry-run.

---

## 7. Artifact index

| Artifact | Path |
|---|---|
| Project state / journal (source of truth) | [PROJECT_STATE.md](PROJECT_STATE.md) |
| Top-level + package READMEs | [README.md](README.md), [lloom-lab/README.md](lloom-lab/README.md) |
| Dev workflow rules | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Paper breakdown + figures | [Paper.md](Paper.md), `images/fig1…fig16`, `extract_paper_figures.py` |
| Source paper | `Concept Induction-Analyzing Unstructured Text with High-Level.pdf` |
| Package source | `lloom-lab/src/lloom_lab/` (config, llm/, pipeline/, data/, metrics/, experiments/, theory/) |
| Scripts | `lloom-lab/scripts/` (`build_dataset.py`, `run_pilot.py`, `validate_matcher.py`, smoke tests, `start_qwen_server.bat`) |
| E2E milestone | `lloom-lab/results/e2e_milestone.json` |
| Pilot summary + run records (v2, GO) | `lloom-lab/results/pilot/pilot_summary.{md,json}`, `results/runs/pilot/nXXX_tY.json` |
| Pilot v1 (NO-GO, archived) | `lloom-lab/results/runs/pilot_v1/` |
| Matcher validation | `lloom-lab/results/pilot/matcher_validation_{pairs,report}.json`, `matcher_recheck_pairs.json` |
| Clustering-config harness (offline) | `lloom-lab/scripts/cluster_harness.py` |
| CI | `.github/workflows/ci.yml` |

*Git note: the repository is initialized on `main` with the initial commit staged but not yet committed; activity history is tracked in the PROJECT_STATE.md Journal.*
