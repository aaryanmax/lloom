# PROJECT_STATE — AML-Project / lloom-lab

> Single source of truth for coding agents working in this repo.
> Update `## TODO` / `## Journal` on every major plan or implementation change;
> keep everyday task lists out of this file (use session todos instead).

---

## 1. Identity

```yaml
project:      "How Many Examples Does It Take?"
repo:         C:\Users\Admin\Projects\AML-Project
code_package: lloom-lab/
domain:       Advanced ML course project (empirical study, not a system build)
course_unit:  Unit II — PAC learning, ERM, sample complexity, uniform convergence,
              No Free Lunch, VC dimension, Sauer's lemma, growth function
source_paper: LLooM: Concept Induction (Lam et al., CHI 2024, arXiv:2404.12259)
              local copy: "Concept Induction-Analyzing Unstructured Text with High-Level.pdf"
paper_notes:  Paper.md  (+ extracted figures in images/)
start_date:   2026-10-05
duration:     ~8 weeks (Week 1 done; Week 8 is buffer)
deliverables: code repo + matplotlib charts + Streamlit live demo + written report
              explicitly mapped to Unit II + presentation deck
deadline_pressure: LOW (deadline is long; no hosting required)
```

### Core research question

How many documents (n) does LLM-based concept induction (LLooM) need before the
induced concepts cover ground truth? Compare coverage-vs-n against a known PAC
sample-complexity bound (axis-parallel rectangles control), estimate the growth
function / effective VC dimension of the promptable concept class, demonstrate
uniform-convergence behavior, and show NFL collapse on permuted-label data.

### Chosen project idea (decision 2026-10-05)

Idea 2 "How Many Examples Does It Take?" over Idea 1 (version spaces) and
Idea 3 (RL analyst). Rationale: most technically impressive, deadline allows
1–2 weeks of free-tier quota accumulation (~2,000–4,000 requests). Fallback if
the Week-3 GO/NO-GO pilot is too noisy: revert to Idea 1.

---

## 2. Tech stack

```yaml
python:        C:/Users/Admin/Projects/.supervenv/Scripts/python.exe   # shared supervenv, use uv/pip here
llm_cloud:     Gemini Flash-Lite  (gemini-flash-lite-latest, free tier, 500 req/day)
               key in .env at repo root → auto-loaded by lloom_lab/config.py
llm_local:     Qwen3-8B Q4_K_M via llama.cpp llama-server
               exe:  C:\Users\Admin\Projects\LocalAi\llama.cpp\llama-server.exe
               gguf: C:\Users\Admin\Projects\LocalAi\models\qwen3-8b-q4_K_M.gguf
               endpoint: http://127.0.0.1:8123/v1 (OpenAI-compatible), ctx 8192
gpu:           NVIDIA GTX 1650, 4 GB VRAM; model file is 4.87 GB → partial offload
               -ngl 25 (tunable via QWEN_NGL); generation is slow → 300 s timeout
embeddings:    fastembed bge-small-en-v1.5 (local, free) — replaces paper's ada-002
clustering:    HDBSCAN (sklearn 1.9); BERTopic as baseline (installed)
topics_stack:  bertopic, umap-learn, streamlit (installed into supervenv)
storage:       SQLite — results/llm_cache.sqlite3 (prompt-hash → response),
               results/request_ledger.sqlite3 (per-day per-model counts)
secrets:       .env (repo root) — GEMINI_API_KEY; must stay out of git/context
               (listed in .zcodeignore)
```

### Per-operator model assignment (cost model)

| Operator  | Model                          | Rationale                                        |
|-----------|--------------------------------|--------------------------------------------------|
| Distill   | Qwen3-8B local                 | highest volume; keep off cloud quota             |
| Cluster   | bge-small-en-v1.5 + HDBSCAN    | local & free                                     |
| Synthesize| Gemini Flash-Lite              | the "smart" call; Qwen runs the local-arm variant|
| Score     | Qwen3-8B local, 20 docs/call   | paper's costliest step (~80% of total cost)      |

---

## 3. Repo map

> **Layout** (2026-10-05): src-layout Python package, git-tracked, CI-gated.
> Dev workflow & rules: [CONTRIBUTING.md](CONTRIBUTING.md). Install:
> `uv pip install --python C:/Users/Admin/Projects/.supervenv/Scripts/python.exe -e "lloom-lab[dev]"`

```
AML-Project/                        ← git repo root (branch: main)
├── PROJECT_STATE.md                ← this file
├── README.md                       ← top-level entry point
├── CONTRIBUTING.md                 ← dev setup, workflow rules, LLM/quota discipline
├── .env                            ← GEMINI_API_KEY (gitignored; auto-loaded by config.py)
├── .env.example                    ← template
├── .gitignore / .zcodeignore       ← secrets + generated artifacts excluded
├── .github/workflows/ci.yml        ← ruff + offline pytest gate
├── Paper.md                        ← LLooM paper breakdown (plain-English, all numbers)
├── images/                         ← 16 figures cropped from the source PDF
├── extract_paper_figures.py        ← reusable PyMuPDF figure-cropper (caption-anchored)
└── lloom-lab/                      ← the Python package (pyproject.toml, src layout)
    ├── pyproject.toml              ← metadata, deps, ruff & pytest config, build-dataset script
    ├── README.md                   ← status, setup, cost model, roadmap
    ├── src/lloom_lab/
    │   ├── config.py               ← ALL paths/models/quota limits (+ .env auto-load)
    │   ├── llm/                    ← client.py (unified Gemini+Qwen, retry/quota guard),
    │   │                             cache.py (SQLite, prompt-hash keyed), accounting.py (ledger)
    │   ├── pipeline/               ← mini-LLooM: distill → cluster → synthesize → score; lloom.py orchestrates
    │   ├── data/                   ← PHASE 1: concepts.py (paper App. C.4 taxonomy),
    │   │                             sentences.py (320 seed + 62 filler sentences),
    │   │                             synthetic.py (generator + verification),
    │   │                             nfl.py (label-derangement variants), build.py (build-dataset entry)
    │   ├── metrics/                ← PHASE 2: coverage.py (batched few-shot matcher, §5.1)
    │   └── experiments/            ← sampling.py (deterministic trial subsets); E1–E5 drivers land here
    ├── tests/                      ← 35 offline pytest tests (CI gate); live tests marked @live
    ├── scripts/                    ← start_qwen_server.bat, smoke_{local,gemini,e2e}.py, build_dataset.py
    ├── results/                    ← *.sqlite3 gitignored; e2e_milestone.json tracked
    ├── charts/                     ← output PNGs (gitignored until final figures)
    └── data_store/synthetic_v1/    ← docs.jsonl (160), manifest.json, nfl/*.json (gitignored; reproducible via `build-dataset`)
```

---

## 4. Conventions & invariants (agents MUST respect)

1. **Cache-first**: every LLM response is cached by model+prompt hash. Never call
   an LLM in a loop without the cache/ledger client (`lloom_lab.config.make_client()`).
   Re-runs and demo rehearsals must be free.
2. **Quota guard**: client hard-stops Gemini at `GEMINI_DAILY_LIMIT` (default 500).
   Batch Score calls 20 examples per request. Distill/Score always local.
3. **Config centralization**: no hard-coded paths, model ids, or limits outside
   `lloom_lab/config.py`; use env-var overrides already defined there.
4. **Strict JSON from Qwen3**: thinking mode disabled; enforce JSON schema;
   retry-with-repair fallback. Do not use reasoning-distill models (unreliable JSON).
5. **Local hardware is the constraint**: 4 GB VRAM → keep `-ngl 25`, expect slow
   generation, design batch/overnight jobs accordingly.
6. **Never print or commit** the API key; `.env` is gitignored/context-ignored.
7. **Multi-dev workflow** (full rules in CONTRIBUTING.md): feature branches
   (`feat/*`, `exp/<experiment-id>`), ruff + offline pytest as the CI gate,
   new modules ship with fake-based tests (no network), anything needing real
   models marked `@pytest.mark.live`, PROJECT_STATE.md Journal updated on
   every major feature.

---

## 5. Status (as of 2026-10-05)

**Week 1 (scaffolding) — COMPLETE ✅**
- Unified LLM client: Gemini Flash-Lite + Qwen3-8B, shared SQLite response cache
  and request ledger with daily quota guard — verified by all three smoke tests.
- Mini-LLooM pipeline (Distill → Cluster → Synthesize → Score) verified end-to-end;
  milestone result in `lloom-lab/results/e2e_milestone.json`.
- bertopic / umap-learn / streamlit installed into supervenv.
- `.env` added with Gemini key + auto-load in `config.py` (2026-10-05).

**Phase 1 (Week 2: synthetic data + NFL variants) — COMPLETE ✅ (2026-10-05)**
- `lloom_lab/data/concepts.py`: verbatim paper Appendix C.4 taxonomy
  (10 Generic / 40 Specific politics concepts).
- `lloom_lab/data/sentences.py`: 320 hand-written seed sentences (8 per
  Specific concept) + 62 neutral fillers; no internal periods → exact
  sentence splitting.
- `lloom_lab/data/synthetic.py` + `scripts/build_dataset.py`: builds
  `data_store/synthetic_v1/docs.jsonl` — 160 docs (4 cells: length 5/10 ×
  prevalence 20/40% × 40 docs), exact ground-truth labels by construction,
  round-robin concept assignment (each Specific in 6–7 docs).
- Programmatic verification passing: sentence counts, seed-sentence counts
  (= prevalence), label consistency, concept coverage. Zero LLM calls.
- `lloom_lab/data/nfl.py`: 10 NFL permuted-label variants (label-level
  derangements, zero fixed points, fixed seeds) in `synthetic_v1/nfl/`.
- Known limitation (fine for now): filler pool of 62 sentences is reused
  ~14×/corpus; optional expansion via local Qwen later.

**No experiments run yet.** E1–E5 results and charts do not exist.

---

## 6. TODO (next work items — read this before doing anything)

### Phase 2 — coverage metric + GO/NO-GO pilot (NEXT; ~150–200 Gemini calls total)

Code already landed (0 API calls): `metrics/coverage.py` (batched few-shot
matcher, 10 ground truths/request ≈ 5 calls per scoring pass instead of 50),
`experiments/sampling.py` (deterministic subsets + ground-truth assembly),
full offline test suite.

- [ ] `experiments/run_lloom.py` — driver that runs the pipeline on a doc
      subset and persists `results/runs/run_{n}_{trial}.json` (config, docs
      used, concepts, scores, ground truth). Distill/Score on local Qwen;
      Synthesize on Gemini.
- [ ] Matcher validation: hand-label ~40 (ground truth, generated concept)
      pairs from cached pilot outputs; compute MAE of the automated matcher
      vs manual (paper's bar: MAE ≈ 0.07 on 16 trials). Near-zero API
      (re-uses cached responses).
- [ ] **GO/NO-GO pilot**: n ∈ {10, 25, 50, 100} × 2 trials = 8 runs.
      Budget: ~10–20 Synthesize calls/run (~80–160) + ~5 matcher calls/run
      (~40) → fits in one day's 500-request quota; all cached.
      Output: coverage-vs-n table + per-cell (5/10 sents × 20/40%) breakdown.
- [ ] **Decision gate**: if trial-to-trial coverage std is too noisy at
      n=100 → tighten prompts and re-pilot (cached, cheap) or fall back to
      Idea 1 (version spaces). Document the GO/NO-GO call in the Journal.

### Later phases

- [ ] **Phase 3 (Weeks 4–5)** — experiments E1–E5:
      E1 coverage-vs-n (n ∈ {10,20,40,80,160}) × methods {LLooM-Gemini,
      LLooM-Qwen, zero-shot Gemini, zero-shot Qwen, BERTopic} × 5–10 trials,
      scored separately for Generic and Specific concepts.
      E2 rectangles vs known PAC bound. E3 growth function → effective VC dim.
      E4 empirical-vs-held-out generalization gap. E5 NFL collapse (variants
      already built in `data_store/synthetic_v1/nfl/`).
- [ ] **Phase 4 (Week 6)** — analysis + publication-style charts (headline
      overlay: theory bound vs rectangles vs LLM vs BERTopic), stats, viva prep.
- [ ] **Phase 5 (Week 7)** — Streamlit dashboard, Unit-II-mapped report, deck.
- [ ] **Week 8** — buffer (reruns, polish, dry-run).

### Known risks & mitigations

- VRAM < model size → partial offload; KV quant (`-ctk/-ctv q8_0`) if needed;
  overnight batches for Distill.
- Quota exhaustion → cache-first client, batched scoring, local Distill/Score,
  scheduler stops at daily limit.
- Noisy curves → Week-3 pilot gate; raise trials only for noisiest settings.
- Qwen3 JSON failures → no thinking mode, schema enforcement, repair retry.

---

## 7. Journal (append-only; one entry per major plan / implementation event)

### 2026-10-05 — Project kickoff & Week 1 scaffolding ✅
- Read LLooM paper (arXiv 2404.12259); breakdown delivered as `Paper.md` with
  16 PDF-extracted figures in `images/` (via `extract_paper_figures.py`).
- Compared three project ideas against course syllabus; chose **Idea 2**
  ("How Many Examples Does It Take?") — sample-complexity / VC study of LLM
  concept induction, Unit II. Fallback: Idea 1 if pilot fails.
- Constraints fixed: free tier only (Gemini Flash-Lite 500 req/day), local
  GTX 1650 (4 GB VRAM) for mechanical calls; long deadline; no hosting.
- **Implemented (Week 1)**: `lloom-lab` package — unified dual-LLM client with
  SQLite response cache + request ledger + daily quota guard; mini-LLooM
  pipeline (Distill→Cluster→Synthesize→Score) verified end-to-end;
  Qwen server launch script; three smoke tests all passing.
- **Planned next**: synthetic dataset generator (paper §5.2) + NFL variants (Week 2).

### 2026-10-05 — Phase 1 implemented: synthetic dataset + NFL variants ✅
- **Phasing agreed with user** (proceed with Idea 2, minimize Gemini calls
  during build): P1 synthetic data (0 API calls) → P2 coverage metric +
  GO/NO-GO pilot (~50–80 Gemini calls) → P3 experiments E1–E5 (quota spread
  over days) → P4 analysis/charts → P5 dashboard/report/deck.
- **Implemented P1** (`lloom_lab/data/`, `scripts/build_dataset.py`):
  - Exact paper Appendix C.4 ground-truth taxonomy (verified by grepping the
    source PDF, p. 26).
  - 382 hand-written sentences (320 seed across 40 Specific concepts + 62
    fillers) so dataset construction needs **zero API calls** and ground-truth
    labels are exact by construction (stronger than the paper's manual check).
  - Documented deviation from paper §5.2: sentences are hand-written, not
    GPT-4-generated (also guarantees no pre-training memorization). Recorded
    in `manifest.json.deviations_from_paper`.
  - Generator round-robin assigns Specific concepts → each appears in 6–7 of
    the 160 docs; per-doc concept count scales with seed-sentence budget
    (1/2/4) so Generic concepts always span multiple Specifics.
  - Programmatic verification passes: lengths, prevalence, labels, balance.
  - NFL (E5): 10 label-derangement variants with fixed seeds; enforced
    zero fixed points at the label level (fixed a positional-derangement bug).
- **Decision**: defer optional expansion of the 62-sentence filler pool via
  local Qwen (only if Distill clustering looks degenerate in the pilot).
- **Planned next (P2)**: coverage metric (paper Appendix A.5 matcher) +
  manual-validation set + GO/NO-GO pilot at n ∈ {10, 25, 50, 100}, 2 trials.

### 2026-10-05 — Production-grade restructure + Phase 2 code (0 API calls) ✅
- **Restructured for multi-developer work**:
  - Package moved to src layout (`lloom-lab/src/lloom_lab/`), now a real
    installable package (`uv pip install -e "lloom-lab[dev]"`, console script
    `build-dataset`). Verified the reinstall reproduces the identical dataset.
  - `pyproject.toml`: metadata, deps, ruff (line 100, isort, per-file ignores
    for data/prompt string files), pytest config with a `live` marker.
  - **Git initialized** (`main`); `.gitignore` excludes `.env`, `*.sqlite3`,
    `data_store/`, `charts/` — datasets/results are reproducible from seeds
    and the cache. Secrets live in `.env` (gitignored) with `.env.example`.
  - **35-test offline suite** (`lloom-lab/tests/`): taxonomy, sentence-pool
    constraints, generator determinism/invariants, NFL derangements, cache +
    ledger semantics, JSON extraction, coverage matcher (fake client),
    sampling determinism. All passing; `ruff check` clean.
  - **CI** (`.github/workflows/ci.yml`): ruff + `pytest -m "not live"`.
  - `CONTRIBUTING.md` + top-level `README.md`.
  - Two real bugs caught by tests along the way: `random.Random` rejects
    tuple seeds on modern Python (sampling now uses a stable string seed),
    and `today_count` counts only non-cached calls (test corrected to match
    the quota-guard semantics, which are intentional).
- **Phase 2 code landed** (no API calls): `metrics/coverage.py` — §5.1-style
  few-shot matcher that batches 10 ground truths/request, validates model
  output (hallucinated names nulled, missing rows backfilled), and computes
  coverage with Generic/Specific splits; `experiments/sampling.py` —
  deterministic (seed, n, trial, cell) subsets + ground-truth assembly.
- **Planned next (P2 remainder)**: `experiments/run_lloom.py` driver →
  matcher validation on ~40 hand-labeled pairs → GO/NO-GO pilot
  (8 runs ≈ 150–200 Gemini calls, one day's quota) → decision gate.

### 2026-10-05 — Phase 3 started: pilot infrastructure live, API spend green-lit ✅
- User approved proceeding; API budget guidance honored (batched matcher,
  cached everything, local Qwen for all high-volume calls).
- **2026-docs verification**: google-genai 2.19.0 in use — our client already
  matches the current SDK pattern (`genai.Client(api_key=...)`,
  `client.models.generate_content(model/contents/config=GenerateContentConfig)`,
  `response_mime_type` JSON mode); `gemini-flash-lite-latest` is the current
  auto-updated alias. Free-tier per-minute throttling is aggressive in 2026 →
  `_call_gemini` now backs off (30/60/90s) and retries on 429 instead of
  aborting the run; our own daily cap stays ledger-enforced in `chat()`.
- **New: `experiments/run_lloom.py`** — the experiment driver. Per-stage
  try/except (distill failure → marker bullet, synthesize failure → skip
  cluster, score failure → non-match), incremental atomic run-record writes
  after every stage (`results/runs/pilot/nXXX_tY.json`), embedded config +
  ground truth + errors in each record. Self-healing: re-running a crashed
  run hits the prompt-hash cache for everything already done.
- **New: `experiments/pilot.py` + `scripts/run_pilot.py`** — GO/NO-GO pilot
  over n ∈ {10,25,50,100} × 2 trials with resume (completed runs skipped),
  matcher scoring, per-n mean/std/CV aggregation, and a documented verdict
  rule: GO iff specific-coverage CV at n_max ≤ 0.25 and mean ≥ 0.20.
  Summary lands in `results/pilot/pilot_summary.{json,md}`.
- Tests grown 35 → 42 (pilot aggregation/verdict pure functions). ruff clean.
- llama-server started (port 8123, -ngl 25); pilot smoke (n=10 × 1 trial) run
  to validate the full chain before the 8-run grid.
- **Planned next**: matcher validation on hand-labeled pairs (near-zero API),
  then full pilot grid → verdict → E1–E5 drivers in `experiments/`.

### 2026-10-05 (evening) — Pilot grid + theory modules + matcher validation ✅
- **Pilot grid in flight**: n010_t0 ✅ (15 concepts), n010_t1 ✅ (6),
  n025_t0 ✅ (6, resumed from cache), n025_t1+ running. First attempt died
  at 20:39 when llama-server crashed mid-run (system-level kill; both
  processes gone) — resume via cache worked exactly as designed, zero lost
  quota. Watch for repeat crashes; if they recur, drop `-ngl` or run the
  server CPU-only overnight.
- **Offline theory modules landed (E2/E3 core, 12 new tests, 57 total)**:
  - `theory/rectangles.py`: axis-parallel-rectangle consistent learner,
    true-error = area of symmetric difference, Mitchell's PAC bound
    m ≥ (4/ε)ln(13/δ), and an empirical failure-rate harness — the
    "known-answer" control for E1's LLM curves.
  - `theory/vc.py`: growth function via exact realizability tests,
    shatter-rate estimation, effective-VC-dim (rate ≥ 0.5), Sauer's-lemma
    envelope. Tests encode the textbook answers: thresholds VC=1,
    intervals VC=2, rectangles VC=4 via the diamond configuration.
    (Test lesson: random 4-point sets are NOT always shattered by
    rectangles — the diamond is; fixed the test accordingly.)
- **Matcher validation (first pass, 14 pairs, 2 runs)** — author-labeled
  with the matcher's own exact-meaning rule: **precision 0.14 on the
  matcher's own picks** — Flash-Lite systematically over-matches broad
  umbrella concepts (e.g. "Voter and Campaign Activity" ↔ "Election
  Campaigns"). Paper's GPT-3.5 matcher had MAE 0.07; ours inflates coverage.
  Pilot numbers are therefore upper bounds (fine for the noise verdict,
  which depends on variance, not absolutes).
- **Fixes applied**: matcher prompt tightened (umbrella/narrower few-shot
  negatives + "when in doubt, null"); pair extractor now interleaves hard
  negatives instead of letting positives exhaust the budget. Re-validation
  with the tightened matcher runs after the grid completes (cache makes the
  matcher re-score free; the tightened prompt costs ~2–3 calls/run).
- Harness: `scripts/validate_matcher.py --extract/--report` +
  `experiments/matcher_validation.py`; pairs + report land in
  `results/pilot/matcher_validation_{pairs,report}.json`.

### 2026-10-06 — CHECKPOINT (work paused mid-diagnosis): pilot grid complete → NO-GO; root cause found; re-pilot plan ready

**State when paused:** llama-server stopped; no background jobs; all results on disk.

- **Pilot grid COMPLETE (8/8 runs)**. Full-grid rescore with the tightened
  matcher (`results/pilot/pilot_summary.{md,json}`):
  | n | generic | specific | CV(spec) |
  |---|---|---|---|
  | 10 | 0.06 ± 0.08 | 0.09 ± 0.03 | 0.31 |
  | 25 | 0.05 ± 0.07 | 0.07 ± 0.00 | 0.03 |
  | 50 | 0.10 ± 0.00 | 0.10 ± 0.07 | 0.68 |
  | 100 | 0.60 ± 0.42 | 0.21 ± 0.16 | **0.75** |
  **Verdict: NO-GO** (CV 0.75 > 0.25 at n=100). Pre-tightening numbers were
  inflated (n=10 specific 0.33 → 0.09 after the fix), confirming the old
  matcher over-matched.
- **Matcher re-check on the SAME 14 labeled pairs** (new file:
  `results/pilot/matcher_recheck_pairs.json`): old precision 0.14 / recall 1.0
  → tightened precision 0.25 / recall 0.5, agreement 0.14 → 0.71. Still poor in
  BOTH directions (misses obvious "Taxes" ↔ "Tax and Ballot Decisions";
  over-matches narrow "Immigration" ↔ "Skilled Worker Priorities"). 14 pairs
  too small to iterate against — matcher left as-is for now.
- **ROOT CAUSE of variance is the pipeline, not the matcher**: n100_t0 induced
  ZERO healthcare/immigration/environment concepts vs 7 in n100_t1 (verified by
  regex over concept names) — the clustering stage. Amplifier: clusters are
  tiny (min_cluster_size=3 bullets) AND Synthesize asks for
  n_concepts_per_cluster=3 per cluster → ~90–95 micro-concepts from 100 docs.
  Distill produces 2–4 bullets/doc (~300 bullets at n=100).
- **Fix plan (paused here)**:
  1. Offline clustering harness written at
     `.zcode/tmp/cluster_harness.py` (promote to `scripts/` if it lands):
     scores min_cluster_size ∈ {3,4,5,6,8,10,12} with ZERO API calls, using
     cached distilled bullets; metric = topic recovery + purity via
     bullet → doc_id → doc's specific concepts majority vote, plus n_clusters.
     **Was cancelled mid-run — re-run it first on resume.**
  2. Pick best config; almost certainly drop n_concepts_per_cluster to 1
     (paper's spirit: one pattern per cluster).
  3. Re-pilot: distill is cache-hits (free); synthesize ≈ 1 call/cluster
     (~250–300 Gemini calls for the full grid; ~380 quota left today 2026-10-06
     after 119 used — or spread over days). Old run records will be overwritten
     in place by run_subset; archive to `results/runs/pilot_v1/` first if
     comparability matters.
  4. Re-generate summary/verdict; then `validate_matcher.py --extract/--report`
     on the new runs; update REPORT.md.
- Quota ledger 2026-10-06: 119 gemini fresh calls, 64 qwen.
- llama-server crash note: yesterday's mid-run crash (n025_t1) resumed from
  cache with zero lost quota; today it was started manually again
  (`start_qwen_server.bat` equivalent command) for the grid — worked fine.
- REPORT.md is now STALE (still says 4/8 runs, provisional GO) — update on
  resume together with the journal.

### 2026-10-06 (resumed) — Pilot FIXED and complete: GO verdict; clustering recipe changed; matcher validated ✅

- **Resumed from the checkpoint above.** Ran the offline harness; results
  changed the plan (the fix was the recipe, not just mcs).
- **Offline harness results** (now promoted to `scripts/cluster_harness.py`,
  ruff-clean, zero API): UMAP(cosine, n_neighbors=10, 5 comps) → HDBSCAN
  (mcs=3) dominates raw-embedding HDBSCAN on cached v1 bullets — mean topic
  recovery 0.66 vs 0.40, noise 14% vs 59%, and trial-consistent at n=100
  (recovery 0.80/0.80 vs 0.55/0.53). Larger mcs only collapses recovery
  (mcs≥5 → ≤0.38). Purity is ~0.35–0.40 in all recipes (short bullets, weak
  embeddings — accepted as the operator's ceiling for now).
- **Pipeline changes** (57 tests pass, ruff clean):
  - `pipeline/cluster.py::cluster_vectors` — UMAP-first recipe; raw HDBSCAN
    is now the fallback rung; single-cluster last resort unchanged.
  - `experiments/run_lloom.py::DEFAULT_RUN_CFG` and `pipeline/lloom.py` —
    `n_concepts_per_cluster` 3 → 1 (one unifying pattern per cluster, paper
    §3 spirit; v1's 3/cluster × ~31 tiny clusters exploded into ~95
    micro-concepts that drowned the matcher).
- v1 run records archived to `results/runs/pilot_v1/` (8 JSONs, self-contained).
- **Re-pilot (v2) — all 8 runs complete, zero errors**, new config embedded in
  every record. Concept counts sane: 5/6 (n=10), 12/13 (25), 33/28 (50),
  52/57 (100).
  | n | generic | specific | CV(spec) |
  |---|---|---|---|
  | 10 | 0.00 ± 0.00 | 0.06 ± 0.01 | 0.18 |
  | 25 | 0.25 ± 0.07 | 0.22 ± 0.06 | 0.26 |
  | 50 | 0.70 ± 0.00 | 0.35 ± 0.08 | 0.23 |
  | 100 | 0.75 ± 0.07 | 0.30 ± 0.07 | **0.24** |
  **Verdict: GO** (CV 0.24 ≤ 0.25 and mean 0.30 ≥ 0.20 at n=100).
  **Phase 2/3 decision gate PASSED — Idea 2 confirmed, Idea-1 fallback dead.**
  First real empirical signal: coverage grows with n then plateaus
  (~0.30–0.35 specific, ~0.70–0.75 generic at n=50–100).
- **Quota**: 358 fresh Gemini calls on 2026-10-06 (~140 left of 500); all
  Distill calls were cache hits; spend went to re-synthesis under the new
  clustering + matcher scoring.
- **Matcher validation on v2** (`matcher_validation_pairs.json` — 38 pairs
  across all 8 runs; `matcher_validation_report.json`): precision 0.21 (3/14),
  recall 1.0, agreement 0.71. Labels authored by the coding agent following
  the v1 exact-meaning standard (paraphrase=1; subset/superset/umbrella=0) —
  noted in the report JSON. **Breakdown: generic-GT picks 2/12 correct** (only
  "Gun Control", the narrowest generic), **specific-GT picks 1/2** (Party
  Factionalism ✓, Political Rallies ✗). The tightened v1 prompt did NOT fix
  systematic umbrella over-matching on generic GTs → generic coverage is
  inflated; specific coverage is the trustworthy signal (and is what the
  verdict rule uses). E1 design implication: handle generic GTs differently
  (stricter criteria or per-generic batching) or report specific coverage as
  primary.
- v1 matcher re-check file kept: `results/pilot/matcher_recheck_pairs.json`
  (old vs tightened prompt on the 14 original pairs).
- **Next: Phase 3 — E1–E5 drivers** (E2/E3 theory modules already landed;
  E5 variants built). E1 needs the quota spread over days at ~5 methods ×
  5 trials × n grid.

### 2026-10-06 (later) — Phase 3 kicked off: API hardening, Gemma probe, E1 driver ✅

- **User-provided API facts**: Gemini Flash-Lite free tier has a 15 RPM cap
  (breaching it stalls runs); `gemini-3.1-flash-lite` is a separate daily pool
  usable when the primary 500/day exhausts; `gemma-4-31b-it` has a 14,400/day
  pool at 30 RPM.
- **Client hardening** (`config.py`, `llm/client.py`; tests 57 → 64):
  - Proactive RPM throttle: fresh Gemini-family calls are spaced
    60/RPM × 1.15 s apart *before* sending (per model id) — no more relying
    on post-429 backoff.
  - Ordered failover chain for logical "gemini":
    gemini-flash-lite-latest (500/day) → gemini-3.1-flash-lite (500/day) →
    (optional) gemma, each pool ledger-guarded per concrete model id. Legacy
    ledger rows (label "gemini") count toward the primary pool.
  - Gemma API quirks handled: system prompt folded into user message, JSON
    response_mime_type dropped (extractor + repair retry enforce JSON).
  - New logical model "gemma" in `chat()` (own daily pool + RPM); cache keys
    stay keyed by logical label so re-routing never collides.
- **Gemma 4 31B capability probe** (`scripts/probe_gemma.py`,
  `results/probe_gemma.json`): rebuilt real Synthesize + matcher prompts from
  the pilot n=100 record, A/B vs Flash-Lite, all fresh:
  - Synthesize: **4/4 structurally valid + coherent**, but ~50–70 s/call
    (vs ~2 s Flash-Lite).
  - Matcher: unreliable — repeated 500 INTERNAL on large matcher prompts,
    and its one valid batch diverged from Flash-Lite (4 vs 6 matches).
  - **Decision: hybrid routing.** Synthesize (high volume, ~80–90% of E1
    cloud calls) → gemma (14K pool); matcher + zero-shot (precision-critical,
    low volume) → Flash-Lite chain. `DEFAULT_RUN_CFG.synth_model = "gemma"`.
- **E1 implemented** (`experiments/e1.py`, `scripts/run_e1.py`; tests 64 → 71):
  grid n ∈ {10,20,40,80,160} × methods {lloom_gemma, lloom_qwen,
  zeroshot_gemini, zeroshot_qwen, bertopic} × 5 trials, resumable cells under
  `results/runs/e1/<method>/`, QuotaExceeded pauses the grid cleanly,
  aggregation reuses pilot.summarize per method → `results/e1/e1_summary.{json,md}`.
  Zero-shot labels docs against the FULL 50-concept taxonomy (no answer leak);
  coverage = fraction of subset GTs used on ≥1 doc. BERTopic pseudo-concepts
  (top-keyword names) go through the same matcher as induced concepts.
- Quota note: primary Flash-Lite pool shows ~484/500 today (legacy rows +
  probe) — first E1 cloud calls will exercise the 3.1 failover pool.
