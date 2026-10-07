# lloom-lab

**How Many Examples Does It Take?** — an empirical sample-complexity and capacity
analysis of LLM concept induction, based on *LLooM: Concept Induction* (Lam et al.,
CHI 2024, [arXiv:2404.12259](https://arxiv.org/abs/2404.12259)), mapped to the
Advanced-ML Unit II syllabus (PAC learning, ERM, sample complexity, uniform
convergence, No Free Lunch, VC dimension, Sauer's lemma, growth function).

## Status

Week 1 (scaffolding) complete:
- [x] Unified LLM client — Gemini Flash-Lite (cloud, free tier) + Qwen3-8B Q4_K_M
      (local llama.cpp), SQLite response cache, request ledger with daily quota guard
- [x] Mini-LLooM pipeline: Distill → Cluster → Synthesize → Score, verified end-to-end
- [x] bertopic / umap-learn / streamlit installed into the supervenv

Phase 1 (Week 2: synthetic data) complete — **zero LLM calls**:
- [x] Paper Appendix C.4 ground truth (10 Generic / 40 Specific politics concepts)
- [x] 160-document corpus in `data_store/synthetic_v1/` (length 5/10 ×
      prevalence 20/40% × 40 docs) from 382 hand-written sentences —
      ground-truth labels exact by construction, programmatic verification passing
- [x] 10 NFL permuted-label variants (label-level derangements, fixed seeds)

```bash
python scripts/build_dataset.py          # rebuild dataset (0 LLM calls)
```

## Setup

```bash
# Python: use the global supervenv
C:/Users/Admin/Projects/.supervenv/Scripts/python.exe

# 1) Start the local model server (keep running during experiments)
scripts/start_qwen_server.bat        # port 8123, -ngl 25 for the 4GB GPU

# 2) Gemini (cloud arm): set your key once per shell, or persist it
set GEMINI_API_KEY=...               # Windows cmd
export GEMINI_API_KEY=...            # Git Bash

# 3) Smoke tests
python scripts/smoke_local.py --live # local Qwen + cache + ledger
python scripts/smoke_gemini.py       # cloud arm + quota accounting
python scripts/smoke_e2e.py --synth gemini   # full milestone run
```

## Cost model

| Operator | Default model | Why |
|---|---|---|
| Distill (quotes + bullets) | Qwen3-8B local | highest volume; keep off the cloud quota |
| Cluster | bge-small-en-v1.5 + HDBSCAN (local, free) | replaces the paper's ada-002 |
| Synthesize | Gemini Flash-Lite | the "smart" call; also runs on Qwen for the local arm |
| Score | Qwen3-8B local, batched 20/call | the paper's costliest step (~80% of cost) |

Every response is cached by prompt hash → re-runs and demo rehearsals are free.
The ledger (`results/request_ledger.sqlite3`) logs per-day per-model counts and
the client hard-stops Gemini calls at `GEMINI_DAILY_LIMIT` (default 500).

## Roadmap

1. ~~Week 1 — scaffolding, dual-LLM client, e2e milestone~~ ✅
2. ~~Week 2 — synthetic dataset generator (paper §5.2: 10 Generic / 40 Specific
   concepts, politics domain, 5/10 sentences, 20/40% prevalence) + NFL variants~~ ✅
3. Week 3 — coverage metric (Appendix A.5 matcher) + validation + GO/NO-GO pilot
4. Weeks 4–5 — experiments E1–E5 (coverage curves, rectangles vs PAC bound,
   growth function / effective VC dimension, uniform convergence, NFL collapse)
5. Week 6 — analysis + charts
6. Week 7 — Streamlit dashboard, Unit-II-mapped report, deck
7. Week 8 — buffer
