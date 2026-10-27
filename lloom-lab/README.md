# lloom-lab

**How Many Examples Does It Take?** — an empirical sample-complexity and capacity
analysis of LLM concept induction, based on *LLooM: Concept Induction* (Lam et al.,
CHI 2024, [arXiv:2404.12259](https://arxiv.org/abs/2404.12259)), mapped to the
Advanced-ML Unit II syllabus (PAC learning, ERM, sample complexity, uniform
convergence, No Free Lunch, VC dimension, Sauer's lemma, growth function).

## Status — COMPLETE ✅ (2026-10-27)

All phases shipped:

- **Infrastructure** — Unified dual-LLM client (Gemini Flash-Lite + Qwen3-8B local), SQLite
  response cache (2,763 entries), per-day quota ledger, RPM throttle, CI (ruff + pytest).
- **Data** — 160-doc synthetic corpus (`data_store/synthetic_v1/`), exact ground-truth labels
  by construction (40 Specific + 10 Generic politics concepts), 10 NFL derangement variants.
- **Experiments** — 125-cell E1 coverage-vs-n grid (5 methods × 5 trials × 5 n values),
  E2 PAC-bound rectangles (0 violations), E3 growth function / VC dimension, E4 uniform
  convergence generalization gap, E5 NFL collapse (94.8% F₁ drop).
- **Outputs** — 6 publication-quality charts in `charts/`, Streamlit dashboard (`app.py`),
  `Docs/REPORT.md`, `How_Many_Examples_Deck.pptx` (8-slide "Does LLooM Hold Up?" deck).
- **Quality** — 78/78 offline tests passing; `ruff check` clean; CI gate green.

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
3. ~~Week 3 — coverage metric (Appendix A.5 matcher) + validation + GO/NO-GO pilot~~ ✅
4. ~~Weeks 4–5 — experiments E1–E5 (coverage curves, rectangles vs PAC bound,
   growth function / effective VC dimension, uniform convergence, NFL collapse)~~ ✅
5. ~~Week 6 — analysis + publication-style figures (6 charts in `charts/`)~~ ✅
6. ~~Week 7 — Streamlit dashboard (`app.py`), Unit-II-mapped report (`REPORT.md`), deck~~ ✅
7. ~~Week 8 — codebase audit, viva polish, deck rebuilt as "Does LLooM Hold Up?" (8 slides)~~ ✅
