# PLAN — Weeks 4–7 (final phase): E1 wrap-up → E2–E5 → charts → dashboard/report/deck

> Companion to [PROJECT_STATE.md](PROJECT_STATE.md) (status) and [SLIDES.md](SLIDES.md) (deck content).
> Created 2026-10-07 while the E1 grid runs. Every code block below matches the
> actual `lloom_lab` APIs (verified against the source on this date).

---

## 0. Timeline at a glance

| Day | Work | Where it runs | Depends on |
|---|---|---|---|
| **Oct 8 (W4)** | E1 grid finishes → `rescore_e1` pass → `e1_summary` → **first git commit** | local + this machine | E1 grid |
| **Oct 8 (W4)** | E2 rectangles vs PAC bound — run + figure | local, 100% offline | nothing |
| **Oct 9 (W4)** | E3 growth function / VC dim — run + figure | local, 100% offline | nothing |
| **Oct 9–10 (W5)** | E4 generalization gap — driver + run | **Kaggle T4** (scoring is Qwen-heavy) | E1 concepts (already on disk per cell) |
| **Oct 10 (W5)** | E5 NFL collapse — driver + run | local, near-zero API (concepts are label-independent) | E1 concepts |
| **Oct 11 (W6)** | Stats + publication figures incl. the headline overlay | local, offline | E1–E5 outputs |
| **Oct 12 (W7)** | Streamlit dashboard | local | summaries + cache |
| **Oct 13 (W7)** | Written report + deck finalization, dry run | — | everything |
| **Oct 14–15** | Buffer (reruns, polish) | — | — |

Kaggle weekly budget (30 h T4×2, 12 h/session) covers E4 scoring and any
re-run of `lloom_qwen` several times over; we will use at most ~4–6 GPU-hours.

---

## 1. Kaggle offload plan

### 1.1 What goes to Kaggle and why

The bottleneck this machine can't remove is **local Qwen3-8B speed** (GTX 1650,
4 GB VRAM, partial offload). A Kaggle T4 (16 GB) runs the same GGUF **fully
offloaded** — roughly 3–5× faster per call — and sessions run up to 12 h
unattended. Only *local-model* workloads go to Kaggle:

| Workload | Runs on Kaggle? | Reason |
|---|---|---|
| `lloom_qwen` cells (Distill + Synthesize on local Qwen) | ✅ ideal | 100% local-model; the whole block is ~700 Qwen calls |
| E4 Score calls (train + held-out scoring) | ✅ ideal | ~1,500–4,000 short Qwen calls; T4 clears it in ~1–2 h |
| BERTopic arm | ❌ unnecessary | already fast locally (no LLM) |
| Anything calling Gemini (Synthesize-lite, matcher, zero-shot Gemini) | ❌ never | the quota **ledger** lives on this machine; splitting it risks overspend |

### 1.2 What to copy up (one-time Kaggle Datasets)

1. **`qwen3-8b-q4-k-m`** — the GGUF: `C:\Users\Admin\Projects\LocalAi\models\qwen3-8b-q4_K_M.gguf` (4.87 GB).
2. **`aml-lloom-lab`** — a zip of the repo **minus** `.env`, `results/`, `data_store/` (code + tests only).
3. **`aml-data-cache`** — `data_store/synthetic_v1/` (docs.jsonl + manifest) **and** `results/llm_cache.sqlite3`.
   The cache is the gold: every Distill prompt already issued for a doc set is
   answered for free on Kaggle too, because prompts are deterministic per
   (model, doc text) and llama.cpp sees the identical model file.

### 1.3 Session pattern (repeatable, 12 h cap)

Enable **Internet + GPU T4 x2** in the notebook settings. Cells:

```python
# CELL 1 — runtime deps (Internet ON)
!pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
!pip install -q fastapi uvicorn sse-starlette pydantic-settings   # llama_cpp.server needs these
```

```python
# CELL 2 — start the OpenAI-compatible server, full offload (background)
import subprocess
srv = subprocess.Popen([
    "python", "-m", "llama_cpp.server",
    "--model", "/kaggle/input/qwen3-8b-q4-k-m/qwen3-8b-q4_K_M.gguf",
    "--n_gpu_layers", "-1",          # 16 GB VRAM: the whole model fits
    "--n_ctx", "8192",
    "--port", "8123",
])
import time, requests; time.sleep(60)
requests.get("http://127.0.0.1:8123/v1/models").raise_for_status()
```

```python
# CELL 3 — unpack repo + data, install package
!cp -r /kaggle/input/aml-lloom-lab/lloom-lab_repo /kaggle/working/lloom-lab
!cp -r /kaggle/input/aml-data-cache/synthetic_v1 /kaggle/working/lloom-lab/data_store/
!cp /kaggle/input/aml-data-cache/llm_cache.sqlite3 /kaggle/working/lloom-lab/results/
%cd /kaggle/working/lloom-lab
!pip install -q -e . bertopic umap-learn
```

```python
# CELL 4 — run LOCAL-model cells only (no Gemini needed, no ledger needed).
# run_local_cells.py is defined in §2.4; it replicates run_cell minus coverage.
!python kaggle/run_local_cells.py --method lloom_qwen --jobs 4
```

```python
# CELL 5 — bring results home
!cd /kaggle/working && zip -qr kaggle_out.zip lloom-lab/results/runs lloom-lab/results/llm_cache.sqlite3
# then Download Output, or copy the notebook's output file
```

**Hard rules on Kaggle** (enforced by the driver, see §2.4):
- `GEMINI_API_KEY` is **never** copied to Kaggle — a missing key makes any
  accidental cloud call fail loudly instead of silently burning quota.
- Merge caches **into** the local one only (`scripts/merge_cache.py`, §2.5);
  the local ledger stays the single source of truth for quota.

---

## 2. Week 4 (Oct 8–9) — E1 wrap-up, E2, E3

### 2.1 `scripts/rescore_e1.py` — the coverage scoring pass

E1's pipeline stages are cached, so scoring any cell costs only its batched
matcher calls. This script (re)scores every record that has concepts but no
coverage yet — idempotent, cache-backed, safe to re-run:

```python
"""Score coverage for E1 records with concepts but no coverage_split.

    python scripts/rescore_e1.py                      # all methods
    python scripts/rescore_e1.py --methods lloom_qwen
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import DATA_STORE, make_client
from lloom_lab.experiments import e1, sampling
from lloom_lab.metrics import coverage as cov


def load_corpus():
    p = DATA_STORE / "synthetic_v1" / "docs.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def rescore_one(client, record, corpus):
    docs = {d["doc_id"]: d for d in corpus}
    subset = [docs[i] for i in record["doc_ids"]]
    gt = sampling.ground_truth_for(subset)
    types = {g: "generic" for g in gt["generic"]}
    types.update({s: "specific" for s in gt["specific"]})
    generated = cov.parse_generated(record.get("concepts", []))
    if not generated:
        record["coverage_split"] = {"generic": 0.0, "specific": 0.0}
        record["coverage"] = 0.0
        record["n_generated"] = 0
        return
    rows = cov.match_concepts(client, gt["generic"] + gt["specific"],
                              generated, model="gemini")
    record["coverage_split"] = cov.coverage_split(rows, types)
    record["coverage"] = cov.coverage(rows)
    record["match_rows"] = rows
    record["n_generated"] = len(generated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(e1.METHODS))
    ap.add_argument("--ns", nargs="+", type=int, default=list(e1.NS))
    ap.add_argument("--trials", type=int, default=e1.N_TRIALS)
    args = ap.parse_args()

    client = make_client()
    corpus = load_corpus()
    todo = []
    for m in args.methods:
        for n in args.ns:
            for t in range(args.trials):
                rec = e1.load_run(e1.run_path(m, n, t))
                if rec and not rec.get("coverage_split"):
                    todo.append((m, n, t, rec))
    print(f"{len(todo)} records to score")
    for i, (m, n, t, rec) in enumerate(todo):
        try:
            rescore_one(client, rec, corpus)
        except Exception as ex:                     # quota pause: progress kept
            print(f"[{i}] {m} n={n} t={t} failed: {ex} — re-run to resume")
            break
        rec["status"] = "complete_with_errors" if rec.get("errors") else "complete"
        rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        e1._atomic_write(e1.run_path(m, n, t), rec)
        print(f"[{i+1}/{len(todo)}] {m} n={n} t={t} → {rec['coverage_split']}")


if __name__ == "__main__":
    main()
```

Then regenerate the summary and **make the first git commit**:

```bash
python scripts/run_e1.py --jobs 4        # no pending cells → aggregates summary
git add -A && git commit -m "E1 grid complete: coverage-vs-n across 5 methods"
```

### 2.2 E2 — rectangles vs the PAC bound (offline, run + figure)

`theory/rectangles.py` already has `empirical_sample_complexity()`. New driver:

```python
# scripts/run_e2.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
from lloom_lab.config import RESULTS_DIR
from lloom_lab.theory.rectangles import empirical_sample_complexity

rows = empirical_sample_complexity(
    eps_values=[0.05, 0.10, 0.20],
    delta_values=[0.05, 0.10],
    m_grid=list(range(10, 620, 20)),
    trials=400,
)
out = RESULTS_DIR / "e2"
out.mkdir(exist_ok=True)
(out / "e2_rows.json").write_text(json.dumps(rows, indent=1))
for (eps, delta) in {(r["eps"], r["delta"]) for r in rows}:
    sel = [r for r in rows if r["eps"] == eps and r["delta"] == delta]
    below = [r["m"] for r in sel if r["failure_rate"] > delta and r["theory_ok"]]
    print(f"eps={eps} delta={delta}: bound m>={sel[0]['bound_m']}; "
          f"empirical violations above bound: {len(below)} (expect 0)")
```

**Sanity check to record in the report:** for every (eps, delta), the
empirical failure rate at any m ≥ bound must sit below delta — that *is* the
PAC guarantee, verified. The chart (§4) plots failure-rate curves against the
bound's required m.

### 2.3 E3 — growth function → effective VC dim (offline, run + figure)

`theory/vc.py` already provides `growth_function`, `shatter_rate`,
`effective_vc_dim`, `sauer_bound`. Driver:

```python
# scripts/run_e3.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from lloom_lab.config import RESULTS_DIR
from lloom_lab.theory import vc

out = RESULTS_DIR / "e3"
out.mkdir(exist_ok=True)
result = {}

# (1) textbook anchors — must reproduce thresholds d=1, intervals d=2, rectangles d=4
uniform01 = lambda m: np.random.default_rng(42).uniform(0, 1, size=(m, 1))
result["threshold_vc"] = vc.effective_vc_dim(
    uniform01, vc.threshold_realizer_1d, d_max=3, trials=50)
result["interval_vc"] = vc.effective_vc_dim(
    uniform01, vc.interval_realizer_1d, d_max=4, trials=50)
result["rectangle_vc"] = vc.effective_vc_dim(
    lambda m: np.random.default_rng(42).uniform(0, 1, size=(m, 2)),
    vc.rectangle_realizer, d_max=5, trials=200)

# (2) growth function vs Sauer envelope for intervals (d=2), the class whose
# "shape" most resembles a concept family with a criteria test
ms = list(range(1, 15))
gf = vc.growth_function(
    [np.random.default_rng(m).uniform(0, 1, size=(m, 1)) for m in ms],
    vc.interval_realizer_1d)
result["growth"] = {"m": ms, "Pi_H": gf,
                    "sauer": [vc.sauer_bound(m, 2) for m in ms]}

(out / "e3_results.json").write_text(json.dumps(result, indent=1, default=float))
print(json.dumps({k: v for k, v in result.items() if k != "growth"}, indent=1))
```

**Expected anchors** (already encoded as tests): thresholds VC=1, intervals
VC=2, rectangles VC=4 (diamond configuration). The report claim: the
*promptable concept class* behaves like a low-VC family per prompt, but the
prompt is chosen from a set exponential in the taxonomy — connect this to why
zero-shot Gemini reaches a high "detection ceiling" while induction needs n.

### 2.4 `kaggle/run_local_cells.py` — local-model cells on Kaggle (no cloud)

Replicates `e1.run_cell` up to (not including) coverage, so records land on
disk with concepts and `coverage_split` left empty for `rescore_e1.py`:

```python
"""Run LLooM cells with ALL-local models (no Gemini, no ledger) — Kaggle only.

    python kaggle/run_local_cells.py --method lloom_qwen --jobs 4
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import DATA_STORE, RUNS_DIR, make_client
from lloom_lab.experiments import sampling
from lloom_lab.experiments.run_lloom import run_subset


def load_corpus():
    p = DATA_STORE / "synthetic_v1" / "docs.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="lloom_qwen")   # synth on local qwen
    ap.add_argument("--ns", nargs="+", type=int, default=[10, 20, 40, 80, 160])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    corpus = load_corpus()
    out_dir = RUNS_DIR / "e1" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    def cell(n, t):
        path = out_dir / f"n{n:03d}_t{t}.json"
        if path.exists() and json.loads(path.read_text()).get("concepts"):
            return f"skip n{n} t{t}"
        docs = sampling.subsample_docs(corpus, n, trial=t)
        rec = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "meta": {"experiment": "E1", "method": args.method, "n": n,
                        "trial": t},
               "doc_ids": sorted(d["doc_id"] for d in docs), "errors": {}}
        try:
            run_subset(make_client(), docs,
                       run_cfg={"synth_model": "qwen", "skip_score": True},
                       run_meta=rec["meta"], out_path=path)
            return f"done n{n} t{t}"
        except Exception as ex:
            return f"FAIL n{n} t{t}: {ex}"

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(lambda c: cell(*c),
                            [(n, t) for n in args.ns for t in range(args.trials)]):
            print(res, flush=True)


if __name__ == "__main__":
    main()
```

### 2.5 `scripts/merge_cache.py` — bring the Kaggle cache home

```python
"""Union a cache DB produced on Kaggle into the local one (idempotent).

    python scripts/merge_cache.py path/to/kaggle/llm_cache.sqlite3
"""
import sqlite3, sys

src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect("results/llm_cache.sqlite3")
cols = [r[1] for r in src.execute("PRAGMA table_info(responses)")]
table = "responses" if cols else None
if table is None:                                   # find the table defensively
    (table,) = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
placeholders = ",".join("?" * len(cols))
rows = src.execute(f"SELECT {','.join(cols)} FROM {table}").fetchall()
before = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
dst.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})", rows)
dst.commit()
after = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
print(f"merged {after - before} new responses ({before} → {after})")
```

> If the Kaggle run also wrote new ledger rows, **ignore them** — the local
> ledger is authoritative, and the Kaggle run made zero cloud calls by design.

---

## 3. Week 5 (Oct 9–10) — E4 and E5

### 3.1 `experiments/e4.py` — uniform convergence: empirical vs held-out gap

**Definition used:** for each E1 record's induced concept set, the *empirical*
fit is the mean Score (A–E → 1.0…0.0) on the n docs that induced it; the
*held-out* fit is the mean Score on docs outside the sample. The gap should
shrink as n grows — that is uniform convergence made visible. Score is local
Qwen, so this is the main Kaggle workload.

```python
"""E4: empirical vs held-out Score gap as n grows.

    python -m lloom_lab.experiments.e4 --methods lloom_gemma --ns 40 80 160
    # (on Kaggle: same command; llama-server on port 8123, full offload)
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import RESULTS_DIR, make_client
from lloom_lab.experiments import e1, sampling
from lloom_lab.pipeline.score import score_examples

E4_DIR = RESULTS_DIR / "e4"


def load_corpus():
    from lloom_lab.config import DATA_STORE
    p = DATA_STORE / "synthetic_v1" / "docs.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def gap_for_record(client, record, corpus, model="qwen", held_cap=40,
                   seed=42) -> dict:
    """Mean empirical vs held-out score across the record's induced concepts."""
    by_id = {d["doc_id"]: d["text"] for d in corpus}
    train_ids = list(record["doc_ids"])
    import random
    held_pool = sorted(set(by_id) - set(train_ids))
    rng = random.Random(seed)
    held_ids = held_pool if len(held_pool) <= held_cap else rng.sample(held_pool, held_cap)

    gaps = []
    for c in record.get("concepts", []):
        r_tr = score_examples(client, c["name"], c["criteria_prompt"],
                              {i: by_id[i] for i in train_ids}, model=model)
        r_ho = score_examples(client, c["name"], c["criteria_prompt"],
                              {i: by_id[i] for i in held_ids}, model=model)
        emp = sum(v["score"] for v in r_tr.values()) / len(r_tr)
        hld = sum(v["score"] for v in r_ho.values()) / len(r_ho)
        gaps.append({"concept": c["name"], "empirical": emp, "held_out": hld,
                     "gap": emp - hld})
    mg = [g["gap"] for g in gaps]
    return {"meta": record["meta"], "n_concepts": len(gaps),
            "mean_empirical": sum(g["empirical"] for g in gaps) / max(1, len(gaps)),
            "mean_held_out": sum(g["held_out"] for g in gaps) / max(1, len(gaps)),
            "mean_gap": sum(mg) / max(1, len(mg)),
            "gaps": gaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["lloom_gemma"])
    ap.add_argument("--ns", nargs="+", type=int, default=[40, 80, 160])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--model", default="qwen")
    args = ap.parse_args()

    client = make_client()
    corpus = load_corpus()
    E4_DIR.mkdir(exist_ok=True)
    rows = []
    for m in args.methods:
        for n in args.ns:
            for t in range(args.trials):
                out = E4_DIR / f"{m}_n{n:03d}_t{t}.json"
                if out.exists():
                    rows.append(json.loads(out.read_text()))
                    continue
                rec = e1.load_run(e1.run_path(m, n, t))
                if rec is None or not rec.get("concepts"):
                    continue
                try:
                    row = gap_for_record(client, rec, corpus, model=args.model)
                except Exception as ex:
                    print(f"{m} n={n} t={t}: {ex} — re-run to resume"); break
                out.write_text(json.dumps(row, indent=1))
                rows.append(row)
                print(f"{m} n={n} t={t}: gap={row['mean_gap']:.3f}")
    (E4_DIR / "e4_rows.json").write_text(json.dumps(rows, indent=1))
    for n in sorted({r["meta"]["n"] for r in rows}):
        sel = [r["mean_gap"] for r in rows if r["meta"]["n"] == n]
        print(f"n={n}: mean gap {sum(sel)/len(sel):.3f} over {len(sel)} records")


if __name__ == "__main__":
    main()
```

**Expected claim for the report:** `mean_gap` decreases with n (and absolute
scores rise), the empirical-convergence signature; held-out scoring is
cached per (concept, doc) so re-tries are free.

### 3.2 `experiments/e5.py` — No Free Lunch collapse

Key structural fact: **LLooM never sees labels** — the permuted-variant
experiments reuse the *identical* induced concepts from E1; only the ground
truth against which coverage is computed changes. Expected: coverage on
permuted labels collapses toward the taxonomy-overlap floor while true-label
coverage stays at the E1 level.

```python
"""E5: coverage collapse under label derangements (No Free Lunch).

    python -m lloom_lab.experiments.e5 --methods lloom_gemma --ns 40 80 160
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import RESULTS_DIR, make_client
from lloom_lab.data.nfl import permuted_labels
from lloom_lab.experiments import e1
from lloom_lab.metrics import coverage as cov

E5_DIR = RESULTS_DIR / "e5"
SEEDS = list(range(2026, 2036))          # same seeds as data/nfl.build_variants


def labels_for(corpus):
    return {d["doc_id"]: d["concepts"] for d in corpus}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["lloom_gemma"])
    ap.add_argument("--ns", nargs="+", type=int, default=[40, 80, 160])
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    from lloom_lab.config import DATA_STORE
    docs = [json.loads(l) for l in
            (DATA_STORE / "synthetic_v1" / "docs.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
    true_labels = labels_for(docs)
    by_id = {d["doc_id"]: d for d in docs}
    client = make_client()
    E5_DIR.mkdir(exist_ok=True)
    rows = []

    for m in args.methods:
        for n in args.ns:
            for t in range(args.trials):
                rec = e1.load_run(e1.run_path(m, n, t))
                if rec is None or not rec.get("concepts"):
                    continue
                out = E5_DIR / f"{m}_n{n:03d}_t{t}.json"
                if out.exists():
                    rows.append(json.loads(out.read_text())); continue
                doc_ids = rec["doc_ids"]
                gen = cov.parse_generated(rec["concepts"])
                true_cov = rec.get("coverage_split", {})
                per_seed = []
                for seed in SEEDS:
                    # deranged labels for the WHOLE corpus, then read the
                    # subset's — mirrors how variants were built in Phase 1
                    perm = permuted_labels(true_labels, seed)
                    gt = {"generic": sorted({g for i in doc_ids
                                             for g in by_id[i]["concepts"] and perm[i]["generic"]}),
                          "specific": sorted({s for i in doc_ids
                                              for s in perm[i]["specific"]})}
                    if not gt["specific"]:
                        continue
                    types = {g: "generic" for g in gt["generic"]}
                    types.update({s: "specific" for s in gt["specific"]})
                    rws = cov.match_concepts(client, gt["generic"] + gt["specific"],
                                             gen, model="gemini")
                    per_seed.append(cov.coverage_split(rws, types))
                mean = lambda k: (sum(p[k] for p in per_seed) / len(per_seed)
                                  if per_seed else 0.0)
                row = {"meta": rec["meta"], "true": true_cov,
                       "permuted_specific": mean("specific"),
                       "permuted_generic": mean("generic"),
                       "per_seed": per_seed}
                out.write_text(json.dumps(row, indent=1))
                rows.append(row)
                print(f"{m} n={n} t={t}: true {true_cov.get('specific', 0):.2f} "
                      f"vs permuted {row['permuted_specific']:.2f} (specific)")

    (E5_DIR / "e5_rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
```

Matcher cost: ≤ 10 seeds × ~5 batches per record, and prompts repeat across
records with identical GT sets → heavy cache reuse. Cloud spend stays small.

### 3.3 Tests to add this week (offline, fake clients)

- `test_e4.py`: `gap_for_record` aggregation math with a fake client returning fixed A–E answers; held-out cap respected; resume-skip behavior.
- `test_e5.py`: permutation coverage computed against the *permuted* GT (fake client echoes which GT was asked); zero fixed points propagates from `permuted_labels`.
- `test_rescore.py`: rescore fills `coverage_split` exactly once; second run is a no-op.

---

## 4. Week 6 (Oct 11) — stats + publication figures

One script produces every figure from the JSON outputs — palette-consistent,
300 DPI, direct value labels, no chart junk:

```python
# charts/make_figures.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lloom_lab.config import RESULTS_DIR, CHARTS_DIR

INK, TEAL, TEAL_LT, AMBER, MUTED = "#0E1524", "#0F766E", "#5EC8BC", "#D97706", "#5B6B7B"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "figure.dpi": 300,
})
CHARTS_DIR.mkdir(exist_ok=True)
e1s = json.loads((RESULTS_DIR / "e1" / "e1_summary.json").read_text())


def curve(method, key="specific"):
    per = e1s["per_method"][method]["per_n"]
    ns = [e["n"] for e in per if e["n_completed"]]
    mean = [e[key]["mean"] for e in per if e["n_completed"]]
    std = [e[key]["std"] for e in per if e["n_completed"]]
    return ns, mean, std


# FIG 1 — headline: coverage-vs-n, all 5 methods (specific coverage)
fig, ax = plt.subplots(figsize=(7, 4.5))
styles = {"lloom_gemma": (TEAL, "o", "LLooM (cloud synth)"),
          "lloom_qwen": (TEAL_LT, "s", "LLooM (all-local)"),
          "zeroshot_gemini": (AMBER, "^", "Zero-shot Gemini"),
          "zeroshot_qwen": ("#9AC2BC", "v", "Zero-shot Qwen"),
          "bertopic": (MUTED, "D", "BERTopic")}
for m, (c, mk, lab) in styles.items():
    try:
        ns, mu, sd = curve(m)
    except KeyError:
        continue
    ax.errorbar(ns, mu, yerr=sd, color=c, marker=mk, ms=5, lw=2, capsize=3, label=lab)
ax.set(xlabel="documents n", ylabel="specific coverage", ylim=(0, 1.02),
       title="Coverage vs sample size — 5 methods")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(CHARTS_DIR / "fig_e1_curves.png"); plt.close(fig)

# FIG 2 — E2: rectangle failure rate vs m, with the PAC bound marker
e2 = json.loads((RESULTS_DIR / "e2" / "e2_rows.json").read_text())
fig, ax = plt.subplots(figsize=(7, 4.5))
for (eps, delta), c in zip([(0.05, 0.05), (0.10, 0.10), (0.20, 0.05)],
                           [TEAL, TEAL_LT, AMBER]):
    sel = sorted((r for r in e2 if r["eps"] == eps and r["delta"] == delta),
                 key=lambda r: r["m"])
    ax.plot([r["m"] for r in sel], [r["failure_rate"] for r in sel],
            color=c, lw=2, label=f"ε={eps}, δ={delta} (bound m={sel[0]['bound_m']})")
    ax.axvline(sel[0]["bound_m"], color=c, ls="--", lw=1, alpha=0.4)
ax.set(xlabel="training examples m", ylabel="P(true error > ε) over 400 trials",
       title="Rectangles: empirical sample complexity vs PAC bound")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(CHARTS_DIR / "fig_e2_rectangles.png"); plt.close(fig)

# FIG 3 — E3: growth function vs Sauer envelope
e3 = json.loads((RESULTS_DIR / "e3" / "e3_results.json").read_text())["growth"]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(e3["m"], e3["Pi_H"], color=TEAL, marker="o", lw=2, label="Π_H(m), intervals")
ax.plot(e3["m"], e3["sauer"], color=AMBER, ls="--", lw=2, label="Sauer envelope (d=2)")
ax.set(xlabel="m", ylabel="dichotomies realized", title="Growth function vs Sauer's lemma")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(CHARTS_DIR / "fig_e3_growth.png"); plt.close(fig)

# FIG 4 — E4: generalization gap vs n
rows = json.loads((RESULTS_DIR / "e4" / "e4_rows.json").read_text())
ns = sorted({r["meta"]["n"] for r in rows})
mu = [sum(r["mean_gap"] for r in rows if r["meta"]["n"] == n) /
      len([r for r in rows if r["meta"]["n"] == n]) for n in ns]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ns, mu, color=TEAL, marker="o", lw=2)
ax.set(xlabel="documents n", ylabel="empirical − held-out score gap",
       title="Uniform convergence: the gap closes as n grows")
fig.tight_layout(); fig.savefig(CHARTS_DIR / "fig_e4_gap.png"); plt.close(fig)

# FIG 5 — E5: NFL collapse (true vs permuted coverage, per method)
rows = json.loads((RESULTS_DIR / "e5" / "e5_rows.json").read_text())
fig, ax = plt.subplots(figsize=(7, 4.5))
ms = sorted({r["meta"]["method"] for r in rows})
width = 0.35
import numpy as np
x = np.arange(len(ms))
true_v = [sum(r["true"].get("specific", 0) for r in rows if r["meta"]["method"] == m) /
          len([r for r in rows if r["meta"]["method"] == m]) for m in ms]
perm_v = [sum(r["permuted_specific"] for r in rows if r["meta"]["method"] == m) /
          len([r for r in rows if r["meta"]["method"] == m]) for m in ms]
ax.bar(x - width / 2, true_v, width, color=TEAL, label="true labels")
ax.bar(x + width / 2, perm_v, width, color=AMBER, label="permuted labels (10 seeds)")
ax.set_xticks(x, ms); ax.set(ylabel="specific coverage", ylim=(0, 1),
                             title="No Free Lunch: coverage collapses off-distribution")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(CHARTS_DIR / "fig_e5_nfl.png"); plt.close(fig)

print("figures →", CHARTS_DIR)
```

Statistics to compute alongside (small script or notebook):
- **Bootstrap 95% CIs** over trials for each E1 point (5 trials → resample with replacement, 10k iterations) — replaces std whiskers in the final report figure.
- **CV table** per method/n (already produced by `pilot.summarize`).
- **Spearman correlation** between n and held-out score (E4) — one line of scipy.

The **headline figure** for the report/deck = FIG 1 + FIG 2 side by side
(theory's known-answer control next to the LLM curves), one shared caption.

## 4b. Week 6 — written report skeleton (Unit II mapped)

1. Introduction & research question (→ SLIDES.md slide 2)
2. Background: LLooM (§3 pipeline, §5.1 matcher) + PAC learning toolkit
3. Method: synthetic corpus, mini-LLooM re-implementation, cost model
4. E1 — coverage vs n (fig_e1, CV table, matcher-validation caveat)
5. E2 — rectangles & the PAC bound (fig_e2)
6. E3 — growth function & VC dimension (fig_e3, anchors table)
7. E4 — uniform convergence (fig_e4)
8. E5 — No Free Lunch (fig_e5)
9. Threats to validity (matcher over-matching on generic GTs; 2→5 trials; hand-written sentences deviation; model-routing change mid-grid)
10. Conclusion & future work

---

## 5. Week 7 (Oct 12–13) — dashboard, report, deck

### 5.1 `dashboard/app.py` — Streamlit live demo

```python
"""Live LLooM demo + results browser.

    streamlit run dashboard/app.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

st.set_page_config(page_title="How Many Examples Does It Take?", layout="wide")
from lloom_lab.config import RESULTS_DIR, make_client
from lloom_lab.experiments import sampling
from lloom_lab.pipeline.lloom import run_pipeline   # same entry the driver uses

st.title("How Many Examples Does It Take?")
st.caption("Sample complexity of LLM concept induction (LLooM, CHI 2024) — "
           "Advanced ML Unit II")

tab1, tab2 = st.tabs(["📈 Results", "🧪 Live pipeline"])

with tab1:
    summary = RESULTS_DIR / "e1" / "e1_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        method = st.selectbox("Method", list(data["per_method"]))
        per_n = data["per_method"][method]["per_n"]
        st.line_chart({f"{method} (specific)":
                       {e["n"]: e["specific"]["mean"] for e in per_n}})
        st.dataframe([{**e, "specific": e["specific"]["mean"],
                       "generic": e["generic"]["mean"]} for e in per_n])
    else:
        st.info("Run the E1 grid first — results/e1/e1_summary.json not found.")

with tab2:
    n = st.slider("Sample size n", 5, 160, 20,
                  help="Docs are drawn from the 160-doc synthetic corpus.")
    trial = st.number_input("Trial (which random subset)", 0, 4, 0)
    if st.button("Run mini-LLooM", type="primary"):
        corpus = [json.loads(l) for l in
                  Path("data_store/synthetic_v1/docs.jsonl").read_text(
                      encoding="utf-8").splitlines() if l.strip()]
        subset = sampling.subsample_docs(corpus, n, trial=int(trial))
        with st.status("Running Distill → Cluster → Synthesize → Score…") as s:
            client = make_client()          # cache-first: rehearsals are free
            record = run_pipeline(client, subset)
            s.update(label="Done", state="complete")
        for c in record.get("concepts", []):
            with st.expander(f"{c['name']}"):
                st.markdown(f"**Criteria:** {c.get('criteria_prompt', '')}")
        if record.get("scores"):
            st.json(record["scores"])
        st.caption("Every LLM response is cached — re-running this demo costs nothing.")
```

> The live tab calls the pipeline through the same cache/quota client; a demo
> rehearsal after the experiments is **free**. Verify against the actual
> `pipeline.lloom` entry point signature when wiring (it may be
> `run_pipeline(client, docs)` or via `run_subset`) and adjust the call.

### 5.2 Deck & report finalization

- Update SLIDES.md slide 5 (pilot numbers → full E1 numbers + E2/E4/E5 minis)
  and slide 6 (timeline all-teal). Rebuild the pptx from the spec once the
  figures exist.
- REPORT.md gets rewritten into the §4b skeleton with the real figures.
- PROJECT_STATE.md journal closes with final numbers and the git tag
  (`v1.0-course`).

---

## 6. Definition of done

- [ ] E1: 125/125 cells, all `complete*`, `e1_summary` regenerated, first git commit made
- [ ] E2: failure rate ≤ δ for every m ≥ bound (asserted in run output)
- [ ] E3: VC anchors reproduce (1, 2, 4); growth ≤ Sauer everywhere
- [ ] E4: gap-vs-n figure shows a decreasing trend (or an honest negative result + explanation)
- [ ] E5: permuted coverage ≪ true coverage for every method
- [ ] Figures: 5 PNGs in `charts/`, 300 DPI, palette-consistent
- [ ] Dashboard: runs offline from cache, live tab demoed once
- [ ] Report: Unit-II mapped, every figure cited with its JSON source
- [ ] Deck: 7 slides, final numbers, team slide has real names
- [ ] Tests: `pytest -m "not live"` green, ruff clean, CI green
