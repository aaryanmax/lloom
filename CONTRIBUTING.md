# Contributing to AML-Project / lloom-lab

## One-time setup

```bash
# Python env (shared supervenv, managed with uv)
uv pip install --python C:/Users/Admin/Projects/.supervenv/Scripts/python.exe -e "lloom-lab[dev]"

# Secrets: copy .env.example to .env (repo root) and add GEMINI_API_KEY.
# .env is auto-loaded by lloom_lab.config; it is gitignored — never commit it.

# Local model server (only needed for live LLM work)
lloom-lab/scripts/start_qwen_server.bat
```

## Daily workflow

```bash
cd lloom-lab
ruff check .            # lint (CI gate)
pytest -m "not live"    # offline tests (CI gate)
pytest -m live          # only when llama-server / network is up
```

## Rules for multi-developer work

1. **Branches**: `feat/<topic>`, `fix/<topic>`, `exp/<experiment-id>` (e.g. `exp/e1-coverage`).
   One experiment or feature per branch; keep PRs small.
2. **Commits**: imperative mood, first line ≤ 72 chars, e.g. `coverage: batch matcher requests`.
3. **Never commit**: `.env`, `*.sqlite3`, `data_store/` outputs, `charts/` — all gitignored.
   Datasets are reproducible via `build-dataset --seed 42`; results are reproducible
   via cached runs.
4. **Config centralization**: no hard-coded paths, model ids, or quota limits outside
   `src/lloom_lab/config.py` (env-var overrides live there too).
5. **LLM discipline**: all calls go through `config.make_client()` (cache-first +
   quota-guarded). Distill/Score = local Qwen; Synthesize/coverage-matcher = Gemini,
   batched. If your change adds Gemini calls, state the per-run call count in the PR.
6. **Tests**: new modules need pytest coverage with fakes (no network). Mark anything
   requiring real models with `@pytest.mark.live`.
7. **PROJECT_STATE.md is the coordination point**: when you plan or land a major
   feature, append a Journal entry there and update the TODO section. Small fixes
   don't need an entry.
8. **Style**: ruff-enforced (line length 100, isort, pyupgrade). Type hints on all
   public functions; module docstrings explaining the *why*.

## Where things live

- `lloom-lab/src/lloom_lab/` — the package (see PROJECT_STATE.md §3 for the module map)
- `lloom-lab/tests/` — offline test suite (CI runs this)
- `lloom-lab/scripts/` — operator-facing entry points
- `PROJECT_STATE.md` — project status, TODOs, decision log, journal
