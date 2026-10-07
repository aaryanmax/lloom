"""Central configuration for the lloom-lab project.

All paths, endpoints, model ids, and quota limits live here so experiment
scripts never hard-code them.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../lloom-lab (src layout)
REPO_ROOT = PROJECT_ROOT.parent

# Minimal .env loader (no python-dotenv dependency): fills os.environ only for
# keys that are not already set, so real env vars still win.
_ENV_FILE = REPO_ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
        os.environ.setdefault(_key, _val)

# --- Project layout ---------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = PROJECT_ROOT / "charts"
DATA_STORE = PROJECT_ROOT / "data_store"
for _d in (RESULTS_DIR, CHARTS_DIR, DATA_STORE):
    _d.mkdir(exist_ok=True)

# --- Local LLM (llama.cpp + Qwen3-8B) ----------------------------------------
LLAMA_SERVER_EXE = Path(r"C:\Users\Admin\Projects\LocalAi\llama.cpp\llama-server.exe")
QWEN_MODEL_GGUF = Path(r"C:\Users\Admin\Projects\LocalAi\models\qwen3-8b-q4_K_M.gguf")
QWEN_PORT = int(os.environ.get("QWEN_PORT", "8123"))
QWEN_URL = f"http://127.0.0.1:{QWEN_PORT}/v1"
QWEN_MODEL_NAME = "qwen3"  # arbitrary label; llama.cpp serves whatever is loaded
QWEN_NGL = int(os.environ.get("QWEN_NGL", "25"))  # GPU layers; 8B Q4 = 4.87GB vs 4GB VRAM
QWEN_CTX = 8192
QWEN_HTTP_TIMEOUT = 300.0  # seconds; partial-offload generation is slow

# --- Cloud LLM (Gemini free tier) --------------------------------------------
# Two independent Flash-Lite pools are used first (500 req/day each, 1000/day
# total) because they are far faster than Gemma; Gemma is the last resort.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_DAILY_LIMIT = int(os.environ.get("GEMINI_DAILY_LIMIT", "500"))
# Free-tier RPM cap — the client spaces fresh Gemini calls at least
# 60/RPM * 1.15 apart BEFORE sending, so bursts never trip 429 timeouts.
GEMINI_RPM_LIMIT = int(os.environ.get("GEMINI_RPM_LIMIT", "15"))

# Failover chain: when a pool's daily cap is exhausted, the client
# automatically moves to the next entry (each id has its own separate pool).
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_DAILY_LIMIT = int(os.environ.get("GEMINI_FALLBACK_DAILY_LIMIT", "500"))

# Gemma 4 31B: big pool (14K/day) at 30 RPM — last resort after both
# Flash-Lite lite pools (2 × 500/day) are exhausted. JSON output was
# validated by the 2026-10-06 capability probe (4/4 valid on real prompts).
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31b-it")
GEMMA_DAILY_LIMIT = int(os.environ.get("GEMMA_DAILY_LIMIT", "14000"))
GEMMA_RPM_LIMIT = int(os.environ.get("GEMMA_RPM_LIMIT", "30"))
GEMINI_ENABLE_GEMMA = os.environ.get("GEMINI_ENABLE_GEMMA", "1").lower() in ("1", "true", "yes")

# --- Shared ------------------------------------------------------------------
CACHE_DB = RESULTS_DIR / "llm_cache.sqlite3"
LEDGER_DB = RESULTS_DIR / "request_ledger.sqlite3"
RUNS_DIR = RESULTS_DIR / "runs"        # one JSON per experiment run
PILOT_DIR = RESULTS_DIR / "pilot"      # pilot summaries + verdicts

SCORE_OPTIONS = ["A: Strongly agree", "B: Agree", "C: Neither agree nor disagree",
                 "D: Disagree", "E: Strongly disagree"]
SCORE_TO_VALUE = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "E": 0.0}
MATCH_THRESHOLD = 1.0  # paper's scenario setting: only "Strongly agree" counts as a match


def gemini_model_chain() -> list[tuple[str, int, int]]:
    """Ordered (model_id, daily_limit, rpm) failover chain for fresh Gemini
    calls. The first entry with quota remaining serves the request."""
    chain = [
        (GEMINI_MODEL, GEMINI_DAILY_LIMIT, GEMINI_RPM_LIMIT),
        (GEMINI_FALLBACK_MODEL, GEMINI_FALLBACK_DAILY_LIMIT, GEMINI_RPM_LIMIT),
    ]
    if GEMINI_ENABLE_GEMMA:
        chain.append((GEMMA_MODEL, GEMMA_DAILY_LIMIT, GEMMA_RPM_LIMIT))
    return chain


def make_client():
    """Build the default cached client (Gemini + Qwen with shared ledger)."""
    from lloom_lab.llm.accounting import RequestLedger
    from lloom_lab.llm.cache import ResponseCache
    from lloom_lab.llm.client import LLMClient

    return LLMClient(
        cache=ResponseCache(CACHE_DB),
        ledger=RequestLedger(LEDGER_DB),
        gemini_chain=gemini_model_chain(),
    )
