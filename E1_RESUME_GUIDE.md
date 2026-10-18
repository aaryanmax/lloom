# E1 Grid: Stopping, Resuming & Progress Guide

This guide explains how checkpointing works in the E1 coverage-vs-n grid, how to safely stop the process midway, check current progress, and resume execution without losing completed work.

---

## 1. How Progress and Checkpointing Work

The E1 experiment consists of **125 grid cells**:
$$\text{5 methods} \times 5 \text{ sample sizes } (n \in \{10, 20, 40, 80, 160\}) \times 5 \text{ trials } (t \in \{0, 1, 2, 3, 4\})$$

Your progress is protected at two distinct layers:

### A. Atomic Cell Checkpoints
- Every completed cell writes its output to `lloom-lab/results/runs/e1/<method>/n<NNN>_t<T>.json`.
- Writes are done atomically via temporary files (`.tmp` $\to$ `.json`). If the run is stopped mid-execution, a completed file is never corrupted.
- When `run_e1.py` starts, it automatically inspects this directory. Any cell that already has a completed JSON record is **skipped immediately** (costs 0 seconds and 0 API quota).

### B. Persistent SQLite LLM Cache
- All raw prompt $\to$ response pairs are cached in `lloom-lab/results/llm_cache.sqlite3`.
- If a cell is interrupted midway (e.g., during Distill or Synthesize steps), any individual LLM queries completed up to that point remain stored. Resuming that cell reuses the cached responses rather than calling the model again.

---

## 2. How to Safely Stop Mid-Way

You do not need any special teardown routine. Because every cell and prompt is cached to disk on completion, you can stop at any moment:

### Option A: Clean Exit via Terminal
In the terminal where `run_e1.py` is actively running, press:
```powershell
Ctrl + C
```
Python will raise a `KeyboardInterrupt` and exit. The cell currently being processed will stop, but all previously written files and SQLite cache entries remain intact.

### Option B: Stopping via PowerShell Command
If running as a background task or separate window:
```powershell
Stop-Process -Name python -ErrorAction SilentlyContinue
```

*(Note: You can leave `llama-server` running if you intend to resume soon, or close its console window to free up GPU VRAM).*

---

## 3. How to Check Current Progress

Open a PowerShell terminal in the repository root (`AML-Project`) and run:

```powershell
C:/Users/Admin/Projects/.supervenv/Scripts/python.exe -c "
import sys
sys.path.insert(0, 'lloom-lab/src')
from lloom_lab.experiments import e1

total_done = 0
for m in e1.METHODS:
    done = sum(1 for n in e1.NS for t in range(e1.N_TRIALS) if e1.run_path(m, n, t).exists())
    total_done += done
    print(f'{m:16s}: {done:2d} / 25 completed')
print(f'Total progress   : {total_done} / 125 completed ({total_done/125*100:.1f}%)')
"
```

---

## 4. How to Resume Execution

Whenever you are ready to resume, follow these two simple steps:

### Step 1: Ensure Local Qwen Server is Running
Because `lloom_qwen`, `zeroshot_qwen`, and distillation stages use the local LLM, start `llama-server` first:

```powershell
Start-Process -FilePath "C:\Users\Admin\Projects\LocalAi\llama.cpp\llama-server.exe" `
  -ArgumentList "-m C:\Users\Admin\Projects\LocalAi\models\qwen3-8b-q4_K_M.gguf -ngl 25 -c 8192 -ctk q8_0 -ctv q8_0 --threads 8 --host 127.0.0.1 --port 8123" `
  -WindowStyle Minimized
```

*Verify health in 2 seconds:*
```powershell
Invoke-RestMethod http://127.0.0.1:8123/health
# Returns: @{status=ok}
```

### Step 2: Run the E1 Resumption Script
Navigate to `lloom-lab` and launch the driver script:

```powershell
cd C:\Users\Admin\Projects\AML-Project\lloom-lab
C:\Users\Admin\Projects\.supervenv\Scripts\python.exe scripts/run_e1.py
```

### What happens automatically:
1. It logs `skip complete cell ...` for every already-finished cell.
2. It picks up immediately on the first unfinished cell.
3. If free cloud quota is hit at any point, it cleanly pauses the grid without throwing fatal unhandled errors.
4. When finished, it outputs `e1_summary.json` and `e1_summary.md` in `lloom-lab/results/e1/`.

---

## 5. Resuming Specific Methods Only (Optional)

If you wish to run or test only a single method (for example, just `bertopic` or `zeroshot_gemini`):

```powershell
# Run only bertopic across all 5 trials and sample sizes
C:\Users\Admin\Projects\.supervenv\Scripts\python.exe scripts/run_e1.py --methods bertopic

# Run only zero-shot methods
C:\Users\Admin\Projects\.supervenv\Scripts\python.exe scripts/run_e1.py --methods zeroshot_gemini zeroshot_qwen
```
