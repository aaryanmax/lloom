# AML-Project

**"How Many Examples Does It Take?"** — an empirical re-implementation and
sample-complexity analysis of LLM concept induction, targeting *LLooM* (Lam et al.,
CHI 2024, [arXiv:2404.12259](https://arxiv.org/abs/2404.12259)) for an
Advanced-ML course (Unit II: PAC learning, sample complexity, VC dimension,
uniform convergence, No Free Lunch).

> **Status: ALL DELIVERABLES COMPLETE ✅** — 125-cell experiment suite, 6 charts,
> Streamlit dashboard, written report, and 8-slide presentation deck shipped (2026-10-27).

| Entry point | What it is |
|---|---|
| [Docs/GUIDE.md](Docs/GUIDE.md) | **Project guide** — summary, tech stack, theory → experiments map, paper references |
| [Docs/PROJECT_STATE.md](Docs/PROJECT_STATE.md) | **Single source of truth** — status, tech stack, complete journal |
| [Docs/REPORT.md](Docs/REPORT.md) | **Final report** — all 5 experiments, quantitative results, analysis |
| [Docs/SLIDES.md](Docs/SLIDES.md) | **Deck spec** — 8-slide "Does LLooM Hold Up?" structure and speaker notes |
| [lloom-lab/](lloom-lab/) | The Python package + experiments (`pip install -e "lloom-lab[dev]"`) |
| [Docs/Paper.md](Docs/Paper.md) | Full plain-English breakdown of the LLooM paper, all numbers |
| [Docs/CONTRIBUTING.md](Docs/CONTRIBUTING.md) | Dev setup, workflow rules, LLM/quota discipline |

Quick start:

```bash
uv pip install --python C:/Users/Admin/Projects/.supervenv/Scripts/python.exe -e "lloom-lab[dev]"
cp .env.example .env   # add GEMINI_API_KEY
cd lloom-lab && pytest -m "not live"
streamlit run lloom-lab/app.py   # interactive dashboard
```
