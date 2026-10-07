# AML-Project

**"How Many Examples Does It Take?"** — an empirical sample-complexity and
capacity analysis of LLM concept induction, based on *LLooM* (Lam et al.,
CHI 2024, [arXiv:2404.12259](https://arxiv.org/abs/2404.12259)) for an
Advanced-ML course (Unit II: PAC learning, sample complexity, VC dimension,
uniform convergence, No Free Lunch).

| Entry point | What it is |
|---|---|
| [GUIDE.md](GUIDE.md) | **Project guide** — summary, tech stack, core theory → experiments map, paper references, diagrams |
| [PROJECT_STATE.md](PROJECT_STATE.md) | **Start here** — status, tech stack, TODOs, journal (kept current for humans and coding agents) |
| [lloom-lab/](lloom-lab/) | The Python package + experiments (`pip install -e "lloom-lab[dev]"`) |
| [Paper.md](Paper.md) | Full plain-English breakdown of the LLooM paper, all numbers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, workflow rules, LLM/quota discipline |

Quick start:

```bash
uv pip install --python C:/Users/Admin/Projects/.supervenv/Scripts/python.exe -e "lloom-lab[dev]"
cp .env.example .env   # add GEMINI_API_KEY
cd lloom-lab && pytest -m "not live"
```
