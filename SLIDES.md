# Slide Content & Visual Setup — "How Many Examples Does It Take?"

Deck: 7 slides, 16:9 (13.33 × 7.5 in). Dark/light "sandwich": dark cover +
dark closing, light content slides. One visual motif repeated: a coverage
curve rising to a plateau (dots), and oversized numerals as focal points.

## Global visual system

| Role | Color | Use |
|---|---|---|
| Background (dark) | `#0E1524` deep ink | slides 1 and 7 |
| Background (light) | `#FFFFFF` | slides 2–6 |
| Primary | `#0F766E` deep teal | headers, structural shapes, chart primary line, done-nodes |
| Primary tint | `#EAF4F2` | bands, input/output boxes |
| Primary light (on dark) | `#5EC8BC` | kickers, handles on dark slides |
| Accent | `#D97706` amber | ONE key element per slide (headline stat, "now" node, key phrase) |
| Text | `#15202B` / `#5B6B7B` muted | light slides |
| Text on dark | `#E9EFF4` / `#8FA1B3` muted | slides 1, 7 |
| Hairlines | `#DDE5E9` | row separators |

Fonts: **Segoe UI** (everything) + **Consolas** (model ids, formulas, handles,
arXiv id). Sizes: titles 32pt bold (cover 50pt), section labels 17pt, body
14–15pt, captions/sources 12pt, stat numbers 30–40pt. Margins 0.7 in;
no decorative underlines, no accent stripes, no emoji.

---

## Slide 1 — Cover (dark `#0E1524`)

- Kicker (top-left, y≈0.75, 13pt bold, `#5EC8BC`, letter-spaced):
  `ADVANCED ML · UNIT II — PAC LEARNING · VC DIMENSION · NO FREE LUNCH`
- Title (y≈1.9, 50pt bold, white, 2 lines): **How Many Examples / Does It Take?**
- Subtitle (y≈4.35, 20pt, `#8FA1B3`): The sample complexity & capacity of LLM concept induction
- Hairline, then attribution block (y≈5.95):
  - `Built on LLooM` 14pt bold `#E9EFF4`
  - `Lam et al., CHI 2024 · arXiv:2404.12259` 13pt Consolas `#8FA1B3`
- **Visual (right half): coverage-curve motif.** 12 dots along a saturating
  curve from (8.05, 5.55) rising to plateau y≈2.05 (x → 12.2). Dots 1–9 teal
  `#5EC8BC` (0.13"), last 3 amber `#D97706` (0.17"). Dashed hairline at the
  plateau; tiny 12pt labels: "plateau" above the line, "documents n" below
  the curve.

## Slide 2 — The question (light)

- Title (32pt): **From topic models to concept induction**
- Two columns split by a vertical hairline at x≈6.6:
  - Left header (17pt bold, muted): *Topic models — LDA · BERTopic*
    - Discover co-occurring words, not meanings
    - Themes readable only as keyword lists
    - No notion of ground-truth concepts
  - Right header (17pt bold, teal): *LLooM — LLM concept induction*
    - LLM induces named concepts: name + criteria + definition
    - Each concept scored per document (A–E scale)
    - **+17.9% coverage vs BERTopic on Congressional bills**
  - Small source line under right column (12pt muted): Source: Lam et al., CHI 2024 (arXiv:2404.12259)
- Bottom full-bleed tinted band (`#EAF4F2`, y 5.55→7.05): kicker
  `THE OPEN QUESTION` (12pt teal, spaced), then the focal line (24pt bold):
  **How many documents *n* before the induced concepts *cover ground truth?***
  — "n" and "cover ground truth?" in amber.

## Slide 3 — Pipeline (light)

- Title: **The pipeline — mini-LLooM, re-engineered for free tiers**
- 6 rounded boxes in one row (≈1.86 × 1.95 in, 15.5pt bold name + 12pt
  Consolas sub-lines), arrows between:
  1. **Documents** — `n docs · synthetic` / `4 cells` (tinted = input)
  2. **Distill** — `Qwen3-8B` / `local llama.cpp` · caption below: *highest volume*
  3. **Cluster** — `bge-small` / `UMAP > HDBSCAN` · caption: *local & free*
  4. **Synthesize** — `Flash-Lite chain` / `gemini` · caption in amber: *the "smart" call*
  5. **Score** — `Qwen3-8B` / `20 docs / call` · caption: *costliest step*
  6. **Matcher** — `few-shot LLM` / `vs 50 GT concepts` (tinted = output)
- Full-bleed tinted band (y≈4.95): **Cache-first infrastructure —** SQLite
  prompt-hash response cache · per-day quota ledger (500/day per pool) · RPM
  throttle · atomic resumable run records. Re-runs and demo rehearsals are free.
- Source (12pt muted): Pipeline per LLooM §3; Distill/Score kept local
  (GTX 1650) to save cloud quota for Synthesize and matching.

## Slide 4 — Theory → experiments (light)

- Title: **Unit II theory, tested empirically**
- Sub-line (14.5pt muted): Five experiments connect each classical concept to a measurable question.
- Five rows (hairline-separated). Left: oversized numeral (28pt bold; E1
  amber, rest teal). Right: name (16.5pt bold) + description (13.5pt muted):
  - **E1 · Coverage vs n (headline)** — 5 methods × 5 trials · n ∈ {10, 20, 40, 80, 160} · specific coverage primary (LLooM-Gemini, LLooM-Qwen, zero-shot ×2, BERTopic)
  - **E2 · PAC sample complexity** — rectangle consistent learner vs Mitchell bound `m ≥ (4/ε)·ln(13/δ)` (Consolas, teal)
  - **E3 · Capacity** — growth function → effective VC dimension vs Sauer envelope `Π_H(m) ≤ (em/d)^d`
  - **E4 · Uniform convergence** — empirical vs held-out gap of the Score operator as n grows
  - **E5 · No Free Lunch** — coverage collapse on 10 label-derangement variants (zero fixed points, fixed seeds)

## Slide 5 — Results (light)

- Title: **Pilot: coverage grows, then plateaus — verdict GO**
- Left (≈7.4 × 4.55 in): **line chart with markers**, x = documents n
  {10, 25, 50, 100}, y = coverage 0.0–1.0, legend bottom, value labels (0.00
  format) above points:
  - *Generic coverage* `0.00, 0.25, 0.70, 0.75` — muted teal `#9AC2BC`
  - *Specific coverage* `0.06, 0.22, 0.35, 0.30` — deep teal `#0F766E` (the primary signal)
- Caption (12pt muted): Pilot grid, 2 trials per n (v2 config, 2026-10-06).
  Specific = paper's 40-concept taxonomy — the trustworthy signal; generic
  readings are inflated by umbrella over-matching.
- Right column, three stat callouts separated by hairlines:
  - **0.30 ± 0.07** (40pt amber) — specific coverage @ n = 100
  - **CV 0.24** (30pt teal) — trial variance ≤ 0.25 bar → verdict GO
  - **8 / 8** (30pt teal) — pilot runs completed, zero errors

## Slide 6 — Roadmap (light)

- Title: **Road to the finished study**
- Sub-line: Three milestones done, three to go — every step resumable from the cache.
- Horizontal timeline (hairline at mid-slide, 6 evenly spaced nodes):
  labels above (15pt bold), title + 2-line description below (14.5pt bold /
  12pt muted):
  1. **W1 · Scaffolding** ● teal — Dual-LLM client, cache + ledger, mini-pipeline
  2. **W2 · Dataset** ● teal — 160 docs, exact labels, 10 NFL variants
  3. **W3 · Pilot** ● teal — Matcher validated, verdict GO
  4. **W4–5 · Experiments** ● **amber (current)** — E1–E5 grid, quota spread over days
  5. **W6 · Analysis** ○ hollow — Headline overlay: bound vs rectangles vs LLMs
  6. **W7–8 · Ship** ○ hollow — Streamlit demo, Unit-II report, deck
- Bottom full-bleed tinted band: **Deliverables —** publication-style charts
  (theory bound overlaying the LLM & BERTopic curves) · Streamlit live demo ·
  written report mapped to Unit II · presentation deck.

## Slide 7 — Team + end note (dark `#0E1524`)

- Kicker: `TEAM` (13pt `#5EC8BC`, spaced)
- Two rounded cards (`#1A2334`) side by side:
  - **Aaryan** (22pt bold white) — Design · implementation · experiments ·
    analysis — `github.com/aaryanmax` (Consolas, teal). *(add teammates here)*
  - **Workflow** (18pt bold white) — Coding-agent driven development ·
    72 offline tests + CI gate · cache-first LLM client, reruns are free
- End note (y≈3.85, 30pt bold white, 2 lines): The answer is **a curve, not a
  number** — and classical learning theory draws it. ("a curve, not a number"
  in amber)
- `Thank you — questions?` (20pt muted) · bottom line in Consolas:
  `AML-Project / lloom-lab · Unit II · 2026`

## Speaker notes (per slide)

1. PAC-style question posed to LLooM (CHI 2024).
2. +17.9% is LLooM's headline; the sample-complexity question is the gap we fill.
3. UMAP→HDBSCAN recipe was chosen by a zero-API offline harness after the first pilot NO-GO.
4. E2/E3 are offline theory modules with textbook answers encoded in tests (thresholds VC=1, intervals VC=2, rectangles VC=4 via the diamond).
5. First pilot NO-GO (CV 0.75) was root-caused to clustering, not the research question; fix → GO.
6. Grid pauses cleanly on quota and resumes from cache; nothing is lost.
7. Close on the curve framing; invite questions.
