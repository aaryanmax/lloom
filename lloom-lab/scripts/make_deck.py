# Build the 7-slide AML project deck with python-pptx.
# Palette: deep ink + teal primary + amber accent; fonts: Segoe UI + Consolas.
import math

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

INK = RGBColor(0x0E, 0x15, 0x24)       # dark background (cover + closing)
INK_CARD = RGBColor(0x1A, 0x23, 0x34)  # card fill on dark
BG = RGBColor(0xFF, 0xFF, 0xFF)
PRIMARY = RGBColor(0x0F, 0x76, 0x6E)   # deep teal
TEAL_LT = RGBColor(0x5E, 0xC8, 0xBC)   # teal on dark
TINT = RGBColor(0xEA, 0xF4, 0xF2)      # light teal band
ACCENT = RGBColor(0xD9, 0x77, 0x06)    # amber, sparing
TEXT = RGBColor(0x15, 0x20, 0x2B)
MUTED = RGBColor(0x5B, 0x6B, 0x7B)
LIGHT = RGBColor(0xE9, 0xEF, 0xF4)
LIGHT_MUTED = RGBColor(0x8F, 0xA1, 0xB3)
HAIR = RGBColor(0xDD, 0xE5, 0xE9)      # hairline separators
GRID = RGBColor(0xE2, 0xE8, 0xF0)

F = "Segoe UI"
FC = "Consolas"
W, H = 13.333, 7.5

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


def slide(bg=BG):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def tb(s, x, y, w, h, runs, size=14, color=TEXT, bold=False, font=F,
       align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spc=None, line=None,
       space_after=None):
    """runs: str or list of (text, {overrides}) tuples."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    if line:
        p.line_spacing = line
    if space_after is not None:
        p.space_after = Pt(space_after)
    if isinstance(runs, str):
        runs = [(runs, {})]
    for txt, o in runs:
        r = p.add_run()
        r.text = txt
        r.font.name = o.get("font", font)
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", False)
        r.font.color.rgb = o.get("color", color)
        _spc = o.get("spc", spc)
        if _spc:
            r.font._rPr.set("spc", str(_spc))
    return box


def rect(s, x, y, w, h, fill=None, line_color=None, line_w=1.0, rounded=False,
         radius=0.08, dash=None):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(line_w)
        if dash:
            shp.line._get_or_add_ln().makeelement(
                qn("a:prstDash"), {}).set("val", dash) if False else None
    shp.shadow.inherit = False
    return shp


def hline(s, x, y, w, color=HAIR, weight=1.0):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y),
                                Inches(x + w), Inches(y))
    ln.line.color.rgb = color
    ln.line.width = Pt(weight)
    ln.shadow.inherit = False
    return ln


def arrow(s, x1, y1, x2, y2, color=PRIMARY):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                Inches(x2), Inches(y2))
    ln.line.color.rgb = color
    ln.line.width = Pt(1.75)
    ln.shadow.inherit = False
    lnEl = ln.line._get_or_add_ln()
    tail = lnEl.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med",
                                              "len": "med"})
    lnEl.append(tail)
    return ln


def dot(s, cx, cy, d, fill, line_color=None, line_w=1.5):
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2),
                           Inches(d), Inches(d))
    if fill is None:
        o.fill.background()
    else:
        o.fill.solid()
        o.fill.fore_color.rgb = fill
    if line_color is None:
        o.line.fill.background()
    else:
        o.line.color.rgb = line_color
        o.line.width = Pt(line_w)
    o.shadow.inherit = False
    return o


def page_no(s, n):
    tb(s, W - 1.0, H - 0.55, 0.5, 0.3, f"{n} / 7", size=12, color=MUTED,
       align=PP_ALIGN.RIGHT)


def notes(s, txt):
    s.notes_slide.notes_text_frame.text = txt


# ---------------------------------------------------------------- S1 cover
s = slide(INK)
tb(s, 0.7, 0.75, 11.0, 0.35,
   "ADVANCED ML · UNIT II — PAC LEARNING · VC DIMENSION · NO FREE LUNCH",
   size=13, color=TEAL_LT, bold=True, spc=200)
tb(s, 0.7, 1.9, 7.3, 2.6, [("How Many Examples\nDoes It Take?", {})],
   size=50, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, line=1.02)
tb(s, 0.7, 4.35, 6.6, 1.0,
   "The sample complexity & capacity of LLM concept induction",
   size=20, color=LIGHT_MUTED, line=1.15)
hline(s, 0.7, 5.75, 5.4, color=RGBColor(0x2A, 0x36, 0x4A), weight=1.0)
tb(s, 0.7, 5.95, 7.0, 0.9,
   [("Built on LLooM\n", {"color": LIGHT, "size": 14, "bold": True}),
    ("Lam et al., CHI 2024 · arXiv:2404.12259", {"color": LIGHT_MUTED,
                                                 "size": 13, "font": FC})],
   line=1.3)
# coverage-curve motif: dots rising to a plateau
y0, y1 = 5.55, 2.05
k, kmax = 2.6, 1 - math.exp(-2.6)
for i in range(12):
    t = i / 11
    x = 8.05 + t * 4.15
    y = y0 - (y0 - y1) * (1 - math.exp(-k * t)) / kmax
    d = 0.17 if i >= 9 else 0.13
    dot(s, x, y, d, ACCENT if i >= 9 else TEAL_LT)
pl = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(10.9), Inches(y1 + 0.05),
                            Inches(12.65), Inches(y1 + 0.05))
pl.line.color.rgb = RGBColor(0x3A, 0x47, 0x5C)
pl.line.width = Pt(1.0)
pl.line._get_or_add_ln().makeelement(qn("a:prstDash"), {"val": "dash"})
pl.shadow.inherit = False
tb(s, 10.05, 1.62, 2.6, 0.3, "plateau", size=12, color=LIGHT_MUTED)
tb(s, 8.05, 5.85, 4.6, 0.3, "documents n", size=12, color=LIGHT_MUTED)
notes(s, "Course project for Advanced ML Unit II. We ask a PAC-style question of "
         "LLooM (CHI 2024): how many documents does LLM concept induction need "
         "before induced concepts cover ground truth?")

# ------------------------------------------------------- S2 the question
s = slide()
tb(s, 0.7, 0.5, 11.5, 0.6, "From topic models to concept induction",
   size=32, bold=True)
tb(s, 0.7, 1.35, 5.4, 0.4, "Topic models — LDA · BERTopic", size=17,
   bold=True, color=MUTED)
rows_l = ["Discover co-occurring words, not meanings",
          "Themes readable only as keyword lists",
          "No notion of ground-truth concepts"]
rows_r = ["LLM induces named concepts: name + criteria + definition",
          "Each concept scored per document (A–E scale)",
          "+17.9% coverage vs BERTopic on Congressional bills"]
y = 1.95
for r in rows_l:
    dot(s, 0.78, y + 0.12, 0.1, MUTED)
    tb(s, 1.05, y, 5.1, 0.85, r, size=14.5, color=TEXT, line=1.1)
    y += 1.0
tb(s, 7.0, 1.35, 5.6, 0.4, "LLooM — LLM concept induction", size=17,
   bold=True, color=PRIMARY)
y = 1.95
for r in rows_r:
    dot(s, 7.08, y + 0.12, 0.1, PRIMARY)
    tb(s, 7.35, y, 5.3, 0.85, r, size=14.5, color=TEXT, line=1.1)
    y += 1.0
tb(s, 7.35, 4.85, 5.3, 0.3, "Source: Lam et al., CHI 2024 (arXiv:2404.12259)",
   size=12, color=MUTED)
vln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(6.62), Inches(1.45),
                             Inches(6.62), Inches(5.2))
vln.line.color.rgb = HAIR
vln.line.width = Pt(1.0)
vln.shadow.inherit = False
band = rect(s, 0, 5.55, W, 1.5, fill=TINT)
tb(s, 0.7, 5.8, 12.0, 0.3, "THE OPEN QUESTION", size=12, color=PRIMARY,
   bold=True, spc=200)
tb(s, 0.7, 6.15, 12.0, 0.75,
   [("How many documents ", {}), ("n", {"color": ACCENT}),
    (" before the induced concepts ", {}),
    ("cover ground truth?", {"color": ACCENT})],
   size=24, bold=True)
page_no(s, 2)
notes(s, "LLooM beats BERTopic by +17.9% coverage on Congressional bills, but the "
         "paper never asks the sample-complexity question. That gap is our project.")

# ------------------------------------------------------------ S3 pipeline
s = slide()
tb(s, 0.7, 0.5, 12.0, 0.6, "The pipeline — mini-LLooM, re-engineered for free tiers",
   size=32, bold=True)
ops = [("Documents", ["n docs · synthetic", "4 cells"], None),
       ("Distill", ["Qwen3-8B", "local llama.cpp"], "highest volume"),
       ("Cluster", ["bge-small", "UMAP > HDBSCAN"], "local & free"),
       ("Synthesize", ["Flash-Lite chain", "gemini"], "the \u201csmart\u201d call"),
       ("Score", ["Qwen3-8B", "20 docs / call"], "costliest step"),
       ("Matcher", ["few-shot LLM", "vs 50 GT concepts"], None)]
bx, bw, gap, by, bh = 0.62, 1.86, 0.22, 1.85, 1.95
for i, (name, subs, role) in enumerate(ops):
    x = bx + i * (bw + gap)
    is_in = name in ("Documents", "Matcher")
    rect(s, x, by, bw, bh, fill=TINT if is_in else BG,
         line_color=None if is_in else PRIMARY, line_w=1.5, rounded=True,
         radius=0.1)
    tb(s, x + 0.12, by + 0.22, bw - 0.24, 0.4, name, size=15.5, bold=True,
       align=PP_ALIGN.CENTER, color=PRIMARY if not is_in else TEXT)
    tb(s, x + 0.08, by + 0.78, bw - 0.16, 0.75,
       [(subs[0] + "\n", {"font": FC}), (subs[1], {"font": FC})],
       size=12, color=MUTED, align=PP_ALIGN.CENTER, line=1.15)
    if role:
        tb(s, x - 0.1, by + bh + 0.12, bw + 0.2, 0.3, role, size=12,
           color=ACCENT if name == "Synthesize" else MUTED,
           align=PP_ALIGN.CENTER)
    if i < len(ops) - 1:
        arrow(s, x + bw + 0.03, by + bh / 2, x + bw + gap - 0.03, by + bh / 2)
band = rect(s, 0, 4.95, W, 1.3, fill=TINT)
tb(s, 0.7, 5.22, 12.0, 0.8,
   [("Cache-first infrastructure — ", {"bold": True, "color": PRIMARY}),
    ("SQLite prompt-hash response cache · per-day quota ledger (500/day per "
     "pool) · RPM throttle · atomic resumable run records. Re-runs and demo "
     "rehearsals are free.", {})],
   size=14.5, line=1.25)
tb(s, 0.7, 6.55, 12.0, 0.3,
   "Pipeline per LLooM §3; Distill/Score kept local (GTX 1650) to save cloud "
   "quota for Synthesize and matching.", size=12, color=MUTED)
page_no(s, 3)
notes(s, "Mini-LLooM: Distill to bullets, cluster embeddings, synthesize one "
         "concept per cluster, score A-E. UMAP->HDBSCAN recipe chosen by an "
         "offline harness after the pilot's first NO-GO.")

# ------------------------------------------------- S4 theory -> experiments
s = slide()
tb(s, 0.7, 0.5, 12.0, 0.6, "Unit II theory, tested empirically", size=32,
   bold=True)
tb(s, 0.7, 1.18, 12.0, 0.35,
   "Five experiments connect each classical concept to a measurable question.",
   size=14.5, color=MUTED)
exps = [
    ("E1", "Coverage vs n  (headline)",
     [("5 methods × 5 trials · n ∈ {10, 20, 40, 80, 160} · specific coverage "
       "primary (LLooM-Gemini, LLooM-Qwen, zero-shot ×2, BERTopic)", {})]),
    ("E2", "PAC sample complexity",
     [("Rectangle consistent learner vs Mitchell bound  ", {}),
      ("m ≥ (4/ε)·ln(13/δ)", {"font": FC, "color": PRIMARY})]),
    ("E3", "Capacity: growth function → VC dimension",
     [("Exact realizability tests → Π_H(m); effective VC dim vs Sauer "
       "envelope  ", {}), ("Π_H(m) ≤ (em/d)^d", {"font": FC, "color": PRIMARY})]),
    ("E4", "Uniform convergence",
     [("Empirical vs held-out gap of the Score operator as n grows", {})]),
    ("E5", "No Free Lunch",
     [("Coverage collapse on 10 label-derangement variants (zero fixed "
       "points, fixed seeds)", {})]),
]
y = 1.75
for i, (tag, name, desc) in enumerate(exps):
    tb(s, 0.7, y + 0.06, 1.05, 0.6, tag, size=28, bold=True,
       color=ACCENT if i == 0 else PRIMARY)
    tb(s, 1.95, y, 10.7, 0.4, name, size=16.5, bold=True)
    tb(s, 1.95, y + 0.42, 10.2, 0.5, desc, size=13.5, color=MUTED)
    if i < len(exps) - 1:
        hline(s, 0.7, y + 1.0, 12.0)
    y += 1.08
page_no(s, 4)
notes(s, "E2 and E3 are fully offline theory modules with textbook answers "
         "encoded in tests (thresholds VC=1, intervals VC=2, rectangles VC=4 "
         "via the diamond configuration).")

# -------------------------------------------------------------- S5 results
s = slide()
tb(s, 0.7, 0.5, 12.0, 0.6, "Pilot: coverage grows, then plateaus — verdict GO",
   size=32, bold=True)
cd = CategoryChartData()
cd.categories = ["10", "25", "50", "100"]
cd.add_series("Generic coverage", (0.00, 0.25, 0.70, 0.75))
cd.add_series("Specific coverage", (0.06, 0.22, 0.35, 0.30))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.45),
                        Inches(7.4), Inches(4.55), cd)
ch = gf.chart
ch.has_title = False
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(13)
ch.legend.font.name = F
for ser, col in zip(ch.series, (RGBColor(0x9A, 0xC2, 0xBC), PRIMARY), strict=False):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.75)
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
plot = ch.plots[0]
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format = "0.00"
dl.number_format_is_linked = False
dl.font.size = Pt(12)
dl.font.name = F
dl.font.color.rgb = TEXT
dl.position = XL_LABEL_POSITION.ABOVE
cax = ch.category_axis
cax.tick_labels.font.size = Pt(13)
cax.tick_labels.font.name = F
cax.has_title = True
cax.axis_title.text_frame.text = "documents n"
cax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
cax.axis_title.text_frame.paragraphs[0].runs[0].font.name = F
cax.axis_title.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED
vax = ch.value_axis
vax.minimum_scale, vax.maximum_scale = 0.0, 1.0
vax.tick_labels.font.size = Pt(12)
vax.tick_labels.font.name = F
vax.tick_labels.number_format = "0.0"
vax.tick_labels.number_format_is_linked = False
vax.has_major_gridlines = True
vax.major_gridlines.format.line.color.rgb = GRID
vax.major_gridlines.format.line.width = Pt(0.5)
tb(s, 0.7, 6.15, 7.3, 0.75,
   "Pilot grid, 2 trials per n (v2 config, 2026-10-06). Specific = paper's "
   "40-concept taxonomy — the trustworthy signal; generic readings are "
   "inflated by umbrella over-matching.", size=12, color=MUTED, line=1.2)
stats = [("0.30 ± 0.07", "specific coverage @ n = 100", ACCENT, 40),
         ("CV 0.24", "trial variance ≤ 0.25 bar → verdict GO", PRIMARY, 30),
         ("8 / 8", "pilot runs completed, zero errors", PRIMARY, 30)]
y = 1.6
for num, lab, col, sz in stats:
    tb(s, 8.45, y, 4.3, 0.85, num, size=sz, bold=True, color=col)
    tb(s, 8.45, y + 0.88, 4.3, 0.6, lab, size=13.5, color=MUTED, line=1.15)
    if y < 4.5:
        hline(s, 8.45, y + 1.55, 4.2)
    y += 1.72
page_no(s, 5)
notes(s, "First pilot was NO-GO (CV 0.75): raw-embedding HDBSCAN fragmented into "
         "micro-clusters. UMAP(cosine)->HDBSCAN + one concept per cluster fixed "
         "it (recovery 0.66 vs 0.40 offline). Verdict gate passed: GO.")

# --------------------------------------------------------------- S6 roadmap
s = slide()
tb(s, 0.7, 0.5, 12.0, 0.6, "Road to the finished study", size=32, bold=True)
tb(s, 0.7, 1.18, 12.0, 0.35,
   "Three milestones done, three to go — every step resumable from the cache.",
   size=14.5, color=MUTED)
nodes = [("W1", "Scaffolding", "Dual-LLM client, cache +\nledger, mini-pipeline", True),
         ("W2", "Dataset", "160 docs, exact labels,\n10 NFL variants", True),
         ("W3", "Pilot", "Matcher validated,\nverdict GO", True),
         ("W4–5", "Experiments", "E1–E5 grid, quota\nspread over days", "now"),
         ("W6", "Analysis", "Headline overlay: bound\nvs rectangles vs LLMs", False),
         ("W7–8", "Ship", "Streamlit demo, Unit-II\nreport, deck", False)]
line_y = 3.35
tl = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(1.5), Inches(line_y),
                            Inches(11.85), Inches(line_y))
tl.line.color.rgb = HAIR
tl.line.width = Pt(2.0)
tl.shadow.inherit = False
step = (11.85 - 1.5) / (len(nodes) - 1)
for i, (wk, name, desc, done) in enumerate(nodes):
    cx = 1.5 + i * step
    if done is True:
        dot(s, cx, line_y, 0.24, PRIMARY)
    elif done == "now":
        dot(s, cx, line_y, 0.30, ACCENT)
    else:
        dot(s, cx, line_y, 0.24, BG, line_color=PRIMARY, line_w=2.0)
    tb(s, cx - 1.0, line_y - 0.95, 2.0, 0.35, wk, size=15, bold=True,
       color=ACCENT if done == "now" else PRIMARY, align=PP_ALIGN.CENTER)
    tb(s, cx - 1.0, line_y + 0.38, 2.0, 0.35, name, size=14.5, bold=True,
       align=PP_ALIGN.CENTER)
    tb(s, cx - 1.0, line_y + 0.78, 2.0, 0.85, desc, size=12, color=MUTED,
       align=PP_ALIGN.CENTER, line=1.15)
band = rect(s, 0, 5.75, W, 1.2, fill=TINT)
tb(s, 0.7, 6.0, 12.0, 0.7,
   [("Deliverables — ", {"bold": True, "color": PRIMARY}),
    ("publication-style charts (theory bound overlaying the LLM & BERTopic "
     "curves) · Streamlit live demo · written report mapped to Unit II · "
     "presentation deck", {})],
   size=14.5, line=1.25)
page_no(s, 6)
notes(s, "Weeks 1-3 are complete and the GO/NO-GO gate is passed. The E1 grid "
         "is resumable cell by cell; QuotaExceeded pauses it cleanly until the "
         "next day's quota.")

# ------------------------------------------------------ S7 team + end note
s = slide(INK)
tb(s, 0.7, 0.6, 6.0, 0.35, "TEAM", size=13, color=TEAL_LT, bold=True, spc=200)
card1 = rect(s, 0.7, 1.15, 5.9, 1.85, fill=INK_CARD, rounded=True, radius=0.09)
tb(s, 1.05, 1.45, 5.2, 0.5, "Aaryan", size=22, bold=True,
   color=RGBColor(0xFF, 0xFF, 0xFF))
tb(s, 1.05, 2.0, 5.2, 0.4, "Design · implementation · experiments · analysis",
   size=13.5, color=LIGHT_MUTED)
tb(s, 1.05, 2.45, 5.2, 0.35, "github.com/aaryanmax", size=13, font=FC,
   color=TEAL_LT)
card2 = rect(s, 6.95, 1.15, 5.7, 1.85, fill=INK_CARD, rounded=True, radius=0.09)
tb(s, 7.3, 1.45, 5.0, 0.4, "Workflow", size=18, bold=True,
   color=RGBColor(0xFF, 0xFF, 0xFF))
tb(s, 7.3, 1.98, 5.0, 0.95,
   [("Coding-agent driven development\n", {}),
    ("72 offline tests + CI gate · cache-first LLM client", {})],
   size=13.5, color=LIGHT_MUTED, line=1.3)
tb(s, 0.7, 3.85, 12.0, 1.6,
   [("The answer is ", {}),
    ("a curve, not a number", {"color": ACCENT}),
    (" —\nand classical learning theory draws it.", {})],
   size=30, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), line=1.2)
tb(s, 0.7, 5.75, 8.0, 0.45, "Thank you — questions?", size=20,
   color=LIGHT_MUTED)
tb(s, 0.7, 6.6, 12.0, 0.35, "AML-Project / lloom-lab · Unit II · 2026",
   size=13, font=FC, color=RGBColor(0x3A, 0x47, 0x5C))
notes(s, "Team and closing. Swap in teammate names on the cards as needed.")

OUT = r"C:\Users\Admin\Projects\AML-Project\How_Many_Examples_Deck.pptx"
prs.save(OUT)
print("saved", OUT)
