"""Extract figure regions from the LLooM (arXiv 2404.12259) PDF into images/.

Figures are located by finding the y-coordinate of their "Figure N:" caption
block; the figure art sits between the page header (or, on page 1, the author
block) and that caption.
"""

import os

import pymupdf

PDF = r"C:\Users\Admin\Projects\AML-Project\Concept Induction-Analyzing Unstructured Text with High-Level.pdf"
OUT = r"C:\Users\Admin\Projects\AML-Project\images"
ZOOM = 200 / 72  # ~200 DPI

doc = pymupdf.open(PDF)
os.makedirs(OUT, exist_ok=True)


def find_block(pno, contains=None, startswith=None):
    for b in doc[pno].get_text("blocks"):
        t = b[4].strip()
        if startswith is not None and t.startswith(startswith):
            return b
        if contains is not None and contains in t:
            return b
    return None


def caption_bottom(pno, fig):
    b = find_block(pno, startswith=f"Figure {fig}:")
    if b is not None:
        return b[3]
    rects = doc[pno].search_for(f"Figure {fig}:")
    if rects:
        return max(r.y1 for r in rects)
    raise RuntimeError(f"caption for Figure {fig} not found on page {pno + 1}")


def caption_x0(pno, fig):
    rects = doc[pno].search_for(f"Figure {fig}:")
    if rects:
        return min(r.x0 for r in rects)
    raise RuntimeError(f"caption for Figure {fig} not found on page {pno + 1}")


def header_bottom(pno):
    b = find_block(pno, contains="Honolulu, HI, USA")
    return b[3] if b is not None else 60.0


def save(pno, y0, y1, name, x0=48.0, x1=566.0):
    page = doc[pno]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=pymupdf.Rect(x0, y0, x1, y1))
    path = os.path.join(OUT, name)
    pix.save(path)
    print(f"{name}: {pix.width}x{pix.height}px  (page {pno + 1}, y {y0:.0f}-{y1:.0f})")


# Figure 1 lives between the author block and its caption on page 1.
auth = find_block(0, contains="msb@cs.stanford.edu")
save(0, auth[3] + 8, caption_bottom(0, 1) + 4, "fig1_summary.png")

FULL = dict(x0=48.0, x1=566.0)
for pno, fig, name in [
    (2, 2, "fig2_pipeline.png"),
    (5, 3, "fig3_concept_generation.png"),
    (7, 4, "fig4_workbench.png"),
    (9, 5, "fig5_toxic_concepts.png"),
    (10, 6, "fig6_toxic_prevalence.png"),
    (11, 7, "fig7_partisan_prevalence.png"),
    (14, 10, "fig10_synthetic_samples.png"),
    (15, 12, "fig12_synthetic_by_params.png"),
    (16, 13, "fig13_expert_ratings.png"),
    (22, 14, "fig14_neurips_prevalence.png"),
    (24, 15, "fig15_classification_metrics.png"),
]:
    save(pno, header_bottom(pno) + 6, caption_bottom(pno, fig) + 4, name, **FULL)

# Page 12 holds Figure 7 (top) and Figure 8 (below it).
cap7 = caption_bottom(11, 7)
save(11, header_bottom(11) + 6, cap7 + 4, "fig7_partisan_prevalence.png", **FULL)
save(11, cap7 + 10, caption_bottom(11, 8) + 4, "fig8_uist_prevalence.png", **FULL)

# Page 15 holds Figure 10 (full width) and Figure 11 (left column below it).
cap10 = caption_bottom(14, 10)
save(14, header_bottom(14) + 6, cap10 + 4, "fig10_synthetic_samples.png", **FULL)
save(14, cap10 + 10, caption_bottom(14, 11) + 4, "fig11_synthetic_coverage.png",
     x0=48.0, x1=306.0)

# Page 13, Figure 9 sits at the top of the right column.
x0 = caption_x0(12, 9) - 8
save(12, header_bottom(12) + 6, caption_bottom(12, 9) + 4, "fig9_benchmark_coverage.png",
     x0=x0, x1=566.0)

# Page 25 holds Figure 15 (full width) and Figure 16 (left column below it).
cap15 = caption_bottom(24, 15)
save(24, header_bottom(24) + 6, cap15 + 4, "fig15_classification_metrics.png", **FULL)
save(24, cap15 + 10, caption_bottom(24, 16) + 4, "fig16_irr.png", x0=48.0, x1=306.0)

print("done ->", OUT)
