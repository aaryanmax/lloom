"""Build the synthetic dataset (paper §5.2) + NFL permuted-label variants.

Zero LLM calls — everything is programmatic from hand-written sentence pools.

Console entry point:  build-dataset  (see pyproject.toml [project.scripts])
CLI:                  python scripts/build_dataset.py --seed 42 --variants 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lloom_lab import config
from lloom_lab.data import nfl, synthetic
from lloom_lab.data.concepts import ALL_GENERIC, ALL_SPECIFIC, GENERIC_TO_SPECIFIC


def build(out_dir: Path | None = None, seed: int = 42, docs_per_cell: int = 40,
          n_variants: int = 10) -> dict:
    """Build the corpus + NFL variants; returns the manifest dict."""
    docs = synthetic.build_corpus(seed=seed, docs_per_cell=docs_per_cell)
    report = synthetic.verify_corpus(docs)

    out: Path = out_dir or (config.DATA_STORE / "synthetic_v1")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    labels = {d.doc_id: {"specific": d.specifics, "generic": d.generics}
              for d in docs}
    variants = nfl.build_variants(labels, n_variants=n_variants)
    nfl_dir = out / "nfl"
    nfl_dir.mkdir(exist_ok=True)
    for name, mapping in variants.items():
        (nfl_dir / f"{name}.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = {
        "generator": "lloom_lab.data.synthetic (hand-written pools, 0 LLM calls)",
        "seed": seed,
        "docs_per_cell": docs_per_cell,
        "cells": [{"doc_length": L, "prevalence": p,
                   "n_docs": sum(1 for d in docs
                                 if d.doc_length == L and d.prevalence == p)}
                  for L in synthetic.LENGTHS for p in synthetic.PREVALENCES],
        "n_docs": len(docs),
        "ground_truth": {"generic": ALL_GENERIC,
                         "specific": ALL_SPECIFIC,
                         "taxonomy": GENERIC_TO_SPECIFIC},
        "verification": report,
        "deviations_from_paper": [
            "Documents composed from hand-written sentence pools instead of "
            "GPT-4 generation (ground-truth labels exact by construction; no "
            "pre-training memorization possible).",
            "Generic-concept labels derived as parents of the Specific "
            "concepts actually seeded in each document.",
        ],
        "nfl_variants": list(variants),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--docs-per-cell", type=int, default=40)
    ap.add_argument("--variants", type=int, default=10,
                    help="number of NFL permuted-label variants")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: data_store/synthetic_v1)")
    args = ap.parse_args()

    manifest = build(out_dir=args.out, seed=args.seed,
                     docs_per_cell=args.docs_per_cell, n_variants=args.variants)

    v = manifest["verification"]
    print(f"wrote {manifest['n_docs']} docs + {len(manifest['nfl_variants'])} "
          f"NFL variants")
    print(f"verification: sentences_ok={v['sentence_counts_ok']} "
          f"prevalence_ok={v['prevalence_counts_ok']} labels_ok={v['labels_ok']}")
    print(f"docs per Specific concept: min={v['min_docs_per_concept']} "
          f"max={v['max_docs_per_concept']} "
          f"(reused sentences: {v['reused_sentences']})")
    if v["problems"]:
        print("PROBLEMS:")
        for p in v["problems"]:
            print("  -", p)


if __name__ == "__main__":
    main()
