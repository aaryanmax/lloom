"""Offline clustering-config harness (zero API): score candidate clustering
recipes against ground-truth topics, using cached distilled bullets from
completed pilot runs.

Motivation (2026-10-06): the v1 pilot's NO-GO verdict was traced to clustering
variance — raw-embedding HDBSCAN on short bullets recovered topics
inconsistently across trials. This harness compares recipes (raw HDBSCAN vs
UMAP+HDBSCAN at several n_neighbors) with NO LLM calls: each bullet inherits
its source doc's specific concepts, so a cluster's "topic" is the majority
concept among its bullets' docs.

Metrics per run record:
  recover : fraction of the subset's ground-truth specific concepts that are
            the majority concept of at least one cluster
  purity  : mean over clusters of (share of bullets whose doc carries the
            cluster's majority concept)
  clusters, noise% : cluster count and share of noise bullets

    python scripts/cluster_harness.py            # compare all recipes
    python scripts/cluster_harness.py --runs n100_t0 n100_t1

Requires completed run records under results/runs/pilot/ (distill output is
embedded in each record, so this works from any machine with the records).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import RUNS_DIR  # noqa: E402
from lloom_lab.experiments.run_lloom import load_run  # noqa: E402
from lloom_lab.pipeline import cluster as cluster_mod  # noqa: E402

warnings.filterwarnings("ignore")


def evaluate(pooled: list[dict], labels: list[int],
             gt_specific: set[str]) -> tuple[float, float, int, float]:
    clusters: dict[int, list[dict]] = {}
    for item, lab in zip(pooled, labels, strict=True):
        if lab != -1:
            clusters.setdefault(int(lab), []).append(item)
    n_noise = sum(1 for lab in labels if lab == -1)
    represented: set[str] = set()
    purities: list[float] = []
    for items in clusters.values():
        counts: dict[str, int] = {}
        for it in items:
            for t in it["topics"]:
                counts[t] = counts.get(t, 0) + 1
        if not counts:
            continue
        best, best_n = max(counts.items(), key=lambda kv: kv[1])
        represented.add(best)
        purities.append(best_n / len(items))
    recovery = len(represented & gt_specific) / len(gt_specific)
    purity = sum(purities) / len(purities) if purities else 0.0
    return recovery, purity, len(clusters), 100 * n_noise / len(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+",
                    default=["n050_t0", "n050_t1", "n100_t0", "n100_t1"],
                    help="run ids (without .json) under results/runs/pilot/")
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN

    X_by_run: dict[str, object] = {}
    pooled_by_run: dict[str, list[dict]] = {}
    gt_by_run: dict[str, set[str]] = {}
    for rid in args.runs:
        rec = load_run(RUNS_DIR / "pilot" / f"{rid}.json")
        if rec is None:
            sys.exit(f"run {rid} not found or incomplete")
        pooled = []
        for d in rec["distilled"]:
            doc_concepts = None
            # doc-level concepts live on the corpus docs; recover them from the
            # record's own ground truth per doc via doc_ids is not possible, so
            # use the corpus builder's labels if the store is present.
            try:
                from lloom_lab.experiments.run_lloom import load_corpus
                corpus = {c["doc_id"]: c for c in load_corpus()}
                doc_concepts = corpus[d["doc_id"]]["concepts"]["specific"]
            except FileNotFoundError:
                sys.exit("data_store/synthetic_v1/docs.jsonl missing — build "
                         "the dataset first (build-dataset)")
            for b in d["bullets"]:
                pooled.append({"doc_id": d["doc_id"], "bullet": b,
                               "topics": set(doc_concepts)})
        pooled_by_run[rid] = pooled
        X_by_run[rid] = cluster_mod.embed([p["bullet"] for p in pooled])
        gt_by_run[rid] = set(rec["ground_truth"]["specific"])

    def umap_reduce(x, n_neighbors, n_components=5):
        import numpy as np
        import umap
        return np.asarray(umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components, metric="cosine",
                                    random_state=42).fit_transform(x),
                          dtype="float32")

    recipes = [("raw", mcs, None) for mcs in (3, 4, 5)] + [
        (f"umap{nn}", mcs, nn) for nn in (10, 15, 30) for mcs in (3, 4)]

    print(f"{'recipe':<14} {'run':<9} {'clusters':>8} {'noise%':>6} "
          f"{'recover':>7} {'purity':>6}")
    for name, mcs, nn in recipes:
        rows = []
        for rid in args.runs:
            x = X_by_run[rid] if nn is None else umap_reduce(X_by_run[rid], nn)
            labels = HDBSCAN(min_cluster_size=mcs,
                             metric="euclidean").fit_predict(x)
            r = evaluate(pooled_by_run[rid], list(labels), gt_by_run[rid])
            rows.append(r)
            print(f"{name + ' mcs=' + str(mcs):<14} {rid:<9} {r[2]:>8} "
                  f"{r[3]:>5.0f}% {r[0]:>7.2f} {r[1]:>6.2f}")
        mean = [sum(r[i] for r in rows) / len(rows) for i in (0, 1, 2, 3)]
        print(f"{name + ' mcs=' + str(mcs) + ' MEAN':<14} {'':<9} {mean[2]:>8.1f} "
              f"{mean[3]:>5.0f}% {mean[0]:>7.2f} {mean[1]:>6.2f}\n")


if __name__ == "__main__":
    main()
