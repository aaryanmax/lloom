"""Mini-LLooM orchestrator: Distill -> Cluster -> Synthesize -> Score.

One iteration (paper §3). The Loop operator and higher-level abstraction are
added later with the experiments.
"""

from __future__ import annotations

from lloom_lab.pipeline import cluster as cluster_mod
from lloom_lab.pipeline import distill as distill_mod
from lloom_lab.pipeline import score as score_mod
from lloom_lab.pipeline import synthesize as synth_mod


def run_iteration(client, docs: dict[str, str], seed_phrase: str = "",
                  min_cluster_size: int = 3, n_concepts_per_cluster: int = 1,
                  distill_model: str = "qwen", synth_model: str = "gemini",
                  score_model: str = "qwen", score_batch_size: int = 20,
                  use_filter: bool = False) -> dict:
    """Run one concept-induction iteration over {doc_id: text}.

    Returns {"concepts": [...], "scores": {concept_name: {doc_id: result}},
             "distilled": [...], "clusters": {label: [doc_ids]}}
    """
    # 1. Distill (local model)
    distilled = []
    for doc_id, text in docs.items():
        d = distill_mod.distill(client, text, seed_phrase=seed_phrase,
                                use_filter=use_filter, model=distill_model)
        d["doc_id"] = doc_id
        distilled.append(d)

    # 2. Cluster pooled bullets (local embeddings + HDBSCAN)
    clusters = cluster_mod.pooled_clusters(distilled, min_cluster_size)

    # 3. Synthesize concepts per cluster (cloud model)
    concepts = []
    seen_names: set[str] = set()
    for label, items in sorted(clusters.items()):
        for c in synth_mod.synthesize(client, items, seed_phrase=seed_phrase,
                                      n_concepts=n_concepts_per_cluster,
                                      model=synth_model):
            key = c["name"].lower()
            if key not in seen_names:
                seen_names.add(key)
                c["cluster_label"] = label
                concepts.append(c)

    # 4. Score every doc against every concept
    scores: dict[str, dict] = {}
    for c in concepts:
        scores[c["name"]] = score_mod.score_examples(
            client, c["name"], c["criteria_prompt"], docs,
            model=score_model, batch_size=score_batch_size)

    cluster_docs: dict[int, list[str]] = {}
    for label, items in clusters.items():
        cluster_docs[label] = sorted({it["doc_id"] for it in items})

    return {"concepts": concepts, "scores": scores,
            "distilled": distilled, "clusters": cluster_docs}
