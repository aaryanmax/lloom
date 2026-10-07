"""Cluster operator (paper §3.1.1): pool distilled bullet points (detached from
their source docs so concepts can cut across documents), embed locally with
bge-small-en-v1.5 (fastembed), and group with HDBSCAN (sklearn >= 1.3).
"""

from __future__ import annotations

import numpy as np

_EMBEDDER = None
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)
    return _EMBEDDER


def embed(texts: list[str]) -> np.ndarray:
    vecs = list(_get_embedder().embed(texts))
    X = np.array(vecs, dtype=np.float32)
    # L2-normalize so euclidean distance ~ cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def cluster_vectors(X: np.ndarray, min_cluster_size: int = 3,
                    random_state: int = 42) -> np.ndarray:
    """Cluster pre-computed embeddings; one label per row (-1 = noise).

    Primary recipe (BERTopic-style, chosen by the offline harness on cached
    pilot bullets, 2026-10-06): UMAP(cosine) reduction then HDBSCAN. On short
    distilled bullets it beats raw-embedding HDBSCAN decisively — topic
    recovery 0.66 vs 0.40, noise 14% vs 59%, and trial-to-trial consistent
    (n=100: 0.80/0.80 vs 0.55/0.53). Robustness ladder:
      1. UMAP(cosine, n_neighbors<=10, 5 comps) -> HDBSCAN;
      2. if that fails/all-noise: HDBSCAN on the raw (L2-normalized) embeddings;
      3. if still nothing: one cluster holding all bullets, so Synthesize
         still gets a chance to find patterns (the paper's Loop would refine).
    """
    from sklearn.cluster import HDBSCAN

    def _hdbscan(mat: np.ndarray) -> np.ndarray:
        # sklearn raises if min_cluster_size > n_samples — clamp instead.
        return HDBSCAN(min_cluster_size=min(min_cluster_size, len(mat)),
                       metric="euclidean").fit_predict(mat)

    if len(X) < 2 or len(X) < min_cluster_size:
        # too few bullets to cluster meaningfully: one group, never all-noise
        return np.zeros(len(X), dtype=int)

    try:
        import umap
        n_neighbors = max(2, min(10, len(X) - 1))
        n_components = max(2, min(5, n_neighbors))
        Xr = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                       metric="cosine", random_state=random_state).fit_transform(X)
        labels = _hdbscan(np.asarray(Xr, dtype=np.float32))
        if (labels != -1).any():
            return labels
    except Exception:  # noqa: BLE001 — UMAP is an enhancement, never a hard dep at runtime
        pass

    # UMAP unavailable or found nothing: raw-embedding HDBSCAN
    labels = _hdbscan(X)
    if (labels != -1).any():
        return labels
    return np.zeros(len(X), dtype=int)  # single-cluster fallback


def cluster_bullets(bullets: list[str], min_cluster_size: int = 3) -> np.ndarray:
    """Return one cluster label per bullet (-1 = noise, like the paper's HDBSCAN)."""
    if len(bullets) < min_cluster_size:
        return np.zeros(len(bullets), dtype=int)
    return cluster_vectors(embed(bullets), min_cluster_size)


def pooled_clusters(distilled: list[dict], min_cluster_size: int = 3) -> dict[int, list[dict]]:
    """Pool all bullets (detached from docs, per the paper) and group them.

    Args:
        distilled: [{"doc_id": ..., "bullets": [...], ...}, ...]
    Returns:
        {cluster_label: [{"doc_id": ..., "bullet": ...}, ...]} for label != -1
    """
    pooled: list[dict] = []
    for d in distilled:
        for b in d["bullets"]:
            pooled.append({"doc_id": d["doc_id"], "bullet": b})
    if not pooled:
        return {}
    labels = cluster_bullets([p["bullet"] for p in pooled], min_cluster_size)
    clusters: dict[int, list[dict]] = {}
    for item, lab in zip(pooled, labels, strict=True):
        if lab != -1:
            clusters.setdefault(int(lab), []).append(item)
    return clusters
