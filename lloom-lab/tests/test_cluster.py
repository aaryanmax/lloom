"""Clustering ladder on synthetic vectors (no model downloads, offline)."""

import numpy as np

from lloom_lab.pipeline.cluster import cluster_vectors


def _blobs(n_per=8, seed=0):
    """Three well-separated 2D blobs."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    pts = np.concatenate([c + rng.normal(scale=0.3, size=(n_per, 2))
                          for c in centers])
    return pts.astype(np.float32)


def test_separated_blobs_cluster():
    labels = cluster_vectors(_blobs())
    # blobs must split into >= 2 non-noise groups
    assert len({lab for lab in labels if lab != -1}) >= 2


def test_all_noise_falls_back_to_single_cluster():
    # one tight blob with min_cluster_size larger than n -> nothing findable
    X = np.tile(np.arange(6, dtype=np.float32) * 1e-6, (5, 1))
    labels = cluster_vectors(X, min_cluster_size=8)
    assert len(labels) == 5
    # ladder must never return all-noise: either a real split or all-in-one
    assert (labels != -1).any()


def test_fewer_than_min_size_short_circuit():
    labels = cluster_vectors(_blobs(n_per=2), min_cluster_size=3)
    assert (labels == 0).all()
