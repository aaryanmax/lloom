"""NFL variants: label-level derangements, determinism, distinct variants."""

from lloom_lab.data import nfl


def _labels():
    return {f"d{i}": {"specific": [f"s{i}"], "generic": ["G"]} for i in range(8)}


def test_no_fixed_points_even_with_duplicate_labels():
    labels = _labels()
    labels["d1"] = labels["d0"]  # duplicate label set must still move
    out = nfl.permuted_labels(labels, seed=1)
    assert all(out[k] != labels[k] for k in labels)


def test_labels_are_a_permutation_of_each_other():
    labels = _labels()
    out = nfl.permuted_labels(labels, seed=2)
    assert sorted(map(str, out.values())) == sorted(map(str, labels.values()))


def test_deterministic_per_seed():
    labels = _labels()
    assert nfl.permuted_labels(labels, seed=3) == nfl.permuted_labels(labels, seed=3)
    assert nfl.permuted_labels(labels, seed=3) != nfl.permuted_labels(labels, seed=4)


def test_build_variants_shapes():
    variants = nfl.build_variants(_labels(), n_variants=5, base_seed=100)
    assert len(variants) == 5
    for mapping in variants.values():
        assert len(mapping) == 8
        assert all(mapping[k] != _labels()[k] for k in mapping)
