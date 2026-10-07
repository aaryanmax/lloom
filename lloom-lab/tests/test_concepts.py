"""Taxonomy invariants (paper Appendix C.4)."""

from lloom_lab.data.concepts import (
    ALL_GENERIC,
    ALL_SPECIFIC,
    GENERIC_TO_SPECIFIC,
    SPECIFIC_TO_GENERIC,
    parent_of,
    parents_of,
)


def test_taxonomy_shape():
    assert len(ALL_GENERIC) == 10
    assert len(ALL_SPECIFIC) == 40
    for generic, children in GENERIC_TO_SPECIFIC.items():
        assert len(children) == 4, generic
        assert len(set(children)) == 4, generic


def test_parent_mapping_is_bijective():
    assert len(SPECIFIC_TO_GENERIC) == 40
    for specific, generic in SPECIFIC_TO_GENERIC.items():
        assert specific in GENERIC_TO_SPECIFIC[generic]
        assert parent_of(specific) == generic


def test_parents_of_dedupes():
    assert parents_of(["Taxes", "Unemployment", "Taxes"]) == ["Economy"]
    assert parents_of(["Taxes", "Climate Change"]) == ["Economy", "Environment"]
