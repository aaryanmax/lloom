"""Sentence-pool constraints: exact splitting, cleanliness, no cross-pool dupes."""

import re

from lloom_lab.data.concepts import ALL_SPECIFIC
from lloom_lab.data.sentences import FILLERS, SEED_SENTENCES

# A sentence must contain no internal sentence-ending punctuation, so the
# generator's regex splitter counts exactly one sentence per string.
_SPLIT = re.compile(r"(?<=[.!?])\s+")


def test_every_specific_concept_has_a_pool():
    assert set(SEED_SENTENCES) == set(ALL_SPECIFIC)


def test_pool_sizes():
    for concept, pool in SEED_SENTENCES.items():
        assert len(pool) >= 8, concept


def test_sentences_split_exactly_once():
    for pool_name, pool in [*SEED_SENTENCES.items(), ("fillers", FILLERS)]:
        for s in pool:
            assert len(_SPLIT.split(s)) == 1, (pool_name, s)
            assert s == s.strip(), (pool_name, s)
            assert s[0].isupper(), (pool_name, s)


def test_no_duplicates_within_or_across_pools():
    all_sents = [s for pool in SEED_SENTENCES.values() for s in pool] + FILLERS
    assert len(all_sents) == len(set(all_sents))
