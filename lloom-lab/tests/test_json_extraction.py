"""Robust JSON extraction from messy small-model output."""

import pytest

from lloom_lab.llm.client import LLMError, extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_leading_prose_and_trailing_commentary():
    text = 'Sure! Here is the JSON you asked for:\n{"a": {"b": 2}}\nHope that helps!'
    assert extract_json(text) == {"a": {"b": 2}}


def test_array():
    assert extract_json('noise [1, 2, 3] noise') == [1, 2, 3]


def test_no_json_raises():
    with pytest.raises(LLMError):
        extract_json("no structured data here at all")
