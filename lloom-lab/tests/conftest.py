"""Shared fixtures. Tests never touch the network, GPU, or the real results DBs."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeClient:
    """Stands in for LLMClient: records prompts, returns canned responses."""

    def __init__(self, responses: list | dict | None = None):
        self.calls: list[dict] = []
        self._responses = list(responses or [])
        self._fixed = responses if isinstance(responses, dict) else None

    def chat_json(self, model, prompt, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        if self._fixed is not None:
            return self._fixed
        if not self._responses:
            raise AssertionError("no canned response left for prompt")
        return self._responses.pop(0)


@pytest.fixture
def fake_client():
    return FakeClient()
