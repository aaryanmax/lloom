"""Cache + ledger behavior on temporary SQLite files (no network)."""

import pytest

from lloom_lab.llm.accounting import RequestLedger
from lloom_lab.llm.cache import ResponseCache
from lloom_lab.llm.client import (
    QuotaExceeded,
    pick_gemini_provider,
    prepare_gemini_request,
)


def test_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite3")
    key = ("gemini", "sys", "prompt", 0.0, True)
    assert cache.get(key) is None
    cache.put(key, "gemini", "answer")
    assert cache.get(key) == "answer"
    assert cache.stats()["entries"] == 1


def test_cache_key_sensitivity(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite3")
    cache.put(("gemini", "", "p1", 0.0, False), "gemini", "a1")
    assert cache.get(("gemini", "", "p2", 0.0, False)) is None
    assert cache.get(("qwen", "", "p1", 0.0, False)) is None


def test_ledger_counts_and_report(tmp_path):
    ledger = RequestLedger(tmp_path / "l.sqlite3")
    ledger.log("gemini", "json", cached=False, tokens_in=10, tokens_out=5)
    ledger.log("gemini", "json", cached=True)
    ledger.log("qwen", "chat", cached=False, tokens_in=1, tokens_out=2)
    # today_count counts only non-cached calls — that is what the quota guard uses.
    assert ledger.today_count("gemini") == 1
    assert ledger.today_count("gemini", cached=True) == 1
    assert ledger.today_count("qwen") == 1
    report = {(m, k): (n, ti, to) for m, k, n, ti, to in ledger.daily_report()}
    assert report[("gemini", "json")] == (2, 10, 5)
    assert report[("qwen", "chat")] == (1, 1, 2)


CHAIN = [("model-a", 3, 15), ("model-b", 2, 15), ("gemma-x", 10, 30)]


def test_provider_chain_picks_first_pool_with_quota(tmp_path):
    ledger = RequestLedger(tmp_path / "l.sqlite3")
    assert pick_gemini_provider(CHAIN, ledger)[0] == "model-a"
    for _ in range(3):
        ledger.log("model-a", "json", cached=False)
    # primary exhausted -> first fallback
    assert pick_gemini_provider(CHAIN, ledger)[0] == "model-b"
    for _ in range(2):
        ledger.log("model-b", "json", cached=False)
    # fallbacks exhausted -> gemma pool (10/day) still open
    assert pick_gemini_provider(CHAIN, ledger)[0] == "gemma-x"


def test_provider_chain_legacy_rows_count_toward_primary(tmp_path):
    ledger = RequestLedger(tmp_path / "l.sqlite3")
    # pre-per-id-labeling rows were logged under the logical label "gemini"
    for _ in range(3):
        ledger.log("gemini", "json", cached=False)
    assert pick_gemini_provider(CHAIN, ledger)[0] == "model-b"


def test_provider_chain_exhaustion_raises(tmp_path):
    ledger = RequestLedger(tmp_path / "l.sqlite3")
    for _ in range(3):
        ledger.log("model-a", "json", cached=False)
    for _ in range(2):
        ledger.log("model-b", "json", cached=False)
    for _ in range(10):
        ledger.log("gemma-x", "json", cached=False)
    with pytest.raises(QuotaExceeded):
        pick_gemini_provider(CHAIN, ledger)


def test_provider_chain_cached_calls_are_free(tmp_path):
    ledger = RequestLedger(tmp_path / "l.sqlite3")
    for _ in range(5):
        ledger.log("model-a", "json", cached=True)
    assert pick_gemini_provider(CHAIN, ledger)[0] == "model-a"


def test_throttle_spaces_fresh_calls():
    from lloom_lab.llm.client import LLMClient

    LLMClient._FRESH_TS.clear()
    client = LLMClient(cache=ResponseCache(":memory:"), ledger=RequestLedger(":memory:"))
    sleeps: list[float] = []
    client._sleep = sleeps.append
    # first call: never sleeps
    client._throttle("m", rpm=15)
    assert sleeps == []
    # second call immediately after: must wait ~60/15*1.15 = 4.6s
    client._throttle("m", rpm=15)
    assert len(sleeps) == 1 and 4.0 < sleeps[0] <= 4.7
    # after the wait elapses, no further sleep
    LLMClient._FRESH_TS["m"] -= 10.0
    client._throttle("m", rpm=15)
    assert len(sleeps) == 1
    # a different model id has its own independent slot timeline
    client._throttle("other", rpm=15)
    assert len(sleeps) == 1


def test_throttle_slots_are_process_global():
    """Two client instances (parallel grid workers) must share one slot
    timeline per model id — 3 immediate bookings at 15 RPM => ~2 slots of wait."""
    from lloom_lab.llm.client import LLMClient

    LLMClient._FRESH_TS.clear()
    clients = [LLMClient(cache=ResponseCache(":memory:"),
                         ledger=RequestLedger(":memory:")) for _ in range(3)]
    sleeps: list[float] = []
    for c in clients:
        c._sleep = sleeps.append
        c._throttle("m", rpm=15)
    # bookings 2 and 3 must wait ~1 and ~2 intervals respectively
    assert len(sleeps) == 2
    assert 3.5 < sleeps[0] <= 4.7 and 7.0 < sleeps[1] <= 9.4


def test_gemma_request_folds_system_and_drops_json_mode():
    p, s, jm = prepare_gemini_request("gemma-4-31b-it", "user prompt", "be terse", True)
    assert p == "be terse\n\nuser prompt" and s is None and jm is False
    # Gemini models pass through untouched
    p, s, jm = prepare_gemini_request("gemini-flash-lite-latest", "u", "s", True)
    assert (p, s, jm) == ("u", "s", True)


def test_gemma_logical_model_routes_and_quota_guards(tmp_path):
    """chat('gemma') hits the gemma pool, logs under its concrete id, and
    respects its own daily limit — independent of the gemini chain."""
    from unittest.mock import patch

    from lloom_lab.llm.cache import ResponseCache
    from lloom_lab.llm.client import LLMClient

    ledger = RequestLedger(tmp_path / "l.sqlite3")
    client = LLMClient(cache=ResponseCache(tmp_path / "c.sqlite3"), ledger=ledger)
    assert client.gemma_chain[0] == "gemma-4-31b-it"
    with patch.object(client, "_call_gemini", return_value=("{}", 1, 1)) as m:
        client.chat("gemma", "p", json_mode=True)
        sent_id = m.call_args.args[0]
    assert sent_id == "gemma-4-31b-it"
    # fresh call logged under the concrete id; cache entry under the logical label
    assert ledger.today_count("gemma-4-31b-it") == 1
    assert ledger.today_count("gemini-flash-lite-latest") == 0
    # second identical call = cache hit, no quota usage
    client.chat("gemma", "p", json_mode=True)
    assert ledger.today_count("gemma-4-31b-it") == 1
    # exhausting the gemma pool raises, even with the gemini chain untouched
    for _ in range(client.gemma_chain[1]):
        ledger.log("gemma-4-31b-it", "json", cached=False)
    with pytest.raises(QuotaExceeded):
        client.chat("gemma", "other prompt")
