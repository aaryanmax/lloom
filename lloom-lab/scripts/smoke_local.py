"""Local smoke test: cache, ledger, JSON extraction, and a Qwen round-trip.

Run with the supervenv python AFTER starting llama-server (scripts/start_qwen_server.bat):
    C:/Users/Admin/Projects/.supervenv/Scripts/python.exe scripts/smoke_local.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import make_client  # noqa: E402
from lloom_lab.llm import extract_json  # noqa: E402


def test_extract_json():
    cases = [
        '{"a": 1}',
        'Sure! Here is the JSON:\n```json\n{"a": [1, 2], "b": "x"}\n```',
        'blah blah {"name": "Victim Blaming", "prompt": "Does it?"} hope that helps',
        'junk [1, 2, 3] trailing',
    ]
    assert extract_json(cases[0]) == {"a": 1}
    assert extract_json(cases[1]) == {"a": [1, 2], "b": "x"}
    assert extract_json(cases[2])["name"] == "Victim Blaming"
    assert extract_json(cases[3]) == [1, 2, 3]
    print("[ok] extract_json handles fences, prose, and balanced-scan fallback")


def test_cache_and_ledger():
    client = make_client()
    parts = ("fake-model", "", "hello", 0.0, False)
    client.cache.put(parts, "fake-model", "cached-reply")
    assert client.cache.get(parts) == "cached-reply"
    print(f"[ok] cache round-trip ({client.cache.stats()})")
    client.ledger.log("fake-model", "test", cached=True)
    print(f"[ok] ledger writes; today's (cached) rows: "
          f"{client.ledger.today_count('fake-model')}")


def test_qwen_live():
    client = make_client()
    reply = client.chat("qwen", "Reply with exactly the word: OK", kind="smoke")
    print(f"[ok] qwen live round-trip -> {reply.strip()[:80]!r}")
    j = client.chat_json("qwen", 'Return JSON {"ok": true, "n": 3}', kind="smoke")
    assert j.get("ok") is True and j.get("n") == 3, j
    print("[ok] qwen JSON mode works")
    # second call must be served from cache (ledger gets a cached=True row)
    before = client.ledger.today_count("qwen", cached=True)
    client.chat_json("qwen", 'Return JSON {"ok": true, "n": 3}', kind="smoke")
    assert client.ledger.today_count("qwen", cached=True) == before + 1
    print("[ok] cache hit on repeat call (0 extra API cost)")


if __name__ == "__main__":
    test_extract_json()
    test_cache_and_ledger()
    if "--live" in sys.argv:
        test_qwen_live()
    print("\nALL SMOKE TESTS PASSED")
