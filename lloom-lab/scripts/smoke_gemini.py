"""Gemini free-tier smoke test (requires GEMINI_API_KEY env var).

    C:/Users/Admin/Projects/.supervenv/Scripts/python.exe scripts/smoke_gemini.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import GEMINI_DAILY_LIMIT, GEMINI_MODEL, make_client  # noqa: E402


def main():
    print(f"primary model={GEMINI_MODEL}  primary daily_limit={GEMINI_DAILY_LIMIT}")
    client = make_client()
    reply = client.chat("gemini", "Reply with exactly the word: OK", kind="smoke")
    print(f"[ok] gemini live round-trip -> {reply.strip()[:80]!r}")

    j = client.chat_json(
        "gemini",
        'List 2 unifying patterns of these posts as JSON: '
        '{"bullets": ["complains about rising rent", "blames landlords for prices"]}. '
        'Format: {"patterns": [{"name": "...", "prompt": "..."}]}',
        kind="smoke")
    assert isinstance(j.get("patterns"), list) and j["patterns"], j
    print(f"[ok] gemini JSON mode -> first concept: {j['patterns'][0]['name']!r}")

    # repeat = cache hit, zero quota cost
    client.chat("gemini", "Reply with exactly the word: OK", kind="smoke")
    fresh = sum(client.ledger.today_count(mid) for mid, _, _ in client.gemini_chain)
    print(f"[ok] cache+ledger working; today's fresh gemini-family calls: "
          f"{fresh}/{sum(d for _, d, _ in client.gemini_chain)}")


if __name__ == "__main__":
    main()
