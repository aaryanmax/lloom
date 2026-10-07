"""Synthesize operator (paper §3.1.1, prompt in Appendix A.3): GPT-4 in the
paper; here Gemini Flash-Lite by default. Proposes unifying high-level
"patterns" (the paper's prompt word for concepts) as JSON:
{name, prompt(criteria), example_ids}.
"""

from __future__ import annotations

import json as _json

SYNTHESIZE_PROMPT = """I have this set of bullet point summaries of text examples:
{bullets_json}

Please write a summary of {n_concepts} unifying patterns for these examples {seed_phrase}.
For each high-level pattern, write a {n_name_words} word NAME for the pattern
and an associated one-sentence ChatGPT PROMPT that could take in a new text
example and determine whether the relevant pattern applies.
Please also include {n_example_ids} example_ids for items that BEST exemplify
the pattern. The example_ids must be copied EXACTLY from the bullet summaries above.
Please respond ONLY with a valid JSON in the following format:
{{
  "patterns": [
    {{
      "name": "<PATTERN_NAME_1>",
      "prompt": "<PATTERN_PROMPT_1>",
      "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]
    }}
  ]
}}"""


def synthesize(client, cluster_items: list[dict], seed_phrase: str = "",
               n_concepts: int = 3, n_name_words: str = "2-4",
               n_example_ids: int = 2, model: str = "gemini") -> list[dict]:
    """cluster_items: [{"doc_id": ..., "bullet": ...}, ...] from cluster.py."""
    bullets = [{"example_id": it["doc_id"], "bullet": it["bullet"]}
               for it in cluster_items]
    out = client.chat_json(
        model,
        SYNTHESIZE_PROMPT.format(
            bullets_json=_json.dumps(bullets, ensure_ascii=False, indent=1),
            n_concepts=n_concepts, seed_phrase=seed_phrase,
            n_name_words=n_name_words, n_example_ids=n_example_ids),
        kind="synthesize",
    )
    concepts = []
    for p in out.get("patterns", []):
        name = str(p.get("name", "")).strip()
        criteria = str(p.get("prompt", "")).strip()
        if not name or not criteria:
            continue
        concepts.append({
            "name": name,
            "criteria_prompt": criteria,
            "representative_example_ids": [str(x) for x in p.get("example_ids", [])][:4],
        })
    return concepts[:n_concepts]
