"""Score operator (paper §3.1.2, prompt in Appendix A.4): batched zero-shot
multiple-choice labeling of every example against a concept's criteria prompt.
Defaults to the local Qwen model (the paper's costliest step — keep it off the
cloud quota). Answers A–E map to 0.0–1.0.
"""

from __future__ import annotations

import json as _json

from lloom_lab.config import SCORE_OPTIONS, SCORE_TO_VALUE

SCORE_PROMPT = """CONTEXT:
I have the following text examples in a JSON:
{examples_json}

I also have a pattern named {concept_name} with the following PROMPT:
{concept_prompt}

TASK:
For each example, please evaluate the PROMPT by generating RATIONALE of your
thought process and providing a resulting ANSWER of ONE of the following
multiple-choice options, including just the letter:
{options}

Respond with ONLY a JSON with the following format:
{{
  "pattern_results": [
    {{
      "example_id": "<example_id>",
      "rationale": "<rationale>",
      "answer": "<A|B|C|D|E>"
    }}
  ]
}}"""


def score_examples(client, concept_name: str, concept_prompt: str,
                   examples: dict[str, str], model: str = "qwen",
                   batch_size: int = 20) -> dict[str, dict]:
    """Score every example. Returns {example_id: {"score": 0-1, "answer": "A"..,
    "rationale": ...}}. Batched: `batch_size` examples per call."""
    results: dict[str, dict] = {}
    ids = list(examples)
    for i in range(0, len(ids), batch_size):
        chunk_ids = ids[i:i + batch_size]
        payload = [{"example_id": eid, "text": examples[eid]} for eid in chunk_ids]
        out = client.chat_json(
            model,
            SCORE_PROMPT.format(examples_json=_json.dumps(payload, ensure_ascii=False),
                                concept_name=concept_name,
                                concept_prompt=concept_prompt,
                                options="\n".join(f"- {o}" for o in SCORE_OPTIONS)),
            kind="score",
        )
        for row in out.get("pattern_results", []):
            eid = str(row.get("example_id", ""))
            ans = str(row.get("answer", "E")).strip().upper()[:1]
            if eid in examples and ans in SCORE_TO_VALUE:
                results[eid] = {"score": SCORE_TO_VALUE[ans], "answer": ans,
                                "rationale": str(row.get("rationale", ""))}
    # anything the model skipped counts as a non-match
    for eid in ids:
        results.setdefault(eid, {"score": 0.0, "answer": None, "rationale": "missing"})
    return results
