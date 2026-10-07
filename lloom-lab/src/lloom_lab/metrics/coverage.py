"""Automated concept-coverage metric (paper §5.1 / Appendix A.5).

A few-shot LLM matcher maps each ground-truth concept to at most one generated
concept ("exact meaning" matches only, with a written rationale). Coverage =
matched ground truths / total ground truths.

Quota strategy: ground truths are batched (default 10 per request), every
response goes through the cache-first client, and matching uses temperature 0
— so re-scoring a run costs zero API calls.
"""

from __future__ import annotations

DEFAULT_BATCH_SIZE = 10

_PROMPT_TEMPLATE = """You are scoring an automated topic-matching benchmark.

GROUND-TRUTH CONCEPTS (from a verified label set):
{ground_truths}

GENERATED CONCEPTS (induced from documents by a system, each with its inclusion criterion):
{generated}

TASK: For EACH ground-truth concept, decide whether EXACTLY ONE generated concept
matches it in meaning. Rules:
- Match only if the generated concept means the same thing as the ground truth —
  not merely related, broader, narrower, or a combination of several ground truths.
- Each generated concept may be matched to at most one ground truth (no reuse).
- If no generated concept matches a ground truth, use null.

Respond with JSON only:
{{"matches": [{{"ground_truth": "<verbatim ground truth>",
               "concept": "<generated concept name or null>",
               "rationale": "<one sentence>"}}]}}
"""

FEW_SHOT_EXAMPLE = """Example judgments for calibration:
- ground truth "Healthcare" vs generated "Universal Healthcare" -> null (narrower)
- ground truth "Healthcare" vs generated "Healthcare Policy Debates" -> match
- ground truth "Taxes" vs generated "Taxes and Government Spending" -> null (combination)
- ground truth "Election Campaigns" vs generated "Voter and Campaign Activity" -> null
  (umbrella: mixes campaigns with unrelated facets like voter registration)
- ground truth "Political Parties" vs generated "Factional Conflict and Splits" -> null
  (narrower: covers one facet, not the whole concept)
When in doubt between a match and a null, choose null — this metric is used to
compare methods, so false matches are worse than missed matches.
"""


def _format_generated(generated: list[dict]) -> str:
    lines = []
    for i, c in enumerate(generated, 1):
        lines.append(f"{i}. {c['name']} — criterion: {c['criteria_prompt']}")
    return "\n".join(lines)


def match_batch(client, ground_truths: list[str], generated: list[dict],
                model: str = "gemini", few_shot: bool = True) -> list[dict]:
    """One matcher call for a batch of ground truths."""
    prompt = _PROMPT_TEMPLATE.format(
        ground_truths="\n".join(f"- {g}" for g in ground_truths),
        generated=_format_generated(generated),
    )
    if few_shot:
        prompt += "\n" + FEW_SHOT_EXAMPLE
    data = client.chat_json(model, prompt, temperature=0.0, kind="coverage")
    matches = data.get("matches", data) if isinstance(data, dict) else data

    out: list[dict] = []
    gt_set = set(ground_truths)
    names = {c["name"] for c in generated}
    for m in matches:
        gt, concept = m.get("ground_truth"), m.get("concept")
        if gt not in gt_set:
            continue  # model hallucinated a ground truth — drop
        if concept is not None and concept not in names:
            concept = None  # model invented a concept name — treat as no match
        out.append({"ground_truth": gt, "concept": concept,
                    "rationale": m.get("rationale", "")})
    # Guarantee exactly one row per requested ground truth.
    seen = {r["ground_truth"] for r in out}
    for gt in ground_truths:
        if gt not in seen:
            out.append({"ground_truth": gt, "concept": None, "rationale": "missing from response"})
    return out


def match_concepts(client, ground_truths: list[str], generated: list[dict],
                   model: str = "gemini", batch_size: int = DEFAULT_BATCH_SIZE,
                   ) -> list[dict]:
    """Match all ground truths (batched); returns one row per ground truth."""
    results: list[dict] = []
    for i in range(0, len(ground_truths), batch_size):
        results.extend(match_batch(client, ground_truths[i:i + batch_size],
                                   generated, model=model))
    return results


def coverage(match_rows: list[dict]) -> float:
    """Fraction of ground truths matched (0..1)."""
    if not match_rows:
        return 0.0
    return sum(1 for r in match_rows if r["concept"] is not None) / len(match_rows)


def coverage_split(match_rows: list[dict], gt_types: dict[str, str]) -> dict[str, float]:
    """Coverage per ground-truth type, e.g. {"generic": ..., "specific": ...}.

    `gt_types` maps ground-truth name -> type ("generic" / "specific").
    """
    out: dict[str, float] = {}
    by_type: dict[str, list[dict]] = {}
    for r in match_rows:
        by_type.setdefault(gt_types[r["ground_truth"]], []).append(r)
    for t, rows in by_type.items():
        out[t] = coverage(rows)
    return out


def parse_generated(concepts: list[dict]) -> list[dict]:
    """Adapter from pipeline output to matcher input (defensive copy)."""
    return [{"name": c["name"], "criteria_prompt": c["criteria_prompt"]}
            for c in concepts]


__all__ = [
    "DEFAULT_BATCH_SIZE", "match_batch", "match_concepts", "coverage",
    "coverage_split", "parse_generated",
]
