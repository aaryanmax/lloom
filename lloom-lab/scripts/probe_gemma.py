"""Capability probe: is Gemma 4 31B good enough to be the E1 workhorse?

Rebuilds realistic Synthesize and matcher prompts from the pilot's n=100 run
record, sends the SAME prompts fresh (use_cache=False) to both
gemini-flash-lite-latest and gemma-4-31b-it, and scores:

  - structural validity (JSON parses, required keys, verbatim ids/names)
  - coherence (concept names/criteria read sensibly, example_ids real)
  - A/B agreement on the matcher batches (verdict-level)

Decision rule (recorded in the output): adopt Gemma as the E1 Synthesize
workhorse iff 100% of its calls are structurally valid and matcher agreement
with Flash-Lite is >= 0.7.

    C:/Users/Admin/Projects/.supervenv/Scripts/python.exe scripts/probe_gemma.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import CACHE_DB, LEDGER_DB, RUNS_DIR  # noqa: E402
from lloom_lab.llm.accounting import RequestLedger  # noqa: E402
from lloom_lab.llm.cache import ResponseCache  # noqa: E402
from lloom_lab.llm.client import LLMClient, extract_json  # noqa: E402
from lloom_lab.metrics.coverage import (  # noqa: E402
    _PROMPT_TEMPLATE,
    FEW_SHOT_EXAMPLE,
    _format_generated,
)
from lloom_lab.pipeline.synthesize import SYNTHESIZE_PROMPT  # noqa: E402

GEMMA_ID = "gemma-4-31b-it"
FLASHLITE_ID = "gemini-flash-lite-latest"
RUN_RECORD = RUNS_DIR / "pilot" / "n100_t0.json"
N_SYNTH_CLUSTERS = 4
N_MATCHER_BATCHES = 3
GT_BATCH_SIZE = 10


def make_single_model_client(model_id: str, rpm: int) -> LLMClient:
    """Client pinned to ONE Gemini-family model id (no failover), shared cache."""
    return LLMClient(
        cache=ResponseCache(CACHE_DB),
        ledger=RequestLedger(LEDGER_DB),
        gemini_chain=[(model_id, 10**9, rpm)],
    )


def build_synth_prompts(record: dict) -> list[dict]:
    """Rebuild synthesize prompts for a few real clusters of the pilot run."""
    bullets_by_doc = {d["doc_id"]: d["bullets"] for d in record["distilled"]}
    prompts = []
    clusters = sorted(record["clusters"].items(), key=lambda kv: -len(kv[1]))
    # one large, one mid, one small, one tiny cluster — cover the size range
    picks = [clusters[0], clusters[len(clusters) // 3],
             clusters[2 * len(clusters) // 3], clusters[-1]][:N_SYNTH_CLUSTERS]
    for label, doc_ids in picks:
        items = [{"doc_id": did, "bullet": b}
                 for did in doc_ids for b in bullets_by_doc[did]]
        bullets_json = json.dumps(
            [{"example_id": it["doc_id"], "bullet": it["bullet"]} for it in items],
            ensure_ascii=False, indent=1)
        prompt = SYNTHESIZE_PROMPT.format(
            bullets_json=bullets_json, n_concepts=1, seed_phrase="",
            n_name_words="2-4", n_example_ids=2)
        valid_ids = {it["doc_id"] for it in items}
        prompts.append({"kind": f"synthesize.cluster_{label}", "prompt": prompt,
                        "valid_ids": valid_ids, "n_items": len(items)})
    return prompts


def build_matcher_prompts(record: dict) -> list[dict]:
    generated = [{"name": c["name"], "criteria_prompt": c["criteria_prompt"]}
                 for c in record["concepts"]]
    gt_specific = record["ground_truth"]["specific"]
    prompts = []
    for i in range(N_MATCHER_BATCHES):
        gts = gt_specific[i * GT_BATCH_SIZE:(i + 1) * GT_BATCH_SIZE]
        prompt = _PROMPT_TEMPLATE.format(
            ground_truths="\n".join(f"- {g}" for g in gts),
            generated=_format_generated(generated)) + "\n" + FEW_SHOT_EXAMPLE
        prompts.append({"kind": f"matcher.batch_{i}", "prompt": prompt,
                        "gts": gts, "concept_names": {c["name"] for c in generated}})
    return prompts


def check_synth(out, valid_ids: set[str]) -> dict:
    try:
        data = extract_json(out) if isinstance(out, str) else out
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "why": f"JSON: {e}"}
    pats = data.get("patterns") if isinstance(data, dict) else None
    if not isinstance(pats, list) or not pats:
        return {"valid": False, "why": f"no patterns list: {str(data)[:120]!r}"}
    p = pats[0]
    name, criteria = str(p.get("name", "")).strip(), str(p.get("prompt", "")).strip()
    ids = [str(x) for x in p.get("example_ids", [])]
    if not name or not criteria:
        return {"valid": False, "why": f"empty name/criteria: {str(p)[:120]!r}"}
    bad_ids = [x for x in ids if x not in valid_ids]
    return {"valid": not bad_ids,
            "why": f"invented example_ids {bad_ids}" if bad_ids else "ok",
            "name": name, "criteria": criteria, "ids": ids}


def check_matcher(out, gts: list[str], concept_names: set[str]) -> dict:
    try:
        data = extract_json(out) if isinstance(out, str) else out
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "why": f"JSON: {e}"}
    matches = data.get("matches") if isinstance(data, dict) else None
    if not isinstance(matches, list):
        return {"valid": False, "why": f"no matches list: {str(data)[:120]!r}"}
    got = {m.get("ground_truth") for m in matches}
    missing = [g for g in gts if g not in got]
    invented_ids = [m.get("concept") for m in matches
                    if m.get("concept") is not None and m.get("concept") not in concept_names]
    n_matched = sum(1 for m in matches if m.get("concept") is not None)
    valid = not missing and not invented_ids
    return {"valid": valid,
            "why": (f"missing {missing[:2]} invented {invented_ids[:2]}"
                    if not valid else "ok"),
            "matched": n_matched, "of": len(gts)}


def main():
    record = json.loads(RUN_RECORD.read_text(encoding="utf-8"))
    flash = make_single_model_client(FLASHLITE_ID, rpm=15)
    gemma = make_single_model_client(GEMMA_ID, rpm=30)
    jobs = build_synth_prompts(record) + build_matcher_prompts(record)
    print(f"probe: {len(jobs)} prompts x 2 models (all fresh, uncached)\n")

    results = []
    for job in jobs:
        row = {"kind": job["kind"]}
        for tag, (mid, cli) in (("flashlite", (FLASHLITE_ID, flash)),
                                ("gemma", (GEMMA_ID, gemma))):
            t0 = time.time()
            # sentinel system string isolates probe entries in the shared cache
            # (cache keys include the system text), so re-runs are free and real
            # cached responses are never overwritten by a probe call
            system = f"probe:{mid}"
            try:
                out = cli.chat("gemini", job["prompt"], system=system,
                               kind="probe", use_cache=True)
                latency = time.time() - t0
                if "valid_ids" in job:
                    chk = check_synth(out, job["valid_ids"])
                else:
                    chk = check_matcher(out, job["gts"], job["concept_names"])
                chk["latency_s"] = round(latency, 1)
                row[tag] = chk
            except Exception as e:  # noqa: BLE001
                row[tag] = {"valid": False, "why": f"API error after {time.time()-t0:.0f}s: {e}",
                            "latency_s": round(time.time() - t0, 1)}
        results.append(row)
        f, g = row["flashlite"], row["gemma"]
        print(f"{job['kind']}:")
        print(f"  flashlite: valid={f['valid']} ({f.get('latency_s', '?')}s) "
              f"{f.get('name', f.get('matched', ''))} {f['why']}")
        print(f"  gemma:     valid={g['valid']} ({g.get('latency_s', '?')}s) "
              f"{g.get('name', g.get('matched', ''))} {g['why']}")

    # ---- verdict ------------------------------------------------------------
    synth = [r for r in results if r["kind"].startswith("synthesize")]
    match = [r for r in results if r["kind"].startswith("matcher")]
    g_synth_ok = sum(r["gemma"]["valid"] for r in synth)
    g_match_ok = sum(r["gemma"]["valid"] for r in match)
    agree = 0
    for r in match:
        fv, gv = r["flashlite"], r["gemma"]
        # verdict-level agreement: matched counts within 1 of each other
        if fv["valid"] and gv["valid"] and abs(fv["matched"] - gv["matched"]) <= 1:
            agree += 1
    n_match_cmp = sum(1 for r in match if r["flashlite"]["valid"] and r["gemma"]["valid"])
    print("\n==== SUMMARY ====")
    print(f"gemma synthesize valid: {g_synth_ok}/{len(synth)}")
    print(f"gemma matcher valid:    {g_match_ok}/{len(match)}")
    print(f"matcher agreement (matched-count within +/-1): {agree}/{n_match_cmp or len(match)}")
    adopt = (g_synth_ok == len(synth) and g_match_ok == len(match)
             and agree >= 0.7 * (n_match_cmp or len(match)))
    verdict = ("ADOPT Gemma as E1 workhorse" if adopt
               else "KEEP Flash-Lite primary; Gemma overflow only")
    print(f"VERDICT: {verdict}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "probe_gemma.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"details -> {out_path}")


if __name__ == "__main__":
    main()
