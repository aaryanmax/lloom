"""Week-1 milestone: one full mini-LLooM iteration end-to-end.

    python scripts/smoke_e2e.py [--synth qwen|gemini]

Runs Distill (qwen) -> Cluster (bge-small + HDBSCAN) -> Synthesize (gemini or
qwen) -> Score (qwen) over six toy documents with two obvious themes, and
prints the induced concepts with scores.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import make_client  # noqa: E402
from lloom_lab.pipeline.lloom import run_iteration  # noqa: E402

DOCS = {
    "d1": "My landlord raised the rent by 300 dollars this month. No repairs have been done in a year and he still wants more. Renting in this city is impossible.",
    "d2": "Housing prices are insane. We pay half our salary in rent and the landlord does nothing. Something must be done about the rental market.",
    "d3": "Just got the lease renewal: another rent increase. Landlords keep squeezing tenants while housing supply stays low. Unaffordable.",
    "d4": "New phone update destroyed my battery life. It drains from 100 to 20 percent by noon. The battery is unusable after this software update.",
    "d5": "The latest firmware killed my phone battery. Barely lasts three hours now. Software updates should not wreck battery health like this.",
    "d6": "Battery drains so fast since the update, and the phone gets hot while charging. Clearly a bad software patch affecting battery life.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="gemini", choices=["gemini", "qwen"])
    args = ap.parse_args()

    client = make_client()
    print(f"Running one LLooM iteration over {len(DOCS)} docs "
          f"(distill=qwen, synth={args.synth}, score=qwen)...\n")
    out = run_iteration(client, DOCS, min_cluster_size=2,
                        n_concepts_per_cluster=2, synth_model=args.synth)

    print(f"Induced {len(out['concepts'])} concepts "
          f"from {len(out['clusters'])} bullet clusters:\n")
    for c in out["concepts"]:
        scores = out["scores"][c["name"]]
        matched = [did for did, s in scores.items() if s["score"] >= 1.0]
        print(f"- {c['name']}  (cluster {c['cluster_label']})")
        print(f"  criteria: {c['criteria_prompt']}")
        print(f"  matched(1.0): {matched or '-'}")
    print("\nfull result saved -> results/e2e_milestone.json")
    (Path(__file__).resolve().parent.parent / "results" / "e2e_milestone.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
