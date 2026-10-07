"""Coverage matcher against a fake client (0 API calls)."""

from lloom_lab.metrics import coverage as cov

GENERATED = [
    {"name": "Healthcare Policy Debates", "criteria_prompt": "Does it discuss health policy?"},
    {"name": "Tax Policy", "criteria_prompt": "Does it discuss taxation?"},
    {"name": "Climate Action", "criteria_prompt": "Does it discuss climate?"},
]

GROUND = ["Healthcare", "Taxes", "Environment", "Gun Control"]

RESPONSE = {
    "matches": [
        {"ground_truth": "Healthcare", "concept": "Healthcare Policy Debates",
         "rationale": "same meaning"},
        {"ground_truth": "Taxes", "concept": "Tax Policy", "rationale": "same meaning"},
        {"ground_truth": "Environment", "concept": "Climate Action", "rationale": "same meaning"},
        {"ground_truth": "Gun Control", "concept": None, "rationale": "nothing matches"},
    ]
}


class BatchFakeClient:
    def __init__(self):
        self.calls = []

    def chat_json(self, model, prompt, **kwargs):
        self.calls.append(prompt)
        return RESPONSE


def test_match_concepts_batches_and_normalizes():
    client = BatchFakeClient()
    rows = cov.match_concepts(client, GROUND, GENERATED, batch_size=2)
    assert len(client.calls) == 2  # 4 ground truths / batch 2
    assert len(rows) == 4
    by_gt = {r["ground_truth"]: r["concept"] for r in rows}
    assert by_gt["Healthcare"] == "Healthcare Policy Debates"
    assert by_gt["Gun Control"] is None


def test_hallucinated_names_are_nulled():
    client = BatchFakeClient()
    rows = cov.match_concepts(client, GROUND, GENERATED)
    assert all(r["concept"] in {None} | {c["name"] for c in GENERATED} for r in rows)


def test_coverage_and_split():
    rows = cov.match_concepts(BatchFakeClient(), GROUND, GENERATED)
    assert cov.coverage(rows) == 0.75
    types = {"Healthcare": "generic", "Taxes": "specific",
             "Environment": "specific", "Gun Control": "specific"}
    split = cov.coverage_split(rows, types)
    assert split["generic"] == 1.0
    assert split["specific"] == 2 / 3


def test_missing_row_is_backfilled():
    resp = {"matches": RESPONSE["matches"][:1]}  # model dropped 3 rows

    class DropClient(BatchFakeClient):
        def chat_json(self, model, prompt, **kwargs):
            self.calls.append(prompt)
            return resp

    rows = cov.match_concepts(DropClient(), GROUND, GENERATED)
    assert len(rows) == 4
    assert {r["ground_truth"] for r in rows} == set(GROUND)


def test_parse_generated_adapter():
    out = cov.parse_generated([{"name": "A", "criteria_prompt": "c?", "extra": 1}])
    assert out == [{"name": "A", "criteria_prompt": "c?"}]
