"""Distill operator (paper §3.1.1): Filter (extractive quotes) + Summarize
(abstractive bullet points). Defaults to the local Qwen model — high-volume,
mechanical calls stay off the Gemini quota."""

from __future__ import annotations

import json as _json

FILTER_PROMPT = """I have the following TEXT EXAMPLE:
{text_example_json}

Please extract {n_quotes} QUOTES exactly copied from this EXAMPLE {seed_phrase}.
Please respond ONLY with a valid JSON in the following format:
{{
  "relevant_quotes": ["<QUOTE_1>", "<QUOTE_2>", ...]
}}"""

SUMMARIZE_PROMPT = """I have the following TEXT EXAMPLE:
{text_example_json}

Please summarize the main point of this EXAMPLE {seed_phrase} into {n_bullets} bullet points, where each bullet point is a {n_words} word phrase.
Please respond ONLY with a valid JSON in the following format:
{{
  "bullets": ["<BULLET_1>", "<BULLET_2>", ...]
}}"""


def filter_quotes(client, doc_text: str, seed_phrase: str = "",
                  n_quotes: str = "any number of", model: str = "qwen") -> list[str]:
    out = client.chat_json(
        model,
        FILTER_PROMPT.format(text_example_json=_json_doc(doc_text),
                             n_quotes=n_quotes, seed_phrase=seed_phrase),
        kind="distill.filter",
    )
    quotes = out.get("relevant_quotes", [])
    return [str(q).strip() for q in quotes if str(q).strip()][:10]


def summarize_bullets(client, doc_text: str, seed_phrase: str = "",
                      n_bullets: str = "2-4", n_words: str = "5-8",
                      model: str = "qwen") -> list[str]:
    out = client.chat_json(
        model,
        SUMMARIZE_PROMPT.format(text_example_json=_json_doc(doc_text),
                                seed_phrase=seed_phrase, n_bullets=n_bullets,
                                n_words=n_words),
        kind="distill.summarize",
    )
    bullets = out.get("bullets", [])
    return [str(b).strip() for b in bullets if str(b).strip()][:6]


def distill(client, doc_text: str, seed_phrase: str = "",
            use_filter: bool = False, model: str = "qwen") -> dict:
    """Full Distill step. Filter is skippable for short texts (per the paper)."""
    quotes = filter_quotes(client, doc_text, seed_phrase, model=model) if use_filter else []
    text_for_summary = doc_text if not use_filter else (
        doc_text + "\n\nRelevant excerpts: " + " | ".join(quotes))
    bullets = summarize_bullets(client, text_for_summary, seed_phrase, model=model)
    return {"quotes": quotes, "bullets": bullets}


def _json_doc(text: str) -> str:
    return _json.dumps({"text": text}, ensure_ascii=False)
