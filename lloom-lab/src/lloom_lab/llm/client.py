"""Unified LLM client.

Three logical models behind one interface:
  * "gemini" — Gemini free tier via google-genai (failover chain over model
    ids, each with its own daily pool; proactive RPM throttle) — matcher,
    zero-shot, and any precision-critical call
  * "gemma"  — Gemma 4 31B via the same API (big 14K/day pool, 30 RPM) —
    the high-volume Synthesize workhorse
  * "qwen"   — local Qwen3-8B Q4_K_M via llama.cpp llama-server (OpenAI-compatible)

All calls are cache-first (SQLite; keyed by the LOGICAL model label so
re-routing between gemini/gemma never collides or re-hits) and logged to the
request ledger. Fresh calls are logged under their concrete model id (e.g.
"gemini-flash-lite-latest") so each pool's daily quota is tracked
independently; cached calls are logged under the logical label.
"""

from __future__ import annotations

import json as _json
import os
import re
import threading
import time

import httpx

from lloom_lab.config import GEMINI_MODEL, QWEN_HTTP_TIMEOUT, QWEN_MODEL_NAME, QWEN_URL
from lloom_lab.llm.accounting import RequestLedger
from lloom_lab.llm.cache import ResponseCache


class LLMError(RuntimeError):
    pass


class QuotaExceeded(RuntimeError):
    pass


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

# Gemma models on the Gemini API don't honor system_instruction or JSON
# response_mime_type — their requests are prepared differently.
_GEMMA_PREFIX = "gemma"


def _is_transient_5xx_text(text: str) -> bool:
    return ("500" in text or "503" in text or "INTERNAL" in text
            or "UNAVAILABLE" in text or "DEADLINE_EXCEEDED" in text)


def _is_transient_5xx(err: Exception) -> bool:
    """True for server-side blips worth failing over for (not quota, not 4xx)."""
    text = str(err)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return False
    return _is_transient_5xx_text(text)


def pick_gemini_provider(chain: list[tuple[str, int, int]], ledger: RequestLedger,
                         legacy_label: str = "gemini") -> tuple[str, int, int]:
    """First (model_id, daily_limit, rpm) in the chain with quota left today.

    Rows logged before per-id accounting existed (label "gemini") count toward
    the first pool entry, which is the id they were actually sent to. Pass
    legacy_label="" for pools that never had legacy rows (e.g. gemma).
    """
    for i, spec in enumerate(chain):
        used = ledger.today_count(spec[0])
        if i == 0:
            used += ledger.today_count(legacy_label)
        if used < spec[1]:
            return spec
    ids = ", ".join(spec[0] for spec in chain)
    raise QuotaExceeded(
        f"All Gemini pools exhausted for today ({ids}) — resume tomorrow or "
        "raise GEMINI_DAILY_LIMIT / GEMINI_FALLBACK_DAILY_LIMIT.")


def prepare_gemini_request(model_id: str, prompt: str, system: str | None,
                           json_mode: bool) -> tuple[str, str | None, bool]:
    """Return (prompt, system, json_mode) adapted to the model's API support.

    Gemma models ignore system_instruction and JSON response_mime_type, so the
    system text is folded into the user prompt and JSON is enforced by our
    extractor + repair retry instead of the response config.
    """
    if model_id.lower().startswith(_GEMMA_PREFIX):
        folded = f"{system}\n\n{prompt}" if system else prompt
        return folded, None, False
    return prompt, system, json_mode


def extract_json(text: str):
    """Robustly pull the first JSON object/array out of an LLM response.

    Handles code fences, leading prose, and trailing commentary — small local
    models rarely emit *clean* JSON, so the parser must be forgiving.
    """
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[start:i + 1])
                    except _json.JSONDecodeError:
                        break
    raise LLMError(f"no valid JSON found in response: {text[:200]!r}")


class LLMClient:
    def __init__(self, cache: ResponseCache, ledger: RequestLedger,
                 gemini_chain: list[tuple[str, int, int]] | None = None,
                 gemini_model: str = GEMINI_MODEL, daily_gemini_limit: int = 500,
                 gemma_chain: list[tuple[str, int, int]] | None = None):
        self.cache = cache
        self.ledger = ledger
        if gemini_chain is None:
            gemini_chain = [(gemini_model, daily_gemini_limit, 15)]
        self.gemini_chain = gemini_chain
        if gemma_chain is None:
            from lloom_lab.config import GEMMA_DAILY_LIMIT, GEMMA_MODEL, GEMMA_RPM_LIMIT
            gemma_chain = [(GEMMA_MODEL, GEMMA_DAILY_LIMIT, GEMMA_RPM_LIMIT)]
        self.gemma_chain = gemma_chain[0]
        self._genai_client = None
        self._http = httpx.Client(timeout=QWEN_HTTP_TIMEOUT)
        self._sleep = time.sleep  # monkeypatchable in tests

    # ---- public API ---------------------------------------------------------
    def chat(self, model: str, prompt: str, system: str | None = None,
             temperature: float = 0.0, json_mode: bool = False,
             kind: str = "chat", max_retries: int = 3,
             use_cache: bool = True) -> str:
        """Send one chat completion; returns the response text."""
        key_parts = (model, system or "", prompt, temperature, json_mode)
        if use_cache:
            hit = self.cache.get(key_parts)
            if hit is not None:
                self.ledger.log(model, kind, cached=True)
                return hit

        if model == "gemini":
            model_id, daily, rpm = pick_gemini_provider(self.gemini_chain, self.ledger)
            self._throttle(model_id, rpm)
            out, t_in, t_out = self._call_gemini(model_id, prompt, system, temperature,
                                                 json_mode, max_retries)
            ledger_model = model_id
        elif model == "gemma":
            # Gemma 4 31B: big daily pool (14K) at 30 RPM — used for the
            # high-volume Synthesize step; NOT for the matcher (probe showed
            # 500 INTERNAL errors on large matcher prompts). Gemma's free-tier
            # serving is flaky (intermittent 500 INTERNAL), so on persistent
            # 5xx the call fails over to the Flash-Lite chain rather than
            # losing the cluster.
            model_id, _, rpm = pick_gemini_provider(
                [self.gemma_chain], self.ledger, legacy_label="")
            self._throttle(model_id, rpm)
            try:
                out, t_in, t_out = self._call_gemini(model_id, prompt, system,
                                                     temperature, json_mode, 5)
            except LLMError as e:
                if not _is_transient_5xx(e):
                    raise
                fb_id, _, fb_rpm = pick_gemini_provider(self.gemini_chain,
                                                        self.ledger)
                self._throttle(fb_id, fb_rpm)
                out, t_in, t_out = self._call_gemini(fb_id, prompt, system,
                                                     temperature, json_mode,
                                                     max_retries)
                model_id = fb_id
            ledger_model = model_id
        elif model == "qwen":
            out, t_in, t_out = self._call_qwen(prompt, system, temperature,
                                               json_mode, max_retries)
            ledger_model = model
        else:
            raise LLMError(f"unknown model '{model}' (use 'gemini', 'gemma' or 'qwen')")

        self.cache.put(key_parts, model, out)
        self.ledger.log(ledger_model, kind, cached=False, tokens_in=t_in, tokens_out=t_out)
        return out

    def chat_json(self, model: str, prompt: str, **kwargs):
        """chat() + robust JSON extraction."""
        kwargs.setdefault("json_mode", True)
        kwargs.setdefault("kind", "json")
        return extract_json(self.chat(model, prompt, **kwargs))

    # ---- providers ----------------------------------------------------------
    # Process-global throttle state: parallel grid workers each hold their own
    # LLMClient, but the RPM cap belongs to the API key, so spacing must be
    # shared across ALL client instances in this process.
    _FRESH_TS: dict[str, float] = {}
    _THROTTLE_LOCK = threading.Lock()

    def _throttle(self, model_id: str, rpm: int) -> None:
        """Proactively space fresh calls so we never breach the RPM cap.

        Books each caller a unique send slot at least min_interval after the
        previous slot for this model id (thread-safe, process-global), then
        sleeps until its own slot outside the lock.
        """
        min_interval = 60.0 / rpm * 1.15
        with LLMClient._THROTTLE_LOCK:
            now = time.monotonic()
            slot = max(now, LLMClient._FRESH_TS.get(model_id, 0.0) + min_interval)
            LLMClient._FRESH_TS[model_id] = slot
        wait = slot - now
        if wait > 0:
            self._sleep(wait)

    def _call_gemini(self, model_id, prompt, system, temperature, json_mode, max_retries):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY env var is not set.")
        if self._genai_client is None:
            self._genai_client = genai.Client(api_key=api_key)
        prompt, system, json_mode = prepare_gemini_request(model_id, prompt, system, json_mode)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
        )
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._genai_client.models.generate_content(
                    model=model_id, contents=prompt, config=cfg)
                usage = getattr(resp, "usage_metadata", None)
                t_in = getattr(usage, "prompt_token_count", 0) or 0
                t_out = getattr(usage, "candidates_token_count", 0) or 0
                return resp.text, t_in, t_out
            except Exception as e:
                last_err = e
                text = str(e)
                if "429" in text or "RESOURCE_EXHAUSTED" in text:
                    # Free-tier per-minute throttling is routine in 2026 — back
                    # off and retry within the same request instead of failing
                    # the whole run. (Our own *daily* limits are enforced
                    # separately by the ledger check in pick_gemini_provider.)
                    time.sleep(min(30 * (attempt + 1), 120))
                elif _is_transient_5xx_text(text):
                    # Gemma's free-tier serving intermittently 500s — retry
                    # with a longer, growing wait (the 2s/4s default gave up
                    # on recoverable blips during the first E1 smoke).
                    time.sleep(min(5 * 2 ** attempt, 60))
                else:
                    time.sleep(min(2 ** attempt * 2, 30))
        raise LLMError(f"Gemini call to {model_id} failed after {max_retries} retries: {last_err}")

    def _call_qwen(self, prompt, system, temperature, json_mode, max_retries):
        # "/no_think" switches Qwen3's reasoning mode off — required for strict,
        # fast JSON output from the 8B model.
        sys_text = (system or "") + "\n/no_think"
        messages = [{"role": "system", "content": sys_text},
                    {"role": "user", "content": prompt}]
        payload = {
            "model": QWEN_MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                r = self._http.post(f"{QWEN_URL}/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
                msg = data["choices"][0]["message"]
                usage = data.get("usage", {})
                content = msg.get("content") or ""
                if not content.strip() and msg.get("reasoning_content"):
                    # model ignored /no_think and put everything in reasoning —
                    # fall back to parsing the reasoning text
                    content = msg["reasoning_content"]
                return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt * 2, 30))
        raise LLMError(
            f"Qwen call failed after {max_retries} retries: {last_err}\n"
            f"Is llama-server running?  {QWEN_URL}")

    def close(self):
        self._http.close()
