"""
llm_client.py — Unified LLM client supporting Ollama (local).

Switch providers by changing config.yaml:

  # Local (Ollama):
  provider: "ollama"
  model: "gemma3:7b"       # or "gemma3:27b"
"""

import json
import os
import re
import time

import requests

from config import cfg


PROVIDER   = cfg["llm"]["provider"]
MODEL      = cfg["llm"]["model"]
TEMP       = cfg["llm"]["temperature"]
MAX_TOKENS = cfg["llm"]["max_tokens"]

MAX_RETRIES = 3
RETRY_DELAY = 2


# ── Health check (provider-aware) ─────────────────────────────────────────────

def check_llm_ready() -> bool:
    """Verify the configured LLM provider is reachable."""
    if PROVIDER == "ollama":
        return _check_ollama()
    else:
        print(f"[llm_client] Unknown provider: '{PROVIDER}'. Use 'ollama' ")
        return False


def _check_ollama() -> bool:
    tags_url = f"{cfg['llm']['base_url']}/api/tags"
    try:
        r = requests.get(tags_url, timeout=5)
        r.raise_for_status()
        models    = [m["name"] for m in r.json().get("models", [])]
        available = any(MODEL.split(":")[0] in m for m in models)
        if not available:
            print(f"[llm_client] Model '{MODEL}' not found. Run: ollama pull {MODEL}")
            print(f"[llm_client] Available: {models}")
        return available
    except requests.exceptions.ConnectionError:
        print("[llm_client] Ollama not running. Start with: ollama serve")
        return False



# ── Core call ─────────────────────────────────────────────────────────────────

def call(prompt: str, system: str = "", expect_json: bool = False) -> str:
    """Send a prompt and return the raw text response."""
    if expect_json:
        system = (system + "\n\nRESPONSE FORMAT: Return ONLY valid JSON. "
                  "No explanation, no markdown fences, no preamble. "
                  "Start your response with { and end with }.").strip()

    if PROVIDER == "ollama":
        raw = _call_ollama(prompt, system)
    else:
        raise ValueError(f"Unknown provider: '{PROVIDER}'")

    return _clean(raw, expect_json)


def call_json(prompt: str, system: str = "") -> dict | list:
    """Call the LLM expecting a JSON response. Retries on parse failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        raw = call(prompt, system=system, expect_json=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[llm_client] JSON parse failed (attempt {attempt}): {e}")
            print(f"[llm_client] Response snippet: {raw[:200]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise ValueError("[llm_client] Could not parse valid JSON after retries.")


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, system: str) -> str:
    chat_url = f"{cfg['llm']['base_url']}/api/chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":    MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": TEMP, "num_predict": MAX_TOKENS},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(chat_url, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.exceptions.Timeout:
            print(f"[llm_client] Timeout (attempt {attempt}/{MAX_RETRIES})")
        except requests.exceptions.RequestException as e:
            print(f"[llm_client] Request error (attempt {attempt}): {e}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"[llm_client] Ollama failed after {MAX_RETRIES} attempts.")



# ── Response cleanup ──────────────────────────────────────────────────────────

def _clean(text: str, expect_json: bool) -> str:
    text = text.strip()
    if not expect_json:
        return text

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Trim any text before the first { or [
    brace   = text.find("{")
    bracket = text.find("[")
    start   = min(
        brace   if brace   != -1 else len(text),
        bracket if bracket != -1 else len(text),
    )
    if start > 0:
        text = text[start:]

    return text