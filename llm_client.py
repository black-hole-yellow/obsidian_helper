"""
llm_client.py — Thin wrapper around the Ollama API.

Responsibilities:
  - Send prompts to the local model
  - Enforce JSON-only responses for structured extraction
  - Handle retries on malformed JSON
  - Keep a single reusable session for performance
"""

import json
import re
import time
from typing import Any

import requests

from config import cfg


# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL   = cfg["llm"]["base_url"]
MODEL      = cfg["llm"]["model"]
TEMP       = cfg["llm"]["temperature"]
MAX_TOKENS = cfg["llm"]["max_tokens"]

_GENERATE_URL = f"{BASE_URL}/api/generate"
_CHAT_URL     = f"{BASE_URL}/api/chat"
_TAGS_URL     = f"{BASE_URL}/api/tags"

MAX_RETRIES   = 3
RETRY_DELAY   = 2  # seconds between retries


# ── Health check ─────────────────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Verify Ollama is reachable and the configured model is available."""
    try:
        r = requests.get(_TAGS_URL, timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]

        # Accept partial match (e.g. "qwen2.5:14b" matches "qwen2.5:14b-instruct-q4")
        available = any(MODEL.split(":")[0] in m for m in models)
        if not available:
            print(f"[llm_client] WARNING: Model '{MODEL}' not found in Ollama.")
            print(f"[llm_client] Available models: {models}")
            print(f"[llm_client] Run: ollama pull {MODEL}")
        return available
    except requests.exceptions.ConnectionError:
        print("[llm_client] ERROR: Ollama is not running. Start it with: ollama serve")
        return False


# ── Core call ────────────────────────────────────────────────────────────────

def call(prompt: str, system: str = "", expect_json: bool = False) -> str:
    """
    Send a prompt to Ollama and return the raw text response.

    Args:
        prompt      : The user message / instruction
        system      : Optional system message (sets LLM behavior)
        expect_json : If True, adds JSON enforcement to system prompt
                      and strips markdown fences from response

    Returns:
        Raw string response from the model
    """
    if expect_json:
        system = (system + "\n\nRESPONSE FORMAT: Return ONLY valid JSON. "
                  "No explanation, no markdown fences, no preamble. "
                  "Start your response with { and end with }.").strip()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":   MODEL,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature":  TEMP,
            "num_predict":  MAX_TOKENS,
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(_CHAT_URL, json=payload, timeout=120)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            return _clean_response(content, expect_json)

        except requests.exceptions.Timeout:
            print(f"[llm_client] Timeout on attempt {attempt}/{MAX_RETRIES}")
        except requests.exceptions.RequestException as e:
            print(f"[llm_client] Request error on attempt {attempt}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"[llm_client] Failed after {MAX_RETRIES} attempts.")


def call_json(prompt: str, system: str = "") -> dict | list:
    """
    Call the LLM expecting a JSON response.
    Automatically retries if the response is not valid JSON.

    Returns:
        Parsed Python dict or list
    """
    for attempt in range(1, MAX_RETRIES + 1):
        raw = call(prompt, system=system, expect_json=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[llm_client] JSON parse failed (attempt {attempt}): {e}")
            print(f"[llm_client] Raw response snippet: {raw[:300]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise ValueError("[llm_client] Could not parse valid JSON after retries.")


# ── Response cleanup ─────────────────────────────────────────────────────────

def _clean_response(text: str, expect_json: bool) -> str:
    """Strip markdown code fences and leading/trailing whitespace."""
    text = text.strip()
    if expect_json:
        # Remove ```json ... ``` or ``` ... ``` wrappers if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # If there's text before the first {, trim it
        brace = text.find("{")
        bracket = text.find("[")
        start = min(
            brace   if brace   != -1 else len(text),
            bracket if bracket != -1 else len(text)
        )
        if start > 0:
            text = text[start:]
    return text
    return text


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Checking Ollama connection...")
    if check_ollama_running():
        print("Connection OK. Sending test prompt...")
        result = call_json(
            prompt='Return a JSON object with keys "status" and "message".',
            system="You are a helpful assistant."
        )
        print("Response:", result)
