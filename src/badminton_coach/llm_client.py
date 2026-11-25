"""Gemini client with safe key loading, retry on 429/503, and a local cache.

The free tier allows only a small number of requests per day, so:
  * advice for many rallies is generated in ONE request (batch prompt),
  * every response is cached in results/advice_cache.json,
  * the demo app reads the cache and never needs the API at presentation time.
"""
import hashlib
import json
import os
import random
import time
from pathlib import Path

from google import genai
from google.genai import errors

CACHE_PATH = Path("results/advice_cache.json")
RETRY_CODES = {429, 500, 503}


def load_api_key() -> str:
    """Colab Secrets first, then environment / .env. Never hard-code the key."""
    try:
        from google.colab import userdata  # only exists on Colab
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set (Colab Secrets or .env).")
    return key


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def generate(prompt: str, model: str = "gemini-2.5-flash", max_retries: int = 5) -> str:
    """Return cached text if this exact prompt was already answered."""
    cache = _load_cache()
    k = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    if k in cache:
        return cache[k]["text"]

    client = genai.Client(api_key=load_api_key())
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            cache[k] = {"model": model, "prompt": prompt, "text": resp.text}
            _save_cache(cache)
            return resp.text
        except errors.APIError as e:
            if e.code not in RETRY_CODES or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * 5 + random.uniform(0, 3)   # 5s, 10s, 20s, 40s ...
            print(f"Gemini {e.code}, retry in {wait:.0f}s")
            time.sleep(wait)
