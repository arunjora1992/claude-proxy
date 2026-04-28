"""
Exact-match response cache.

Hashes the canonicalized request (model + messages + system + tools + params)
and returns the cached response if present, saving 100% of tokens for identical
repeated calls.
"""

import hashlib
import json
import time
from typing import Any, Optional

from cachetools import TTLCache


class ResponseCache:
    def __init__(self, maxsize: int, ttl: int):
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.hits = 0
        self.misses = 0

    def _key(self, request: dict) -> str:
        # Canonicalize: sort keys, exclude non-deterministic fields
        stable = {
            "model": request.get("model"),
            "system": request.get("system"),
            "messages": request.get("messages"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "max_tokens": request.get("max_tokens"),
            "temperature": request.get("temperature"),
            "thinking": request.get("thinking"),
            "output_config": request.get("output_config"),
        }
        canonical = json.dumps(stable, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, request: dict) -> Optional[dict]:
        key = self._key(request)
        result = self._cache.get(key)
        if result is not None:
            self.hits += 1
            return result
        self.misses += 1
        return None

    def set(self, request: dict, response: dict) -> None:
        # Only cache complete, non-streaming responses with end_turn
        if response.get("stop_reason") not in ("end_turn", "max_tokens", "stop_sequence"):
            return
        key = self._key(request)
        self._cache[key] = response

    def stats(self) -> dict:
        total = self.hits + self.misses
        rate = self.hits / total if total else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(rate, 4),
            "size": len(self._cache),
            "maxsize": self._cache.maxsize,
        }

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0
