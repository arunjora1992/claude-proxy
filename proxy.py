"""
Claude Token-Reduction Proxy
=============================
A FastAPI server that sits in front of the Anthropic API and applies four
token-saving strategies transparently:

  1. Exact response cache   — identical requests never hit the API (100% savings)
  2. Prompt caching         — auto cache_control on stable prefixes (up to 90% savings)
  3. Model routing          — short requests go to Haiku/Sonnet instead of Opus (5x cheaper)
  4. Context trimming       — truncate long conversation histories to stay lean

Optional:
  5. Batch API queue        — 50% discount for non-real-time traffic (BATCH_ENABLED=true)

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python proxy.py

  Then point your clients at http://localhost:8080/v1/messages
  instead of https://api.anthropic.com/v1/messages.

Environment variables (all optional):
  PROXY_PORT              Port to listen on (default: 8080)
  CACHE_ENABLED           Enable exact response cache (default: true)
  CACHE_MAXSIZE           Max cached responses (default: 1000)
  CACHE_TTL               Cache TTL in seconds (default: 3600)
  PROMPT_CACHE_ENABLED    Auto-inject cache_control (default: true)
  PROMPT_CACHE_MIN_TOKENS Min tokens to bother caching a prefix (default: 1024)
  ROUTING_ENABLED         Enable model routing (default: true)
  HAIKU_THRESHOLD         Token count <= this -> Haiku (default: 2000)
  SONNET_THRESHOLD        Token count <= this -> Sonnet (default: 8000)
  DEFAULT_MODEL           Model to treat as "expensive" for routing (default: claude-opus-4-7)
  CONTEXT_TRIM_ENABLED    Trim long conversations (default: true)
  CONTEXT_MAX_TURNS       Max user/assistant turn pairs to keep (default: 20)
  BATCH_ENABLED           Enable batch queue endpoint (default: false)
"""

import asyncio
import json
import time
from typing import Any, Optional

import anthropic
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from cache import ResponseCache
from config import ProxyConfig, config
from optimizer import optimize_request

# ── Batch queue (optional) ────────────────────────────────────────────────
if config.batch_enabled:
    from batch_queue import BatchQueue


app = FastAPI(title="Claude Token-Reduction Proxy", version="1.0.0")

_cache = ResponseCache(maxsize=config.response_cache_maxsize, ttl=config.response_cache_ttl_seconds)
_batch_queue: Optional[Any] = BatchQueue(anthropic.Anthropic(api_key=config.anthropic_api_key)) if config.batch_enabled else None


def _get_api_key(request: Request) -> str:
    """Extract API key from the incoming request (x-api-key or Authorization header)."""
    key = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
    if not key and config.anthropic_api_key:
        key = config.anthropic_api_key
    if not key:
        raise HTTPException(status_code=401, detail="No API key provided")
    return key

# Track savings
_stats = {
    "requests_total": 0,
    "cache_hits": 0,
    "tokens_saved_estimate": 0,
    "model_downgrade_count": 0,
    "context_trims": 0,
    "upstream_calls": 0,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _response_to_dict(msg: anthropic.types.Message) -> dict:
    return {
        "id": msg.id,
        "type": "message",
        "role": "assistant",
        "content": [b.model_dump() for b in msg.content],
        "model": msg.model,
        "stop_reason": msg.stop_reason,
        "stop_sequence": msg.stop_sequence,
        "usage": {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        },
    }


def _strip_internal_fields(req: dict) -> dict:
    """Remove proxy-internal tracking fields before forwarding."""
    return {k: v for k, v in req.items() if not k.startswith("_")}


# ── Main messages endpoint ─────────────────────────────────────────────────

@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body: dict = await request.json()
    _stats["requests_total"] += 1

    is_streaming = body.get("stream", False)
    original_messages_len = len(body.get("messages", []))

    # 1. Exact response cache (only for non-streaming)
    if not is_streaming and config.response_cache_enabled:
        cached = _cache.get(body)
        if cached is not None:
            _stats["cache_hits"] += 1
            _stats["tokens_saved_estimate"] += cached.get("usage", {}).get("input_tokens", 0)
            cached["_proxy_cache_hit"] = True
            return JSONResponse(cached)

    # 2 & 3 & 4. Optimize (trim + prompt cache + route)
    optimized = optimize_request(body, config)

    if len(optimized.get("messages", [])) < original_messages_len:
        _stats["context_trims"] += 1

    if optimized.get("_routed_from"):
        _stats["model_downgrade_count"] += 1

    forward = _strip_internal_fields(optimized)

    api_key = _get_api_key(request)

    # 5. Streaming path — forward raw bytes so Claude Code sees exact SSE wire format
    if is_streaming:
        # Pass through beta headers and other anthropic-specific headers from the caller
        passthrough = {
            k: v for k, v in request.headers.items()
            if k.lower() in ("anthropic-beta", "anthropic-version", "user-agent")
        }
        return StreamingResponse(
            _stream_upstream(forward, api_key, passthrough),
            media_type="text/event-stream",
        )

    # 6. Non-streaming upstream call
    _stats["upstream_calls"] += 1
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(**forward)
    except anthropic.APIError as e:
        raise HTTPException(status_code=e.status_code if hasattr(e, "status_code") else 500, detail=str(e))

    result = _response_to_dict(msg)

    # Annotate routing info for transparency
    if optimized.get("_routed_from"):
        result["_proxy_routed_model"] = result["model"]
        result["_proxy_original_model"] = optimized["_routed_from"]

    # Store in response cache
    if config.response_cache_enabled:
        _cache.set(body, result)

    return JSONResponse(result)


async def _stream_upstream(forward: dict, api_key: str, extra_headers: dict):
    """
    Forward a streaming request to Anthropic and yield raw SSE bytes.

    Using raw httpx byte forwarding (not the SDK stream wrapper) guarantees
    that Claude Code sees the exact same SSE wire format as the real API.
    Any re-serialization through SDK objects would break client parsers.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        **extra_headers,
    }
    payload = {**forward, "stream": True}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as http:
            async with http.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except httpx.HTTPError as e:
        error_data = {"type": "error", "error": {"type": "connection_error", "message": str(e)}}
        yield f"data: {json.dumps(error_data)}\n\n"


# ── Token counting endpoint ────────────────────────────────────────────────

@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> JSONResponse:
    body: dict = await request.json()
    optimized = _strip_internal_fields(optimize_request(body, config))
    try:
        client = anthropic.Anthropic(api_key=_get_api_key(request))
        result = client.messages.count_tokens(**optimized)
        return JSONResponse({"input_tokens": result.input_tokens})
    except anthropic.APIError as e:
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=str(e))


# ── Batch endpoints (optional) ─────────────────────────────────────────────

if config.batch_enabled:

    @app.post("/v1/messages/batch")
    async def batch_enqueue(request: Request) -> JSONResponse:
        body = await request.json()
        result = await _batch_queue.enqueue(body)
        return JSONResponse(result, status_code=202)

    @app.get("/v1/messages/batch/{batch_id}")
    async def batch_status(batch_id: str) -> JSONResponse:
        status = _batch_queue.get_batch_status(batch_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Batch not found")
        return JSONResponse(status)

    @app.get("/v1/messages/batch/{batch_id}/results")
    async def batch_results(batch_id: str) -> JSONResponse:
        status = _batch_queue.get_batch_status(batch_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Batch not found")
        if status["status"] != "ended":
            raise HTTPException(status_code=202, detail="Batch not yet complete")
        return JSONResponse(status["results"])

    @app.post("/v1/messages/batch/flush")
    async def batch_flush() -> JSONResponse:
        batch_id = await _batch_queue.flush_all()
        return JSONResponse({"batch_id": batch_id, "status": "flushed"})


# ── Models passthrough (required for Claude Code capability discovery) ────

@app.get("/v1/models")
async def list_models(request: Request) -> Response:
    passthrough = {
        k: v for k, v in request.headers.items()
        if k.lower() in ("anthropic-beta", "anthropic-version", "user-agent")
    }
    headers = {"x-api-key": _get_api_key(request), "anthropic-version": "2023-06-01", **passthrough}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as http:
        resp = await http.get("https://api.anthropic.com/v1/models", headers=headers)
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, request: Request) -> Response:
    passthrough = {
        k: v for k, v in request.headers.items()
        if k.lower() in ("anthropic-beta", "anthropic-version", "user-agent")
    }
    headers = {"x-api-key": _get_api_key(request), "anthropic-version": "2023-06-01", **passthrough}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as http:
        resp = await http.get(f"https://api.anthropic.com/v1/models/{model_id}", headers=headers)
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


# ── Admin / observability endpoints ───────────────────────────────────────

@app.get("/proxy/stats")
async def stats() -> JSONResponse:
    cache_stats = _cache.stats()
    return JSONResponse({
        "proxy": _stats,
        "response_cache": cache_stats,
        "config": {
            "response_cache_enabled": config.response_cache_enabled,
            "prompt_cache_enabled": config.prompt_cache_enabled,
            "routing_enabled": config.routing_enabled,
            "context_trim_enabled": config.context_trim_enabled,
            "batch_enabled": config.batch_enabled,
            "haiku_threshold_tokens": config.haiku_token_threshold,
            "sonnet_threshold_tokens": config.sonnet_token_threshold,
            "context_max_turns": config.context_max_turns,
        },
    })


@app.post("/proxy/cache/clear")
async def cache_clear() -> JSONResponse:
    _cache.clear()
    return JSONResponse({"status": "cleared"})


@app.get("/proxy/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Claude Token-Reduction Proxy starting on port {config.proxy_port}")
    print(f"  Response cache : {'on' if config.response_cache_enabled else 'off'} "
          f"(max {config.response_cache_maxsize}, TTL {config.response_cache_ttl_seconds}s)")
    print(f"  Prompt caching : {'on' if config.prompt_cache_enabled else 'off'}")
    print(f"  Model routing  : {'on' if config.routing_enabled else 'off'} "
          f"(haiku ≤{config.haiku_token_threshold}t, sonnet ≤{config.sonnet_token_threshold}t)")
    print(f"  Context trim   : {'on' if config.context_trim_enabled else 'off'} "
          f"(max {config.context_max_turns} turns)")
    print(f"  Batch queue    : {'on' if config.batch_enabled else 'off'}")
    uvicorn.run(app, host="0.0.0.0", port=config.proxy_port)
