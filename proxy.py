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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

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
    # Token accounting for dashboard
    "input_tokens_total": 0,
    "input_tokens_cache_read": 0,
    "input_tokens_cache_written": 0,
    "output_tokens_total": 0,
    # Cost tracking (USD)
    "cost_without_proxy_usd": 0.0,
    "cost_with_proxy_usd": 0.0,
    # Model tracking
    "last_model_used": "—",
    "models_seen": {},  # model -> request count
}

# Pricing per million tokens (input / output)
_PRICE = {
    "haiku":  {"in": 1.00,  "out": 5.00},
    "sonnet": {"in": 3.00,  "out": 15.00},
    "opus":   {"in": 5.00,  "out": 25.00},
}

def _model_tier(model: str) -> str:
    m = model.lower()
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m:
        return "sonnet"
    return "opus"

def _record_usage(usage: dict, model: str, original_model: str) -> None:
    inp   = usage.get("input_tokens", 0)
    out   = usage.get("output_tokens", 0)
    cr    = usage.get("cache_read_input_tokens", 0)
    cw    = usage.get("cache_creation_input_tokens", 0)

    _stats["input_tokens_total"]          += inp
    _stats["output_tokens_total"]         += out
    _stats["input_tokens_cache_read"]     += cr
    _stats["input_tokens_cache_written"]  += cw

    tier_actual   = _model_tier(model)
    tier_baseline = _model_tier(original_model)

    actual_cost = (inp * _PRICE[tier_actual]["in"] + out * _PRICE[tier_actual]["out"]) / 1_000_000
    # cache_read tokens cost 0.1x, cache_creation 1.25x
    actual_cost += (cr  * _PRICE[tier_actual]["in"] * 0.1)  / 1_000_000
    actual_cost += (cw  * _PRICE[tier_actual]["in"] * 0.25) / 1_000_000

    baseline_cost = ((inp + cr + cw) * _PRICE[tier_baseline]["in"] + out * _PRICE[tier_baseline]["out"]) / 1_000_000

    _stats["cost_with_proxy_usd"]    += actual_cost
    _stats["cost_without_proxy_usd"] += baseline_cost
    _stats["last_model_used"] = model
    _stats["models_seen"][model] = _stats["models_seen"].get(model, 0) + 1


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

    # Record token/cost metrics for dashboard
    _record_usage(
        result.get("usage", {}),
        model=result.get("model", forward.get("model", config.default_model)),
        original_model=optimized.get("_routed_from") or forward.get("model", config.default_model),
    )

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


# ── Dashboard ─────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claude Proxy Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 24px; }
  h1 { font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
  .subtitle { color: #64748b; font-size: 0.85rem; margin-bottom: 28px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
  .card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; color: #64748b; margin-bottom: 8px; }
  .card .value { font-size: 2rem; font-weight: 700; color: #f8fafc; line-height: 1; }
  .card .sub { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
  .card.green  .value { color: #4ade80; }
  .card.blue   .value { color: #60a5fa; }
  .card.yellow .value { color: #fbbf24; }
  .card.purple .value { color: #c084fc; }
  .card.cyan   .value { color: #22d3ee; font-size: 1.1rem; word-break: break-all; }
  .section { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin-bottom: 16px; }
  .section h2 { font-size: 0.9rem; font-weight: 600; color: #94a3b8; margin-bottom: 16px; text-transform: uppercase; letter-spacing: .05em; }
  .bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
  .bar-label { width: 160px; font-size: 0.82rem; color: #cbd5e1; flex-shrink: 0; }
  .bar-track { flex: 1; background: #0f172a; border-radius: 999px; height: 10px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }
  .bar-val { width: 80px; text-align: right; font-size: 0.82rem; color: #94a3b8; flex-shrink: 0; }
  .on  { color: #4ade80; font-weight: 600; }
  .off { color: #f87171; font-weight: 600; }
  .refresh { font-size: 0.75rem; color: #475569; margin-top: 20px; text-align: center; }
  /* Controls */
  .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .toggle-row { display: flex; align-items: center; justify-content: space-between;
                padding: 10px 0; border-bottom: 1px solid #0f172a; }
  .toggle-row .tlabel { font-size: 0.85rem; color: #cbd5e1; }
  .toggle-row .tdesc { font-size: 0.72rem; color: #475569; margin-top: 2px; }
  .switch { position: relative; width: 44px; height: 24px; flex-shrink: 0; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; inset: 0; background: #374151; border-radius: 999px;
            cursor: pointer; transition: background .2s; }
  .slider:before { content:''; position: absolute; height: 18px; width: 18px;
                   left: 3px; bottom: 3px; background: white; border-radius: 50%;
                   transition: transform .2s; }
  input:checked + .slider { background: #3b82f6; }
  input:checked + .slider:before { transform: translateX(20px); }
  select, input[type=text] {
    background: #0f172a; border: 1px solid #334155; color: #e2e8f0;
    border-radius: 6px; padding: 6px 10px; font-size: 0.82rem; width: 100%;
    margin-top: 6px;
  }
  .field-row { margin-bottom: 12px; }
  .field-row label { font-size: 0.78rem; color: #94a3b8; display: block; margin-bottom: 4px; }
  .btn { background: #3b82f6; color: white; border: none; border-radius: 8px;
         padding: 8px 20px; font-size: 0.85rem; cursor: pointer; margin-top: 8px; }
  .btn:hover { background: #2563eb; }
  .btn.danger { background: #ef4444; }
  .btn.danger:hover { background: #dc2626; }
  .toast { position: fixed; bottom: 24px; right: 24px; background: #22c55e;
           color: white; padding: 10px 20px; border-radius: 8px; font-size: 0.85rem;
           opacity: 0; transition: opacity .3s; pointer-events: none; }
  .toast.show { opacity: 1; }
  .model-tag { display: inline-block; background: #1e3a5f; color: #60a5fa;
               border: 1px solid #1d4ed8; border-radius: 6px; padding: 2px 8px;
               font-size: 0.75rem; margin: 2px; }
</style>
</head>
<body>
<h1>Claude Proxy Dashboard</h1>
<p class="subtitle">Live token savings · auto-refreshes every 10 s</p>

<!-- Top KPI cards -->
<div class="grid">
  <div class="card green">
    <div class="label">Cost saved</div>
    <div class="value" id="saved_pct">—</div>
    <div class="sub" id="saved_usd">loading…</div>
  </div>
  <div class="card blue">
    <div class="label">Cache hit rate</div>
    <div class="value" id="hit_rate">—</div>
    <div class="sub" id="hits_of">loading…</div>
  </div>
  <div class="card yellow">
    <div class="label">Requests total</div>
    <div class="value" id="req_total">—</div>
    <div class="sub" id="upstream_calls">— upstream calls</div>
  </div>
  <div class="card purple">
    <div class="label">Prompt cache saves</div>
    <div class="value" id="cache_read_tokens">—</div>
    <div class="sub">tokens from prompt cache</div>
  </div>
</div>

<!-- Current model -->
<div class="card cyan" style="margin-bottom:16px">
  <div class="label">Active model (last request)</div>
  <div class="value" id="last_model">—</div>
  <div class="sub" id="models_seen_row" style="margin-top:8px"></div>
</div>

<!-- Token breakdown -->
<div class="section">
  <h2>Token breakdown</h2>
  <div class="bar-row">
    <div class="bar-label">Without proxy (est.)</div>
    <div class="bar-track"><div class="bar-fill" id="b_without" style="width:100%;background:#f87171;"></div></div>
    <div class="bar-val" id="v_without">—</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">Actual input tokens</div>
    <div class="bar-track"><div class="bar-fill" id="b_actual" style="width:0%;background:#60a5fa;"></div></div>
    <div class="bar-val" id="v_actual">—</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">Prompt cache reads</div>
    <div class="bar-track"><div class="bar-fill" id="b_cr" style="width:0%;background:#4ade80;"></div></div>
    <div class="bar-val" id="v_cr">—</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">Output tokens</div>
    <div class="bar-track"><div class="bar-fill" id="b_out" style="width:0%;background:#c084fc;"></div></div>
    <div class="bar-val" id="v_out">—</div>
  </div>
</div>

<!-- Optimisation events -->
<div class="section">
  <h2>Optimisation events</h2>
  <div class="bar-row">
    <div class="bar-label">Response cache hits</div>
    <div class="bar-track"><div class="bar-fill" id="b_chits" style="background:#4ade80;width:0%"></div></div>
    <div class="bar-val" id="v_chits">—</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">Model downgrades</div>
    <div class="bar-track"><div class="bar-fill" id="b_route" style="background:#fbbf24;width:0%"></div></div>
    <div class="bar-val" id="v_route">—</div>
  </div>
  <div class="bar-row">
    <div class="bar-label">Context trims</div>
    <div class="bar-track"><div class="bar-fill" id="b_trim" style="background:#fb923c;width:0%"></div></div>
    <div class="bar-val" id="v_trim">—</div>
  </div>
</div>

<!-- Controls -->
<div class="section">
  <h2>Runtime controls</h2>
  <div class="controls">
    <div>
      <div class="toggle-row">
        <div><div class="tlabel">Response cache</div><div class="tdesc">100% savings on repeated requests</div></div>
        <label class="switch"><input type="checkbox" id="t_rcache" onchange="toggle('response_cache_enabled',this.checked)"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <div><div class="tlabel">Prompt caching</div><div class="tdesc">Auto cache_control — up to 90% savings</div></div>
        <label class="switch"><input type="checkbox" id="t_pcache" onchange="toggle('prompt_cache_enabled',this.checked)"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <div><div class="tlabel">Model routing</div><div class="tdesc">⚠ Disable for Claude Code</div></div>
        <label class="switch"><input type="checkbox" id="t_route" onchange="toggle('routing_enabled',this.checked)"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <div><div class="tlabel">Context trimming</div><div class="tdesc">⚠ Disable for Claude Code</div></div>
        <label class="switch"><input type="checkbox" id="t_trim" onchange="toggle('context_trim_enabled',this.checked)"><span class="slider"></span></label>
      </div>
    </div>
    <div>
      <div class="field-row">
        <label>Haiku model</label>
        <select id="sel_haiku" onchange="setField('haiku_model',this.value)">
          <option value="claude-haiku-4-5">claude-haiku-4-5</option>
        </select>
      </div>
      <div class="field-row">
        <label>Sonnet model</label>
        <select id="sel_sonnet" onchange="setField('sonnet_model',this.value)">
          <option value="claude-sonnet-4-6">claude-sonnet-4-6</option>
        </select>
      </div>
      <div class="field-row">
        <label>Default (Opus) model</label>
        <select id="sel_opus" onchange="setField('default_model',this.value)">
          <option value="claude-opus-4-7">claude-opus-4-7</option>
          <option value="claude-opus-4-6">claude-opus-4-6</option>
        </select>
      </div>
      <div class="field-row">
        <label>Haiku threshold (tokens)</label>
        <input type="text" id="inp_haiku_thresh" placeholder="2000"
               onblur="setField('haiku_token_threshold',parseInt(this.value)||2000)">
      </div>
      <div class="field-row">
        <label>Sonnet threshold (tokens)</label>
        <input type="text" id="inp_sonnet_thresh" placeholder="8000"
               onblur="setField('sonnet_token_threshold',parseInt(this.value)||8000)">
      </div>
      <button class="btn danger" onclick="clearCache()">Clear response cache</button>
    </div>
  </div>
</div>

<div class="refresh">Auto-refreshing every 10 s &nbsp;·&nbsp; <a href="/proxy/stats" style="color:#60a5fa">raw JSON</a></div>
<div class="toast" id="toast"></div>

<script>
function fmt(n) {
  if (n >= 1e6) return (n/1e6).toFixed(2)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return String(n||0);
}
function bar(id, val, max) {
  const el = document.getElementById(id);
  if (el) el.style.width = (max > 0 ? Math.min(100, val/max*100) : 0) + '%';
}
function showToast(msg, ok=true) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = ok ? '#22c55e' : '#ef4444';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

async function toggle(key, val) {
  try {
    await fetch('/proxy/config', {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({[key]:val})});
    showToast(key + ' → ' + (val ? 'ON' : 'OFF'));
  } catch(e) { showToast('Error: '+e, false); }
}
async function setField(key, val) {
  try {
    await fetch('/proxy/config', {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({[key]:val})});
    showToast(key + ' updated');
  } catch(e) { showToast('Error: '+e, false); }
}
async function clearCache() {
  await fetch('/proxy/cache/clear', {method:'POST'});
  showToast('Cache cleared');
}

let _cfg = {};

async function refresh() {
  try {
    const r = await fetch('/proxy/stats');
    const d = await r.json();
    const p = d.proxy, c = d.response_cache, cfg = d.config;
    _cfg = cfg;

    // Cost
    const saved = (p.cost_without_proxy_usd||0) - (p.cost_with_proxy_usd||0);
    const savedPct = p.cost_without_proxy_usd > 0
      ? (saved / p.cost_without_proxy_usd * 100).toFixed(1) + '%' : '0%';
    document.getElementById('saved_pct').textContent = savedPct;
    document.getElementById('saved_usd').textContent =
      '$' + saved.toFixed(4) + ' saved of $' + (p.cost_without_proxy_usd||0).toFixed(4);

    // Cache
    const hr = c.hit_rate != null ? (c.hit_rate*100).toFixed(1)+'%' : '0%';
    document.getElementById('hit_rate').textContent = hr;
    document.getElementById('hits_of').textContent = (c.hits||0) + ' hits of ' + ((c.hits||0)+(c.misses||0)) + ' requests';

    // Totals
    document.getElementById('req_total').textContent = fmt(p.requests_total);
    document.getElementById('upstream_calls').textContent = fmt(p.upstream_calls) + ' upstream calls';
    document.getElementById('cache_read_tokens').textContent = fmt(p.input_tokens_cache_read || 0);

    // Model
    document.getElementById('last_model').textContent = p.last_model_used || '—';
    const seen = p.models_seen || {};
    document.getElementById('models_seen_row').innerHTML =
      Object.entries(seen).map(([m,n]) => `<span class="model-tag">${m}: ${n}</span>`).join('') || 'No requests yet';

    // Token bars
    const baseline = (p.input_tokens_total||0) + (p.input_tokens_cache_read||0) + (p.input_tokens_cache_written||0);
    const maxTok = Math.max(baseline, p.output_tokens_total||0, 1);
    bar('b_without', baseline, maxTok); document.getElementById('v_without').textContent = fmt(baseline);
    bar('b_actual', p.input_tokens_total||0, maxTok); document.getElementById('v_actual').textContent = fmt(p.input_tokens_total||0);
    bar('b_cr', p.input_tokens_cache_read||0, maxTok); document.getElementById('v_cr').textContent = fmt(p.input_tokens_cache_read||0);
    bar('b_out', p.output_tokens_total||0, maxTok); document.getElementById('v_out').textContent = fmt(p.output_tokens_total||0);

    // Event bars
    const maxEv = Math.max(p.cache_hits||0, p.model_downgrade_count||0, p.context_trims||0, 1);
    bar('b_chits', p.cache_hits||0, maxEv); document.getElementById('v_chits').textContent = fmt(p.cache_hits||0);
    bar('b_route', p.model_downgrade_count||0, maxEv); document.getElementById('v_route').textContent = fmt(p.model_downgrade_count||0);
    bar('b_trim', p.context_trims||0, maxEv); document.getElementById('v_trim').textContent = fmt(p.context_trims||0);

    // Sync toggles (only if user is not hovering)
    document.getElementById('t_rcache').checked = cfg.response_cache_enabled;
    document.getElementById('t_pcache').checked = cfg.prompt_cache_enabled;
    document.getElementById('t_route').checked  = cfg.routing_enabled;
    document.getElementById('t_trim').checked   = cfg.context_trim_enabled;

    // Sync selects
    setSelVal('sel_haiku',  cfg.haiku_model);
    setSelVal('sel_sonnet', cfg.sonnet_model);
    setSelVal('sel_opus',   cfg.default_model);
    document.getElementById('inp_haiku_thresh').placeholder  = cfg.haiku_threshold_tokens;
    document.getElementById('inp_sonnet_thresh').placeholder = cfg.sonnet_threshold_tokens;
  } catch(e) { console.error(e); }
}

function setSelVal(id, val) {
  const s = document.getElementById(id);
  let found = false;
  for (const o of s.options) { if (o.value === val) { o.selected = true; found = true; } }
  if (!found) { const o = new Option(val, val, true, true); s.add(o); }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


@app.get("/proxy/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)


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
            "default_model": config.default_model,
            "haiku_model": config.haiku_model,
            "sonnet_model": config.sonnet_model,
        },
    })


@app.patch("/proxy/config")
async def update_config(request: Request) -> JSONResponse:
    """Runtime config toggle — changes take effect immediately without restart."""
    body = await request.json()
    allowed = {
        "routing_enabled": bool,
        "context_trim_enabled": bool,
        "prompt_cache_enabled": bool,
        "response_cache_enabled": bool,
        "haiku_token_threshold": int,
        "sonnet_token_threshold": int,
        "context_max_turns": int,
        "default_model": str,
        "haiku_model": str,
        "sonnet_model": str,
    }
    updated = {}
    for key, typ in allowed.items():
        if key in body:
            setattr(config, key, typ(body[key]))
            updated[key] = getattr(config, key)
    return JSONResponse({"updated": updated})


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
