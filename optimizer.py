"""
Request optimizer — three strategies:

1. Prompt caching: auto-inject cache_control on the last system block and
   the last cacheable message block so repeated prefixes hit the cache
   (up to 90% savings on cached tokens, ~0.1x cost vs 1x).

2. Context trimming: when a conversation exceeds max_turns, drop the oldest
   turns while always keeping the first (user) message and all recent turns.

3. Model routing: map simple / short requests to Haiku (cheapest) and
   medium requests to Sonnet, reserving Opus only for complex work.
"""

import json
from typing import Any

from config import ProxyConfig


# ── Rough token estimator (4 chars ≈ 1 token) ─────────────────────────────

def _estimate_tokens(obj: Any) -> int:
    text = json.dumps(obj) if not isinstance(obj, str) else obj
    return max(1, len(text) // 4)


# ── 1. Prompt caching ─────────────────────────────────────────────────────

def inject_prompt_cache(request: dict, cfg: ProxyConfig) -> dict:
    """
    Inject cache_control on:
    - the last block of the system prompt (caches tools + system together)
    - the last content block of the most-recently-appended message turn

    Skips injection when the prefix is too short to cache (< min_tokens).
    Never overwrites an existing cache_control the caller already set.
    """
    if not cfg.prompt_cache_enabled:
        return request

    req = dict(request)

    # -- System prompt --
    system = req.get("system")
    if isinstance(system, str) and _estimate_tokens(system) >= cfg.prompt_cache_min_tokens:
        req["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(system, list) and system:
        last = dict(system[-1])
        if "cache_control" not in last and _estimate_tokens(last.get("text", "")) >= cfg.prompt_cache_min_tokens:
            last["cache_control"] = {"type": "ephemeral"}
        req["system"] = [*system[:-1], last]

    # -- Last message turn --
    messages = req.get("messages")
    if messages:
        msgs = list(messages)
        last_msg = dict(msgs[-1])
        content = last_msg.get("content")

        if isinstance(content, str) and _estimate_tokens(content) >= cfg.prompt_cache_min_tokens:
            last_msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content:
            blocks = list(content)
            last_block = dict(blocks[-1])
            if (
                "cache_control" not in last_block
                and last_block.get("type") in ("text", "document", "image", "tool_result")
                and _estimate_tokens(last_block) >= cfg.prompt_cache_min_tokens
            ):
                last_block["cache_control"] = {"type": "ephemeral"}
            last_msg["content"] = [*blocks[:-1], last_block]

        msgs[-1] = last_msg
        req["messages"] = msgs

    return req


# ── 2. Context trimming ───────────────────────────────────────────────────

def trim_context(request: dict, cfg: ProxyConfig) -> dict:
    """
    When a conversation has more than max_turns pairs, keep:
    - the first message (often carries the main task)
    - the most recent max_turns * 2 messages

    This prevents runaway token counts in long agentic loops.
    """
    if not cfg.context_trim_enabled:
        return request

    messages = request.get("messages", [])
    limit = cfg.context_max_turns * 2  # user + assistant per turn

    if len(messages) <= limit + 1:
        return request

    # Keep first message + tail
    trimmed = [messages[0]] + messages[-(limit):]
    return {**request, "messages": trimmed}


# ── 3. Model routing ──────────────────────────────────────────────────────

def route_model(request: dict, cfg: ProxyConfig) -> dict:
    """
    Estimate input token count and downgrade the model when appropriate.
    Only applies when the caller sends the default (Opus) model or omits model.
    Never upgrades a model the caller explicitly chose.
    """
    if not cfg.routing_enabled:
        return request

    caller_model = request.get("model", cfg.default_model)

    # Don't touch if caller explicitly chose haiku or sonnet
    if caller_model not in (cfg.default_model, "claude-opus-4-7", "claude-opus-4-6"):
        return request

    # Estimate total input size
    total = _estimate_tokens(request.get("system", ""))
    for msg in request.get("messages", []):
        total += _estimate_tokens(msg.get("content", ""))
    for tool in request.get("tools", []):
        total += _estimate_tokens(tool)

    if total <= cfg.haiku_token_threshold:
        routed = cfg.haiku_model
    elif total <= cfg.sonnet_token_threshold:
        routed = cfg.sonnet_model
    else:
        routed = caller_model  # keep Opus for large/complex requests

    return {**request, "model": routed, "_routed_from": caller_model}


# ── Combined pipeline ─────────────────────────────────────────────────────

def optimize_request(request: dict, cfg: ProxyConfig) -> dict:
    request = trim_context(request, cfg)
    request = inject_prompt_cache(request, cfg)
    request = route_model(request, cfg)
    return request
