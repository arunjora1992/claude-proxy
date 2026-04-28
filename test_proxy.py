"""Smoke tests for the proxy optimizer and cache (no API key required)."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cache import ResponseCache
from optimizer import inject_prompt_cache, trim_context, route_model, optimize_request
from config import ProxyConfig


def make_cfg(**kwargs) -> ProxyConfig:
    cfg = ProxyConfig()
    cfg.anthropic_api_key = "test"
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ── Cache tests ────────────────────────────────────────────────────────────

def test_cache_hit():
    cache = ResponseCache(maxsize=100, ttl=60)
    req = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    resp = {"stop_reason": "end_turn", "usage": {"input_tokens": 50}, "id": "msg_1"}
    cache.set(req, resp)
    assert cache.get(req) == resp
    print("  [PASS] cache hit returns cached response")


def test_cache_miss_on_different_request():
    cache = ResponseCache(maxsize=100, ttl=60)
    req1 = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    req2 = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "bye"}], "max_tokens": 100}
    resp = {"stop_reason": "end_turn", "usage": {"input_tokens": 50}}
    cache.set(req1, resp)
    assert cache.get(req2) is None
    print("  [PASS] cache miss on different content")


def test_cache_no_store_tool_use():
    cache = ResponseCache(maxsize=100, ttl=60)
    req = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "x"}], "max_tokens": 100}
    resp = {"stop_reason": "tool_use", "usage": {"input_tokens": 50}}
    cache.set(req, resp)
    # tool_use responses should not be cached
    assert cache.get(req) is None
    print("  [PASS] tool_use responses not cached")


# ── Prompt caching tests ───────────────────────────────────────────────────

def test_prompt_cache_injects_string_system():
    cfg = make_cfg(prompt_cache_min_tokens=1)
    req = {"system": "You are helpful.", "messages": [{"role": "user", "content": "hi"}]}
    out = inject_prompt_cache(req, cfg)
    assert isinstance(out["system"], list)
    assert out["system"][0]["cache_control"] == {"type": "ephemeral"}
    print("  [PASS] string system prompt gets cache_control block")


def test_prompt_cache_skips_short_system():
    cfg = make_cfg(prompt_cache_min_tokens=9999)
    req = {"system": "hi", "messages": [{"role": "user", "content": "x"}]}
    out = inject_prompt_cache(req, cfg)
    # Should stay as string (too short to cache)
    assert isinstance(out["system"], str)
    print("  [PASS] short system prompt not wrapped")


def test_prompt_cache_injects_message_block():
    cfg = make_cfg(prompt_cache_min_tokens=1)
    req = {
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }
    out = inject_prompt_cache(req, cfg)
    last_msg = out["messages"][-1]
    assert isinstance(last_msg["content"], list)
    assert last_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}
    print("  [PASS] last message content gets cache_control block")


# ── Context trimming tests ─────────────────────────────────────────────────

def test_trim_context_short():
    cfg = make_cfg(context_max_turns=10)
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
    req = {"messages": msgs}
    out = trim_context(req, cfg)
    assert len(out["messages"]) == 5  # under limit, unchanged
    print("  [PASS] short conversation not trimmed")


def test_trim_context_long():
    cfg = make_cfg(context_max_turns=2)
    # 3 pairs = 6 msgs, limit=2 pairs=4 msgs -> keeps first + last 4
    msgs = []
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"msg {i}"})
    req = {"messages": msgs}
    out = trim_context(req, cfg)
    kept = out["messages"]
    assert kept[0] == msgs[0], "First message preserved"
    assert kept[-1] == msgs[-1], "Last message preserved"
    assert len(kept) < len(msgs), f"Got {len(kept)}, expected < {len(msgs)}"
    print(f"  [PASS] long conversation trimmed {len(msgs)} -> {len(kept)} messages")


# ── Model routing tests ────────────────────────────────────────────────────

def test_route_to_haiku_short():
    cfg = make_cfg(haiku_token_threshold=500, sonnet_token_threshold=2000,
                   routing_enabled=True, default_model="claude-opus-4-7")
    req = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "Hi"}]}
    out = route_model(req, cfg)
    assert out["model"] == cfg.haiku_model
    print(f"  [PASS] short request routed to {cfg.haiku_model}")


def test_route_to_sonnet_medium():
    cfg = make_cfg(haiku_token_threshold=5, sonnet_token_threshold=5000,
                   routing_enabled=True, default_model="claude-opus-4-7")
    # ~100 token message
    content = "word " * 400
    req = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": content}]}
    out = route_model(req, cfg)
    assert out["model"] == cfg.sonnet_model
    print(f"  [PASS] medium request routed to {cfg.sonnet_model}")


def test_no_route_explicit_haiku():
    cfg = make_cfg(routing_enabled=True, default_model="claude-opus-4-7")
    req = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "Hi"}]}
    out = route_model(req, cfg)
    assert out["model"] == "claude-haiku-4-5"
    print("  [PASS] explicit non-Opus model not re-routed")


def test_routing_disabled():
    cfg = make_cfg(routing_enabled=False, default_model="claude-opus-4-7")
    req = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "Hi"}]}
    out = route_model(req, cfg)
    assert out["model"] == "claude-opus-4-7"
    print("  [PASS] routing disabled preserves original model")


# ── Full pipeline ──────────────────────────────────────────────────────────

def test_full_pipeline():
    cfg = make_cfg(
        prompt_cache_min_tokens=1,
        haiku_token_threshold=500,
        sonnet_token_threshold=5000,
        context_max_turns=3,
        routing_enabled=True,
    )
    msgs = []
    for i in range(10):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})
    req = {
        "model": "claude-opus-4-7",
        "system": "You are a helpful assistant.",
        "messages": msgs,
        "max_tokens": 1024,
    }
    out = optimize_request(req, cfg)
    # Context should be trimmed
    assert len(out["messages"]) < len(msgs)
    # System should have cache_control
    assert isinstance(out["system"], list)
    # Model should be routed (trimmed msgs are short)
    assert out["model"] in (cfg.haiku_model, cfg.sonnet_model)
    print(f"  [PASS] full pipeline: {len(msgs)} msgs -> {len(out['messages'])}, "
          f"model -> {out['model']}")


if __name__ == "__main__":
    tests = [
        ("Cache", [test_cache_hit, test_cache_miss_on_different_request, test_cache_no_store_tool_use]),
        ("Prompt caching", [test_prompt_cache_injects_string_system, test_prompt_cache_skips_short_system,
                            test_prompt_cache_injects_message_block]),
        ("Context trimming", [test_trim_context_short, test_trim_context_long]),
        ("Model routing", [test_route_to_haiku_short, test_route_to_sonnet_medium,
                           test_no_route_explicit_haiku, test_routing_disabled]),
        ("Full pipeline", [test_full_pipeline]),
    ]

    passed = failed = 0
    for group, fns in tests:
        print(f"\n{group}:")
        for fn in fns:
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {fn.__name__}: {e}")
                failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
