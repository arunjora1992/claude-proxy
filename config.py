import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProxyConfig:
    # API
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    proxy_port: int = int(os.environ.get("PROXY_PORT", "8080"))

    # Response cache
    response_cache_enabled: bool = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
    response_cache_maxsize: int = int(os.environ.get("CACHE_MAXSIZE", "1000"))
    response_cache_ttl_seconds: int = int(os.environ.get("CACHE_TTL", "3600"))

    # Prompt caching (cache_control injection)
    prompt_cache_enabled: bool = os.environ.get("PROMPT_CACHE_ENABLED", "true").lower() == "true"
    prompt_cache_min_tokens: int = int(os.environ.get("PROMPT_CACHE_MIN_TOKENS", "1024"))

    # Model routing
    routing_enabled: bool = os.environ.get("ROUTING_ENABLED", "true").lower() == "true"
    # Requests with estimated input tokens <= this go to haiku
    haiku_token_threshold: int = int(os.environ.get("HAIKU_THRESHOLD", "2000"))
    # Requests with estimated input tokens <= this go to sonnet
    sonnet_token_threshold: int = int(os.environ.get("SONNET_THRESHOLD", "8000"))
    default_model: str = os.environ.get("DEFAULT_MODEL", "claude-opus-4-7")
    haiku_model: str = os.environ.get("HAIKU_MODEL", "claude-haiku-4-5")
    sonnet_model: str = os.environ.get("SONNET_MODEL", "claude-sonnet-4-6")

    # Context trimming
    context_trim_enabled: bool = os.environ.get("CONTEXT_TRIM_ENABLED", "true").lower() == "true"
    # Keep at most this many message turns (user+assistant pairs)
    context_max_turns: int = int(os.environ.get("CONTEXT_MAX_TURNS", "20"))

    # Batch queue
    batch_enabled: bool = os.environ.get("BATCH_ENABLED", "false").lower() == "true"

    # Claude Code compatibility mode:
    # disables routing + context trimming (both can break Claude Code),
    # keeps prompt caching and response caching (both are transparent and safe).
    claude_code_mode: bool = os.environ.get("CLAUDE_CODE_MODE", "false").lower() == "true"

    def __post_init__(self) -> None:
        if self.claude_code_mode:
            self.routing_enabled = False
            self.context_trim_enabled = False


config = ProxyConfig()
