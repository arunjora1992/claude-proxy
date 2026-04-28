# Claude Token-Reduction Proxy

A lightweight FastAPI proxy that sits between your client (e.g. Claude Code) and the Anthropic API, transparently applying token-saving strategies with zero code changes on the client side.

## Savings strategies

| Strategy | How | Savings |
|---|---|---|
| **Exact response cache** | SHA-256 key → TTLCache; identical requests never hit the API | 100% on cache hits |
| **Prompt caching** | Auto-injects `cache_control` on stable system prompt and message prefixes | Up to 90% on cached tokens |
| **Model routing** | Routes short requests to Haiku/Sonnet instead of Opus | 5–15× cheaper per request |
| **Context trimming** | Truncates long histories, keeps first + recent turns | Reduces runaway token counts |
| **Batch API queue** | Collects requests and flushes via Batch API (optional) | 50% discount |

> **Claude Code users:** `CLAUDE_CODE_MODE=true` (the default in docker-compose) disables routing and context trimming — both can degrade code quality. Prompt caching and response caching remain active and are fully transparent. The proxy forwards your API key from Claude Code requests — no proxy-side key config needed.

## Quick start (Docker Compose)

```bash
git clone git@github.com:arunjora1992/claude-proxy.git
cd claude-proxy
docker compose up -d
```

Point Claude Code at the proxy:

```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_BASE_URL=http://localhost:8082
```

Then open the dashboard: **http://localhost:8082/proxy/dashboard**

## Dashboard

The live dashboard auto-refreshes every 10 seconds and shows:

| Panel | What it shows |
|---|---|
| **Cost saved** | % and $ saved vs full Opus pricing without the proxy |
| **Cache hit rate** | Response cache hits vs total requests |
| **Prompt cache saves** | Tokens served from Anthropic's prompt cache (0.1× cost) |
| **Active model** | Exact model used on the last request + per-model request counts |
| **Token breakdown** | Without proxy (baseline) vs actual input, prompt cache reads, output |
| **Optimisation events** | Cache hits, model downgrades, context trims |
| **Runtime controls** | Toggle features and change models without restarting |

### Runtime controls (no restart needed)

All settings can be changed live from the dashboard or via API:

```bash
# Enable model routing
curl -X PATCH http://localhost:8082/proxy/config \
  -H "Content-Type: application/json" \
  -d '{"routing_enabled": true}'

# Change model thresholds
curl -X PATCH http://localhost:8082/proxy/config \
  -H "Content-Type: application/json" \
  -d '{"haiku_token_threshold": 1000, "sonnet_token_threshold": 5000}'

# Change which models are used for each tier
curl -X PATCH http://localhost:8082/proxy/config \
  -H "Content-Type: application/json" \
  -d '{"haiku_model": "claude-haiku-4-5", "sonnet_model": "claude-sonnet-4-6", "default_model": "claude-opus-4-7"}'
```

Configurable fields via `PATCH /proxy/config`:

| Field | Type | Description |
|---|---|---|
| `routing_enabled` | bool | Enable/disable model routing |
| `context_trim_enabled` | bool | Enable/disable context trimming |
| `prompt_cache_enabled` | bool | Enable/disable prompt cache injection |
| `response_cache_enabled` | bool | Enable/disable exact response cache |
| `haiku_token_threshold` | int | Tokens ≤ this → Haiku |
| `sonnet_token_threshold` | int | Tokens ≤ this → Sonnet |
| `context_max_turns` | int | Max turn pairs to keep when trimming |
| `default_model` | string | The "expensive" model routing routes away from |
| `haiku_model` | string | Model to use for short requests |
| `sonnet_model` | string | Model to use for medium requests |

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8082` | Port to listen on |
| `CACHE_ENABLED` | `true` | Enable exact response cache |
| `CACHE_MAXSIZE` | `1000` | Max cached responses |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |
| `PROMPT_CACHE_ENABLED` | `true` | Auto-inject `cache_control` |
| `PROMPT_CACHE_MIN_TOKENS` | `1024` | Min tokens to bother caching a prefix |
| `ROUTING_ENABLED` | `false` | Enable model routing (disable for Claude Code) |
| `HAIKU_THRESHOLD` | `2000` | Tokens ≤ this → Haiku |
| `SONNET_THRESHOLD` | `8000` | Tokens ≤ this → Sonnet |
| `CONTEXT_TRIM_ENABLED` | `false` | Trim long conversations (disable for Claude Code) |
| `CONTEXT_MAX_TURNS` | `20` | Max user/assistant turn pairs to keep |
| `BATCH_ENABLED` | `false` | Enable batch queue endpoint |
| `CLAUDE_CODE_MODE` | `true` | Shortcut: disables routing + context trim |

## Endpoints

### Proxied (transparent)
| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Main chat endpoint — streaming and non-streaming |
| `POST` | `/v1/messages/count_tokens` | Token counting |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{model_id}` | Get model details |

### Admin
| Method | Path | Description |
|---|---|---|
| `GET` | `/proxy/dashboard` | Live GUI dashboard |
| `GET` | `/proxy/health` | Health check |
| `GET` | `/proxy/stats` | Full stats JSON (tokens, cost, cache, models) |
| `PATCH` | `/proxy/config` | Update any config field at runtime |
| `POST` | `/proxy/cache/clear` | Clear the response cache |

### Batch (when `BATCH_ENABLED=true`)
| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages/batch` | Enqueue a request |
| `GET` | `/v1/messages/batch/{id}` | Check batch status |
| `GET` | `/v1/messages/batch/{id}/results` | Retrieve results |
| `POST` | `/v1/messages/batch/flush` | Flush pending queue |

## Check savings

```bash
curl http://localhost:8082/proxy/stats | python3 -m json.tool
```

Example output:
```json
{
  "proxy": {
    "requests_total": 142,
    "cache_hits": 38,
    "upstream_calls": 104,
    "input_tokens_total": 280000,
    "input_tokens_cache_read": 190000,
    "output_tokens_total": 42000,
    "cost_without_proxy_usd": 0.0182,
    "cost_with_proxy_usd": 0.0031,
    "last_model_used": "claude-sonnet-4-6",
    "models_seen": {"claude-sonnet-4-6": 104}
  },
  "response_cache": {
    "hits": 38,
    "misses": 104,
    "hit_rate": 0.268
  }
}
```

## Run tests (no API key required)

```bash
docker compose run --rm claude-proxy python test_proxy.py
```

## Architecture

```
Claude Code / any client
        │
        │  ANTHROPIC_BASE_URL=http://localhost:8082
        ▼
┌─────────────────────────┐
│   Claude Proxy (8082)   │
│                         │
│  1. Response cache hit? │──── yes ──► return cached
│          │ no           │
│  2. Trim context        │
│  3. Inject cache_control│
│  4. Route model         │
│          │              │
└──────────┼──────────────┘
           │
           ▼
   api.anthropic.com
```

## Docker commands

```bash
docker compose up -d             # start in background
docker compose logs -f           # live logs
docker compose restart           # restart
docker compose down              # stop and remove container
docker compose build --no-cache  # rebuild image
```
