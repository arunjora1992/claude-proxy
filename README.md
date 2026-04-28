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

> **Claude Code users:** `CLAUDE_CODE_MODE=true` (the default in docker-compose) disables routing and context trimming — both can degrade code quality. Prompt caching and response caching remain active and are fully transparent.

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

No API key configuration needed in the proxy — it forwards the key from your Claude Code requests automatically.

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
| `GET` | `/proxy/health` | Health check |
| `GET` | `/proxy/stats` | Cache hit rate, savings estimates, config |
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
    "tokens_saved_estimate": 94200,
    "model_downgrade_count": 0,
    "context_trims": 0,
    "upstream_calls": 104
  },
  "response_cache": {
    "hits": 38,
    "misses": 104,
    "hit_rate": 0.268,
    "size": 98,
    "maxsize": 1000
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
docker compose up -d          # start in background
docker compose logs -f        # live logs
docker compose restart        # restart
docker compose down           # stop and remove container
docker compose build --no-cache  # rebuild image
```
