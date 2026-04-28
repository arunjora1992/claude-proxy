"""
Optional Batch API queue — 50% cost reduction for non-real-time requests.

Usage:
  POST /v1/messages/batch   → enqueue, returns {"batch_id": ..., "custom_id": ...}
  GET  /v1/messages/batch/{batch_id}  → check status
  GET  /v1/messages/batch/{batch_id}/results  → retrieve results

The queue collects requests and flushes them in bulk via the Batch API.
Results are polled in the background and stored in memory.
"""

import asyncio
import time
import uuid
from typing import Any, Optional

import anthropic


class BatchQueue:
    def __init__(self, client: anthropic.Anthropic, flush_interval: int = 60, max_pending: int = 100):
        self._client = client
        self._flush_interval = flush_interval
        self._max_pending = max_pending
        self._pending: list[dict] = []
        self._batches: dict[str, dict] = {}  # batch_id -> {status, results}
        self._results: dict[str, dict] = {}  # custom_id -> result
        self._lock = asyncio.Lock()

    async def enqueue(self, request: dict) -> dict:
        custom_id = f"proxy-{uuid.uuid4().hex[:12]}"
        async with self._lock:
            self._pending.append({"custom_id": custom_id, "request": request})
            if len(self._pending) >= self._max_pending:
                batch_id = await self._flush()
                return {"batch_id": batch_id, "custom_id": custom_id, "status": "pending"}
        return {"batch_id": None, "custom_id": custom_id, "status": "queued"}

    async def _flush(self) -> str:
        if not self._pending:
            return ""
        items = self._pending[:]
        self._pending.clear()

        requests = []
        for item in items:
            req = item["request"]
            requests.append(
                anthropic.types.messages.batch_create_params.Request(
                    custom_id=item["custom_id"],
                    params=anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                        model=req.get("model", "claude-opus-4-7"),
                        max_tokens=req.get("max_tokens", 4096),
                        messages=req.get("messages", []),
                        system=req.get("system", anthropic.NOT_GIVEN),
                        tools=req.get("tools", anthropic.NOT_GIVEN),
                    ),
                )
            )

        batch = self._client.messages.batches.create(requests=requests)
        batch_id = batch.id
        self._batches[batch_id] = {"status": "processing", "results": {}}
        # Schedule background polling
        asyncio.create_task(self._poll(batch_id))
        return batch_id

    async def _poll(self, batch_id: str) -> None:
        while True:
            await asyncio.sleep(30)
            batch = self._client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                results = {}
                for result in self._client.messages.batches.results(batch_id):
                    if result.result.type == "succeeded":
                        msg = result.result.message
                        results[result.custom_id] = {
                            "id": msg.id,
                            "type": "message",
                            "role": "assistant",
                            "content": [b.model_dump() for b in msg.content],
                            "model": msg.model,
                            "stop_reason": msg.stop_reason,
                            "usage": msg.usage.model_dump(),
                        }
                    else:
                        results[result.custom_id] = {
                            "error": result.result.type,
                        }
                self._batches[batch_id] = {"status": "ended", "results": results}
                self._results.update(results)
                break

    def get_batch_status(self, batch_id: str) -> Optional[dict]:
        return self._batches.get(batch_id)

    def get_result(self, custom_id: str) -> Optional[dict]:
        return self._results.get(custom_id)

    async def flush_all(self) -> Optional[str]:
        async with self._lock:
            if self._pending:
                return await self._flush()
        return None
