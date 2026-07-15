"""Split-miner serving proxy.

Answers the public miner endpoints on the slot's port while forwarding inference
to a shared GPU pool, so the on-chain identity can live on a cheap VPS and GPU
compute can be pooled and swapped without changing the registered endpoint.

It is a drop-in replacement for the local ``verallm.api.server`` from the
neuron's point of view: it serves ``/health``, ``/model_spec``,
``/identity/challenge``, ``/chat``, ``/inference``, ``/epoch/receipt``, and
``/epoch/{n}/receipts`` on ``localhost:PORT`` so ``wait_for_health`` and the
validator both see a normal miner.

Identity stays here:

* validator auth reuses the canonical :class:`ValidatorAuthMiddleware`
* receipts reuse the canonical :class:`ReceiptStore`
* ``/identity/challenge`` signs ``nonce || evm_address`` with the slot EVM key

Only inference forwards to ``--gpu-pool-url`` (something that speaks the vLLM
``/chat`` and ``/inference`` proof-streaming API — a single server, a
load-balancer over several, or a scheduler that exposes those paths). The proof
bundle is produced by the pool's vLLM plugin and relayed back byte-for-byte, so
proofs and receipts verify exactly as for a co-located miner.
"""

from __future__ import annotations

import argparse
import base64
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from verallm.api.receipt_store import ReceiptStore
from verallm.api.validator_auth import ValidatorAuthMiddleware


def _clean_url(value: str) -> str:
    return str(value or "").strip().rstrip("/")


class _SlotState:
    """Per-slot identity and forwarding config, read from env at startup."""

    def __init__(self) -> None:
        # Static single-pool URL (for one GPU or a plain LB), OR a pick balancer
        # URL that returns the best GPU per request (verathos-monitor /api/pick).
        self.gpu_pool_url = _clean_url(os.environ.get("SPLIT_GPU_POOL_URL", ""))
        self.gpu_pick_url = _clean_url(os.environ.get("SPLIT_GPU_PICK_URL", ""))
        self.evm_address = str(os.environ.get("SPLIT_EVM_ADDRESS", "") or "").strip()
        self.evm_private_key = str(os.environ.get("SPLIT_EVM_PRIVATE_KEY", "") or "").strip()
        self.model_id = str(os.environ.get("SPLIT_MODEL_ID", "") or "").strip()
        try:
            self.model_index = int(os.environ.get("SPLIT_MODEL_INDEX", "0") or "0")
        except ValueError:
            self.model_index = 0
        self.request_timeout_s = float(os.environ.get("SPLIT_REQUEST_TIMEOUT_S", "330") or "330")
        self.pool_health_timeout_s = float(os.environ.get("SPLIT_POOL_HEALTH_TIMEOUT_S", "5") or "5")
        # Optional claimed-GPU override. The pool's real hardware may differ from
        # the class this slot advertises (e.g. claim a lesser calibrated class so
        # the physical GPU clears the capacity-audit timing with margin). When set,
        # /health reports these instead of the pool's GPU, which is what the
        # capacity-audit worker reads to derive the claimed class + workload.
        self.gpu_name = str(os.environ.get("SPLIT_GPU_NAME", "") or "").strip()
        self.gpu_class = str(os.environ.get("SPLIT_GPU_CLASS", "") or "").strip()
        self.vram_gb = str(os.environ.get("SPLIT_VRAM_GB", "") or "").strip()
        self.compute_capability = str(os.environ.get("SPLIT_COMPUTE_CAPABILITY", "") or "").strip()
        self.receipt_store = ReceiptStore()

    def apply_gpu_override(self, payload: dict) -> dict:
        """Replace the pool's reported GPU with the advertised class, if set."""
        if not (self.gpu_name or self.gpu_class or self.vram_gb or self.compute_capability):
            return payload
        hw = dict(payload.get("hardware") or {})
        if self.gpu_name:
            hw["gpu_name"] = self.gpu_name
        if self.vram_gb:
            try:
                hw["vram_gb"] = int(float(self.vram_gb))
            except ValueError:
                pass
        if self.compute_capability:
            hw["compute_capability"] = self.compute_capability
        payload["hardware"] = hw
        klass = self.gpu_class or self.gpu_name
        if klass:
            payload["gpu_class"] = klass
        return payload


state = _SlotState()
app = FastAPI(title="Verathos Split-Miner Serving Proxy", version="0.1.0")
# Same auth the canonical server uses: Sr25519 signature against the validator
# allowlist file, public endpoints exempt. Reused (not reimplemented) so it
# tracks upstream.
app.add_middleware(ValidatorAuthMiddleware)


@app.on_event("startup")
async def _startup() -> None:
    app.state.client = httpx.AsyncClient(
        timeout=state.request_timeout_s,
        verify=False,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    client: httpx.AsyncClient | None = getattr(app.state, "client", None)
    if client is not None:
        await client.aclose()


async def _resolve_gpu_base(client: httpx.AsyncClient) -> str:
    """Resolve the inference GPU base URL for one request.

    If a pick balancer is configured (verathos-monitor ``/api/pick``), query it
    per request so the busy-score router spreads load and picks the best GPU;
    it returns ``{"endpoint": ...}``. Otherwise use the static pool URL. Returns
    ``""`` if no GPU is available.
    """
    if state.gpu_pick_url:
        try:
            resp = await client.get(
                state.gpu_pick_url,
                params={"model_id": state.model_id} if state.model_id else None,
                timeout=state.pool_health_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return _clean_url(str(data.get("endpoint") or ""))
        except Exception:
            return ""
        return ""
    return state.gpu_pool_url


async def _pool_health(client: httpx.AsyncClient) -> dict[str, Any]:
    base = await _resolve_gpu_base(client)
    if not base:
        return {}
    try:
        resp = await client.get(f"{base}/health", timeout=state.pool_health_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Slot health: the pool's model/hardware view stamped with slot identity."""
    client: httpx.AsyncClient = app.state.client
    pool = await _pool_health(client)
    payload = dict(pool) if pool else {"status": "degraded"}
    payload.setdefault("status", "ok")
    if state.model_id:
        payload["model"] = state.model_id
    payload["model_index"] = state.model_index
    payload["slot_evm_address"] = state.evm_address
    payload["pool_reachable"] = bool(pool)
    return state.apply_gpu_override(payload)


@app.get("/model_spec")
async def model_spec() -> dict[str, Any]:
    client: httpx.AsyncClient = app.state.client
    pool = await _pool_health(client)
    return {
        "model": state.model_id or pool.get("model", ""),
        "model_index": state.model_index,
        "max_model_len": pool.get("max_model_len") or pool.get("max_context"),
        "quant": pool.get("quant", ""),
    }


@app.get("/models")
@app.get("/v1/models")
async def models() -> dict[str, Any]:
    client: httpx.AsyncClient = app.state.client
    pool = await _pool_health(client)
    return {
        "object": "list",
        "data": [
            {
                "id": state.model_id or pool.get("model", ""),
                "object": "model",
                "owned_by": "verathos",
                "model_index": state.model_index,
            }
        ],
    }


@app.post("/identity/challenge")
async def identity_challenge(body: dict[str, Any]):
    """Sign ``nonce || evm_address`` with the slot EVM key (same as canonical)."""
    if not state.evm_private_key or not state.evm_address:
        return JSONResponse(
            status_code=501,
            content={"error": "Identity challenge not available (no EVM key configured)"},
        )
    nonce_hex = str(body.get("nonce") or "")
    try:
        nonce_bytes = bytes.fromhex(nonce_hex)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid hex nonce"})
    if len(nonce_bytes) != 32:
        return JSONResponse(status_code=400, content={"error": "Nonce must be 32 bytes (64 hex chars)"})

    from eth_account import Account
    from eth_account.messages import encode_defunct

    address_bytes = bytes.fromhex(state.evm_address[2:])
    message = nonce_bytes + address_bytes
    signed = Account.sign_message(encode_defunct(primitive=message), private_key=state.evm_private_key)
    return {"address": state.evm_address, "signature": signed.signature.hex()}


@app.post("/epoch/receipt")
async def receive_epoch_receipt(body: dict[str, Any]):
    """Store a validator-pushed receipt for later pull (miner aggregates all)."""
    miner_address = str(body.get("miner_address") or "").lower()
    if state.evm_address and miner_address != state.evm_address.lower():
        return JSONResponse(
            status_code=403,
            content={"error": "Receipt address mismatch — this endpoint belongs to a different miner"},
        )
    try:
        epoch = int(body.get("epoch_number"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="epoch_number is required")
    count = state.receipt_store.add(epoch, dict(body))
    state.receipt_store.gc(epoch)
    return {"status": "accepted", "epoch": epoch, "count": count}


@app.get("/epoch/{epoch_number}/receipts")
async def get_epoch_receipts(epoch_number: int) -> dict[str, Any]:
    receipts = state.receipt_store.get(epoch_number)
    return {"epoch": epoch_number, "receipt_count": len(receipts), "receipts": receipts}


def _validator_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name in ("x-validator-hotkey", "x-validator-signature", "x-validator-timestamp"):
        value = request.headers.get(name)
        if value:
            headers[name] = value
    return headers


async def _forward_stream(request: Request, path: str):
    """Relay a proof-streaming inference request to the pool byte-for-byte.

    The target GPU is resolved per request (pick balancer) or from the static
    pool URL, so a multi-GPU inference pool is balanced without reconfiguring
    the miner.
    """
    client: httpx.AsyncClient = app.state.client
    base = await _resolve_gpu_base(client)
    if not base:
        raise HTTPException(
            status_code=503,
            detail="no inference GPU available (set SPLIT_GPU_PICK_URL or SPLIT_GPU_POOL_URL)",
        )
    raw_body = await request.body()
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **_validator_headers(request),
    }

    async def stream():
        try:
            async with client.stream(
                "POST",
                f"{base}{path}",
                content=raw_body,
                headers=headers,
                timeout=state.request_timeout_s,
            ) as resp:
                if resp.status_code >= 400:
                    detail = await resp.aread()
                    yield (
                        "event: error\n"
                        f'data: {{"error":"pool inference failed","status_code":{resp.status_code},'
                        f'"body":{_json_str(detail.decode("utf-8", errors="replace")[:2000])}}}\n\n'
                    )
                    return
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8", errors="replace")
        except Exception as exc:
            yield f'event: error\ndata: {{"error":{_json_str(f"{type(exc).__name__}: {exc}")}}}\n\n'

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _json_str(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


@app.post("/chat")
async def chat(request: Request):
    return await _forward_stream(request, "/chat")


@app.post("/inference")
async def inference(request: Request):
    return await _forward_stream(request, "/inference")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verathos split-miner serving proxy")
    parser.add_argument("--host", default=os.environ.get("SPLIT_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SPLIT_PORT", "8000") or "8000"))
    parser.add_argument("--gpu-pool-url", default=None)
    parser.add_argument("--gpu-pick-url", default=None)
    parser.add_argument("--evm-address", default=None)
    parser.add_argument("--evm-private-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-index", type=int, default=None)
    args = parser.parse_args()

    # CLI overrides env for the fields the neuron passes explicitly.
    if args.gpu_pool_url:
        state.gpu_pool_url = _clean_url(args.gpu_pool_url)
    if args.gpu_pick_url:
        state.gpu_pick_url = _clean_url(args.gpu_pick_url)
    if args.evm_address:
        state.evm_address = args.evm_address.strip()
    if args.evm_private_key:
        state.evm_private_key = args.evm_private_key.strip()
    if args.model:
        state.model_id = args.model.strip()
    if args.model_index is not None:
        state.model_index = int(args.model_index)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
