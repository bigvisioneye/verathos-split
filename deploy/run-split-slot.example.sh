#!/usr/bin/env bash
# =============================================================================
#  Verathos SPLIT-miner slot — full configuration (ONE slot per copy of file).
#
#  Copy per slot, edit the CONFIG sections, run on the frontend VPS:
#     cp run-split-slot.example.sh run-split-slot2.sh && ./run-split-slot2.sh
#  (prefer pm2 via ecosystem.split.example.js for anything long-lived.)
#
#  Fields marked [required] must be set. Leave optional fields empty ("").
#
#  HTTPS: the slot's on-chain endpoint should be https://. Run
#  deploy/frontend-tls-setup.sh first — nginx terminates TLS on PUBLIC_PORT and
#  forwards to the miner proxy on BACKEND_PORT. PUBLIC_URL uses PUBLIC_PORT;
#  the miner binds --port BACKEND_PORT. They differ on purpose.
# =============================================================================
set -euo pipefail
ROOT="${ROOT:-/root/verathos_subnet}"
VP="$ROOT/.venv/bin"

# ─── 1. Wallet / chain ───────────────────────────────────────────────────────
COLDKEY="my-coldkey"                 # [required] bittensor wallet (coldkey) name  -> --wallet
HOTKEY="my-hotkey"                   # [required] bittensor hotkey name            -> --hotkey
NETUID="96"                          # [required] 96 = mainnet
SUBTENSOR_NETWORK="finney"           # [required] finney = mainnet · test = testnet
SUBTENSOR_CHAIN_ENDPOINT=""          # optional: ws(s):// custom/local subtensor
CHAIN_CONFIG=""                      # optional: path to chain_config_*.json (default: repo mainnet)

# ─── 2. Public endpoint (TLS) + internal proxy bind ─────────────────────────
PUBLIC_IP="203.0.113.10"             # [required] this VPS public IP
PUBLIC_PORT="19101"                  # [required] https port validators hit (nginx)
BACKEND_PORT="28082"                 # [required] internal port the miner proxy binds -> --port
PUBLIC_URL="https://$PUBLIC_IP:$PUBLIC_PORT"   #            -> --endpoint (registered on-chain)
MODEL_ID="QuantTrio/Qwen3.5-9B-AWQ"  # [required] registered model id              -> --model-id
QUANT="int4"                         # [required] int4 · fp16 · ...                 -> --quant
MAX_CONTEXT_LEN="262144"             # [required] context length                   -> --max-context-len
MODEL_INDEX="0"                      # [required] on-chain model_index for THIS endpoint
                                     #            (set AFTER first registration; read it from the log)

# ─── 3. Claimed GPU class (under-claim vs the physical inference GPU) ─────────
# The audit enforces GPU class via TIMING only; the proof is deterministic math.
# Advertising a LESSER class (e.g. A40 while the pool runs A100s) gives timing
# margin and is legitimate. Must MATCH the audit worker's registered gpu_class.
CLAIM_GPU_NAME="NVIDIA A40"
CLAIM_VRAM_GB="46"
CLAIM_COMPUTE_CAP="8.6"              # A40 = 8.6 (keep consistent with the claimed name)

# ─── 4. Split infra (where compute lives) ────────────────────────────────────
# Inference: choose ONE. --gpu-pick-url (balancer, per-request pick) is preferred;
# --gpu-pool-url is a single static pool/LB.
GPU_PICK_URL="http://MONITOR_HOST:3840/api/pick"   # verathos-monitor inference pick -> --gpu-pick-url
GPU_POOL_URL=""                                    # OR a static vLLM pool/LB        -> --gpu-pool-url
# Audit compute: choose ONE. Pick1 balancer takes precedence if both set.
BALANCER_URL="http://MONITOR_HOST:8081"            # verathos-monitor pick1 balancer -> --capacity-audit-balancer-url
BALANCER_API_KEY="PUT_BALANCER_KEY_HERE"           # == monitor .env CAPACITY_AUDIT_BALANCER_API_KEY
AUDIT_BACKEND_URL=""                               # OR orchestration scheduler url

# ─── 5. Validator auth + audit mode ──────────────────────────────────────────
VALIDATORS_PATH="/root/verathos_validators.json"   # allowlist file (auto-written by the miner)
CAPACITY_AUDIT_MODE="observe"        # observe · enforce (miner startup tolerance for backend-unready)

# ─── 6. Operational (optional) ───────────────────────────────────────────────
SKIP_EXTERNAL_PORT_CHECK="1"         # 1 when behind the TLS front-proxy (endpoint not directly bound)
AUTO_UPDATE="0"
ANALYTICS="0"

# =============================================================================
#                       wiring — you should not need to edit
# =============================================================================
# Slot identity + claimed-class env the serving proxy (neurons.split_serving) reads:
export SPLIT_MODEL_ID="$MODEL_ID"
export SPLIT_MODEL_INDEX="$MODEL_INDEX"
export SPLIT_GPU_NAME="$CLAIM_GPU_NAME" SPLIT_GPU_CLASS="$CLAIM_GPU_NAME"
export SPLIT_VRAM_GB="$CLAIM_VRAM_GB" SPLIT_COMPUTE_CAPABILITY="$CLAIM_COMPUTE_CAP"
export VERATHOS_VALIDATORS_PATH="$VALIDATORS_PATH"
export VERATHOS_CAPACITY_AUDIT_MODE="$CAPACITY_AUDIT_MODE"
[ -n "$GPU_PICK_URL" ] && export SPLIT_GPU_PICK_URL="$GPU_PICK_URL"

args=( --wallet "$COLDKEY" --hotkey "$HOTKEY"
       --netuid "$NETUID" --subtensor-network "$SUBTENSOR_NETWORK"
       --endpoint "$PUBLIC_URL"
       --capacity-audit
       --port "$BACKEND_PORT"
       --model-id "$MODEL_ID" --quant "$QUANT" --max-context-len "$MAX_CONTEXT_LEN" )

# Inference routing: pick-url preferred, else static pool.
if [ -n "$GPU_PICK_URL" ]; then
  args+=( --gpu-pick-url "$GPU_PICK_URL" )
elif [ -n "$GPU_POOL_URL" ]; then
  args+=( --gpu-pool-url "$GPU_POOL_URL" )
fi

# Audit compute: pick1 balancer preferred, else orchestration backend.
if [ -n "$BALANCER_URL" ]; then
  args+=( --capacity-audit-balancer-url "$BALANCER_URL" --capacity-audit-balancer-api-key "$BALANCER_API_KEY" )
elif [ -n "$AUDIT_BACKEND_URL" ]; then
  args+=( --capacity-audit-backend-url "$AUDIT_BACKEND_URL" )
fi

[ -n "$SUBTENSOR_CHAIN_ENDPOINT" ] && args+=( --subtensor-chain-endpoint "$SUBTENSOR_CHAIN_ENDPOINT" )
[ -n "$CHAIN_CONFIG" ]             && args+=( --chain-config "$CHAIN_CONFIG" )
[ "$SKIP_EXTERNAL_PORT_CHECK" = "1" ] && args+=( --skip-external-port-check )
[ "$AUTO_UPDATE" = "1" ]              && args+=( --auto-update )
[ "$ANALYTICS" = "1" ]               && args+=( --analytics )

echo "Launching split slot: model_index=$MODEL_INDEX endpoint=$PUBLIC_URL proxy_port=$BACKEND_PORT"
cd "$ROOT"
exec "$VP/python" -m neurons.miner "${args[@]}"
