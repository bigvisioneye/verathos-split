#!/usr/bin/env bash
# =============================================================================
#  Verathos SPLIT-miner slot — full configuration (ONE slot per copy of file).
#
#  Copy per slot, edit the CONFIG sections, run on the VPS:
#     cp run-split-slot.example.sh run-split-slot2.sh && ./run-split-slot2.sh
#
#  Fields marked [required] must be set. Everything else has a working default;
#  leave optional fields empty ("") to use the default / omit the flag.
# =============================================================================
set -euo pipefail

# ─── 1. Wallet / chain ───────────────────────────────────────────────────────
COLDKEY="96"                        # [required] bittensor wallet (coldkey) name  -> --wallet
HOTKEY="96-5"                       # [required] bittensor hotkey name            -> --hotkey
NETUID="96"                         # [required] 96 = mainnet · 405 = testnet
SUBTENSOR_NETWORK="finney"          # [required] finney = mainnet · test = testnet
SUBTENSOR_CHAIN_ENDPOINT=""         # optional: ws(s):// custom/local subtensor (overrides network)
CHAIN_CONFIG=""                     # optional: path to chain_config_*.json (default: repo mainnet)

# ─── 2. Public endpoint + serving ────────────────────────────────────────────
PUBLIC_URL="https://65.75.203.31:19102"     # [required] URL validators hit        -> --endpoint
PORT="19102"                                # [required] local port the proxy binds -> --port
MODEL_ID="QuantTrio/Qwen3.5-9B-AWQ"         # [required] registered model id        -> --model-id
QUANT="int4"                                # [required] int4 · fp16 · ...           -> --quant
MAX_MODEL_LEN="262144"                       # [required] context length             -> --max-model-len
MODEL_INDEX="35"                            # [required] on-chain model_index for THIS endpoint

# ─── 3. Slot EVM identity ────────────────────────────────────────────────────
# By default the neuron DERIVES the slot EVM key from the hotkey (this is how
# your current slots share address 0x7010BD…). Only set these to override with
# an explicit per-slot key.
EVM_PRIVATE_KEY_ENV=""              # optional: name of an env var holding the slot EVM private key
EVM_ADDRESS=""                      # optional: explicit slot EVM address (else derived)

# ─── 4. Split infra (where compute lives) ────────────────────────────────────
GPU_POOL_URL="https://216.193.128.133:42716"   # [required] vLLM pool serving /chat,/inference -> --gpu-pool-url
# Audit compute — choose ONE. Pick1 balancer takes precedence if both are set.
BALANCER_URL="http://MONITOR_HOST:8081"        # verathos-monitor pick1 balancer   -> --capacity-audit-balancer-url
BALANCER_API_KEY="PUT_BALANCER_KEY_HERE"       # == monitor .env CAPACITY_AUDIT_BALANCER_API_KEY
AUDIT_BACKEND_URL=""                           # OR orchestration scheduler, e.g. http://88.198.47.172:19190

# ─── 5. Validator auth + audit mode ──────────────────────────────────────────
VALIDATORS_PATH="/etc/miner-control-v2/verathos_validators.json"  # [required] allowlist file
CAPACITY_AUDIT_MODE="observe"       # observe · score_gate · soft_gate · enforce (miner startup tolerance)

# ─── 6. Serving proxy tuning (optional) ──────────────────────────────────────
SPLIT_REQUEST_TIMEOUT_S="330"       # inference forward timeout to the pool
SPLIT_POOL_HEALTH_TIMEOUT_S="5"     # /health poll timeout to the pool

# ─── 7. Operational (optional) ───────────────────────────────────────────────
SKIP_EXTERNAL_PORT_CHECK="1"        # 1 when behind a TLS front-proxy (endpoint not directly reachable)
AUTO_UPDATE="0"                     # 1 = auto-update on new releases
ANALYTICS="0"                       # 1 = emit analytics

# ─── 8. Capacity-audit timing (ADVANCED — normally leave empty) ──────────────
# These come from the on-chain subnet runtime config so the miner matches the
# validator. Setting them here is an EMERGENCY OVERRIDE ONLY — a mismatch with
# the validator causes timing failures. Leave empty to use chain values.
CA_DEADLINE_S=""
CA_TRANSPORT_GRACE_S=""
CA_PAYLOAD_DEADLINE_S=""
CA_LEAD_BLOCKS=""
CA_PROOF_CHALLENGE_DELAY_BLOCKS=""
CA_WORKER_POLL_S=""
CA_VALIDATOR_URLS=""                # emergency manual validator artifact targets (comma-separated)

# =============================================================================
#                       wiring — you should not need to edit
# =============================================================================

# Slot identity env the serving proxy (neurons.split_serving) reads:
export SPLIT_MODEL_ID="$MODEL_ID"
export SPLIT_MODEL_INDEX="$MODEL_INDEX"
export SPLIT_REQUEST_TIMEOUT_S SPLIT_POOL_HEALTH_TIMEOUT_S
export VERATHOS_VALIDATORS_PATH="$VALIDATORS_PATH"
export VERATHOS_CAPACITY_AUDIT_MODE="$CAPACITY_AUDIT_MODE"
[ -n "$EVM_ADDRESS" ] && export SPLIT_EVM_ADDRESS="$EVM_ADDRESS"

# Explicit slot EVM key (optional): resolve the named env var if provided.
EVM_PK=""
if [ -n "$EVM_PRIVATE_KEY_ENV" ]; then
  EVM_PK="${!EVM_PRIVATE_KEY_ENV:-}"
  [ -n "$EVM_PK" ] || { echo "ERROR: $EVM_PRIVATE_KEY_ENV is not set in the environment" >&2; exit 1; }
  export SPLIT_EVM_PRIVATE_KEY="$EVM_PK"
fi

args=( --wallet "$COLDKEY" --hotkey "$HOTKEY"
       --netuid "$NETUID" --subtensor-network "$SUBTENSOR_NETWORK"
       --endpoint "$PUBLIC_URL"
       --capacity-audit
       --gpu-pool-url "$GPU_POOL_URL"
       --port "$PORT" --model-id "$MODEL_ID" --quant "$QUANT" --max-model-len "$MAX_MODEL_LEN" )

[ -n "$SUBTENSOR_CHAIN_ENDPOINT" ] && args+=( --subtensor-chain-endpoint "$SUBTENSOR_CHAIN_ENDPOINT" )
[ -n "$CHAIN_CONFIG" ]             && args+=( --chain-config "$CHAIN_CONFIG" )
[ -n "$EVM_PK" ]                   && args+=( --private-key "$EVM_PK" )

if [ -n "$BALANCER_URL" ]; then
  args+=( --capacity-audit-balancer-url "$BALANCER_URL" --capacity-audit-balancer-api-key "$BALANCER_API_KEY" )
elif [ -n "$AUDIT_BACKEND_URL" ]; then
  args+=( --capacity-audit-backend-url "$AUDIT_BACKEND_URL" )
fi

[ "$SKIP_EXTERNAL_PORT_CHECK" = "1" ] && args+=( --skip-external-port-check )
[ "$AUTO_UPDATE" = "1" ]              && args+=( --auto-update )
[ "$ANALYTICS" = "1" ]               && args+=( --analytics )

[ -n "$CA_DEADLINE_S" ]                 && args+=( --capacity-audit-deadline-s "$CA_DEADLINE_S" )
[ -n "$CA_TRANSPORT_GRACE_S" ]          && args+=( --capacity-audit-transport-grace-s "$CA_TRANSPORT_GRACE_S" )
[ -n "$CA_PAYLOAD_DEADLINE_S" ]         && args+=( --capacity-audit-payload-deadline-s "$CA_PAYLOAD_DEADLINE_S" )
[ -n "$CA_LEAD_BLOCKS" ]                && args+=( --capacity-audit-lead-blocks "$CA_LEAD_BLOCKS" )
[ -n "$CA_PROOF_CHALLENGE_DELAY_BLOCKS" ] && args+=( --capacity-audit-proof-challenge-delay-blocks "$CA_PROOF_CHALLENGE_DELAY_BLOCKS" )
[ -n "$CA_WORKER_POLL_S" ]              && args+=( --capacity-audit-worker-poll-s "$CA_WORKER_POLL_S" )
[ -n "$CA_VALIDATOR_URLS" ]             && args+=( --capacity-audit-validator-urls "$CA_VALIDATOR_URLS" )

echo "Launching split slot: model_index=$MODEL_INDEX endpoint=$PUBLIC_URL port=$PORT"
exec python -m neurons.miner "${args[@]}"
