// PM2 ecosystem config for a Verathos SPLIT miner.
//
// Split model: each public slot is a full canonical miner process
// (neurons.miner) whose GPU compute is remote and shared. Identity, auth,
// receipts, and audit signing stay on this (cheap, GPU-less) VPS; inference and
// the capacity-audit workload run on a shared GPU pool + a central audit
// scheduler that live on the GPU machines.
//
//   VPS (this file):  N slot miners  ── serve via proxy ──▶  GPU inference pool
//                                     ── audit compute  ──▶  central audit scheduler ──▶ audit GPU runners
//
// Copy and fill in your values:
//   cp ecosystem.split.example.js ecosystem.config.js
//
// Each slot uses the SAME wallet/hotkey (one UID per VPS) but a UNIQUE
// endpoint and model_index. The two split flags added for this mode are:
//   --gpu-pool-url URL                 serve inference via a proxy to this pool
//   --capacity-audit-backend-url URL   run audit compute on this shared scheduler
// Omit both and neurons.miner is an ordinary single-box miner.
//
// IMPORTANT: unlike a GPU-bound canonical miner, split slots are cheap to
// restart (no local VRAM), so autorestart is on.
//
// Verify before scaling: model_index is assigned at on-chain registration.
// Two slots share one hotkey, so confirm your subnet permits multiple model
// registrations per hotkey and that each slot's SPLIT_MODEL_INDEX matches the
// index it registers, so /health and receipts report the right slot.

const REPO_ROOT = "<REPO_ROOT>";
const WALLET = "<WALLET>";
const HOTKEY = "<HOTKEY>";                       // one hotkey / UID for this VPS
const GPU_POOL_URL = "http://<GPU_POOL_HOST>:<PORT>";        // vLLM pool or scheduler exposing /chat,/inference
const AUDIT_SCHEDULER_URL = "http://<AUDIT_HOST>:19190";     // central audit scheduler (single, shared)

function slot({ name, endpoint, modelId, modelIndex, evmKeyEnv }) {
  return {
    name,
    script: ".venv-vllm/bin/python",
    args: [
      "-u -m neurons.miner",
      `--wallet ${WALLET} --hotkey ${HOTKEY}`,
      "--netuid 96 --subtensor-network finney",
      `--endpoint ${endpoint}`,
      `--model-id ${modelId}`,
      "--capacity-audit",
      `--gpu-pool-url ${GPU_POOL_URL}`,
      `--capacity-audit-backend-url ${AUDIT_SCHEDULER_URL}`,
    ].join(" "),
    cwd: REPO_ROOT,
    env: {
      // The serving proxy reads these for /health, /identity/challenge, and
      // receipt gating. SPLIT_MODEL_INDEX must match this slot's on-chain index.
      SPLIT_MODEL_INDEX: String(modelIndex),
      // Slot EVM private key (keep OUT of source control; set on the host).
      SPLIT_EVM_PRIVATE_KEY: process.env[evmKeyEnv] || "",
    },
    // GPU-less proxy — safe to restart.
    autorestart: true,
    max_restarts: 10,
    min_uptime: "30s",
    restart_delay: 5000,
    merge_logs: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    max_size: "50M",
    retain: 3,
  };
}

module.exports = {
  apps: [
    slot({
      name: "slot-1",
      endpoint: "https://<SLOT_1_PUBLIC_HOST>",
      modelId: "<MODEL_ID>",
      modelIndex: 1,
      evmKeyEnv: "SLOT_1_EVM_PRIVATE_KEY",
    }),
    slot({
      name: "slot-2",
      endpoint: "https://<SLOT_2_PUBLIC_HOST>",
      modelId: "<MODEL_ID>",
      modelIndex: 2,
      evmKeyEnv: "SLOT_2_EVM_PRIVATE_KEY",
    }),

    // ── Shared GPU-side services (run on the GPU machines, not here) ──
    //
    // These are not started by this VPS ecosystem; they are listed so the whole
    // topology is documented in one place. See docs/SPLIT_MINER_INTEGRATION.md.
    //
    //   * GPU inference pool: one or more vLLM servers (verallm.api.server)
    //     serving /chat and /inference with the proof plugin, behind
    //     GPU_POOL_URL. Each must serve the exact registered model.
    //
    //   * Central audit scheduler at AUDIT_SCHEDULER_URL: one process holding a
    //     global reservation table so two selected slots never double-book one
    //     audit GPU. Must be exactly one, shared by every slot on every VPS.
    //
    //   * Audit GPU runners: real hot-capacity workers the scheduler fans out
    //     to. Must fail closed (never emit a fake proof).
  ],
};
