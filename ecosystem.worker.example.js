// PM2 ecosystem for a SPLIT-AUDIT GPU WORKER (runs on a GPU box).
//
// The worker exposes /split-audit/v1/{prepare,timing,proof,cancel,health} and
// runs the real hot-capacity + zkllm proof workload on its GPU. It holds NO
// wallet or identity — the frontend miner (ecosystem.split.example.js) leases
// it through the verathos-monitor pick1 balancer.
//
// Prereq: run deploy/worker-gpu-setup.sh first (installs torch cu128 + wheels +
// blake3 and verifies imports). SPLIT_AUDIT_GPU_RUNNER_MODE MUST be "real";
// "fake" is only for wiring tests and never produces a valid proof.
//
//   cp ecosystem.worker.example.js ecosystem.worker.config.js   # then edit
//   pm2 start ecosystem.worker.config.js && pm2 save

const REPO_ROOT = "/root/verathos_subnet";
const PY = `${REPO_ROOT}/.venv/bin/python`;
const PORT = 8080; // container port; map it externally and register THAT in the balancer

module.exports = {
  apps: [
    {
      name: "split-audit-runner",
      script: PY,
      interpreter: "none", // script IS the python binary
      cwd: REPO_ROOT,
      args: `-m uvicorn miner_gpu_control.split_audit_gpu_runner:app --host 0.0.0.0 --port ${PORT}`,
      env: {
        SPLIT_AUDIT_GPU_RUNNER_MODE: "real", // NEVER "fake" in production
        PYTHONPATH: REPO_ROOT,
        // SPLIT_AUDIT_KEEP_ARTIFACTS: "1", // uncomment to retain per-audit out_dirs for debugging
      },
      autorestart: true,
      max_restarts: 20,
      min_uptime: "20s",
    },
  ],
};
