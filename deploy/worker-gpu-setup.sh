#!/usr/bin/env bash
# =============================================================================
#  SPLIT-AUDIT WORKER GPU setup — run this ON the worker GPU box.
#  Installs torch + the compiled CUDA wheels and verifies the runner imports.
#  It does NOT start the service (that command is printed at the end).
#
#  Prereq: $ROOT must contain the code + dist/*.whl (rsync'd from the repo).
#
#  CUDA VERSION CONSTRAINT (important, hard-won):
#    The compiled wheels (hot_capacity_workspace_cuda, zkllm) are built against
#    torch 2.11 ABI *and CUDA 12* — the native libs link libcudart.so.12 /
#    libcublas.so.12. You MUST install a CUDA-12 torch build (cu128), NOT the
#    default cu130 wheel, or the extension fails at import with
#    "libcudart.so.12: cannot open shared object file". A CUDA-13 driver runs
#    the CUDA-12 runtime fine (forward compatible).
# =============================================================================
set -euo pipefail
ROOT="${ROOT:-/root/verathos_subnet}"
cd "$ROOT"

# 0) sanity: code + wheels present
test -f miner_gpu_control/split_audit_gpu_runner.py || { echo "ERROR: runner code missing at $ROOT/miner_gpu_control — re-transfer the repo"; exit 1; }
WHL=$(ls dist/hot_capacity_workspace_cuda-0.1.0-cp312-cp312-linux_x86_64.whl 2>/dev/null) || { echo "ERROR: cp312 workspace wheel missing in dist/"; exit 1; }

# 1) venv (box is Python 3.12)
python3 -m venv .venv 2>/dev/null || true
VP="$ROOT/.venv/bin"
"$VP/pip" install -q --upgrade pip wheel setuptools

# 2) torch — CUDA 12.8 build to match the compiled wheels (see constraint above)
"$VP/pip" install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$VP/pip" install -q numpy fastapi "uvicorn[standard]" httpx pydantic

# 3) compiled CUDA wheels (cp312) + blake3 (zkllm proof transcript hashing)
"$VP/pip" install -q "$WHL"
"$VP/pip" install -q dist/zkllm-0.1.0-cp312-cp312-linux_x86_64.whl
"$VP/pip" install -q blake3

# 4) verify — import AND that the CUDA runtime resolves + proof crypto loads
"$VP/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
PYTHONPATH="$ROOT" "$VP/python" -c "import torch, hot_capacity_workspace_cuda; print('hot_capacity_workspace_cuda import OK')"
PYTHONPATH="$ROOT" "$VP/python" -c "import torch, zkllm.crypto.merkle, zkllm.crypto.hash_utils; print('zkllm.crypto (blake3) import OK')"
PYTHONPATH="$ROOT" "$VP/python" -c "import miner_gpu_control.split_audit_gpu_runner; print('runner import OK')"

cat <<EOF

SETUP OK.

Start the audit runner (real mode) on container port 8080 — prefer pm2 via
ecosystem.worker.example.js, or directly:

  SPLIT_AUDIT_GPU_RUNNER_MODE=real PYTHONPATH=$ROOT \\
    $VP/uvicorn miner_gpu_control.split_audit_gpu_runner:app --host 0.0.0.0 --port 8080

Then verify locally:  curl -s localhost:8080/split-audit/v1/health   (expect mode:real)
Register the EXTERNAL url (host:mapped-port) in the verathos-monitor pick1
balancer with the gpu_class your slots CLAIM (e.g. "NVIDIA A40").
EOF
