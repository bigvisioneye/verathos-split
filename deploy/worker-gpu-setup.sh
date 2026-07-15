#!/usr/bin/env bash
# =============================================================================
#  SPLIT-AUDIT WORKER GPU setup — run this ON the worker vast box (ssh4).
#  Installs torch + the CUDA hot-capacity wheel and verifies the runner imports.
#  It does NOT start the service (that command is printed at the end).
#
#  Prereq: /root/verathos_subnet must contain the code + dist/*.whl (rsync'd).
# =============================================================================
set -euo pipefail
ROOT=/root/verathos_subnet
cd "$ROOT"

# 0) sanity: code + wheels present
test -f miner_gpu_control/split_audit_gpu_runner.py || { echo "ERROR: runner code missing at $ROOT/miner_gpu_control — re-transfer the repo"; exit 1; }
WHL=$(ls dist/hot_capacity_workspace_cuda-0.1.0-cp312-cp312-linux_x86_64.whl 2>/dev/null) || { echo "ERROR: cp312 workspace wheel missing in dist/"; exit 1; }

# 1) venv (box is Python 3.12)
python3 -m venv .venv 2>/dev/null || true
VP="$ROOT/.venv/bin"
"$VP/pip" install -q --upgrade pip wheel setuptools

# 2) torch (CUDA build for the hot-capacity workload) + service deps
"$VP/pip" install -q torch numpy
"$VP/pip" install -q fastapi "uvicorn[standard]" httpx pydantic

# 3) compiled CUDA wheels (cp312)
"$VP/pip" install -q "$WHL"
"$VP/pip" install -q dist/zkllm-0.1.0-cp312-cp312-linux_x86_64.whl || true

# 4) verify
"$VP/python" -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"
PYTHONPATH="$ROOT" "$VP/python" -c "from hot_capacity_workspace.bench_combined import main; print('bench_combined import OK')"
PYTHONPATH="$ROOT" "$VP/python" -c "import miner_gpu_control.split_audit_gpu_runner; print('runner import OK')"

cat <<EOF

SETUP OK.

Start the audit runner (real mode) on container port 8080:

  SPLIT_AUDIT_GPU_RUNNER_MODE=real PYTHONPATH=$ROOT \\
    $VP/uvicorn miner_gpu_control.split_audit_gpu_runner:app --host 0.0.0.0 --port 8080

Then verify locally:  curl -s localhost:8080/split-audit/v1/health
Externally it is:      http://38.255.16.21:51016/split-audit/v1/health
(register THAT url in the balancer with gpu_class "NVIDIA A40")
EOF
