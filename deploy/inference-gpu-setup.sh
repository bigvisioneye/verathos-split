#!/usr/bin/env bash
# =============================================================================
#  INFERENCE (vLLM) GPU setup — run this ON the inference vast box (ssh1).
#  Installs vLLM + verallm and verifies imports. Does NOT start the server or
#  download the model (that command is printed at the end; first run downloads).
#
#  NOTE: the box is CUDA 13 / Python 3.12. vLLM wheels can be version-sensitive
#  for that combo. If `pip install vllm` fails or pulls a torch that reports
#  cuda_avail False, pin a version (e.g. `vllm==0.10.*`) matching a torch build
#  that supports this driver — iterate here on the box, it's the one risky step.
#
#  Prereq: /root/verathos_subnet contains the code + dist/*.whl (rsync'd).
# =============================================================================
set -euo pipefail
ROOT=/root/verathos_subnet
cd "$ROOT"
test -f verallm/api/server.py || { echo "ERROR: verallm code missing — re-transfer the repo"; exit 1; }

python3 -m venv .venv 2>/dev/null || true
VP="$ROOT/.venv/bin"
"$VP/pip" install -q --upgrade pip wheel setuptools

# 1) vLLM (heavy; pulls a matching torch). Pin a version here if this box's
#    CUDA/py combo needs it.
"$VP/pip" install vllm

# 2) verathos runtime deps + compiled wheels (cp312)
"$VP/pip" install -q fastapi "uvicorn[standard]" httpx pydantic \
  eth-account web3 "bittensor>=10.2,<10.3" blake3 bcrypt substrate-interface PyYAML
"$VP/pip" install -q dist/zkllm-0.1.0-cp312-cp312-linux_x86_64.whl
"$VP/pip" install -q dist/hot_capacity_workspace_cuda-0.1.0-cp312-cp312-linux_x86_64.whl || true

# 3) verify
"$VP/python" -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| vllm', vllm.__version__)"
PYTHONPATH="$ROOT" "$VP/python" -c "import verallm.api.server as s; print('verallm.api.server import OK')"

cat <<EOF

SETUP OK.

Start the inference server on container port 8080 (first run downloads the model,
~5-6 GB). Auth is disabled because the split proxy fronts validator auth:

  VERATHOS_NO_VALIDATOR_AUTH=1 PYTHONPATH=$ROOT \\
    $VP/python -m verallm.api.server \\
      --model-id QuantTrio/Qwen3.5-9B-AWQ --quant int4 \\
      --max-model-len 262144 --host 0.0.0.0 --port 8080

Then verify locally:  curl -s localhost:8080/health
Externally it is:      http://38.64.63.84:20448/health
(this is the frontend's --gpu-pool-url)
EOF
