#!/usr/bin/env bash
# =============================================================================
#  INFERENCE (vLLM) GPU setup — run this ON the inference vast box (ssh1).
#  Installs vLLM + verallm and verifies imports. Does NOT start the server or
#  download the model (that command is printed at the end; first run downloads).
#
#  VERSION CONSTRAINT (hard-won): the zkllm proof wheel is built against
#  torch 2.10 ABI + CUDA 12. vLLM 0.19.1 is the version whose torch dependency
#  is exactly torch 2.10 (installed as +cu128 = CUDA 12.8 here), so zkllm's
#  native lib loads. Do NOT let pip pull a newer vLLM — it drags in a torch
#  (2.11+/cu130) that breaks zkllm with "libcudart.so.12: cannot open ...".
#  A CUDA-13 driver runs the CUDA-12 runtime fine (forward compatible).
#
#  Prereq: /root/verathos_subnet contains the code + dist/*.whl (rsync'd).
# =============================================================================
set -euo pipefail
ROOT="${ROOT:-/root/verathos_subnet}"
cd "$ROOT"
test -f verallm/api/server.py || { echo "ERROR: verallm code missing — re-transfer the repo"; exit 1; }

python3 -m venv .venv 2>/dev/null || true
VP="$ROOT/.venv/bin"
"$VP/pip" install -q --upgrade pip wheel setuptools

# 1) vLLM — PINNED. Pulls torch 2.10; force the +cu128 (CUDA 12) build so the
#    zkllm proof wheel's libcudart.so.12 resolves.
"$VP/pip" install vllm==0.19.1
"$VP/pip" install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

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
Register the EXTERNAL url (host:mapped-port) in the verathos-monitor inference
pick balancer; that balancer's /api/pick is the frontend's --gpu-pick-url.
EOF
