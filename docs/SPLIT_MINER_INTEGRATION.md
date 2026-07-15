# Split-Miner Integration

Goal: fold the split-miner design (public identity on cheap VPS frontends, GPU
compute pooled and swappable behind schedulers) into this repo as first-class
miner code, so it stays current with upstream instead of living in a separate
project that vendors and monkeypatches these modules.

Guiding rule (unchanged from the standalone design):

```
Frontend owns miner identity.  GPU owns compute only.
Schedulers choose compute workers without changing slot identity.
```

## Key structural insight

`neurons/miner.py` is already two cooperating pieces with clean seams:

1. A **served subprocess** started by `MinerNeuron.start_server()` — the vLLM
   server that owns inference, proof generation, `/health`, `/epoch/receipt`.
   The neuron only talks to it over `localhost:PORT`.
2. A **cleanly-constructed audit worker** — `CapacityAuditMinerWorker`,
   instantiated in `main()` and run as a background thread.

The split design maps onto exactly these two seams. No fork of `miner.py` and no
monkeypatching are required — only a strategy seam on the audit worker and a
swap of the served process.

## What must NOT move

Everything the validator checks must stay in exactly one place — the worker:

- seed derivation (`derive_proof_seed`, `derive_proof_challenge_seed`)
- artifact construction (`_base_artifact`, `_pass0_artifact`, `_final_artifact`,
  `_proof_payload_artifact`)
- EVM signing (`sign_artifact`) and publishing (`_publish_receipt`,
  `_publish_proof`)
- proof-challenge-seed waiting against the chain (`_wait_for_proof_challenge_seed`)
- selection / block watching / drain state

Only the **GPU-compute mechanics** move behind a strategy:

- launch the hot-capacity workload
- release the timed run
- read pass0 / final-timing roots
- hand it the proof challenge seed
- collect the final proof summary
- cancel

## The audit-compute strategy seam

New module `neurons/capacity_audit_backend.py`:

```python
@dataclass
class AuditProgress:
    pass0_root: str = ""
    final_timing: dict | None = None
    done: bool = False
    error: str = ""

class CapacityAuditComputeBackend(Protocol):
    def ensure_ready(self, worker) -> bool: ...
    def prepare(self, worker, audit_slot, *, start_timeout_s: float) -> object | None: ...
    def start(self, worker, prepared, *, seed_hex: str, audit_id: str, b_start: int) -> None: ...
    def poll(self, worker, prepared) -> AuditProgress: ...
    def submit_challenge(self, worker, prepared, *, challenge_seed: str) -> None: ...
    def finalize(self, worker, prepared, *, timeout_s: float) -> dict: ...
    def cancel(self, worker, prepared) -> None: ...
```

`prepared` is opaque to the worker (a `PreparedAuditProcess` locally, a
`RemotePreparedAudit` remotely). `worker` is passed so a backend can reuse the
worker's local helpers (`_workspace_script`, `runtime_cfg`, etc.).

Two implementations:

- `LocalWorkspaceAuditComputeBackend` — the current behavior, extracted verbatim
  from `_prepare_audit_process`, `_workspace_audit_command`, the subprocess/file
  mechanics inside `_run_audit_slot`, and `_ensure_workspace_extension`.
- `RemoteAuditComputeBackend` — HTTP client to the central audit scheduler
  (`/v1/capacity-audit/{prepare,start,status,challenge,finalize,cancel}`). Fails
  **closed**: prepare returning `None`/raising degrades the slot to a clean
  `no_show`, never a fabricated proof.

`CapacityAuditMinerWorker.__init__` gains one parameter:

```python
audit_backend: CapacityAuditComputeBackend | None = None
# default None -> LocalWorkspaceAuditComputeBackend() (identical to today)
```

`_run_audit_slot` is refactored **once** into a single orchestration that calls
`self._audit_backend` for the six compute ops and keeps all identity/publishing
inline. `_prepare_audit_process`, `_terminate_prepared_audit`, and
`_ensure_workspace_extension` delegate to the backend. Canonical (non-split)
miners get byte-identical behavior because the default backend is the local one.

## Serving seam

The neuron starts a served process on `localhost:PORT`. For split serving, that
process is a proxy to the shared GPU pool instead of a local vLLM. It answers the
same endpoints the neuron and validators expect (`/health`, `/chat`,
`/inference`, `/epoch/receipt`, `/identity/challenge`, `/epoch/{n}/receipts`).
Selected via a serve-mode flag; local vLLM stays the default.

## Config flags (neurons/config.py + miner.py argparse)

```
# audit compute (choose one; balancer takes precedence)
--capacity-audit-balancer-url URL    # set -> Pick1AuditComputeBackend (lease a worker via verathos-monitor)
--capacity-audit-balancer-api-key K  # bearer key for the balancer
--capacity-audit-backend-url URL     # set -> RemoteAuditComputeBackend (orchestration scheduler)
# inference serving (choose one; pick-url preferred)
--gpu-pick-url URL                   # set -> proxy asks a balancer /api/pick per request
--gpu-pool-url URL                   # set -> proxy forwards to a static pool/LB
```

All default empty -> unchanged canonical single-box miner. Backend selection
precedence in `main()`: Pick1 (balancer) > Remote (backend-url) > Local.

## Run model

Each slot runs `miner.py main()` as its own process with per-slot identity
(endpoint, model_index, EVM key) plus the two URLs above. Shared infra: one GPU
inference pool and one central audit scheduler. Two slots per hotkey remains the
deployment shape; each is a full canonical miner whose compute happens to be
remote and shared.

## Phases (status)

- **P1 — done.** `capacity_audit_backend.py`: interface + `RemoteAuditComputeBackend`.
- **P2 — done.** `LocalWorkspaceAuditComputeBackend`, `CapacityAuditMinerWorker`
  routes compute through `self._audit_backend`; `--capacity-audit-backend-url`
  flag in `config.py`/`miner.py`.
- **P3 — done.** `neurons/split_serving.py` proxy + `--gpu-pool-url` flag;
  `MinerNeuron._server_cmd` launches the proxy instead of local vLLM when set;
  the local-GPU recommended-model gate is skipped in split serving.
- **P4 — done.** `ecosystem.split.example.js` run model (N slot miners per VPS
  sharing one GPU pool + one central audit scheduler).

### Serving proxy (`neurons/split_serving.py`)

Reuses the canonical `verallm.api.validator_auth.ValidatorAuthMiddleware` and
`verallm.api.receipt_store.ReceiptStore` so auth and receipt aggregation track
upstream. It serves `/health`, `/model_spec`, `/models`, `/identity/challenge`,
`/epoch/receipt`, `/epoch/{n}/receipts` locally and forwards `/chat` and
`/inference` (raw body + `x-validator-*` headers) to `--gpu-pool-url`, relaying
the proof SSE byte-for-byte. Slot identity comes from CLI/env
(`SPLIT_EVM_ADDRESS`, `SPLIT_EVM_PRIVATE_KEY`, `SPLIT_MODEL_ID`,
`SPLIT_MODEL_INDEX`, `SPLIT_GPU_POOL_URL`).

### Open nuances to validate on real hardware

- **model_index provenance.** The proxy launches before on-chain registration
  assigns a model_index, so it reports `SPLIT_MODEL_INDEX` from env rather than
  the discovered value. Validators key on chain (address, model_index), so this
  is likely informational — confirm, and set `SPLIT_MODEL_INDEX` per slot.
- **Multiple registrations per hotkey.** Two slots share one hotkey/UID with
  distinct model_index. Confirm the subnet permits this and each `neurons.miner`
  process registers the index its proxy advertises.
- **Drain awareness.** The canonical vLLM reads the capacity-audit state file to
  drain during an audit; the proxy currently does not. In the split design audit
  and inference use different GPU pools, so there is no contention to drain —
  revisit only if they share GPUs.

## Verification (requires GPU + a validator; not runnable in a dev box)

1. Canonical regression: with no flags set, a normal miner audits identically
   (default local backend). Diff `_run_audit_slot` behavior on a single box.
2. Split audit: `--capacity-audit-backend-url` set, run a real selection, confirm
   the validator records `pass0_seen -> timing_pass -> combined_proof_verified`,
   never `invalid_payload`.
3. Split serving: `--gpu-pool-url` set, confirm inference proofs verify and
   receipts round-trip (`POST /epoch/receipt` -> `GET /epoch/{n}/receipts`).

## Full topology & deployment (self-contained in this repo)

Three machine roles. All Python components live in this repo; the two balancers
live in the separate `verathos-monitor` project (they only pick/lease — no
identity or proof logic).

```
  Frontend VPS (GPU-less)        GPU worker box(es)          Inference GPU box(es)
  ─────────────────────         ──────────────────          ─────────────────────
  neurons.miner                  miner_gpu_control           verallm.api.server
   + neurons.split_serving        .split_audit_gpu_runner     (vLLM + proof plugin)
  nginx TLS  :443/19101          /split-audit/v1/*           /chat,/inference
        │                              ▲                            ▲
        │ audit compute (lease)        │ pick1                      │ /api/pick
        └───────────────▶ verathos-monitor pick1 balancer ─────────┘
                          verathos-monitor inference pick balancer
```

Per-role components **in this repo**:

| Role | Entry point | Setup | pm2 example |
|------|-------------|-------|-------------|
| Frontend miner | `neurons.miner` (+ `neurons.split_serving` proxy) | `deploy/frontend-tls-setup.sh`, `deploy/run-split-slot.example.sh` | `ecosystem.split.example.js` |
| Audit worker | `miner_gpu_control.split_audit_gpu_runner` | `deploy/worker-gpu-setup.sh` | `ecosystem.worker.example.js` |
| Inference | `verallm.api.server` | `deploy/inference-gpu-setup.sh` | — |

`miner_gpu_control` (the audit worker) imports `neurons.capacity_audit*`, so it
ships as a top-level package alongside `neurons/` and `verallm/` and is declared
in `pyproject.toml` packages.

### Compiled-wheel / CUDA constraints (must match, or imports fail)

The proof extensions are prebuilt wheels in `dist/` with strict pairings:

- **Audit worker:** `torch==2.11.0+cu128`. The `hot_capacity_workspace_cuda` /
  `zkllm` natives link **CUDA 12** (`libcudart.so.12`); the default cu130 wheel
  fails at import. Also needs `blake3` (zkllm Merkle/Fiat-Shamir hashing).
- **Inference:** `vllm==0.19.1` (pins **torch 2.10**) + `torch==2.10.0+cu128`,
  matching zkllm's `.torch210` variant. Also needs `blake3`.
- A CUDA-13 driver runs the CUDA-12 runtime fine (forward compatible).

The `deploy/*-setup.sh` scripts encode these and verify imports before printing
the start command.

### Claimed GPU class

The slot advertises a *lesser* GPU class than the physical inference GPU (e.g.
`NVIDIA A40` while the pool runs A100s) via `SPLIT_GPU_NAME/CLASS/VRAM_GB/
COMPUTE_CAPABILITY`. The audit enforces class by **timing** only (the proof is
deterministic math), so under-claiming is legitimate and gives timing margin.
The claimed class MUST equal the audit worker's registered `gpu_class` in the
pick1 balancer, or the lease won't match.
