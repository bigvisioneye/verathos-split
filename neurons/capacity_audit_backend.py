"""Capacity-audit GPU-compute strategy backends.

The capacity-audit miner worker owns everything the validator checks: seed
derivation, artifact construction, EVM signing, publishing, and the on-chain
proof-challenge-seed wait. Only the *GPU-compute mechanics* of an audit are
pluggable, and they live behind :class:`CapacityAuditComputeBackend`:

    launch the hot-capacity workload -> release the timed run ->
    read pass0 / final-timing roots -> hand it the proof challenge seed ->
    collect the final proof summary -> cancel

Two backends ship:

* :class:`LocalWorkspaceAuditComputeBackend` (default) runs the real
  ``hot_capacity_workspace.bench_combined`` subprocess on the local GPU. This is
  the canonical single-box behavior and is byte-compatible with the historical
  worker methods.
* :class:`RemoteAuditComputeBackend` routes the compute to a shared audit
  scheduler over HTTP so the public slot identity can live on a VPS while GPU
  compute is pooled and swappable. It fails **closed**: any transport or backend
  failure degrades the slot to a clean ``no_show`` (needs two misses, excusable
  by overlapping verified receipts) rather than fabricating a proof (a single
  cryptographically invalid proof zeroes the slot score and can zero the UID).

The worker passes itself into every backend call so a backend can reuse the
worker's local helpers (``_workspace_script``, ``runtime_cfg``, ``model_index``,
``_audit_lease``, ``_audit_challenge_timeout_s``) without this module importing
the worker at load time (which would be circular).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from neurons.capacity_audit import PROTOCOL_VERSION, slot_id as capacity_slot_id


@dataclass
class AuditProgress:
    """Snapshot of an in-flight audit's compute state.

    ``pass0_root`` is the compact CUDA pass-0 root once available; ``final_timing``
    is the parsed final-timing object once the timed run completes; ``done`` marks
    the compute finished (successfully or not); ``error`` carries a backend-side
    failure string, if any.
    """

    pass0_root: str = ""
    final_timing: Optional[dict] = None
    done: bool = False
    error: str = ""


@dataclass
class RemotePreparedAudit:
    """Opaque handle for an audit prepared on a remote scheduler."""

    job_id: str
    lease: str
    backend_url: str


class CapacityAuditComputeBackend:
    """Strategy interface for running the GPU side of a capacity audit.

    Implementations must never fabricate roots or proofs. If they cannot run the
    real workload they must fail closed — ``prepare`` returns ``None`` or a call
    raises — so the worker degrades the slot to a ``no_show`` instead of
    publishing an invalid proof.
    """

    #: How long the worker sleeps between :meth:`poll` calls, in seconds.
    poll_interval_s: float = 0.05

    def ensure_ready(self, worker: Any) -> bool:
        """Return whether the backend can run a real audit right now."""
        raise NotImplementedError

    def prepare(
        self,
        worker: Any,
        audit_slot: Any,
        *,
        start_timeout_s: float,
    ) -> Optional[object]:
        """Pre-launch (hot-start) the workload. Return an opaque handle or ``None``."""
        raise NotImplementedError

    def start(
        self,
        worker: Any,
        prepared: object,
        *,
        seed_hex: str,
        audit_id: str,
        b_start: int,
    ) -> None:
        """Release the timed run with the derived proof seed."""
        raise NotImplementedError

    def poll(self, worker: Any, prepared: object) -> AuditProgress:
        """Return the current compute progress (pass0 root, final timing, done)."""
        raise NotImplementedError

    def submit_challenge(
        self,
        worker: Any,
        prepared: object,
        *,
        challenge_seed: str,
    ) -> None:
        """Hand the workload the public post-commit proof challenge seed."""
        raise NotImplementedError

    def finalize(self, worker: Any, prepared: object, *, timeout_s: float) -> dict:
        """Block for proof assembly and return the final summary dict."""
        raise NotImplementedError

    def cancel(self, worker: Any, prepared: object) -> None:
        """Best-effort teardown of a prepared/running audit."""
        raise NotImplementedError


class RemoteAuditComputeBackend(CapacityAuditComputeBackend):
    """Route audit compute to a shared audit scheduler over HTTP.

    Speaks the ``/v1/capacity-audit/{prepare,start,status,challenge,finalize,
    cancel}`` API. The scheduler chooses a free audit GPU runner and holds the
    global reservation that keeps two selected slots from double-booking one GPU.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    # -- transport -----------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any], *, timeout: Optional[float] = None) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=timeout or self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"remote audit backend returned non-object for {path}")
        return data

    # -- backend interface ---------------------------------------------------

    def ensure_ready(self, worker: Any) -> bool:
        # Readiness of individual GPU runners is the scheduler's concern; it only
        # hands out backends that can prove. ``prepare`` fails closed otherwise.
        return True

    def prepare(
        self,
        worker: Any,
        audit_slot: Any,
        *,
        start_timeout_s: float,
    ) -> Optional[RemotePreparedAudit]:
        lease = worker._audit_lease(audit_slot)
        slot = audit_slot.slot
        payload = {
            "audit_id": audit_slot.audit_id,
            "lease_id": lease,
            "slot_id": capacity_slot_id(slot),
            "selection_block": int(audit_slot.selection_block),
            "audit_block": int(audit_slot.audit_block),
            "proof_challenge_block": int(audit_slot.proof_challenge_block),
            "passes": int(audit_slot.passes),
            "deadline_s": float(audit_slot.deadline_s),
            "challenge_timeout_s": float(worker._audit_challenge_timeout_s(audit_slot)),
            "workload_spec": dict(audit_slot.workload_spec or {}),
            "protocol_version": PROTOCOL_VERSION,
            "slot": {
                "endpoint": getattr(slot, "endpoint", ""),
                "address": getattr(slot, "address", ""),
                "model_index": int(getattr(slot, "model_index", worker.model_index)),
                "model_id": getattr(slot, "model_id", worker.model_id),
                "quant": getattr(slot, "quant", worker.quant),
                "max_context_len": int(getattr(slot, "max_context_len", worker.max_context_len) or 0),
                "gpu_name": getattr(slot, "gpu_name", ""),
                "vram_gb": int(getattr(slot, "vram_gb", 0) or 0),
            },
        }
        data = self._post(
            "/v1/capacity-audit/prepare",
            payload,
            timeout=max(float(start_timeout_s), 10.0),
        )
        job_id = str(data.get("job_id") or "")
        if not job_id:
            raise RuntimeError("remote audit backend prepare returned no job_id")
        return RemotePreparedAudit(
            job_id=job_id,
            lease=str(data.get("lease_id") or lease),
            backend_url=self.base_url,
        )

    def start(
        self,
        worker: Any,
        prepared: object,
        *,
        seed_hex: str,
        audit_id: str,
        b_start: int,
    ) -> None:
        assert isinstance(prepared, RemotePreparedAudit)
        self._post(
            "/v1/capacity-audit/start",
            {
                "job_id": prepared.job_id,
                "seed_hex": seed_hex,
                "audit_id": audit_id,
                "B_start": int(b_start),
            },
        )

    def poll(self, worker: Any, prepared: object) -> AuditProgress:
        assert isinstance(prepared, RemotePreparedAudit)
        data = self._post("/v1/capacity-audit/status", {"job_id": prepared.job_id})
        final_timing = data.get("final_timing")
        return AuditProgress(
            pass0_root=str(data.get("pass0_root") or "").strip(),
            final_timing=final_timing if isinstance(final_timing, dict) else None,
            done=bool(data.get("done")),
            error=str(data.get("error") or ""),
        )

    def submit_challenge(
        self,
        worker: Any,
        prepared: object,
        *,
        challenge_seed: str,
    ) -> None:
        assert isinstance(prepared, RemotePreparedAudit)
        self._post(
            "/v1/capacity-audit/challenge",
            {"job_id": prepared.job_id, "challenge_seed": challenge_seed},
        )

    def finalize(self, worker: Any, prepared: object, *, timeout_s: float) -> dict:
        assert isinstance(prepared, RemotePreparedAudit)
        data = self._post(
            "/v1/capacity-audit/finalize",
            {"job_id": prepared.job_id, "timeout_s": float(timeout_s)},
            timeout=max(float(timeout_s) + 15.0, 30.0),
        )
        summary = data.get("final_summary")
        return summary if isinstance(summary, dict) else {}

    def cancel(self, worker: Any, prepared: object) -> None:
        if not isinstance(prepared, RemotePreparedAudit):
            return
        try:
            self._post("/v1/capacity-audit/cancel", {"job_id": prepared.job_id}, timeout=10.0)
        except Exception:
            pass


@dataclass
class Pick1PreparedAudit:
    """Opaque handle for an audit run via a pick1/lease balancer.

    ``lease`` is the protocol capacity lease (used for challenge-seed derivation
    and the proof artifact); ``balancer_lease_id`` is the balancer's own lease,
    used only to release the worker. The two are intentionally distinct.
    """

    worker_url: str
    balancer_lease_id: str
    lease: str
    job: dict
    challenge_seed: str = ""
    timing_payload: Optional[dict] = None
    error: str = ""
    done: bool = False
    _thread: Optional[threading.Thread] = field(default=None, repr=False)


class Pick1AuditComputeBackend(CapacityAuditComputeBackend):
    """Route audit compute via a ``pick1``/lease/release worker balancer.

    The balancer (e.g. verathos-monitor's capacity-audit balancer) only *picks*
    a free worker of the right GPU class and hands back a lease; this backend
    then drives that worker's ``/split-audit/v1/{prepare,timing,proof,cancel}``
    API directly and releases the lease when done. Fails closed: any pick, HTTP,
    or backend failure degrades the slot to a ``no_show`` rather than a
    fabricated proof.

    The worker's ``/split-audit/v1/timing`` call blocks until the timed run
    finishes, so ``start`` kicks it off on a background thread and ``poll``
    reports progress from the stored result — mapping the same fields the local
    backend reads from its output files.
    """

    #: Balancer pick + worker timing are network round-trips; poll modestly.
    poll_interval_s: float = 0.1

    def __init__(self, balancer_url: str, *, api_key: str = "", timeout_s: float = 10.0):
        self.balancer_url = balancer_url.rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout_s = float(timeout_s)

    # -- transport -----------------------------------------------------------

    def _balancer_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _pick_worker(self, gpu_class: str) -> Optional[dict]:
        try:
            resp = httpx.get(
                f"{self.balancer_url}/pick1",
                params={"gpu_class": gpu_class},
                headers=self._balancer_headers(),
                timeout=self.timeout_s,
            )
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None

    def _release(self, prepared: "Pick1PreparedAudit") -> None:
        if not prepared.balancer_lease_id:
            return
        try:
            httpx.post(
                f"{self.balancer_url}/v1/release",
                json={"lease_id": prepared.balancer_lease_id},
                headers=self._balancer_headers(),
                timeout=self.timeout_s,
            )
        except Exception:
            pass

    @staticmethod
    def _worker_post(worker_url: str, path: str, payload: dict, *, timeout: float) -> dict:
        resp = httpx.post(f"{worker_url.rstrip('/')}{path}", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"worker {path} HTTP {resp.status_code}: {resp.text[:600]}")
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"worker {path} returned non-object payload")
        return data

    # -- job construction ----------------------------------------------------

    def _build_job(self, worker: Any, audit_slot: Any, lease: str) -> dict:
        slot = audit_slot.slot
        try:
            epoch_blocks = max(1, int(worker._epoch_blocks()))
        except Exception:
            epoch_blocks = int(os.getenv("VERATHOS_EPOCH_BLOCKS") or "360")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "audit_id": audit_slot.audit_id,
            "slot_id": capacity_slot_id(slot),
            "lease_id": lease,
            "endpoint": getattr(slot, "endpoint", ""),
            "evm_address": str(getattr(slot, "address", "") or "").lower(),
            "model_index": int(getattr(slot, "model_index", worker.model_index)),
            "model_id": getattr(slot, "model_id", worker.model_id),
            "quant": getattr(slot, "quant", worker.quant),
            "max_context_len": int(getattr(slot, "max_context_len", worker.max_context_len) or 0),
            "gpu_name": getattr(slot, "gpu_name", ""),
            "vram_gb": int(getattr(slot, "vram_gb", 0) or 0),
            "claimed_gpu_class": getattr(audit_slot, "gpu_class_name", "") or getattr(slot, "gpu_name", ""),
            "gpu_uuids": list(getattr(slot, "gpu_uuids", []) or []),
            "B_select": int(audit_slot.selection_block),
            "B_start": int(audit_slot.audit_block),
            "B_proof": int(audit_slot.proof_challenge_block),
            "audit_epoch": int(audit_slot.selection_block) // epoch_blocks,
            "audit_block_hash": "",
            "proof_seed": "",
            "pass_count": int(audit_slot.passes),
            "workload_spec": dict(audit_slot.workload_spec or {}),
            "deadline_s": float(audit_slot.deadline_s),
        }

    # -- backend interface ---------------------------------------------------

    def ensure_ready(self, worker: Any) -> bool:
        return True

    def prepare(
        self,
        worker: Any,
        audit_slot: Any,
        *,
        start_timeout_s: float,
    ) -> Optional[Pick1PreparedAudit]:
        gpu_class = str(getattr(audit_slot, "gpu_class_name", "") or getattr(audit_slot.slot, "gpu_name", "")).strip()
        picked = self._pick_worker(gpu_class)
        if not picked:
            return None
        worker_url = str(picked.get("endpoint") or picked.get("url") or "").strip()
        balancer_lease_id = str(picked.get("lease_id") or "").strip()
        if not worker_url:
            return None
        lease = worker._audit_lease(audit_slot)
        job = self._build_job(worker, audit_slot, lease)
        prepared = Pick1PreparedAudit(
            worker_url=worker_url,
            balancer_lease_id=balancer_lease_id,
            lease=lease,
            job=job,
        )
        try:
            self._worker_post(
                worker_url,
                "/split-audit/v1/prepare",
                {"job": job, "start_timeout_s": float(start_timeout_s) + 60.0},
                timeout=max(float(start_timeout_s), 10.0),
            )
        except Exception:
            self._release(prepared)
            return None
        return prepared

    def start(
        self,
        worker: Any,
        prepared: object,
        *,
        seed_hex: str,
        audit_id: str,
        b_start: int,
    ) -> None:
        assert isinstance(prepared, Pick1PreparedAudit)
        prepared.job["proof_seed"] = seed_hex
        prepared.job["B_start"] = int(b_start)
        timeout_s = max(
            120.0,
            float(prepared.job.get("deadline_s") or 30.0) + 90.0,
        )

        def _run_timing() -> None:
            try:
                timing = self._worker_post(
                    prepared.worker_url,
                    "/split-audit/v1/timing",
                    dict(prepared.job),
                    timeout=timeout_s,
                )
                prepared.timing_payload = timing
            except Exception as exc:
                prepared.error = f"{type(exc).__name__}: {exc}"
            finally:
                prepared.done = True

        thread = threading.Thread(target=_run_timing, name=f"pick1-timing-{audit_id[:8]}", daemon=True)
        prepared._thread = thread
        thread.start()

    def poll(self, worker: Any, prepared: object) -> AuditProgress:
        assert isinstance(prepared, Pick1PreparedAudit)
        timing = prepared.timing_payload
        if not isinstance(timing, dict):
            return AuditProgress(done=prepared.done, error=prepared.error)
        final_timing = timing.get("final_timing")
        return AuditProgress(
            pass0_root=str(timing.get("pass0_root") or "").strip(),
            final_timing=final_timing if isinstance(final_timing, dict) else None,
            done=prepared.done,
            error=prepared.error,
        )

    def submit_challenge(
        self,
        worker: Any,
        prepared: object,
        *,
        challenge_seed: str,
    ) -> None:
        assert isinstance(prepared, Pick1PreparedAudit)
        prepared.challenge_seed = challenge_seed

    def finalize(self, worker: Any, prepared: object, *, timeout_s: float) -> dict:
        assert isinstance(prepared, Pick1PreparedAudit)
        try:
            if not isinstance(prepared.timing_payload, dict) or not prepared.challenge_seed:
                return {}
            proof = self._worker_post(
                prepared.worker_url,
                "/split-audit/v1/proof",
                {
                    "job": dict(prepared.job),
                    "timing": dict(prepared.timing_payload),
                    "challenge_seed": prepared.challenge_seed,
                },
                timeout=max(float(timeout_s) + 10.0, 30.0),
            )
            summary = proof.get("final_summary")
            return summary if isinstance(summary, dict) else {}
        finally:
            self._release(prepared)

    def cancel(self, worker: Any, prepared: object) -> None:
        if not isinstance(prepared, Pick1PreparedAudit):
            return
        try:
            self._worker_post(
                prepared.worker_url, "/split-audit/v1/cancel", dict(prepared.job), timeout=10.0
            )
        except Exception:
            pass
        finally:
            self._release(prepared)


class LocalWorkspaceAuditComputeBackend(CapacityAuditComputeBackend):
    """Run the real hot-capacity workload as a local subprocess (canonical).

    This is the default backend and reproduces the historical single-box
    behavior. Process launch, teardown, and workspace readiness are delegated to
    the worker's existing helpers verbatim, so canonical miners are unaffected;
    only the small start/poll/finalize file operations live here.
    """

    #: Local file polling is cheap, so poll tightly for timing responsiveness.
    poll_interval_s: float = 0.02

    def ensure_ready(self, worker: Any) -> bool:
        script = worker._workspace_script()
        return bool(worker._ensure_workspace_extension(script.parent))

    def prepare(
        self,
        worker: Any,
        audit_slot: Any,
        *,
        start_timeout_s: float,
    ) -> Optional[object]:
        # Returns a PreparedAuditProcess (opaque here) or None if the workspace
        # extension is unavailable — the worker then degrades to a no_show.
        return worker._prepare_audit_process(audit_slot, start_timeout_s=start_timeout_s)

    def start(
        self,
        worker: Any,
        prepared: object,
        *,
        seed_hex: str,
        audit_id: str,
        b_start: int,
    ) -> None:
        proc = prepared.proc
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                "hot-start workload exited before B_start: "
                f"rc={proc.returncode} stderr_tail={(stderr or '')[-500:]} "
                f"stdout_tail={(stdout or '')[-300:]}"
            )
        start_payload = {
            "seed_hex": seed_hex,
            "audit_id": audit_id,
            "B_start": int(b_start),
            "t": time.time(),
        }
        tmp_start = prepared.start_file.with_suffix(prepared.start_file.suffix + ".tmp")
        tmp_start.write_text(json.dumps(start_payload, sort_keys=True) + "\n")
        os.replace(tmp_start, prepared.start_file)

    def poll(self, worker: Any, prepared: object) -> AuditProgress:
        from neurons.capacity_audit_miner import _root_hex

        pass0_root = ""
        pass0_path = prepared.out_dir / f"{prepared.lease}_pass0.json"
        if pass0_path.exists():
            try:
                data = json.loads(pass0_path.read_text())
                raw_root = data.get("root")
                if raw_root not in (None, "", []):
                    pass0_root = _root_hex(raw_root)
            except Exception:
                pass0_root = ""

        final_timing: Optional[dict] = None
        final_path = prepared.out_dir / f"{prepared.lease}_final_timing.json"
        if final_path.exists():
            try:
                data = json.loads(final_path.read_text())
                if isinstance(data, dict):
                    final_timing = data
            except Exception:
                final_timing = None

        return AuditProgress(
            pass0_root=pass0_root,
            final_timing=final_timing,
            done=prepared.proc.poll() is not None,
        )

    def submit_challenge(
        self,
        worker: Any,
        prepared: object,
        *,
        challenge_seed: str,
    ) -> None:
        worker._write_text_atomic(prepared.challenge_file, challenge_seed)

    def finalize(self, worker: Any, prepared: object, *, timeout_s: float) -> dict:
        proc = prepared.proc
        try:
            proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
        final_summary_path = prepared.out_dir / f"{prepared.lease}_final.json"
        if final_summary_path.exists():
            try:
                data = json.loads(final_summary_path.read_text())
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def cancel(self, worker: Any, prepared: object) -> None:
        worker._terminate_prepared_audit(prepared)


__all__ = [
    "AuditProgress",
    "CapacityAuditComputeBackend",
    "LocalWorkspaceAuditComputeBackend",
    "Pick1AuditComputeBackend",
    "Pick1PreparedAudit",
    "RemoteAuditComputeBackend",
    "RemotePreparedAudit",
]
