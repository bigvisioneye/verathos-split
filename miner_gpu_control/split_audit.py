from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import httpx

from neurons.capacity_audit import (
    CapacityAuditRuntimeConfig,
    PROTOCOL_VERSION,
    CapacitySlot,
    capacity_gpu_pass_count,
    capacity_gpu_workload_spec,
    derive_proof_seed,
    lease_id,
    match_gpu_class,
    sign_artifact,
    slot_id,
    transcript_root,
    verify_artifact_signature,
)
from neurons.capacity_audit_combined import COMBINED_PROOF_FORMAT


def _hash_hex(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            h.update(part)
        else:
            h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _root(value: str) -> str:
    text = str(value or "").strip().lower()
    return text.removeprefix("0x")


@dataclass(frozen=True)
class SplitAuditSlotIdentity:
    """Public slot identity owned by the VPS frontend.

    This object is the identity boundary.  It may contain the EVM private key on
    the VPS, but the serialized GPU job produced from it never includes that key.
    """

    chain_id: int
    netuid: int
    endpoint: str
    evm_address: str
    model_index: int
    model_id: str
    quant: str
    max_context_len: int
    gpu_name: str
    vram_gb: int
    claimed_gpu_class: str = ""
    gpu_uuids: tuple[str, ...] = ()
    evm_private_key: str = ""

    def capacity_slot(self) -> CapacitySlot:
        return CapacitySlot(
            chain_id=int(self.chain_id),
            netuid=int(self.netuid),
            address=self.evm_address,
            model_index=int(self.model_index),
            endpoint=self.endpoint,
            model_id=self.model_id,
            quant=self.quant,
            max_context_len=int(self.max_context_len),
            gpu_name=self.gpu_name,
            gpu_count=1 if self.gpu_name else 0,
            vram_gb=int(self.vram_gb),
        )

    def public_payload(self) -> dict[str, Any]:
        return {
            "chain_id": int(self.chain_id),
            "netuid": int(self.netuid),
            "endpoint": self.endpoint,
            "evm_address": self.evm_address.lower(),
            "model_index": int(self.model_index),
            "model_id": self.model_id,
            "quant": self.quant,
            "max_context_len": int(self.max_context_len),
            "gpu_name": self.gpu_name,
            "vram_gb": int(self.vram_gb),
            "claimed_gpu_class": self.claimed_gpu_class,
            "gpu_uuids": list(self.gpu_uuids),
        }


@dataclass(frozen=True)
class SplitAuditJob:
    """Sanitized compute job sent from VPS to the rented GPU."""

    protocol_version: str
    audit_id: str
    slot_id: str
    lease_id: str
    endpoint: str
    evm_address: str
    model_index: int
    model_id: str
    quant: str
    max_context_len: int
    gpu_name: str
    vram_gb: int
    claimed_gpu_class: str
    gpu_uuids: tuple[str, ...]
    B_select: int
    B_start: int
    B_proof: int
    audit_epoch: int
    audit_block_hash: str
    proof_seed: str
    pass_count: int
    workload_spec: dict[str, Any] = field(default_factory=dict)
    deadline_s: float = 30.0

    def to_gpu_payload(self) -> dict[str, Any]:
        """Return the payload safe to send to a rented GPU.

        It deliberately has no hotkey seed and no EVM private key.
        """
        return {
            "protocol_version": self.protocol_version,
            "audit_id": self.audit_id,
            "slot_id": self.slot_id,
            "lease_id": self.lease_id,
            "endpoint": self.endpoint,
            "evm_address": self.evm_address.lower(),
            "model_index": int(self.model_index),
            "model_id": self.model_id,
            "quant": self.quant,
            "max_context_len": int(self.max_context_len),
            "gpu_name": self.gpu_name,
            "vram_gb": int(self.vram_gb),
            "claimed_gpu_class": self.claimed_gpu_class,
            "gpu_uuids": list(self.gpu_uuids),
            "B_select": int(self.B_select),
            "B_start": int(self.B_start),
            "B_proof": int(self.B_proof),
            "audit_epoch": int(self.audit_epoch),
            "audit_block_hash": self.audit_block_hash,
            "proof_seed": self.proof_seed,
            "pass_count": int(self.pass_count),
            "workload_spec": dict(self.workload_spec or {}),
            "deadline_s": float(self.deadline_s),
        }

    @classmethod
    def from_gpu_payload(cls, payload: Mapping[str, Any]) -> "SplitAuditJob":
        return cls(
            protocol_version=str(payload.get("protocol_version") or PROTOCOL_VERSION),
            audit_id=str(payload["audit_id"]),
            slot_id=str(payload["slot_id"]),
            lease_id=str(payload["lease_id"]),
            endpoint=str(payload["endpoint"]),
            evm_address=str(payload["evm_address"]).lower(),
            model_index=int(payload["model_index"]),
            model_id=str(payload["model_id"]),
            quant=str(payload.get("quant") or ""),
            max_context_len=int(payload.get("max_context_len") or 0),
            gpu_name=str(payload.get("gpu_name") or ""),
            vram_gb=int(payload.get("vram_gb") or 0),
            claimed_gpu_class=str(payload.get("claimed_gpu_class") or payload.get("gpu_name") or ""),
            gpu_uuids=tuple(str(x) for x in payload.get("gpu_uuids") or []),
            B_select=int(payload["B_select"]),
            B_start=int(payload["B_start"]),
            B_proof=int(payload["B_proof"]),
            audit_epoch=int(payload["audit_epoch"]),
            audit_block_hash=str(payload.get("audit_block_hash") or ""),
            proof_seed=str(payload.get("proof_seed") or ""),
            pass_count=int(payload.get("pass_count") or 0),
            workload_spec=dict(payload.get("workload_spec") or {}),
            deadline_s=float(payload.get("deadline_s") or 30.0),
        )


@dataclass(frozen=True)
class SplitAuditTimingResult:
    pass0_root: str
    final_root: str
    transcript_root: str
    final_timing: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "pass0_root": self.pass0_root,
            "final_root": self.final_root,
            "transcript_root": self.transcript_root,
            "final_timing": dict(self.final_timing or {}),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SplitAuditTimingResult":
        return cls(
            pass0_root=_root(str(payload["pass0_root"])),
            final_root=_root(str(payload["final_root"])),
            transcript_root=_root(str(payload["transcript_root"])),
            final_timing=dict(payload.get("final_timing") or {}),
        )


@dataclass(frozen=True)
class SplitAuditProofResult:
    proof_payload: dict[str, Any]
    final_summary: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "proof_payload": dict(self.proof_payload or {}),
            "final_summary": dict(self.final_summary or {}),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SplitAuditProofResult":
        return cls(
            proof_payload=dict(payload["proof_payload"]),
            final_summary=dict(payload.get("final_summary") or {}),
        )


def build_split_audit_job(
    identity: SplitAuditSlotIdentity,
    *,
    audit_id: str,
    audit_epoch: int,
    selection_block: int,
    audit_block: int,
    proof_challenge_block: int,
    audit_block_hash: str,
    pass_count: int | None = None,
    workload_spec: Mapping[str, Any] | None = None,
    deadline_s: float = 30.0,
    runtime_cfg: CapacityAuditRuntimeConfig | None = None,
) -> SplitAuditJob:
    slot = identity.capacity_slot()
    public_slot_id = slot_id(slot)
    public_lease = lease_id(slot, int(audit_epoch))
    proof_seed = derive_proof_seed(audit_block_hash, public_slot_id, 0)
    gpu_row = match_gpu_class(
        identity.gpu_name,
        int(identity.vram_gb),
        runtime_cfg or CapacityAuditRuntimeConfig(),
    )
    claimed_gpu_class = identity.claimed_gpu_class or (
        str(gpu_row.match_gpu_name) if gpu_row is not None else identity.gpu_name
    )
    resolved_pass_count = int(pass_count or 0)
    if resolved_pass_count <= 0 and gpu_row is not None:
        resolved_pass_count = capacity_gpu_pass_count(gpu_row)
    resolved_workload_spec = dict(workload_spec or {})
    if not resolved_workload_spec and gpu_row is not None:
        resolved_workload_spec = capacity_gpu_workload_spec(gpu_row)
    return SplitAuditJob(
        protocol_version=PROTOCOL_VERSION,
        audit_id=str(audit_id),
        slot_id=public_slot_id,
        lease_id=public_lease,
        endpoint=identity.endpoint,
        evm_address=identity.evm_address.lower(),
        model_index=int(identity.model_index),
        model_id=identity.model_id,
        quant=identity.quant,
        max_context_len=int(identity.max_context_len),
        gpu_name=identity.gpu_name,
        vram_gb=int(identity.vram_gb),
        claimed_gpu_class=claimed_gpu_class,
        gpu_uuids=tuple(identity.gpu_uuids),
        B_select=int(selection_block),
        B_start=int(audit_block),
        B_proof=int(proof_challenge_block),
        audit_epoch=int(audit_epoch),
        audit_block_hash=str(audit_block_hash),
        proof_seed=proof_seed,
        pass_count=resolved_pass_count,
        workload_spec=resolved_workload_spec,
        deadline_s=float(deadline_s),
    )


def assert_job_matches_identity(job: SplitAuditJob, identity: SplitAuditSlotIdentity) -> None:
    expected = build_split_audit_job(
        identity,
        audit_id=job.audit_id,
        audit_epoch=job.audit_epoch,
        selection_block=job.B_select,
        audit_block=job.B_start,
        proof_challenge_block=job.B_proof,
        audit_block_hash=job.audit_block_hash,
        pass_count=job.pass_count,
        workload_spec=job.workload_spec,
        deadline_s=job.deadline_s,
    )
    checked = (
        "slot_id",
        "lease_id",
        "endpoint",
        "evm_address",
        "model_index",
        "model_id",
        "quant",
        "max_context_len",
        "gpu_name",
        "vram_gb",
        "claimed_gpu_class",
        "proof_seed",
    )
    for name in checked:
        if getattr(job, name) != getattr(expected, name):
            raise ValueError(f"split audit job identity mismatch for {name}")


class SplitAuditArtifactBuilder:
    """VPS-side signer for artifacts generated from remote GPU output."""

    def __init__(self, identity: SplitAuditSlotIdentity) -> None:
        if not identity.evm_private_key:
            raise ValueError("identity.evm_private_key is required on the VPS signer")
        self.identity = identity

    def _base_artifact(self, job: SplitAuditJob) -> dict[str, Any]:
        assert_job_matches_identity(job, self.identity)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "audit_id": job.audit_id,
            "slot_id": job.slot_id,
            "address": self.identity.evm_address.lower(),
            "model_index": int(self.identity.model_index),
            "claimed_gpu_class": job.claimed_gpu_class,
            "gpu_index": 0,
            "B_select": int(job.B_select),
            "B_start": int(job.B_start),
            "B_proof": int(job.B_proof),
            "pass_count": int(job.pass_count),
        }

    def pass0_artifact(self, job: SplitAuditJob, result: SplitAuditTimingResult) -> dict[str, Any]:
        artifact = self._base_artifact(job)
        artifact.update(
            {
                "artifact_type": "capacity_audit_pass0_receipt",
                "pass0_root": result.pass0_root,
                "pass0_transcript_commit": result.pass0_root,
            }
        )
        return self._sign(artifact)

    def final_artifact(self, job: SplitAuditJob, result: SplitAuditTimingResult) -> dict[str, Any]:
        artifact = self._base_artifact(job)
        artifact.update(
            {
                "artifact_type": "capacity_audit_final_receipt",
                "pass0_root": result.pass0_root,
                "final_root": result.final_root,
                "final_transcript_commit": result.transcript_root,
            }
        )
        timing = dict(result.final_timing or {})
        if str(timing.get("proof_format") or "") == COMBINED_PROOF_FORMAT:
            artifact["combined"] = {
                "format": COMBINED_PROOF_FORMAT,
                "workload_version": timing.get("workload_version"),
                "pass_count": timing.get("pass_count"),
                "capacity_transcript_root": timing.get("capacity_transcript_root"),
                "capacity_tail_transcript_root": timing.get("capacity_tail_transcript_root"),
                "fp64_transcript_root": timing.get("fp64_transcript_root"),
                "combined_transcript_root": timing.get("combined_transcript_root"),
                "capacity_params": timing.get("capacity_params"),
                "capacity_tail_params": timing.get("capacity_tail_params"),
                "fp64_params": timing.get("fp64_params"),
                "workspace_mode": timing.get("workspace_mode"),
                "timed_cuda_component_s": timing.get("timed_cuda_component_s"),
                "timed_wall_s": timing.get("timed_wall_s"),
            }
        return self._sign(artifact)

    def proof_artifact(
        self,
        job: SplitAuditJob,
        timing: SplitAuditTimingResult,
        proof: SplitAuditProofResult,
    ) -> dict[str, Any]:
        payload = dict(proof.proof_payload or {})
        if str(payload.get("format") or "") != COMBINED_PROOF_FORMAT:
            raise ValueError("proof payload has unsupported format")
        sampled = 0
        capacity_proof = payload.get("capacity_proof")
        if isinstance(capacity_proof, Mapping):
            sampled_blob = capacity_proof.get("sampled")
            if isinstance(sampled_blob, Mapping):
                try:
                    sampled = int(sampled_blob.get("pass_index") or 0)
                except Exception:
                    sampled = 0
        artifact = self._base_artifact(job)
        artifact.update(
            {
                "artifact_type": "capacity_audit_proof_payload",
                "sampled_pass_index": sampled,
                "sampled_opening": {
                    "lease_id": job.lease_id,
                    "transcript_root": timing.transcript_root,
                    "pass0_root": timing.pass0_root,
                    "final_root": timing.final_root,
                    "pass_index": sampled,
                },
                "sampled_pass_proof": payload,
            }
        )
        return self._sign(artifact)

    def _sign(self, artifact: dict[str, Any]) -> dict[str, Any]:
        artifact["miner_signature"] = sign_artifact(artifact, self.identity.evm_private_key)
        if not verify_artifact_signature(artifact, self.identity.evm_address):
            raise ValueError("signed split-audit artifact failed local signature verification")
        return artifact


class SplitAuditStateWriter:
    """Writes the audit gate files read by gpu_slot_frontend."""

    def __init__(self, state_dir: str | os.PathLike[str]) -> None:
        self.state_dir = Path(state_dir)

    def path_for(self, *, slot_index: int, model_index: int) -> Path:
        return self.state_dir / f"slot-{int(slot_index)}-{int(model_index)}.json"

    def mark_active(
        self,
        *,
        slot_index: int,
        model_index: int,
        audit_id: str,
        until_ts: float,
        phase: str,
        job: SplitAuditJob | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "active": True,
            "audit_id": audit_id,
            "until_ts": float(until_ts),
            "phase": phase,
            "updated_at": time.time(),
        }
        if job is not None:
            payload.update(
                {
                    "endpoint": job.endpoint,
                    "evm_address": job.evm_address,
                    "model_index": int(job.model_index),
                    "slot_id": job.slot_id,
                }
            )
        return self._write(slot_index=slot_index, model_index=model_index, payload=payload)

    def clear(self, *, slot_index: int, model_index: int, audit_id: str = "") -> Path:
        payload = {
            "active": False,
            "audit_id": audit_id,
            "until_ts": 0.0,
            "updated_at": time.time(),
        }
        return self._write(slot_index=slot_index, model_index=model_index, payload=payload)

    def _write(self, *, slot_index: int, model_index: int, payload: Mapping[str, Any]) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(slot_index=slot_index, model_index=model_index)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return path


class SplitAuditGpuClient:
    """HTTP client used by the VPS coordinator to call a rented GPU runner."""

    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _raise_for_status(resp: httpx.Response, label: str) -> None:
        if resp.status_code < 400:
            return
        text = resp.text
        if len(text) > 1600:
            text = text[:1600] + "...<truncated>"
        raise RuntimeError(f"{label} returned HTTP {resp.status_code}: {text}")

    def prepare_timing(self, job: SplitAuditJob, *, start_timeout_s: float) -> dict[str, Any]:
        payload = {
            "job": job.to_gpu_payload(),
            "start_timeout_s": float(start_timeout_s),
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/split-audit/v1/prepare", json=payload)
            self._raise_for_status(resp, "split-audit prepare")
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("split-audit prepare returned non-object payload")
            return data

    def run_timing(self, job: SplitAuditJob) -> SplitAuditTimingResult:
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/split-audit/v1/timing", json=job.to_gpu_payload())
            self._raise_for_status(resp, "split-audit timing")
            return SplitAuditTimingResult.from_payload(resp.json())

    def run_proof(
        self,
        *,
        job: SplitAuditJob,
        timing: SplitAuditTimingResult,
        challenge_seed: str,
    ) -> SplitAuditProofResult:
        payload = {
            "job": job.to_gpu_payload(),
            "timing": timing.to_payload(),
            "challenge_seed": challenge_seed,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/split-audit/v1/proof", json=payload)
            self._raise_for_status(resp, "split-audit proof")
            return SplitAuditProofResult.from_payload(resp.json())

    def cancel(self, job: SplitAuditJob) -> dict[str, Any]:
        with httpx.Client(timeout=min(self.timeout_s, 30.0)) as client:
            resp = client.post(f"{self.base_url}/split-audit/v1/cancel", json=job.to_gpu_payload())
            self._raise_for_status(resp, "split-audit cancel")
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("split-audit cancel returned non-object payload")
            return data


class FakeSplitAuditGpuRunner:
    """Deterministic no-GPU runner for local tests.

    This validates split identity, artifact construction, and frontend gate state.
    It is not a replacement for the real hot-capacity workspace proof.
    """

    def prepare_timing(self, job: SplitAuditJob, *, start_timeout_s: float = 0.0) -> dict[str, Any]:
        return {
            "status": "prepared",
            "mode": "fake",
            "audit_id": job.audit_id,
            "slot_id": job.slot_id,
            "lease_id": job.lease_id,
            "start_timeout_s": float(start_timeout_s),
        }

    def run_timing(self, job: SplitAuditJob) -> SplitAuditTimingResult:
        pass0 = _hash_hex("pass0", job.audit_id, job.slot_id, job.proof_seed)
        final = _hash_hex("final", job.audit_id, job.slot_id, job.pass_count)
        transcript = transcript_root([pass0, final])
        timing = {
            "proof_format": COMBINED_PROOF_FORMAT,
            "workload_version": (job.workload_spec or {}).get("workload_version", "fake_split_audit"),
            "pass_count": int(job.pass_count),
            "capacity_transcript_root": pass0,
            "capacity_tail_transcript_root": final,
            "fp64_transcript_root": _hash_hex("fp64", job.audit_id),
            "combined_transcript_root": transcript,
            "capacity_params": dict(job.workload_spec or {}),
            "capacity_tail_params": {},
            "fp64_params": {},
            "workspace_mode": "fake",
            "timed_cuda_component_s": 0.0,
            "timed_wall_s": 0.0,
        }
        return SplitAuditTimingResult(
            pass0_root=pass0,
            final_root=final,
            transcript_root=transcript,
            final_timing=timing,
        )

    def run_proof(
        self,
        *,
        job: SplitAuditJob,
        timing: SplitAuditTimingResult,
        challenge_seed: str,
    ) -> SplitAuditProofResult:
        sampled = int(_hash_hex(challenge_seed, timing.transcript_root)[:8], 16) % max(1, job.pass_count)
        proof_payload = {
            "format": COMBINED_PROOF_FORMAT,
            "fake": True,
            "job_slot_id": job.slot_id,
            "challenge_seed": challenge_seed,
            "capacity_proof": {
                "sampled": {
                    "pass_index": sampled,
                    "root": timing.final_root,
                },
                "opening": _hash_hex("opening", challenge_seed, sampled),
            },
        }
        return SplitAuditProofResult(
            proof_payload=proof_payload,
            final_summary={"proof_payload": proof_payload},
        )

    def cancel(self, job: SplitAuditJob) -> dict[str, Any]:
        return {
            "status": "cancelled",
            "mode": "fake",
            "audit_id": job.audit_id,
            "slot_id": job.slot_id,
            "lease_id": job.lease_id,
        }


__all__ = [
    "FakeSplitAuditGpuRunner",
    "SplitAuditArtifactBuilder",
    "SplitAuditGpuClient",
    "SplitAuditJob",
    "SplitAuditProofResult",
    "SplitAuditSlotIdentity",
    "SplitAuditStateWriter",
    "SplitAuditTimingResult",
    "assert_job_matches_identity",
    "build_split_audit_job",
]
