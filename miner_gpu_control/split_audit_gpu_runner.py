from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from neurons.capacity_audit import root_words_digest

from miner_gpu_control.split_audit import (
    FakeSplitAuditGpuRunner,
    SplitAuditJob,
    SplitAuditProofResult,
    SplitAuditTimingResult,
)


class ProofRequest(BaseModel):
    job: dict[str, Any] = Field(default_factory=dict)
    timing: dict[str, Any] = Field(default_factory=dict)
    challenge_seed: str


class PrepareRequest(BaseModel):
    job: dict[str, Any] = Field(default_factory=dict)
    start_timeout_s: float = 120.0


app = FastAPI(title="Split Audit GPU Runner", version="0.1.0")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _root_hex(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        raw = text[2:] if text.startswith("0x") else text
        try:
            bytes.fromhex(raw)
            if len(raw) == 64:
                return raw.lower()
        except ValueError:
            pass
    return root_words_digest(value)


@dataclass
class _ActiveJob:
    job: SplitAuditJob
    out_dir: Path
    lease: str
    challenge_file: Path
    proc: subprocess.Popen[str]
    start_file: Path | None = None
    ready_file: Path | None = None
    timing: SplitAuditTimingResult | None = None
    hot_start: bool = False
    released_at: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def pass0_path(self) -> Path:
        return self.out_dir / f"{self.lease}_pass0.json"

    @property
    def final_timing_path(self) -> Path:
        return self.out_dir / f"{self.lease}_final_timing.json"

    @property
    def final_path(self) -> Path:
        return self.out_dir / f"{self.lease}_final.json"

    @property
    def stdout_path(self) -> Path:
        return self.out_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.out_dir / "stderr.log"


class RealSplitAuditGpuRunner:
    """Run real hot-capacity CUDA work for a VPS-side split-audit signer."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, _ActiveJob] = {}
        self.max_active = max(1, int(os.getenv("SPLIT_AUDIT_GPU_MAX_ACTIVE", "1") or "1"))
        self.timing_timeout_s = max(10.0, float(os.getenv("SPLIT_AUDIT_TIMING_TIMEOUT_S", "120") or "120"))
        self.proof_timeout_s = max(10.0, float(os.getenv("SPLIT_AUDIT_PROOF_TIMEOUT_S", "120") or "120"))
        self.challenge_timeout_s = max(
            30.0,
            float(os.getenv("SPLIT_AUDIT_CHALLENGE_TIMEOUT_S", "240") or "240"),
        )
        self.keep_artifacts = _truthy(os.getenv("SPLIT_AUDIT_KEEP_ARTIFACTS"))

    @staticmethod
    def _job_key(job: SplitAuditJob) -> str:
        return f"{job.audit_id}:{job.slot_id}:{job.lease_id}"

    def _command(self, job: SplitAuditJob, active: _ActiveJob, *, start_timeout_s: float = 0.0) -> list[str]:
        cmd = [
            sys.executable,
            "-c",
            "from hot_capacity_workspace.bench_combined import main; main()",
            "--child",
            "--out-dir",
            str(active.out_dir),
            "--lease-id",
            active.lease,
            "--gpu-index",
            "0",
            "--challenge-file",
            str(active.challenge_file),
            "--challenge-timeout-s",
            str(max(self.challenge_timeout_s, float(job.deadline_s or 0.0) + 180.0)),
        ]
        if active.hot_start and active.start_file is not None:
            cmd.extend([
                "--start-file",
                str(active.start_file),
                "--start-timeout-s",
                str(max(1.0, float(start_timeout_s or 0.0))),
            ])
            if active.ready_file is not None:
                cmd.extend(["--ready-file", str(active.ready_file)])
        elif job.proof_seed:
            cmd.extend(["--seed-hex", job.proof_seed.removeprefix("0x")])
        for key, value in dict(job.workload_spec or {}).items():
            if key in {"workload_version", "pass_count"}:
                continue
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        return cmd

    def _start_job(
        self,
        job: SplitAuditJob,
        *,
        hot_start: bool = False,
        start_timeout_s: float = 0.0,
    ) -> _ActiveJob:
        out_dir = Path(tempfile.mkdtemp(prefix="verathos_split_audit_gpu_"))
        active = _ActiveJob(
            job=job,
            out_dir=out_dir,
            lease=job.lease_id,
            challenge_file=out_dir / f"{job.lease_id}_challenge.txt",
            start_file=out_dir / f"{job.lease_id}_start.json" if hot_start else None,
            ready_file=out_dir / f"{job.lease_id}_ready.json" if hot_start else None,
            hot_start=bool(hot_start),
            proc=None,  # type: ignore[arg-type]
        )
        stdout = active.stdout_path.open("w", encoding="utf-8")
        stderr = active.stderr_path.open("w", encoding="utf-8")
        try:
            active.proc = subprocess.Popen(
                self._command(job, active, start_timeout_s=start_timeout_s),
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        finally:
            stdout.close()
            stderr.close()
        return active

    def _cleanup_done_locked(self) -> None:
        for key, active in list(self._jobs.items()):
            if active.proc.poll() is not None and active.timing is None:
                self._jobs.pop(key, None)
                self._cleanup_dir(active)

    def _cleanup_dir(self, active: _ActiveJob) -> None:
        if self.keep_artifacts:
            return
        try:
            shutil.rmtree(active.out_dir)
        except Exception:
            pass

    def _tail(self, path: Path, limit: int = 1000) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-limit:]
        except Exception:
            return ""

    def _read_timing(self, active: _ActiveJob) -> SplitAuditTimingResult | None:
        if not active.final_timing_path.exists():
            return None
        final = json.loads(active.final_timing_path.read_text(encoding="utf-8"))
        pass0_root = ""
        if active.pass0_path.exists():
            try:
                pass0_root = _root_hex(json.loads(active.pass0_path.read_text(encoding="utf-8")).get("root"))
            except Exception:
                pass0_root = ""
        pass0_root = pass0_root or _root_hex(final.get("pass0_root"))
        final_root = _root_hex(final.get("root"))
        transcript = str(final.get("transcript_root") or final.get("combined_transcript_root") or final_root)
        if not pass0_root or not final_root or not transcript:
            raise RuntimeError("hot-capacity timing output missing roots")
        return SplitAuditTimingResult(
            pass0_root=pass0_root,
            final_root=final_root,
            transcript_root=transcript,
            final_timing=final,
        )

    def prepare_timing(self, job: SplitAuditJob, *, start_timeout_s: float = 0.0) -> dict[str, Any]:
        key = self._job_key(job)
        with self._lock:
            self._cleanup_done_locked()
            active = self._jobs.get(key)
            if active is None:
                if len(self._jobs) >= self.max_active:
                    raise RuntimeError("split-audit GPU runner is busy")
                active = self._start_job(job, hot_start=True, start_timeout_s=start_timeout_s)
                self._jobs[key] = active
            elif not active.hot_start:
                raise RuntimeError("split-audit job already started without hot-start")
        if active.proc.poll() is not None:
            raise RuntimeError(
                "hot-capacity child exited during prepare: "
                f"rc={active.proc.returncode} stderr_tail={self._tail(active.stderr_path)}"
            )
        return {
            "status": "prepared",
            "audit_id": job.audit_id,
            "slot_id": job.slot_id,
            "lease_id": job.lease_id,
            "pid": active.proc.pid,
            "hot_start": True,
            "ready": bool(active.ready_file and active.ready_file.exists()),
        }

    @staticmethod
    def _write_start_file(active: _ActiveJob, job: SplitAuditJob) -> None:
        if active.start_file is None:
            return
        payload = {
            "seed_hex": str(job.proof_seed or "").strip().removeprefix("0x"),
            "audit_id": job.audit_id,
            "B_start": int(job.B_start),
            "t": time.time(),
        }
        tmp = active.start_file.with_suffix(active.start_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, active.start_file)
        active.released_at = time.time()

    def run_timing(self, job: SplitAuditJob) -> SplitAuditTimingResult:
        key = self._job_key(job)
        with self._lock:
            self._cleanup_done_locked()
            active = self._jobs.get(key)
            if active is None:
                if len(self._jobs) >= self.max_active:
                    raise RuntimeError("split-audit GPU runner is busy")
                active = self._start_job(job)
                self._jobs[key] = active
            elif active.hot_start and active.released_at <= 0.0:
                self._write_start_file(active, job)
        deadline = time.time() + max(self.timing_timeout_s, float(job.deadline_s or 0.0) + 90.0)
        while time.time() < deadline:
            timing = self._read_timing(active)
            if timing is not None:
                active.timing = timing
                return timing
            if active.proc.poll() is not None:
                raise RuntimeError(
                    "hot-capacity child exited before final timing: "
                    f"rc={active.proc.returncode} stderr_tail={self._tail(active.stderr_path)}"
                )
            time.sleep(0.02)
        raise TimeoutError("timed out waiting for split-audit final timing")

    def run_proof(
        self,
        *,
        job: SplitAuditJob,
        timing: SplitAuditTimingResult,
        challenge_seed: str,
    ) -> SplitAuditProofResult:
        key = self._job_key(job)
        with self._lock:
            active = self._jobs.get(key)
        if active is None:
            with self._lock:
                active_keys = list(self._jobs.keys())
            raise RuntimeError(
                "split-audit timing job is not active "
                f"key={key} active_keys={active_keys[:5]}"
            )
        if active.timing is None:
            active.timing = timing
        active.challenge_file.write_text(str(challenge_seed).strip().removeprefix("0x") + "\n", encoding="utf-8")
        try:
            active.proc.communicate(timeout=self.proof_timeout_s)
        except subprocess.TimeoutExpired:
            active.proc.terminate()
            try:
                active.proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                active.proc.kill()
                active.proc.communicate()
            raise TimeoutError("timed out waiting for split-audit proof payload")
        try:
            if active.proc.returncode not in (0, None):
                raise RuntimeError(
                    "hot-capacity child failed while producing proof: "
                    f"rc={active.proc.returncode} out_dir={active.out_dir} "
                    f"stdout_tail={self._tail(active.stdout_path)} "
                    f"stderr_tail={self._tail(active.stderr_path)}"
                )
            if not active.final_path.exists():
                raise RuntimeError(
                    "final summary missing after proof: "
                    f"out_dir={active.out_dir} stdout_tail={self._tail(active.stdout_path)} "
                    f"stderr_tail={self._tail(active.stderr_path)}"
                )
            final_summary = json.loads(active.final_path.read_text(encoding="utf-8"))
            proof_payload = final_summary.get("proof_payload")
            if not isinstance(proof_payload, dict):
                raise RuntimeError(
                    "final summary missing proof_payload: "
                    f"out_dir={active.out_dir} keys={sorted(final_summary.keys())} "
                    f"stdout_tail={self._tail(active.stdout_path)} "
                    f"stderr_tail={self._tail(active.stderr_path)}"
                )
            return SplitAuditProofResult(proof_payload=proof_payload, final_summary=final_summary)
        finally:
            with self._lock:
                self._jobs.pop(key, None)
            self._cleanup_dir(active)

    def cancel(self, job: SplitAuditJob) -> dict[str, Any]:
        key = self._job_key(job)
        with self._lock:
            active = self._jobs.pop(key, None)
        if active is None:
            return {
                "status": "not_found",
                "audit_id": job.audit_id,
                "slot_id": job.slot_id,
                "lease_id": job.lease_id,
            }
        if active.proc.poll() is None:
            try:
                active.proc.terminate()
                active.proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                active.proc.kill()
                active.proc.communicate()
            except Exception:
                pass
        self._cleanup_dir(active)
        return {
            "status": "cancelled",
            "audit_id": job.audit_id,
            "slot_id": job.slot_id,
            "lease_id": job.lease_id,
        }

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_done_locked()
            active = len(self._jobs)
            jobs = [
                {
                    "audit_id": item.job.audit_id,
                    "slot_id": item.job.slot_id,
                    "model_index": item.job.model_index,
                    "has_timing": item.timing is not None,
                    "hot_start": item.hot_start,
                    "released": item.released_at > 0.0,
                    "ready": bool(item.ready_file and item.ready_file.exists()),
                    "running": item.proc.poll() is None,
                }
                for item in self._jobs.values()
            ]
        return {
            "status": "ok",
            "service": "split-audit-gpu-runner",
            "mode": "real",
            "gpu_required": True,
            "active": active,
            "max_active": self.max_active,
            "jobs": jobs,
        }


runner: Any
if os.getenv("SPLIT_AUDIT_GPU_RUNNER_MODE", "fake").strip().lower() == "real":
    runner = RealSplitAuditGpuRunner()
else:
    runner = FakeSplitAuditGpuRunner()


@app.get("/split-audit/v1/health")
async def health() -> dict[str, Any]:
    health_fn = getattr(runner, "health", None)
    if callable(health_fn):
        return health_fn()
    return {"status": "ok", "service": "split-audit-gpu-runner", "mode": "fake", "gpu_required": False}


@app.post("/split-audit/v1/prepare")
async def prepare_timing(request: PrepareRequest):
    try:
        job = SplitAuditJob.from_gpu_payload(request.job)
        prepare_fn = getattr(runner, "prepare_timing", None)
        if callable(prepare_fn):
            return prepare_fn(job, start_timeout_s=float(request.start_timeout_s or 0.0))
        return {"status": "prepared", "mode": "unsupported-noop"}
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"{exc.__class__.__name__}: {exc}"},
        )


@app.post("/split-audit/v1/timing")
async def run_timing(payload: dict[str, Any]):
    try:
        job = SplitAuditJob.from_gpu_payload(payload)
        result = runner.run_timing(job)
        return result.to_payload()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"{exc.__class__.__name__}: {exc}"},
        )


@app.post("/split-audit/v1/cancel")
async def cancel_timing(payload: dict[str, Any]):
    try:
        job = SplitAuditJob.from_gpu_payload(payload)
        cancel_fn = getattr(runner, "cancel", None)
        if callable(cancel_fn):
            return cancel_fn(job)
        return {"status": "cancelled", "mode": "unsupported-noop"}
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"{exc.__class__.__name__}: {exc}"},
        )


@app.post("/split-audit/v1/proof")
async def run_proof(request: ProofRequest):
    try:
        job = SplitAuditJob.from_gpu_payload(request.job)
        timing = SplitAuditTimingResult.from_payload(request.timing)
        result = runner.run_proof(
            job=job,
            timing=timing,
            challenge_seed=request.challenge_seed,
        )
        return result.to_payload()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"{exc.__class__.__name__}: {exc}"},
        )
