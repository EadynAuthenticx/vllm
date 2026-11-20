# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4: NVIDIA subprocess server launcher, canonical hardware IDs, and PGID cleanup."""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from tools.byog.bundle.writer import redact_env
from tools.byog.executor.interfaces import ServerHandle, ServerLauncher
from tools.byog.schemas.result import ProtocolVariant

STRIPPED_CANDIDATE_ENV_KEYS: frozenset[str] = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GITHUB_TOKEN",
        "BYOG_UPLOAD_TOKEN",
    }
)


class CacheResetUnsupportedError(RuntimeError):
    """Raised when the running server does not support POST /reset_prefix_cache."""


def canonicalize_gpu_name(raw_name: str) -> str:
    """Normalize nvidia-smi GPU product names to vllm-project/recipes taxonomy IDs."""
    low = raw_name.lower()
    for sku in (
        "gb300",
        "gb200",
        "b300",
        "b200",
        "h200",
        "h100",
        "a100",
        "l40s",
        "l4",
        "rtx 5090",
        "rtx pro 6000",
    ):
        if sku in low:
            return sku.replace(" ", "_")
    return low.replace("nvidia ", "").strip().replace(" ", "_")


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _run_cmd_text(cmd: list[str], timeout_s: int = 10) -> str | None:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=timeout_s, text=True
        )
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _query_gpu_memory_mib() -> list[int]:
    out = _run_cmd_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    values: list[int] = []
    for line in out.splitlines():
        token = line.strip()
        if token.isdigit():
            values.append(int(token))
    return values


class NvidiaSubprocessLauncher(ServerLauncher):
    """Launches vllm serve as a process-group-isolated local subprocess on NVIDIA."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        vllm_bin: str = "vllm",
        shutdown_grace_s: float = 15.0,
        vram_reclaim_timeout_s: float = 30.0,
        vram_idle_threshold_mib: int = 1024,
    ):
        self.host = host
        self.vllm_bin = vllm_bin
        self.shutdown_grace_s = shutdown_grace_s
        self.vram_reclaim_timeout_s = vram_reclaim_timeout_s
        self.vram_idle_threshold_mib = vram_idle_threshold_mib
        self._procs: dict[int, subprocess.Popen[bytes]] = {}

    @staticmethod
    def build_subprocess_env(
        base_env: dict[str, str],
        serve_env: dict[str, str],
        *,
        enable_dev_mode: bool = True,
    ) -> dict[str, str]:
        """Build child env with credentials stripped and protocol dev mode enforced."""
        env = {
            k: v
            for k, v in base_env.items()
            if k.upper() not in STRIPPED_CANDIDATE_ENV_KEYS
        }
        if enable_dev_mode:
            if serve_env.get("VLLM_SERVER_DEV_MODE") == "0":
                raise ValueError(
                    "serve.env.VLLM_SERVER_DEV_MODE=0 conflicts with the "
                    "harness measurement protocol."
                )
            env["VLLM_SERVER_DEV_MODE"] = "1"
        for k, v in serve_env.items():
            if k.upper() not in STRIPPED_CANDIDATE_ENV_KEYS:
                env[k] = str(v)
        return env

    def detect_hardware(self) -> dict[str, Any]:
        query = _run_cmd_text(
            [
                "nvidia-smi",
                (
                    "--query-gpu=index,name,driver_version,memory.total,memory.used,"
                    "power.limit,ecc.mode.current"
                ),
                "--format=csv,noheader,nounits",
            ]
        )
        gpus: list[dict[str, Any]] = []
        canonical_skus: list[str] = []
        if query:
            for line in query.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    sku = canonicalize_gpu_name(parts[1])
                    canonical_skus.append(sku)
                    gpus.append(
                        {
                            "index": parts[0],
                            "name": parts[1],
                            "canonical_sku": sku,
                            "driver_version": parts[2],
                            "memory_total_mib": parts[3],
                            "memory_used_mib": parts[4] if len(parts) > 4 else None,
                            "power_limit_w": parts[5] if len(parts) > 5 else None,
                            "ecc_mode": parts[6] if len(parts) > 6 else None,
                        }
                    )
        topo = _run_cmd_text(["nvidia-smi", "topo", "-m"])
        compute_apps = _run_cmd_text(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        foreign_procs = [
            line.strip() for line in (compute_apps or "").splitlines() if line.strip()
        ]
        return {
            "platform": "nvidia",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_count": len(gpus),
            "canonical_skus": sorted(set(canonical_skus)),
            "gpus": gpus,
            "topology": topo,
            "active_compute_processes": foreign_procs,
        }

    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
    ) -> ServerHandle:
        log_dir.mkdir(parents=True, exist_ok=True)
        port = _find_free_port(self.host)

        cfg = dict(serve_config)
        model = str(cfg.pop("model"))
        cfg_path = log_dir / "vllm_serve_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        cmd = [
            self.vllm_bin,
            "serve",
            model,
            "--host",
            self.host,
            "--port",
            str(port),
        ]
        if cfg:
            cmd.extend(["--config", str(cfg_path)])

        child_env = self.build_subprocess_env(dict(os.environ), serve_env)
        (log_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        (log_dir / "env.redacted.yaml").write_text(
            yaml.safe_dump(redact_env(child_env), sort_keys=True),
            encoding="utf-8",
        )

        log_path = log_dir / "server.log"
        log_fh = log_path.open("ab")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )
        log_fh.close()
        self._procs[proc.pid] = proc

        base_url = f"http://{self.host}:{port}"
        handle = ServerHandle(
            base_url=base_url,
            command=cmd,
            log_path=log_path,
            pid=proc.pid,
            metadata={"port": port, "config_path": str(cfg_path)},
        )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._procs.pop(proc.pid, None)
                raise RuntimeError(
                    f"vLLM server exited early with code {proc.returncode}; "
                    f"see {log_path}"
                )
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2.0) as resp:
                    if resp.status == 200:
                        return handle
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(1.0)

        self.stop(handle)
        raise TimeoutError(
            f"vLLM server did not become healthy within {timeout_s}s; see {log_path}"
        )

    def reset_cache(self, handle: ServerHandle) -> ProtocolVariant:
        req = urllib.request.Request(
            f"{handle.base_url}/reset_prefix_cache",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                if resp.status == 200:
                    return ProtocolVariant.RESET_CACHE
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CacheResetUnsupportedError(
                f"Prefix cache reset failed on {handle.base_url}: {exc}"
            ) from exc
        raise CacheResetUnsupportedError(
            f"Prefix cache reset returned non-200 on {handle.base_url}."
        )

    def stop(self, handle: ServerHandle) -> dict[str, Any]:
        pid = handle.pid
        terminated_pgid = False
        if pid is not None:
            proc = self._procs.pop(pid, None)
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
                terminated_pgid = True
            except ProcessLookupError:
                pass
            if proc is not None:
                try:
                    proc.wait(timeout=self.shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    proc.wait(timeout=5.0)

        # Verify GPU memory is released when nvidia-smi is available.
        vram_reclaimed = True
        final_mem = _query_gpu_memory_mib()
        if final_mem:
            deadline = time.monotonic() + self.vram_reclaim_timeout_s
            while time.monotonic() < deadline:
                final_mem = _query_gpu_memory_mib()
                if all(m <= self.vram_idle_threshold_mib for m in final_mem):
                    vram_reclaimed = True
                    break
                vram_reclaimed = False
                time.sleep(1.0)

        return {
            "terminated_pgid": terminated_pgid,
            "vram_reclaimed": vram_reclaimed,
            "final_memory_used_mib": final_mem,
        }
