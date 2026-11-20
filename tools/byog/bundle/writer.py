# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C5: Result bundle writer, env redaction, manifest builder, and repro.sh."""

from __future__ import annotations

import hashlib
import os
import shlex
import tarfile
from pathlib import Path

import yaml

from tools.byog.schemas.catalog import PlatformKind
from tools.byog.schemas.result import ArtifactEntry, BundleManifest, RunResult
from tools.byog.schemas.spec import ResolvedJob

SENSITIVE_KEY_MARKERS: tuple[str, ...] = (
    "TOKEN",
    "SECRET",
    "KEY",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
)


def redact_env(env: dict[str, str]) -> dict[str, str]:
    """Redact credentials/tokens from environment snapshots before bundling."""
    redacted: dict[str, str] = {}
    for k, v in sorted(env.items()):
        upper = k.upper()
        if any(marker in upper for marker in SENSITIVE_KEY_MARKERS):
            redacted[k] = "<REDACTED>"
        else:
            redacted[k] = v
    return redacted


def write_repro_sh(
    attempt_dir: Path,
    job: ResolvedJob,
    *,
    image_ref: str = "<registry>/byog@sha256:<digest>",
    platform: PlatformKind = PlatformKind.NVIDIA,
) -> Path:
    """Generate a platform-appropriate repro.sh without embedding credentials."""
    if platform == PlatformKind.ROCM:
        gpu_flags = (
            "--device=/dev/kfd --device=/dev/dri "
            "--group-add video --security-opt seccomp=unconfined"
        )
    else:
        gpu_flags = "--gpus all"

    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Reproduces BYOG job_id={job.job_id} attempt_id={job.attempt_id}",
            f"BYOG_IMAGE={shlex.quote(image_ref)}",
            'SPEC_PATH="${SPEC_PATH:-$PWD/spec.resolved.yaml}"',
            'WEIGHTS_DIR="${WEIGHTS_DIR:-$PWD/weights}"',
            'OUT_DIR="${OUT_DIR:-$PWD/out}"',
            'mkdir -p "$WEIGHTS_DIR" "$OUT_DIR"',
            "",
            f"docker run --rm --ipc=host {gpu_flags} \\",
            '  -v "$SPEC_PATH:/input/spec.yaml:ro" \\',
            '  -v "$WEIGHTS_DIR:/weights" \\',
            '  -v "$OUT_DIR:/out" \\',
            "  -e HF_HUB_CACHE=/weights/hub \\",
            "  ${HF_TOKEN:+-e HF_TOKEN} \\",
            '  "$BYOG_IMAGE" \\',
            f"  byog run --spec /input/spec.yaml --job {shlex.quote(job.job_id)}",
            "",
        ]
    )
    repro_path = attempt_dir / "repro.sh"
    repro_path.write_text(script, encoding="utf-8")
    os.chmod(repro_path, 0o755)
    return repro_path


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def seal_bundle(
    attempt_dir: Path,
    result: RunResult,
    job: ResolvedJob,
    *,
    original_spec_text: str,
    image_ref: str = "<registry>/byog@sha256:<digest>",
    platform: PlatformKind = PlatformKind.NVIDIA,
) -> Path:
    """Write metadata, manifest.json, repro.sh, and pack bundle.tar.gz."""
    attempt_dir.mkdir(parents=True, exist_ok=True)

    (attempt_dir / "spec.yaml").write_text(original_spec_text, encoding="utf-8")
    resolved_payload = {
        "schema_version": 1,
        "name": job.spec_name,
        "job_id": job.job_id,
        "attempt_id": job.attempt_id,
        "serve": {
            "config": job.serve_config,
            "env": redact_env(job.serve_env),
        },
        "hardware": job.hardware.model_dump(),
        "workloads": [w.model_dump() for w in job.workloads],
        "timeout_s": job.timeout_s,
    }
    (attempt_dir / "spec.resolved.yaml").write_text(
        yaml.safe_dump(resolved_payload, sort_keys=False),
        encoding="utf-8",
    )
    (attempt_dir / "result.json").write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    write_repro_sh(attempt_dir, job, image_ref=image_ref, platform=platform)

    # Enumerate every file except manifest.json and bundle.tar.gz itself.
    entries: list[ArtifactEntry] = []
    files_to_pack: list[tuple[Path, str]] = []
    for path in sorted(attempt_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(attempt_dir).as_posix()
        if rel in ("manifest.json", "bundle.tar.gz"):
            continue
        digest, size = _sha256_file(path)
        entries.append(ArtifactEntry(path=rel, size_bytes=size, sha256=digest))
        files_to_pack.append((path, rel))

    manifest = BundleManifest(
        job_id=result.job_id,
        attempt_id=result.attempt_id,
        files=entries,
    )
    manifest_path = attempt_dir / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    bundle_path = attempt_dir / "bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        for full_path, rel in files_to_pack:
            tar.add(full_path, arcname=rel)

    return bundle_path
