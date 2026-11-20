# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C7 Dockerfile.byog, lockfile freshness, and Buildkite wiring."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_requirements_lock_exists_and_is_fresh() -> None:
    lock_path = REPO_ROOT / "tools" / "byog" / "requirements.lock"
    assert lock_path.exists()
    content = lock_path.read_text(encoding="utf-8")
    assert "pydantic==" in content
    assert "pyyaml==" in content

    if shutil.which("uv") is not None:
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "byog" / "compile_requirements.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout


def test_vendor_manifest_and_dockerfile_invariants() -> None:
    manifest = json.loads(
        (REPO_ROOT / "tools" / "byog" / "vendor-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == 1
    assert isinstance(manifest["vendored"], list)

    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.byog").read_text(encoding="utf-8")
    assert "AS byog-tools" in dockerfile
    assert "COPY --from=byog-tools /opt/byog /opt/byog" in dockerfile
    assert "/opt/byog/venv" in dockerfile


def test_buildkite_byog_pipeline_declares_cpu_and_gpu_steps() -> None:
    bk = yaml.safe_load(
        (REPO_ROOT / ".buildkite" / "test_areas" / "byog.yaml").read_text(
            encoding="utf-8"
        )
    )
    keys = {step["key"] for step in bk["steps"]}
    assert {"byog-cpu-tests", "byog-nvidia-gpu-smoke"}.issubset(keys)
