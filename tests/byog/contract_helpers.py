# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable Phase 1 plugin contract test helper for BYOG contributors.

Every P1 benchmark or eval PR can import `assert_plugin_contract` to prove
end-to-end compliance on CPU in seconds: catalog auto-discovery, CLI validate,
CLI run against `FakeVLLMServer` (or direct mode without a server), shared
bundle validation, and deduplicated eval record checks.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import yaml

from tests.byog.unit.test_c4_executor import FakeLauncher
from tools.byog.bundle import validate_bundle
from tools.byog.catalog import Catalog
from tools.byog.cli import main as cli_main
from tools.byog.executor.interfaces import AssetProvider, FixtureAssetProvider
from tools.byog.schemas.catalog import ExecutionMode, WorkloadKind
from tools.byog.schemas.result import ExecutionStatus, RunResult


def assert_plugin_contract(
    *,
    workload_name: str,
    spec_dict: dict[str, Any],
    tmp_path: Path,
    catalog: Catalog | None = None,
    asset_provider: AssetProvider | None = None,
    expected_status: ExecutionStatus = ExecutionStatus.COMPLETED,
) -> tuple[RunResult, Path]:
    """Run a workload end-to-end through the real BYOG CLI and validate its bundle."""
    cat = catalog or Catalog.discover()
    defn = cat.get_workload(workload_name)

    tmp_path.mkdir(parents=True, exist_ok=True)
    spec_path = tmp_path / f"{workload_name}_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False), encoding="utf-8")

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # 1. `byog validate` must succeed.
    rc_val = cli_main(
        ["validate", "--spec", str(spec_path)],
        catalog=cat,
        stdout=stdout_buf,
        stderr=stderr_buf,
    )
    assert rc_val == 0, f"byog validate failed: {stderr_buf.getvalue()}"

    # 2. `byog run --dry-run` must succeed.
    rc_dry = cli_main(
        ["run", "--spec", str(spec_path), "--dry-run"],
        catalog=cat,
        stdout=stdout_buf,
        stderr=stderr_buf,
    )
    assert rc_dry == 0, f"byog run --dry-run failed: {stderr_buf.getvalue()}"

    # 3. `byog run` executes end-to-end via FakeLauncher.
    out_dir = tmp_path / "out"
    launcher = FakeLauncher()
    provider = asset_provider or FixtureAssetProvider(
        Path(__file__).resolve().parent / "fixtures"
    )
    rc_run = cli_main(
        ["run", "--spec", str(spec_path), "--out", str(out_dir)],
        catalog=cat,
        launcher=launcher,
        asset_provider=provider,
        stdout=stdout_buf,
        stderr=stderr_buf,
    )
    if expected_status == ExecutionStatus.COMPLETED:
        assert rc_run == 0, (
            f"byog run failed ({rc_run}):\n"
            f"stdout:\n{stdout_buf.getvalue()}\n"
            f"stderr:\n{stderr_buf.getvalue()}"
        )

    if defn.execution_mode == ExecutionMode.DIRECT:
        assert launcher.starts == 0, "Direct workload must not launch a server."
    else:
        assert launcher.starts >= 1, "Server workload must launch a server."
        assert launcher.stops == launcher.starts, "Server was not cleaned up."

    bundles = list(out_dir.rglob("bundle.tar.gz"))
    assert len(bundles) == 1, f"Expected 1 bundle.tar.gz, found {bundles}"
    bundle_path = bundles[0]

    # 4. Shared validator (reused by upload client) must accept the bundle.
    result, manifest = validate_bundle(bundle_path)
    assert result.execution_status == expected_status
    matching = [m for m in result.measurements if m.workload == workload_name]
    assert matching, f"No measurement recorded for '{workload_name}'"
    for m in matching:
        assert m.compatibility_key, "Missing per-measurement compatibility_key"

    # 5. Eval plugins must emit deduplicated samples/inputs/responses JSONL.
    if defn.kind == WorkloadKind.EVAL and expected_status == ExecutionStatus.COMPLETED:
        with tarfile.open(bundle_path, "r:gz") as tar:
            names = set(tar.getnames())
        prefix = f"evals/{workload_name}"
        for fname in ("samples.jsonl", "inputs.jsonl", "responses.jsonl"):
            assert f"{prefix}/{fname}" in names, (
                f"Missing {prefix}/{fname} in bundle: {sorted(names)}"
            )
        assert len(manifest.files) >= 7

    return result, bundle_path
