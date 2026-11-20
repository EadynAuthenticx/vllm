# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C1: BYOG command-line entrypoint (`python -m tools.byog`)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TextIO

from tools.byog.bundle import BundleValidationError, validate_bundle
from tools.byog.catalog import Catalog, CatalogError
from tools.byog.catalog.assets import CachingAssetProvider
from tools.byog.executor.coordinator import JobCoordinator
from tools.byog.executor.interfaces import (
    AssetProvider,
    ServerLauncher,
)
from tools.byog.executor.platform import NvidiaSubprocessLauncher
from tools.byog.schemas.result import ExecutionStatus, RunResult
from tools.byog.spec import SpecResolutionError, load_spec_file, resolve_spec

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VALIDATION_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byog",
        description="vLLM BYOG benchmark and accuracy evaluation harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate a BYOG spec YAML.")
    p_val.add_argument("--spec", type=Path, required=True, help="Path to spec YAML.")
    p_val.add_argument(
        "--allow-unpinned-revision",
        action="store_true",
        help="Allow specs without serve.config.revision (local dev only).",
    )

    p_run = sub.add_parser("run", help="Validate and execute a BYOG spec YAML.")
    p_run.add_argument("--spec", type=Path, required=True, help="Path to spec YAML.")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved job plan without starting servers or using GPUs.",
    )
    p_run.add_argument("--job", type=str, default=None, help="Run only this job ID.")
    p_run.add_argument(
        "--out",
        type=Path,
        default=Path("out"),
        help="Output root directory (default: ./out).",
    )
    p_run.add_argument(
        "--upload",
        action="store_true",
        help="Upload bundle after execution and cleanup (tracked in P1.14).",
    )
    p_run.add_argument(
        "--image",
        type=str,
        default="<registry>/byog@sha256:<digest>",
        help="Pinned image reference recorded in repro.sh.",
    )
    p_run.add_argument(
        "--allow-unpinned-revision",
        action="store_true",
        help="Allow specs without serve.config.revision (local dev only).",
    )

    p_upload = sub.add_parser("upload", help="Validate and upload a result bundle.")
    p_upload.add_argument("bundle", type=Path, help="Path to bundle.tar.gz.")

    p_vbundle = sub.add_parser(
        "validate-bundle", help="Validate an existing bundle.tar.gz locally."
    )
    p_vbundle.add_argument("bundle", type=Path, help="Path to bundle.tar.gz.")

    p_cat = sub.add_parser("catalog", help="Inspect registered workloads and suites.")
    cat_sub = p_cat.add_subparsers(dest="catalog_cmd", required=True)
    cat_sub.add_parser("list", help="List available benchmarks, evals, and suites.")

    return parser


def _print_result_summary(result: RunResult, bundle_path: Path, stream: TextIO) -> None:
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    size_bytes = bundle_path.stat().st_size
    print(
        f"[byog] job={result.job_id} attempt={result.attempt_id} "
        f"status={result.execution_status.value} quality={result.quality_state.value} "
        f"duration={result.duration_s:.2f}s",
        file=stream,
    )
    for m in result.measurements:
        metrics_str = ", ".join(f"{k}={v}" for k, v in sorted(m.metrics.items()))
        print(
            f"  - {m.workload}:{m.point_id} [{m.protocol_variant.value}] "
            f"status={m.status.value} {metrics_str}",
            file=stream,
        )
    print(
        f"[byog] bundle={bundle_path} size={size_bytes}B sha256={bundle_sha}",
        file=stream,
    )


def main(
    argv: list[str] | None = None,
    *,
    catalog: Catalog | None = None,
    launcher: ServerLauncher | None = None,
    asset_provider: AssetProvider | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cat = catalog or Catalog.discover()
    except CatalogError as exc:
        print(f"Catalog error: {exc}", file=err)
        return EXIT_VALIDATION_ERROR

    if args.command == "catalog" and args.catalog_cmd == "list":
        print("Workloads:", file=out)
        for w in cat.list_workloads():
            print(
                f"  - {w.name:20s} kind={w.kind.value:9s} "
                f"mode={w.execution_mode.value:6s} "
                f"lifecycle={w.lifecycle.value}",
                file=out,
            )
        print("Suites:", file=out)
        for s in cat.list_suites():
            print(f"  - {s.name:20s} {s.description}", file=out)
        return EXIT_OK

    if args.command == "validate-bundle":
        try:
            res, manifest = validate_bundle(args.bundle)
        except BundleValidationError as exc:
            print(f"Bundle validation failed: {exc}", file=err)
            return EXIT_VALIDATION_ERROR
        print(
            f"Bundle OK: job={res.job_id} attempt={res.attempt_id} "
            f"files={len(manifest.files)} status={res.execution_status.value}",
            file=out,
        )
        return EXIT_OK

    if args.command == "upload":
        try:
            validate_bundle(args.bundle)
        except BundleValidationError as exc:
            print(f"Bundle validation failed: {exc}", file=err)
            return EXIT_VALIDATION_ERROR
        print(
            "Bundle passed validation, but remote upload client is tracked in P1.14.",
            file=err,
        )
        return EXIT_RUNTIME_ERROR

    # validate / run
    try:
        spec_text = args.spec.read_text(encoding="utf-8")
        spec = load_spec_file(args.spec)
        jobs = resolve_spec(
            spec,
            cat,
            require_model_revision=not args.allow_unpinned_revision,
        )
    except (OSError, SpecResolutionError) as exc:
        print(f"Validation error: {exc}", file=err)
        return EXIT_VALIDATION_ERROR

    if args.command == "validate":
        print(
            f"Spec OK: {len(jobs)} job(s) resolved "
            f"({', '.join(j.job_id for j in jobs)})",
            file=out,
        )
        return EXIT_OK

    # command == "run"
    if args.job is not None:
        matched = [j for j in jobs if j.job_id == args.job]
        if not matched:
            known = ", ".join(j.job_id for j in jobs)
            print(
                f"Validation error: unknown --job '{args.job}'. Available: {known}",
                file=err,
            )
            return EXIT_VALIDATION_ERROR
        jobs = matched

    if args.dry_run:
        plan = [
            {
                "job_id": j.job_id,
                "attempt_id": j.attempt_id,
                "hardware": j.hardware.model_dump(),
                "workloads": [w.model_dump() for w in j.workloads],
                "resolved_config_hash": j.resolved_config_hash,
            }
            for j in jobs
        ]
        print(json.dumps(plan, indent=2), file=out)
        return EXIT_OK

    srv_launcher = launcher or NvidiaSubprocessLauncher()
    provider = asset_provider or CachingAssetProvider()
    coord = JobCoordinator(
        catalog=cat,
        launcher=srv_launcher,
        asset_provider=provider,
        image_ref=args.image,
    )

    any_failed = False
    for job in jobs:
        result, bundle_path = coord.run_job(
            job, output_root=args.out, original_spec_text=spec_text
        )
        # Always verify every sealed bundle with the shared upload-side validator.
        validate_bundle(bundle_path)
        _print_result_summary(result, bundle_path, out)
        if result.execution_status != ExecutionStatus.COMPLETED:
            any_failed = True

    if args.upload:
        print(
            "Warning: remote upload client is tracked in P1.14; bundle kept locally.",
            file=err,
        )
        return EXIT_RUNTIME_ERROR

    return EXIT_RUNTIME_ERROR if any_failed else EXIT_OK
