# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4: Job coordinator owning declarative server lifecycle and bundle sealing."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.byog import __version__ as HARNESS_VERSION
from tools.byog.bundle import DeduplicatedEvalRecordWriter, seal_bundle
from tools.byog.catalog import Catalog
from tools.byog.executor.interfaces import (
    AssetProvider,
    BenchmarkRunner,
    DirectRunContext,
    EvalRunner,
    ServerHandle,
    ServerLauncher,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.executor.platform import CacheResetUnsupportedError
from tools.byog.schemas.catalog import (
    ExecutionMode,
    LifecyclePolicy,
    PlatformKind,
    WorkloadDefinition,
)
from tools.byog.schemas.result import (
    ExecutionStatus,
    MeasurementRecord,
    ProtocolVariant,
    QualityState,
    RunResult,
)
from tools.byog.schemas.spec import ResolvedJob, ResolvedWorkload
from tools.byog.spec import compute_config_hash


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_measurement_compatibility_key(
    *,
    job: ResolvedJob,
    workload: ResolvedWorkload,
    defn: WorkloadDefinition,
    point: WorkloadPoint,
    protocol_variant: ProtocolVariant,
) -> str:
    """Per-measurement compatibility key excluding unrelated workloads.

    Adding an unrelated eval or benchmark to a run does not change this key.
    """
    payload: dict[str, Any] = {
        "serve_config": job.serve_config,
        "serve_env": job.serve_env,
        "hardware": job.hardware.model_dump(),
        "workload": workload.name,
        "kind": workload.kind,
        "execution_mode": workload.execution_mode,
        "protocol_version": workload.protocol_version,
        "protocol_variant": protocol_variant.value,
        "workload_params": workload.params,
        "point_id": point.point_id,
        "point_params": point.params,
        "assets": {
            k: {"id": a.id, "revision": a.revision, "sha256": a.sha256}
            for k, a in sorted(defn.assets.items())
        },
    }
    return compute_config_hash(payload)


class JobCoordinator:
    """Executes a ResolvedJob in protocol order and seals a validated bundle."""

    def __init__(
        self,
        *,
        catalog: Catalog,
        launcher: ServerLauncher,
        asset_provider: AssetProvider,
        platform: PlatformKind = PlatformKind.NVIDIA,
        image_ref: str = "<registry>/byog@sha256:<digest>",
    ):
        self.catalog = catalog
        self.launcher = launcher
        self.asset_provider = asset_provider
        self.platform = platform
        self.image_ref = image_ref

    def run_job(
        self,
        job: ResolvedJob,
        *,
        output_root: Path,
        original_spec_text: str,
    ) -> tuple[RunResult, Path]:
        attempt_dir = output_root / job.job_id / job.attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        env_dir = attempt_dir / "env"
        env_dir.mkdir(parents=True, exist_ok=True)

        started_iso = _utc_now_iso()
        start_mono = time.monotonic()

        hw_before = self.launcher.detect_hardware()
        (env_dir / "before.json").write_text(
            json.dumps(hw_before, indent=2) + "\n", encoding="utf-8"
        )

        measurements: list[MeasurementRecord] = []
        errors: list[str] = []
        withheld_artifacts: list[dict[str, str]] = []
        exec_status = ExecutionStatus.COMPLETED
        handle: ServerHandle | None = None
        server_counter = 0

        def _ensure_server() -> ServerHandle:
            nonlocal handle, server_counter
            if handle is None:
                server_counter += 1
                srv_dir = attempt_dir / "server" / f"instance-{server_counter}"
                handle = self.launcher.start(
                    job.serve_config,
                    job.serve_env,
                    srv_dir,
                    timeout_s=job.timeout_s,
                )
            return handle

        def _stop_server() -> None:
            nonlocal handle
            if handle is not None:
                cleanup = self.launcher.stop(handle)
                (handle.log_path.parent / "cleanup.json").write_text(
                    json.dumps(cleanup, indent=2) + "\n", encoding="utf-8"
                )
                handle = None

        def _prepare_point(
            lifecycle: LifecyclePolicy, *, is_first_server_point: bool
        ) -> tuple[ServerHandle, ProtocolVariant]:
            nonlocal handle
            if lifecycle == LifecyclePolicy.FRESH_SERVER_PER_POINT:
                _stop_server()
                return _ensure_server(), ProtocolVariant.RESTART_SERVER
            if lifecycle == LifecyclePolicy.RESET_CACHE:
                if handle is None:
                    return _ensure_server(), ProtocolVariant.RESET_CACHE
                if not is_first_server_point:
                    try:
                        variant = self.launcher.reset_cache(handle)
                        return handle, variant
                    except CacheResetUnsupportedError:
                        _stop_server()
                        return _ensure_server(), ProtocolVariant.RESTART_SERVER
                return handle, ProtocolVariant.RESET_CACHE
            return _ensure_server(), ProtocolVariant.NONE

        first_server_point = True
        try:
            for workload in job.workloads:
                defn = self.catalog.get_workload(workload.name)
                plugin = self.catalog.instantiate_plugin(workload.name)
                mode = ExecutionMode(workload.execution_mode)
                lifecycle = LifecyclePolicy(workload.lifecycle)

                if mode == ExecutionMode.DIRECT:
                    # Always stop any running server before direct GPU workloads.
                    _stop_server()

                points: list[WorkloadPoint] = plugin.points(workload.params)
                record_writer: DeduplicatedEvalRecordWriter | None = None
                if workload.kind == "eval":
                    eval_dir = attempt_dir / "evals" / workload.name
                    record_writer = DeduplicatedEvalRecordWriter(
                        eval_dir, defn.publication_policy
                    )

                try:
                    for point in points:
                        if time.monotonic() - start_mono > job.timeout_s:
                            exec_status = ExecutionStatus.TIMED_OUT
                            errors.append(
                                f"Job exceeded timeout of {job.timeout_s}s "
                                f"before {workload.name}:{point.point_id}."
                            )
                            break

                        point_dir = (
                            attempt_dir
                            / ("evals" if workload.kind == "eval" else "bench")
                            / workload.name
                            / point.point_id
                        )
                        point_dir.mkdir(parents=True, exist_ok=True)

                        if mode == ExecutionMode.DIRECT:
                            variant = ProtocolVariant.DIRECT
                            dctx = DirectRunContext(
                                serve_config=job.serve_config,
                                serve_env=job.serve_env,
                                assets=self.asset_provider,
                                workload_assets=defn.assets,
                                output_dir=point_dir,
                                timeout_s=job.timeout_s,
                            )
                            assert isinstance(plugin, BenchmarkRunner)
                            pt_res = plugin.run_point(point, dctx)
                        else:
                            srv_handle, variant = _prepare_point(
                                lifecycle, is_first_server_point=first_server_point
                            )
                            first_server_point = False
                            sctx = ServerRunContext(
                                base_url=srv_handle.base_url,
                                model=str(job.serve_config.get("model", "")),
                                assets=self.asset_provider,
                                workload_assets=defn.assets,
                                output_dir=point_dir,
                                timeout_s=job.timeout_s,
                            )
                            if workload.kind == "eval":
                                assert isinstance(plugin, EvalRunner)
                                assert record_writer is not None
                                pt_res = plugin.run_point(point, sctx, record_writer)
                            else:
                                assert isinstance(plugin, BenchmarkRunner)
                                pt_res = plugin.run_point(point, sctx)

                        rel_artifacts = [
                            p.relative_to(attempt_dir).as_posix()
                            for p in pt_res.artifacts
                            if p.exists()
                        ]
                        compat_key = compute_measurement_compatibility_key(
                            job=job,
                            workload=workload,
                            defn=defn,
                            point=point,
                            protocol_variant=variant,
                        )
                        measurements.append(
                            MeasurementRecord(
                                workload=workload.name,
                                kind=workload.kind,
                                point_id=point.point_id,
                                execution_mode=workload.execution_mode,
                                protocol_version=workload.protocol_version,
                                protocol_variant=variant,
                                evidence_source=pt_res.evidence_source,
                                compatibility_key=compat_key,
                                status=pt_res.status,
                                metrics=pt_res.metrics,
                                attempted=pt_res.attempted,
                                completed=pt_res.completed,
                                scoring_denominator=pt_res.scoring_denominator,
                                model_errors=pt_res.model_errors,
                                infra_errors=pt_res.infra_errors,
                                error_message=pt_res.error_message,
                                artifacts=rel_artifacts,
                            )
                        )
                        if pt_res.status != ExecutionStatus.COMPLETED:
                            exec_status = pt_res.status
                            if pt_res.error_message:
                                errors.append(
                                    f"{workload.name}:{point.point_id}: "
                                    f"{pt_res.error_message}"
                                )
                finally:
                    if record_writer is not None:
                        record_writer.close()
                        if record_writer.withheld_fields:
                            withheld_artifacts.append(
                                {
                                    "workload": workload.name,
                                    "fields": ",".join(
                                        sorted(record_writer.withheld_fields)
                                    ),
                                    "reason": (
                                        "catalog publication policy"
                                        if defn.publication_policy.reviewed
                                        else "dataset publication policy unreviewed"
                                    ),
                                }
                            )

                if exec_status == ExecutionStatus.TIMED_OUT:
                    break
        except KeyboardInterrupt:
            exec_status = ExecutionStatus.CANCELLED
            errors.append("Run cancelled by user interrupt.")
        except Exception as exc:
            exec_status = ExecutionStatus.FAILED
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            _stop_server()
            hw_after = self.launcher.detect_hardware()
            (env_dir / "after.json").write_text(
                json.dumps(hw_after, indent=2) + "\n", encoding="utf-8"
            )

        quality_state = self._compute_quality_state(measurements)
        finished_iso = _utc_now_iso()
        duration_s = round(max(0.0, time.monotonic() - start_mono), 3)

        result = RunResult(
            job_id=job.job_id,
            attempt_id=job.attempt_id,
            spec_name=job.spec_name,
            source_tier=job.source_tier,
            execution_status=exec_status,
            quality_state=quality_state,
            started_at=started_iso,
            finished_at=finished_iso,
            duration_s=duration_s,
            harness_version=HARNESS_VERSION,
            combo_key={
                "model": job.serve_config.get("model"),
                "hardware_sku": (
                    job.hardware.gpu[0] if job.hardware.gpu else "unknown"
                ),
                "gpu_count": job.hardware.count,
                "serving_profile": job.spec_name or "adhoc",
            },
            resolved_config_hash=job.resolved_config_hash,
            hardware_detected=hw_before,
            measurements=measurements,
            withheld_artifacts=withheld_artifacts,
            errors=errors,
        )

        bundle_path = seal_bundle(
            attempt_dir,
            result,
            job,
            original_spec_text=original_spec_text,
            image_ref=self.image_ref,
            platform=self.platform,
        )
        return result, bundle_path

    def _compute_quality_state(
        self, measurements: list[MeasurementRecord]
    ) -> QualityState:
        eval_records = [m for m in measurements if m.kind == "eval"]
        if not eval_records:
            return QualityState.NOT_EVALUATED
        if any(
            m.status != ExecutionStatus.COMPLETED or m.infra_errors > 0
            for m in eval_records
        ):
            return QualityState.INCONCLUSIVE
        if any(
            m.scoring_denominator is None or m.scoring_denominator <= 0
            for m in eval_records
        ):
            return QualityState.INCONCLUSIVE
        return QualityState.PASSED
