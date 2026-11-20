# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C4 JobCoordinator lifecycle and NvidiaSubprocessLauncher."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.bundle import validate_bundle
from tools.byog.catalog import Catalog
from tools.byog.executor.coordinator import JobCoordinator
from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    EvalRecordWriter,
    EvalRunner,
    FixtureAssetProvider,
    PointResult,
    ServerHandle,
    ServerLauncher,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.executor.platform import (
    CacheResetUnsupportedError,
    NvidiaSubprocessLauncher,
    canonicalize_gpu_name,
)
from tools.byog.schemas.catalog import (
    ExecutionMode,
    LifecyclePolicy,
    PublicationPolicy,
    WorkloadDefinition,
    WorkloadKind,
)
from tools.byog.schemas.result import (
    ExecutionStatus,
    ProtocolVariant,
    QualityState,
)
from tools.byog.schemas.spec import RunSpec
from tools.byog.spec import resolve_spec


class DummyHttpBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [
            WorkloadPoint(point_id=f"c{c}", params={"concurrency": c})
            for c in params.get("concurrency", [1, 4])
        ]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        assert isinstance(ctx, ServerRunContext)
        req = urllib.request.Request(
            f"{ctx.base_url}/v1/completions",
            data=json.dumps({"prompt": "What is 1 + 1?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
        out = ctx.output_dir / "summary.json"
        out.write_text(json.dumps({"throughput_tps": 100.0}))
        return PointResult(
            metrics={"throughput_tps": 100.0},
            attempted=1,
            completed=1,
            artifacts=[out],
        )


class DummyDirectBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint(point_id="prefill_128", params={"case": "prefill=128"})]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        assert isinstance(ctx, DirectRunContext)
        out = ctx.output_dir / "forward.json"
        out.write_text(json.dumps({"latency_ms": 1.23}))
        return PointResult(metrics={"latency_ms": 1.23}, artifacts=[out])


class DummyFailingBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint(point_id="c1")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        raise RuntimeError("boom during benchmark")


class DummyEval(EvalRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint(point_id="c1", params={"concurrency": 1})]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult:
        record_writer.write_response(
            dataset_id="tiny",
            dataset_revision="v1",
            sample_id="s0",
            question="1+1?",
            reference_answer="2",
            rendered_input="Q: 1+1?\nA:",
            repeat=0,
            concurrency=1,
            model_response="#### 2",
            extracted_answer=2,
            score=1.0,
            correct=True,
        )
        return PointResult(
            metrics={"accuracy": 1.0},
            attempted=1,
            completed=1,
            scoring_denominator=1,
        )


class FakeLauncher(ServerLauncher):
    """Test ServerLauncher backed by FakeVLLMServer with lifecycle call tracking."""

    def __init__(self, *, allow_reset_cache: bool = True):
        self.allow_reset_cache = allow_reset_cache
        self.starts = 0
        self.stops = 0
        self.resets = 0
        self.active = False
        self._servers: list[FakeVLLMServer] = []

    def detect_hardware(self) -> dict[str, Any]:
        return {"platform": "nvidia", "gpu_count": 1, "canonical_skus": ["h200"]}

    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
    ) -> ServerHandle:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.starts += 1
        self.active = True
        srv = FakeVLLMServer(
            FakeServerState(
                model_name=str(serve_config.get("model", "m")),
                allow_reset_cache=self.allow_reset_cache,
            )
        ).start()
        self._servers.append(srv)
        log_path = log_dir / "server.log"
        log_path.write_text("fake server started\n")
        return ServerHandle(base_url=srv.base_url, command=["fake"], log_path=log_path)

    def reset_cache(self, handle: ServerHandle) -> ProtocolVariant:
        self.resets += 1
        if not self.allow_reset_cache:
            raise CacheResetUnsupportedError("dev mode off")
        req = urllib.request.Request(
            f"{handle.base_url}/reset_prefix_cache", data=b"{}", method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
        return ProtocolVariant.RESET_CACHE

    def stop(self, handle: ServerHandle) -> dict[str, Any]:
        self.stops += 1
        self.active = False
        if self._servers:
            self._servers.pop().stop()
        return {"terminated_pgid": True, "vram_reclaimed": True}


def _test_catalog() -> Catalog:
    mod = __name__
    workloads = {
        "http_reset": WorkloadDefinition(
            name="http_reset",
            kind=WorkloadKind.BENCHMARK,
            execution_mode=ExecutionMode.SERVER,
            lifecycle=LifecyclePolicy.RESET_CACHE,
            entrypoint=f"{mod}:DummyHttpBench",
            defaults={"concurrency": [1, 4]},
        ),
        "http_fresh": WorkloadDefinition(
            name="http_fresh",
            kind=WorkloadKind.BENCHMARK,
            execution_mode=ExecutionMode.SERVER,
            lifecycle=LifecyclePolicy.FRESH_SERVER_PER_POINT,
            entrypoint=f"{mod}:DummyHttpBench",
            defaults={"concurrency": [1, 4]},
        ),
        "direct_only": WorkloadDefinition(
            name="direct_only",
            kind=WorkloadKind.BENCHMARK,
            execution_mode=ExecutionMode.DIRECT,
            lifecycle=LifecyclePolicy.NONE,
            entrypoint=f"{mod}:DummyDirectBench",
        ),
        "failing_bench": WorkloadDefinition(
            name="failing_bench",
            kind=WorkloadKind.BENCHMARK,
            execution_mode=ExecutionMode.SERVER,
            lifecycle=LifecyclePolicy.NONE,
            entrypoint=f"{mod}:DummyFailingBench",
        ),
        "tiny_eval": WorkloadDefinition(
            name="tiny_eval",
            kind=WorkloadKind.EVAL,
            execution_mode=ExecutionMode.SERVER,
            lifecycle=LifecyclePolicy.RESET_CACHE,
            entrypoint=f"{mod}:DummyEval",
            publication_policy=PublicationPolicy(reviewed=False),
        ),
    }
    return Catalog(workloads=workloads, suites={})


def test_coordinator_resets_cache_and_stops_before_direct(tmp_path: Path) -> None:
    cat = _test_catalog()
    launcher = FakeLauncher(allow_reset_cache=True)
    spec = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r1"}},
        benchmarks=[{"type": "http_reset"}, {"type": "direct_only"}],
    )
    job = resolve_spec(spec, cat, attempt_id="att-1")[0]
    coord = JobCoordinator(
        catalog=cat, launcher=launcher, asset_provider=FixtureAssetProvider(tmp_path)
    )
    result, bundle = coord.run_job(
        job, output_root=tmp_path / "out", original_spec_text="schema_version: 1\n"
    )
    assert result.execution_status == ExecutionStatus.COMPLETED
    assert result.quality_state == QualityState.NOT_EVALUATED
    assert launcher.starts == 1
    assert launcher.resets == 1
    assert launcher.stops == 1
    assert [m.protocol_variant for m in result.measurements] == [
        ProtocolVariant.RESET_CACHE,
        ProtocolVariant.RESET_CACHE,
        ProtocolVariant.DIRECT,
    ]
    validate_bundle(bundle)


def test_coordinator_records_restart_when_cache_reset_unsupported(
    tmp_path: Path,
) -> None:
    cat = _test_catalog()
    launcher = FakeLauncher(allow_reset_cache=False)
    spec = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r1"}},
        benchmarks=[{"type": "http_reset"}],
    )
    job = resolve_spec(spec, cat, attempt_id="att-2")[0]
    coord = JobCoordinator(
        catalog=cat, launcher=launcher, asset_provider=FixtureAssetProvider(tmp_path)
    )
    result, bundle = coord.run_job(
        job, output_root=tmp_path / "out", original_spec_text="schema_version: 1\n"
    )
    assert launcher.starts == 2
    assert result.measurements[1].protocol_variant == ProtocolVariant.RESTART_SERVER
    validate_bundle(bundle)


def test_coordinator_fresh_server_per_point_and_direct_no_server(
    tmp_path: Path,
) -> None:
    cat = _test_catalog()
    fresh_launcher = FakeLauncher()
    spec_fresh = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r1"}},
        benchmarks=[{"type": "http_fresh"}],
    )
    job = resolve_spec(spec_fresh, cat, attempt_id="att-fresh")[0]
    JobCoordinator(
        catalog=cat,
        launcher=fresh_launcher,
        asset_provider=FixtureAssetProvider(tmp_path),
    ).run_job(job, output_root=tmp_path / "out", original_spec_text="schema_version: 1")
    assert fresh_launcher.starts == 2
    assert fresh_launcher.stops == 2

    direct_launcher = FakeLauncher()
    spec_direct = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r1"}},
        benchmarks=[{"type": "direct_only"}],
    )
    job2 = resolve_spec(spec_direct, cat, attempt_id="att-direct")[0]
    res2, bundle2 = JobCoordinator(
        catalog=cat,
        launcher=direct_launcher,
        asset_provider=FixtureAssetProvider(tmp_path),
    ).run_job(
        job2, output_root=tmp_path / "out", original_spec_text="schema_version: 1"
    )
    assert direct_launcher.starts == 0
    assert res2.execution_status == ExecutionStatus.COMPLETED
    validate_bundle(bundle2)


def test_coordinator_preserves_partial_bundle_on_failure_and_stable_compat_key(
    tmp_path: Path,
) -> None:
    cat = _test_catalog()
    # Adding an unrelated eval must NOT change http_reset's compatibility_key.
    spec_a = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r1"}},
        benchmarks=[{"type": "http_reset"}],
    )
    spec_b = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r1"}},
        benchmarks=[{"type": "http_reset"}],
        evals=[{"name": "tiny_eval"}],
    )
    res_a, _ = JobCoordinator(
        catalog=cat,
        launcher=FakeLauncher(),
        asset_provider=FixtureAssetProvider(tmp_path),
    ).run_job(
        resolve_spec(spec_a, cat, attempt_id="a")[0],
        output_root=tmp_path / "out",
        original_spec_text="a",
    )
    res_b, bundle_b = JobCoordinator(
        catalog=cat,
        launcher=FakeLauncher(),
        asset_provider=FixtureAssetProvider(tmp_path),
    ).run_job(
        resolve_spec(spec_b, cat, attempt_id="b")[0],
        output_root=tmp_path / "out",
        original_spec_text="b",
    )
    assert (
        res_a.measurements[0].compatibility_key
        == res_b.measurements[0].compatibility_key
    )
    assert res_b.quality_state == QualityState.PASSED
    assert len(res_b.withheld_artifacts) == 1
    validate_bundle(bundle_b)

    # Failure seals a valid bundle and cleans up the server.
    fail_launcher = FakeLauncher()
    fail_spec = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r1"}},
        benchmarks=[{"type": "failing_bench"}],
    )
    res_f, bundle_f = JobCoordinator(
        catalog=cat,
        launcher=fail_launcher,
        asset_provider=FixtureAssetProvider(tmp_path),
    ).run_job(
        resolve_spec(fail_spec, cat, attempt_id="f")[0],
        output_root=tmp_path / "out",
        original_spec_text="f",
    )
    assert res_f.execution_status == ExecutionStatus.FAILED
    assert fail_launcher.stops == 1
    assert any("boom" in e for e in res_f.errors)
    validate_bundle(bundle_f)


def test_nvidia_launcher_strips_hf_token_and_normalizes_gpu_names() -> None:
    env = NvidiaSubprocessLauncher.build_subprocess_env(
        {"HF_TOKEN": "secret", "HOME": "/root"},
        {"VLLM_LOGGING_LEVEL": "INFO"},
    )
    assert "HF_TOKEN" not in env
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    assert env["VLLM_LOGGING_LEVEL"] == "INFO"

    with pytest.raises(ValueError, match="conflicts"):
        NvidiaSubprocessLauncher.build_subprocess_env({}, {"VLLM_SERVER_DEV_MODE": "0"})

    assert canonicalize_gpu_name("NVIDIA H200") == "h200"
    assert canonicalize_gpu_name("NVIDIA B200 SXM") == "b200"
    assert canonicalize_gpu_name("NVIDIA GB300") == "gb300"
