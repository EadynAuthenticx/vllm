# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C2 auto-discovered catalog and C3 single-job spec resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.byog.catalog import Catalog, CatalogError
from tools.byog.schemas.spec import RunSpec
from tools.byog.spec import SpecResolutionError, load_spec_file, resolve_spec


def _make_temp_catalog(tmp_path: Path) -> Catalog:
    wdir = tmp_path / "workloads"
    sdir = tmp_path / "suites"
    wdir.mkdir()
    sdir.mkdir()

    (wdir / "http_bench.yaml").write_text(
        """
schema_version: 1
name: http_bench
kind: benchmark
execution_mode: server
lifecycle: reset_cache
entrypoint: tools.byog.schemas:RunSpec
defaults:
  concurrency: [1, 4]
""".strip()
    )
    (wdir / "direct_bench.yaml").write_text(
        """
schema_version: 1
name: direct_bench
kind: benchmark
execution_mode: direct
lifecycle: none
entrypoint: tools.byog.schemas:RunSpec
defaults:
  cases: ["prefill=128"]
""".strip()
    )
    (wdir / "mini_eval.yaml").write_text(
        """
schema_version: 1
name: mini_eval
kind: eval
execution_mode: server
lifecycle: reset_cache
entrypoint: tools.byog.schemas:RunSpec
defaults:
  repeats: 1
  concurrency: [1]
""".strip()
    )
    (sdir / "smoke.yaml").write_text(
        """
schema_version: 1
name: smoke
benchmarks:
  - type: http_bench
evals:
  - name: mini_eval
""".strip()
    )
    return Catalog.discover(catalog_dir=tmp_path)


def test_catalog_discovers_workloads_and_rejects_duplicates(tmp_path: Path) -> None:
    cat = _make_temp_catalog(tmp_path)
    assert [w.name for w in cat.list_workloads()] == [
        "direct_bench",
        "http_bench",
        "mini_eval",
    ]
    assert [s.name for s in cat.list_suites()] == ["smoke"]

    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "dup.yaml").write_text(
        (tmp_path / "workloads" / "http_bench.yaml").read_text()
    )
    with pytest.raises(CatalogError, match="Duplicate workload name"):
        Catalog.discover(catalog_dir=tmp_path, extra_workload_dirs=[extra])


def test_resolve_spec_protocol_ordering_and_defaults(tmp_path: Path) -> None:
    cat = _make_temp_catalog(tmp_path)
    spec = RunSpec(
        schema_version=1,
        name="demo",
        serve={
            "config": {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "revision": "rev123",
                "speculative-config": None,
            },
            "env": {"VLLM_LOGGING_LEVEL": "INFO"},
        },
        hardware={"gpu": ["H200"], "count": 1},
        benchmarks=[{"type": "direct_bench"}, {"type": "http_bench"}],
        evals=[{"name": "mini_eval", "repeats": 2}],
    )
    jobs = resolve_spec(spec, cat, attempt_id="att-fixed")
    assert len(jobs) == 1
    job = jobs[0]
    assert job.job_id == "default"
    assert job.attempt_id == "att-fixed"
    assert "speculative-config" not in job.serve_config
    assert job.hardware.gpu == ["h200"]
    # Protocol order: server bench -> eval -> direct bench
    assert [w.name for w in job.workloads] == [
        "http_bench",
        "mini_eval",
        "direct_bench",
    ]
    assert job.workloads[1].params["repeats"] == 2
    assert job.workloads[1].params["concurrency"] == [1]


@pytest.mark.parametrize(
    "bad_env",
    [
        {"LD_PRELOAD": "/tmp/evil.so"},
        {"PYTHONPATH": "/tmp/evil"},
        {"HF_TOKEN": "hf_secret"},
        {"https_proxy": "http://evil:8080"},
        {"CURL_CA_BUNDLE": "/tmp/ca.pem"},
        {"PATH": "/tmp/bin"},
    ],
)
def test_resolve_spec_rejects_blocked_serve_env(
    tmp_path: Path, bad_env: dict[str, str]
) -> None:
    cat = _make_temp_catalog(tmp_path)
    spec = RunSpec(
        schema_version=1,
        serve={
            "config": {"model": "m", "revision": "r"},
            "env": bad_env,
        },
        benchmarks=[{"type": "http_bench"}],
    )
    with pytest.raises(SpecResolutionError, match="security-sensitive"):
        resolve_spec(spec, cat)


def test_resolve_spec_gives_clear_errors_for_p1_features(tmp_path: Path) -> None:
    cat = _make_temp_catalog(tmp_path)
    recipe_spec = RunSpec(
        schema_version=1,
        recipe={"model": "Qwen/Qwen3.8-27B"},
        evals=[{"name": "mini_eval"}],
    )
    with pytest.raises(SpecResolutionError, match="P1.3"):
        resolve_spec(recipe_spec, cat)

    matrix_spec = RunSpec(
        schema_version=1,
        serve={"config": {"model": "m", "revision": "r"}},
        matrix={"tp": [1, 2]},
        evals=[{"name": "mini_eval"}],
    )
    with pytest.raises(SpecResolutionError, match="P1.4"):
        resolve_spec(matrix_spec, cat)


def test_load_spec_file_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: 99\nserve:\n  config: {model: m, revision: r}\n")
    with pytest.raises(SpecResolutionError, match="Unsupported schema_version"):
        load_spec_file(p)
