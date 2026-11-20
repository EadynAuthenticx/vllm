# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 exit tests: proves all three workload paths end-to-end on CPU."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import yaml

from tests.byog.contract_helpers import assert_plugin_contract
from tests.byog.unit.test_c4_executor import FakeLauncher
from tools.byog.bundle import validate_bundle
from tools.byog.catalog import Catalog
from tools.byog.catalog.assets import CachingAssetProvider
from tools.byog.cli import main as cli_main
from tools.byog.schemas.result import ExecutionStatus, ProtocolVariant, QualityState

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _mini_gsm8k_provider(tmp_path: Path) -> CachingAssetProvider:
    return CachingAssetProvider(
        cache_dir=tmp_path / "cache",
        local_overrides={
            "gsm8k_train": FIXTURES_DIR / "gsm8k_mini_train.jsonl",
            "gsm8k_test": FIXTURES_DIR / "gsm8k_mini_test.jsonl",
        },
    )


def test_phase0_contract_all_three_reference_plugins(tmp_path: Path) -> None:
    cat = Catalog.discover()
    names = {w.name for w in cat.list_workloads()}
    assert {"gsm8k", "http_smoke", "direct_smoke"}.issubset(names)

    # 1. HTTP serving benchmark
    res_http, _ = assert_plugin_contract(
        workload_name="http_smoke",
        spec_dict={
            "schema_version": 1,
            "serve": {"config": {"model": "fake/model", "revision": "abc123"}},
            "benchmarks": [
                {"type": "http_smoke", "concurrency": [1, 2], "num_requests": 4}
            ],
        },
        tmp_path=tmp_path / "http_smoke",
        catalog=cat,
    )
    assert len(res_http.measurements) == 2
    assert res_http.measurements[0].metrics["p99_latency_s"] is None

    # 2. Direct workload (asserts 0 servers started)
    res_direct, _ = assert_plugin_contract(
        workload_name="direct_smoke",
        spec_dict={
            "schema_version": 1,
            "serve": {"config": {"model": "fake/model", "revision": "abc123"}},
            "benchmarks": [{"type": "direct_smoke", "cases": ["prefill=128"]}],
        },
        tmp_path=tmp_path / "direct_smoke",
        catalog=cat,
    )
    assert res_direct.measurements[0].protocol_variant == ProtocolVariant.DIRECT

    # 3. Real GSM8K eval plugin (offline via mini fixture AssetProvider)
    res_gsm, bundle_gsm = assert_plugin_contract(
        workload_name="gsm8k",
        spec_dict={
            "schema_version": 1,
            "serve": {"config": {"model": "fake/model", "revision": "abc123"}},
            "evals": [
                {"name": "gsm8k", "num_questions": 6, "concurrency": [2], "repeats": 2}
            ],
        },
        tmp_path=tmp_path / "gsm8k",
        catalog=cat,
        asset_provider=_mini_gsm8k_provider(tmp_path),
    )
    assert res_gsm.quality_state == QualityState.PASSED
    assert res_gsm.measurements[0].metrics["accuracy"] == 1.0
    assert res_gsm.measurements[0].scoring_denominator == 12

    # Verify deduplication: 6 questions x 2 repeats -> 6 samples, 6 inputs, 12 responses
    with tarfile.open(bundle_gsm, "r:gz") as tar:
        samples_fh = tar.extractfile("evals/gsm8k/samples.jsonl")
        inputs_fh = tar.extractfile("evals/gsm8k/inputs.jsonl")
        responses_fh = tar.extractfile("evals/gsm8k/responses.jsonl")
        assert samples_fh and inputs_fh and responses_fh
        samples = [json.loads(x) for x in samples_fh.read().decode().splitlines()]
        inputs = [json.loads(x) for x in inputs_fh.read().decode().splitlines()]
        responses = [json.loads(x) for x in responses_fh.read().decode().splitlines()]
    assert len(samples) == 6
    assert len(inputs) == 6
    assert len(responses) == 12
    # Unreviewed publication policy withholds sample text by default.
    assert all(s["question"] is None for s in samples)
    assert all(r["model_response"] is None for r in responses)


def test_phase0_combined_server_eval_direct_e2e_flow(tmp_path: Path) -> None:
    """Single `byog run` executes http_smoke -> reset -> gsm8k -> stop -> direct."""
    cat = Catalog.discover()
    spec_path = tmp_path / "combined.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "phase0-combined",
                "serve": {"config": {"model": "fake/model", "revision": "deadbeef"}},
                "hardware": {"gpu": ["h200"], "count": 1, "nodes": 1},
                "benchmarks": [
                    {"type": "direct_smoke", "cases": ["smoke=1"]},
                    {"type": "http_smoke", "concurrency": [1], "num_requests": 4},
                ],
                "evals": [{"name": "gsm8k", "num_questions": 4, "concurrency": [2]}],
            },
            sort_keys=False,
        )
    )
    launcher = FakeLauncher(allow_reset_cache=True)
    out_dir = tmp_path / "out"
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = cli_main(
        ["run", "--spec", str(spec_path), "--out", str(out_dir)],
        catalog=cat,
        launcher=launcher,
        asset_provider=_mini_gsm8k_provider(tmp_path),
        stdout=stdout,
        stderr=stderr,
    )
    assert rc == 0, stderr.getvalue()
    assert launcher.starts == 1
    assert launcher.resets == 1
    assert launcher.stops == 1

    bundles = list(out_dir.rglob("bundle.tar.gz"))
    assert len(bundles) == 1
    result, _ = validate_bundle(bundles[0])
    assert result.execution_status == ExecutionStatus.COMPLETED
    assert result.quality_state == QualityState.PASSED
    assert [m.workload for m in result.measurements] == [
        "http_smoke",
        "gsm8k",
        "direct_smoke",
    ]
