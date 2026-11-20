# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C0 contracts, fixtures, and FakeVLLMServer."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.executor.interfaces import FixtureAssetProvider
from tools.byog.schemas import (
    AssetTier,
    BundleManifest,
    ErrorKind,
    EvalRenderedInputRecord,
    EvalResponseRecord,
    EvalSampleRecord,
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    MeasurementRecord,
    PinnedAsset,
    ProtocolVariant,
    PublicationPolicy,
    QualityState,
    RunResult,
    RunSpec,
    UploadConflictResponse,
    UploadRequestMeta,
    UploadSuccessResponse,
    WorkloadDefinition,
    WorkloadKind,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_run_spec_validation_requires_recipe_or_serve() -> None:
    with pytest.raises(ValidationError, match="recipe.*serve"):
        RunSpec(schema_version=1)

    with pytest.raises(ValidationError):
        RunSpec(schema_version=2, serve={"config": {"model": "m"}})  # type: ignore[arg-type]

    spec = RunSpec(
        schema_version=1,
        serve={"config": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "abc"}},
        evals=[{"name": "gsm8k", "concurrency": [1, 4]}],
    )
    assert spec.serve is not None
    assert spec.evals is not None and spec.evals[0].concurrency == [1, 4]


def test_workload_definition_forbids_lifecycle_on_direct_mode() -> None:
    with pytest.raises(ValidationError, match="Direct workloads"):
        WorkloadDefinition(
            name="bad_direct",
            kind=WorkloadKind.BENCHMARK,
            execution_mode=ExecutionMode.DIRECT,
            lifecycle=LifecyclePolicy.RESET_CACHE,
            entrypoint="tools.byog.example:Runner",
        )

    ok = WorkloadDefinition(
        name="good_server",
        kind=WorkloadKind.EVAL,
        execution_mode=ExecutionMode.SERVER,
        lifecycle=LifecyclePolicy.RESET_CACHE,
        entrypoint="tools.byog.example:Runner",
    )
    assert ok.publication_policy.effective_allow_samples() is False


def test_publication_policy_blocks_unreviewed_datasets() -> None:
    unreviewed = PublicationPolicy(reviewed=False, allow_question_text=True)
    assert unreviewed.effective_allow_samples() is False
    reviewed = PublicationPolicy(reviewed=True, allow_question_text=True)
    assert reviewed.effective_allow_samples() is True


def test_result_and_eval_record_roundtrip() -> None:
    sample = EvalSampleRecord(
        dataset_id="gsm8k",
        dataset_revision="rev1",
        sample_id="0",
        question="1+1?",
        reference_answer="2",
    )
    rendered = EvalRenderedInputRecord(
        input_hash="deadbeef",
        dataset_revision="rev1",
        sample_id="0",
        rendered_input="Q: 1+1?\nA:",
    )
    response = EvalResponseRecord(
        dataset_revision="rev1",
        sample_id="0",
        input_hash="deadbeef",
        repeat=0,
        concurrency=1,
        model_response="2",
        extracted_answer=2,
        score=1.0,
        correct=True,
        error_kind=ErrorKind.NONE,
    )
    assert EvalSampleRecord.model_validate_json(sample.model_dump_json()) == sample
    assert (
        EvalRenderedInputRecord.model_validate_json(rendered.model_dump_json())
        == rendered
    )
    assert (
        EvalResponseRecord.model_validate_json(response.model_dump_json()) == response
    )

    result = RunResult(
        job_id="default",
        attempt_id="att-1",
        execution_status=ExecutionStatus.COMPLETED,
        quality_state=QualityState.PASSED,
        started_at="2026-11-20T00:00:00Z",
        finished_at="2026-11-20T00:00:10Z",
        duration_s=10.0,
        harness_version="0.1.0",
        measurements=[
            MeasurementRecord(
                workload="gsm8k",
                kind="eval",
                point_id="c1",
                execution_mode="server",
                protocol_version="1.0",
                protocol_variant=ProtocolVariant.RESET_CACHE,
                compatibility_key="compat-hash",
                metrics={"accuracy": 1.0},
                attempted=6,
                completed=6,
                scoring_denominator=6,
            )
        ],
    )
    assert RunResult.model_validate_json(result.model_dump_json()) == result
    manifest = BundleManifest(job_id="default", attempt_id="att-1", files=[])
    assert BundleManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_upload_api_fixtures_match_contracts() -> None:
    data = json.loads((FIXTURES_DIR / "upload_api_fixtures.json").read_text())
    cases = data["cases"]

    for key in ("success_created", "idempotent_retry"):
        UploadRequestMeta.model_validate(cases[key]["request"])
        resp = UploadSuccessResponse.model_validate(cases[key]["response"])
        assert resp.job_id == "default"

    UploadRequestMeta.model_validate(cases["attempt_conflict"]["request"])
    conflict = UploadConflictResponse.model_validate(
        cases["attempt_conflict"]["response"]
    )
    assert conflict.error == "attempt_conflict"


def test_fixture_asset_provider_verifies_sha256() -> None:
    test_path = FIXTURES_DIR / "gsm8k_mini_test.jsonl"
    expected_sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
    provider = FixtureAssetProvider(FIXTURES_DIR)

    asset = PinnedAsset(
        id="gsm8k_test",
        source="gsm8k_mini_test.jsonl",
        revision="mini-v1",
        sha256=expected_sha,
        tier=AssetTier.SMALL_PERMISSIVE,
    )
    resolved = provider.resolve(asset)
    assert resolved.path == test_path
    assert resolved.sha256 == expected_sha

    bad = asset.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        provider.resolve(bad)


def test_fake_vllm_server_endpoints_and_failure_injection() -> None:
    state = FakeServerState(fail_after_n=2)
    with FakeVLLMServer(state) as srv:
        with urllib.request.urlopen(f"{srv.base_url}/health") as resp:
            assert resp.status == 200

        req = urllib.request.Request(
            f"{srv.base_url}/v1/completions",
            data=json.dumps({"prompt": "Question: What is 7 + 8?\nAnswer:"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
            assert "#### 15" in body["choices"][0]["text"]

        reset_req = urllib.request.Request(
            f"{srv.base_url}/reset_prefix_cache", data=b"{}", method="POST"
        )
        with urllib.request.urlopen(reset_req) as resp:
            assert resp.status == 200
        assert state.reset_calls == 1
