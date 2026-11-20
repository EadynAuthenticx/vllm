# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C5 bundle writer, deduplicated eval records, and validator."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from tools.byog.bundle import (
    BundleValidationError,
    DeduplicatedEvalRecordWriter,
    redact_env,
    seal_bundle,
    validate_bundle,
)
from tools.byog.schemas.catalog import PlatformKind, PublicationPolicy
from tools.byog.schemas.result import (
    ExecutionStatus,
    QualityState,
    RunResult,
)
from tools.byog.schemas.spec import HardwareSpec, ResolvedJob


def _sample_job_and_result() -> tuple[ResolvedJob, RunResult]:
    job = ResolvedJob(
        job_id="default",
        attempt_id="20261120T000000Z-abcdef1234",
        spec_name="unit-demo",
        serve_config={"model": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "abc"},
        serve_env={"VLLM_LOGGING_LEVEL": "INFO", "MY_SECRET_TOKEN": "supersecret"},
        hardware=HardwareSpec(gpu=["h200"], count=1, nodes=1),
    )
    result = RunResult(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        spec_name=job.spec_name,
        execution_status=ExecutionStatus.COMPLETED,
        quality_state=QualityState.PASSED,
        started_at="2026-11-20T00:00:00Z",
        finished_at="2026-11-20T00:00:05Z",
        duration_s=5.0,
        harness_version="0.1.0",
    )
    return job, result


def test_deduplicated_eval_writer_and_policy_enforcement(tmp_path: Path) -> None:
    unreviewed_dir = tmp_path / "unreviewed"
    writer = DeduplicatedEvalRecordWriter(
        unreviewed_dir, PublicationPolicy(reviewed=False, allow_question_text=True)
    )
    for concurrency in (1, 4):
        for repeat in (0, 1):
            writer.write_response(
                dataset_id="gsm8k",
                dataset_revision="v1",
                sample_id="q0",
                question="1+1?",
                reference_answer="2",
                rendered_input="Question: 1+1?\nAnswer:",
                repeat=repeat,
                concurrency=concurrency,
                model_response="#### 2",
                extracted_answer=2,
                score=1.0,
                correct=True,
            )
    writer.close()

    samples = [
        json.loads(x)
        for x in (unreviewed_dir / "samples.jsonl").read_text().splitlines()
    ]
    inputs = [
        json.loads(x)
        for x in (unreviewed_dir / "inputs.jsonl").read_text().splitlines()
    ]
    responses = [
        json.loads(x)
        for x in (unreviewed_dir / "responses.jsonl").read_text().splitlines()
    ]
    assert len(samples) == 1
    assert len(inputs) == 1
    assert len(responses) == 4
    # Unreviewed dataset withholds question, reference, rendered_input, model_response.
    assert samples[0]["question"] is None
    assert samples[0]["reference_answer"] is None
    assert inputs[0]["rendered_input"] is None
    assert responses[0]["model_response"] is None
    assert responses[0]["correct"] is True

    reviewed_dir = tmp_path / "reviewed"
    writer2 = DeduplicatedEvalRecordWriter(
        reviewed_dir,
        PublicationPolicy(
            reviewed=True,
            allow_question_text=True,
            allow_reference_answer=True,
            allow_model_response=True,
        ),
    )
    writer2.write_response(
        dataset_id="gsm8k",
        dataset_revision="v1",
        sample_id="q0",
        question="1+1?",
        reference_answer="2",
        rendered_input="Question: 1+1?\nAnswer:",
        repeat=0,
        concurrency=1,
        model_response="#### 2",
        extracted_answer=2,
        score=1.0,
        correct=True,
    )
    writer2.close()
    s2 = json.loads((reviewed_dir / "samples.jsonl").read_text().splitlines()[0])
    r2 = json.loads((reviewed_dir / "responses.jsonl").read_text().splitlines()[0])
    assert s2["question"] == "1+1?"
    assert r2["model_response"] == "#### 2"


def test_seal_and_validate_bundle_detects_tampering(tmp_path: Path) -> None:
    job, result = _sample_job_and_result()
    attempt_dir = tmp_path / "attempt"
    bundle = seal_bundle(
        attempt_dir,
        result,
        job,
        original_spec_text="schema_version: 1\nserve:\n  config: {model: m}\n",
        platform=PlatformKind.ROCM,
    )
    parsed_result, manifest = validate_bundle(bundle)
    assert parsed_result.attempt_id == job.attempt_id
    assert any(f.path == "repro.sh" for f in manifest.files)
    assert "--device=/dev/kfd" in (attempt_dir / "repro.sh").read_text()
    assert "supersecret" not in (attempt_dir / "spec.resolved.yaml").read_text()

    # Tamper with result.json inside a copied archive -> validator must reject.
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(bundle, "r:gz") as src, tarfile.open(tampered, "w:gz") as dst:
        for m in src.getmembers():
            fh = src.extractfile(m) if m.isfile() else None
            data = fh.read() if fh is not None else b""
            if m.name == "repro.sh":
                data += b"# tampered\n"
                m.size = len(data)
            import io

            dst.addfile(m, io.BytesIO(data))

    with pytest.raises(BundleValidationError, match="Size mismatch|SHA256 mismatch"):
        validate_bundle(tampered)


def test_redact_env_strips_tokens() -> None:
    out = redact_env({"HF_TOKEN": "abc", "VLLM_LOGGING_LEVEL": "INFO"})
    assert out["HF_TOKEN"] == "<REDACTED>"
    assert out["VLLM_LOGGING_LEVEL"] == "INFO"
