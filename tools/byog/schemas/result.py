# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned schemas for result.json, manifest.json, eval records, and Upload API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class QualityState(StrEnum):
    """Quality status tracked separately from ExecutionStatus."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
    INCONCLUSIVE = "inconclusive"


class EvidenceSource(StrEnum):
    HARNESS = "harness"
    CANDIDATE = "candidate"


class ProtocolVariant(StrEnum):
    """Actual server-state handling executed by the coordinator."""

    NONE = "none"
    DIRECT = "direct"
    RESET_CACHE = "reset_cache"
    RESTART_SERVER = "restart_server"


class ErrorKind(StrEnum):
    NONE = "none"
    MODEL = "model"
    INFRA = "infra"


class EvalSampleRecord(BaseModel):
    """Deduplicated source question/reference keyed by (dataset_revision, sample_id)."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_revision: str
    sample_id: str
    question: str | None = None
    reference_answer: str | None = None
    media_refs: list[str] = Field(default_factory=list)
    withheld_reason: str | None = None


class EvalRenderedInputRecord(BaseModel):
    """Exact rendered prompt/messages sent to the model, keyed by input_hash."""

    model_config = ConfigDict(extra="forbid")

    input_hash: str
    dataset_revision: str
    sample_id: str
    rendered_input: Any = None
    withheld_reason: str | None = None


class EvalResponseRecord(BaseModel):
    """Per-(sample, repeat, concurrency) model response and score."""

    model_config = ConfigDict(extra="forbid")

    dataset_revision: str
    sample_id: str
    input_hash: str
    repeat: int = Field(ge=0)
    concurrency: int = Field(ge=1)
    model_response: str | None = None
    extracted_answer: Any = None
    score: float | None = None
    correct: bool | None = None
    error_kind: ErrorKind = ErrorKind.NONE
    error_message: str | None = None
    latency_s: float | None = None
    output_tokens: int | None = None
    withheld_reason: str | None = None


class MeasurementRecord(BaseModel):
    """Per-workload / per-point measurement with its own compatibility fingerprint."""

    model_config = ConfigDict(extra="forbid")

    workload: str
    kind: Literal["benchmark", "eval"]
    point_id: str
    execution_mode: Literal["server", "direct"]
    protocol_version: str
    protocol_variant: ProtocolVariant
    evidence_source: EvidenceSource = EvidenceSource.HARNESS
    compatibility_key: str = Field(
        ...,
        description="Per-measurement fingerprint of fields affecting this metric.",
    )
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    attempted: int | None = None
    completed: int | None = None
    scoring_denominator: int | None = None
    model_errors: int = 0
    infra_errors: int = 0
    error_message: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    """Top-level result.json sealed inside bundle.tar.gz."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str
    attempt_id: str
    spec_name: str | None = None
    source_tier: Literal["ci_nightly", "ci_pr", "community"] = "community"
    execution_status: ExecutionStatus
    quality_state: QualityState = QualityState.NOT_EVALUATED
    started_at: str
    finished_at: str
    duration_s: float = Field(ge=0.0)
    harness_version: str
    vllm_commit: str | None = None
    image_digest: str | None = None
    combo_key: dict[str, Any] = Field(default_factory=dict)
    resolved_config_hash: str = ""
    hardware_detected: dict[str, Any] = Field(default_factory=dict)
    measurements: list[MeasurementRecord] = Field(default_factory=list)
    withheld_artifacts: list[dict[str, str]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")


class BundleManifest(BaseModel):
    """Top-level manifest.json listing every file inside bundle.tar.gz."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str
    attempt_id: str
    files: list[ArtifactEntry] = Field(default_factory=list)


# --- Upload HTTP API Contracts (co-reviewed with §3 Results Service) ---


class UploadRequestMeta(BaseModel):
    """Metadata sent alongside bundle.tar.gz to POST /api/v1/bundles."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str
    attempt_id: str
    bundle_sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=1)
    source_tier: Literal["ci_nightly", "ci_pr", "community"]
    ci_run_id: str | None = None


class UploadSuccessResponse(BaseModel):
    """HTTP 201 Created (new) or 200 OK (idempotent retry with identical SHA256)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["created", "duplicate_identical"]
    job_id: str
    attempt_id: str
    bundle_sha256: str
    result_url: str
    uploaded_by: str
    uploaded_at: str


class UploadConflictResponse(BaseModel):
    """HTTP 409 when attempt_id already exists with a different bundle_sha256."""

    model_config = ConfigDict(extra="forbid")

    error: Literal["attempt_conflict"] = "attempt_conflict"
    job_id: str
    attempt_id: str
    existing_bundle_sha256: str
    submitted_bundle_sha256: str
    message: str
