# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned schemas for auto-discovered BYOG catalog workloads and suites."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkloadKind(StrEnum):
    BENCHMARK = "benchmark"
    EVAL = "eval"


class ExecutionMode(StrEnum):
    SERVER = "server"
    DIRECT = "direct"


class LifecyclePolicy(StrEnum):
    """Declarative server-state isolation between workload points.

    Owned and enforced by the coordinator, never by individual plugins.
    """

    NONE = "none"
    RESET_CACHE = "reset_cache"
    FRESH_SERVER_PER_POINT = "fresh_server_per_point"


class PlatformKind(StrEnum):
    NVIDIA = "nvidia"
    ROCM = "rocm"


class AssetTier(StrEnum):
    SMALL_PERMISSIVE = "small_permissive"
    LARGE_PERMISSIVE = "large_permissive"
    GATED = "gated"


class PinnedAsset(BaseModel):
    """Pinned dataset or trace reference owned by the trusted catalog."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    source: str = Field(
        ...,
        min_length=1,
        description="Upstream URI, HF repo ID, or bundled fixture path.",
    )
    revision: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=1)
    tier: AssetTier = AssetTier.SMALL_PERMISSIVE
    mirror_url: str | None = None


class PublicationPolicy(BaseModel):
    """Trusted catalog policy controlling what eval content can be published."""

    model_config = ConfigDict(extra="forbid")

    reviewed: bool = False
    allow_question_text: bool = False
    allow_reference_answer: bool = False
    allow_model_response: bool = False
    note: str | None = None

    def effective_allow_samples(self) -> bool:
        """Unreviewed datasets withhold sample text, references, and responses."""
        return self.reviewed and (
            self.allow_question_text
            or self.allow_reference_answer
            or self.allow_model_response
        )


class WorkloadDefinition(BaseModel):
    """Schema for a single auto-discovered catalog/workloads/<name>.yaml file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: WorkloadKind
    description: str = ""
    execution_mode: ExecutionMode
    lifecycle: LifecyclePolicy = LifecyclePolicy.NONE
    platforms: list[PlatformKind] = Field(default_factory=lambda: [PlatformKind.NVIDIA])
    entrypoint: str = Field(
        ...,
        description="Import path in 'module.path:ClassName' format.",
        pattern=r"^[a-zA-Z0-9_.]+:[a-zA-Z0-9_]+$",
    )
    protocol_version: str = Field(default="1.0", min_length=1)
    assets: dict[str, PinnedAsset] = Field(default_factory=dict)
    publication_policy: PublicationPolicy = Field(default_factory=PublicationPolicy)
    defaults: dict[str, Any] = Field(default_factory=dict)
    required_for_quality: bool = False

    @model_validator(mode="after")
    def _validate_mode_and_lifecycle(self) -> WorkloadDefinition:
        if (
            self.execution_mode == ExecutionMode.DIRECT
            and self.lifecycle != LifecyclePolicy.NONE
        ):
            raise ValueError(
                "Direct workloads must use lifecycle='none' (no server is managed)."
            )
        return self


class SuiteDefinition(BaseModel):
    """Named collection of benchmarks and/or evals (smoke, standard, extended)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    benchmarks: list[dict[str, Any]] = Field(default_factory=list)
    evals: list[dict[str, Any]] = Field(default_factory=list)
