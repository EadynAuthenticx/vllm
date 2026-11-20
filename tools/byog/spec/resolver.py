# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C3: Single-job spec loader, security denylist, and catalog resolver."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from tools.byog.catalog import Catalog, CatalogError
from tools.byog.schemas.catalog import WorkloadKind
from tools.byog.schemas.spec import (
    ResolvedJob,
    ResolvedWorkload,
    RunSpec,
)

BLOCKED_ENV_PREFIXES: tuple[str, ...] = ("LD_", "PYTHON")
BLOCKED_ENV_EXACT: frozenset[str] = frozenset(
    {
        "PATH",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "CURL_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
    }
)


class SpecResolutionError(ValueError):
    """Raised when a BYOG run spec fails parsing, validation, or resolution."""


def generate_attempt_id() -> str:
    """Create a unique, sortable execution attempt ID owned by the harness."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:10]}"


def compute_config_hash(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA256 hex digest over canonical JSON."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_serve_env(serve_env: dict[str, str], job_id: str = "default") -> None:
    """Reject security-sensitive loader, proxy, CA, and credential overrides."""
    for key in serve_env:
        upper = key.upper()
        if upper in BLOCKED_ENV_EXACT or any(
            upper.startswith(prefix) for prefix in BLOCKED_ENV_PREFIXES
        ):
            raise SpecResolutionError(
                f"Job '{job_id}' field 'serve.env.{key}': "
                f"security-sensitive environment override '{key}' is not permitted."
            )


def load_spec_file(spec_path: Path) -> RunSpec:
    """Load and schema-validate a BYOG spec YAML file."""
    if not spec_path.exists():
        raise SpecResolutionError(f"Spec file not found: {spec_path}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SpecResolutionError(f"Spec file {spec_path} must be a YAML mapping.")
    if raw.get("schema_version") != 1:
        raise SpecResolutionError(
            f"Unsupported schema_version '{raw.get('schema_version')}' "
            f"in {spec_path}; supported version is 1."
        )
    try:
        return RunSpec.model_validate(raw)
    except ValidationError as exc:
        raise SpecResolutionError(f"Invalid spec {spec_path}: {exc}") from exc


def _merge_workload_params(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(defaults)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


def resolve_spec(
    spec: RunSpec,
    catalog: Catalog,
    *,
    attempt_id: str | None = None,
    source_tier: Literal["ci_nightly", "ci_pr", "community"] = "community",
    require_model_revision: bool = True,
) -> list[ResolvedJob]:
    """Resolve a RunSpec into concrete ResolvedJob instances.

    In Phase 0, single-job `serve:` specs are supported. `recipe:` and
    `matrix:` specs fail with clear messages pointing to their P1 tracks.
    """
    if spec.recipe is not None:
        raise SpecResolutionError(
            "Field 'recipe': recipe selector resolution is not yet implemented "
            "in Phase 0 (tracked in P1.3). Supply an explicit 'serve:' section."
        )
    if (
        spec.matrix is not None
        or spec.matrix_include is not None
        or spec.matrix_exclude is not None
    ):
        raise SpecResolutionError(
            "Field 'matrix': matrix expansion is not yet implemented in Phase 0 "
            "(tracked in P1.4). Supply a single-job spec without matrix keys."
        )

    assert spec.serve is not None
    job_id = "default"
    serve_config = {k: v for k, v in spec.serve.config.items() if v is not None}
    serve_env = dict(spec.serve.env)
    validate_serve_env(serve_env, job_id=job_id)

    if not serve_config.get("model"):
        raise SpecResolutionError(
            f"Job '{job_id}' field 'serve.config.model': model is required."
        )
    if require_model_revision and not serve_config.get("revision"):
        raise SpecResolutionError(
            f"Job '{job_id}' field 'serve.config.revision': "
            "pinned model revision is required."
        )

    # Resolve benchmarks (explicit list overrides suite; [] skips).
    raw_benchmarks: list[dict[str, Any]]
    if spec.benchmarks is not None:
        raw_benchmarks = [b.model_dump() for b in spec.benchmarks]
    elif spec.benchmark_suite is not None:
        try:
            suite = catalog.get_suite(spec.benchmark_suite)
        except CatalogError as exc:
            raise SpecResolutionError(
                f"Job '{job_id}' field 'benchmark_suite': {exc}"
            ) from exc
        raw_benchmarks = list(suite.benchmarks)
    else:
        raw_benchmarks = []

    # Resolve evals (explicit list overrides suite; [] skips).
    raw_evals: list[dict[str, Any]]
    if spec.evals is not None:
        raw_evals = [e.model_dump(exclude_none=True) for e in spec.evals]
    elif spec.eval_suite is not None:
        try:
            suite = catalog.get_suite(spec.eval_suite)
        except CatalogError as exc:
            raise SpecResolutionError(
                f"Job '{job_id}' field 'eval_suite': {exc}"
            ) from exc
        raw_evals = list(suite.evals)
    else:
        raw_evals = []

    workloads: list[ResolvedWorkload] = []

    for entry in raw_benchmarks:
        wtype = entry.get("type")
        if not isinstance(wtype, str):
            raise SpecResolutionError(
                f"Job '{job_id}' field 'benchmarks': missing string 'type'."
            )
        try:
            defn = catalog.get_workload(wtype, expected_kind=WorkloadKind.BENCHMARK)
        except CatalogError as exc:
            raise SpecResolutionError(
                f"Job '{job_id}' field 'benchmarks.{wtype}': {exc}"
            ) from exc
        overrides = {k: v for k, v in entry.items() if k != "type"}
        workloads.append(
            ResolvedWorkload(
                name=defn.name,
                kind="benchmark",
                execution_mode=defn.execution_mode.value,
                lifecycle=defn.lifecycle.value,
                protocol_version=defn.protocol_version,
                params=_merge_workload_params(defn.defaults, overrides),
            )
        )

    for entry in raw_evals:
        ename = entry.get("name")
        if not isinstance(ename, str):
            raise SpecResolutionError(
                f"Job '{job_id}' field 'evals': missing string 'name'."
            )
        try:
            defn = catalog.get_workload(ename, expected_kind=WorkloadKind.EVAL)
        except CatalogError as exc:
            raise SpecResolutionError(
                f"Job '{job_id}' field 'evals.{ename}': {exc}"
            ) from exc
        overrides = {k: v for k, v in entry.items() if k != "name"}
        workloads.append(
            ResolvedWorkload(
                name=defn.name,
                kind="eval",
                execution_mode=defn.execution_mode.value,
                lifecycle=defn.lifecycle.value,
                protocol_version=defn.protocol_version,
                params=_merge_workload_params(defn.defaults, overrides),
            )
        )

    if not workloads:
        raise SpecResolutionError(
            f"Job '{job_id}': at least one benchmark or eval must be selected."
        )

    # Protocol order: server benchmarks -> evals -> direct workloads.
    def _order_key(w: ResolvedWorkload) -> tuple[int, str]:
        if w.execution_mode == "server" and w.kind == "benchmark":
            return (0, w.name)
        if w.execution_mode == "server" and w.kind == "eval":
            return (1, w.name)
        return (2, w.name)

    workloads.sort(key=_order_key)

    normalized_hw = spec.hardware.model_copy(
        update={"gpu": [g.lower() for g in spec.hardware.gpu]}
    )
    config_hash = compute_config_hash(
        {
            "serve_config": serve_config,
            "serve_env": serve_env,
            "hardware": normalized_hw.model_dump(),
        }
    )

    job = ResolvedJob(
        job_id=job_id,
        attempt_id=attempt_id or generate_attempt_id(),
        spec_name=spec.name,
        serve_config=serve_config,
        serve_env=serve_env,
        hardware=normalized_hw,
        workloads=workloads,
        timeout_s=spec.timeout_s,
        source_tier=source_tier,
        resolved_config_hash=config_hash,
    )
    return [job]
