# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4 Direct benchmark plugin: proves direct-mode execution without HTTP server."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.result import EvidenceSource, ExecutionStatus


class DirectSmokeBenchmarkRunner(BenchmarkRunner):
    """Reference direct-mode plugin exercising execution without vllm serve."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        cases = params.get("cases") or ["smoke=1"]
        return [
            WorkloadPoint(point_id=str(case).replace("=", "_"), params={"case": case})
            for case in cases
        ]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext | DirectRunContext,
    ) -> PointResult:
        assert isinstance(ctx, DirectRunContext)
        case = str(point.params.get("case", "smoke=1"))
        model = str(ctx.serve_config.get("model", ""))

        t0 = time.monotonic()
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys; "
                    "print(json.dumps({'schema_version': 1, 'case': sys.argv[1], "
                    "'model': sys.argv[2]}))"
                ),
                case,
                model,
            ],
            capture_output=True,
            text=True,
            timeout=min(30, ctx.timeout_s),
            check=False,
        )
        elapsed_ms = round((time.monotonic() - t0) * 1000.0, 3)

        log_path = ctx.output_dir / "direct.log"
        log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            return PointResult(
                status=ExecutionStatus.FAILED,
                evidence_source=EvidenceSource.CANDIDATE,
                infra_errors=1,
                error_message=f"Direct command exited with {proc.returncode}",
                artifacts=[log_path],
            )

        parsed = json.loads(proc.stdout.strip())
        if parsed.get("schema_version") != 1:
            return PointResult(
                status=ExecutionStatus.FAILED,
                evidence_source=EvidenceSource.CANDIDATE,
                infra_errors=1,
                error_message=(
                    f"Unsupported direct output schema_version: "
                    f"{parsed.get('schema_version')}"
                ),
                artifacts=[log_path],
            )

        out_path = ctx.output_dir / "timing.json"
        out_path.write_text(
            json.dumps({"case": case, "elapsed_ms": elapsed_ms}, indent=2) + "\n",
            encoding="utf-8",
        )
        return PointResult(
            status=ExecutionStatus.COMPLETED,
            evidence_source=EvidenceSource.CANDIDATE,
            metrics={"elapsed_ms": elapsed_ms},
            attempted=1,
            completed=1,
            artifacts=[log_path, out_path],
        )
