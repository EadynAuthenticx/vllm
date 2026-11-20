# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4 Benchmark plugin: lightweight HTTP throughput/latency smoke benchmark."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.result import ExecutionStatus


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 4)


class HttpSmokeBenchmarkRunner(BenchmarkRunner):
    """Reference HTTP serving benchmark exercising server mode + cache reset."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        concurrencies = params.get("concurrency") or [1, 4]
        return [
            WorkloadPoint(
                point_id=f"c{int(c)}",
                params={**params, "concurrency": int(c)},
            )
            for c in concurrencies
        ]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext | DirectRunContext,
    ) -> PointResult:
        assert isinstance(ctx, ServerRunContext)
        concurrency = int(point.params.get("concurrency", 1))
        num_requests = int(point.params.get("num_requests", max(4, concurrency * 2)))
        max_tokens = int(point.params.get("max_tokens", 32))
        prompt = str(
            point.params.get(
                "prompt",
                "Write a short one-sentence summary of what vLLM does.",
            )
        )

        def _send_one(idx: int) -> dict[str, Any]:
            t0 = time.monotonic()
            req = urllib.request.Request(
                f"{ctx.base_url}/v1/completions",
                data=json.dumps(
                    {
                        "model": ctx.model,
                        "prompt": f"{prompt} (#{idx})",
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                lat = time.monotonic() - t0
                out_toks = int(body.get("usage", {}).get("completion_tokens", 0))
                return {
                    "idx": idx,
                    "ok": True,
                    "latency_s": round(lat, 4),
                    "output_tokens": out_toks,
                }
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return {
                    "idx": idx,
                    "ok": False,
                    "latency_s": round(time.monotonic() - t0, 4),
                    "output_tokens": 0,
                    "error": str(exc),
                }

        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            records = list(pool.map(_send_one, range(num_requests)))
        wall_s = max(1e-6, time.monotonic() - t_start)

        ok_recs = [r for r in records if r["ok"]]
        latencies = sorted(float(r["latency_s"]) for r in ok_recs)
        total_out_toks = sum(int(r["output_tokens"]) for r in ok_recs)
        n = len(ok_recs)

        summary: dict[str, float | int | str | None] = {
            "output_tokens_per_s": round(total_out_toks / wall_s, 2),
            "requests_per_s": round(n / wall_s, 2),
            "p50_latency_s": _percentile(latencies, 50.0),
            "p90_latency_s": _percentile(latencies, 90.0),
            # Gate high tail percentiles on statistically meaningful sample counts.
            "p99_latency_s": _percentile(latencies, 99.0) if n >= 100 else None,
            "p999_latency_s": _percentile(latencies, 99.9) if n >= 1000 else None,
        }

        req_path = ctx.output_dir / "requests.jsonl"
        req_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        sum_path = ctx.output_dir / "summary.json"
        sum_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        infra_errors = num_requests - n
        return PointResult(
            status=(
                ExecutionStatus.COMPLETED
                if infra_errors == 0
                else ExecutionStatus.FAILED
            ),
            metrics=summary,
            attempted=num_requests,
            completed=n,
            infra_errors=infra_errors,
            error_message=(
                f"{infra_errors} failed HTTP requests" if infra_errors > 0 else None
            ),
            artifacts=[req_path, sum_path],
        )
