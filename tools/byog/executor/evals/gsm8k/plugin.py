# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4 Eval plugin: GSM8K accuracy evaluation against OpenAI-compatible vLLM API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tools.byog.executor.interfaces import (
    EvalRecordWriter,
    EvalRunner,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.result import ErrorKind, ExecutionStatus

INVALID_ANSWER = -9999999


def extract_gsm8k_answer(text: str) -> int:
    """Extract the final numerical answer matching tests/evals/gsm8k/gsm8k_eval.py."""
    cleaned = text.replace(",", "")
    nums: list[str] = []
    cur: list[str] = []
    for ch in cleaned:
        if ch.isdigit():
            cur.append(ch)
        elif cur:
            nums.append("".join(cur))
            cur = []
    if cur:
        nums.append("".join(cur))
    if not nums:
        return INVALID_ANSWER
    try:
        return int(nums[-1])
    except ValueError:
        return INVALID_ANSWER


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                rows.append(json.loads(s))
    return rows


def _call_endpoint(
    *,
    base_url: str,
    model: str,
    prompt: str,
    use_chat: bool,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    timeout_s: float,
) -> tuple[str, int, float, ErrorKind, str | None]:
    t0 = time.monotonic()
    if use_chat:
        url = f"{base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    else:
        url = f"{base_url}/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["Question:", "Assistant:", "<|separator|>"],
        }
    if seed is not None:
        payload["seed"] = seed

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency = time.monotonic() - t0
        choice = body["choices"][0]
        if use_chat:
            text = choice["message"].get("content") or ""
        else:
            text = choice.get("text") or ""
        tokens = int(body.get("usage", {}).get("completion_tokens", 0))
        return text, tokens, latency, ErrorKind.NONE, None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "", 0, time.monotonic() - t0, ErrorKind.INFRA, str(exc)
    except Exception as exc:
        return "", 0, time.monotonic() - t0, ErrorKind.MODEL, str(exc)


class GSM8KEvalRunner(EvalRunner):
    """Self-contained GSM8K eval plugin honoring BYOG's C0 EvalRunner contract."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        concurrencies = params.get("concurrency") or [16]
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
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult:
        train_asset = ctx.assets.resolve(ctx.workload_assets["train"])
        test_asset = ctx.assets.resolve(ctx.workload_assets["test"])
        train_rows = _read_jsonl(train_asset.path)
        test_rows = _read_jsonl(test_asset.path)

        num_shots = int(point.params.get("num_shots", 5))
        num_questions = min(
            int(point.params.get("num_questions", len(test_rows))), len(test_rows)
        )
        repeats = int(point.params.get("repeats", 1))
        concurrency = int(point.params.get("concurrency", 16))
        max_tokens = int(point.params.get("max_tokens", 256))
        temperature = float(point.params.get("temperature", 0.0))
        seed = point.params.get("seed", 42)
        use_chat = bool(point.params.get("use_chat_completions", False))
        req_timeout = float(point.params.get("request_timeout_s", 120.0))

        few_shot = ""
        for row in train_rows[:num_shots]:
            few_shot += f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"

        tasks: list[tuple[int, int, str, str, int, str]] = []
        for repeat in range(repeats):
            for idx in range(num_questions):
                q = str(test_rows[idx]["question"])
                ref_str = str(test_rows[idx]["answer"])
                ref_val = extract_gsm8k_answer(ref_str)
                prompt = f"{few_shot}Question: {q}\nAnswer:"
                tasks.append((repeat, idx, q, ref_str, ref_val, prompt))

        t_start = time.monotonic()

        def _exec_one(
            item: tuple[int, int, str, str, int, str],
        ) -> tuple[
            int, int, str, str, int, str, str, int, float, ErrorKind, str | None
        ]:
            repeat, idx, q, ref_str, ref_val, prompt = item
            text, toks, lat, err_kind, err_msg = _call_endpoint(
                base_url=ctx.base_url,
                model=ctx.model,
                prompt=prompt,
                use_chat=use_chat,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=(int(seed) + repeat) if seed is not None else None,
                timeout_s=req_timeout,
            )
            return (
                repeat,
                idx,
                q,
                ref_str,
                ref_val,
                prompt,
                text,
                toks,
                lat,
                err_kind,
                err_msg,
            )

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            outcomes = list(pool.map(_exec_one, tasks))

        total_latency = time.monotonic() - t_start
        correct_count = 0
        invalid_count = 0
        model_errors = 0
        infra_errors = 0
        completed = 0
        total_tokens = 0

        for (
            repeat,
            idx,
            q,
            ref_str,
            ref_val,
            prompt,
            text,
            toks,
            lat,
            err_kind,
            err_msg,
        ) in outcomes:
            extracted = (
                extract_gsm8k_answer(text)
                if err_kind == ErrorKind.NONE
                else INVALID_ANSWER
            )
            is_correct = err_kind == ErrorKind.NONE and extracted == ref_val
            if err_kind == ErrorKind.INFRA:
                infra_errors += 1
            elif err_kind == ErrorKind.MODEL:
                model_errors += 1
            else:
                completed += 1
                total_tokens += toks
                if extracted == INVALID_ANSWER:
                    invalid_count += 1
                if is_correct:
                    correct_count += 1

            record_writer.write_response(
                dataset_id="gsm8k",
                dataset_revision=test_asset.resolved_revision,
                sample_id=str(idx),
                question=q,
                reference_answer=ref_str,
                rendered_input=prompt,
                repeat=repeat,
                concurrency=concurrency,
                model_response=text,
                extracted_answer=None if extracted == INVALID_ANSWER else extracted,
                score=1.0 if is_correct else 0.0,
                correct=is_correct,
                error_kind=err_kind,
                error_message=err_msg,
                latency_s=round(lat, 4),
                output_tokens=toks,
            )

        denom = len(tasks)
        accuracy = correct_count / denom if denom > 0 else 0.0
        invalid_rate = invalid_count / denom if denom > 0 else 0.0
        summary = {
            "accuracy": round(accuracy, 4),
            "invalid_rate": round(invalid_rate, 4),
            "num_questions": num_questions,
            "repeats": repeats,
            "concurrency": concurrency,
            "total_output_tokens": total_tokens,
            "wall_time_s": round(total_latency, 3),
            "tokens_per_second": round(
                total_tokens / total_latency if total_latency > 0 else 0.0, 2
            ),
        }
        summary_path = ctx.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        return PointResult(
            status=(
                ExecutionStatus.COMPLETED
                if infra_errors == 0
                else ExecutionStatus.FAILED
            ),
            metrics={
                "accuracy": summary["accuracy"],
                "invalid_rate": summary["invalid_rate"],
                "tokens_per_second": summary["tokens_per_second"],
            },
            attempted=denom,
            completed=completed,
            scoring_denominator=denom,
            model_errors=model_errors,
            infra_errors=infra_errors,
            error_message=(
                f"{infra_errors} infra request error(s)" if infra_errors > 0 else None
            ),
            artifacts=[summary_path],
        )
