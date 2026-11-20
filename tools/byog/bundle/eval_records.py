# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C5: Shared deduplicated eval record writer honoring trusted catalog policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.byog.schemas.catalog import PublicationPolicy
from tools.byog.schemas.result import (
    ErrorKind,
    EvalRenderedInputRecord,
    EvalResponseRecord,
    EvalSampleRecord,
)


def _hash_rendered_input(rendered_input: Any) -> str:
    canonical = json.dumps(rendered_input, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class DeduplicatedEvalRecordWriter:
    """Writes normalized eval records split into samples, rendered inputs, responses.

    Publication permissions come exclusively from the trusted catalog's
    `PublicationPolicy`; plugins cannot grant themselves permission to publish
    restricted question text, reference answers, or model responses.
    """

    def __init__(self, output_dir: Path, policy: PublicationPolicy):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self._samples_path = self.output_dir / "samples.jsonl"
        self._inputs_path = self.output_dir / "inputs.jsonl"
        self._responses_path = self.output_dir / "responses.jsonl"
        self._seen_samples: set[tuple[str, str]] = set()
        self._seen_inputs: set[str] = set()
        self._samples_fh = self._samples_path.open("w", encoding="utf-8")
        self._inputs_fh = self._inputs_path.open("w", encoding="utf-8")
        self._responses_fh = self._responses_path.open("w", encoding="utf-8")
        self._closed = False
        self.withheld_fields: set[str] = set()

    def write_response(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        sample_id: str,
        question: str,
        reference_answer: str,
        rendered_input: Any,
        repeat: int,
        concurrency: int,
        model_response: str,
        extracted_answer: Any,
        score: float | None,
        correct: bool | None,
        error_kind: ErrorKind = ErrorKind.NONE,
        error_message: str | None = None,
        latency_s: float | None = None,
        output_tokens: int | None = None,
        media_refs: list[str] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("EvalRecordWriter is already closed.")

        reviewed = self.policy.reviewed
        allow_q = reviewed and self.policy.allow_question_text
        allow_ref = reviewed and self.policy.allow_reference_answer
        allow_resp = reviewed and self.policy.allow_model_response
        withheld_reason = (
            None
            if reviewed
            else "dataset publication policy not yet reviewed in catalog"
        )

        sample_key = (dataset_revision, sample_id)
        if sample_key not in self._seen_samples:
            self._seen_samples.add(sample_key)
            if not allow_q:
                self.withheld_fields.add("question")
            if not allow_ref:
                self.withheld_fields.add("reference_answer")
            sample_rec = EvalSampleRecord(
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                sample_id=sample_id,
                question=question if allow_q else None,
                reference_answer=reference_answer if allow_ref else None,
                media_refs=list(media_refs or []),
                withheld_reason=(
                    None
                    if (allow_q and allow_ref)
                    else (withheld_reason or "withheld by catalog policy")
                ),
            )
            self._samples_fh.write(sample_rec.model_dump_json() + "\n")

        input_hash = _hash_rendered_input(rendered_input)
        if input_hash not in self._seen_inputs:
            self._seen_inputs.add(input_hash)
            if not allow_q:
                self.withheld_fields.add("rendered_input")
            input_rec = EvalRenderedInputRecord(
                input_hash=input_hash,
                dataset_revision=dataset_revision,
                sample_id=sample_id,
                rendered_input=rendered_input if allow_q else None,
                withheld_reason=(
                    None
                    if allow_q
                    else (withheld_reason or "withheld by catalog policy")
                ),
            )
            self._inputs_fh.write(input_rec.model_dump_json() + "\n")

        if not allow_resp:
            self.withheld_fields.add("model_response")
        resp_rec = EvalResponseRecord(
            dataset_revision=dataset_revision,
            sample_id=sample_id,
            input_hash=input_hash,
            repeat=repeat,
            concurrency=concurrency,
            model_response=model_response if allow_resp else None,
            extracted_answer=extracted_answer,
            score=score,
            correct=correct,
            error_kind=error_kind,
            error_message=error_message,
            latency_s=latency_s,
            output_tokens=output_tokens,
            withheld_reason=(
                None
                if allow_resp
                else (withheld_reason or "withheld by catalog policy")
            ),
        )
        self._responses_fh.write(resp_rec.model_dump_json() + "\n")

    def close(self) -> list[Path]:
        if not self._closed:
            self._samples_fh.close()
            self._inputs_fh.close()
            self._responses_fh.close()
            self._closed = True
        return [self._samples_path, self._inputs_path, self._responses_path]
