# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C5/C6: Shared bundle.tar.gz validator reused by bundling and upload."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

from pydantic import ValidationError

from tools.byog.schemas.result import BundleManifest, RunResult

REQUIRED_BUNDLE_FILES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "result.json",
        "spec.yaml",
        "spec.resolved.yaml",
        "repro.sh",
    }
)


class BundleValidationError(ValueError):
    """Raised when bundle.tar.gz fails schema, manifest, or integrity checks."""


def validate_bundle(bundle_path: Path) -> tuple[RunResult, BundleManifest]:
    """Validate bundle structure, schemas, file sizes, and SHA256 digests."""
    if not bundle_path.exists():
        raise BundleValidationError(f"Bundle not found: {bundle_path}")

    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            members: dict[str, bytes] = {}
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name.lstrip("./")
                if name.startswith("/") or ".." in Path(name).parts:
                    raise BundleValidationError(
                        f"Unsafe archive member path '{member.name}' in {bundle_path}."
                    )
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                members[name] = extracted.read()
    except (tarfile.TarError, OSError) as exc:
        raise BundleValidationError(
            f"Unable to open gzip tar archive {bundle_path}: {exc}"
        ) from exc

    missing = REQUIRED_BUNDLE_FILES - set(members)
    if missing:
        raise BundleValidationError(
            f"Bundle missing required file(s): {', '.join(sorted(missing))}"
        )

    try:
        manifest = BundleManifest.model_validate_json(members["manifest.json"])
    except ValidationError as exc:
        raise BundleValidationError(f"Invalid manifest.json: {exc}") from exc

    try:
        result = RunResult.model_validate_json(members["result.json"])
    except ValidationError as exc:
        raise BundleValidationError(f"Invalid result.json: {exc}") from exc

    if manifest.job_id != result.job_id or manifest.attempt_id != result.attempt_id:
        raise BundleValidationError(
            "Manifest job_id/attempt_id does not match result.json."
        )

    manifest_paths = {entry.path for entry in manifest.files}
    actual_paths = set(members) - {"manifest.json"}
    if manifest_paths != actual_paths:
        extra = sorted(actual_paths - manifest_paths)
        absent = sorted(manifest_paths - actual_paths)
        raise BundleValidationError(
            f"Manifest file list mismatch: extra={extra}, missing={absent}"
        )

    for entry in manifest.files:
        data = members[entry.path]
        if len(data) != entry.size_bytes:
            raise BundleValidationError(
                f"Size mismatch for '{entry.path}': "
                f"manifest={entry.size_bytes}, actual={len(data)}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256:
            raise BundleValidationError(
                f"SHA256 mismatch for '{entry.path}': "
                f"manifest={entry.sha256}, actual={digest}"
            )

    return result, manifest
