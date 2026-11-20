# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2: Default SHA256-verifying AssetProvider for local paths and HTTPS URLs."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from tools.byog.executor.interfaces import AssetProvider, ResolvedAsset
from tools.byog.schemas.catalog import PinnedAsset


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


class CachingAssetProvider(AssetProvider):
    """Resolves pinned local or HTTPS assets into a content-addressed cache."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        local_overrides: dict[str, Path] | None = None,
    ):
        env_cache = os.environ.get("BYOG_ASSET_CACHE")
        default_dir = (
            Path(env_cache) if env_cache else Path.home() / ".cache" / "byog" / "assets"
        )
        self.cache_dir = cache_dir or default_dir
        self.local_overrides = local_overrides or {}

    def resolve(self, asset: PinnedAsset) -> ResolvedAsset:
        if asset.id in self.local_overrides:
            path = self.local_overrides[asset.id]
            digest = _sha256_file(path)
            return ResolvedAsset(
                asset_id=asset.id,
                path=path,
                resolved_revision=asset.revision,
                sha256=digest,
            )

        if not asset.source.startswith(("http://", "https://")):
            path = Path(asset.source)
            if not path.exists():
                raise FileNotFoundError(
                    f"Local asset '{asset.id}' not found at {path}."
                )
            digest = _sha256_file(path)
            if asset.sha256 != "skip-fixture-check" and digest != asset.sha256:
                raise ValueError(
                    f"SHA256 mismatch for asset '{asset.id}': "
                    f"expected {asset.sha256}, got {digest}"
                )
            return ResolvedAsset(
                asset_id=asset.id,
                path=path,
                resolved_revision=asset.revision,
                sha256=digest,
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(asset.source).suffix or ".dat"
        cached = self.cache_dir / f"{asset.id}-{asset.sha256[:16]}{suffix}"
        if cached.exists():
            digest = _sha256_file(cached)
            if digest == asset.sha256:
                return ResolvedAsset(
                    asset_id=asset.id,
                    path=cached,
                    resolved_revision=asset.revision,
                    sha256=digest,
                )
            cached.unlink()

        urls = [asset.source]
        if asset.mirror_url:
            urls.append(asset.mirror_url)

        last_err: Exception | None = None
        for url in urls:
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            try:
                urllib.request.urlretrieve(url, tmp)
                digest = _sha256_file(tmp)
                if digest != asset.sha256:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(
                        f"SHA256 mismatch for asset '{asset.id}' from {url}: "
                        f"expected {asset.sha256}, got {digest}"
                    )
                tmp.replace(cached)
                return ResolvedAsset(
                    asset_id=asset.id,
                    path=cached,
                    resolved_revision=asset.revision,
                    sha256=digest,
                )
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                last_err = exc

        raise RuntimeError(
            f"Failed to fetch pinned asset '{asset.id}': {last_err}"
        ) from last_err
