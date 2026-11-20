# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2: Auto-discovered catalog of BYOG workloads and suites."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from tools.byog.schemas.catalog import (
    SuiteDefinition,
    WorkloadDefinition,
    WorkloadKind,
)

DEFAULT_CATALOG_DIR = Path(__file__).resolve().parent


class CatalogError(ValueError):
    """Raised when catalog definitions are invalid or lookup fails."""


class Catalog:
    """Loads workload and suite YAMLs via directory auto-discovery.

    Adding a workload requires only dropping a YAML file into
    `catalog/workloads/<name>.yaml`; no shared registry file needs edits.
    """

    def __init__(
        self,
        workloads: dict[str, WorkloadDefinition],
        suites: dict[str, SuiteDefinition],
    ):
        self._workloads = workloads
        self._suites = suites

    @classmethod
    def discover(
        cls,
        catalog_dir: Path | None = None,
        extra_workload_dirs: list[Path] | None = None,
    ) -> Catalog:
        root = catalog_dir or DEFAULT_CATALOG_DIR
        workload_dirs = [root / "workloads", *(extra_workload_dirs or [])]
        workloads: dict[str, WorkloadDefinition] = {}

        for wdir in workload_dirs:
            if not wdir.exists():
                continue
            for path in sorted(wdir.glob("*.yaml")):
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise CatalogError(f"Workload file {path} must contain a mapping.")
                defn = WorkloadDefinition.model_validate(raw)
                if defn.name in workloads:
                    raise CatalogError(
                        f"Duplicate workload name '{defn.name}' found in {path}."
                    )
                workloads[defn.name] = defn

        suites: dict[str, SuiteDefinition] = {}
        suites_dir = root / "suites"
        if suites_dir.exists():
            for path in sorted(suites_dir.glob("*.yaml")):
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise CatalogError(f"Suite file {path} must contain a mapping.")
                suite = SuiteDefinition.model_validate(raw)
                if suite.name in suites:
                    raise CatalogError(
                        f"Duplicate suite name '{suite.name}' found in {path}."
                    )
                suites[suite.name] = suite

        return cls(workloads=workloads, suites=suites)

    def list_workloads(
        self, kind: WorkloadKind | None = None
    ) -> list[WorkloadDefinition]:
        items = list(self._workloads.values())
        if kind is not None:
            items = [w for w in items if w.kind == kind]
        return sorted(items, key=lambda w: (w.kind.value, w.name))

    def list_suites(self) -> list[SuiteDefinition]:
        return sorted(self._suites.values(), key=lambda s: s.name)

    def get_workload(
        self, name: str, expected_kind: WorkloadKind | None = None
    ) -> WorkloadDefinition:
        if name not in self._workloads:
            known = ", ".join(sorted(self._workloads)) or "<none>"
            raise CatalogError(
                f"Unknown workload '{name}'. Available workloads: {known}"
            )
        defn = self._workloads[name]
        if expected_kind is not None and defn.kind != expected_kind:
            raise CatalogError(
                f"Workload '{name}' has kind '{defn.kind.value}', "
                f"expected '{expected_kind.value}'."
            )
        return defn

    def get_suite(self, name: str) -> SuiteDefinition:
        if name not in self._suites:
            known = ", ".join(sorted(self._suites)) or "<none>"
            raise CatalogError(f"Unknown suite '{name}'. Available suites: {known}")
        return self._suites[name]

    def instantiate_plugin(self, name: str) -> Any:
        defn = self.get_workload(name)
        mod_path, cls_name = defn.entrypoint.split(":", 1)
        module = importlib.import_module(mod_path)
        plugin_cls = getattr(module, cls_name)
        return plugin_cls()
