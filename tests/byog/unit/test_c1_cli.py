# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C1 CLI subcommands and reusable plugin contract helper."""

from __future__ import annotations

import io
import json
from pathlib import Path

from tests.byog.contract_helpers import assert_plugin_contract
from tests.byog.unit.test_c4_executor import _test_catalog
from tools.byog.cli import main as cli_main


def test_cli_catalog_list_and_validate_errors(tmp_path: Path) -> None:
    cat = _test_catalog()
    out, err = io.StringIO(), io.StringIO()
    assert cli_main(["catalog", "list"], catalog=cat, stdout=out, stderr=err) == 0
    assert "http_reset" in out.getvalue()
    assert "direct_only" in out.getvalue()

    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nserve:\n  config: {model: m}\n")
    out2, err2 = io.StringIO(), io.StringIO()
    assert (
        cli_main(
            ["validate", "--spec", str(bad)], catalog=cat, stdout=out2, stderr=err2
        )
        == 2
    )
    assert "revision" in err2.getvalue()


def test_cli_dry_run_and_contract_helper_for_server_direct_and_eval(
    tmp_path: Path,
) -> None:
    cat = _test_catalog()

    # HTTP benchmark via contract helper
    res_http, bundle_http = assert_plugin_contract(
        workload_name="http_reset",
        spec_dict={
            "schema_version": 1,
            "name": "http-test",
            "serve": {"config": {"model": "m", "revision": "r1"}},
            "benchmarks": [{"type": "http_reset"}],
        },
        tmp_path=tmp_path / "http",
        catalog=cat,
    )
    assert res_http.measurements[0].metrics["throughput_tps"] == 100.0

    # `byog validate-bundle` CLI works on produced bundle
    out, err = io.StringIO(), io.StringIO()
    assert (
        cli_main(
            ["validate-bundle", str(bundle_http)],
            catalog=cat,
            stdout=out,
            stderr=err,
        )
        == 0
    )
    assert "Bundle OK" in out.getvalue()

    # Direct benchmark via contract helper (asserts zero servers launched)
    assert_plugin_contract(
        workload_name="direct_only",
        spec_dict={
            "schema_version": 1,
            "serve": {"config": {"model": "m", "revision": "r1"}},
            "benchmarks": [{"type": "direct_only"}],
        },
        tmp_path=tmp_path / "direct",
        catalog=cat,
    )

    # Eval via contract helper (asserts deduplicated JSONL files in bundle)
    assert_plugin_contract(
        workload_name="tiny_eval",
        spec_dict={
            "schema_version": 1,
            "serve": {"config": {"model": "m", "revision": "r1"}},
            "evals": [{"name": "tiny_eval"}],
        },
        tmp_path=tmp_path / "eval",
        catalog=cat,
    )

    # Dry run emits valid JSON plan
    spec_path = tmp_path / "dry.yaml"
    spec_path.write_text(
        "schema_version: 1\nserve:\n  config: {model: m, revision: r1}\n"
        "benchmarks:\n  - type: direct_only\n"
    )
    out_dry, err_dry = io.StringIO(), io.StringIO()
    assert (
        cli_main(
            ["run", "--spec", str(spec_path), "--dry-run"],
            catalog=cat,
            stdout=out_dry,
            stderr=err_dry,
        )
        == 0
    )
    plan = json.loads(out_dry.getvalue())
    assert plan[0]["job_id"] == "default"
