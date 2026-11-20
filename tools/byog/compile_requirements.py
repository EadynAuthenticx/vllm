#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C7: Compile tools/byog/requirements.d/*.txt into tools/byog/requirements.lock."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BYOG_DIR = Path(__file__).resolve().parent
REQ_D_DIR = BYOG_DIR / "requirements.d"
LOCK_FILE = BYOG_DIR / "requirements.lock"


def collect_fragment_files() -> list[Path]:
    return sorted(REQ_D_DIR.glob("*.txt"))


def compile_lock(output_path: Path) -> None:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise RuntimeError("`uv` is required to compile tools/byog/requirements.lock.")
    fragments = collect_fragment_files()
    if not fragments:
        raise RuntimeError(f"No requirement fragments found in {REQ_D_DIR}")

    cmd = [
        uv_bin,
        "pip",
        "compile",
        "--python-version",
        "3.12",
        "--universal",
        "--no-header",
        "--output-file",
        str(output_path),
        *[str(p) for p in fragments],
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if requirements.lock is out of sync with requirements.d/.",
    )
    args = parser.parse_args(argv)

    if args.check:
        if not LOCK_FILE.exists():
            print(f"Missing lockfile: {LOCK_FILE}", file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "requirements.lock"
            compile_lock(candidate)
            if candidate.read_text(encoding="utf-8") != LOCK_FILE.read_text(
                encoding="utf-8"
            ):
                print(
                    "tools/byog/requirements.lock is out of date. "
                    "Run: python3 tools/byog/compile_requirements.py",
                    file=sys.stderr,
                )
                return 1
        print("tools/byog/requirements.lock is up to date.")
        return 0

    compile_lock(LOCK_FILE)
    print(f"Wrote {LOCK_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
