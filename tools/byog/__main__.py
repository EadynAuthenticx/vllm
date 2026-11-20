# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Module entrypoint for `python -m tools.byog`."""

import sys

from tools.byog.cli import main

if __name__ == "__main__":
    sys.exit(main())
