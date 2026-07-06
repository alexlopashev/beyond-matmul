#!/usr/bin/env python3
"""Tiny PyTorch Conv1d workload case study."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.case_study_artifacts import print_conv1d_demo


def main() -> int:
    try:
        print_conv1d_demo()
    except RuntimeError as exc:
        if "PyTorch is required" not in str(exc):
            raise
        print("PyTorch is not installed in this environment, so the Conv1d workload demo was skipped.")
        print("Install project dependencies and rerun:")
        print("  mise exec -- uv sync --locked")
        print("  mise exec -- uv run python examples/conv1d_workload_demo.py")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
