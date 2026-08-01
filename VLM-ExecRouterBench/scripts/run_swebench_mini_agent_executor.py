#!/usr/bin/env python3
"""Canonical SWE-bench entrypoint for the RouterSFT mini_agent backend."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_swebench_mini_swe_agent_executor import main  # noqa: E402


if __name__ == "__main__":
    main()
