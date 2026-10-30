"""Repository-local entry point for the TwinSepsis agent CLI.

The project is not installed as a package, so this wrapper adds ``src`` to the
import path and delegates to :func:`twinsepsis.agent.cli.main`.

Examples
--------
    python run_agent.py --patient-id p000001 --hour 6 --tools-only
    python run_agent.py --patient-id p000001 --hour 6 --request "Summarize risk"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
