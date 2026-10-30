"""Repository-local entry point for the TwinSepsis Streamlit dashboard.

The project is not installed as a package, so this wrapper adds ``src`` to the
import path before delegating to :func:`twinsepsis.dashboard.app.main`.

The dashboard needs the CPython 3.12 environment because ``ensemble`` mode
requires CPU PyTorch::

    C:\\TwinSepsis-venvs\\gru312\\Scripts\\python.exe -m streamlit run dashboard.py

``xgboost`` mode works in either interpreter, but the 3.12 environment is the
supported one and is what the launch command above uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.dashboard.app import main  # noqa: E402

if __name__ == "__main__":
    main()
