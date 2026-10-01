"""Import a command-line script of ``scripts/`` as a module, for testing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str) -> ModuleType:
    """Load ``scripts/<name>.py`` without running its ``__main__`` block.

    The scripts folder is put on ``sys.path``, as when a script is run, so
    that one script can import another (e.g. replay imports run_radar).
    """
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
