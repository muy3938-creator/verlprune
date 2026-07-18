"""Keep pruning tests independent from the distributed VerL runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

if importlib.util.find_spec("ray") is None:
    package = ModuleType("verl")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "verl")]
    sys.modules["verl"] = package
