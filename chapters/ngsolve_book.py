"""Import shim for live notebooks executed from the chapters directory."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_root_helper = Path(__file__).resolve().parent.parent / "ngsolve_book.py"
_spec = spec_from_file_location("_ngsolve_book_root", _root_helper)

if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load NGSolve book helper from {_root_helper}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

Draw = _module.Draw

__all__ = ["Draw"]
