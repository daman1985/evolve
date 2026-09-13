"""Make `import tscodec` resolve correctly when pytest is invoked from the
repo root as `python -m pytest tscodec/tests -q`.

The project root here (`.../tscodec/`) contains a child directory *also*
named `tscodec/` (the actual package, with `__init__.py`). `python -m
pytest` puts the current working directory (the repo root, one level above
this project root) on `sys.path`. From there, Python's import machinery
finds the outer, `__init__.py`-less project directory first and treats it as
a PEP 420 namespace package, shadowing the real package one level down and
breaking `from tscodec.codec import ...` with `ModuleNotFoundError:
No module named 'tscodec.codec'`.

Inserting this project root directly onto `sys.path` fixes the resolution:
Python then finds `.../tscodec/tscodec/__init__.py` as a regular package,
which always wins over a namespace package once found.
"""

import pathlib
import sys

_PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent  # .../tscodec
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))
