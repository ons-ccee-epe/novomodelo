"""Novomodelo — Python bindings for the Novomodelo power systems solver.

This is the public ``novomodelo`` package. The actual solver is implemented in the
compiled extension module ``novomodelo._native`` (a PyO3/maturin ``cdylib``). This
``__init__`` re-exports the compiled module's public surface so that
``import novomodelo``, ``novomodelo.Study``, ``novomodelo.run.run(...)`` and friends resolve
exactly as they did before the crate adopted the maturin mixed layout.

The split into a private compiled ``_native`` plus a pure-Python ``novomodelo``
package exists so ergonomic wrappers can be authored in Python rather than
PyO3. This file is the single extension point for those additions.
"""

from __future__ import annotations

import sys

# Pull the top-level public names (Study, Policy, version_info, __version__)
# from the compiled extension module.
from ._native import *  # noqa: F401,F403

# `from ._native import *` may skip dunders; re-bind __version__ explicitly so
# `novomodelo.__version__` is preserved byte-for-byte from the compiled module.
from ._native import __version__ as __version__

# Re-export the compiled submodules under their public `novomodelo.*` names. Binding
# them here exposes `novomodelo.run` etc. as attributes; registering each in
# `sys.modules` under the public name makes `import novomodelo.run` resolve to the
# very same compiled module object (no shadowing copy). This mirrors the Rust
# `register_submodule` sys.modules trick, but on the public side.
from ._native import errors as errors
from ._native import io as io
from ._native import model as model
from ._native import results as results
from ._native import run as run
from ._native import schema as schema

sys.modules["novomodelo.errors"] = errors
sys.modules["novomodelo.io"] = io
sys.modules["novomodelo.model"] = model
sys.modules["novomodelo.results"] = results
sys.modules["novomodelo.run"] = run
sys.modules["novomodelo.schema"] = schema

# Top-level public classes/functions re-exported from `_native`.
from ._native import Policy as Policy
from ._native import Study as Study
from ._native import version_info as version_info
from ._native import write_policy_checkpoint as write_policy_checkpoint

# --- Extension point ---------------------------------------------------------
# Pure-Python ergonomic wrappers can be layered on top of the compiled surface
# here. When such wrappers land, import them above and add their public names
# to `__all__` below. The typed `novomodelo.errors` exception hierarchy is registered
# from Rust (under `novomodelo._native.errors`) and re-exported above, so
# `import novomodelo.errors` and `from novomodelo.errors import ValidationError` resolve
# to the compiled classes.

__all__ = [
    "Study",
    "Policy",
    "version_info",
    "write_policy_checkpoint",
    "errors",
    "io",
    "model",
    "run",
    "results",
    "schema",
]
