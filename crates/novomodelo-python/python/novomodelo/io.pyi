from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import model as model

def load_case(path: str | Path) -> model.System: ...
def validate(
    path: str | Path,
    config_overrides: Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]: ...
