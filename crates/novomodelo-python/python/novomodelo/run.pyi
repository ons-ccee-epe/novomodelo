from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ._types import RunResult

def run(
    case_dir: str | Path,
    output_dir: str | Path | None = None,
    threads: int | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    on_iteration: Callable[[dict[str, Any]], bool | None] | None = None,
) -> RunResult: ...
