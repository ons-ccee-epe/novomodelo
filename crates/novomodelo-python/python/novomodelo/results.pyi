from pathlib import Path
from typing import Any, Optional, Union

def load_results(output_dir: Union[str, Path]) -> dict[str, Any]: ...
def load_convergence(output_dir: Union[str, Path]) -> Any: ...
def load_convergence_arrow(output_dir: Union[str, Path]) -> Any: ...
def load_simulation(
    output_dir: Union[str, Path],
    entity_type: Optional[str] = None,
) -> Any: ...
def load_simulation_arrow(
    output_dir: Union[str, Path],
    entity_type: Optional[str] = None,
) -> Any: ...
def load_policy(
    output_dir: Union[str, Path],
    policy_subdir: str = "policy",
) -> Any:
    """Load a policy checkpoint, returning a dict with "metadata" and "stage_cuts".

    metadata["season_manifest"] is always present with keys:
    - "cycle_code": int (0 monthly / 1 weekly / 2 custom / 255 absent)
    - "n_seasons": int
    - "hydro_orders": list of {"hydro_id": int, "orders": list[int]} dicts
    """
    ...

class Stochastic:
    def par_coefficients(self) -> Any: ...
    def opening_tree(self, stage: int) -> Any: ...

def load_stochastic(output_dir: Union[str, Path]) -> Stochastic: ...
