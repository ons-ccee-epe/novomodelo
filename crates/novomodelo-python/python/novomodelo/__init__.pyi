from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from . import errors as errors
from . import io as io
from . import model as model
from . import results as results
from . import run as run
from . import schema as schema
from ._types import HydroModelsSummary, ProvenanceReport, StochasticSummary
from .model import System

__version__: str

def version_info() -> dict[str, Any]: ...
def write_policy_checkpoint(
    path: Union[str, Path],
    stage_cuts: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    stage_bases: Optional[Sequence[Mapping[str, Any]]] = None,
    stage_states: Optional[Sequence[Mapping[str, Any]]] = None,
    inflow_lag_depth: Optional[int] = None,
) -> None:
    """Write a policy checkpoint from plain Python dicts.

    ``stage_cuts``, ``metadata``, ``stage_bases`` and ``stage_states`` take the
    shapes ``novomodelo.results.load_policy`` returns, so ``load_policy`` -> edit ->
    ``write_policy_checkpoint`` keeps every field. Keys marked optional take the
    stated default when omitted; keys a loaded result carries beyond those
    listed are ignored.

    ``stage_cuts``: one dict per pool.
    - "stage_id": int, the pool id
    - "state_dimension": int
    - "capacity": int
    - "cost_scale_factor": float, required, finite and positive. The pool's
      "intercept" and "coefficients" values are in the units of this factor; a
      policy authored in currency units passes 1.0. The producer block's
      "cost_scale_factor" is provenance only and is never a fallback.
    - "warm_start_count": int, optional, default 0
    - "populated_count": int, optional, default len("cuts")
    - "node_id": int, optional, default -1 (a pool with no single owning node)
    - "graph_stage_id": int, optional, default -1
    - "priced_state_date": int (YYYYMMDD), optional, default -2147483648 (undated)
    - "entity_manifest": list of slot dicts, optional, default []
    - "cuts": list of cut dicts

    A cut dict has "slot_index", "iteration", "forward_pass_index", "intercept",
    "coefficients" (length "state_dimension") and "is_active"; optional
    "scored" (default False), "active_count" and "last_active_iteration"
    (default absent; set both on every cut or on none, matching the producer's
    "activity_counters") and "inflow_lag_coefficients" (default {}, needs
    ``inflow_lag_depth``).

    A slot dict has "entity_type", "entity_id", "subindex" and "was_active";
    optional "reference_date", "interval_start" and "interval_end" (YYYYMMDD,
    default -2147483648, no date).

    ``stage_bases``: one dict per node, with "stage_id" (the node position),
    "iteration" and "entries", a list of dicts with "scenario",
    "column_status", "row_status" and "piece_row_slots". Default: none.

    ``stage_states``: one dict per pool that stores trial states, with
    "stage_id" (the pool id), "state_dimension", "count", "slots" (strictly
    ascending, one per row), "data" (``count * state_dimension`` values,
    row-major); optional "node_id" (default -1) and "entity_manifest" (default
    []). Default: none.

    ``metadata``:
    - "created_at": str, "num_stages": int
    - "format_version": int, optional, default the current format
    - "graph_manifest": optional, {"n_pools", "nodes": [{"id", "stage_id",
      "pool_id"}], "edges": [{"source_id", "target_id", "probability"}]}
    - "season_manifest": optional, {"cycle_code": int (0 monthly / 1 weekly /
      2 custom / 255 absent), "n_seasons": int, "hydro_orders": list of
      {"hydro_id": int, "orders": list[int]}}; omitted, the checkpoint carries
      the absent descriptor (cycle_code=255)
    - "producer": dict with "completed_iterations", "final_lower_bound",
      "max_iterations", "forward_passes", "warm_start_cuts", "rng_seed";
      optional "best_upper_bound" (None), "warm_start_counts" ([]),
      "total_visited_states" (0, must equal the rows of ``stage_states``),
      "training_block_mode" (""), "training_block_mode_per_stage" ([]),
      "cost_scale_factor" (None), "lower_bound_history" ([]), "state_history"
      (None, "whole" or a positive int window; a bool is rejected),
      "activity_counters" (False) and "basis_store_complete" (False)

    The checkpoint always records the running software and version; "software",
    "software_version" or "novomodelo_version" keys in metadata are ignored.

    ``inflow_lag_depth``: when N > 0, reserve N inflow-lag slots per storage
    hydro in every manifest and place each cut's "inflow_lag_coefficients".

    Raises ``ValueError`` when:
    - a pool dict holds "active_cut_indices", a cut dict "cut_id", or a basis
      dict "num_cut_rows", "num_columns" or "num_rows" (retired keys, named with
      their position; remove them);
    - a pool's "cost_scale_factor" is missing, not finite or not positive;
    - a cut's "coefficients" length differs from its pool's "state_dimension",
      or "inflow_lag_coefficients" is given without ``inflow_lag_depth``;
    - the producer sets "basis_store_complete" with no basis in ``stage_bases``;
    - "state_history" is not None, "whole" or a positive int;
    - the payloads break a checkpoint rule (slot order, row counts, counter
      presence, dates, a repeated id, size limits), with the rule's message.

    A write failure raises the matching ``novomodelo.errors`` class.
    """
    ...

class Study:
    def __init__(
        self,
        case_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        threads: Optional[int] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None: ...
    @property
    def output_dir(self) -> str: ...
    @property
    def system(self) -> System: ...
    @property
    def stochastic(self) -> StochasticSummary: ...
    @property
    def hydro_models(self) -> HydroModelsSummary: ...
    @property
    def provenance(self) -> ProvenanceReport: ...
    def validate(self) -> dict[str, Any]: ...
    def train(
        self,
        on_iteration: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> "Policy": ...
    def load_policy(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "Policy": ...
    def simulate(
        self,
        policy: "Policy",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]: ...

class Policy:
    @property
    def iterations(self) -> int: ...
    @property
    def final_lower_bound(self) -> float: ...
    @property
    def final_upper_bound(self) -> float: ...
    def evaluate(self, stage: int, state: Sequence[float]) -> float: ...
    def cut_matrix(self, stage: int) -> tuple[Any, Any]: ...
