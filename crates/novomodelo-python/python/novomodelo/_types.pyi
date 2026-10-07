"""Typing-only result shapes for `cobre.run.run(...)` and `cobre.Study` properties.

This module carries **no** compiled counterpart: the leading underscore marks it
as a pure typing helper (`PEP 589` `TypedDict`s). It is never registered as a
runtime submodule -- importing it from compiled code would fail. Type checkers
pick it up via `[tool.pyright] stubPath = "stubs"`. The field set of each
`TypedDict` mirrors the `set_item` call sites in `src/run.rs`; do not invent
keys.
"""

from typing import Any, NotRequired, Optional, TypedDict

class SimulationSummary(TypedDict):
    """Nested `simulation` sub-dict built by `cobre.run.run`."""

    n_scenarios: int
    completed: int

class TrainingSummary(TypedDict):
    """Training headline fields.

    The `run()` result inlines these fields at the **top level** of
    `RunResult` (`converged`, `iterations`, `lower_bound`, `upper_bound`,
    `gap_percent`, and `total_time_ms`) rather than nesting them under a
    `training` key. This standalone `TypedDict` mirrors them for reuse by
    callers who want to annotate the training headline in isolation.
    """

    converged: bool
    iterations: int
    lower_bound: float
    upper_bound: Optional[float]
    gap_percent: Optional[float]
    total_time_ms: int

class StochasticSummary(TypedDict, total=False):
    """Nested `stochastic` sub-dict.

    Keys mirror `stochastic_summary_to_dict`. `ar_order` is the nested AR-order
    summary: `None` when no AR model is fitted, otherwise the dict built by
    `ar_order_to_dict`. `total=False` because the `ar_order` value is conditionally
    `None`.
    """

    inflow_source: Optional[str]
    n_hydros: int
    n_seasons: int
    ar_order: Optional[Any]
    correlation_source: Optional[str]
    correlation_dim: Optional[str]
    opening_tree_source: Optional[str]
    openings_per_stage: Any
    n_stages: int
    n_load_buses: int
    seed: Any

class HydroModelsSummary(TypedDict, total=False):
    """Nested `hydro_models` sub-dict (mirrors `hydro_model_summary_to_dict`)."""

    n_constant: int
    n_fpha: int
    total_planes: int
    n_evaporation: int
    n_no_evaporation: int

class ProvenanceReport(TypedDict, total=False):
    """Nested `provenance` sub-dict (mirrors `provenance_to_dict`).

    `total=False` because the report carries conditional/variable nested keys;
    `hydro_production` is itself a nested dict whose precise shape is not
    load-bearing for discoverability, so it is typed `Any`.
    """

    estimation_path: str
    seasonal_stats_source: str
    ar_coefficients_source: str
    correlation_source: str
    opening_tree_source: str
    n_hydros: int
    ar_method: Optional[str]
    ar_max_order: Optional[int]
    white_noise_fallbacks: list[int]
    hydro_production: Any

class RunResult(TypedDict):
    """Top-level result of `cobre.run.run(...)`.

    All keys are always present; the `Optional[...]` values are `None` (not absent)
    when the corresponding phase did not run, so `RunResult` is **not** `total=False`.
    `upper_bound` and `gap_percent` are `None` on simulation-only and both-disabled
    runs (mirrors `RunSummary.{upper_bound,gap_percent}: Option<f64>`); `lower_bound`
    is always present.
    """

    converged: bool
    iterations: int
    lower_bound: float
    upper_bound: Optional[float]
    gap_percent: Optional[float]
    total_time_ms: int
    output_dir: str
    simulation: Optional[SimulationSummary]
    stochastic: Optional[StochasticSummary]
    hydro_models: Optional[HydroModelsSummary]
    provenance: Optional[ProvenanceReport]
