//! Export flag configuration for `config.json → exports`.

use serde::{Deserialize, Serialize};

/// Export flags controlling which outputs are written to disk
/// (`config.json → exports`).
///
/// Only the active fields below are accepted. Legacy keys (`training`, `cuts`,
/// `vertices`, `simulation`, `forward_detail`, `backward_detail`,
/// `compression`) must be removed from existing `config.json` files before
/// loading — they are now rejected as unknown fields.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportsConfig {
    /// Store the trial state of each row the run adds, under every selection
    /// method, and write the stored states to the policy checkpoint. Each row
    /// pool that receives rows stores the states of its own rows, at its own
    /// state dimension: the first stage's pool, which keeps every state, and
    /// the pool of each stage strictly between the first and the last, which
    /// under `level1` with `lookback_iterations` keeps only the states inside
    /// the window. A `level1` run stores and writes the states of the pools it
    /// selects whether or not this is set, since a resume reads them. Storing
    /// the states changes no result of the run; the policy then marks its
    /// rows `scored`, which a later `level1` run that loads it selects like
    /// its own rows.
    pub states: bool,

    /// Export stochastic preprocessing artifacts to `output/stochastic/`.
    pub stochastic: bool,

    /// Export the per-sampled-point computed-FPHA fit-deviation table to
    /// `output/hydro_models/fpha_deviation_points.parquet`.
    ///
    /// Opt-in (default `false`) purely for size: it emits one row per
    /// `(hydro, stage, V, Q)` grid point at spillage = 0. Off ⇒ no file and a
    /// byte-identical run; the table is additive and never enters the parity
    /// hash. The values are deterministic (a pure function of geometry + config),
    /// so the file is reproducible when emitted.
    pub fpha_deviation_points: bool,
}
