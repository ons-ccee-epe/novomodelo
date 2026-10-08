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
    /// Export visited forward-pass trial points to the policy checkpoint.
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
