//! Shared single-scenario noise point specification.

/// Configuration for single-scenario noise point generation, shared across
/// the LHS, Halton, and Sobol point-wise generators.
#[derive(Debug, Clone, Copy)]
pub struct NoisePointSpec {
    /// Forward-pass base seed.
    pub sampling_seed: u64,
    /// Training iteration index.
    pub iteration: u32,
    /// Global scenario index in `0..total_scenarios`.
    pub scenario: u32,
    /// Seed-tuple stream identifier: forward-pass producers pass the
    /// `noise_group_id`, opening-tree producers pass the stage id.
    pub stream_id: u32,
    /// Total forward scenarios per iteration (= N strata for LHS).
    pub total_scenarios: u32,
    /// Noise vector dimension.
    pub dim: usize,
}
