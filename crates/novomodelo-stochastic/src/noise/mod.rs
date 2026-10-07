//! Deterministic noise generation for scenario construction.
//!
//! Each per-scenario, per-stage seed is derived from a global base seed via
//! SipHash-1-3, so any subset of scenarios generates independently on any node
//! without inter-process coordination — identical output regardless of how work
//! is partitioned across ranks or threads.

pub mod quantile;
pub mod rng;
pub mod seed;
