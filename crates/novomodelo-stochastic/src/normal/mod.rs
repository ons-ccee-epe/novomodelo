//! Normal (i.i.d. Gaussian) noise model building blocks: `x = μ + σ·ε`, where
//! `ε ~ N(0, 1)`. See [`precompute`] for the LP-ready cached parameter arrays.

pub mod precompute;

pub use precompute::{BlockFactorPair, EntityFactorEntry, PrecomputedNormal};
